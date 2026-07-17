import evaluate
import torch
import torch.nn as nn
from datasets import load_from_disk, concatenate_datasets
from torch.optim import AdamW
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Config,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# use nvidia GPU if available; default to CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ______________________________________ CONSTANTS ________________________________________________
PHQ_MEAN = 8.4

# specify whether this is the final model
# trains on official DAIC-WoZ train split if False, with evaluation on development set
# trains on (dev U train) if True, with evaluation on test set
FINAL_TEST = False
OVERFIT_TEST = False # set train and eval to the union of train and eval

OUTPUT_DIR = "/placeholder/path" # <- paste path for output directory here
DATA_PATH = "/placeholder/path" # <- paste path to DAIC-WoZ dataset (saved using Dataset's .save_to_disk method)
NUM_EPOCHS = 20 # number of full passes over the train set
BATCH_SIZE = 32
GRAD_ACCUM = 4 # number of mini-batches to average over before weight update

# modifications to training objective
DIVERSITY_LOSS = False # subtract variance of predictions from loss to discourage collapsed/uniform predictions
DIV_LOSS_WEIGHT = 0.1 # weight applied to the variance penalty term
ADVERSARIAL_TRAINING = False
LAMBDA_ADVERSARIAL = 1e-3

# hyperparams varied during model development; converted to constants to facilitate replication
INTER_FC_LAYER = False # add additional projector layer between encoder output and regressor neuron
LEARNING_RATES = [1e-7, 1e-6, 1e-5] # learning rates for [feature extractor, encoder, head]
WEIGHT_DECAYS = [0.002, 0.02, 0.2] # penalize large weights (L2) for [feature extractor, encoder, head]
COSINE_ANNEALING = True # decay LEARNING_RATES following cosine curve
FREEZE_BACKBONE = False
POOLING_METHOD = "mean" # "mean" or "attention"; pooling strategy over time dimension
BIAS_INIT = True #intialize bias with PHQ mean
FINAL_DROPOUT = 0.5 # proportion of randomly dropped activations before head
NORMALIZED = True # specify whether the model should predict raw or normalized PHQ values
WARMUP_STEPS = 0.1 # proportion of steps for linear warmup
HIDDEN_DROPOUT = 0.3 # hidden dropout proportion applied to each transformer layer
ATTENTION_DROPOUT = 0.1 # proportion of random dropout applied to attention weights
ACTIVATION_DROPOUT = 0.0 # dropout applied to FFN layers after activation
MASK_TIME_PROBABILITY = 0.05 # probability that each time step is masked in SpecAugment
MASK_FEATURE_PROBABILITY = 0.0 # proportion of feature channels randomly masked
LAYER_DROP = 0.0 # probability of randomly skipping transformer layer

# ______________________________________ FUNCTIONS ________________________________________________
# utility function for plots - given repeated computation it gives a misleadingly small range
# still, it at least provides some visual reminder of uncertainty
def pearson_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = 1.96
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return lo, hi
# ______________________________________ CLASSES ________________________________________________

# WAV2VEC WITH REGRESSION HEAD
class Wav2Vec2ForRegression(Wav2Vec2PreTrainedModel):
    #initialize
    def __init__(self, config):
        super().__init__(config)
        # load wav2vec2 base (without task-specific head) with config params
        self.wav2vec2 = Wav2Vec2Model(config)
        # initialize attention pool
        if POOLING_METHOD == "attention":
            self.pooling_attention = nn.Linear(config.hidden_size, 1)
        # optionally add intermediate projector layer between encoder output and regressor
        if INTER_FC_LAYER:
            self.projector = nn.Linear(config.hidden_size, config.hidden_size)
            self.relu = nn.ReLU()
        # initialize dropout layer applied before regressor
        self.dropout  = nn.Dropout(FINAL_DROPOUT)
        # initialize linear layer of size 768 (default for w2v2 base) and 1 output feature
        self.regressor = nn.Linear(config.hidden_size, 1)
        # initialize bias with PHQ mean
        if BIAS_INIT:
            mean_phq = PHQ_MEAN / 24.0 if NORMALIZED else PHQ_MEAN
            nn.init.constant_(self.regressor.bias, mean_phq)
        # optionally store label variance for diversity loss
        if DIVERSITY_LOSS:
            self.register_buffer('label_var', torch.tensor(0.0))
        # initialize speaker classification head for adversarial training
        if ADVERSARIAL_TRAINING:
            self.lambda_adv = getattr(config, 'lambda_adv', LAMBDA_ADVERSARIAL)
            num_speakers = getattr(config, 'num_speakers', 0)
            self.speaker_head = nn.Linear(config.hidden_size, num_speakers) if num_speakers > 0 else None

        # apply model's w2v2 weight initialization scheme to newly added layers
        # & prepares for compatibility with gradient checkpointing
        self.post_init()

    def forward(
        self,
        # prevent unexpected keyword argument errors for:
        input_values=None,
        attention_mask=None,
        labels=None,
        # use config defaults for:
        output_attentions=None,
        output_hidden_states=None,
        # to be sure
        return_dict=True,
        # for adversarial
        speaker_id=None,
    ):

        # run wav2vec backbone (feature extractor + transformer encoder)
        outputs = self.wav2vec2(
            # preprocessed audio waveform
            input_values,
            # padding mask (can be None)
            attention_mask=attention_mask,
            #whether to return attention weight matrices from each transformer head
            output_attentions=output_attentions,
            #whether to return hidden states from all 12 layers (not just last one)
            output_hidden_states=output_hidden_states, #
            return_dict=True,
        ) # returns hf ModelOutput object

        #store output of last layer
        last_hidden_states = outputs.last_hidden_state

        # POOL OVER TIME (WARNING: ASSUMES NO PADDING)
        if POOLING_METHOD == "mean":
            pooled = last_hidden_states.mean(dim=1)  # [batch, 768]
        elif POOLING_METHOD == "attention":
            attn_weights = torch.nn.functional.softmax(
                self.pooling_attention(last_hidden_states), dim=1
            ) # [batch, time_steps, 1]
            pooled = (last_hidden_states * attn_weights).sum(dim=1) # [batch, 768]

        #APPLY DROPOUT
        pooled = self.dropout(pooled)

        #apply linear layer to time-pooled batch with dropout
        logits = self.regressor(pooled) #result is a 2D tensor [batch, 1] = [[x1],[x2],...,[x_batch]]
        #i.e., a list of 1D vectors

        #for every item in the batch, take 0th item to return [batch] = [x1, x2, ..., x_batch]
        #i.e., 1 batch-dimensional vector
        logits = logits[:, 0] # alternatively: logits = logits.squeeze(-1)

        #default to be used at inference time
        loss = None

        #if labels available
        if labels is not None:
            #we just use nn.functional for loss computation, make sure labels are also float32 (not fp16)
            loss = nn.functional.mse_loss(logits, labels.float())
            # subtract variance multiplied by weight
            if DIVERSITY_LOSS:
                loss = loss - DIV_LOSS_WEIGHT * torch.clamp(torch.var(logits), max=self.label_var)
            # compute adversarial training loss
            if ADVERSARIAL_TRAINING and speaker_id is not None and self.speaker_head is not None:
                speaker_logits = self.speaker_head(pooled)
                speaker_loss = nn.functional.cross_entropy(speaker_logits, speaker_id.long())
                loss = loss - self.lambda_adv * speaker_loss

        #return subclass of ModelOutput that bundles output together in standardized way
        #this lets Trainer easily find loss, logits, etc.
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LossCurveCallback(TrainerCallback):

    #initialize lists for train loss, eval loss, epoch and pearson r
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.epochs = []
        self.pearson_r = []

    #append loss and pearson r to list
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append((state.epoch, logs["loss"]))
        if logs and "eval_loss" in logs:
            self.eval_losses.append((state.epoch, logs["eval_loss"]))
        #Trainer automatically prepends eval_ to everything returned by compute_metrics
        if logs and "pearson_r" in logs:
            self.pearson_r.append(logs["pearson_r"])

    #create plot of loss
    def on_train_end(self, args, state, control, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        if self.train_losses:
            epochs, losses = zip(*self.train_losses)
            ax.plot(epochs, losses, label="train loss", alpha=0.7)
        if self.eval_losses:
            epochs, losses = zip(*self.eval_losses)
            ax.plot(epochs, losses, label="eval loss", alpha=0.7)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/loss_curve.png", dpi=150)
        plt.close()
        #print(f"[INFO] Loss curve saved to {args.output_dir}/loss_curve.png")

class InspectionCallback(TrainerCallback):
    def __init__(self, eval_dataset, output_dir, device, batch_size=16):
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        # running stats across epochs, for the summary plots in on_train_end
        self.epochs = []
        self.epoch_pred_global_variance = []
        self.epoch_pred_global_mean = []
        self.epoch_mean_participant_variance = []
        self.epoch_r_mean_pred = []
        self.epoch_r_mean_ci_low = []
        self.epoch_r_mean_ci_high = []
        self.epoch_r_extr_pred = []
        self.epoch_r_extr_ci_high = []
        self.epoch_r_extr_ci_low = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)

        # BUILD PARTICIPANT MAP (segment idx -> participant, so we can aggregate later)
        participant_map = defaultdict(list)
        for idx, pid in enumerate(self.eval_dataset["participant"]):
            participant_map[pid].append(idx)

        # EXTRACT ALL SEGMENT EMBEDDINGS
        # flatten participant_map back to a plain index list; probably just [0,1,...,last] already but not guaranteed
        all_indices = []
        for participant_indices in participant_map.values():
            for seg_idx in participant_indices:
                all_indices.append(seg_idx)

        model.eval() #disables dropout/layerdrop, want stable embeddings not augmented ones

        raw_embeddings = []
        participant_preds = defaultdict(list)

        with torch.no_grad():
            for i in range(0, len(all_indices), self.batch_size):
                batch_indices = all_indices[i : i + self.batch_size]
                batch_inputs = np.array([self.eval_dataset[int(j)]["input_values"] for j in batch_indices])
                inputs = torch.tensor(batch_inputs).float().to(self.device)

                #run encoder only, skip regressor - we want the embedding space here, not the prediction itself
                outputs = model.wav2vec2(input_values=inputs, return_dict=True)
                pooled = outputs.last_hidden_state.mean(dim=1)  # [batch, 768]
                raw_embeddings.append(pooled.cpu().numpy())

                #also grab prediction while we're here (dropout already disabled via model.eval())
                pred = model.regressor(model.dropout(pooled)).squeeze(-1)
                for k, j in enumerate(batch_indices):
                    pid_for_segment = self.eval_dataset[int(j)]["participant"]
                    #defaultdict: accessing a missing key auto-creates the list, so .append just works
                    participant_preds[pid_for_segment].append(pred[k].item())

        all_segment_embeddings = np.concatenate(raw_embeddings, axis=0)  # [n_segments, 768]

        # AVERAGE EMBEDDINGS AND LABELS PER PARTICIPANT
        participant_embeddings = []
        participant_labels = {}
        seg_cursor = 0 #tracks position in all_segment_embeddings as we walk through participants in order

        for pid, indices in participant_map.items():
            n_segs = len(indices)
            participant_segs = all_segment_embeddings[seg_cursor : seg_cursor + n_segs]
            participant_embeddings.append(participant_segs.mean(axis=0))
            participant_labels[pid] = self.eval_dataset[int(indices[0])]["labels"] #same label for every segment of this participant
            seg_cursor += n_segs

        embeddings = np.array(participant_embeddings)
        labels = np.array(list(participant_labels.values()))

        # T-SNE PLOT
        #pca first to denoise before t-SNE, standard practice for high-dim embeddings
        pca_pre = PCA(n_components=min(50, len(embeddings) - 1))
        embeddings_reduces = pca_pre.fit_transform(embeddings)

        perplexity = 5 #minimum of what's usually recommended, dataset is small (n=59 participants)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=435, max_iter=1000)
        tsne_result = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#e8e8e8")
        ax.set_facecolor("#e8e8e8")
        sc = ax.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            c=labels,
            cmap="gray_r",
            alpha=0.85,
            s=60,
            edgecolors="white",
            linewidths=0.5,
            vmin=0, vmax=1, #fix color scale to normalized PHQ range so plots are comparable across epochs
        )
        plt.colorbar(sc, ax=ax, label="PHQ score (normalized)")
        ax.set_title(f"t-SNE of participant embeddings - epoch {epoch}", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.2, color="white")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/tsne_epoch_{epoch:03d}.png", dpi=150) #zero-padded so alphabetical = chronological
        plt.close()
        # PCA + PHQ CORRELATION
        pca = PCA(n_components=10)
        pca_result = pca.fit_transform(embeddings)  # [n_participants, 10]

        correlations = []
        for pc_idx in range(10):
            r, p = pearsonr(pca_result[:, pc_idx], labels)
            correlations.append((pc_idx, r, p))

        correlations.sort(key=lambda x: abs(x[1]),
                          reverse=True)  # want strongest PHQ relationship regardless of direction
        top2 = correlations[:2]  # these become the plot axes

        pc1_idx, r1, p1 = top2[0]
        pc2_idx, r2, p2 = top2[1]

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#e8e8e8")
        ax.set_facecolor("#e8e8e8")
        sc = ax.scatter(
            pca_result[:, pc1_idx],
            pca_result[:, pc2_idx],
            c=labels,
            cmap="gray_r",
            alpha=0.85,
            s=60,
            edgecolors="white",
            linewidths=0.5,
            vmin=0, vmax=1,  # same scale as t-SNE plot, for cross-plot comparability
        )
        plt.colorbar(sc, ax=ax, label="PHQ score (normalized)")
        ax.set_xlabel(f"PC{pc1_idx + 1} (r={r1:.3f}")
        ax.set_ylabel(f"PC{pc2_idx + 1} (r={r2:.3f}")
        ax.set_title(f"Top-2 PHQ-correlated PCA components - epoch {epoch}", fontsize=12)
        ax.grid(True, alpha=0.2, color="white")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pca_epoch_{epoch:03d}.png", dpi=150)
        plt.close()

        # STATS AND CORRELATIONS
        all_preds_flat = []
        for preds in participant_preds.values():
            for pred in preds:
                all_preds_flat.append(pred)

        self.epoch_pred_global_variance.append(np.var(all_preds_flat))
        global_mean = np.mean(all_preds_flat)
        self.epoch_pred_global_mean.append(global_mean)

        extreme_preds = []
        mean_preds = []
        paired_labels = []  # ground truth, same order as extreme_preds/mean_preds

        for pid, preds in participant_preds.items():
            most_extreme = max(preds, key=lambda pred: abs(pred - global_mean))
            extreme_preds.append(most_extreme)
            mean_preds.append(np.mean(preds))
            paired_labels.append(participant_labels[pid])

        # mean participant-level variance, i.e. how inconsistent is the model within one speaker
        participant_variances = []
        for preds in participant_preds.values():
            participant_variances.append(np.var(preds))
        self.epoch_mean_participant_variance.append(np.mean(participant_variances))

        # two aggregation strategies: mean vs most-extreme segment per participant
        r_mean, _ = pearsonr(mean_preds, paired_labels)
        ci_low_mean, ci_high_mean = pearson_ci(r_mean, len(mean_preds))
        self.epoch_r_mean_pred.append(r_mean)
        self.epoch_r_mean_ci_low.append(ci_low_mean)
        self.epoch_r_mean_ci_high.append(ci_high_mean)

        r_extr, _ = pearsonr(extreme_preds, paired_labels)
        ci_low_extr, ci_high_extr = pearson_ci(r_extr, len(extreme_preds))
        self.epoch_r_extr_pred.append(r_extr)
        self.epoch_r_extr_ci_low.append(ci_low_extr)
        self.epoch_r_extr_ci_high.append(ci_high_extr)

        self.epochs.append(epoch)

    def on_train_end(self, args, state, control, **kwargs):

        # FIGURE 1: PEARSON R OVER EPOCHS - MEAN AGGREGATION
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.epochs, self.epoch_r_mean_pred, label="Pearson r (mean)", color="steelblue")
        ax.fill_between(self.epochs, self.epoch_r_mean_ci_low, self.epoch_r_mean_ci_high, alpha=0.2, color="steelblue")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Pearson r")
        ax.set_ylim(-1, 1)
        ax.set_title("Pearson r over epochs - mean aggregation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/r_mean_over_epochs.png", dpi=150)
        plt.close()

        # FIGURE 2: PEARSON R OVER EPOCHS - EXTREME AGGREGATION
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.epochs, self.epoch_r_extr_pred, label="Pearson r (extreme)", color="darkorange")
        ax.fill_between(self.epochs, self.epoch_r_extr_ci_low, self.epoch_r_extr_ci_high, alpha=0.2, color="darkorange")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Pearson r")
        ax.set_ylim(-1, 1)
        ax.set_title("Pearson r over epochs - extreme aggregation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/r_extr_over_epochs.png", dpi=150)
        plt.close()

        # FIGURE 3: VARIANCE OVER EPOCHS
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.epochs, self.epoch_pred_global_variance, label="global variance", color="steelblue")
        ax.plot(self.epochs, self.epoch_mean_participant_variance, label="mean participant variance",
                color="darkorange")
        ax.set_xlabel("epoch")
        ax.set_ylabel("variance")
        ax.set_title("Prediction variance over epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/variance_over_epochs.png", dpi=150)
        plt.close()

        # FIGURE 4: GLOBAL PREDICTION MEAN OVER EPOCHS
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.epochs, self.epoch_pred_global_mean, label="global prediction mean", color="steelblue")
        # ax.axhline(0.35, color="red", linestyle="--", linewidth=0.8, label="label mean (~0.35)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean prediction (normalized)")
        ax.set_title("Global prediction mean over epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/mean_over_epochs.png", dpi=150)
        plt.close()


# ______________________________________ MAIN ________________________________________________
if __name__ == "__main__":

    # INITIALIZE METRICS
    mse_metric = evaluate.load("mse")
    mae_metric = evaluate.load("mae")

    # DEFINE METRICS FUNCTION WITH EVALUATE
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.squeeze()
        refs = eval_pred.label_ids.squeeze()
        mse = mse_metric.compute(predictions=preds, references=refs)
        mae = mae_metric.compute(predictions=preds, references=refs)
        r, p = pearsonr(preds, refs)
        return {"mse": mse["mse"], "mae": mae["mae"], "pearson_r": r}

    # LOAD MODEL WEIGHTS
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # LOAD DATA
    data = load_from_disk(DATA_PATH)
    data = data.rename_column("phq_score", "labels")
    data = data.remove_columns(["source_file", "audio", "gender", "segment_number"])

    # NORMALIZE
    if NORMALIZED:
        data = data.map(lambda x: {"labels": x["labels"] / 24.0})

    # SPLIT INTO DEV, TRAIN, TEST
    train_data = data.filter(lambda x: x['split'] == 'train')
    dev_data = data.filter(lambda x: x['split'] == 'dev')
    if FINAL_TEST:
        test_data = data.filter(lambda x: x['split'] == 'test')

    # SET EVAL DATA
    if FINAL_TEST:
        train_data = concatenate_datasets([dev_data, train_data])
        eval_data = test_data
    else:
        eval_data = dev_data

    # MERGE SETS (IF OVERFITTING TEST)
    if OVERFIT_TEST:
        train_data = concatenate_datasets([eval_data, train_data])
        eval_data = train_data

    # REMOVE SPLIT COLUMN
    train_data = train_data.remove_columns("split")
    eval_data = eval_data.remove_columns("split")

    # LOAD MODEL
    config = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-base",
        mask_time_prob=MASK_TIME_PROBABILITY, #default=0.05
        mask_feature_prob=MASK_FEATURE_PROBABILITY, #default=0.0
        apply_spec_augment=True, #default=True
        hidden_dropout=HIDDEN_DROPOUT, #default=0.1
        attention_dropout=ATTENTION_DROPOUT, #default=0.1
        activation_dropout=ACTIVATION_DROPOUT, #default=0.0
        layerdrop=LAYER_DROP #default=0
    )
    model = Wav2Vec2ForRegression.from_pretrained("facebook/wav2vec2-base", config=config)
    model = model.to(device)

    # FREEZE TRANSFORMER BACKBONE (if specified)
    if FREEZE_BACKBONE:
        LEARNING_RATES[0] = 0.0
        LEARNING_RATES[1] = 0.0

    # SET OPTIMIZER
    opt_param_groups = [
        {"params": model.wav2vec2.feature_extractor.parameters(), "lr": LEARNING_RATES[0], "weight_decay": WEIGHT_DECAYS[0]},
        {"params": model.wav2vec2.encoder.parameters(), "lr": LEARNING_RATES[1], "weight_decay": WEIGHT_DECAYS[1]},
        {"params": model.regressor.parameters(), "lr": LEARNING_RATES[2], "weight_decay": WEIGHT_DECAYS[2]},
    ]
    if INTER_FC_LAYER:
        opt_param_groups.append(
            {"params": model.projector.parameters(), "lr": LEARNING_RATES[2], "weight_decay": WEIGHT_DECAYS[2]})
    if ADVERSARIAL_TRAINING:
        opt_param_groups.append(
            {"params": model.speaker_head.parameters(), "lr": LEARNING_RATES[2], "weight_decay": WEIGHT_DECAYS[2]})
    optimizer = AdamW(opt_param_groups)

    # TRAINING ARGS
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps= GRAD_ACCUM,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_pearson_r",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        save_total_limit=2,
        fp16=True,
    )
    loss_curve_cb = LossCurveCallback()
    inspection_cb = EmbeddingInspectionCallback(
        eval_dataset=eval_data,
        output_dir=OUTPUT_DIR,
        device=device,
    )
    callbacks
    callbacks = [loss_curve_cb,inspection_cb]


    # MANUALLY SET SCHEDULER
    if COSINE_ANNEALING:
        total_steps = (len(train_data) // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
        warmup_steps = int(WARMUP_STEPS * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )


    # TRAIN
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        #callbacks=callbacks,
    )

    trainer.train()