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
    #callbacks,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput
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
    #initialize callbacks here
    #callbacks = []


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