import torch
from datasets import load_from_disk
from finetune_w2v2_DAIC_WoZ import Wav2Vec2ForRegression

# ______________________________________ CONSTANTS ________________________________________________
OUTPUT_DIR = "/placeholder/path" # path specifying where to save depression estimations
MODEL_PATH = "/placeholder/path" # path to model checkpoint
WOZ_PATH = "/placeholder/path" # path to DAIC-WoZ dataset
MIC_PATH = "/placeholder/path" # path to interview dataset from intervention study
HOME_PATH = "/placeholder/path" # path to dataset from phone journal recordings

# ______________________________________ FUNCTIONS ________________________________________________
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

woz = load_from_disk(WOZ_PATH)
mic = load_from_disk(MIC_PATH)
home = load_from_disk(HOME_PATH)

model = Wav2Vec2ForRegression.from_pretrained(MODEL_PATH)
model.eval()
model = model.to(device)

def predict(batch):
    with torch.no_grad():
        output = model(input_values=torch.tensor(batch["input_values"]).to(device), return_dict=True)
    return {"prediction": output.logits.squeeze().tolist()}

woz = woz.map(predict, batched=True, batch_size=16)
woz = woz.remove_columns(["audio", "input_values"])

mic = mic.map(predict, batched=True, batch_size=16)
mic = mic.remove_columns(["audio", "input_values"])

home = home.map(predict, batched=True, batch_size=16)
home = home.remove_columns(["audio", "input_values"])

woz.to_parquet(f"{OUTPUT_DIR}/woz_preds_final.parquet")
mic.to_parquet(f"{OUTPUT_DIR}/mic_preds_final.parquet")
home.to_parquet(f"{OUTPUT_DIR}/home_preds_final.parquet")