import os #for os.walk
import random
import re
import pathlib # for path manipulations
from time import process_time_ns
import pandas as pd
import seaborn as sb
import librosa # loading to wav
import torchaudio # backup load method
import numpy as np # for daicwoz signal arrays
import soundfile as sf # reading and writing sound files
import torch
from collections import Counter
from pandas import DataFrame
from pandas.core.dtypes.common import pandas_dtype
from pathlib import Path
from datasets import Dataset, Audio, Features, Value, Sequence, load_from_disk, concatenate_datasets
from transformers import AutoFeatureExtractor, Wav2Vec2Config
from ft_normreg_longer_ifmain import Wav2Vec2ForRegression
from cross_domain_saliency_maps.torch_ig.cross_domain_integrated_gradients import STFTIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR      = ".../output/cdig"
MODEL_PATH      = "..../checkpoints/final_model/checkpoint-680"
WOZ_PATH        = "..../data/woz"
MIC_PATH        = ".../data/mic"
HOME_PATH       = ".../data/home"

# WRAPPER AROUND MODEL:
# Captum/CDIG expects forward(x) -> tensor, not forward(input_values=..., ...) -> ModelOutput
class Wav2Vec2RegressionWrapper(torch.nn.Module):
    #initialize
    def __init__(self, model):
        super().__init__()
        #store trained model to wrap
        self.model = model

    def forward(self, x):
        x = x.squeeze()  # remove all size-1 dimensions -> [T] or [B, T]
        #IG can pass a single unbatched example; re-add batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)  # ensure [1, T]
        #run wrapped model, then extract only the logits (IG needs a raw tensor, not a ModelOutput)
        #unsqueeze to restore expected output shape for STFTIG
        return self.model(input_values=x, return_dict=True).logits.unsqueeze(0)

# WARNING: assumes a single (unbatched) audio segment in input_values, not a batch (see STFTIG reshape below)
def ig_stft(input_values: list,
            wrapped: torch.nn.Module,
            n_fft: int = 512,
            hop_length: int = 160,
            win_length: int = 320,
            n_iterations: int = 50,
            output_channel: int = 0,
            ) -> torch.Tensor:
    #convert to array before reshaping
    input_array = np.array(input_values)
    # add batch dimension: [T] -> [1, T]
    input_2d = input_array[np.newaxis, :]
    # add channel dimension: [1, T] -> [1, 1, T] (shape STFTIG expects)
    input_3d = input_2d[:, np.newaxis, :]
    # baseline for IG: all-zero input (of same shape)
    baseline_3d = np.zeros_like(input_3d)

    #initialize CDIG object with wrapped model and STFT params
    stftIG = STFTIG(
        wrapped,
        n_iterations,
        output_channel=output_channel,
        device=device,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )

    #run attribution: input vs. zero baseline
    output = stftIG.run(input_3d, baseline_3d)

    return output

# ________________________________ WARNING ___________________________________
# CDIG can be extremely computationally expensive. This is a script that selects one
# recording per participant over which to compute the attributions.

#LOAD DAIC-WOZ DATA AND RANDOMLY SELECT ONE PER PARTICIPANT
woz_ds = load_from_disk(WOZ_PATH)
woz = woz_ds.to_pandas()
woz_sample = woz.groupby('participant').sample(n=1, random_state=534)
woz_sample_ds = Dataset.from_pandas(woz_sample)

#LOAD INTERVENTION STUDY INTERVIEW DATA AND RANDOMLY SELECT ONE PER PARTICIPANT
mic_ds = load_from_disk(MIC_PATH)
mic = mic_ds.to_pandas()
mic_sample = woz.groupby('participant').sample(n=1, random_state=534)
mic_sample_ds = Dataset.from_pandas(mic_sample)

#LOAD INTERVENTION STUDY JOURNAL DATA AND RANDOMLY SELECT ONE PER PARTICIPANT
home_ds = load_from_disk(HOME_PATH)
home = home_ds.to_pandas()
home_sample = woz.groupby('participant').sample(n=1, random_state=534)
home_sample_ds = Dataset.from_pandas(home_sample)

#LOAD MODEL
model = Wav2Vec2ForRegression.from_pretrained(MODEL_PATH)
wrapped_model = Wav2Vec2RegressionWrapper(model)
wrapped_model.eval()
wrapped_model = wrapped_model.to(device)

#DEFINE CDIG FOR MAP
def cdig(sample):
    print(f"Processing participant {sample['participant']}", flush=True)
    ig = ig_stft(sample['input_values'], wrapped_model)
    return {'stft_ig': ig.squeeze().detach().cpu().numpy()}

woz_sample_ds = woz_sample_ds.map(cdig, batched=False)
woz_sample_ds.save_to_disk(f"{OUTPUT_DIR}/woz_sample_cdig")

mic = mic.map(predict, batched=False)
mic_sample.save_to_disk(f"{OUTPUT_DIR}/mic_sample_cdig")

home = home.map(predict, batched=False)
home_sample.save_to_disk(f"{OUTPUT_DIR}/home_sample_preds")