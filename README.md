# wav2vec-depression-intervention

Master's thesis project (Artificial Intelligence, University of Groningen), looking at whether a Wav2Vec 2.0 model fine-tuned on the DAIC-WoZ can be used to estimate and track depressive symptom severity for raw speech data from a psychological intervention study.

The main finding is that the model ended up encoding who is speaking rather than depression-specific infomration, and that confound already seems to be encoded in the pretrained Wav2Vec 2.0 representations before any fine-tuning happens. This means that a chunk of the prior literature reporting strong results on this task may pick up on speaker identity rather than depression. More detail in the thesis itself under `thesis and supplemental material/`.

This repo also holds preprocessing for a separate intervention study (interviews + journal recordings), not just the DAIC-WoZ pipeline, so don't assume everything here feeds into the same model.

## File structure

- `preprocessing/` - DAIC-WoZ audio and transcript handling, segmentation, feature extraction
- `fine-tuning/` - model definition and training scripts
- `inference/` - running the trained model on held-out DAIC-WoZ data and data from psychological intervention study
- `analyses/` - embedding diagnostics (LOSO/LOPO, t-SNE, PCA), performance metrics, figures
- `cross_domain_integrated_gradients/` - CDIG attribution analysis
- `thesis and supplemental material/` - full thesis PDF, proposal, protocol docs, overview of runs and performance metrics

## Data
DAIC-WoZ requires a data use agreement, it's not included here. Request access through the official form: [(https://dcapswoz.ict.usc.edu/daic-woz-database-download/)](https://dcapswoz.ict.usc.edu/daic-woz-database-download/).
Conventions used throughout: 20s segments, 16000 Hz, `facebook/wav2vec2-base` pretrained on LibriSpeech 960h.

## Setup

```bash
pip install -r requirements.txt
```

## Running it
Download DAIC-WoZ data, preprocess with the scripts in `preprocessing/`, train with `fine-tuning/finetune_wav2vec_DAIC_WoZ.py`, run `inference/` on the result, then `analyses/` and CDIG for the diagnostics, performance and plots. Not every script has been tested for a clean run from a fresh clone yet, so open an issue if something's broken.

## License
GPL-3.0, see `LICENSE`.
