# wav2vec-depression-intervention

Master's thesis project (Artificial Intelligence, University of Groningen), looking at whether a Wav2Vec 2.0 model fine-tuned on the DAIC-WoZ can be used to estimate and track depressive symptom severity for raw speech data from a psychological intervention study.

The main finding is that the model ended up encoding who is speaking rather than depression-specific infomration, and that confound already seems to be encoded in the pretrained Wav2Vec 2.0 representations before any fine-tuning happens. This means that a portion of the prior literature reporting strong results on this task may pick up on speaker identity rather than depression. 

The setup used isn't specific to this model or dataset: this repo holds preprocessing (with generic functions applicable to sets of raw .WAV files following a specifiable naming convention) for a separate intervention study (interviews and journal recordings), and the pipeline as a whole is offered as a starting point for anyone wanting to extend these findings by fine-tuning an acoustic foundation model on a more varied set of depression estimation tasks before testing its robustness or applying explainability techniques.

## File structure

- `preprocessing/` - DAIC-WoZ audio and transcript handling, segmentation, feature extraction
- `fine-tuning/` - model definition and training scripts
- `inference/` - running the trained model on held-out DAIC-WoZ data and data from psychological intervention study
- `analyses/` - embedding diagnostics (LOSO/LOPO, t-SNE, PCA), performance metrics, figures
- `cross_domain_integrated_gradients/` - CDIG attribution analysis
- `thesis and supplemental material/` - full thesis PDF, proposal, protocol docs, overview of runs and performance metrics

## Data
DAIC-WoZ requires a data use agreement, it's not included here. Request access through the official form: [(https://dcapswoz.ict.usc.edu/daic-woz-database-download/)](https://dcapswoz.ict.usc.edu/daic-woz-database-download/).Conventions used throughout: 20s segments, 16000 Hz, `facebook/wav2vec2-base` pretrained on LibriSpeech 960h.

Anonymized data used for maing findings are included, along with the notebooks used to generate them.

## Setup

```bash
pip install -r requirements.txt
```

## Running it
Download DAIC-WoZ data, preprocess with the scripts in `preprocessing/`, train with `fine-tuning/finetune_wav2vec_DAIC_WoZ.py`, run `inference/` on the result, then `analyses/` and CDIG for the diagnostics, performance and plots. Not every script has been tested for a clean run from a fresh clone yet, so open an issue if something's broken.

## Fine-tuning script design
`fine-tuning/finetune_wav2vec_DAIC_WoZ.py` is built around a block of top-level constants rather than hardcoded values so that every architectural and training choice used across the thesis can be reproduced by flipping the value of a constant rather than editing the model code, for example:

- `POOLING_METHOD` - mean or attention pooling over the time dimension
- `INTER_FC_LAYER` - optional intermediate projector layer before the regressor
- `DIVERSITY_LOSS` - optional variance-based penalty to discourage collapsed predictions
- `ADVERSARIAL_TRAINING` - optional speaker-classification head with adversarial term, to test whether discouraging speaker-identity encoding helps
- `FREEZE_BACKBONE`, `FINAL_TEST`, `OVERFIT_TEST`, dropout rates, learning rates and weight decay per parameter group

## License
GPL-3.0, see `LICENSE`.
