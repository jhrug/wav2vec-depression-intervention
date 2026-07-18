# wav2vec-depression-intervention

Master's thesis project (Artificial Intelligence, University of Groningen) investigating whether a Wav2Vec 2.0 model fine-tuned on the DAIC-WoZ can be used to estimate and track depressive symptom severity for raw speech data from a psychological intervention study.

The main finding is that the model ended up encoding who is speaking rather than depression-specific information, and that this confound already seems to be encoded in the pretrained Wav2Vec 2.0 representations before any fine-tuning happens. This means that a portion of the prior literature reporting strong results on this task may reflect speaker identity recognition rather than true depression estimation. 

The setup used is not specific to this model or dataset: this repo contains preprocessing (with generic functions applicable to sets of raw .WAV files following a specifiable naming convention) for a separate intervention study (interviews and journal recordings), and the pipeline as a whole is offered as a starting point for anyone wanting to extend these findings by fine-tuning an acoustic foundation model on a more varied set of depression estimation tasks before testing its robustness or applying explainability techniques.

## File structure

- `analyses/` - descriptives, embedding diagnostics (LOSO/LOPO, t-SNE), performance metrics, figures
- `cross_domain_integrated_gradients/` - CDIG attribution analysis
- `data` - see below
- `fine-tuning/` - model definition and training scripts and per-run loss curves along with other visualizations (based on dev set)
- `inference/` - running the trained model on held-out DAIC-WoZ data and data from psychological intervention study
- `intervention_transcripts/` - content of audio files sent to participants for daily practice during intervention
- `preprocessing/` - DAIC-WoZ audio and transcript handling, segmentation and labeling, Wav2Vec preprocessor
- `thesis and supplemental material/` - full thesis PDF, proposal, protocol docs, overview of runs and performance metrics

## Data
Raw DAIC-WoZ audio requires a data use agreement, so it's not included here - request access through the [official form](https://dcapswoz.ict.usc.edu/daic-woz-database-download/). Conventions used throughout: 20s segments, 16000 Hz. Due to privacy agreements, raw audio from the intervention study is not available under any circumstance.

Anonymized data used for main findings are included, along with the notebooks used to generate them.
- `df_phq_grand.pkl` — PHQ scores for the intervention study. Rows have been
  reshuffled and participant IDs replaced with random two-character strings.
- `woz_phqs.pkl` — actual vs estimated (by final model) PHQ scores for every participant X
  segment-number combination in the DAIC-WoZ, including a column specifying split (dev/train/test).
- `woz_eGeMAPS_features.pkl` — eGeMAPS acoustic features extracted from DAIC-WoZ
  audio (via openSMILE), used for the classical baseline models (Ridge/SVR/Random
  Forest).
- `embeddings_epoch_0_and_20.parquet` - DAIC_WoZ test set (w2v) encoder output for LOSO/LOPO analysis

## Fine-tuning script design
`fine-tuning/finetune_wav2vec_DAIC_WoZ.py` contains a block of constants specifying architectural / training choices so that every configuration used in the thesis (see Appendix A) can be reproduced by flipping the value of a constant rather than editing the model code, for example:

- `POOLING_METHOD` - mean or attention pooling over the time dimension
- `INTER_FC_LAYER` - optional intermediate projector layer before the regressor
- `DIVERSITY_LOSS` - optional variance-based penalty to discourage collapsed estimates
- `ADVERSARIAL_TRAINING` - optional speaker-classification head with adversarial term, to test whether discouraging speaker-identity encoding helps
- `FREEZE_BACKBONE`, `FINAL_TEST`, `OVERFIT_TEST`, dropout rates, learning rates and weight decay per parameter group


## Setup

```bash
pip install -r requirements.txt
```

## Running it
Download DAIC-WoZ data, preprocess with the scripts in `preprocessing/`, train with `fine-tuning/finetune_wav2vec_DAIC_WoZ.py`, run `inference/` on the result, then `analyses/` and CDIG for the diagnostics, performance and plots. Not every script has been tested for a clean run from a fresh clone yet, so please open an issue if something's broken.

## License
GPL-3.0, see `LICENSE`.
