# BiBiER

## Audio Modality Training

The following pretrained encoders were used for acoustic feature extraction:
- [`wav2vec2`](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
- [`ExHuBERT`](https://huggingface.co/amiriparian/ExHuBERT)

For each encoder, experiments were conducted with three model types — LSTM, Transformer, and Mamba — using various architectures and hyperparameter settings.

### 1) wav2vec2

- The file [`train_wav2vec2.ipynb`](./train_wav2vec2.ipynb) contains a full experiment grid based on embeddings extracted using wav2vec2.

### 2) ExHuBERT

- The file [`train_hubert.ipynb`](./train_hubert.ipynb) contains a similar experiment grid for models trained on ExHuBERT embeddings.

### Experimental Results

Across all model types, wav2vec2-based embeddings consistently yielded the best results. Among the models tested, variants of Mamba achieved the highest performance metrics.

- The file [`best_model.ipynb`](./best_model.ipynb) provides an example of how to apply the best-performing model to an audio sample.

- The `info` directory contains training logs and performance tables.
