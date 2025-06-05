# BiBiER: A Bilingual Bimodal Emotion Recognition Method with Improved Generalization through Data Augmentation

This repository accompanies the publication in **Expert Systems with Applications (ESWA), 2025**:

> [Elena Ryumina](https://scholar.google.com/citations?user=DOBkQssAAAAJ), [Alexandr Axyonov](https://scholar.google.com/citations?user=Hs95wd4AAAAJ), Timur Abdulkadirov, Darya Koryakovskaya, Svetlana Gorovaya, Anna Bykova, Dmitry Vikhorev, [Dmitry Ryumin](https://scholar.google.com/citations?user=LrTIp5IAAAAJ)
> 
> HSE University

---

## Demo

An interactive demo of the **BiBiER** system is available on **Hugging Face Spaces**:  
👉 [BiBiER](https://huggingface.co/spaces/DmitryRyumin/BiBiER)

This web app allows you to upload or record speech in Russian or English and receive emotion predictions based on:
- 🎧 Acoustic features (audio-based analysis)
- 📝 Transcribed content (text-based analysis)
- 🔄 Bimodal fusion (audio + text integration)

---

## 🧠 Abstract

Emotion Recognition (ER) is essential for real-world human-computer interaction, but multimodal systems that use audio, text, and video are often too complex for practical applications. Bimodal audio-text systems provide a more feasible balance between accuracy and efficiency, particularly in applications where visual input is either unavailable or impractical. This research introduces a bilingual bimodal emotion recognition (BER) s system that integrates Mamba-based audio and text encoders within a Transformer-based cross-modal fusion. This architecture enables the system to improve its generalizability across an English-Russian corpus, comprising the MELD and RESD corpora. Bilingual bimodal fusion significantly outperforms the unimodal bilingual baselines (UAR=38.54% vs. 36.17% vs. 28.00% on MELD and UAR=67.89% vs. 37.20% vs. 60.79% on RESD). Moreover, the performance of the attention mechanism used for bimodal fusion depends on the corpus. Transformer-based attention is sufficient for shorter utterances in MELD, whereas Transformer- and graph-based attention yields better results for longer sequences in RESD. We also propose novel data augmentation strategies, including SDS and LS-LLM. The results show corpus-specific benefits: SDS improves performance on MELD, which contains utterances with variable durations; LS-LLM enhances results on RESD, which contains utterances with multiple emotional expressions. The combined use of both augmentation methods yields improvements across both corpora. Our multi-corpus training demonstrates strong bilingual generalization, while the single-corpus setup outperforms existing SOTA methods, achieving WF1=68.31% on MELD and WF1=85.25% on RESD.

---

## 📊 Key Features

- A novel Bilingual Bimodal Emotion Recognition Model for English-Russian generalization.

- A novel Transformer-based Cross-Modal Fusion Strategy for capturing intra- and inter- spatiotemporal audio-text features.

- A novel Stacked Data Sampling Strategy for merging same-label utterances into richer fixed-length emotional segments.

- A novel LLM-based Label Smoothing Strategy for correcting ambiguous or multi-emotion labels via soft distributions.

---

### 🧪 Branch Descriptions

This repository is organized into several branches to support different components of the BiBiER system:

| Branch | Description |
|--------|-------------|
| [`main`](https://github.com/LEYA-HSE/ESWA_2025/tree/main) | Default branch with final experimental results and app for the ESWA 2025 publication. |
| [`LLMs`](https://github.com/LEYA-HSE/ESWA_2025/tree/LLMs) | Large Language Model (LLM)-based emotion prediction from text. Includes evaluations of Qwen3-4B and Phi-4-mini-instruct on MELD and RESD. [[View results ↗](https://github.com/LEYA-HSE/ESWA_2025/tree/LLMs)] |
| [`audio-based`](https://github.com/LEYA-HSE/ESWA_2025/tree/audio-based) | Audio modality experiments using wav2vec2 and ExHuBERT as feature extractors. Includes training of LSTM, Transformer, and Mamba models with different architectures. Best results achieved using Mamba on wav2vec2 embeddings. [[See summary ↗](https://github.com/LEYA-HSE/ESWA_2025/tree/audio-based)] |
| [`text-based`](https://github.com/LEYA-HSE/ESWA_2025/tree/text-based) | Text-based emotion recognition using Jina embeddings and Mamba classifier. Includes evaluation on MELD and RESD with UAR. [[See results ↗](https://github.com/LEYA-HSE/ESWA_2025/tree/text-based)] |
| [`train`](https://github.com/LEYA-HSE/ESWA_2025/tree/train) | Core training pipeline for BiBiER. Integrates precomputed features from the best-performing audio and text models (e.g., wav2vec2 + Mamba, Jina + Mamba). Includes scripts for multimodal fusion, fine-tuning, and evaluation across MELD and RESD. Serves as the foundation for all final experiments reported in the paper. |

## 📝 Citation

If you use this work, please cite the following paper (currently under review):

```bibtex
@article{ryumina2025ber,
  title   = {BiBiER: A Bilingual Bimodal Emotion Recognition Method with Improved Generalization through Data Augmentation},
  author  = {Ryumina, Elena and Axyonov, Alexandr and Abdulkadirov, Timur and Koryakovskaya, Darya and Gorovaya, Svetlana and Bykova, Anna and Vikhorev, Dmitry and Ryumin, Dmitry},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {Under review}
}

