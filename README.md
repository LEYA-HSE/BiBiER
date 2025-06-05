# BiBiER: A Bilingual Bimodal Emotion Recognition Method with Improved Generalization through Data Augmentation

This repository accompanies the publication in **Expert Systems with Applications (ESWA), 2025**:

> Elena Ryumina, Alexandr Axyonov, Timur Abdulkadirov, Darya Koryakovskaya, Svetlana Gorovaya, Anna Bykova, Dmitry Vikhorev, Dmitry Ryumin
> HSE University, St. Petersburg

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

---

### TUG: Template-Based Utterance Generation

**Description:**
This prompt is designed for generating structured emotional utterance templates using a large language model (LLM), based on the style and tone of the MELD corpus (Multimodal EmotionLines Dataset). The generated content is used to create synthetic dialogue data for tasks such as emotion classification, expressive text-to-speech (TTS), and multimodal affective computing.

Each generated JSON defines reusable components (`subjects`, `verbs`, `interjections`, `contexts`, `templates`) that can be sampled and composed into thousands of emotion-specific utterances suitable for training or augmenting emotion-aware systems.

All generations were performed using **ChatGPT-4o**, a GPT-4-based model provided by OpenAI, which demonstrated strong stylistic and emotional consistency aligned with the MELD corpus.


### 🟦 Prompt:
```
You are an expert language model trained to synthesize emotional utterance templates for expressive speech synthesis and affective data generation. Your task is to generate a structured set of phrase components based on examples from the MELD corpus.

Instructions:
- You will receive several short utterances labeled with a specific emotion.
- Analyze their emotional tone, linguistic style, and conversational patterns.
- Generate a valid JSON object with the following five fields:
    - subjects: 60–100 short noun phrases (e.g., "That moment", "The silence")
    - verbs: 60–100 expressive emotional reactions (e.g., "still hurts", "lit me up")
    - interjections: 20–40 emotional openings (e.g., "(sighs)...", "(laughs)...") using only the DIA TTS-supported sound tags
    - contexts: 60–100 sentence continuations expressing emotional aftermath
    - templates: 5–10 composition patterns using the placeholders:
        {s} = subject, {v} = verb, {i} = interjection, {c} = context
        Example: "{i}. {s} {v}, {c}."

Target emotion: one of neutral, happy, sad, anger, surprise, disgust, fear

Requirements:
- The output must match the tone and style of MELD utterances.
- Use only DIA TTS-compatible audio tags: (laughs), (sighs), (gasps), (groans), (clears throat), (inhales), (exhales), (coughs), (chuckle), (mumbles), (sniffs), (claps), (screams), (applause), (burps), (humming), (sneezes), (beep), (whistles), (singing), (sings)
- Output must be a valid JSON object without additional explanations or commentary.
- Utterances must be short (5–20 words) and expressive enough for TTS.

Input format:
Emotion label = {emotion}
Example utterances:
{text_1}
{text_2}
{text_3}
...

Output format:
A valid JSON dictionary with keys:
subjects, verbs, interjections, contexts, templates
```

---

### LS-LLM: Label Smoothing Generation based on Large Language Model

**Description:**
This prompt is used to implement a semantic-aware label smoothing technique for emotion classification. Instead of using a flat (uniform) smoothing across all non-target labels, this method leverages zero-shot large language models (LLMs) to generate a **context-informed probability distribution** over the emotion classes. The model is prompted to analyze the input text and output soft labels, where the **ground truth emotion is given the highest probability**, and other plausible emotions receive proportionally smaller weights.

This method is referred to as **LS-LLM** in our experiments and is compatible with any LLM that supports zero-shot text classification. It was tested using lightweight models (e.g., Phi-4-mini-instruct, Qwen3-4B), both under 4B parameters, as detailed in our comparison table.



### 🟦 Prompt:

```
You are an expert emotion analysis system. Analyze the following text and predict a probability distribution for the emotions: neutral, happy, sad, anger, surprise, disgust, fear.

Instructions:
- Carefully analyze the emotional content and nuances of the text.
- Predict a probability distribution where the sum of all probabilities is exactly 1.00000.
- Each value must have exactly 5 decimal places.
- Output the probabilities in this exact order, comma-separated, with no extra text: neutral, happy, sad, anger, surprise, disgust, fear.
- The ground truth emotion is "{label}". Ensure this emotion has the highest probability in your distribution.
- Do NOT assign 1.00000 to the ground truth emotion unless no other emotions are even slightly present.
- If other emotions are present, allocate realistic probabilities while keeping the ground truth highest.
- Ensure no two emotions have identical probability values.
- No probability should be exactly 0.00000 unless entirely absent.

Text: {text}

Output format:
neutral_prob, happy_prob, sad_prob, anger_prob, surprise_prob, disgust_prob, fear_prob
```


**Example usage:**

```
Text: I wish I could stop thinking about it, but it just keeps hurting.
Label: sad
```

Expected Output (sample):
```
0.02000, 0.01000, 0.82000, 0.08000, 0.05000, 0.01500, 0.00500
```


This prompt can be used in any fine-tuning or pseudo-labeling loop to dynamically generate soft labels aligned with LLM-informed emotion inference.

---


## 📝 Citation

If you use this work, please cite:

```bibtex
@article{ryumina2025ber,
  title = {BiBiER: A Bilingual Bimodal Emotion Recognition Method with Improved Generalization through Data Augmentation},
  author = {Ryumina, Elena and Axyonov, Alexandr and Abdulkadirov, Timur and Koryakovskaya, Darya and Gorovaya, Svetlana and Bykova, Anna and Vikhorev, Dmitry and Ryumin, Dmitry},
  journal = {Expert Systems with Applications},
  year = {2025}
}
