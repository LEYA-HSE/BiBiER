# A Novel Data Augmentation Method for Improving the Generalizability of Bilingual Bimodal Emotion Recognition Systems

This repository accompanies the publication in **Expert Systems with Applications (ESWA), 2025**:

> **A Novel Data Augmentation Method for Improving the Generalizability of Bilingual Bimodal Emotion Recognition Systems**
> Elena Ryumina, Alexandr Axyonov, Timur Abdulkadirov, Darya Koryakovskaya, Svetlana Gorovaya, Anna Bykova, Dmitry Vikhorev, Dmitry Ryumin
> HSE University, St. Petersburg

---

## 🧠 Abstract

Emotion recognition (ER) is essential for real-world human-computer interaction, but multimodal systems that use audio, text, and video are often too complex for practical applications. Bimodal audio-text systems provide a more feasible balance between accuracy and efficiency, particularly in scenarios where visual input is unavailable.

This research introduces a bilingual Bimodal Emotion Recognition (BER) system that integrates Mamba-based audio and text encoders within a Transformer-based cross-modal fusion. The architecture improves generalizability across an English-Russian corpus (MELD and RESD).

We propose three novel data augmentation strategies:
- **SDS (Stacked Data Sampling)** — merges short utterances with the same label to simulate longer sequences.
- **TUG (Template-based Utterance Generation)** — synthesizes new emotional utterances using ChatGPT-4o and generates audio via DIA-TTS with expressive control.
- **LS-LLM (Label Smoothing with Large Language Models)** — replaces one-hot labels with soft distributions predicted by LLMs.

Each method brings corpus-specific benefits. TUG improves performance on MELD (variable-length utterances), LS-LLM benefits RESD (emotionally blended content), and SDS helps both via sequence normalization. Combined use yields further performance gains.

Our best models achieve:
- **WF1 = 68.31%** on MELD
- **WF1 = 85.25%** on RESD

---

## 📊 Key Features

- Bilingual bimodal fusion using Mamba + Transformer + Graph attention
- Cross-corpus training with MELD (English) and RESD (Russian)
- **SDS (Stacked Data Sampling)** — context extension through multi-segment merging
- **TUG (Template-based Utterance Generation)** — ChatGPT-4o + DIA-TTS for expressive data synthesis with inline prosody tags
- **LS-LLM (Label Smoothing with LLMs)** — zero-shot probability distributions from Qwen3-4B and Phi-4-mini-instruct
- Corpus-specific optimization of augmentation ratios
- Evaluation across both single-corpus and multi-corpus setups

---

### Template-Based Utterance Generation

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
  title = {A Novel Data Augmentation Method for Improving the Generalizability of Bilingual Bimodal Emotion Recognition Systems},
  author = {Ryumina, Elena and Axyonov, Alexandr and Abdulkadirov, Timur and Koryakovskaya, Darya and Gorovaya, Svetlana and Bykova, Anna and Vikhorev, Dmitry and Ryumin, Dmitry},
  journal = {Expert Systems with Applications},
  year = {2025}
}
