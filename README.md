# ESWA_2025


### Template-Based Utterance Generation

**Description:**
This prompt is designed for generating structured emotional utterance templates using a large language model (LLM), based on the style and tone of the MELD corpus (Multimodal EmotionLines Dataset). The generated content is used to create synthetic dialogue data for tasks such as emotion classification, expressive text-to-speech (TTS), and multimodal affective computing.

Each generated JSON defines reusable components (`subjects`, `verbs`, `interjections`, `contexts`, `templates`) that can be sampled and composed into thousands of emotion-specific utterances suitable for training or augmenting emotion-aware systems.

All generations were performed using **ChatGPT-4o**, a GPT-4-based model provided by OpenAI, which demonstrated strong stylistic and emotional consistency aligned with the MELD corpus.


### 🟦 Prompt:
```
You are an expert language model trained to synthesize emotional utterance templates for expressive speech synthesis and affective data generation. Your task is to generate a structured set of phrase components based on examples from the MELD dataset.

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

### LS-LLM: Label Smoothing with Large Language Models

**Description:**
This prompt is used to implement a semantic-aware label smoothing technique for emotion classification. Instead of using a flat (uniform) smoothing across all non-target labels, this method leverages zero-shot large language models (LLMs) to generate a **context-informed probability distribution** over the emotion classes. The model is prompted to analyze the input text and output soft labels, where the **ground truth emotion is given the highest probability**, and other plausible emotions receive proportionally smaller weights.

This approach is referred to as **LS-LLM** in our experiments and is compatible with any LLM that supports zero-shot text classification. It was tested using lightweight models (e.g., Phi-4-mini-instruct, Qwen3-4B), both under 4B parameters, as detailed in our comparison table.



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
