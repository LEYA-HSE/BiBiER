# ESWA_2025


### Prompt for Template-Based Utterance Generation

This prompt is designed for generating structured emotional utterance templates using a large language model (LLM), based on the style and tone of the MELD dataset (Multimodal EmotionLines Dataset). The generated content is used to create **synthetic dialogue data** for tasks such as **emotion classification**, **expressive text-to-speech (TTS)**, and **multimodal affective computing**.

Each generated JSON defines reusable components (subjects, verbs, interjections, contexts, templates) that can be sampled and composed into thousands of emotion-specific utterances suitable for training or augmenting emotion-aware systems.

### Prompt:
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
