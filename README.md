# Emotion Probability Prediction from Text using LLMs

## Evaluation Results for [Qwen3-4B_emotions_resd](https://github.com/LEYA-HSE/ESWA_2025/blob/LLMs/corpora/Qwen3-4B_emotions_resd.csv)

### Confusion Matrix `Qwen3-4B_emotions_resd`

| True / Pred | Neutral | Happy | Sad | Anger | Surprise | Disgust | Fear |
|:------------|:--------|:------|:----|:------|:---------|:--------|:-----|
| Neutral     | 153     | 0     | 0   | 0     | 0        | 0       | 0    |
| Happy       | 1       | 173   | 0   | 0     | 0        | 0       | 0    |
| Sad         | 1       | 0     | 129 | 0     | 0        | 0       | 0    |
| Anger       | 0       | 0     | 2   | 173   | 0        | 0       | 0    |
| Surprise    | 1       | 2     | 2   | 0     | 153      | 0       | 0    |
| Disgust     | 3       | 2     | 9   | 6     | 1        | 127     | 0    |
| Fear        | 6       | 2     | 17  | 0     | 1        | 0       | 152  |

**Total mismatches:** 56 out of 1116
**Unweighted Average Recall (UAR):** 0.95079

### Per-class Recall `Qwen3-4B_emotions_resd`

| Emotion  | Recall  |
|:---------|:--------|
| Neutral  | 1.00000 |
| Happy    | 0.99425 |
| Sad      | 0.99231 |
| Anger    | 0.98857 |
| Surprise | 0.96835 |
| Disgust  | 0.85811 |
| Fear     | 0.85393 |

## Evaluation Results for [Phi-4-mini-instruct_emotions_resd](https://github.com/LEYA-HSE/ESWA_2025/blob/LLMs/corpora/Phi-4-mini-instruct_emotions_resd.csv)

### Confusion Matrix `Phi-4-mini-instruct_emotions_resd`

| True / Pred | Neutral | Happy | Sad | Anger | Surprise | Disgust | Fear |
|:------------|:--------|:------|:----|:------|:---------|:--------|:-----|
| Neutral     | 152     | 1     | 0   | 0     | 0        | 0       | 0    |
| Happy       | 6       | 128   | 2   | 3     | 1        | 0       | 34   |
| Sad         | 3       | 5     | 116 | 2     | 3        | 0       | 1    |
| Anger       | 4       | 4     | 1   | 164   | 1        | 0       | 1    |
| Surprise    | 17      | 10    | 0   | 1     | 129      | 1       | 0    |
| Disgust     | 4       | 0     | 2   | 2     | 2        | 134     | 4    |
| Fear        | 9       | 0     | 1   | 2     | 2        | 0       | 164  |

**Total mismatches:** 129 out of 1116
**Unweighted Average Recall (UAR):** 0.88597

### Per-class Recall `Phi-4-mini-instruct_emotions_resd`

| Emotion  | Recall  |
|:---------|:--------|
| Neutral  | 0.99346 |
| Happy    | 0.73563 |
| Sad      | 0.89231 |
| Anger    | 0.93714 |
| Surprise | 0.81646 |
| Disgust  | 0.90541 |
| Fear     | 0.92135 |
