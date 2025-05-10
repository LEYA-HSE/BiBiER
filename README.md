# Emotion Probability Prediction from Text using LLMs

## Summary of UAR Results

### RESD Corpus

| Model                | UAR    |
|:--------------------|:--------|
| Qwen3-4B            | 0.95079 |
| Phi-4-mini-instruct | 0.88597 |

### MELD Corpus

| Model               | UAR     |
|:--------------------|:--------|
| Qwen3-4B            | 0.95741 |
| Phi-4-mini-instruct | 0.91990 |

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

## Evaluation Results for [Qwen3-4B_emotions_meld](https://github.com/LEYA-HSE/ESWA_2025/blob/LLMs/corpora/Qwen3-4B_emotions_meld.csv)

### Confusion Matrix `Qwen3-4B_emotions_meld`

| True / Pred | Neutral | Happy | Sad | Anger | Surprise | Disgust | Fear |
|:------------|:--------|:------|:----|:------|:---------|:--------|:-----|
| Neutral     | 4710    | 0     | 0   | 0     | 0        | 0       | 0    |
| Happy       | 0       | 1743  | 0   | 0     | 0        | 0       | 0    |
| Sad         | 0       | 1     | 682 | 0     | 0        | 0       | 0    |
| Anger       | 0       | 12    | 10  | 1087  | 0        | 0       | 0    |
| Surprise    | 4       | 25    | 4   | 1     | 1171     | 0       | 0    |
| Disgust     | 2       | 3     | 8   | 16    | 4        | 238     | 0    |
| Fear        | 8       | 9     | 14  | 2     | 1        | 0       | 234  |

**Total mismatches:** 124 out of 9989
**Unweighted Average Recall (UAR):** 0.95741

### Per-class Recall `Qwen3-4B_emotions_meld`

| Emotion  | Recall  |
|:---------|:--------|
| Neutral  | 1.00000 |
| Happy    | 1.00000 |
| Sad      | 0.99854 |
| Anger    | 0.98016 |
| Surprise | 0.97178 |
| Disgust  | 0.87823 |
| Fear     | 0.87313 |

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

## Evaluation Results for [Phi-4-mini-instruct_emotions_meld](https://github.com/LEYA-HSE/ESWA_2025/blob/LLMs/corpora/Phi-4-mini-instruct_emotions_meld.csv)

### Confusion Matrix `Phi-4-mini-instruct_emotions_meld`

| True / Pred | Neutral | Happy | Sad | Anger | Surprise | Disgust | Fear |
|:------------|:--------|:------|:----|:------|:---------|:--------|:-----|
| Neutral     | 4707    | 0     | 0   | 1     | 0        | 0       | 2    |
| Happy       | 39      | 1410  | 10  | 36    | 24       | 0       | 224  |
| Sad         | 3       | 7     | 646 | 9     | 12       | 1       | 5    |
| Anger       | 21      | 35    | 2   | 1013  | 17       | 1       | 20   |
| Surprise    | 37      | 18    | 5   | 19    | 1126     | 0       | 0    |
| Disgust     | 7       | 2     | 2   | 5     | 6        | 235     | 14   |
| Fear        | 1       | 2     | 0   | 2     | 3        | 0       | 260  |

**Total mismatches:** 592 out of 9989
**Unweighted Average Recall (UAR):** 0.91990

### Per-class Recall `Phi-4-mini-instruct_emotions_meld`

| Emotion  | Recall  |
|:---------|:--------|
| Neutral  | 0.99936 |
| Happy    | 0.80895 |
| Sad      | 0.94583 |
| Anger    | 0.91344 |
| Surprise | 0.93444 |
| Disgust  | 0.86716 |
| Fear     | 0.97015 |
