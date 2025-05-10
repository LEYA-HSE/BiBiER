# Emotion Probability Prediction from Text using LLMs

## Evaluation Results for `Phi-4-mini-instruct_emotions_resd`

### Confusion Matrix

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

### Per-class Recall

| Emotion  | Recall  |
|:---------|:--------|
| Neutral  | 0.99346 |
| Happy    | 0.73563 |
| Sad      | 0.89231 |
| Anger    | 0.93714 |
| Surprise | 0.81646 |
| Disgust  | 0.90541 |
| Fear     | 0.92135 |
