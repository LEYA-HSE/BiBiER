"""
File: emotion_inference.py
Author: Dmitry Ryumin
Description: Script for predicting emotion probability distributions in text using LLMs,
             reading input from a CSV file and saving results with predictions.
License: MIT License
"""

import warnings

import os
import re
import csv
import random
import torch
import torch.nn as nn
import polars as pl
import numpy as np
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

warnings.filterwarnings(
    "ignore", message=".*1Torch was not compiled with flash attention.*"
)


class SupportedLLMs:
    QWEN3_4B = "Qwen3-4B"
    PHI4_MINI = "Phi-4-mini-instruct"

    MODELS_REQUIRING_TRUST_REMOTE_CODE = {PHI4_MINI}
    MODELS_REQUIRING_PIPE = {PHI4_MINI}


@dataclass
class Config:
    base_path: Path = Path("D:/Dr.Ryumin/GitHub/EMNLP25")
    model_name: str = SupportedLLMs.QWEN3_4B
    model_path: Path = base_path / model_name
    input_csv: Path = base_path / "meld_train_labels.csv"
    output_csv: Path = base_path / f"{model_name}_emotions_meld.csv"
    log_file: Path = base_path / f"{model_name}_emotions_meld.txt"
    num_emotions: int = 7
    seed: int = 42
    batch_size: int = 1
    num_rows: int | None = 10
    epsilon: float = 1e-5
    max_tokens: int = 1024
    use_torch_compile: bool = True
    prompt_template: str = """
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
"""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def validate_model_name(model_name: str):
    valid_models = [
        v for k, v in SupportedLLMs.__dict__.items() if not k.startswith("__")
    ]
    if model_name not in valid_models:
        raise ValueError(
            f"Unsupported model: {model_name}. Available options: {valid_models}"
        )


def maybe_compile_model(model: nn.Module) -> nn.Module:
    if not Config.use_torch_compile:
        return model
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile().")
    except Exception as e:
        print(f"torch.compile() failed: {e}")
    return model


def postprocess_probs(
    probs: list[float], line_number: int, epsilon: float = 1e-5
) -> Tuple[list[float], bool]:
    probs = np.array(probs, dtype=np.float64)
    probs /= probs.sum()
    np.random.seed(line_number)

    while True:
        rounded = np.round(probs, 5)
        unique, counts = np.unique(rounded, return_counts=True)
        if all(counts == 1):
            break
        for val in unique[counts > 1]:
            idxs = np.where(rounded == val)[0]
            probs[idxs] += np.random.uniform(-epsilon, epsilon, size=len(idxs))
        probs /= probs.sum()

    probs = np.round(probs, 5)
    diff = round(1.0 - np.sum(probs), 5)
    if diff:
        probs[np.argmax(probs)] = round(probs[np.argmax(probs)] + diff, 5)

    return probs, round(np.sum(probs), 5) != 1.0


def load_model_and_tokenizer(
    model_path: Path,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Optional[pipeline]]:
    trust_remote_code = (
        Config.model_name in SupportedLLMs.MODELS_REQUIRING_TRUST_REMOTE_CODE
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if Config.model_name in SupportedLLMs.MODELS_REQUIRING_PIPE:
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
        )
        return model, tokenizer, pipe
    else:
        return model, tokenizer, None


def get_emotion_prediction_batch(
    texts,
    labels,
    idx_start,
    model,
    tokenizer,
    prompt_template,
    max_tokens=1024,
    pipe=None,
):
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt_template.format(label=label, text=text),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text, label in zip(texts, labels)
    ]

    if pipe:
        responses = pipe(prompts, max_new_tokens=max_tokens, return_full_text=False)
        decoded = [resp[0]["generated_text"] for resp in responses]
    else:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        generated = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    all_probs, all_responses, error_count = [], [], 0
    for j, response in enumerate(decoded):
        response_clean = (
            (response.split("assistant")[-1] if "assistant" in response else response)
            .strip()
            .splitlines()[-1]
        )
        numbers = re.findall(r"\d\.\d+", response_clean)
        if len(numbers) == Config.num_emotions:
            probs, error = postprocess_probs([float(n) for n in numbers], idx_start + j)
            error_count += error
        else:
            print(f"Parsing error: {response_clean}")
            probs = [None] * 7
        all_probs.append(probs)
        all_responses.append(response_clean)

    return all_probs, all_responses, error_count


def print_confusion_matrix(
    gt_labels: pl.Series, pred_labels: pl.Series, emotions: list[str]
) -> None:
    matrix = {e: {e_: 0 for e_ in emotions} for e in emotions}
    for true, pred in zip(gt_labels, pred_labels):
        if true in emotions and pred in emotions:
            matrix[true][pred] += 1

    data = [[true] + [matrix[true][pred] for pred in emotions] for true in emotions]
    print(tabulate(data, headers=["True/Pred"] + emotions, tablefmt="fancy_grid"))


def evaluate_model_performance(
    pred_csv: Path, num_rows: Optional[int] = None
) -> Tuple[float, Dict[str, float]]:
    emotions = ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"]
    pred = pl.read_csv(pred_csv)
    if num_rows:
        pred = pred.head(num_rows)
    true_labels, pred_labels = pred["ground_truth"], pred["llm_best_label"]

    print("\nUnique Ground Truth Labels:", true_labels.unique().to_list())
    print("Unique Predicted Labels:", pred_labels.unique().to_list())
    print_confusion_matrix(true_labels, pred_labels, emotions)

    mismatches = (true_labels != pred_labels).sum()
    print(f"\nTotal mismatches: {mismatches} out of {pred.height}")

    recalls = {
        e: (
            (
                true_labels.filter(true_labels == e)
                == pred_labels.filter(true_labels == e)
            ).sum()
            / (true_labels == e).sum()
            if (true_labels == e).sum() > 0
            else 0.0
        )
        for e in emotions
    }
    uar = sum(recalls.values()) / len(emotions)
    print(f"\nUnweighted Average Recall (UAR): {uar:.5f}")

    for e, val in recalls.items():
        print(f"{e.capitalize()}: {val:.5f}")

    return uar, recalls


def main():
    set_seed(Config.seed)

    validate_model_name(Config.model_name)

    model, tokenizer, pipe = load_model_and_tokenizer(Config.model_path)
    model = maybe_compile_model(model)

    df = pl.read_csv(Config.input_csv)
    if Config.num_rows:
        df = df.head(Config.num_rows)

    for file in [Config.output_csv, Config.log_file]:
        if file.exists():
            os.remove(file)

    header = [
        "video_name",
        "start_time",
        "end_time",
        "sentiment",
        "text",
        "ground_truth",
        "llm_best_label",
        "neutral",
        "happy",
        "sad",
        "anger",
        "surprise",
        "disgust",
        "fear",
        "sum_of_emotions",
        "response_clean",
    ]

    with open(Config.output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    log_file = open(Config.log_file, "w", encoding="utf-8", buffering=1)

    batch_texts, batch_labels, batch_meta = [], [], []

    pbar = tqdm(enumerate(df.iter_rows(named=True)), total=len(df))

    error_count = 0
    emotions = ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"]

    for idx, row in pbar:
        text = row["text"]
        label = emotions[[row[e] for e in emotions].index(1)]
        batch_texts.append(text)
        batch_labels.append(label)
        batch_meta.append(
            (row["video_name"], row["start_time"], row["end_time"], row["sentiment"])
        )

        if len(batch_texts) == Config.batch_size or idx == len(df) - 1:
            probs_batch, responses_batch, errors = get_emotion_prediction_batch(
                batch_texts,
                batch_labels,
                idx - len(batch_texts) + 1,
                model,
                tokenizer,
                Config.prompt_template,
                Config.max_tokens,
                pipe,
            )
            error_count += errors

            with open(Config.output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for i, probs in enumerate(probs_batch):
                    best_label = emotions[np.argmax(probs)]
                    sum_probs = round(sum(probs), 5)
                    writer.writerow(
                        [
                            *batch_meta[i],
                            batch_texts[i],
                            batch_labels[i],
                            best_label,
                            *probs,
                            sum_probs,
                            responses_batch[i],
                        ]
                    )

            batch_texts, batch_labels, batch_meta = [], [], []

    log_file.close()
    print(f"Done! Results saved to {Config.output_csv}")
    print(f"Logs saved to {Config.log_file}")
    print(f"Total errors (sum != 1.0): {error_count}")
    evaluate_model_performance(Config.output_csv, Config.num_rows)


if __name__ == "__main__":
    main()
