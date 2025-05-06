# generate_synthetic_dataset.py

import os
import csv
import logging
import random
from datetime import datetime

import pandas as pd
from utils.config_loader import ConfigLoader
from synthetic_utils.text_generation import TextGenerator
from synthetic_utils.dia_tts_wrapper import DiaTTSWrapper  # Заменили на новый TTS

# === Сопоставление эмоций и паралингвистических эффектов ===
PARALINGUISTIC_MARKERS = {
    "neutral": "",
    "happy": "laughs",
    "sad": "sighs",
    "anger": "shouts",
    "surprise": "gasps",
    "disgust": "groans",
    "fear": "whispers"
}

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # deepseek-ai/deepseek-llm-1.3b-base или любая другая модель
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # deepseek-ai/deepseek-llm-1.3b-base или любая другая модель

def collect_partial_texts(csv_path, text_column, emotion_columns):
    texts_by_emotion = {e: [] for e in emotion_columns}
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        text = row.get(text_column, "")
        if not isinstance(text, str) or not text.strip():
            continue

        try:
            scores = [float(row[e]) for e in emotion_columns]
            dominant_idx = int(pd.Series(scores).idxmax())
            emotion = emotion_columns[dominant_idx]
            texts_by_emotion[emotion].append(text.strip())
        except Exception as e:
            logging.warning(f"⚠️ Ошибка обработки строки: {e}")
            continue

    return texts_by_emotion

def generate_synthetic_dataset_for_single_corpus(
    csv_path,
    emotion_columns,
    text_column,
    samples_per_emotion=50,
    output_dir="synthetic_meld_data",
    model_config=None
):
    os.makedirs(output_dir, exist_ok=True)
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"📦 Загрузка TextGenerator и DiaTTSWrapper")

    text_gen = TextGenerator(
        model_name=model_config.model_name,
        device=model_config.whisper_device,
        max_new_tokens=model_config.max_new_tokens,
        temperature=model_config.temperature,
        top_p=model_config.top_p,
        seed=model_config.random_seed
    )

    tts = DiaTTSWrapper(device=model_config.whisper_device)

    texts_by_emotion = collect_partial_texts(csv_path, text_column, emotion_columns)

    csv_out = os.path.join(output_dir, "metadata.csv")
    with open(csv_out, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["emotion", "partial_text", "full_text", "audio_file"])

        for emotion in emotion_columns:
            logging.info(f"\n🎭 Эмоция: {emotion}")
            partial_pool = texts_by_emotion.get(emotion, [])
            marker = PARALINGUISTIC_MARKERS.get(emotion, "")

            for i in range(samples_per_emotion):
                partial = random.choice(partial_pool) if partial_pool else ""
                full_text = text_gen.generate_text(emotion, partial)

                filename_prefix = f"{emotion}_{i}_{datetime.now().strftime('%H%M%S')}"
                tts.generate_and_save_audio(
                    text=full_text,
                    paralinguistic=marker,
                    out_dir=wav_dir,
                    filename_prefix=filename_prefix
                )

                audio_file = f"{filename_prefix}.wav"
                writer.writerow([emotion, partial, full_text, audio_file])
                logging.info(f"[✔] Сохранено: {audio_file}")

    logging.info(f"\n✅ Синтетический датасет сохранён в: {output_dir}")


if __name__ == "__main__":
    config = ConfigLoader("config.toml")

    dataset_name = "meld"
    dataset_cfg = config.datasets[dataset_name]
    base_dir = dataset_cfg["base_dir"]
    csv_path = dataset_cfg["csv_path"].format(base_dir=base_dir, split="train")

    generate_synthetic_dataset_for_single_corpus(
        csv_path=csv_path,
        emotion_columns=config.emotion_columns,
        text_column=config.text_column,
        samples_per_emotion=1,
        output_dir="synthetic_meld_data",
        model_config=config
    )
