# generate_from_emotion_csv.py

import os
import logging
import time
import random
import pandas as pd
from glob import glob
from synthetic_utils.dia_tts_wrapper import DiaTTSWrapper


def generate_from_emotion_csv(
    csv_path: str,
    emotion: str,
    output_dir: str,
    device: str = "cuda",
    max_samples: int = None
):
    out_dir = os.path.join(output_dir, emotion)
    wav_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    logging.info(f"🎙️ Эмоция: '{emotion}' | CSV: {csv_path}")
    logging.info(f"📥 Сохранение в: {wav_dir}")

    tts = DiaTTSWrapper(device=device)
    df = pd.read_csv(csv_path)

    if max_samples is not None:
        df = df.sample(n=max_samples)

    total_start = time.time()

    for idx, row in df.iterrows():
        text = row["text"]
        video_name = row.get("video_name", f"{emotion}_{idx}")
        filename_prefix = video_name

        start = time.time()
        tts.generate_and_save_audio(
            text=text,
            out_dir=wav_dir,
            filename_prefix=filename_prefix
        )
        elapsed = time.time() - start

        logging.info(f"[{emotion}] ✔ {filename_prefix}.wav")
        logging.info(f"       🗣️ Текст: {text[:100]}{'...' if len(text) > 100 else ''}")
        logging.info(f"       🎧 Маркер: (встроен в текст) | ⏱️ {elapsed:.2f} сек")

    total_elapsed = time.time() - total_start
    logging.info(f"✅ Эмоция '{emotion}' завершена | файлов: {len(df)} | ⏱️ {total_elapsed:.1f} сек\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    INPUT_DIR = "synthetic_data"
    OUTPUT_DIR = "tts_synthetic_final"
    DEVICE = "cuda"

    csv_files = glob(os.path.join(INPUT_DIR, "meld_synthetic_*.csv"))

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        # Извлекаем эмоцию по шаблону: meld_synthetic_<emotion>_*.csv
        try:
            emotion = filename.split("_")[2]
        except IndexError:
            logging.warning(f"⚠️ Пропускаем файл: {filename}")
            continue

        # Генерируем 3–5 случайных аудио для каждой эмоции
        # n = random.randint(2, 5)
        n = 1
        generate_from_emotion_csv(
            csv_path=csv_path,
            emotion=emotion,
            output_dir=OUTPUT_DIR,
            device=DEVICE,
            max_samples=n
        )
