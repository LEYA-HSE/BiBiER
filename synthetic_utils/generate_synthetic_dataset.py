import os
import logging
import time
import pandas as pd
from multiprocessing import Process
from synthetic_utils.dia_tts_wrapper import DiaTTSWrapper


def process_chunk(chunk_df, emotion, wav_dir, device, chunk_id):
    tts = DiaTTSWrapper(device=device)
    for idx, row in chunk_df.iterrows():
        text = row["text"]
        video_name = row.get("video_name", f"{emotion}_{chunk_id}_{idx}")
        filename_prefix = video_name

        try:
            result = tts.generate_and_save_audio(
                text=text,
                out_dir=wav_dir,
                filename_prefix=filename_prefix,
                use_timestamp=False,
                skip_if_exists=True,
                max_trim_duration=10.0
            )
            if result is None:
                logging.info(f"[{emotion}] ⏭️ Пропущено: {filename_prefix}.wav")
            else:
                logging.info(f"[{emotion}] ✔ {filename_prefix}.wav")
        except Exception as e:
            logging.error(f"[{emotion}] ❌ Ошибка: {filename_prefix} — {e}")


def generate_from_emotion_csv(
    csv_path: str,
    emotion: str,
    output_dir: str,
    device: str = "cuda",
    max_samples: int = None,
    num_processes: int = 1
):
    out_dir = os.path.join(output_dir, emotion)
    wav_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    logging.info(f"🎙️ Эмоция: '{emotion}' | CSV: {csv_path}")
    logging.info(f"📥 Сохранение в: {wav_dir}")

    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.sample(n=max_samples)

    chunk_size = len(df) // num_processes
    chunks = [df.iloc[i*chunk_size : (i+1)*chunk_size] for i in range(num_processes)]

    remainder = len(df) % num_processes
    if remainder > 0:
        chunks[-1] = pd.concat([chunks[-1], df.iloc[-remainder:]])

    total_start = time.time()

    processes = []
    for i, chunk in enumerate(chunks):
        p = Process(target=process_chunk, args=(chunk, emotion, wav_dir, device, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_elapsed = time.time() - total_start
    logging.info(f"✅ Эмоция '{emotion}' завершена | чанков: {num_processes} | ⏱️ {total_elapsed:.1f} сек\n")
