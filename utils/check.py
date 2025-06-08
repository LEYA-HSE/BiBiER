#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Проверка синтетического корпуса MELD-S:
    • существует ли WAV-файл;
    • правильные ли размеры аудио- и текст-эмбеддингов;
    • совпадает ли итоговый размер фич-вектора с ожиданием.

Результат:
    GOOD / BAD в консоль + CSV bad_synth_meld.csv (если нашли проблемы).
"""

from __future__ import annotations

import csv
import logging
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# ----------------------------------------------------------------------
# >>>>>>>>>     НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ (проверьте пути!)     <<<<<<<<<<<
# ----------------------------------------------------------------------
USER_CONFIG = {
    # пути к синтетике
    "synthetic_path": r"E:/MELD_S",
    "csv_name": "meld_s_train_labels.csv",
    "wav_subdir": "wavs",

    # модели / чекпойнты такие же, как в вашем config.toml
    "audio_model_name": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "audio_ckpt": "best_audio_model_2.pt",
    "text_model_name": "jinaai/jina-embeddings-v3",
    "text_ckpt": "best_text_model.pth",

    # общие параметры
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 16000,
    "num_emotions": 7,  # anger, disgust, fear, happy, neutral, sad, surprise
}

# ----------------------------------------------------------------------
# импорт собственных экстракторов
# ----------------------------------------------------------------------
try:
    from feature_extractor import (
        PretrainedAudioEmbeddingExtractor,
        PretrainedTextEmbeddingExtractor,
    )
except ModuleNotFoundError:
    try:
        # если файл лежит в data_loading/
        from data_loading.feature_extractor import (
            PretrainedAudioEmbeddingExtractor,
            PretrainedTextEmbeddingExtractor,
        )
    except ModuleNotFoundError as e:
        sys.exit(
            "❌  Не найден feature_extractor.py. "
            "Убедитесь, что он в PYTHONPATH или лежит рядом со скриптом."
        )

# ----------------------------------------------------------------------
# вспомогательные функции
# ----------------------------------------------------------------------
def build_audio_cfg() -> SimpleNamespace:
    """Готовим config-объект для PretrainedAudioEmbeddingExtractor."""
    return SimpleNamespace(
        audio_model_name=USER_CONFIG["audio_model_name"],
        emb_device=USER_CONFIG["device"],
        audio_pooling="mean",            # как в тренировке
        emb_normalize=False,
        max_audio_frames=0,
        audio_classifier_checkpoint=USER_CONFIG["audio_ckpt"],
        sample_rate=USER_CONFIG["sample_rate"],
        wav_length=4,
    )


def build_text_cfg() -> SimpleNamespace:
    """Config для PretrainedTextEmbeddingExtractor."""
    return SimpleNamespace(
        text_model_name=USER_CONFIG["text_model_name"],
        emb_device=USER_CONFIG["device"],
        text_pooling="mean",
        emb_normalize=False,
        max_tokens=95,
        text_classifier_checkpoint=USER_CONFIG["text_ckpt"],
    )


def get_dims(audio_extractor, text_extractor) -> Dict[str, int]:
    """Возвращает фактические размеры эмбеддингов (audio_dim, text_dim)."""
    sr = USER_CONFIG["sample_rate"]
    with torch.no_grad():
        dummy_wav = torch.zeros(1, sr)
        _, a_emb = audio_extractor.extract(dummy_wav[0], sr)
        audio_dim = a_emb[0].shape[-1]

        _, t_emb = text_extractor.extract("dummy text")
        text_dim = t_emb[0].shape[-1]

    return {"audio_dim": audio_dim, "text_dim": text_dim}


def check_row(
    row: pd.Series,
    feats: Dict[str, object],
    dims: Dict[str, int],
    wav_dir: Path,
) -> Optional[str]:
    """
    Возвращает None, если пример корректный, иначе строку-причину.
    """
    video = row["video_name"]
    wav_path = wav_dir / f"{video}.wav"
    text = row.get("text", "")

    try:
        if not wav_path.exists():
            return "file_missing"

        # ---------- аудио ----------
        wf, sr = torchaudio.load(str(wav_path))
        if sr != USER_CONFIG["sample_rate"]:
            wf = torchaudio.transforms.Resample(sr, USER_CONFIG["sample_rate"])(wf)

        a_pred, a_emb = feats["audio"].extract(wf[0], USER_CONFIG["sample_rate"])
        a_emb = a_emb[0]
        if a_emb.shape[-1] != dims["audio_dim"]:
            return f"audio_dim_{a_emb.shape[-1]}"

        # ---------- текст ----------
        t_pred, t_emb = feats["text"].extract(text)
        t_emb = t_emb[0]
        if t_emb.shape[-1] != dims["text_dim"]:
            return f"text_dim_{t_emb.shape[-1]}"

        # ---------- конкатенация ----------
        full_vec = torch.cat(
            [a_emb, t_emb, a_pred[0], t_pred[0]],
            dim=-1,
        )
        expected_all = (
            dims["audio_dim"]
            + dims["text_dim"]
            + 2 * USER_CONFIG["num_emotions"]
        )
        if full_vec.shape[-1] != expected_all:
            return f"concat_dim_{full_vec.shape[-1]}"

    except Exception as e:
        logging.error(f"{video}: {traceback.format_exc(limit=2)}")
        return "exception_" + e.__class__.__name__

    return None


# ----------------------------------------------------------------------
# основной скрипт
# ----------------------------------------------------------------------
def main() -> None:
    syn_root = Path(USER_CONFIG["synthetic_path"])
    csv_path = syn_root / USER_CONFIG["csv_name"]
    wav_dir = syn_root / USER_CONFIG["wav_subdir"]

    if not csv_path.exists():
        sys.exit(f"CSV не найден: {csv_path}")
    if not wav_dir.exists():
        sys.exit(f"WAV-директория не найдена: {wav_dir}")

    # 1. экстракторы
    audio_feat = PretrainedAudioEmbeddingExtractor(build_audio_cfg())
    text_feat = PretrainedTextEmbeddingExtractor(build_text_cfg())
    feats = {"audio": audio_feat, "text": text_feat}

    # 2. реальные размерности
    dims = get_dims(audio_feat, text_feat)
    expected_total = (
        dims["audio_dim"] + dims["text_dim"] + 2 * USER_CONFIG["num_emotions"]
    )
    print(
        f"Audio dim = {dims['audio_dim']}, "
        f"Text dim = {dims['text_dim']}, "
        f"Expected concat = {expected_total}"
    )

    # 3. правим CSV
    df = pd.read_csv(csv_path)
    bad_rows: List[Dict[str, str]] = []
    good_cnt = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking"):
        reason = check_row(row, feats, dims, wav_dir)
        if reason:
            bad_rows.append(
                {
                    "video_name": row["video_name"],
                    "reason": reason,
                    "wav_path": str(wav_dir / f"{row['video_name']}.wav"),
                }
            )
        else:
            good_cnt += 1

    # 4. отчёт
    print("\n========== SUMMARY ==========")
    print(f"✅ GOOD : {good_cnt}")
    print(f"❌ BAD  : {len(bad_rows)}")

    if bad_rows:
        out_csv = Path(__file__).with_name("bad_synth_meld.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["video_name", "reason", "wav_path"]
            )
            writer.writeheader()
            writer.writerows(bad_rows)
        print(f"\nСписок проблемных примеров сохранён: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
