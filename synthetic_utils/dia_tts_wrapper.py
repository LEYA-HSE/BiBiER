# dia_tts_wrapper.py

import os
import time
import logging
import torch
import soundfile as sf
import numpy as np
from dia.model import Dia


class DiaTTSWrapper:
    def __init__(self, model_name="nari-labs/Dia-1.6B", device="cuda", dtype="float16"):
        """
        :param model_name: HuggingFace ID модели DIA-TTS.
        :param device: "cuda" или "cpu".
        :param dtype: float16 | bfloat16 | float32 — экономия памяти и ускорение.
        """
        self.device = device
        self.sr = 44100  # Частота дискретизации по умолчанию для Dia

        logging.info(f"[DiaTTS] Загрузка модели {model_name} на {device} (dtype={dtype})")
        self.model = Dia.from_pretrained(
            model_name,
            device=device,
            compute_dtype=dtype
        )

    def generate_audio_from_text(self, text: str, paralinguistic: str = "", max_duration: float = None) -> torch.Tensor:
        """
        Генерация аудио из текста с необязательной меткой эмоции (например, 'laughs').
        Возвращает тензор (1, num_samples).
        """
        try:
            if paralinguistic:
                clean = paralinguistic.strip("()").lower()
                text = f"{text} ({clean})"

            audio_np = self.model.generate(
                text,
                use_torch_compile=False,
                verbose=False
            )

            wf = torch.from_numpy(audio_np).float().unsqueeze(0)

            if max_duration:
                max_samples = int(self.sr * max_duration)
                wf = wf[:, :max_samples]

            return wf

        except Exception as e:
            logging.error(f"[DiaTTS] Ошибка генерации аудио: {e}")
            return torch.zeros(1, self.sr)

    def generate_and_save_audio(self, text: str, paralinguistic: str = "", out_dir="tts_outputs", filename_prefix="tts", max_duration: float = None) -> torch.Tensor:
        """
        Генерация и сохранение WAV-аудио.
        """
        os.makedirs(out_dir, exist_ok=True)
        wf = self.generate_audio_from_text(text, paralinguistic, max_duration)
        np_wf = wf.squeeze().cpu().numpy()

        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestr}.wav"
        out_path = os.path.join(out_dir, filename)

        sf.write(out_path, np_wf, self.sr)
        logging.info(f"[DiaTTS] 💾 Сохранено аудио: {out_path}")
        return wf

    def get_sample_rate(self):
        return self.sr
