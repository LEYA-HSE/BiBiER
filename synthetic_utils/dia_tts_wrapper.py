import os
import time
import logging
import torch
import soundfile as sf
import numpy as np
from dia.model import Dia


class DiaTTSWrapper:
    def __init__(self, model_name="nari-labs/Dia-1.6B", device="cuda", dtype="float16"):
        self.device = device
        self.sr = 44100
        logging.info(f"[DiaTTS] Загрузка модели {model_name} на {device} (dtype={dtype})")
        self.model = Dia.from_pretrained(
            model_name,
            device=device,
            compute_dtype=dtype
        )

    def generate_audio_from_text(self, text: str, paralinguistic: str = "", max_duration: float = None) -> torch.Tensor:
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

    def generate_and_save_audio(
        self,
        text: str,
        paralinguistic: str = "",
        out_dir="tts_outputs",
        filename_prefix="tts",
        max_duration: float = None,
        use_timestamp=True,
        skip_if_exists=True,
        max_trim_duration: float = None
    ) -> torch.Tensor:
        os.makedirs(out_dir, exist_ok=True)
        if use_timestamp:
            timestr = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestr}.wav"
        else:
            filename = f"{filename_prefix}.wav"
        out_path = os.path.join(out_dir, filename)

        if skip_if_exists and os.path.exists(out_path):
            logging.info(f"[DiaTTS] ⏭️ Пропущено — уже существует: {out_path}")
            return None

        wf = self.generate_audio_from_text(text, paralinguistic, max_duration)
        np_wf = wf.squeeze().cpu().numpy()

        if max_trim_duration is not None:
            max_len = int(self.sr * max_trim_duration)
            if len(np_wf) > max_len:
                logging.info(f"[DiaTTS] ✂️ Обрезка аудио до {max_trim_duration} сек.")
                np_wf = np_wf[:max_len]

        sf.write(out_path, np_wf, self.sr)
        logging.info(f"[DiaTTS] 💾 Сохранено аудио: {out_path}")
        return wf

    def get_sample_rate(self):
        return self.sr
