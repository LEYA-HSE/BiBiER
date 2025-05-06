# parler_tts_wrapper.py

import torch
import soundfile as sf
import time
import os
import logging
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

class ParlerTTS:
    def __init__(self, model_name="parler-tts/parler-tts-mini-v1", device="cuda"):
        self.device = device
        logging.info(f"[ParlerTTS] Загрузка модели {model_name} на {device} ...")

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sr = self.model.config.sampling_rate

    def generate_audio_from_text(self, text: str, description: str) -> torch.Tensor:
        """
        Генерирует аудио (без сохранения на диск).
        Возвращает PyTorch-тензор формы (1, num_samples).
        """
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )

        audio_arr = generation.cpu().numpy().squeeze()  # (samples,)
        wf = torch.from_numpy(audio_arr).unsqueeze(0)    # -> (1, samples)
        return wf

    def generate_and_save_audio(self, text: str, description: str, out_dir="tts_outputs", filename_prefix="tts") -> torch.Tensor:
        """
        Генерирует аудио И сохраняет результат в WAV-файл (для отладки/проверки).
        Возвращает PyTorch-тензор (1, num_samples).
        """
        os.makedirs(out_dir, exist_ok=True)

        wf = self.generate_audio_from_text(text, description)
        np_wf = wf.squeeze().cpu().numpy()

        # Формируем имя файла
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestr}.wav"
        out_path = os.path.join(out_dir, filename)

        # Сохраняем
        sf.write(out_path, np_wf, self.sr)
        logging.info(f"[ParlerTTS] Сохранено аудио: {out_path}")

        return wf

    def get_sample_rate(self):
        return self.sr
