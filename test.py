# tts_test_run.py

import os
from utils.config_loader import ConfigLoader
from synthetic_utils.dia_tts_wrapper import DiaTTSWrapper
from generate_synthetic_dataset import PARALINGUISTIC_MARKERS

# Загружаем конфиг
config = ConfigLoader("config.toml")

# Настройка TTS
tts = DiaTTSWrapper(device=config.whisper_device)

# Пример текста и эмоции
text = "I'm just testing how this emotional voice sounds."
emotion = "neutral"  # можно: neutral, happy, sad, anger, fear, surprise, disgust
marker = PARALINGUISTIC_MARKERS.get(emotion, "")

# Генерация и сохранение
tts.generate_and_save_audio(
    text=text,
    paralinguistic=marker,
    out_dir="tts_test_outputs",
    filename_prefix=f"test_{emotion}",
    max_duration=5.0
)

print(f"✅ Аудио для эмоции '{emotion}' сохранено.")
