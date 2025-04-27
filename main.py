# train.py
# coding: utf-8
import logging
import os
import shutil
import datetime
import whisper
import toml
# os.environ["HF_HOME"] = "models"

from utils.config_loader import ConfigLoader
from utils.logger_setup import setup_logger
from utils.search_utils import greedy_search, exhaustive_search
from training.train_utils import (
    make_dataset_and_loader,
    train_once
)
from data_loading.feature_extractor import PretrainedAudioEmbeddingExtractor, PretrainedTextEmbeddingExtractor

def main():

    #  Грузим конфиг
    base_config = ConfigLoader("config.toml")

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = f"results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # Настраиваем logging
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)

    base_config.show_config()
    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    #  Файл, куда будет писать наш жадный поиск
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    audio_feature_extractor= PretrainedAudioEmbeddingExtractor(base_config)
    text_feature_extractor = PretrainedTextEmbeddingExtractor(base_config)

    # Инициализируем Whisper-модель один раз
    logging.info(f"Инициализация Whisper: модель={base_config.whisper_model}, устройство={base_config.whisper_device}")
    whisper_model = whisper.load_model(base_config.whisper_model, device=base_config.whisper_device)

    # Делаем датасеты/лоадеры
    # Общий train_loader
    _, train_loader = make_dataset_and_loader(base_config, "train", audio_feature_extractor, text_feature_extractor, whisper_model)

    # Раздельные dev/test
    dev_loaders = []
    test_loaders = []

    for dataset_name in base_config.datasets:
        _, dev_loader = make_dataset_and_loader(base_config, "dev",  audio_feature_extractor, text_feature_extractor, whisper_model, only_dataset=dataset_name)
        _, test_loader = make_dataset_and_loader(base_config, "test",  audio_feature_extractor, text_feature_extractor, whisper_model, only_dataset=dataset_name)

        dev_loaders.append((dataset_name, dev_loader))
        test_loaders.append((dataset_name, test_loader))

    search_config = toml.load("search_params.toml")
    param_grid = dict(search_config["grid"])
    default_values = dict(search_config["defaults"])

    if base_config.search_type == "greedy":
        greedy_search(
            base_config       = base_config,
            train_loader      = train_loader,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_once,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            default_values    = default_values,
            csv_prefix        = csv_prefix
        )

    elif base_config.search_type == "exhaustive":
        exhaustive_search(
            base_config       = base_config,
            train_loader      = train_loader,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_once,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            csv_prefix        = csv_prefix
        )

    elif base_config.search_type == "none":
        logging.info("== Режим одиночной тренировки (без поиска параметров) ==")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = f"{csv_prefix}_single_{timestamp}.csv"

        train_once(
            config           = base_config,
            train_loader     = train_loader,
            dev_loaders      = dev_loaders,
            test_loaders     = test_loaders,
            metrics_csv_path = csv_file_path
        )

    else:
        raise ValueError(f"⛔️ Неверное значение search_type в конфиге: '{base_config.search_type}'. Используй 'greedy', 'exhaustive' или 'none'.")


if __name__ == "__main__":
    main()
