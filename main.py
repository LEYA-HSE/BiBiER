# train.py
# coding: utf-8
import logging
import os
import shutil
import datetime
# os.environ["HF_HOME"] = "models"

from utils.config_loader import ConfigLoader
from utils.logger_setup import setup_logger
from utils.greedy_search import greedy_search
from training.train_utils import (
    make_dataset_and_loader,
    train_once
)

def main():

    #  Создаём директорию для результатов, копируем config
    results_dir = f"results_greedy_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(results_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # Настраиваем logging
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)

    #  Грузим конфиг
    base_config = ConfigLoader("config.toml")
    base_config.show_config()

    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    #  Файл, куда будет писать наш жадный поиск
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    # Делаем датасеты/лоадеры
    # Общий train_loader
    _, train_loader = make_dataset_and_loader(base_config, "train")

    # Раздельные dev/test
    dev_loaders = []
    test_loaders = []

    for dataset_name in base_config.datasets:
        _, dev_loader = make_dataset_and_loader(base_config, "dev", only_dataset=dataset_name)
        _, test_loader = make_dataset_and_loader(base_config, "test", only_dataset=dataset_name)

        dev_loaders.append((dataset_name, dev_loader))
        test_loaders.append((dataset_name, test_loader))

    #    Если хотим просто один раз обучить (без перебора),
    #    Можно вызвать train_once(base_config, train_loader, dev_loader, test_loader) напрямую.

    # Или же запустить жадный перебор гиперпараметров:
    param_grid = {
        "hidden_dim":             [128, 256, 512],
        "hidden_dim_gated":       [128, 256, 512],
        "num_transformer_heads":  [2, 4, 8],
        "tr_layer_number":        [1, 2, 3],
        # "out_features":           [128, 256, 512],
        # "num_graph_heads":        [2, 4, 8]
    }
    default_values = {
        "hidden_dim":             128,
        "hidden_dim_gated":       128,
        "num_transformer_heads":  2,
        "tr_layer_number":        1,
        # "out_features":           128,
        # "num_graph_heads":        2
    }

    # Вызываем сам перебор
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

if __name__ == "__main__":
    main()
