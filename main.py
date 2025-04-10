# train.py
# coding: utf-8
import logging
import os
import shutil
import datetime

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

    # Настраиваем logging
    log_file = os.path.join(results_dir, "greedy.log")
    setup_logger(logging.INFO, log_file=log_file)

    #  Грузим конфиг
    base_config = ConfigLoader("config.toml")
    base_config.show_config()

    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    #  Файл, куда будет писать наш жадный поиск
    overrides_file = os.path.join(results_dir, "overrides.txt")

    #  Делаем датасеты/лоадеры
    _, train_loader = make_dataset_and_loader(base_config, "train")
    _, dev_loader   = make_dataset_and_loader(base_config, "dev")
    _, test_loader  = make_dataset_and_loader(base_config, "test")

    #    Если хотим просто один раз обучить (без перебора),
    #    Можно вызвать train_once(base_config, train_loader, dev_loader, test_loader) напрямую.

    # Или же запустить жадный перебор гиперпараметров:
    param_grid = {
        "hidden_dim":             [128, 256, 512],
        # "hidden_dim_gated":       [128, 256, 512],
        "num_transformer_heads":  [2, 4, 8],
        "tr_layer_number":        [1, 2, 3],
        "out_features":           [128, 256, 512],
        # "num_graph_heads":        [2, 4, 8]
    }
    default_values = {
        "hidden_dim":             128,
        # "hidden_dim_gated":       128,
        "num_transformer_heads":  2,
        "tr_layer_number":        1,
        "out_features":           128,
        # "num_graph_heads":        2
    }

    # Вызываем сам перебор
    greedy_search(
        base_config       = base_config,
        train_loader      = train_loader,
        dev_loader        = dev_loader,
        test_loader       = test_loader,
        train_fn          = train_once,
        overrides_file    = overrides_file,
        param_grid        = param_grid,
        default_values    = default_values
    )

if __name__ == "__main__":
    main()
