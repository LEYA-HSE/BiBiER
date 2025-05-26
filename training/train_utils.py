# coding: utf-8
# train_utils.py

import torch
import logging
import random
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from typing import Type
import os
import datetime

from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

from utils.losses import WeightedCrossEntropyLoss
from utils.measures import uar, war, mf1, wf1
from models.models import (
    BiFormer, BiGraphFormer, BiGatedGraphFormer,
    PredictionsFusion, BiFormerWithProb, BiGatedFormer,
    BiMamba, BiMambaWithProb,BiGraphFormerWithProb, BiGatedGraphFormerWithProb
)
from utils.schedulers import SmartScheduler
from data_loading.dataset_multimodal import DatasetMultiModalWithPretrainedExtractors
from sklearn.utils.class_weight import compute_class_weight
from lion_pytorch import Lion


def get_smoothed_labels(audio_paths, original_labels, smooth_labels_df, smooth_mask, emotion_columns,  device):
    """
    audio_paths: список путей к аудиофайлам
    smooth_mask: тензор boolean с индексами для сглаживания
    Возвращает тензор сглаженных меток только для отмеченных примеров
    """

    # Получаем индексы для сглаживания
    smooth_indices = torch.where(smooth_mask)[0]

    # Создаем тензор для результатов (такого же размера как оригинальные метки)
    smoothed_labels = torch.zeros_like(original_labels)

    # print(smooth_labels_df, audio_paths)

    for idx in smooth_indices:
        audio_path = audio_paths[idx]
        # Получаем сглаженную метку из вашего DataFrame или другого источника
        smoothed_label = smooth_labels_df.loc[
            smooth_labels_df['video_name'] == audio_path[:-4],
            emotion_columns
        ].values[0]

        smoothed_labels[idx] = torch.tensor(smoothed_label, device=device)

    return smoothed_labels

def custom_collate_fn(batch):
    """Собирает список образцов в единый батч, отбрасывая None (невалидные)."""
    batch = [x for x in batch if x is not None]
    # print(batch[0].keys())
    if not batch:
        return None

    audios = [b["audio"] for b in batch]
    # audio_tensor = torch.stack(audios)
    audio_tensor = pad_sequence(audios, batch_first=True)

    labels = [b["label"] for b in batch]
    label_tensor = torch.stack(labels)

    texts = [b["text"] for b in batch]
    text_tensor = torch.stack(texts)

    audio_pred = [b["audio_pred"] for b in batch]
    audio_pred = torch.stack(audio_pred)

    text_pred = [b["text_pred"] for b in batch]
    text_pred = torch.stack(text_pred)

    return {
        "audio_paths": [b["audio_path"] for b in batch], # new
        "audio": audio_tensor,
        "label": label_tensor,
        "text": text_tensor,
        "audio_pred": audio_pred,
        "text_pred": text_pred,
    }

def get_class_weights_from_loader(train_loader, num_classes):
    """
    Вычисляет веса классов из train_loader, устойчиво к отсутствующим классам.
    Если какой-либо класс отсутствует в выборке, ему будет присвоен вес 0.0.

    :param train_loader: DataLoader с one-hot метками
    :param num_classes: Общее количество классов
    :return: np.ndarray весов длины num_classes
    """
    all_labels = []
    for batch in train_loader:
        if batch is None:
            continue
        all_labels.extend(batch["label"].argmax(dim=1).tolist())

    if not all_labels:
        raise ValueError("Нет ни одной метки в train_loader для вычисления весов классов.")

    present_classes = np.unique(all_labels)

    if len(present_classes) < num_classes:
        missing = set(range(num_classes)) - set(present_classes)
        logging.info(f"[!] Отсутствуют метки для классов: {sorted(missing)}")

    # Вычисляем веса только по тем классам, что есть
    weights_partial = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=all_labels
    )

    # Собираем полный вектор весов
    full_weights = np.zeros(num_classes, dtype=np.float32)
    for cls, w in zip(present_classes, weights_partial):
        full_weights[cls] = w

    return full_weights

def make_dataset_and_loader(config, split: str, audio_feature_extractor: Type = None, text_feature_extractor: Type = None, whisper_model: Type = None, only_dataset: str = None):
    """
    Универсальная функция: объединяет датасеты или возвращает один при only_dataset.
    При объединении train-датасетов — использует WeightedRandomSampler для балансировки.
    """
    datasets = []

    if not hasattr(config, "datasets") or not config.datasets:
        raise ValueError("⛔ В конфиге не указана секция [datasets].")

    for dataset_name, dataset_cfg in config.datasets.items():
        if only_dataset and dataset_name != only_dataset:
            continue

        csv_path = dataset_cfg["csv_path"].format(base_dir=dataset_cfg["base_dir"], split=split)
        wav_dir  = dataset_cfg["wav_dir"].format(base_dir=dataset_cfg["base_dir"], split=split)

        logging.info(f"[{dataset_name.upper()}] Split={split}: CSV={csv_path}, WAV_DIR={wav_dir}")

        dataset = DatasetMultiModalWithPretrainedExtractors(
            csv_path                = csv_path,
            wav_dir                 = wav_dir,
            emotion_columns         = config.emotion_columns,
            split                   = split,
            config                  = config,
            audio_feature_extractor = audio_feature_extractor,
            text_feature_extractor  = text_feature_extractor,
            whisper_model           = whisper_model,
            dataset_name            = dataset_name
        )

        datasets.append(dataset)

    if not datasets:
        raise ValueError(f"⚠️ Для split='{split}' не найдено ни одного подходящего датасета.")

    if len(datasets) == 1:

        full_dataset = datasets[0]
        loader = DataLoader(
            full_dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )
    else:
        # Несколько датасетов — собираем веса
        lengths = [len(d) for d in datasets]
        total = sum(lengths)

        logging.info(f"[!] Объединяем {len(datasets)} датасетов: {lengths} (total={total})")

        weights = []
        for d_len in lengths:
            w = 1.0 / d_len
            weights += [w] * d_len
            logging.info(f"  ➜ Сэмплы из датасета с {d_len} примерами получают вес {w:.6f}")

        full_dataset = ConcatDataset(datasets)

        if split == "train":
            sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)
            loader = DataLoader(
                full_dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=config.num_workers,
                collate_fn=custom_collate_fn
            )
        else:
            loader = DataLoader(
                full_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=custom_collate_fn
            )

    return full_dataset, loader

def run_eval(model, loader, criterion, model_name,  device="cuda"):
    """
    Оценка модели на loader'е. Возвращает (loss, uar, war, mf1, wf1).
    """
    model.eval()
    total_loss = 0.0
    total_preds = []
    total_targets = []
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue

            audio  = batch["audio"].to(device)
            labels = batch["label"].to(device)
            texts  = batch["text"]
            audio_pred  = batch["audio_pred"].to(device)
            text_pred = batch["text_pred"].to(device)

            if "fusion" in model_name:
                logits = model((audio_pred, text_pred))
            elif "withprob" in model_name:
                logits = model(audio, texts, audio_pred, text_pred)
            else:
                logits = model(audio, texts)
            target = labels.argmax(dim=1)

            loss = criterion(logits, target)
            bs = audio.shape[0]
            total_loss += loss.item() * bs
            total += bs

            preds = logits.argmax(dim=1)
            total_preds.extend(preds.cpu().numpy().tolist())
            total_targets.extend(target.cpu().numpy().tolist())

    avg_loss = total_loss / total

    uar_m = uar(total_targets, total_preds)
    war_m = war(total_targets, total_preds)
    mf1_m = mf1(total_targets, total_preds)
    wf1_m = wf1(total_targets, total_preds)

    return avg_loss, uar_m, war_m, mf1_m, wf1_m

def train_once(config, train_loader, dev_loaders, test_loaders, metrics_csv_path=None):
    """
    Логика обучения (train/dev/test).
    Возвращает лучшую метрику на dev и словарь метрик.
    """

    logging.info("== Запуск тренировки (train/dev/test) ==")

    checkpoint_dir = None
    if config.save_best_model:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = os.path.join("checkpoints", f"{config.model_name}_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    csv_writer = None
    csv_file = None

    if config.path_to_df_ls:
        df_ls = pd.read_csv(config.path_to_df_ls)

    if metrics_csv_path:
        csv_file = open(metrics_csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["split", "epoch", "dataset", "loss", "uar", "war", "mf1", "wf1", "mean"])


    # Seed
    if config.random_seed > 0:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(config.random_seed)
        logging.info(f"== Фиксируем random seed: {config.random_seed}")
    else:
        logging.info("== Random seed не фиксирован (0).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Экстракторы
    # audio_extractor = AudioEmbeddingExtractor(config)
    # text_extractor  = TextEmbeddingExtractor(config)

    # Параметры
    hidden_dim            = config.hidden_dim
    num_classes           = len(config.emotion_columns)
    num_transformer_heads = config.num_transformer_heads
    num_graph_heads       = config.num_graph_heads
    hidden_dim_gated      = config.hidden_dim_gated
    mamba_d_state         = config.mamba_d_state
    mamba_ker_size        = config.mamba_ker_size
    mamba_layer_number    = config.mamba_layer_number
    mode                  = config.mode
    weight_decay          = config.weight_decay
    momentum              = config.momentum
    positional_encoding   = config.positional_encoding
    dropout               = config.dropout
    out_features          = config.out_features
    lr                    = config.lr
    num_epochs            = config.num_epochs
    tr_layer_number       = config.tr_layer_number
    max_patience          = config.max_patience
    scheduler_type        = config.scheduler_type

    dict_models = {
        'BiFormer': BiFormer, # вход audio, texts
        'BiGraphFormer': BiGraphFormer, # вход audio, texts
        'BiGatedGraphFormer': BiGatedGraphFormer, # вход audio, texts
        "BiGatedFormer": BiGatedFormer, # вход audio, texts
        "BiMamba": BiMamba, # вход audio, texts
        "PredictionsFusion": PredictionsFusion, # вход audio_pred, text_pred
        "BiFormerWithProb": BiFormerWithProb, # вход audio, texts, audio_pred, text_pred
        "BiMambaWithProb": BiMambaWithProb, # вход audio, texts, audio_pred, text_pred
        "BiGraphFormerWithProb": BiGraphFormerWithProb, # вход audio, texts, audio_pred, text_pred
        "BiGatedGraphFormerWithProb": BiGatedGraphFormerWithProb,
    }

    model_cls = dict_models[config.model_name]
    model_name = config.model_name.lower()

    if model_name == 'predictionsfusion':
        model = model_cls().to(device)

    elif 'mamba' in model_name:
        # Особые параметры для Mamba-семейства
        model = model_cls(
            audio_dim             = config.audio_embedding_dim,
            text_dim              = config.text_embedding_dim,
            hidden_dim            = hidden_dim,
            mamba_d_state         = mamba_d_state,
            mamba_ker_size        = mamba_ker_size,
            mamba_layer_number    = mamba_layer_number,
            seg_len               = config.max_tokens,
            mode                  = mode,
            dropout               = dropout,
            positional_encoding   = positional_encoding,
            out_features          = out_features,
            device                = device,
            num_classes           = num_classes
        ).to(device)

    else:
        # Обычные модели
        model = model_cls(
            audio_dim             = config.audio_embedding_dim,
            text_dim              = config.text_embedding_dim,
            hidden_dim            = hidden_dim,
            hidden_dim_gated      = hidden_dim_gated,
            num_transformer_heads = num_transformer_heads,
            num_graph_heads       = num_graph_heads,
            seg_len               = config.max_tokens,
            mode                  = mode,
            dropout               = dropout,
            positional_encoding   = positional_encoding,
            out_features          = out_features,
            tr_layer_number       = tr_layer_number,
            device                = device,
            num_classes           = num_classes
        ).to(device)

    # Оптимизатор и лосс
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "lion":
        optimizer = Lion(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,momentum = momentum
        )
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"⛔ Неизвестный оптимизатор: {config.optimizer}")

    logging.info(f"Используем оптимизатор: {config.optimizer}, learning rate: {lr}")

    class_weights = get_class_weights_from_loader(train_loader, num_classes)
    criterion = WeightedCrossEntropyLoss(class_weights)

    logging.info("Class weights: " + ", ".join(f"{name}={weight:.4f}" for name, weight in zip(config.emotion_columns, class_weights)))

    # LR Scheduler
    steps_per_epoch = sum(1 for batch in train_loader if batch is not None)
    scheduler = SmartScheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        config=config,
        steps_per_epoch=steps_per_epoch
    )

    # Early stopping по dev
    best_dev_mean = float("-inf")
    best_dev_metrics = {}
    patience_counter = 0

    for epoch in range(num_epochs):
        logging.info(f"\n=== Эпоха {epoch} ===")
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_preds = []
        total_targets = []

        for batch in tqdm(train_loader):
            if batch is None:
                continue

            audio_paths =  batch["audio_paths"]  # new
            audio = batch["audio"].to(device)

            # Обработка меток с частичным сглаживанием
            if config.smoothing_probability == 0:
                labels = batch["label"].to(device)
            else:
                # Получаем оригинальные горячие метки
                original_labels = batch["label"].to(device)

                # Создаем маску для сглаживания (выбираем случайные примеры)
                batch_size = original_labels.size(0)
                smooth_mask = torch.rand(batch_size, device=device) < config.smoothing_probability

                # Получаем сглаженные метки для выбранных примеров
                smoothed_labels = get_smoothed_labels(audio_paths, original_labels, df_ls, smooth_mask, config.emotion_columns,  device)

                # Комбинируем метки
                labels = torch.where(
                    smooth_mask.unsqueeze(1),  # Добавляем размерность для broadcast
                    smoothed_labels.to(device),
                    original_labels
        )
            # print(labels)
            texts = batch["text"]
            audio_pred = batch["audio_pred"].to(device)
            text_pred = batch["text_pred"].to(device)

            if "fusion" in model_name:
                logits = model((audio_pred, text_pred))
            elif "withprob" in model_name:
                logits = model(audio, texts, audio_pred, text_pred)
            else:
                logits = model(audio, texts)

            target = labels.argmax(dim=1)
            loss   = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Если scheduler - One cycle или с Hugging Face
            scheduler.step(batch_level=True)

            bs = audio.shape[0]
            total_loss += loss.item() * bs

            preds = logits.argmax(dim=1)
            total_preds.extend(preds.cpu().numpy().tolist())
            total_targets.extend(target.cpu().numpy().tolist())
            total_samples += bs

        train_loss = total_loss / total_samples
        uar_m = uar(total_targets, total_preds)
        war_m = war(total_targets, total_preds)
        mf1_m = mf1(total_targets, total_preds)
        wf1_m = wf1(total_targets, total_preds)
        mean_train = np.mean([uar_m, war_m, mf1_m, wf1_m])

        logging.info(
            f"[TRAIN] Loss={train_loss:.4f}, UAR={uar_m:.4f}, WAR={war_m:.4f}, "
            f"MF1={mf1_m:.4f}, WF1={wf1_m:.4f}, MEAN={mean_train:.4f}"
        )

        # --- DEV ---
        dev_means = []
        dev_metrics_by_dataset = []

        for name, loader in dev_loaders:
            d_loss, d_uar, d_war, d_mf1, d_wf1 = run_eval(
                model, loader, criterion, model_name, device
            )
            d_mean = np.mean([d_uar, d_war, d_mf1, d_wf1])
            dev_means.append(d_mean)

            if csv_writer:
                csv_writer.writerow(["dev", epoch, name, d_loss, d_uar, d_war, d_mf1, d_wf1, d_mean])

            logging.info(
                f"[DEV:{name}] Loss={d_loss:.4f}, UAR={d_uar:.4f}, WAR={d_war:.4f}, "
                f"MF1={d_mf1:.4f}, WF1={d_wf1:.4f}, MEAN={d_mean:.4f}"
            )

            dev_metrics_by_dataset.append({
                "name": name,
                "loss": d_loss,
                "uar": d_uar,
                "war": d_war,
                "mf1": d_mf1,
                "wf1": d_wf1,
                "mean": d_mean,
            })

        mean_dev = np.mean(dev_means)
        scheduler.step(mean_dev)

        # --- TEST ---
        test_metrics_by_dataset = []
        for name, loader in test_loaders:
            t_loss, t_uar, t_war, t_mf1, t_wf1 = run_eval(
                model, loader, criterion, model_name, device
            )
            t_mean = np.mean([t_uar, t_war, t_mf1, t_wf1])
            logging.info(
                f"[TEST:{name}] Loss={t_loss:.4f}, UAR={t_uar:.4f}, WAR={t_war:.4f}, "
                f"MF1={t_mf1:.4f}, WF1={t_wf1:.4f}, MEAN={t_mean:.4f}"
            )

            test_metrics_by_dataset.append({
                "name": name,
                "loss": t_loss,
                "uar": t_uar,
                "war": t_war,
                "mf1": t_mf1,
                "wf1": t_wf1,
                "mean": t_mean,
            })

            if csv_writer:
                csv_writer.writerow(["test", epoch, name, t_loss, t_uar, t_war, t_mf1, t_wf1, t_mean])


        if mean_dev > best_dev_mean:
            best_dev_mean = mean_dev
            patience_counter = 0
            best_dev_metrics = {
                "mean": mean_dev,
                "by_dataset": dev_metrics_by_dataset
            }
            best_test_metrics = {
                "mean": np.mean([ds["mean"] for ds in test_metrics_by_dataset]),
                "by_dataset": test_metrics_by_dataset
            }

            if config.save_best_model:
                dev_str = f"{mean_dev:.4f}".replace(".", "_")
                model_path = os.path.join(checkpoint_dir, f"best_model_dev_{dev_str}_epoch_{epoch}.pt")
                torch.save(model.state_dict(), model_path)
                logging.info(f"💾 Модель сохранена по лучшему dev (эпоха {epoch}): {model_path}")

        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logging.info(f"Early stopping: {max_patience} эпох без улучшения.")
                break

    logging.info("Тренировка завершена. Все split'ы обработаны!")

    if csv_file:
        csv_file.close()

    return best_dev_mean, best_dev_metrics, best_test_metrics
