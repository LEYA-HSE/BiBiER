# coding: utf-8
# train_utils.py

import os
import torch
import logging
import random
import datetime
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from utils.losses import WeightedCrossEntropyLoss
from utils.measures import uar, war, mf1, wf1
from utils.logger_setup import setup_logger
from models.models import BiFormer, BiGraphFormer, BiGatedGraphFormer
from data_loading.dataset_multimodal import DatasetMultiModal
from data_loading.feature_extractor import AudioEmbeddingExtractor, TextEmbeddingExtractor

def custom_collate_fn(batch):
    """Собирает список образцов в единый батч, отбрасывая None (невалидные)."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    audios = [b["audio"] for b in batch]
    audio_tensor = torch.stack(audios)

    labels = [b["label"] for b in batch]
    label_tensor = torch.stack(labels)

    texts = [b["text"] for b in batch]

    return {
        "audio": audio_tensor,
        "label": label_tensor,
        "text": texts
    }

def make_dataset_and_loader(config, split: str):
    """
    Создаёт (dataset, dataloader) для указанного сплита: train/dev/test.
    """
    csv_path = config.csv_path.format(base_dir=config.base_dir, split=split)
    wav_dir  = config.wav_dir.format(base_dir=config.base_dir, split=split)
    print(f"{csv_path} {wav_dir} {split}")

    dataset = DatasetMultiModal(
        csv_path = csv_path,
        wav_dir  = wav_dir,
        emotion_columns = config.emotion_columns,
        split          = split,
        sample_rate    = config.sample_rate,
        wav_length     = config.wav_length,
        whisper_model  = config.whisper_model,
        text_column    = config.text_column,
        use_whisper_for_nontrain_if_no_text = config.use_whisper_for_nontrain_if_no_text,
        whisper_device = config.whisper_device,
        subset_size    = config.subset_size,
        merge_probability = config.merge_probability
    )

    shuffle = (split == "train")
    loader = DataLoader(
        dataset,
        batch_size  = config.batch_size,
        shuffle     = shuffle,
        num_workers = config.num_workers,
        collate_fn  = custom_collate_fn
    )
    return dataset, loader

def run_eval(model, loader, audio_extractor, text_extractor, criterion, device="cuda"):
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

            audio_emb = audio_extractor.extract(audio)
            text_emb  = text_extractor.extract(texts)

            logits = model(audio_emb, text_emb)
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

def train_once(config, train_loader, dev_loader, test_loader):
    """
    Логика обучения (train/dev/test).
    Возвращает лучшую метрику на dev и словарь метрик.
    """
    # Лог-файл
    os.makedirs("logs", exist_ok=True)
    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join("logs", f"train_log_{datestr}.txt")

    setup_logger(logging.INFO, log_file=log_file)
    logging.info("== Запуск тренировки (train/dev/test) ==")

    # Seed
    if config.random_seed > 0:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        logging.info(f"== Фиксируем random seed: {config.random_seed}")
    else:
        logging.info("== Random seed не фиксирован (0).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Экстракторы
    audio_extractor = AudioEmbeddingExtractor(config)
    text_extractor  = TextEmbeddingExtractor(config)

    # Параметры
    hidden_dim            = config.hidden_dim
    num_classes           = len(config.emotion_columns)
    num_transformer_heads = config.num_transformer_heads
    num_graph_heads       = config.num_graph_heads
    hidden_dim_gated      = config.hidden_dim_gated
    mode                  = config.mode
    positional_encoding   = config.positional_encoding
    dropout               = config.dropout
    out_features          = config.out_features
    lr                    = config.lr
    num_epochs            = config.num_epochs
    tr_layer_number       = config.tr_layer_number

    dict_models = {
        'BiFormer': BiFormer,
        'BiGraphFormer': BiGraphFormer,
        'BiGatedGraphFormer': BiGatedGraphFormer,
        # 'MultiModalTransformer_v5': MultiModalTransformer_v5,
        # 'MultiModalTransformer_v4': MultiModalTransformer_v4,
        # 'MultiModalTransformer_v3': MultiModalTransformer_v3
    }

    model_cls = dict_models[config.model_name]
    model = model_cls(
        audio_dim=config.audio_embedding_dim,
        text_dim=config.text_embedding_dim,
        hidden_dim=hidden_dim,
        hidden_dim_gated=hidden_dim_gated,
        num_transformer_heads=num_transformer_heads,
        num_graph_heads=num_graph_heads,
        seg_len=config.max_tokens,
        mode=mode,
        dropout=dropout,
        positional_encoding=positional_encoding,
        out_features=out_features,
        tr_layer_number=tr_layer_number,
        device=device,
        num_classes=num_classes
    ).to(device)

    # Оптимизатор и лосс
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedCrossEntropyLoss()

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    # Early stopping по dev
    best_dev_mean = float("-inf")
    best_dev_metrics = {}
    max_patience = 10
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

            audio  = batch["audio"].to(device)
            labels = batch["label"].to(device)
            texts  = batch["text"]

            audio_emb = audio_extractor.extract(audio)
            text_emb  = text_extractor.extract(texts)

            logits = model(audio_emb, text_emb)
            target = labels.argmax(dim=1)
            loss   = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        dev_loss, dev_uar, dev_war, dev_mf1, dev_wf1 = run_eval(
            model, dev_loader, audio_extractor, text_extractor, criterion, device
        )
        mean_dev = np.mean([dev_uar, dev_war, dev_mf1, dev_wf1])
        logging.info(
            f"[DEV]  Loss={dev_loss:.4f}, UAR={dev_uar:.4f}, WAR={dev_war:.4f}, "
            f"MF1={dev_mf1:.4f}, WF1={dev_wf1:.4f}, MEAN={mean_dev:.4f}"
        )

        scheduler.step(mean_dev)

        if mean_dev > best_dev_mean:
            best_dev_mean = mean_dev
            patience_counter = 0
            best_dev_metrics = {
                "loss": dev_loss,
                "uar":  dev_uar,
                "war":  dev_war,
                "mf1":  dev_mf1,
                "wf1":  dev_wf1,
                "mean": mean_dev
            }
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logging.info(f"Early stopping: {max_patience} эпох без улучшения.")
                break

        # --- TEST ---
        test_loss, test_uar, test_war, test_mf1, test_wf1 = run_eval(
            model, test_loader, audio_extractor, text_extractor, criterion, device
        )
        mean_test = np.mean([test_uar, test_war, test_mf1, test_wf1])
        logging.info(
            f"[TEST] Loss={test_loss:.4f}, UAR={test_uar:.4f}, WAR={test_war:.4f}, "
            f"MF1={test_mf1:.4f}, WF1={test_wf1:.4f}, MEAN={mean_test:.4f}"
        )

    logging.info("Тренировка завершена. Все split'ы обработаны!")
    return best_dev_mean, best_dev_metrics
