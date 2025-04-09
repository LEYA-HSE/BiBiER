# coding: utf-8
import logging
import torch
import random
import os
import shutil
import copy

os.environ["HF_HOME"] = "models"
import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.logger_setup import setup_logger
from utils.config_loader import ConfigLoader
from utils.losses import WeightedCrossEntropyLoss
from utils.measures import uar, war, mf1, wf1
from models.models import BiFormer, BiGraphFormer, MultiModalTransformer_v5, MultiModalTransformer_v4, MultiModalTransformer_v3
from data_loading.dataset_multimodal import DatasetMultiModal
from data_loading.feature_extractor import AudioEmbeddingExtractor, TextEmbeddingExtractor


def custom_collate_fn(batch):
    """Собирает список образцов в единый батч, отбрасывая None (невалидные)."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    audios = [b["audio"] for b in batch]   # list of (1, samples) tensors
    audio_tensor = torch.stack(audios)     # (B, 1, samples)

    labels = [b["label"] for b in batch]   # list of (num_emotions,) (one-hot)
    label_tensor = torch.stack(labels)     # (B, num_emotions)

    texts = [b["text"] for b in batch]     # list of str

    return {
        "audio": audio_tensor,  # (B, 1, samples)
        "label": label_tensor,  # (B, num_emotions)
        "text": texts           # list[str]
    }

def make_dataset_and_loader(config, split: str):
    csv_path = config.csv_path.format(base_dir=config.base_dir, split=split)
    wav_dir  = config.wav_dir.format(base_dir=config.base_dir,  split=split)
    print(csv_path, wav_dir, split)

    dataset = DatasetMultiModal(
        csv_path = csv_path,
        wav_dir  = wav_dir,
        emotion_columns=config.emotion_columns,
        split          = split,
        sample_rate    = config.sample_rate,
        wav_length     = config.wav_length,
        whisper_model  = config.whisper_model,
        text_column    = config.text_column,
        use_whisper_for_nontrain_if_no_text=config.use_whisper_for_nontrain_if_no_text,
        whisper_device = config.whisper_device,
        subset_size    = config.subset_size,
        merge_probability=config.merge_probability
    )

    shuffle = (split == "train")
    dataloader = DataLoader(
        dataset,
        batch_size  = config.batch_size,
        shuffle     = shuffle,
        num_workers = config.num_workers,
        collate_fn  = custom_collate_fn
    )
    return dataset, dataloader

def run_eval(model, loader, audio_extractor, text_extractor, criterion, device="cuda"):
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

            preds = logits.argmax(dim=1)
            total_preds.extend(preds.cpu().numpy().tolist())
            total_targets.extend(target.cpu().numpy().tolist())
            total += bs

    avg_loss = total_loss / total
    uar_m = uar(total_targets, total_preds)
    war_m = war(total_targets, total_preds)
    mf1_m = mf1(total_targets, total_preds)
    wf1_m = wf1(total_targets, total_preds)
    return avg_loss, uar_m, war_m, mf1_m, wf1_m

def train_once(config, train_loader, dev_loader, test_loader):
    """
    Логика обучения.
    """
    # Лог-файл
    os.makedirs("logs", exist_ok=True)
    datestr  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    # Экстракторы эмбеддингов
    audio_extractor = AudioEmbeddingExtractor(config)
    text_extractor  = TextEmbeddingExtractor(config)

    hidden_dim             = config.hidden_dim
    num_classes            = len(config.emotion_columns)
    num_transformer_heads  = config.num_transformer_heads
    num_graph_heads        = config.num_graph_heads
    mode                   = config.mode
    positional_encoding    = config.positional_encoding
    dropout                = config.dropout
    out_features           = config.out_features
    lr                     = config.lr
    num_epochs             = config.num_epochs
    tr_layer_number        = config.tr_layer_number

    dict_models = {
        'BiFormer': BiFormer,
        'BiGraphFormer': BiGraphFormer,
        'MultiModalTransformer_v5': MultiModalTransformer_v5,
        'MultiModalTransformer_v4': MultiModalTransformer_v4,
        'MultiModalTransformer_v3': MultiModalTransformer_v3
    }

    model_cls = dict_models[config.model_name]
    model = model_cls(
        audio_dim=config.audio_embedding_dim,
        text_dim=config.text_embedding_dim,
        hidden_dim=hidden_dim,
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

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedCrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=True
    )

    best_dev_meam = float("-inf")
    best_dev_metrics = {}
    max_patience = 5
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
        meam_m = np.mean([uar_m, war_m, mf1_m, wf1_m])

        logging.info(f"[TRAIN] Loss={train_loss:.4f}, UAR={uar_m:.4f}, WAR={war_m:.4f}, MF1={mf1_m:.4f}, WF1={wf1_m:.4f}, MEAN={meam_m:.4f}")

        # --- DEV ---
        dev_loss, dev_uar_m, dev_war_m, dev_mf1_m, dev_wf1_m = run_eval(
            model, dev_loader, audio_extractor, text_extractor, criterion, device
        )
        dev_meam_m = np.mean([dev_uar_m, dev_war_m, dev_mf1_m, dev_wf1_m])
        logging.info(f"[DEV]   Loss={dev_loss:.4f}, UAR={dev_uar_m:.4f}, WAR={dev_war_m:.4f}, MF1={dev_mf1_m:.4f}, WF1={dev_wf1_m:.4f}, MEAN={dev_meam_m:.4f}")

        scheduler.step(dev_meam_m)

        # Early stopping
        if dev_meam_m > best_dev_meam:
            best_dev_meam = dev_meam_m
            patience_counter = 0
            best_dev_metrics = {
                "loss": dev_loss,
                "uar": dev_uar_m,
                "war": dev_war_m,
                "mf1": dev_mf1_m,
                "wf1": dev_wf1_m,
                "mean": dev_meam_m
            }
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logging.info(f"Early stopping: {max_patience} эпох без улучшения.")
                break

        # --- TEST ---
        test_loss, test_uar_m, test_war_m, test_mf1_m, test_wf1_m = run_eval(
            model, test_loader, audio_extractor, text_extractor, criterion, device
        )
        test_meam_m = np.mean([test_uar_m, test_war_m, test_mf1_m, test_wf1_m])
        logging.info(f"[TEST]  Loss={test_loss:.4f}, UAR={test_uar_m:.4f}, WAR={test_war_m:.4f}, MF1={test_mf1_m:.4f}, WF1={test_wf1_m:.4f}, MEAN={test_meam_m:.4f}")

    logging.info("Тренировка завершена. Все split'ы обработаны!")
    return best_dev_meam, best_dev_metrics

def main():
    # 1) Загружаем базовый config из config.toml
    base_config = ConfigLoader("config.toml")

    # 2) Один раз делаем датасеты/лоадеры, если batch_size и пути не меняются
    _, train_loader = make_dataset_and_loader(base_config, "train")
    _, dev_loader   = make_dataset_and_loader(base_config, "dev")
    _, test_loader  = make_dataset_and_loader(base_config, "test")

    # 3) Создаём папку результатов (одну на всю серию экспериментов)
    results_dir = f"results_greedy_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))

    # 4) Файл, куда пишем все комбинации и результаты
    overrides_file = os.path.join(results_dir, "overrides.txt")
    with open(overrides_file, "w", encoding="utf-8") as f:
        f.write("=== Жадный (поэтапный) перебор гиперпараметров ===\n")

    # Изначально фиксируем некоторые значения (как в вашем примере):
    best_heads = 2
    best_layers = 1
    best_out_features = 128

    # ======================================================
    #  ШАГ 1: Перебираем hidden_dim
    # ======================================================
    hidden_dims = [128, 256, 512]
    best_hidden_dim = None
    best_dev_meam_for_hidden = float('-inf')

    for hd in hidden_dims:
        config = copy.deepcopy(base_config)
        config.hidden_dim            = hd
        config.num_transformer_heads = best_heads    # фиксируем
        config.tr_layer_number       = best_layers   # фиксируем
        config.out_features          = best_out_features  # фиксируем

        # Запуск обучения
        best_dev, best_dev_metrics = train_once(config, train_loader, dev_loader, test_loader)

        # ==== Запись результата в overrides.txt (динамика) ====
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Шаг1 (hidden_dim), проверяем hd={hd} ---\n")
            f.write(f"Лучший dev_meam={best_dev:.4f}\n")
            f.write("Лучшие метрики на dev:\n")
            f.write(f"  UAR={best_dev_metrics.get('uar',0):.4f}\n")
            f.write(f"  WAR={best_dev_metrics.get('war',0):.4f}\n")
            f.write(f"  MF1={best_dev_metrics.get('mf1',0):.4f}\n")
            f.write(f"  WF1={best_dev_metrics.get('wf1',0):.4f}\n")
            f.write(f"  Loss={best_dev_metrics.get('loss',0):.4f}\n")
            f.write(f"  MEAN={best_dev_metrics.get('mean',0):.4f}\n")

        # Смотрим, лучше ли это решение
        if best_dev > best_dev_meam_for_hidden:
            best_dev_meam_for_hidden = best_dev
            best_hidden_dim = hd

    # После перебора сохраняем итог
    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write(f"\n>> [Итог Шаг1]: Лучший hidden_dim={best_hidden_dim}, dev_meam={best_dev_meam_for_hidden:.4f}\n")

    # ======================================================
    #  ШАГ 2: Перебираем num_transformer_heads
    # ======================================================
    heads_candidates = [2, 4, 6]
    best_heads_val = None
    best_dev_meam_for_heads = float('-inf')

    for heads in heads_candidates:
        config = copy.deepcopy(base_config)
        config.hidden_dim            = best_hidden_dim  # используем лучший с шага1
        config.num_transformer_heads = heads
        config.tr_layer_number       = best_layers
        config.out_features          = best_out_features

        best_dev, best_dev_metrics = train_once(config, train_loader, dev_loader, test_loader)

        # ==== Запись результата (Шаг2) ====
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Шаг2 (heads), проверяем heads={heads} ---\n")
            f.write(f"Лучший dev_meam={best_dev:.4f}\n")
            f.write("Лучшие метрики на dev:\n")
            f.write(f"  UAR={best_dev_metrics.get('uar',0):.4f}\n")
            f.write(f"  WAR={best_dev_metrics.get('war',0):.4f}\n")
            f.write(f"  MF1={best_dev_metrics.get('mf1',0):.4f}\n")
            f.write(f"  WF1={best_dev_metrics.get('wf1',0):.4f}\n")
            f.write(f"  Loss={best_dev_metrics.get('loss',0):.4f}\n")
            f.write(f"  MEAN={best_dev_metrics.get('mean',0):.4f}\n")

        if best_dev > best_dev_meam_for_heads:
            best_dev_meam_for_heads = best_dev
            best_heads_val = heads

    best_heads = best_heads_val
    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write(f"\n>> [Итог Шаг2]: Лучший heads={best_heads}, dev_meam={best_dev_meam_for_heads:.4f}\n")

    # ======================================================
    #  ШАГ 3: Перебираем tr_layer_number
    # ======================================================
    layers_candidates = [1, 2, 3]
    best_layers_val = None
    best_dev_meam_for_layers = float('-inf')

    for ly in layers_candidates:
        config = copy.deepcopy(base_config)
        config.hidden_dim            = best_hidden_dim
        config.num_transformer_heads = best_heads
        config.tr_layer_number       = ly
        config.out_features          = best_out_features

        best_dev, best_dev_metrics = train_once(config, train_loader, dev_loader, test_loader)

        # Запись результата (Шаг3)
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Шаг3 (layers), проверяем layers={ly} ---\n")
            f.write(f"Лучший dev_meam={best_dev:.4f}\n")
            f.write("Лучшие метрики на dev:\n")
            f.write(f"  UAR={best_dev_metrics.get('uar',0):.4f}\n")
            f.write(f"  WAR={best_dev_metrics.get('war',0):.4f}\n")
            f.write(f"  MF1={best_dev_metrics.get('mf1',0):.4f}\n")
            f.write(f"  WF1={best_dev_metrics.get('wf1',0):.4f}\n")
            f.write(f"  Loss={best_dev_metrics.get('loss',0):.4f}\n")
            f.write(f"  MEAN={best_dev_metrics.get('mean',0):.4f}\n")

        if best_dev > best_dev_meam_for_layers:
            best_dev_meam_for_layers = best_dev
            best_layers_val = ly

    best_layers = best_layers_val
    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write(f"\n>> [Итог Шаг3]: Лучший layers={best_layers}, dev_meam={best_dev_meam_for_layers:.4f}\n")

    # ======================================================
    #  ШАГ 4: Перебираем out_features
    # ======================================================
    out_feat_candidates = [128, 256, 512]
    best_outf_val = None
    best_dev_meam_for_outf = float('-inf')

    for of_ in out_feat_candidates:
        config = copy.deepcopy(base_config)
        config.hidden_dim            = best_hidden_dim
        config.num_transformer_heads = best_heads
        config.tr_layer_number       = best_layers
        config.out_features          = of_

        best_dev, best_dev_metrics = train_once(config, train_loader, dev_loader, test_loader)

        # Запись результата (Шаг4)
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Шаг4 (out_features), проверяем out_features={of_} ---\n")
            f.write(f"Лучший dev_meam={best_dev:.4f}\n")
            f.write("Лучшие метрики на dev:\n")
            f.write(f"  UAR={best_dev_metrics.get('uar',0):.4f}\n")
            f.write(f"  WAR={best_dev_metrics.get('war',0):.4f}\n")
            f.write(f"  MF1={best_dev_metrics.get('mf1',0):.4f}\n")
            f.write(f"  WF1={best_dev_metrics.get('wf1',0):.4f}\n")
            f.write(f"  Loss={best_dev_metrics.get('loss',0):.4f}\n")
            f.write(f"  MEAN={best_dev_metrics.get('mean',0):.4f}\n")

        if best_dev > best_dev_meam_for_outf:
            best_dev_meam_for_outf = best_dev
            best_outf_val = of_

    best_out_features = best_outf_val
    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write(f"\n>> [Итог Шаг4]: Лучший out_features={best_out_features}, dev_meam={best_dev_meam_for_outf:.4f}\n")

    # Итог
    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Итоговая комбинация ===\n")
        f.write(f"hidden_dim={best_hidden_dim}, heads={best_heads}, layers={best_layers}, out_features={best_out_features}\n")

    print("Готово!")

if __name__ == "__main__":
    main()
