import logging
import torch
import random
import os
import shutil
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
        "text": texts           # list[str] длиной B
    }


def make_dataset_and_loader(config, split: str):
    """
    Создаёт (dataset, dataloader) для указанного сплита: train/dev/test.

    Читает config.csv_path / config.wav_dir как шаблоны с {split}.
    Возвращает (dataset, dataloader).
    """
    csv_path = config.csv_path.format(base_dir=config.base_dir, split=split)
    wav_dir  = config.wav_dir.format(base_dir=config.base_dir,  split=split)

    print(csv_path, wav_dir, split)

    dataset = DatasetMultiModal(
        csv_path = csv_path,
        wav_dir  = wav_dir,
        emotion_columns=config.emotion_columns,
        split          =split,
        sample_rate    =config.sample_rate,
        wav_length     =config.wav_length,
        whisper_model  =config.whisper_model,
        text_column    =config.text_column,
        use_whisper_for_nontrain_if_no_text=config.use_whisper_for_nontrain_if_no_text,
        whisper_device =config.whisper_device,
        subset_size    =config.subset_size,
        merge_probability=config.merge_probability
    )

    # Для train обычно shuffle=True, dev/test — обычно False
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
    """
    Оцениваем (валидация/тест) на заданном loader'е, возвращаем (loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    total_preds = []
    total_targets = []

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue

            audio  = batch["audio"].to(device)
            labels = batch["label"].to(device)
            texts  = batch["text"]

            # Извлекаем эмбеддинги
            audio_emb = audio_extractor.extract(audio)
            text_emb  = text_extractor.extract(texts)

            # Прогоняем через модель
            logits = model(audio_emb, text_emb)

            # Считаем лосс
            target = labels.argmax(dim=1)  # (B,)
            loss = criterion(logits, target)

            # Накопим
            bs = audio.shape[0]
            total_loss += loss.item() * bs

            # accuracy
            preds = logits.argmax(dim=1)
            # correct += (preds == target).sum().item()
            total_preds.extend(preds.cpu().numpy().tolist())
            total_targets.extend(target.cpu().numpy().tolist())
            total   += bs

    avg_loss = total_loss / total

    uar_m = uar(total_targets, total_preds)
    war_m = war(total_targets, total_preds)
    mf1_m = mf1(total_targets, total_preds)
    wf1_m = wf1(total_targets, total_preds)
    # accuracy = correct / total
    return avg_loss, uar_m, war_m, mf1_m, wf1_m


def main():
    # Лог-файл
    os.makedirs("logs", exist_ok=True)
    datestr  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join("logs", f"train_log_{datestr}.txt")
    setup_logger(logging.INFO, log_file=log_file)

    logging.info("🚀 === Запуск тренировки (train/dev/test) ===")

    # Загружаем конфиг
    config = ConfigLoader("config.toml")
    config.show_config()

    # Фиксируем seed
    if config.random_seed > 0:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        logging.info(f"🔒 Фиксируем random seed: {config.random_seed}")
    else:
        logging.info("🔓 Random seed не фиксирован (0).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Если в config есть отдельное поле emb_device, то можно device = config.emb_device

    # Создаём датасеты/лоадеры для train/dev/test
    _, train_loader = make_dataset_and_loader(config, "train")
    _, dev_loader   = make_dataset_and_loader(config, "dev")
    _, test_loader  = make_dataset_and_loader(config, "test")

    # Экстракторы эмбеддингов
    audio_extractor = AudioEmbeddingExtractor(config)
    text_extractor  = TextEmbeddingExtractor(config)

    # Предположим, что экстракторы выдают размерность = 1024
    hidden_dim=config.hidden_dim
    num_classes=len(config.emotion_columns)
    num_transformer_heads=config.num_transformer_heads
    num_graph_heads=config.num_graph_heads
    mode=config.mode
    positional_encoding=config.positional_encoding
    dropout=config.dropout
    out_features=config.out_features
    lr=config.lr
    num_epochs=config.num_epochs
    tr_layer_number=config.tr_layer_number

    dict_models = {
                        'BiFormer': BiFormer,
                        'BiGraphFormer': BiGraphFormer,
                        'MultiModalTransformer_v5': MultiModalTransformer_v5,
                        'MultiModalTransformer_v4': MultiModalTransformer_v4,
                        'MultiModalTransformer_v3': MultiModalTransformer_v3
                    }

    model_cls = dict_models[config.model_name]


    model = model_cls(audio_dim=config.audio_embedding_dim,
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
                                num_classes=num_classes).to(device)

    # Оптимизатор, лосс
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedCrossEntropyLoss()

    for epoch in range(num_epochs):
        logging.info(f"\n=== Эпоха {epoch} ===")

        # --- TRAIN ---
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

            # Извлекаем эмбеддинги
            audio_emb = audio_extractor.extract(audio)
            text_emb  = text_extractor.extract(texts)

            # Прогоняем через модель
            logits = model(audio_emb, text_emb)

            # Лосс
            target = labels.argmax(dim=1)  # (B,)
            loss   = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = audio.shape[0]
            total_loss += loss.item() * bs

            # accuracy
            preds = logits.argmax(dim=1)
            total_preds.extend(preds.cpu().numpy().tolist())
            total_targets.extend(target.cpu().numpy().tolist())
            total_samples += bs

        train_loss = total_loss / total_samples
        uar_m = uar(total_targets, total_preds)
        war_m = war(total_targets, total_preds)
        mf1_m = mf1(total_targets, total_preds)
        wf1_m = wf1(total_targets, total_preds)
        meam_m = np.mean([uar_m,war_m, mf1_m, wf1_m])

        logging.info(f"[TRAIN] Loss={train_loss:.4f}, UAR={uar_m:.4f}, WAR={war_m:.4f}, MF1={mf1_m:.4f}, WF1={wf1_m:.4f}, MEAN={meam_m:.4f}")

        # --- DEV ---
        dev_loss, dev_uar_m, dev_war_m, dev_mf1_m, dev_wf1_m  = run_eval(model, dev_loader, audio_extractor, text_extractor, criterion, device)
        dev_meam_m = np.mean([dev_uar_m,dev_war_m, dev_mf1_m, dev_wf1_m])
        logging.info(f"[DEV]   Loss={dev_loss:.4f}, UAR={dev_uar_m:.4f}, WAR={dev_war_m:.4f}, MF1={dev_mf1_m:.4f}, WF1={dev_wf1_m:.4f}, MEAN={dev_meam_m:.4f}")

        # После окончания эпох — прогоняем test (один раз)
        test_loss, test_uar_m, test_war_m, test_mf1_m, test_wf1_m = run_eval(model, test_loader, audio_extractor, text_extractor, criterion, device)
        test_meam_m = np.mean([test_uar_m,test_war_m, test_mf1_m, test_wf1_m])
        logging.info(f"[TEST]  Loss={test_loss:.4f}, UAR={test_uar_m:.4f}, WAR={test_war_m:.4f}, MF1={test_mf1_m:.4f}, WF1={test_wf1_m:.4f}, MEAN={test_meam_m:.4f}")

    logging.info("✅ Тренировка завершена. Все split'ы обработаны!")

    results_dir = f"results_{datestr}"
    os.makedirs(results_dir, exist_ok=True)

    shutil.copy(
        "config.toml",
        os.path.join(results_dir, f"config_{datestr}.toml")
    )

if __name__ == "__main__":
    main()
