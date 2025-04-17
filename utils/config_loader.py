# utils/config_loader.py

import os
import toml
import logging

class ConfigLoader:
    """
    Класс для загрузки и обработки конфигурации из `config.toml`.
    """

    def __init__(self, config_path="config.toml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации `{config_path}` не найден!")

        self.config = toml.load(config_path)

        # ---------------------------
        # Общие параметры
        # ---------------------------
        self.split = self.config.get("split", "train")

        # ---------------------------
        # Пути к данным (многодатасетная поддержка)
        # ---------------------------
        self.datasets = self.config.get("datasets", {})

        # ---------------------------
        # Эмоции, модальности
        # ---------------------------
        self.modalities = self.config.get("modalities", ["audio"])
        self.emotion_columns = self.config.get("emotion_columns", ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"])

        # ---------------------------
        # DataLoader
        # ---------------------------
        dataloader_cfg = self.config.get("dataloader", {})
        self.batch_size = dataloader_cfg.get("batch_size", 1)
        self.num_workers = dataloader_cfg.get("num_workers", 0)
        self.shuffle = dataloader_cfg.get("shuffle", True)

        # ---------------------------
        # Аудио
        # ---------------------------
        audio_cfg = self.config.get("audio", {})
        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.wav_length = audio_cfg.get("wav_length", 2)

        # ---------------------------
        # Whisper / Текст
        # ---------------------------
        text_cfg = self.config.get("text", {})
        self.text_source = text_cfg.get("source", "csv")
        self.text_column = text_cfg.get("text_column", "text")
        self.whisper_model = text_cfg.get("whisper_model", "tiny")
        self.max_text_tokens = text_cfg.get("max_tokens", 15)
        self.whisper_device = text_cfg.get("whisper_device", "cuda")
        self.use_whisper_for_nontrain_if_no_text = text_cfg.get("use_whisper_for_nontrain_if_no_text", True)

        # ---------------------------
        # Параметры для тренировки
        # ---------------------------
        train_cfg = self.config.get("train", {})
        self.random_seed = train_cfg.get("random_seed", None)
        self.subset_size = train_cfg.get("subset_size", 0)
        self.hidden_dim = train_cfg.get("hidden_dim", 256)
        self.hidden_dim_gated = train_cfg.get("hidden_dim_gated", 256)
        self.num_transformer_heads = train_cfg.get("num_transformer_heads", 8)
        self.num_graph_heads = train_cfg.get("num_graph_heads", 8)
        self.mode = train_cfg.get("mode", 'mean')
        self.positional_encoding = train_cfg.get("positional_encoding", True)
        self.merge_probability = train_cfg.get("merge_probability", 0.1)
        self.dropout = train_cfg.get("dropout", 0)
        self.out_features = train_cfg.get("out_features", 128)
        self.lr = train_cfg.get("lr", 1e-4)
        self.num_epochs = train_cfg.get("num_epochs", 100)
        self.model_name=train_cfg.get("model_name", "BiFormer")
        self.tr_layer_number=train_cfg.get("tr_layer_number", 1)
        self.max_patience=train_cfg.get("max_patience", 10)
        self.save_prepared_data=train_cfg.get("save_prepared_data", True)
        self.save_feature_path=train_cfg.get("save_feature_path", 'features')
        self.search_type=train_cfg.get("search_type", None)

        # ---------------------------
        # Embeddings
        # ---------------------------
        emb_cfg = self.config.get("embeddings", {})
        self.audio_model_name = emb_cfg.get("audio_model", "amiriparian/ExHuBERT")
        self.text_model_name  = emb_cfg.get("text_model",  "jinaai/jina-embeddings-v3")
        self.audio_classifier_checkpoint = emb_cfg.get("audio_classifier_checkpoint", "best_audio_model.pt")
        self.text_classifier_checkpoint = emb_cfg.get("text_classifier_checkpoint", "best_text_model.pth")

        self.audio_embedding_dim = emb_cfg.get("audio_embedding_dim", 1024)
        self.text_embedding_dim  = emb_cfg.get("text_embedding_dim",  1024)
        self.emb_normalize = emb_cfg.get("emb_normalize", True)

        self.audio_pooling = emb_cfg.get("audio_pooling", None)
        self.text_pooling  = emb_cfg.get("text_pooling",  None)

        self.max_tokens = emb_cfg.get("max_tokens", 256)
        self.max_audio_frames = emb_cfg.get("max_audio_frames", 16000)

        self.emb_device = emb_cfg.get("device", "cuda")

        if __name__ == "__main__":
            self.log_config()

    def log_config(self):
        logging.info("=== CONFIGURATION ===")
        logging.info(f"Split: {self.split}")
        logging.info(f"Datasets loaded: {list(self.datasets.keys())}")
        for name, ds in self.datasets.items():
            logging.info(f"[Dataset: {name}]")
            logging.info(f"  Base Dir: {ds.get('base_dir', 'N/A')}")
            logging.info(f"  CSV Path: {ds.get('csv_path', '')}")
            logging.info(f"  WAV Dir: {ds.get('wav_dir', '')}")
        logging.info(f"Emotion columns: {self.emotion_columns}")

        # Логируем обучающие параметры
        logging.info("--- Training Config ---")
        logging.info(f"Sample Rate={self.sample_rate}, Wav Length={self.wav_length}s")
        logging.info(f"Whisper Model={self.whisper_model}, Device={self.whisper_device}, MaxTokens={self.max_text_tokens}")
        logging.info(f"use_whisper_for_nontrain_if_no_text={self.use_whisper_for_nontrain_if_no_text}")
        logging.info(f"DataLoader: batch_size={self.batch_size}, num_workers={self.num_workers}, shuffle={self.shuffle}")
        logging.info(f"Model Name: {self.model_name}")
        logging.info(f"Random Seed: {self.random_seed}")
        logging.info(f"Hidden Dim: {self.hidden_dim}")
        logging.info(f"Hidden Dim in Gated: {self.hidden_dim_gated}")
        logging.info(f"Num Heads in Transformer: {self.num_transformer_heads}")
        logging.info(f"Num Heads in Graph: {self.num_graph_heads}")
        logging.info(f"Mode stat pooling: {self.mode}")
        logging.info(f"Positional Encoding: {self.positional_encoding}")
        logging.info(f"Number of transformer layers: {self.tr_layer_number}")
        logging.info(f"Dropout: {self.dropout}")
        logging.info(f"Out Features: {self.out_features}")
        logging.info(f"LR: {self.lr}")
        logging.info(f"Num Epochs: {self.num_epochs}")
        logging.info(f"Merge Probability={self.merge_probability}")
        logging.info(f"Max Patience={self.max_patience}")
        logging.info(f"Save Prepared Data={self.save_prepared_data}")
        logging.info(f"Path to Save Features={self.save_feature_path}")
        logging.info(f"Search Type={self.search_type}")

        # Логируем embeddings
        logging.info("--- Embeddings Config ---")
        logging.info(f"Audio Model: {self.audio_model_name}, Text Model: {self.text_model_name}")
        logging.info(f"Audio dim={self.audio_embedding_dim}, Text dim={self.text_embedding_dim}")
        logging.info(f"Audio pooling={self.audio_pooling}, Text pooling={self.text_pooling}")
        logging.info(f"Max tokens={self.max_tokens}, Max audio frames={self.max_audio_frames}")
        logging.info(f"Emb device={self.emb_device}, Normalize={self.emb_normalize}")

    def show_config(self):
        self.log_config()
