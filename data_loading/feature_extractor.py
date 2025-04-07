# data_loading/feature_extractor.py

import torch
import logging
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    AutoModelForAudioClassification
)

class AudioEmbeddingExtractor:
    """
    Извлекает эмбеддинги из аудио, используя модель (например 'amiriparian/ExHuBERT'),
    с учётом pooling, нормализации и т.д.
    """

    def __init__(self, config):
        """
        Ожидается, что в config есть поля:
         - audio_model_name (str)         : название модели (ExHuBERT и т.п.)
         - emb_device (str)              : "cpu" или "cuda"
         - audio_pooling (str | None)    : "mean", "cls", "max", "min", "last" или None (пропустить пуллинг)
         - emb_normalize (bool)          : делать ли L2-нормализацию выхода
         - max_audio_frames (int)        : ограничение длины по временной оси (если 0 - не ограничивать)
        """
        self.config = config
        self.device = config.emb_device
        self.model_name = config.audio_model_name
        self.pooling = config.audio_pooling       # может быть None
        self.normalize_output = config.emb_normalize
        self.max_audio_frames = getattr(config, "max_audio_frames", 0)

        # Попробуем загрузить feature_extractor (не у всех моделей доступен)
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            logging.info(f"[Audio] Using AutoFeatureExtractor for '{self.model_name}'")
        except Exception as e:
            self.feature_extractor = None
            logging.warning(f"[Audio] No built-in FeatureExtractor found. Model={self.model_name}. Error: {e}")

        # Загружаем модель
        # Если у модели нет head-классификации, бывает достаточно AutoModel
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                output_hidden_states=True   # чтобы точно был last_hidden_state
            ).to(self.device)
            logging.info(f"[Audio] Loaded AutoModel with output_hidden_states=True: {self.model_name}")
        except Exception as e:
            logging.warning(f"[Audio] Fallback to AudioClassification model. Reason: {e}")
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                output_hidden_states=True
            ).to(self.device)

    def extract(self, waveform_batch: torch.Tensor, sample_rate=16000):
        """
        Извлекает эмбеддинги из аудиоданных.

        :param waveform_batch: Тензор формы (B, T) или (B, 1, T).
        :param sample_rate: Частота дискретизации (int).
        :return: Тензор:
          - если pooling != None, будет (B, hidden_dim)
          - если pooling == None и last_hidden_state имел форму (B, seq_len, hidden_dim),
            вернётся (B, seq_len, hidden_dim).
        """

        # Если пришло (B, 1, T), уберём ось "1"
        if waveform_batch.dim() == 3 and waveform_batch.shape[1] == 1:
            waveform_batch = waveform_batch.squeeze(1)  # -> (B, T)

        # Усечение по времени, если нужно
        if self.max_audio_frames > 0 and waveform_batch.shape[1] > self.max_audio_frames:
            waveform_batch = waveform_batch[:, :self.max_audio_frames]

        # Если есть feature_extractor - используем
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                waveform_batch,
                sampling_rate=sample_rate,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_audio_frames if self.max_audio_frames > 0 else None
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(input_values=inputs["input_values"])
        else:
            # Иначе подадим напрямую "input_values" на модель
            inputs = {"input_values": waveform_batch.to(self.device)}
            outputs = self.model(**inputs)

        # Теперь outputs может быть BaseModelOutput (с last_hidden_state, hidden_states, etc.)
        # Или SequenceClassifierOutput (с logits), если это модель-классификатор
        if hasattr(outputs, "last_hidden_state"):
            # (B, seq_len, hidden_dim)
            hidden = outputs.last_hidden_state
            # logging.debug(f"[Audio] last_hidden_state shape: {hidden.shape}")
        elif hasattr(outputs, "logits"):
            # logits: (B, num_labels)
            # Для пуллинга по "seq_len" притворимся, что seq_len=1
            hidden = outputs.logits.unsqueeze(1)  # (B,1,num_labels)
            logging.debug(f"[Audio] Found logits shape: {outputs.logits.shape} => hidden={hidden.shape}")
        else:
            # Модель может сразу возвращать тензор
            hidden = outputs

        # Если у нас 2D-тензор (B, hidden_dim), значит всё уже спулено
        if hidden.dim() == 2:
            emb = hidden
        elif hidden.dim() == 3:
            # (B, seq_len, hidden_dim)
            if self.pooling is None:
                # Возвращаем как есть
                emb = hidden
            else:
                # Выполним пуллинг
                if self.pooling == "mean":
                    emb = hidden.mean(dim=1)
                elif self.pooling == "cls":
                    emb = hidden[:, 0, :]  # [B, hidden_dim]
                elif self.pooling == "max":
                    emb, _ = hidden.max(dim=1)
                elif self.pooling == "min":
                    emb, _ = hidden.min(dim=1)
                elif self.pooling == "last":
                    emb = hidden[:, -1, :]
                else:
                    emb = hidden.mean(dim=1)  # на всякий случай fallback
        else:
            # На всякий: если ещё какая-то форма
            raise ValueError(f"[Audio] Unexpected hidden shape={hidden.shape}, pooling={self.pooling}")

        if self.normalize_output and emb.dim() == 2:
            emb = F.normalize(emb, p=2, dim=1)

        return emb


class TextEmbeddingExtractor:
    """
    Извлекает эмбеддинги из текста (например 'jinaai/jina-embeddings-v3'),
    с учётом pooling (None, mean, cls, и т.д.), нормализации и усечения.
    """

    def __init__(self, config):
        """
        Параметры в config:
         - text_model_name (str)
         - emb_device (str)
         - text_pooling (str | None)
         - emb_normalize (bool)
         - max_tokens (int)
        """
        self.config = config
        self.device = config.emb_device
        self.model_name = config.text_model_name
        self.pooling = config.text_pooling        # может быть None
        self.normalize_output = config.emb_normalize
        self.max_tokens = config.max_tokens

        # trust_remote_code=True нужно для моделей вроде jina
        logging.info(f"[Text] Loading tokenizer for {self.model_name} with trust_remote_code=True")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        logging.info(f"[Text] Loading model for {self.model_name} with trust_remote_code=True")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            output_hidden_states=True,    # хотим иметь last_hidden_state
            force_download=False
        ).to(self.device)

    def extract(self, text_list):
        """
        :param text_list: список строк (или одна строка)
        :return: тензор (B, hidden_dim) или (B, seq_len, hidden_dim), если pooling=None
        """

        if isinstance(text_list, str):
            text_list = [text_list]

        inputs = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Обычно у AutoModel last_hidden_state.shape = (B, seq_len, hidden_dim)
            hidden = outputs.last_hidden_state
            # logging.debug(f"[Text] last_hidden_state shape: {hidden.shape}")

            # Если pooling=None => вернём (B, seq_len, hidden_dim)
            if hidden.dim() == 3:
                if self.pooling is None:
                    emb = hidden
                else:
                    if self.pooling == "mean":
                        emb = hidden.mean(dim=1)
                    elif self.pooling == "cls":
                        emb = hidden[:, 0, :]
                    elif self.pooling == "max":
                        emb, _ = hidden.max(dim=1)
                    elif self.pooling == "min":
                        emb, _ = hidden.min(dim=1)
                    elif self.pooling == "last":
                        emb = hidden[:, -1, :]
                    elif self.pooling == "sum":
                        emb = hidden.sum(dim=1)
                    else:
                        emb = hidden.mean(dim=1)
            else:
                # На всякий случай, если получилось (B, hidden_dim)
                emb = hidden

        if self.normalize_output and emb.dim() == 2:
            emb = F.normalize(emb, p=2, dim=1)

        return emb
