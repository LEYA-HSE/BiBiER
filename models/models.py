# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .help_layers import TransformerEncoderLayer, GAL,GraphFusionLayer, GraphFusionLayerAtt, MambaBlock, RMSNorm

class PredictionsFusion(nn.Module):
    def __init__(self, num_matrices=2, num_classes=7):
        super(PredictionsFusion, self).__init__()
        self.weights = nn.Parameter(torch.rand(num_matrices, num_classes))

    def forward(self, pred):
        normalized_weights = torch.softmax(self.weights, dim=0)
        weighted_matrix = sum(mat * normalized_weights[i] for i, mat in enumerate(pred))
        return weighted_matrix

class MultiModalTransformer_v3(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, hidden_dim=512, hidden_dim_gated=512, num_transformer_heads=2, num_graph_heads=2, seg_len=44, positional_encoding=True, dropout=0, mode='mean', device="cuda",  tr_layer_number=1, out_features=128, num_classes=7):
        super(MultiModalTransformer_v3, self).__init__()

        self.mode = mode

        self.hidden_dim = hidden_dim

        # Проекционные слои
        # self.audio_proj = nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity()

        # self.audio_proj = nn.Sequential(
        #     nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity(),
            # nn.LayerNorm(hidden_dim),
            # nn.Dropout(dropout)
        # )

        self.audio_proj = nn.Sequential(
            nn.Conv1d(audio_dim, hidden_dim, 1),
            nn.GELU(),
        )

        self.text_proj = nn.Sequential(
            nn.Conv1d(text_dim, hidden_dim, 1),
            nn.GELU(),
        )
        # self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()

        # self.text_proj = nn.Sequential(
        #     nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity(),
            # nn.LayerNorm(hidden_dim),
            # nn.Dropout(dropout)
        # )

        # Механизмы внимания
        self.audio_to_text_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_transformer_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])
        self.text_to_audio_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_transformer_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])

        # Классификатор
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim*2, out_features) if self.mode == 'mean' else nn.Linear(hidden_dim*4, out_features),
        #     nn.ReLU(),
        #     nn.Linear(out_features, num_classes)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features) if self.mode == 'mean' else nn.Linear(hidden_dim*4, out_features),
            # nn.LayerNorm(out_features),
            # nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        # self._init_weights()

    def forward(self, audio_features, text_features):
        # Преобразование размерностей
        audio_features = audio_features.float()
        text_features = text_features.float()

        # audio_features = self.audio_proj(audio_features)
        # text_features = self.text_proj(text_features)
        audio_features = self.audio_proj(audio_features.permute(0,2,1)).permute(0,2,1)
        text_features = self.text_proj(text_features.permute(0,2,1)).permute(0,2,1)

        # Адаптивная пуллинг до минимальной длины
        min_seq_len = min(audio_features.size(1), text_features.size(1))
        audio_features = F.adaptive_avg_pool1d(audio_features.permute(0,2,1), min_seq_len).permute(0,2,1)
        text_features = F.adaptive_avg_pool1d(text_features.permute(0,2,1), min_seq_len).permute(0,2,1)

        # Трансформерные блоки
        for i in range(len(self.audio_to_text_attn)):
            attn_audio = self.audio_to_text_attn[i](text_features, audio_features, audio_features)
            attn_text = self.text_to_audio_attn[i](audio_features, text_features, text_features)
            audio_features += attn_audio
            text_features += attn_text

        # Статистики
        std_audio, mean_audio = torch.std_mean(attn_audio, dim=1)
        std_text, mean_text = torch.std_mean(attn_text, dim=1)

        # Классификация
        if self.mode == 'mean':
            return self.classifier(torch.cat([mean_audio, mean_audio], dim=1))
        else:
            return self.classifier(torch.cat([mean_audio, std_audio, mean_text, std_text], dim=1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MultiModalTransformer_v4(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, hidden_dim=512, hidden_dim_gated=512, num_transformer_heads=2, num_graph_heads=2, seg_len=44, positional_encoding=True, dropout=0, mode='mean', device="cuda",  tr_layer_number=1, out_features=128, num_classes=7):
        super(MultiModalTransformer_v4, self).__init__()

        self.mode = mode

        self.hidden_dim = hidden_dim

        # Проекционные слои
        self.audio_proj = nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()

        # Механизмы внимания
        self.audio_to_text_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_transformer_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])
        self.text_to_audio_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_transformer_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])

        # Графовое слияние вместо GAL
        if self.mode == 'mean':
            self.graph_fusion = GraphFusionLayer(hidden_dim, heads=num_graph_heads)
        else:
            self.graph_fusion = GraphFusionLayer(hidden_dim*2, heads=num_graph_heads)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, out_features) if self.mode == 'mean' else nn.Linear(hidden_dim*2, out_features),
            nn.ReLU(),
            nn.Linear(out_features, num_classes)
        )

    def forward(self, audio_features, text_features):
        # Преобразование размерностей
        audio_features = audio_features.float()
        text_features = text_features.float()

        audio_features = self.audio_proj(audio_features)
        text_features = self.text_proj(text_features)

        # Адаптивная пуллинг до минимальной длины
        min_seq_len = min(audio_features.size(1), text_features.size(1))
        audio_features = F.adaptive_avg_pool1d(audio_features.permute(0,2,1), min_seq_len).permute(0,2,1)
        text_features = F.adaptive_avg_pool1d(text_features.permute(0,2,1), min_seq_len).permute(0,2,1)

        # Трансформерные блоки
        for i in range(len(self.audio_to_text_attn)):
            attn_audio = self.audio_to_text_attn[i](text_features, audio_features, audio_features)
            attn_text = self.text_to_audio_attn[i](audio_features, text_features, text_features)
            audio_features += attn_audio
            text_features += attn_text

        # Статистики
        std_audio, mean_audio = torch.std_mean(attn_audio, dim=1)
        std_text, mean_text = torch.std_mean(attn_text, dim=1)

        # Графовое слияние статистик
        if self.mode == 'mean':
            h_ta = self.graph_fusion(mean_audio, mean_text)
        else:
            h_ta = self.graph_fusion(torch.cat([mean_audio, std_audio], dim=1), torch.cat([mean_text, std_text], dim=1))

        # Классификация
        return self.classifier(h_ta)

class MultiModalTransformer_v5(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, hidden_dim=512, hidden_dim_gated=512, num_transformer_heads=2, num_graph_heads=2, seg_len=44, tr_layer_number=1, positional_encoding=True, dropout=0, mode='mean', device="cuda",  out_features=128, num_classes=7):
        super(MultiModalTransformer_v5, self).__init__()

        self.hidden_dim = hidden_dim
        self.mode = mode

        # Приведение к общей размерности (адаптивные проекции)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()

        # Механизмы внимания

        self.audio_to_text_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_transformer_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])
        self.text_to_audio_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_transformer_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])

        # Гейтед аттеншн
        if self.mode == 'mean':
            self.gal = GAL(hidden_dim, hidden_dim, hidden_dim_gated)
        else:
            self.gal = GAL(hidden_dim*2, hidden_dim*2, hidden_dim_gated)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.ReLU(),
            nn.Linear(out_features, num_classes)
        )

    def forward(self, audio_features, text_features):
        bs, seq_audio, audio_feat_dim = audio_features.shape
        bs, seq_text, text_feat_dim = text_features.shape

        text_features = text_features.to(torch.float32)
        audio_features = audio_features.to(torch.float32)

        # Приведение размерности
        audio_features = self.audio_proj(audio_features)  # (bs, seq_audio, hidden_dim)
        text_features = self.text_proj(text_features)    # (bs, seq_text, hidden_dim)

        # Определяем минимальную длину последовательности
        min_seq_len = min(seq_audio, seq_text)

        # Усреднение до минимальной длины
        audio_features = F.adaptive_avg_pool2d(audio_features.permute(0, 2, 1), (self.hidden_dim, min_seq_len)).permute(0, 2, 1)
        text_features = F.adaptive_avg_pool2d(text_features.permute(0, 2, 1), (self.hidden_dim, min_seq_len)).permute(0, 2, 1)

        # Трансформерные блоки
        for i in range(len(self.audio_to_text_attn)):
            attn_audio = self.audio_to_text_attn[i](text_features, audio_features, audio_features)
            attn_text = self.text_to_audio_attn[i](audio_features, text_features, text_features)
            audio_features += attn_audio
            text_features += attn_text

        # Статистики
        std_audio, mean_audio = torch.std_mean(attn_audio, dim=1)
        std_text, mean_text = torch.std_mean(attn_text, dim=1)

        # # Гейтед аттеншн
        # h_audio = torch.tanh(self.Wa(torch.cat([min_audio, std_audio], dim=1)))
        # h_text = torch.tanh(self.Wt(torch.cat([min_text, std_text], dim=1)))
        # z_ta = torch.sigmoid(self.W_at(torch.cat([min_audio, std_audio, min_text, std_text], dim=1)))
        # h_ta = z_ta * h_text + (1 - z_ta) * h_audio
        if self.mode == 'mean':
            h_ta = self.gal(mean_audio, mean_text)
        else:
            h_ta = self.gal(torch.cat([mean_audio, std_audio], dim=1), torch.cat([mean_text, std_text], dim=1))

        # Классификация
        output = self.classifier(h_ta)
        return output

class MultiModalTransformer_v7(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, hidden_dim=512, num_heads=2, positional_encoding=True, dropout=0, mode='mean', device="cuda",  tr_layer_number=1, out_features=128, num_classes=7):
        super(MultiModalTransformer_v7, self).__init__()

        self.mode = mode

        self.hidden_dim = hidden_dim

        # Проекционные слои
        self.audio_proj = nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()

        # Механизмы внимания
        self.audio_to_text_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])
        self.text_to_audio_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])

        # Графовое слияние вместо GAL
        if self.mode == 'mean':
            self.graph_fusion = GraphFusionLayerAtt(hidden_dim, heads=num_heads)
        else:
            self.graph_fusion = GraphFusionLayerAtt(hidden_dim*2, heads=num_heads)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, out_features) if self.mode == 'mean' else nn.Linear(hidden_dim*2, out_features),
            nn.ReLU(),
            nn.Linear(out_features, num_classes)
        )

    def forward(self, audio_features, text_features):
        # Преобразование размерностей
        audio_features = audio_features.float()
        text_features = text_features.float()

        audio_features = self.audio_proj(audio_features)
        text_features = self.text_proj(text_features)

        # Адаптивная пуллинг до минимальной длины
        min_seq_len = min(audio_features.size(1), text_features.size(1))
        audio_features = F.adaptive_avg_pool1d(audio_features.permute(0,2,1), min_seq_len).permute(0,2,1)
        text_features = F.adaptive_avg_pool1d(text_features.permute(0,2,1), min_seq_len).permute(0,2,1)

        # Трансформерные блоки
        for i in range(len(self.audio_to_text_attn)):
            attn_audio = self.audio_to_text_attn[i](text_features, audio_features, audio_features)
            attn_text = self.text_to_audio_attn[i](audio_features, text_features, text_features)
            audio_features += attn_audio
            text_features += attn_text

        # Статистики
        std_audio, mean_audio = torch.std_mean(attn_audio, dim=1)
        std_text, mean_text = torch.std_mean(attn_text, dim=1)

        # Графовое слияние статистик
        if self.mode == 'mean':
            h_ta = self.graph_fusion(mean_audio, mean_text)
        else:
            h_ta = self.graph_fusion(torch.cat([mean_audio, std_audio], dim=1), torch.cat([mean_audio, std_text], dim=1))

        # Классификация
        return self.classifier(h_ta)

class BiFormer(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, seg_len=44, hidden_dim=512, hidden_dim_gated=128,
                num_transformer_heads=2, num_graph_heads=2, positional_encoding=True, dropout=0.1, mode='mean',
                device="cuda", tr_layer_number=1, out_features=128, num_classes=7):
        super(BiFormer, self).__init__()

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.seg_len = seg_len
        self.tr_layer_number = tr_layer_number

        # Проекционные слои с нормализацией
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        # self.audio_proj = nn.Sequential(
        #     nn.Conv1d(audio_dim, hidden_dim, 1),
        #     nn.GELU(),
        # )

        # self.text_proj = nn.Sequential(
        #     nn.Conv1d(text_dim, hidden_dim, 1),
        #     nn.GELU(),
        # )

        # Трансформерные слои (сохраняем вашу реализацию)
        self.audio_to_text_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.text_to_audio_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        # Автоматический расчёт размерности для классификатора
        self._calculate_classifier_input_dim()

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        self._init_weights()

    def _calculate_classifier_input_dim(self):
        """Вычисляет размер входных признаков для классификатора"""
        # Тестовый проход через пулинг с dummy-данными
        dummy_audio = torch.randn(1, self.seg_len, self.hidden_dim)
        dummy_text = torch.randn(1, self.seg_len, self.hidden_dim)

        audio_pool = self._pool_features(dummy_audio)
        text_pool = self._pool_features(dummy_text)

        combined = torch.cat([audio_pool, text_pool], dim=1)
        self.classifier_input_dim = combined.size(1)

    def _pool_features(self, x):
        # Статистики по временной оси (seq_len)
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]

        # Статистики по feature оси  (hidden_dim)
        mean_feat = x.mean(dim=-1)  # [batch, seq_len]

        return torch.cat([mean_temp, mean_feat], dim=1)

    def forward(self, audio_features, text_features):
        # Проекция признаков
        # audio = self.audio_proj(audio_features.permute(0,2,1)).permute(0,2,1)
        # text = self.text_proj(text_features.permute(0,2,1)).permute(0,2,1)
        audio = self.audio_proj(audio_features.float())
        text = self.text_proj(text_features.float())

        # Адаптивный пулинг
        min_len = min(audio.size(1), text.size(1))
        audio = self.adaptive_temporal_pool(audio, min_len)
        text = self.adaptive_temporal_pool(text, min_len)

        # Кросс-модальное взаимодействие
        for i in range(self.tr_layer_number):
            attn_audio = self.audio_to_text_attn[i](text, audio, audio)
            attn_text = self.text_to_audio_attn[i](audio, text, text)
            audio = audio + attn_audio
            text = text + attn_text

        # Агрегация признаков
        audio_pool = self._pool_features(audio)
        text_pool = self._pool_features(text)

        # Классификация
        features = torch.cat([audio_pool, text_pool], dim=1)
        return self.classifier(features)

    def adaptive_temporal_pool(self, x, target_len):
        """Адаптивное изменение временной длины"""
        if x.size(1) == target_len:
            return x

        return F.interpolate(
            x.permute(0, 2, 1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BiGraphFormer(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, seg_len=44, hidden_dim=512, hidden_dim_gated=128,
                num_transformer_heads=2, num_graph_heads = 2, positional_encoding=True, dropout=0.1, mode='mean',
                device="cuda", tr_layer_number=1, out_features=128, num_classes=7):
        super(BiGraphFormer, self).__init__()

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.seg_len = seg_len
        self.tr_layer_number = tr_layer_number

        # Проекционные слои с нормализацией
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Трансформерные слои (сохраняем вашу реализацию)
        self.audio_to_text_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.text_to_audio_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.graph_fusion_feat = GraphFusionLayer(self.seg_len, heads=num_graph_heads)
        self.graph_fusion_temp = GraphFusionLayer(hidden_dim, heads=num_graph_heads)

        # Автоматический расчёт размерности для классификатора
        self._calculate_classifier_input_dim()

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        # Финальная проекция графов
        self.fc_feat = nn.Sequential(
            nn.Linear(self.seg_len, self.seg_len),
            nn.LayerNorm(self.seg_len),
            nn.Dropout(dropout)
        )

        self.fc_temp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _calculate_classifier_input_dim(self):
        """Вычисляет размер входных признаков для классификатора"""
        # Тестовый проход через пулинг с dummy-данными
        dummy_audio = torch.randn(1, self.seg_len, self.hidden_dim)
        dummy_text = torch.randn(1, self.seg_len, self.hidden_dim)

        audio_pool_temp, audio_pool_feat = self._pool_features(dummy_audio)
        # text_pool_temp, _ = self._pool_features(dummy_text)

        combined = torch.cat([audio_pool_temp, audio_pool_feat], dim=1)
        self.classifier_input_dim = combined.size(1)

    def _pool_features(self, x):
        # Статистики по временной оси (seq_len)
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]

        # Статистики по feature оси  (hidden_dim)
        mean_feat = x.mean(dim=-1)  # [batch, seq_len]

        return mean_temp,  mean_feat

    def forward(self, audio_features, text_features):
        # Проекция признаков
        audio = self.audio_proj(audio_features.float())
        text = self.text_proj(text_features.float())

        # Адаптивный пулинг
        min_len = min(audio.size(1), text.size(1))
        audio = self.adaptive_temporal_pool(audio, min_len)
        text = self.adaptive_temporal_pool(text, min_len)

        # Кросс-модальное взаимодействие
        for i in range(self.tr_layer_number):
            attn_audio = self.audio_to_text_attn[i](text, audio, audio)
            attn_text = self.text_to_audio_attn[i](audio, text, text)

            audio = audio + attn_audio
            text = text + attn_text

        # Агрегация признаков
        audio_pool_temp, audio_pool_feat = self._pool_features(audio)
        text_pool_temp, text_pool_feat = self._pool_features(text)

        # print(audio_pool_temp.shape, audio_pool_feat.shape, text_pool_temp.shape, text_pool_feat.shape)

        graph_feat = self.graph_fusion_feat(audio_pool_feat, text_pool_feat)
        graph_temp = self.graph_fusion_temp(audio_pool_temp, text_pool_temp)

        # print(graph_feat.shape, graph_temp.shape)
        # print(torch.mean(graph_feat, dim=1).shape, torch.mean(graph_temp, dim=1).shape)

        # graph_feat = self.fc_feat(graph_feat)
        # graph_temp = self.fc_temp(graph_temp)

        # Классификация
        features = torch.cat([graph_feat, graph_temp], dim=1)

        # print(graph_feat.shape, graph_temp.shape, features.shape)
        return self.classifier(features)

    def adaptive_temporal_pool(self, x, target_len):
        """Адаптивное изменение временной длины"""
        if x.size(1) == target_len:
            return x

        return F.interpolate(
            x.permute(0, 2, 1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BiGatedGraphFormer(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, seg_len=44, hidden_dim=512, hidden_dim_gated=128,
                num_transformer_heads=2, num_graph_heads = 2, positional_encoding=True, dropout=0.1, mode='mean',
                device="cuda", tr_layer_number=1, out_features=128, num_classes=7):
        super(BiGatedGraphFormer, self).__init__()

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.seg_len = seg_len
        self.tr_layer_number = tr_layer_number

        # Проекционные слои с нормализацией
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Трансформерные слои (сохраняем вашу реализацию)
        self.audio_to_text_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.text_to_audio_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.graph_fusion_feat = GraphFusionLayer(self.seg_len, heads=num_graph_heads, out_mean=False)
        self.graph_fusion_temp = GraphFusionLayer(hidden_dim, heads=num_graph_heads, out_mean=False)

        self.gated_feat = GAL(self.seg_len, self.seg_len, hidden_dim_gated, dropout_rate=dropout)
        self.gated_temp = GAL(hidden_dim, hidden_dim, hidden_dim_gated, dropout_rate=dropout)

        # Автоматический расчёт размерности для классификатора
        self._calculate_classifier_input_dim()

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim_gated*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        # Финальная проекция графов
        self.fc_graph_feat = nn.Sequential(
            nn.Linear(self.seg_len, hidden_dim_gated),
            nn.LayerNorm(hidden_dim_gated),
            nn.Dropout(dropout)
        )

        self.fc_graph_temp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim_gated),
            nn.LayerNorm(hidden_dim_gated),
            nn.Dropout(dropout)
        )

        # Финальная проекция gated
        self.fc_gated_feat = nn.Sequential(
            nn.Linear(hidden_dim_gated, hidden_dim_gated),
            nn.LayerNorm(hidden_dim_gated),
            nn.Dropout(dropout)
        )

        self.fc_gated_temp = nn.Sequential(
            nn.Linear(hidden_dim_gated, hidden_dim_gated),
            nn.LayerNorm(hidden_dim_gated),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _calculate_classifier_input_dim(self):
        """Вычисляет размер входных признаков для классификатора"""
        # Тестовый проход через пулинг с dummy-данными
        dummy_audio = torch.randn(1, self.seg_len, self.hidden_dim)
        dummy_text = torch.randn(1, self.seg_len, self.hidden_dim)

        audio_pool_temp, audio_pool_feat = self._pool_features(dummy_audio)
        # text_pool_temp, _ = self._pool_features(dummy_text)

        combined = torch.cat([audio_pool_temp, audio_pool_feat], dim=1)
        self.classifier_input_dim = combined.size(1)

    def _pool_features(self, x):
        # Статистики по временной оси (seq_len)
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]

        # Статистики по feature оси  (hidden_dim)
        mean_feat = x.mean(dim=-1)  # [batch, seq_len]

        return mean_temp,  mean_feat

    def forward(self, audio_features, text_features):
        # Проекция признаков
        audio = self.audio_proj(audio_features.float())
        text = self.text_proj(text_features.float())

        # Адаптивный пулинг
        min_len = min(audio.size(1), text.size(1))
        audio = self.adaptive_temporal_pool(audio, min_len)
        text = self.adaptive_temporal_pool(text, min_len)

        # Кросс-модальное взаимодействие
        for i in range(self.tr_layer_number):
            attn_audio = self.audio_to_text_attn[i](text, audio, audio)
            attn_text = self.text_to_audio_attn[i](audio, text, text)

            audio = audio + attn_audio
            text = text + attn_text

        # Агрегация признаков
        audio_pool_temp, audio_pool_feat = self._pool_features(audio)
        text_pool_temp, text_pool_feat = self._pool_features(text)

        # print(audio_pool_temp.shape, audio_pool_feat.shape, text_pool_temp.shape, text_pool_feat.shape)

        graph_feat = self.graph_fusion_feat(audio_pool_feat, text_pool_feat)
        graph_temp = self.graph_fusion_temp(audio_pool_temp, text_pool_temp)

        gated_feat = self.gated_feat(graph_feat[:, 0, :], graph_feat[:, 1, :])
        gated_temp = self.gated_temp(graph_temp[:, 0, :], graph_temp[:, 1, :])

        fused_feat = self.fc_graph_feat(torch.mean(graph_feat, dim=1)) + self.fc_gated_feat(gated_feat)
        fused_temp = self.fc_graph_temp(torch.mean(graph_temp, dim=1)) + self.fc_gated_feat(gated_temp)

        # print(graph_feat.shape, graph_temp.shape)
        # print(torch.mean(graph_feat, dim=1).shape, torch.mean(graph_temp, dim=1).shape)

        # graph_feat = self.fc_feat(graph_feat)
        # graph_temp = self.fc_temp(graph_temp)

        # Классификация
        features = torch.cat([fused_feat, fused_temp], dim=1)

        # print(graph_feat.shape, graph_temp.shape, features.shape)
        return self.classifier(features)

    def adaptive_temporal_pool(self, x, target_len):
        """Адаптивное изменение временной длины"""
        if x.size(1) == target_len:
            return x

        return F.interpolate(
            x.permute(0, 2, 1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BiFormerWithProb(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, seg_len=44, hidden_dim=512, hidden_dim_gated=128,
                num_transformer_heads=2, num_graph_heads=2, positional_encoding=True, dropout=0.1, mode='mean',
                device="cuda", tr_layer_number=1, out_features=128, num_classes=7):
        super(BiFormerWithProb, self).__init__()

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.seg_len = seg_len
        self.tr_layer_number = tr_layer_number

        # Проекционные слои с нормализацией
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        # self.audio_proj = nn.Sequential(
        #     nn.Conv1d(audio_dim, hidden_dim, 1),
        #     nn.GELU(),
        # )

        # self.text_proj = nn.Sequential(
        #     nn.Conv1d(text_dim, hidden_dim, 1),
        #     nn.GELU(),
        # )

        # Трансформерные слои (сохраняем вашу реализацию)
        self.audio_to_text_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.text_to_audio_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        # Автоматический расчёт размерности для классификатора
        self._calculate_classifier_input_dim()

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        self.pred_fusion = PredictionsFusion(num_matrices=3, num_classes=num_classes)

        self._init_weights()

    def _calculate_classifier_input_dim(self):
        """Вычисляет размер входных признаков для классификатора"""
        # Тестовый проход через пулинг с dummy-данными
        dummy_audio = torch.randn(1, self.seg_len, self.hidden_dim)
        dummy_text = torch.randn(1, self.seg_len, self.hidden_dim)

        audio_pool = self._pool_features(dummy_audio)
        text_pool = self._pool_features(dummy_text)

        combined = torch.cat([audio_pool, text_pool], dim=1)
        self.classifier_input_dim = combined.size(1)

    def _pool_features(self, x):
        # Статистики по временной оси (seq_len)
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]

        # Статистики по feature оси  (hidden_dim)
        mean_feat = x.mean(dim=-1)  # [batch, seq_len]

        return torch.cat([mean_temp, mean_feat], dim=1)

    def forward(self, audio_features, text_features, audio_pred, text_pred):
        # Проекция признаков
        # audio = self.audio_proj(audio_features.permute(0,2,1)).permute(0,2,1)
        # text = self.text_proj(text_features.permute(0,2,1)).permute(0,2,1)
        audio = self.audio_proj(audio_features.float())
        text = self.text_proj(text_features.float())

        # Адаптивный пулинг
        min_len = min(audio.size(1), text.size(1))
        audio = self.adaptive_temporal_pool(audio, min_len)
        text = self.adaptive_temporal_pool(text, min_len)

        # Кросс-модальное взаимодействие
        for i in range(self.tr_layer_number):
            attn_audio = self.audio_to_text_attn[i](text, audio, audio)
            attn_text = self.text_to_audio_attn[i](audio, text, text)
            audio = audio + attn_audio
            text = text + attn_text

        # Агрегация признаков
        audio_pool = self._pool_features(audio)
        text_pool = self._pool_features(text)

        # Классификация
        features = torch.cat([audio_pool, text_pool], dim=1)

        out = self.classifier(features)

        w_out = self.pred_fusion([audio_pred, text_pred, out])
        return w_out

    def adaptive_temporal_pool(self, x, target_len):
        """Адаптивное изменение временной длины"""
        if x.size(1) == target_len:
            return x

        return F.interpolate(
            x.permute(0, 2, 1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BiGatedFormer(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, seg_len=44, hidden_dim=512, hidden_dim_gated=128,
                num_transformer_heads=2, num_graph_heads = 2, positional_encoding=True, dropout=0.1, mode='mean',
                device="cuda", tr_layer_number=1, out_features=128, num_classes=7):
        super(BiGatedFormer, self).__init__()

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.seg_len = seg_len
        self.tr_layer_number = tr_layer_number

        # Проекционные слои с нормализацией
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim) if audio_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Трансформерные слои (сохраняем вашу реализацию)
        self.audio_to_text_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.text_to_audio_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        # self.graph_fusion_feat = GraphFusionLayer(self.seg_len, heads=num_graph_heads, out_mean=False)
        # self.graph_fusion_temp = GraphFusionLayer(hidden_dim, heads=num_graph_heads, out_mean=False)

        self.gated_feat = GAL(self.seg_len, self.seg_len, hidden_dim_gated, dropout_rate=dropout)
        self.gated_temp = GAL(hidden_dim, hidden_dim, hidden_dim_gated, dropout_rate=dropout)

        # Автоматический расчёт размерности для классификатора
        self._calculate_classifier_input_dim()

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim_gated*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        # Финальная проекция графов
        # self.fc_graph_feat = nn.Sequential(
        #     nn.Linear(self.seg_len, hidden_dim_gated),
        #     nn.LayerNorm(hidden_dim_gated),
        #     nn.Dropout(dropout)
        # )

        # self.fc_graph_temp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim_gated),
        #     nn.LayerNorm(hidden_dim_gated),
        #     nn.Dropout(dropout)
        # )

        # Финальная проекция gated
        self.fc_gated_feat = nn.Sequential(
            nn.Linear(hidden_dim_gated, hidden_dim_gated),
            nn.LayerNorm(hidden_dim_gated),
            nn.Dropout(dropout)
        )

        self.fc_gated_temp = nn.Sequential(
            nn.Linear(hidden_dim_gated, hidden_dim_gated),
            nn.LayerNorm(hidden_dim_gated),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _calculate_classifier_input_dim(self):
        """Вычисляет размер входных признаков для классификатора"""
        # Тестовый проход через пулинг с dummy-данными
        dummy_audio = torch.randn(1, self.seg_len, self.hidden_dim)
        dummy_text = torch.randn(1, self.seg_len, self.hidden_dim)

        audio_pool_temp, audio_pool_feat = self._pool_features(dummy_audio)
        # text_pool_temp, _ = self._pool_features(dummy_text)

        combined = torch.cat([audio_pool_temp, audio_pool_feat], dim=1)
        self.classifier_input_dim = combined.size(1)

    def _pool_features(self, x):
        # Статистики по временной оси (seq_len)
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]

        # Статистики по feature оси  (hidden_dim)
        mean_feat = x.mean(dim=-1)  # [batch, seq_len]

        return mean_temp,  mean_feat

    def forward(self, audio_features, text_features):
        # Проекция признаков
        audio = self.audio_proj(audio_features.float())
        text = self.text_proj(text_features.float())

        # Адаптивный пулинг
        min_len = min(audio.size(1), text.size(1))
        audio = self.adaptive_temporal_pool(audio, min_len)
        text = self.adaptive_temporal_pool(text, min_len)

        # Кросс-модальное взаимодействие
        for i in range(self.tr_layer_number):
            attn_audio = self.audio_to_text_attn[i](text, audio, audio)
            attn_text = self.text_to_audio_attn[i](audio, text, text)

            audio = audio + attn_audio
            text = text + attn_text

        # Агрегация признаков
        audio_pool_temp, audio_pool_feat = self._pool_features(audio)
        text_pool_temp, text_pool_feat = self._pool_features(text)

        # print(audio_pool_temp.shape, audio_pool_feat.shape, text_pool_temp.shape, text_pool_feat.shape)

        # graph_feat = self.graph_fusion_feat(audio_pool_feat, text_pool_feat)
        # graph_temp = self.graph_fusion_temp(audio_pool_temp, text_pool_temp)

        gated_feat = self.gated_feat(audio_pool_feat, text_pool_feat)
        gated_temp = self.gated_temp(audio_pool_temp, text_pool_temp)

        # fused_feat = self.fc_graph_feat(torch.mean(graph_feat, dim=1)) + self.fc_gated_feat(gated_feat)
        # fused_temp = self.fc_graph_temp(torch.mean(graph_temp, dim=1)) + self.fc_gated_feat(gated_temp)

        # print(graph_feat.shape, graph_temp.shape)
        # print(torch.mean(graph_feat, dim=1).shape, torch.mean(graph_temp, dim=1).shape)

        # graph_feat = self.fc_feat(graph_feat)
        # graph_temp = self.fc_temp(graph_temp)

        # Классификация
        features = torch.cat([gated_feat, gated_temp], dim=1)

        # print(graph_feat.shape, graph_temp.shape, features.shape)
        return self.classifier(features)

    def adaptive_temporal_pool(self, x, target_len):
        """Адаптивное изменение временной длины"""
        if x.size(1) == target_len:
            return x

        return F.interpolate(
            x.permute(0, 2, 1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BiMamba(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, seg_len=44, hidden_dim=512, mamba_d_state=16,
                d_discr=None, mamba_ker_size=4, mamba_layer_number=2, dropout=0.1, mode='', positional_encoding=False,
                out_features=128, num_classes=7, device="cuda"):
        super(BiMamba, self).__init__()

        self.hidden_dim = hidden_dim
        self.seg_len = seg_len
        self.num_mamba_layers = mamba_layer_number
        self.device = device

        # Проекционные слои для каждой модальности
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Слой для объединения модальностей
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Mamba блоки для обработки объединенных признаков
        mamba_params = {
            'd_input': hidden_dim,
            'd_model': hidden_dim,
            'd_state': mamba_d_state,
            'd_discr': d_discr,
            'ker_size': mamba_ker_size
        }

        self.mamba_blocks = nn.ModuleList([
            nn.Sequential(
                MambaBlock(**mamba_params),
                RMSNorm(hidden_dim)
            )
            for _ in range(self.num_mamba_layers)
        ])

        # Автоматический расчет размерности классификатора
        # self._calculate_classifier_input_dim()

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(self.seg_len + self.hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        self._init_weights()

    # def _calculate_classifier_input_dim(self):
    #     """Вычисляет размер входных признаков для классификатора"""
    #     dummy = torch.randn(1, self.seg_len, self.hidden_dim)
    #     pooled = self._pool_features(dummy)
    #     self.classifier_input_dim = pooled.size(1)

    def _pool_features(self, x):
        """Объединение временных и feature статистик"""
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]
        mean_feat = x.mean(dim=-1)  # [batch, seq_len]
        full_feature = torch.cat([mean_temp, mean_feat], dim=1)
        if full_feature.shape[-1] == self.seg_len+self.hidden_dim:
            return torch.cat([mean_temp, mean_feat], dim=1)
        else:
            pad_size = self.seg_len+self.hidden_dim - full_feature.shape[-1]
            return F.pad(full_feature, (0, pad_size), mode="constant", value=0)

    def forward(self, audio_features, text_features):
        # Проекция признаков
        audio = self.audio_proj(audio_features.float())  # [B, T, D]
        text = self.text_proj(text_features.float())     # [B, T, D]

        # Адаптивный пулинг к минимальной длине
        min_len = min(audio.size(1), text.size(1))
        audio = self._adaptive_pool(audio, min_len)
        text = self._adaptive_pool(text, min_len)

        # Объединение модальностей
        fused = torch.cat([audio, text], dim=-1)  # [B, T, 2*D]
        fused = self.fusion_proj(fused)           # [B, T, D]

        # Обработка объединенных признаков через Mamba
        for mamba_block in self.mamba_blocks:
            out, _ = mamba_block[0](fused, None)
            out = mamba_block[1](out)
            fused = fused + out  # Residual connection

        # Агрегация признаков и классификация
        pooled = self._pool_features(fused)
        return self.classifier(pooled)

    def _adaptive_pool(self, x, target_len):
        """Адаптивное изменение временной длины"""
        if x.size(1) == target_len:
            return x
        return F.interpolate(
            x.permute(0, 2, 1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BiMambaWithProb(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=1024, seg_len=44, hidden_dim=512, mamba_d_state=16,
                d_discr=None, mamba_ker_size=4, mamba_layer_number=2, dropout=0.1, mode='',positional_encoding=False,
                out_features=128, num_classes=7, device="cuda"):
        super(BiMambaWithProb, self).__init__()

        self.hidden_dim = hidden_dim
        self.seg_len = seg_len
        self.num_mamba_layers = mamba_layer_number
        self.device = device

        # Проекционные слои для каждой модальности
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Слой для объединения модальностей
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Mamba блоки для обработки объединенных признаков
        mamba_params = {
            'd_input': hidden_dim,
            'd_model': hidden_dim,
            'd_state': mamba_d_state,
            'd_discr': d_discr,
            'ker_size': mamba_ker_size
        }

        self.mamba_blocks = nn.ModuleList([
            nn.Sequential(
                MambaBlock(**mamba_params),
                RMSNorm(hidden_dim)
            )
            for _ in range(self.num_mamba_layers)
        ])

        # Автоматический расчет размерности классификатора
        # self._calculate_classifier_input_dim()

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(self.seg_len + self.hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        self.pred_fusion = PredictionsFusion(num_matrices=3, num_classes=num_classes)

        self._init_weights()

    # def _calculate_classifier_input_dim(self):
    #     """Вычисляет размер входных признаков для классификатора"""
    #     dummy = torch.randn(1, self.seg_len, self.hidden_dim)
    #     pooled = self._pool_features(dummy)
    #     self.classifier_input_dim = pooled.size(1)

    def _pool_features(self, x):
        """Объединение временных и feature статистик"""
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]
        mean_feat = x.mean(dim=-1)  # [batch, seq_len]
        full_feature = torch.cat([mean_temp, mean_feat], dim=1)
        if full_feature.shape[-1] == self.seg_len+self.hidden_dim:
            return torch.cat([mean_temp, mean_feat], dim=1)
        else:
            pad_size = self.seg_len+self.hidden_dim - full_feature.shape[-1]
            return F.pad(full_feature, (0, pad_size), mode="constant", value=0)

    def forward(self, audio_features, text_features, audio_pred, text_pred):
        # Проекция признаков
        audio = self.audio_proj(audio_features.float())  # [B, T, D]
        text = self.text_proj(text_features.float())     # [B, T, D]

        # Адаптивный пулинг к минимальной длине
        min_len = min(audio.size(1), text.size(1))
        audio = self._adaptive_pool(audio, min_len)
        text = self._adaptive_pool(text, min_len)

        # Объединение модальностей
        fused = torch.cat([audio, text], dim=-1)  # [B, T, 2*D]
        fused = self.fusion_proj(fused)           # [B, T, D]

        # Обработка объединенных признаков через Mamba
        for mamba_block in self.mamba_blocks:
            out, _ = mamba_block[0](fused, None)
            out = mamba_block[1](out)
            fused = fused + out  # Residual connection

        # Агрегация признаков и классификация
        pooled = self._pool_features(fused)
        out = self.classifier(pooled)

        w_out = self.pred_fusion([audio_pred, text_pred, out])
        return w_out

    def _adaptive_pool(self, x, target_len):
        """Адаптивное изменение временной длины"""
        if x.size(1) == target_len:
            return x
        return F.interpolate(
            x.permute(0, 2, 1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
