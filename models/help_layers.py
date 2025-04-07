import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from torch_geometric.nn import GATConv, RGCNConv, TransformerConv

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.gelu(x)  # Более плавная активация
        x = self.dropout(x)
        return self.layer_2(x)


class AddAndNorm(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].detach()  # Отключаем градиенты
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1, positional_encoding=False):
        super().__init__()
        self.input_dim = input_dim
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)
        self.positional_encoding = PositionalEncoding(input_dim) if positional_encoding else None

    def forward(self, key, value, query):
        if self.positional_encoding:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        attn_output, _ = self.self_attention(query, key, value, need_weights=False)
        x = self.add_norm_after_attention(attn_output, query)

        ff_output = self.feed_forward(x)
        x = self.add_norm_after_ff(ff_output, x)

        return x
    
class GAL(nn.Module):
    def __init__(self, input_dim_F1, input_dim_F2, gated_dim):
        super(GAL, self).__init__()
        
        self.WF1 = nn.Parameter(torch.Tensor(input_dim_F1, gated_dim))
        self.WF2 = nn.Parameter(torch.Tensor(input_dim_F2, gated_dim))

        init.xavier_uniform_(self.WF1)
        init.xavier_uniform_(self.WF2)

        dim_size_f = input_dim_F1 + input_dim_F2

        self.WF = nn.Parameter(torch.Tensor(dim_size_f, gated_dim))
        
        init.xavier_uniform_(self.WF)
        
    def forward(self, f1, f2):

        h_f1 = torch.tanh(torch.matmul(f1, self.WF1))
        h_f2 = torch.tanh(torch.matmul(f2, self.WF2))
        # print(h_f1.shape, h_f2.shape, self.WF.shape, torch.cat([f1, f2], dim=1).shape)
        z_f = torch.softmax(torch.matmul(torch.cat([f1, f2], dim=1), self.WF), dim=1)
        h_f = z_f*h_f1 + (1 - z_f)*h_f2
        return h_f
    
class GraphFusionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0, heads=2):
        super().__init__()
        # # Проекционные слои для признаков
        self.proj_audio = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.Dropout(dropout)
        )
        self.proj_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.Dropout(dropout)
        )
        
        # Графовые слои
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim)
        
        # Финальная проекция
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.Dropout(dropout)
        )
        
    def build_complete_graph(self, num_nodes):
        # Создаем полный граф (каждый узел соединен со всеми)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index).t().contiguous()

    def forward(self, audio_stats, text_stats):
        """
        audio_stats: [batch_size, hidden_dim]
        text_stats: [batch_size, hidden_dim]
        """
        batch_size = audio_stats.size(0)
        
        # Проекция признаков
        x_audio = F.relu(self.proj_audio(audio_stats))  # [batch_size, hidden_dim]
        x_text = F.relu(self.proj_text(text_stats))    # [batch_size, hidden_dim]
        
        # Объединение узлов (аудио и текст попеременно)
        nodes = torch.stack([x_audio, x_text], dim=1)  # [batch_size, 2, hidden_dim]
        nodes = nodes.view(-1, nodes.size(-1))        # [batch_size*2, hidden_dim]
        
        # Построение графа (полный граф для каждого элемента батча)
        edge_index = self.build_complete_graph(2)  # Граф для одной пары аудио-текст
        edge_index = edge_index.to(audio_stats.device)
        
        # Применение GAT
        x = F.relu(self.gat1(nodes, edge_index))
        x = self.gat2(x, edge_index)
        
        # Разделяем обратно аудио и текст
        x = x.view(batch_size, 2, -1)  # [batch_size, 2, hidden_dim]
        
        # Усреднение по модальностям
        fused = torch.mean(x, dim=1)   # [batch_size, hidden_dim]
        
        return self.fc(fused)    

class GraphFusionLayerAtt(nn.Module):
    def __init__(self, hidden_dim, heads=2):
        super().__init__()
        # Проекционные слои для признаков
        self.proj_audio = nn.Linear(hidden_dim, hidden_dim)
        self.proj_text = nn.Linear(hidden_dim, hidden_dim)
        
        # Графовые слои
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim)

        self.attention_fusion = nn.Linear(hidden_dim, 1)
        
        # Финальная проекция
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def build_complete_graph(self, num_nodes):
        # Создаем полный граф (каждый узел соединен со всеми)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index).t().contiguous()

    def forward(self, audio_stats, text_stats):
        """
        audio_stats: [batch_size, hidden_dim]
        text_stats: [batch_size, hidden_dim]
        """
        batch_size = audio_stats.size(0)
        
        # Проекция признаков
        x_audio = F.relu(self.proj_audio(audio_stats))  # [batch_size, hidden_dim]
        x_text = F.relu(self.proj_text(text_stats))    # [batch_size, hidden_dim]
        
        # Объединение узлов (аудио и текст попеременно)
        nodes = torch.stack([x_audio, x_text], dim=1)  # [batch_size, 2, hidden_dim]
        nodes = nodes.view(-1, nodes.size(-1))        # [batch_size*2, hidden_dim]
        
        # Построение графа (полный граф для каждого элемента батча)
        edge_index = self.build_complete_graph(2)  # Граф для одной пары аудио-текст
        edge_index = edge_index.to(audio_stats.device)
        
        # Применение GAT
        x = F.relu(self.gat1(nodes, edge_index))
        x = self.gat2(x, edge_index)
        
        # Разделяем обратно аудио и текст
        x = x.view(batch_size, 2, -1)  # [batch_size, 2, hidden_dim]
        
        # Усреднение по модальностям
        # fused = torch.mean(x, dim=1)   # [batch_size, hidden_dim]

        weights = F.softmax(self.attention_fusion(x), dim=1)
        fused = torch.sum(weights * x, dim=1)  # [batch_size, hidden_dim]
        
        return self.fc(fused)
    
# Full code see https://github.com/leson502/CORECT_EMNLP2023/tree/master/corect/model

class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, num_modals, gcn_conv, use_graph_transformer, graph_transformer_nheads):
        super(GNN, self).__init__()
        self.gcn_conv = gcn_conv
        self.use_graph_transformer=use_graph_transformer

        self.num_modals = num_modals
        
        if self.gcn_conv == "rgcn":
            print("GNN --> Use RGCN")
            self.conv1 = RGCNConv(g_dim, h1_dim, num_relations)

        if self.use_graph_transformer:
            print("GNN --> Use Graph Transformer")

            in_dim = h1_dim
                
            self.conv2 = TransformerConv(in_dim, h2_dim, heads=graph_transformer_nheads, concat=True)
            self.bn = nn.BatchNorm1d(h2_dim * graph_transformer_nheads)
            

    def forward(self, node_features, node_type, edge_index, edge_type):
        print(node_features.shape, edge_index.shape, edge_type.shape)

        if self.gcn_conv == "rgcn":
            x = self.conv1(node_features, edge_index, edge_type)
        
        if self.use_graph_transformer:
            x = nn.functional.leaky_relu(self.bn(self.conv2(x, edge_index)))
        
        return x

class GraphModel(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, device, modalities, wp, wf, edge_type, gcn_conv, use_graph_transformer, graph_transformer_nheads):
        super(GraphModel, self).__init__()

        self.n_modals = len(modalities)
        self.wp = wp
        self.wf = wf
        self.device = device
        self.gcn_conv=gcn_conv
        self.use_graph_transformer=use_graph_transformer

        print(f"GraphModel --> Edge type: {edge_type}")
        print(f"GraphModel --> Window past: {wp}")
        print(f"GraphModel --> Window future: {wf}")
        edge_temp = "temp" in edge_type
        edge_multi = "multi" in edge_type

        edge_type_to_idx = {}

        if edge_temp:
            temporal = [-1, 1, 0]
            for j in temporal:
                for k in range(self.n_modals):
                    edge_type_to_idx[str(j) + str(k) + str(k)] = len(edge_type_to_idx)
        else: 
            for j in range(self.n_modals):
                edge_type_to_idx['0' + str(j) + str(j)] = len(edge_type_to_idx)

        if edge_multi:
            for j in range(self.n_modals):
                for k in range(self.n_modals):
                    if (j != k): 
                        edge_type_to_idx['0' + str(j) + str(k)] = len(edge_type_to_idx)

        self.edge_type_to_idx = edge_type_to_idx
        self.num_relations = len(edge_type_to_idx)
        self.edge_multi = edge_multi
        self.edge_temp = edge_temp

        self.gnn = GNN(g_dim, h1_dim, h2_dim, self.num_relations, self.n_modals, self.gcn_conv, self.use_graph_transformer, graph_transformer_nheads)

    
    def forward(self, x, lengths):
        # print(f"x shape: {x.shape}, lengths: {lengths}, lengths.shape: {lengths.shape}")

        node_features = feature_packing(x, lengths)

        node_type, edge_index, edge_type, edge_index_lengths = \
            self.batch_graphify(lengths)
        
        out_gnn = self.gnn(node_features, node_type, edge_index, edge_type)
        out_gnn = multi_concat(out_gnn, lengths, self.n_modals)

        return out_gnn

    def batch_graphify(self, lengths):

        node_type, edge_index, edge_type, edge_index_lengths = [], [], [], []
        edge_type_lengths = [0] * len(self.edge_type_to_idx)

        lengths = lengths.tolist()
        
        sum_length = 0
        total_length = sum(lengths)
        batch_size = len(lengths)

        for k in range(self.n_modals):
            for j in range(batch_size):
                cur_len = lengths[j]
                node_type.extend([k] * cur_len)
        
        for j in range(batch_size):
            cur_len = lengths[j]
            
            perms = self.edge_perms(cur_len, total_length)
            edge_index_lengths.append(len(perms))
            
            for item in perms:
                vertices = item[0]
                neighbor = item[1]
                edge_index.append(torch.tensor([vertices + sum_length, neighbor + sum_length]))
                
                if vertices % total_length > neighbor % total_length:
                    temporal_type = 1
                elif vertices % total_length < neighbor % total_length:
                    temporal_type = -1
                else:
                    temporal_type = 0
                edge_type.append(self.edge_type_to_idx[str(temporal_type) 
                                                + str(node_type[vertices + sum_length])
                                                + str(node_type[neighbor + sum_length])])

            sum_length += cur_len

        node_type = torch.tensor(node_type).long().to(self.device)
        edge_index = torch.stack(edge_index).t().contiguous().to(self.device)  # [2, E]
        edge_type = torch.tensor(edge_type).long().to(self.device)  # [E]
        edge_index_lengths = torch.tensor(edge_index_lengths).long().to(self.device)  # [B]

        return node_type, edge_index, edge_type, edge_index_lengths

    def edge_perms(self, length, total_lengths):
        
        all_perms = set()
        array = np.arange(length)
        for j in range(length):
            if self.wp == -1 and self.wf == -1:
                eff_array = array
            elif self.wp == -1:  # use all past context
                eff_array = array[: min(length, j + self.wf)]
            elif self.wf == -1:  # use all future context
                eff_array = array[max(0, j - self.wp) :]
            else:
                eff_array = array[
                    max(0, j - self.wp) : min(length, j + self.wf)
                ]
            perms = set()

            
            for k in range(self.n_modals):
                node_index = j + k * total_lengths
                if self.edge_temp == True:
                    for item in eff_array:
                        perms.add((node_index, item + k * total_lengths))
                else:
                    perms.add((node_index, node_index))
                if self.edge_multi == True:
                    for l in range(self.n_modals):
                        if l != k:
                            perms.add((node_index, j + l * total_lengths))

            all_perms = all_perms.union(perms)
                
        return list(all_perms)

def feature_packing(multimodal_feature, lengths):
        batch_size = lengths.size(0)
        # print(multimodal_feature.shape, batch_size, lengths.shape)
        node_features = []

        for feature in multimodal_feature:
            for j in range(batch_size):
                cur_len = lengths[j].item()
                # print(f"feature.shape: {feature.shape}, j: {j}, cur_len: {cur_len}")
                node_features.append(feature[j,:cur_len])

        node_features = torch.cat(node_features, dim=0)

        return node_features

def multi_concat(nodes_feature, lengths, n_modals):
    sum_length = lengths.sum().item()
    feature = []
    for j in range(n_modals):
        feature.append(nodes_feature[j * sum_length : (j + 1) * sum_length])

    feature = torch.cat(feature, dim=-1)

    return feature