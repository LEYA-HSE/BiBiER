import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from torch.nn.functional import silu
from torch.nn.functional import softplus
from einops import rearrange, einsum
from torch import Tensor
from einops import rearrange

## Audio models

class CustomMambaBlock(nn.Module):
    def __init__(self, d_input, d_model, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_input, d_model)
        self.s_B = nn.Linear(d_model, d_model)
        self.s_C = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_input)
        self.norm = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_in = x  # сохраняем вход
        x = self.in_proj(x)
        B = self.s_B(x)
        C = self.s_C(x)
        x = x + B + C
        x = self.activation(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        x = self.norm(x + x_in)  # residual + norm
        return x
    
class CustomMambaClassifier(nn.Module):
    def __init__(self, input_size=1024, d_model=256, num_layers=2, num_classes=7, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([
            CustomMambaBlock(d_model, d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths, with_features=False):
        # x: (batch, seq_length, input_size)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        pooled = []
        for i, l in enumerate(lengths):
            if l > 0:
                pooled.append(x[i, :l, :].mean(dim=0))
            else:
                pooled.append(torch.zeros(x.size(2), device=x.device))
        pooled = torch.stack(pooled, dim=0)
        if with_features:
            return self.fc(pooled), x
        else:
            return self.fc(pooled)

def get_model_mamba(params):
    return CustomMambaClassifier(
        input_size=params.get("input_size", 1024),
        d_model=params.get("d_model", 256),
        num_layers=params.get("num_layers", 2),
        num_classes=params.get("num_classes", 7),
        dropout=params.get("dropout", 0.1)
    )

class EmotionModel(Wav2Vec2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        return hidden_states
    
## Text models

class Embedding():
    def __init__(self, model_name='jinaai/jina-embeddings-v3', pooling=None):
        self.model_name = model_name
        self.pooling = pooling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, code_revision='da863dd04a4e5dce6814c6625adfba87b83838aa', trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, code_revision='da863dd04a4e5dce6814c6625adfba87b83838aa', trust_remote_code=True).to(self.device)
        self.model.eval()

    def _mean_pooling(self, X):
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        encoded_input = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.unsqueeze(1)
    
    def get_embeddings(self, X, max_len):
        encoded_input = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            features = self.model(**encoded_input)[0].detach().cpu().float().numpy()
        res = np.pad(features[:, :max_len, :], ((0, 0), (0, max(0, max_len - features.shape[1])), (0, 0)), "constant")
        return torch.tensor(res)
        
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:        
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) * self.weight

class Mamba(nn.Module):
    def __init__(self, num_layers, d_input, d_model, d_state=16, d_discr=None, ker_size=4, num_classes=7, max_tokens=95, model_name='jina', pooling=None):
        super().__init__()
        mamba_par = {
            'd_input' : d_input,
            'd_model' : d_model,
            'd_state' : d_state,
            'd_discr' : d_discr,
            'ker_size': ker_size
        }
        self.model_name = model_name
        self.max_tokens = max_tokens
        embed = Embedding(model_name, pooling)
        self.embedding = embed.get_embeddings
        self.layers = nn.ModuleList([nn.ModuleList([MambaBlock(**mamba_par), RMSNorm(d_input)]) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_input, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, seq, cache=None, with_features=True):
        seq = self.embedding(seq, self.max_tokens).to(self.device)
        for mamba, norm in self.layers:
            out, cache = mamba(norm(seq), cache)
            seq = out + seq
        if with_features:
            return self.fc_out(seq.mean(dim = 1)), seq
        else:
            return self.fc_out(seq.mean(dim = 1))
        
class MambaBlock(nn.Module):
    def __init__(self, d_input, d_model, d_state=16, d_discr=None, ker_size=4):
        super().__init__()
        d_discr = d_discr if d_discr is not None else d_model // 16
        self.in_proj  = nn.Linear(d_input, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_input, bias=False)
        self.s_B = nn.Linear(d_model, d_state, bias=False)
        self.s_C = nn.Linear(d_model, d_state, bias=False)
        self.s_D = nn.Sequential(nn.Linear(d_model, d_discr, bias=False), nn.Linear(d_discr, d_model, bias=False),)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=ker_size,
            padding=ker_size - 1,
            groups=d_model,
            bias=True,
        )
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model, dtype=torch.float))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, seq, cache=None):
        b, l, d = seq.shape
        (prev_hid, prev_inp) = cache if cache is not None else (None, None)
        a, b = self.in_proj(seq).chunk(2, dim=-1)
        x = rearrange(a, 'b l d -> b d l')
        x = x if prev_inp is None else torch.cat((prev_inp, x), dim=-1)
        a = self.conv(x)[..., :l]
        a = rearrange(a, 'b d l -> b l d')
        a = silu(a)
        a, hid = self.ssm(a, prev_hid=prev_hid) 
        b = silu(b)
        out = a * b
        out =  self.out_proj(out)
        if cache:
            cache = (hid.squeeze(), x[..., 1:])   
        return out, cache
    
    def ssm(self, seq, prev_hid):
        A = -self.A
        D = +self.D
        B = self.s_B(seq)
        C = self.s_C(seq)
        s = softplus(D + self.s_D(seq))
        A_bar = einsum(torch.exp(A), s, 'd s,   b l d -> b l d s')
        B_bar = einsum(          B,  s, 'b l s, b l d -> b l d s')
        X_bar = einsum(B_bar, seq, 'b l d s, b l d -> b l d s')
        hid = self._hid_states(A_bar, X_bar, prev_hid=prev_hid)
        out = einsum(hid, C, 'b l d s, b l s -> b l d')
        out = out + D * seq
        return out, hid
    
    def _hid_states(self, A, X, prev_hid=None):
        b, l, d, s = A.shape
        A = rearrange(A, 'b l d s -> l b d s')
        X = rearrange(X, 'b l d s -> l b d s')
        if prev_hid is not None:
            return rearrange(A * prev_hid + X, 'l b d s -> b l d s')
        h = torch.zeros(b, d, s, device=self.device)
        return torch.stack([h := A_t * h + X_t for A_t, X_t in zip(A, X)], dim=1)

