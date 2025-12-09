import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        hidden_dim: int,
        n_layer: int,
        n_head: int,
        ff_dim: int,
        embed_drop: float,
        hidden_drop: float,
    ):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=hidden_drop,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, *args):
        # (batch_size, max_seq_len, embed_dim)
        mask = args[0] if len(args) > 0 else None
        tok_emb = self.tok_embedding(x)
        max_seq_len = x.shape[-1]
        pos_emb = self.pos_embedding(torch.arange(max_seq_len).to(x.device))

        x = tok_emb + pos_emb
        x = self.embed_dropout(x)
        x = self.linear1(x)
        x = self.encoder(x, src_key_padding_mask=mask, batch_first=True)
        x = self.linear2(x)

        probs = torch.matmul(x, self.tok_embedding.weight.T)
        return probs


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_layer: int,
        embed_drop: float,
        rnn_drop: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=n_layer,
            dropout=rnn_drop if n_layer > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x, _ = self.bilstm(x)
        return x

    def predict(self, x):
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.T)
        return probs

    def forward(self, x, *args):
        x = self.encode(x)
        return self.predict(x)


class BiLSTMAttn(BiLSTM):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_layer: int,
        embed_drop: float,
        rnn_drop: float,
        n_head: int,
    ):
        super().__init__(
            vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop
        )
        self.attn = nn.MultiheadAttention(hidden_dim, n_head, batch_first=True)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        x = self.attn(x, x, x, key_padding_mask=mask)[0]
        return self.predict(x)


class BiLSTMCNN(BiLSTM):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_layer: int,
        embed_drop: float,
        rnn_drop: float,
    ):
        super().__init__(
            vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop
        )
        self.conv = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )

    def forward(self, x, *args):
        x = self.encode(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        return self.predict(x)


class BiLSTMConvAttRes(BiLSTM):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        hidden_dim: int,
        n_layer: int,
        embed_drop: float,
        rnn_drop: float,
        n_head: int,
    ):
        super().__init__(
            vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop
        )
        self.attn = nn.MultiheadAttention(hidden_dim, n_head, batch_first=True)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        res = x
        x = self.conv(x.transpose(1, 2)).relu()
        x = x.transpose(1, 2)
        x = self.attn(x, x, x, key_padding_mask=mask)[0]
        x = self.norm(res + x)
        return self.predict(x)


class GRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_layer: int,
        embed_drop: float,
        rnn_drop: float,
    ):
        """
        单向 GRU 序列编码器，接口与 BiLSTM 保持一致：
        - 输入: (batch_size, seq_len)
        - 输出: (batch_size, seq_len, vocab_size) 的 logits
        适配:
        - preprocess.py 产出的 (input_ids, masks, ...) 张量
        - main.py 中对 logits 的使用 (最后一维为 vocab_size)
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=n_layer,
            dropout=rnn_drop if n_layer > 1 else 0.0,
            batch_first=True,
        )
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        x = self.embed_dropout(x)
        # out: (batch_size, seq_len, hidden_dim)
        x, _ = self.gru(x)
        return x

    def predict(self, x):
        # 映射回 embed_dim 并与 embedding 权重点积得到 vocab logits
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.T)
        return probs

    def forward(self, x, *args):
        # 与 BiLSTM 等保持相同签名，忽略可选的 mask 参数
        x = self.encode(x)
        return self.predict(x)


class CNN(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, embed_drop: float
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(
            in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, *args):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.T)
        return probs
