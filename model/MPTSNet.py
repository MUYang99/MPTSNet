import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DataEmbedding, clsWindowTransformer, Inception_CBAM, clsTransformer, DataEmbedding_v1, Transformer, WindowTransformer
from utils import fft_find_each_amplitude


class PeriodicBlock(nn.Module):
    def __init__(self, flag, periods, seq_length, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers):
        super(PeriodicBlock, self).__init__()
        self.periods = periods
        self.embed_dim = embed_dim

        self.cnn = nn.Sequential(
            Inception_CBAM(embed_dim, 1024),
            nn.GELU(),
            Inception_CBAM(1024, embed_dim)
        )

        if flag:
            self.transformer = Transformer(embed_dim, seq_length, embed_dim, num_heads, ff_dim, num_layers)

            self.transformers = nn.ModuleList([
                WindowTransformer(embed_dim * period, (seq_length // period) + 1, embed_dim_t, num_heads, ff_dim,
                                  num_layers)
                for period in periods
            ])
        else:
            self.transformer = clsTransformer(seq_length, embed_dim, num_heads, ff_dim, num_layers)

            self.transformers = nn.ModuleList([
                clsWindowTransformer(embed_dim * period, (seq_length // period) + 1, embed_dim_t, num_heads, ff_dim,
                                     num_layers)
                for period in periods
            ])

    def forward(self, x):  # (batch_size, embed_dim, seq_length)
        B = x.shape[0]
        C = x.shape[1]
        T = x.shape[2]

        time_point_features = self.transformer(x)  # (batch_size, embed_dim, T)

        global_features = []
        amplitudes = []
        for i, period in enumerate(self.periods):
            x_fft = x.permute(0, 2, 1).detach().cpu().numpy()
            amplitudes.append(fft_find_each_amplitude(x_fft, period))
            if T % period != 0:
                # padding
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, C, (length - T)]).to(x.device)
                out = torch.cat([x, padding], dim=2)  # (batch_size, embed_dim, length)
            else:
                length = T
                out = x  # (batch_size, embed_dim, T(length))
            num_period = length // period
            # reshape
            out = out.reshape(B, C, period, num_period).contiguous()  # (batch_size, embed_dim, period, num_period)
            local_features = []
            for j in range(num_period):
                feature = self.cnn(out[:, :, :, j])  # (batch_size, embed_dim, period)
                local_features.append(feature)
            local_features = torch.stack(local_features, dim=-1)  # (batch_size, embed_dim, period, num_period)

            # add res part
            local_features = out+local_features

            local_features = local_features.reshape(B, -1, num_period)  # (batch_size, embed_dim*period, num_period)
            global_feature = self.transformers[i](local_features)  # (batch_size, embed_dim*period, num_period)
            global_feature = global_feature.reshape(B, self.embed_dim, -1).contiguous()  # (batch_size, embed_dim, length)
            global_feature = global_feature[:, :, :T]  # (batch_size, embed_dim, T)
            global_features.append(global_feature)

        # Features fusion
        amplitudes = torch.cat(amplitudes, dim=1)

        global_features = torch.stack(global_features, dim=-1)  # (B, embed_dim, T, k)
        weights = torch.softmax(amplitudes, dim=1)  # (B, k)
        period_weight = weights.unsqueeze(1).unsqueeze(1).repeat(1, self.embed_dim, T, 1).to(x.device)  # (B, embed_dim, T, k)
        res = torch.sum(global_features * period_weight, -1)  # (B, embed_dim, T)

        res = res+time_point_features+x

        return res


class Model(nn.Module):
    def __init__(self, periods, flag, num_channels, seq_length, num_classes, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers):
        super(Model, self).__init__()
        if flag:
            print('[INFO] True')
            self.enc_embedding = DataEmbedding_v1(num_channels, embed_dim, dropout=0.1)
        else:
            self.enc_embedding = DataEmbedding(num_channels, embed_dim, seq_length, dropout=0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.model = nn.ModuleList([PeriodicBlock(flag, periods, seq_length, embed_dim,
                                    embed_dim_t, num_heads, ff_dim, num_layers)
                                    for _ in range(2)])

        self.activation = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(seq_length * embed_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_channels)
        x = self.enc_embedding(x)  # (batch_size, seq_length, embed_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_length)
        for i in range(2):
            x = self.layer_norm(self.model[i](x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)  # (B, embed_dim * T)
        output = self.fc(x.float())

        return output