import torch.nn as nn
import torch
import math
from timm.models.layers import trunc_normal_


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.view(b, c, 1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.kaiming_uniform_(self.positional_encoding, a=math.sqrt(5))

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_length, embed_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=20000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, embed_dim):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=embed_dim,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, embed_dim, seq_length, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, embed_dim=embed_dim)
        self.position_embedding = PositionalEmbedding(embed_dim=embed_dim, max_len=seq_length)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)


class TokenEmbedding_v1(nn.Module):
    def __init__(self, c_in, embed_dim):
        super(TokenEmbedding_v1, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=embed_dim,
                                   kernel_size=1, padding=0, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding_v1(nn.Module):
    def __init__(self, c_in, embed_dim, dropout=0.1):
        super(DataEmbedding_v1, self).__init__()

        self.value_embedding = TokenEmbedding_v1(c_in=c_in, embed_dim=embed_dim)
        self.position_embedding = PositionalEmbedding(embed_dim=embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)


class Inception_Block_1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_CBAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        self.CBAMBlock = CBAMBlock(out_channels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        res = self.CBAMBlock(res)
        # res = res+x
        return res


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock_1D(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=4, drop_path_rate=0):
        super(ConvNeXtBlock_1D, self).__init__()
        hidden_dim = in_channels * expansion_ratio
        self.dwconv = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)  # Depthwise conv
        self.norm = nn.LayerNorm(in_channels)
        self.pwconv1 = nn.Linear(in_channels, hidden_dim)  # Pointwise conv 1
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channels)  # Pointwise conv 2
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # Change to (batch, seq_len, channels)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # Change back to (batch, channels, seq_len)
        x = shortcut + self.drop_path(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x


class Transformer(nn.Module):
    def __init__(self, in_channels, length, embed_dim, num_heads, ff_dim, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, length)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )

    def forward(self, x):  # (batch_size, num_channels, seq_length)
        x = x.permute(0, 2, 1)  # (batch_size, length, num_channels)
        # x = self.embedding(x)  # (batch_size, length, embed_dim)
        # print("x shape: ", x.shape)
        x = self.positional_encoding(x.permute(1, 0, 2))  # (length, batch_size, embed_dim)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.permute(1, 2, 0).contiguous()  # (batch_size, embed_dim, length)
        return x


class clsTransformer(nn.Module):
    def __init__(self, length, embed_dim, num_heads, ff_dim, num_layers):
        super(clsTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, length + 1)  # +1 for [CLS]
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.output_layer = nn.Linear(embed_dim * 2, embed_dim)  # New output layer

    def forward(self, x):
        # x shape: (batch_size, embed_dim, seq_length)
        batch_size, embed_dim, seq_length = x.shape
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, embed_dim)
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_length+1, embed_dim)
        x = self.positional_encoding(x.permute(1, 0, 2))  # (seq_length+1, batch_size, embed_dim)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length+1, embed_dim)
        # Extract [CLS] token representation
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)
        # Process the rest of the sequence
        sequence_output = x[:, 1:, :]  # (batch_size, seq_length, embed_dim)
        # Integrate CLS output with sequence output
        cls_expanded = cls_output.unsqueeze(1).expand(-1, seq_length, -1)
        integrated_output = torch.cat([sequence_output, cls_expanded], dim=-1)  # (batch_size, seq_length, embed_dim*2)
        # Apply final output layer
        output = self.output_layer(integrated_output)  # (batch_size, seq_length, embed_dim)
        # Permute back to original shape
        output = output.permute(0, 2, 1)  # (batch_size, embed_dim, seq_length)

        return output


class clsTransformer_attn(nn.Module):
    def __init__(self, length, embed_dim, num_heads, ff_dim, num_layers):
        super(clsTransformer_attn, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, length + 1)  # +1 for [CLS]
        self.transformer_blocks = nn.ModuleList(
            [WeightTransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.output_layer = nn.Linear(embed_dim * 2, embed_dim)  # New output layer

    def forward(self, x):
        # x shape: (batch_size, embed_dim, seq_length)
        batch_size, embed_dim, seq_length = x.shape
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, embed_dim)
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_length+1, embed_dim)
        x = self.positional_encoding(x.permute(1, 0, 2))  # (seq_length+1, batch_size, embed_dim)
        attn_weights = None
        for transformer in self.transformer_blocks:
            x, attn_weights = transformer(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length+1, embed_dim)
        # Extract [CLS] token representation
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)
        # Process the rest of the sequence
        sequence_output = x[:, 1:, :]  # (batch_size, seq_length, embed_dim)
        # Integrate CLS output with sequence output
        cls_expanded = cls_output.unsqueeze(1).expand(-1, seq_length, -1)
        integrated_output = torch.cat([sequence_output, cls_expanded], dim=-1)  # (batch_size, seq_length, embed_dim*2)
        # Apply final output layer
        output = self.output_layer(integrated_output)  # (batch_size, seq_length, embed_dim)
        # Permute back to original shape
        output = output.permute(0, 2, 1)  # (batch_size, embed_dim, seq_length)

        return output, attn_weights


class WindowTransformer(nn.Module):
    def __init__(self, in_channels, length, embed_dim, num_heads, ff_dim, num_layers):
        super(WindowTransformer, self).__init__()
        self.embedding = nn.Linear(in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, length)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, in_channels)
        self.dropout = nn.Dropout(0.1)  # Add Dropout

    def forward(self, x):  # (batch_size, num_channels, seq_length)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_channels)
        x = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        # x = self.dropout(x)  # Apply dropout after embedding
        x = self.positional_encoding(x.permute(1, 0, 2))  # (seq_length, batch_size, embed_dim)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.permute(1, 0, 2).contiguous()  # (batch_size, seq_length, embed_dim)
        output = self.output_layer(x)  # (batch_size, seq_length, num_channels)
        output = output.permute(0, 2, 1)  # (batch_size, num_channels, seq_length)
        return output


class clsWindowTransformer(nn.Module):
    def __init__(self, in_channels, length, embed_dim, num_heads, ff_dim, num_layers):
        super(clsWindowTransformer, self).__init__()
        self.embedding = nn.Linear(in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, length + 1)  # +1 for [CLS]
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim * 2, in_channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        # x shape: (batch_size, num_channels, seq_length)
        batch_size, _, seq_length = x.shape

        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_channels)
        x = self.embedding(x)  # (batch_size, seq_length, embed_dim)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_length+1, embed_dim)

        x = self.positional_encoding(x.permute(1, 0, 2))  # (seq_length+1, batch_size, embed_dim)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.permute(1, 0, 2)  # (batch_size, seq_length+1, embed_dim)

        # Extract [CLS] token representation
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)

        # Process the rest of the sequence
        sequence_output = x[:, 1:, :]  # (batch_size, seq_length, embed_dim)

        # Integrate CLS output with sequence output
        cls_expanded = cls_output.unsqueeze(1).expand(-1, seq_length, -1)
        integrated_output = torch.cat([sequence_output, cls_expanded], dim=-1)  # (batch_size, seq_length, embed_dim*2)

        # Apply final output layer
        output = self.output_layer(integrated_output)  # (batch_size, seq_length, in_channels)
        output = output.permute(0, 2, 1)  # (batch_size, in_channels, seq_length)

        return output


class WeightTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(WeightTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        # print(attn_weights.shape)  #  torch.Size([batch_size, seq_length, seq_length]) "average multi-heads"
        return x, attn_weights


class WeightTransformer(nn.Module):
    def __init__(self, in_channels, length, embed_dim, num_heads, ff_dim, num_layers):
        super(WeightTransformer, self).__init__()
        self.embedding = nn.Linear(in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, length)
        self.transformer_blocks = nn.ModuleList(
            [WeightTransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, in_channels)
        self.dropout = nn.Dropout(0.1)  # Add Dropout

    def forward(self, x):  # (batch_size, num_channels, seq_length)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_channels)
        x = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        # x = self.dropout(x)  # Apply dropout after embedding
        x = self.positional_encoding(x.permute(1, 0, 2))  # (seq_length, batch_size, embed_dim)
        attn_weights = None
        for transformer in self.transformer_blocks:
            x, attn_weights = transformer(x)
        x = x.permute(1, 0, 2).contiguous()  # (batch_size, seq_length, embed_dim)
        output = self.output_layer(x)  # (batch_size, seq_length, num_channels)
        output = output.permute(0, 2, 1)  # (batch_size, num_channels, seq_length)
        # print('output.shape: ', output.shape)
        # print('attn_weights.shape: ', attn_weights.shape)  #  torch.Size([batch_size, seq_length, seq_length])
        return output, attn_weights


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)  # (batch, channels, 1)
        max_out = self.max_pool(x)  # (batch, channels, 1)

        avg_out = self.fc(avg_out)  # (batch, channels, 1)
        max_out = self.fc(max_out)  # (batch, channels, 1)

        out = avg_out + max_out  # (batch, channels, 1)
        return self.sigmoid(out)  # (batch, channels, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, seq_length)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, seq_length)

        x = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, seq_length)
        x = self.conv1(x)  # (batch, 1, seq_length)

        return self.sigmoid(x)  # (batch, 1, seq_length)


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel Attention
        x_channel_att = self.channel_attention(x)  # (batch, channels, 1)
        x = x * x_channel_att  # (batch, channels, seq_length)

        # Spatial Attention
        x_spatial_att = self.spatial_attention(x)  # (batch, 1, seq_length)
        x = x * x_spatial_att  # (batch, channels, seq_length)

        return x  # (batch, channels, seq_length)


if __name__ == "__main__":
    batch_size = 8
    channels = 16
    seq_length = 50
    x = torch.randn(batch_size, channels, seq_length)

    cbam = CBAMBlock(channels)
    output = cbam(x)
    print(output.shape)
