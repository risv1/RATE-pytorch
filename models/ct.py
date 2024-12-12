import torch.nn as nn


class CausalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(CausalTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)
