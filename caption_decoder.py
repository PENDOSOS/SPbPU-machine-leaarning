
import torch
import torch.nn as nn

from cpu_mamba import Mamba


class MambaCaptionDecoder(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, num_layers=4):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            for _ in range(num_layers)
        ])

        # Нормализация перед каждым блоком Mamba
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        #self.dropout = nn.Dropout(0.1)
        self.dropout = nn.Identity()

        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

    def forward(self, visual_features, input_ids):
        text_features = self.token_embedding(input_ids)

        x = torch.cat(
            [visual_features, text_features],
            dim=1,
        )

        for i, layer in enumerate(self.layers):
            # Pre-Norm + Residual Connection
            residual = x
            x = self.layer_norms[i](x)
            x = residual + self.dropout(layer(x))

            if not torch.isfinite(x).all():
                raise RuntimeError(
                    f"NaN/Inf after Mamba block {i}"
                )

        x = self.norm(x)

        text_hidden = x[:, visual_features.shape[1]:, :]

        logits = self.lm_head(text_hidden)

        return logits
