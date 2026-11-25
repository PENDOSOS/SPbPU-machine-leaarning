import torch
import torch.nn as nn


class VisualAdapter(nn.Module):
    def __init__(self, input_dim=192, output_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, visual_tokens):
        x = self.proj(visual_tokens)
        return self.norm(x)


if __name__ == "__main__":
    adapter = VisualAdapter().cuda()

    visual_tokens = torch.randn(
        1, 196, 192,
        device="cuda",
    )

    output = adapter(visual_tokens)

    print("Input:", tuple(visual_tokens.shape))
    print("Output:", tuple(output.shape))

    assert output.shape == (1, 196, 256)
    assert torch.isfinite(output).all()

    print("Visual Adapter CUDA: SUCCESS")