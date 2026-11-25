import torch
import torch.nn as nn

from vim.models_mamba import VisionMamba
from visual_adapter import VisualAdapter
from caption_decoder import MambaCaptionDecoder


class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size=10000):
        super().__init__()

        self.encoder = VisionMamba(
            img_size=224,
            patch_size=16,
            stride=16,
            depth=24,
            embed_dim=192,
            d_state=16,
            bimamba_type="v2",
            if_divide_out=True,
            if_cls_token=True,
            use_middle_cls_token=True,
            fused_add_norm=False,
        )
        self.encoder.return_all_tokens = True

        self.adapter = VisualAdapter(
            input_dim=192,
            output_dim=256,
        )

        self.decoder = MambaCaptionDecoder(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=4,
        )

    def forward(self, images, input_ids):
        # Извлекаем визуальные признаки
        features = self.encoder(
            images,
            return_features=True,
        )

        if not torch.isfinite(features).all():
            raise RuntimeError("NaN/Inf in Vision Mamba encoder")

        # Удаляем CLS-токен из середины
        cls_position = features.shape[1] // 2

        visual_tokens = torch.cat(
            (
                features[:, :cls_position, :],
                features[:, cls_position + 1:, :],
            ),
            dim=1,
        )

        # Преобразуем размерность признаков
        visual_features = self.adapter(visual_tokens)

        if not torch.isfinite(visual_features).all():
            raise RuntimeError("NaN/Inf in Visual Adapter")

        for name, param in self.decoder.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    f"NaN/Inf in decoder parameter: {name}"
                )

        # Предсказываем следующие текстовые токены
        logits = self.decoder(visual_features, input_ids)

        if not torch.isfinite(logits).all():
            raise RuntimeError("NaN/Inf in Mamba Decoder")

        return logits


if __name__ == "__main__":
    model = ImageCaptionModel().cuda().eval()

    images = torch.randn(
        1, 3, 224, 224,
        device="cuda",
    )

    input_ids = torch.randint(
        0, 10000, (1, 16),
        device="cuda",
    )

    with torch.inference_mode():
        logits = model(images, input_ids)

    torch.cuda.synchronize()

    print("Images:", tuple(images.shape))
    print("Input IDs:", tuple(input_ids.shape))
    print("Logits:", tuple(logits.shape))

    assert logits.shape == (1, 16, 10000)
    assert torch.isfinite(logits).all()

    print("Image Caption Model: SUCCESS")