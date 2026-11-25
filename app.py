import torch
import gradio as gr

from torchvision import transforms
from image_caption_model import ImageCaptionModel
from tokenizer import CaptionTokenizer

torch.set_num_threads(4)

tokenizer = CaptionTokenizer.load("tokenizer.json")

model = ImageCaptionModel(vocab_size=len(tokenizer))

checkpoint = torch.load(
    "checkpoints_aug_dropout/best_model.pt",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@torch.inference_mode()
def predict(image):
    if image is None:
        return "Please upload an image."

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    generated = [tokenizer.word2idx["<BOS>"]]
    eos_id = tokenizer.word2idx["<EOS>"]

    for _ in range(32):
        input_ids = torch.tensor([generated], dtype=torch.long)

        logits = model(image_tensor, input_ids)
        next_token = logits[0, -1].argmax().item()

        if next_token == eos_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Generated caption"),
    title="Vision Mamba Image Captioning",
    description=(
        "Image caption generation using Vision Mamba Tiny "
        "and a Mamba-based decoder. Runs entirely on CPU."
    ),
)

if __name__ == "__main__":
    demo.launch()