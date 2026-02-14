import torch
import clip


def get_transform(n_px):
    def convert_image_to_rgb(image):
        return image.convert("RGB")
    return T.Compose(
        [
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(n_px),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


# Load MONET model
device = "cuda:0"

model, preprocess = clip.load("ViT-L/14", device=device, jit=False)[0], get_transform(n_px=224)
model.load_state_dict(torch.hub.load_state_dict_from_url("https://aimslab.cs.washington.edu/MONET/weight_clip.pt"))

# Load MONET model - not sure how to get weights from huggingface
# from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
# processor = AutoProcessor.from_pretrained("suinleelab/monet")
# model = AutoModelForZeroShotImageClassification.from_pretrained("suinleelab/monet")
