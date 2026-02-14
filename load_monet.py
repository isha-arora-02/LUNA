from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch

processor = AutoProcessor.from_pretrained("suinleelab/monet")
model = AutoModelForZeroShotImageClassification.from_pretrained("suinleelab/monet")

# weights = model.state_dict()
# print(type(weights))
# print(weights.keys())

# freeze weights
for param in model.parameters():
    param.requires_grad = False

# set in eval mode
model.eval()

def get_img_embeddings(images: list, device='cpu'):
    """
    Obtain embeddings for each image using MONET model.

    Parameters:
    images: list of the image objects
    device: which device to run on 

    Returns:
    embeddings: MONET embeddings of shape (batch_size, 768)
    """

    processed_imgs = processor(images=images, return_tensors="pt").to_device()

    with torch.no_grad():
        embeddings_tmp = model(**processed_imgs)

    


