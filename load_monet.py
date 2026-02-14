from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
from torch import nn
import torch.nn.functional as F

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

def create_dataloader():
    pass

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
    
    embeddings = embeddings_tmp.pooler_output
    embeddings = F.normalize(embeddings, dim=1)

# does the deseq? -- nope add as separate fn
class transcriptomics_encoder(nn.Module):
    def __init__(self, first_layer_dim=768, out_dim=512):
        super(transcriptomics_encoder, self).__init__()

        self.nnlayers = nn.Sequential(
            nn.Linear(first_layer_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, X):
        self.nnlayers(X)

class image_encoder(nn.Module):
    def __init__(self, first_layer_dim=768, out_dim=512):
        super(image_encoder, self).__init__()

        self.nnlayers = nn.Sequential(
            nn.Linear(first_layer_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, X):
        self.nnlayers(X)

class joint_model(nn.Module):
    def __init__(self, in_dim=512*2, out_dim=128):
        super(joint_model, self).__init__()

        self.nn_layers = nn.Sequential(
            nn.Linear(in_dim, )
        )
    
    def process_data(self, img_embed, omics_embed):
        pass

    def forward(self, img_embed, omics_embed):
        pass

        


