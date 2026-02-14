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

    processed_imgs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        embeddings_tmp = model(**processed_imgs)
    
    embeddings = embeddings_tmp.pooler_output
    embeddings = F.normalize(embeddings, dim=1)

# does the deseq? -- nope add as separate fn - change the 
class transcriptomics_encoder(nn.Module):
    def __init__(self, num_genes=768, out_dim=512):
        super(transcriptomics_encoder, self).__init__()

        self.nnlayers = nn.Sequential(
            nn.Linear(num_genes, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, X):
        return self.nnlayers(X)

class image_encoder(nn.Module):
    def __init__(self, first_layer_dim=768, out_dim=512):
        super(image_encoder, self).__init__()

        self.nnlayers = nn.Sequential(
            nn.Linear(first_layer_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, X):
        return self.nnlayers(X)

class joint_model(nn.Module):
    def __init__(self, in_dim=512, num_classes=9):
        super(joint_model, self).__init__()

        self.joint_layers = nn.Sequential(
            nn.Linear(in_dim*2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
)

        self.img_skip_connect = nn.Sequential(
            nn.Linear(in_dim, num_classes)
        )

        self.omics_skip_connect = nn.Sequential(
            nn.Linear(in_dim, num_classes)
        )

        self.prediction_head = nn.Linear(3, 1, bias=False)

    def forward(self, img_embed, omics_embed):
        pred_img   = self.img_skip_connect(img_embed)            
        pred_omics = self.omics_skip_connect(omics_embed)               
        pred_joint = self.joint_layers(
            torch.cat([img_embed, omics_embed], dim=1)                       
        ) 

        final_pred = self.prediction_head(
            torch.cat([pred_img, pred_omics, pred_joint], dim=1)            
        )                                                                  

        return final_pred


        


