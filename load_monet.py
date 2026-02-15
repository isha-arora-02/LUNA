from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import SupConLoss
from tqdm import tqdm

def get_monet_model():
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

    return model, processor

def get_img_embeddings(model, processor, images: list, device='cpu'):
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
        embeddings_tmp = model.vision_model(**processed_imgs)
    
    embeddings = embeddings_tmp.pooler_output
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings


def precompute_img_embeddings(images: list, monet_model, monet_processor, device, batch_size=64):
    """
    Pre-compute all MONET embeddings once before training.

    Parameters:
    images: list of image objs
    monet_model: MONET model
    monet_processor: MONET processor
    device: which device to run on 
    batch_size: number of samples per batch

    Returns:
    precomputed_embeddings: all the embeddings from the MONET model for all batches
    """
    precomputed_embeddings = []
    monet_model.eval()
    
    for i in tqdm(range(0, len(images), batch_size), desc="pre-computing MONET embeddings"):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            emb = get_img_embeddings(monet_model, monet_processor, batch, device=device)
        precomputed_embeddings.append(emb.cpu())  

    return torch.cat(precomputed_embeddings, dim=0)  


class transcriptomics_encoder(nn.Module):
    def __init__(self, num_genes=663, out_dim=256):
        super(transcriptomics_encoder, self).__init__()

        self.nnlayers = nn.Sequential(
            nn.Linear(num_genes, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, X):
        return self.nnlayers(X)

class image_encoder(nn.Module):
    def __init__(self, first_layer_dim=1024, out_dim=256):
        super(image_encoder, self).__init__()

        self.nnlayers = nn.Sequential(
            nn.Linear(first_layer_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, X):
        return self.nnlayers(X)

class joint_model(nn.Module):
    def __init__(self, in_dim=256, num_classes=7):
        super(joint_model, self).__init__()

        self.joint_layers = nn.Sequential(
            nn.Linear(in_dim*2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
)

        self.img_skip_connect = nn.Sequential(
            nn.Linear(in_dim, num_classes)
        )

        self.omics_skip_connect = nn.Sequential(
            nn.Linear(in_dim, num_classes)
        )

        self.prediction_head = nn.Linear(3*num_classes, num_classes, bias=False)

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

    def save_model(self, fpath):
        torch.save(self.state_dict(), fpath)
    
    def load_model(self, fpath, device):
        pretrained_dict = torch.load(fpath, map_location=device)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


def compute_loss(final_pred, img_embed, omics_embed, y_val, lambda_val=0.7):
    """
    Compute loss of model. Both cross entropy and SupCon.

    Parameters:
    final_pred: output of joint_model of shape (batch_size, num_classes)
    img_embed: output of image_encoder of shape (batch_size, 512) 
    omics_embed: output of transcriptomics_encoder  of shape (batch_size, 512) 
    y_val: disease class truth values (as indices) of shape (batch_size, )
    lambda_val: weight on SupCon loss relative to CE
    """
    ce     = nn.CrossEntropyLoss()
    supcon = SupConLoss(temperature=0.1)

    loss_ce = ce(final_pred, y_val.long())

    img_z   = F.normalize(img_embed,   dim=1)
    omics_z = F.normalize(omics_embed, dim=1)
    loss_supcon = supcon(
        torch.cat([img_z,  omics_z], dim=0),
        torch.cat([y_val, y_val],  dim=0)
    )

    return loss_ce + lambda_val * loss_supcon


def predict(img_encoder, omics_encoder, joint, x_img_embedding=None, x_omics=None):
    """
    Inference from image data, omics data, or both.

    Parameters:
    img_encoder: image encoder instance
    omics_encoder: transcriptomics encoder instance
    joint: joint_model instance
    x_img_embedding: get_img_embeddings() output of shape (B, 768) (or none for no image data)
    x_omics: gene expression matrix of shape (B, 768) (or none for no omics data)
    """
    img_encoder.eval()
    omics_encoder.eval()
    joint.eval()

    with torch.no_grad():
        if x_omics is None:
            img_embed = img_encoder(x_img_embedding) 
            return joint.img_skip_connect(img_embed) 

        if x_img_embedding is None:
            omics_embed = omics_encoder(x_omics)    
            return joint.omics_skip_connect(omics_embed)                

        img_embed = img_encoder(x_img_embedding) 
        omics_embed = omics_encoder(x_omics)  
        return joint(img_embed, omics_embed)                      
        


