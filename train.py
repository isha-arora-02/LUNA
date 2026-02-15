import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from load_monet import image_encoder, transcriptomics_encoder, joint_model, get_img_embeddings, compute_loss, predict, get_monet_model, precompute_img_embeddings


class DatasetPrep(Dataset):
    def __init__(self, img_embeddings, gene_expression, labels):
        self.img_embeddings = img_embeddings
        self.gene_expression = gene_expression
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.img_embeddings[idx], self.gene_expression[idx], self.labels[idx]
    

class TrainValLUNA:
    def __init__(self, num_classes=6, img_dim=768, omics_dim=768, embed_dim=512, lr=1e-4):
        """
        Parameters:
        num_classes: number of predicted classes
        img_dim: input image dimension
        omics_dim: input omics dimension
        out_dim: output dimension for encoder models
        embed_dim: embedding dimension in mid of training
        device: device to run on
        lr: learning rate
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        
        self.img_encoder = image_encoder(
            first_layer_dim=img_dim, 
            out_dim=embed_dim
        ).to(self.device)
        
        self.omics_encoder = transcriptomics_encoder(
            num_genes=omics_dim, 
            out_dim=embed_dim
        ).to(self.device)
        
        self.joint_model = joint_model(
            in_dim=embed_dim, 
            num_classes=num_classes
        ).to(self.device)
        
        self.monet_model, self.monet_processor = get_monet_model()
        
        self.monet_model.to(self.device)
        
        trainable_params = (
            list(self.img_encoder.parameters()) +
            list(self.omics_encoder.parameters()) +
            list(self.joint_model.parameters())
        )
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
        self.scheduler = None
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        self.best_val_loss = float('inf')

    def precompute_monet_embeddings(self, images, batch_size):
        return precompute_img_embeddings(images, self.monet_model, self.monet_processor, self.device, batch_size)

    def train_model(self, train_loader, val_loader, num_epochs):
        """
        Train the model over multiple epochs and print statistics at each iteration.

        Parameters:
        train_loader: dataloader function for train dataset
        val_loader: dataloader function for validation dataset
        num_epochs: number of epochs of training
       
        Returns:
        img_enc: encoder model for images
        omics_enc: encoder model for omics
        joint: joint model
        """
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        for epoch in range(num_epochs):
            # Training phase
            self.img_encoder.train()
            self.omics_encoder.train()
            self.joint_model.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for img_embeddings, gene_expr, y_val in tqdm(train_loader, desc=f"Epoch {epoch}"):
                # get precomputed img embeddings
                img_embeddings = img_embeddings.to(self.device)

                # to device for expr and y labels
                gene_expr = gene_expr.to(self.device)
                y_val = y_val.to(self.device)
                
                # forward pass through models
                img_embed = self.img_encoder(img_embeddings)
                omics_embed = self.omics_encoder(gene_expr)
                final_pred = self.joint_model(img_embed, omics_embed)
                
                # calc loss
                loss = compute_loss(final_pred, img_embed, omics_embed, y_val, lambda_val=0.7)
                
                # backward pass through model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # track eval/accuracy %/loss values
                train_loss += loss.item()
                _, predicted_class = torch.max(final_pred.squeeze(1), 1)
                train_total += y_val.size(0)
                train_correct += (predicted_class == y_val).sum().item()
            
            # run validation
            val_loss, val_acc = self.validate(val_loader)            
            
            # update lr
            self.scheduler.step()
            
            # print stats so far
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            self.history['train_loss'].append(train_loss) 
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            with open('train_val_acc_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)

                        
            # saving best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'img_encoder': self.img_encoder.state_dict(),
                    'omics_encoder': self.omics_encoder.state_dict(),
                    'joint_model': self.joint_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, 'best_model.pt')
                print(f"Saved the best model so far with val_loss={val_loss:.4f}")


            if (epoch + 1) % 5 == 0:
                torch.save({
                    'img_encoder': self.img_encoder.state_dict(),
                    'omics_encoder': self.omics_encoder.state_dict(),
                    'joint_model': self.joint_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'history': self.history
                }, f'checkpoint_epoch_{epoch}.pt')
        
        return self.img_encoder, self.omics_encoder, self.joint_model


    def validate(self, val_loader):
        """
        Runs analyses for validation.

        Parameters:
        val_loader: dataloader function for validation dataset

        Returns:
        val_loss: validation loss
        val_acc: validation accuracy
        """

        self.img_encoder.eval()
        self.omics_encoder.eval()
        self.joint_model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        # don't update weights
        with torch.no_grad():
            for img_embeddings, gene_expr, y_val in val_loader:
                # transfer data to correct device
                gene_expr = gene_expr.to(self.device)
                y_val = y_val.to(self.device)
                
                # get embeddings and fwd pass through model for inference
                img_embeddings = img_embeddings.to(self.device)
                img_embed = self.img_encoder(img_embeddings)
                omics_embed = self.omics_encoder(gene_expr)
                final_pred = self.joint_model(img_embed, omics_embed)
                
                # calculate loss
                loss = compute_loss(final_pred, img_embed, omics_embed, y_val, lambda_val=0.7)
                val_loss += loss.item()
                
                # track accuracy metrics
                _, predicted = torch.max(final_pred.squeeze(1), 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        
        # calc total loss and accuracy % for this model
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc


if __name__ == "__main__":
    # load the gene expression datasets - ensure they are in torch
    gene_expr_dat = pd.read_csv("gene_expr_data/mergedExpr.csv", index_col=0).T
    gene_expr_dat_tensor = torch.from_numpy(gene_expr_dat.values).float()

    gene_expr_pheno = pd.read_csv("gene_expr_data/mergedPheno.csv", index_col=0)
    gene_expr_pheno["disease_status_ind"] = pd.Categorical(gene_expr_pheno['disease_status']).codes
    gene_expr_labels_tensor = torch.tensor(gene_expr_pheno['disease_status_ind'].values, dtype=torch.long)

    # print statments to check
    print(f"Gene expression shape: {gene_expr_dat_tensor.shape}")
    print(f"Number of samples: {len(gene_expr_labels_tensor)}")
    print(f"Class distribution: {torch.bincount(gene_expr_labels_tensor)}")
    print(f"Classes: {pd.Categorical(gene_expr_pheno['disease_status']).categories.tolist()}")

    # load the image datasets - ensure they are in torch
    images_dat_lst = ...
    images_labels_lst_tensor = ...

    label_mapping = {
        'classes': pd.Categorical(gene_expr_pheno['disease_status']).categories.tolist(),
        'label_to_idx': dict(enumerate(pd.Categorical(gene_expr_pheno['disease_status']).categories))
        }
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)

    # ensure labels were correctly generated 
    assert len(images_labels_lst_tensor) == len(gene_expr_labels_tensor), "Label length mismatch"
    assert torch.equal(images_labels_lst_tensor, gene_expr_labels_tensor), "Label value mismatch"

    # get train test splits
    train_images, val_images, \
    train_gene_expr, val_gene_expr, \
    train_images_labels, val_images_labels = train_test_split(
        images_dat_lst,
        gene_expr_dat_tensor,
        gene_expr_labels_tensor,
        test_size=0.25,
        stratify=gene_expr_labels_tensor.numpy(),
        random_state=42
    )

    # initialize training class instance
    train_instance = TrainValLUNA(num_classes=6, img_dim=768, omics_dim=663)

    print("Pre-computing train embeddings...")
    train_img_embs = train_instance.precompute_monet_embeddings(train_images, batch_size=64)
    torch.save(train_img_embs, "train_precomputed_monet_embeddings.pt")

    print("Pre-computing val embeddings...")
    val_img_embs   = train_instance.precompute_monet_embeddings(val_images, batch_size=64)
    torch.save(val_img_embs, "val_precomputed_monet_embeddings.pt")

    # ensure embeddings are the right shape
    assert train_img_embs.shape[1] == 768, f"Expected 768 dims, got {train_img_embs.shape[1]}"
    assert train_gene_expr.shape[1] == 663, f"Expected 663 dims, got {train_gene_expr.shape[1]}"

    # prepare datasets
    train_dataset = DatasetPrep(train_img_embs, train_gene_expr, train_images_labels)
    val_dataset = DatasetPrep(val_img_embs, val_gene_expr, val_images_labels)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=1)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=1)
    
    # run training and eval
    img_enc, omics_enc, joint = train_instance.train_model(
        train_loader, 
        val_loader, 
        num_epochs=50
    )