import torch, os
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import pickle, csv
import pandas as pd
import random, itertools
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
    
#Chris changed parameter args
class TrainValLUNA:
    def __init__(self, num_classes=7, img_dim=1024, omics_dim=663,
             embed_dim=512, lr=1e-4, weight_decay=0.01,
             dropout=0.3, lambda_val=0.7, supcon_temp=0.1):
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
        #CP adds the init of these 2 args
        self.lambda_val = lambda_val
        self.supcon_temp = supcon_temp
        
        self.cfg = {
            "embed_dim": embed_dim,
            "dropout": dropout,
            "img_dim": img_dim,
            "omics_dim": omics_dim,
            "num_classes": num_classes
        }
                
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
        
        # Chris is adding dropout here
        self.joint_model = joint_model(
            in_dim=embed_dim, 
            num_classes=num_classes, 
            dropout=dropout
            ).to(self.device)

        self.monet_model, self.monet_processor = get_monet_model()
        
        self.monet_model.to(self.device)
        
        trainable_params = (
            list(self.img_encoder.parameters()) +
            list(self.omics_encoder.parameters()) +
            list(self.joint_model.parameters())
        )
        #CP adjusts weight_decay
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = None
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = -1.0

    def precompute_monet_embeddings(self, images, batch_size):
        return precompute_img_embeddings(images, self.monet_model, self.monet_processor, self.device, batch_size)

    def train_model(self, train_loader, val_loader, num_epochs, run_name=None):
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
                
                # calc loss ## CP changes here
                loss = compute_loss(final_pred, img_embed, omics_embed, y_val,
                    lambda_val=self.lambda_val, supcon_temp=self.supcon_temp)
                
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
            # save best-by-accuracy checkpoint
            savep="/home/provido/provido/luna_final/LUNA-main/checkpoints/"
            fname = "/home/provido/provido/luna_final/LUNA-main/checkpoints/best_model_by_acc.pt" if run_name is None else f"{savep}best_model_by_acc_{run_name}.pt"
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    "cfg": self.cfg,
                    "img_encoder": self.img_encoder.state_dict(),
                    "omics_encoder": self.omics_encoder.state_dict(),
                    "joint_model": self.joint_model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }, fname)
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
                    'history': self.history,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, '/home/provido/provido/luna_final/LUNA-main/checkpoints/best_model.pt')
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
        
        return self.img_encoder, self.omics_encoder, self.joint_model, self.history


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
                
                # calculate loss ## CP changes
                loss = compute_loss(final_pred, img_embed, omics_embed, y_val,
                    lambda_val=self.lambda_val, supcon_temp=self.supcon_temp)
                val_loss += loss.item()
                
                # track accuracy metrics
                _, predicted = torch.max(final_pred.squeeze(1), 1)
                #print(predicted)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        
        # calc total loss and accuracy % for this model
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc


if __name__ == "__main__":
    # load the gene expression datasets - ensure they are in torch
    gene_expr_dat = pd.read_csv("gene_expr_data/final_expr.csv", index_col=0)
    gene_expr_dat_tensor = torch.from_numpy(gene_expr_dat.values).float()

    gene_expr_pheno = pd.read_csv("gene_expr_data/final_pheno.csv", index_col=0)
    gene_expr_pheno["disease_status_ind"] = pd.Categorical(gene_expr_pheno['disease_status']).codes
    gene_expr_labels_tensor = torch.tensor(gene_expr_pheno['disease_status_ind'].values, dtype=torch.long)

    # print statments to check
    print(f"Gene expression shape: {gene_expr_dat_tensor.shape}")
    print(f"Number of samples: {len(gene_expr_labels_tensor)}")
    print(f"Class distribution: {torch.bincount(gene_expr_labels_tensor)}")
    print(f"Classes: {pd.Categorical(gene_expr_pheno['disease_status']).categories.tolist()}")

    with open('/home/provido/provido/luna_final/LUNA-main/image_data/final_images_sorted.pkl', 'rb') as f:
        images_dat_lst = pickle.load(f)

    label_mapping = {
        'classes': pd.Categorical(gene_expr_pheno['disease_status']).categories.tolist(),
        'label_to_idx': dict(enumerate(pd.Categorical(gene_expr_pheno['disease_status']).categories))
        }
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
    
    #CP adds search space


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

    best = {"val_loss": float("inf"), "cfg": None}
    # ---- Hyperparameter tuning space (large) ----
    grid = {
        "embed_dim":     [128, 256, 512],
        "lr":            [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        "weight_decay":  [0.0, 1e-4, 1e-3, 1e-2],
        "dropout":       [0.0, 0.2, 0.3, 0.5],
        "lambda_val":    [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        "supcon_temp":   [0.05, 0.07, 0.10, 0.20],
        "batch_size":    [32, 64, 128],
    }
    results_csv = "/labs/khatrilab/provido/luna_final/LUNA-main/hparam_search_results.csv"
    fieldnames = ["run", "best_val_loss", "best_epoch", "best_val_acc",
              "embed_dim", "lr", "weight_decay", "dropout", "lambda_val", "supcon_temp", "batch_size"]
    write_header = not os.path.exists(results_csv)

    keys = list(grid.keys())
    all_cfgs = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    print(f"Total configs in full grid: {len(all_cfgs)}")

    # Optional: sample a large number without running the entire grid
    random.seed(42)
    N_TRIALS = 250  # increase (e.g., 200) if you really want to push it
    search_space = random.sample(all_cfgs, k=min(N_TRIALS, len(all_cfgs)))
    print(f"Running {len(search_space)} sampled configs")

    pbar = tqdm(list(enumerate(search_space)), total=len(search_space), desc="Hparam search")
    for i, cfg in pbar:
        pbar.set_postfix(run=i, embed_dim=cfg["embed_dim"], lr=cfg["lr"], bs=cfg["batch_size"])
        print("\n" + "="*80)
        print(f"RUN {i}: {cfg}")

        train_instance = TrainValLUNA(
            num_classes=7, img_dim=1024, omics_dim=663,
            embed_dim=cfg["embed_dim"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            dropout=cfg["dropout"],
            lambda_val=cfg["lambda_val"],
            supcon_temp=cfg["supcon_temp"],
        )
        train_emb_path = "/home/provido/provido/luna_final/LUNA-main/train_precomputed_monet_embeddings.pt"
        val_emb_path = "/home/provido/provido/luna_final/LUNA-main/val_precomputed_monet_embeddings.pt"
        train_img_embs = torch.load(train_emb_path, map_location="cpu")
        val_img_embs   = torch.load(val_emb_path,   map_location="cpu")

        train_dataset = DatasetPrep(train_img_embs, train_gene_expr, train_images_labels)
        val_dataset   = DatasetPrep(val_img_embs,   val_gene_expr,   val_images_labels)
        
        
        bs = cfg["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=1)
        val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=1)

        #train_instance.train_model(train_loader, val_loader, num_epochs=20)

        run_name = (f"run{i}_ed{cfg['embed_dim']}_lr{cfg['lr']}_wd{cfg['weight_decay']}"
                    f"_do{cfg['dropout']}_lam{cfg['lambda_val']}_t{cfg['supcon_temp']}_bs{cfg['batch_size']}")
        os.makedirs("/labs/khatrilab/provido/luna_final/LUNA-main/run_histories", exist_ok=True)

        img_enc, omics_enc, joint, history = train_instance.train_model(train_loader, val_loader, num_epochs=30, run_name=run_name)

        hist_df = pd.DataFrame({
            "epoch": list(range(len(history["train_acc"]))),
            "train_loss": history["train_loss"],
            "train_acc": history["train_acc"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
        })

        for k, v in cfg.items():
            hist_df[k] = v

        hist_df.to_csv(f"run_histories/{run_name}_epoch_history.csv", index=False)

        best_epoch = int(torch.load("best_model.pt", map_location="cpu")["epoch"])  # optional
        best_val_acc = float(max(train_instance.history["val_acc"])) if "val_acc" in train_instance.history else None  # optional

        row = {
            "run": i,
            "best_val_loss": train_instance.best_val_loss,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            **cfg
        }

        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)

        run_best = train_instance.best_val_loss
        if run_best < best["val_loss"]:
            best["val_loss"] = run_best
            best["cfg"] = cfg
            pbar.set_postfix(best_val_loss=best["val_loss"], run=i, lr=cfg["lr"], bs=cfg["batch_size"])
            # save which config won
            with open("best_hparam_config.pkl", "wb") as f:
                pickle.dump(best, f)

    print("\nBEST CONFIG:", best)

#CP commenting out original codebase
"""
    # initialize training class instance
    train_instance = TrainValLUNA(num_classes=7, img_dim=1024, omics_dim=663)

    print("Pre-computing train embeddings...")
    train_img_embs = train_instance.precompute_monet_embeddings(train_images, batch_size=64)
    torch.save(train_img_embs, "train_precomputed_monet_embeddings.pt")

    print("Pre-computing val embeddings...")
    val_img_embs   = train_instance.precompute_monet_embeddings(val_images, batch_size=64)
    torch.save(val_img_embs, "val_precomputed_monet_embeddings.pt")

    # ensure embeddings are the right shape
    assert train_img_embs.shape[1] == 1024, f"Expected 768 dims, got {train_img_embs.shape[1]}"
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
        num_epochs=30
    )
"""
