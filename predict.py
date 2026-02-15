import warnings
import torch
import pickle
from PIL import Image
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from load_monet import (
    image_encoder, 
    transcriptomics_encoder, 
    joint_model, 
    get_monet_model,
    get_img_embeddings,
    predict
)


class LUNAPredictor:
    def __init__(self, checkpoint_path, label_mapping_path, device=None):
        """
        Initialize the predictor with a trained model checkpoint.
        
        Parameters:
        checkpoint_path: path to the saved model checkpoint (.pt file)
        label_mapping_path: path to the label mapping pickle file
        device: device to run inference on (None = auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        

        with open(label_mapping_path, 'rb') as f:
            self.label_mapping = pickle.load(f)
        self.classes = self.label_mapping['classes']
        print(f"Loaded {len(self.classes)} classes: {self.classes}")
        

        print("Loading MONET model...")
        self.monet_model, self.monet_processor = get_monet_model()
        self.monet_model.to(self.device)
        self.monet_model.eval()
        

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        SUMMARY_CSV = "out_data/run_summary.csv"
        summary = pd.read_csv(SUMMARY_CSV)
        best_row = summary.sort_values("best_val_acc", ascending=False).iloc[0]
        # cfg = checkpoint.get("cfg", None)
        # if cfg is None:
        #     warnings.warn(
        #         "Checkpoint is missing 'cfg'. Re-save your checkpoint during training with a cfg dict "
        #         "(embed_dim, dropout, img_dim, omics_dim, num_classes)."
        #     , RuntimeWarning)
        # 
        #     cfg = {}
        #     cfg["embed_dim"]   = int(best_row["embed_dim"])
        #     cfg["img_dim"]  = int(1024)
        #     cfg["omics_dim"] = int(663)
        #     cfg["dropout"]     = float(best_row["dropout"])
        # print("Loaded model cfg:", cfg)

        # embed_dim = int(cfg["embed_dim"])
        # img_dim   = int(cfg.get("img_dim", 1024))
        # omics_dim = int(cfg.get("omics_dim", 663))
        # dropout   = float(cfg.get("dropout", 0.3))

        img_dim = 1024
        omics_dim = 663
        embed_dim = int(best_row["embed_dim"])
        dropout = float(best_row["dropout"])
        self.img_encoder = image_encoder(first_layer_dim=img_dim, out_dim=embed_dim).to(self.device)
        self.omics_encoder = transcriptomics_encoder(num_genes=omics_dim, out_dim=embed_dim).to(self.device)
        self.joint_model = joint_model(in_dim=embed_dim, num_classes=len(self.classes), dropout=dropout).to(self.device)


        self.img_encoder.load_state_dict(checkpoint["img_encoder"])
        self.omics_encoder.load_state_dict(checkpoint["omics_encoder"])
        self.joint_model.load_state_dict(checkpoint["joint_model"])
        #cfg = checkpoint.get("cfg", None)
        #if cfg is None:
        #    raise ValueError("Checkpoint missing 'cfg'. Re-save the checkpoint including model hyperparams.")

        # embed_dim = int(cfg["embed_dim"])
        # img_dim   = int(cfg.get("img_dim", 1024))
        # omics_dim = int(cfg.get("omics_dim", 663))
        # dropout   = float(cfg.get("dropout", 0.3))


        self.img_encoder = image_encoder(first_layer_dim=img_dim, out_dim=embed_dim).to(self.device)
        self.omics_encoder = transcriptomics_encoder(num_genes=omics_dim, out_dim=embed_dim).to(self.device)
        self.joint_model = joint_model(in_dim=embed_dim, num_classes=len(self.classes), dropout=dropout).to(self.device)


        self.img_encoder.load_state_dict(checkpoint["img_encoder"])
        self.omics_encoder.load_state_dict(checkpoint["omics_encoder"])
        self.joint_model.load_state_dict(checkpoint["joint_model"])
        

        self.img_encoder.eval()
        self.omics_encoder.eval()
        self.joint_model.eval()
        
        print("Model loaded successfully!")
    
    def load_image(self, image_path):
        """
        Load an image from a file path.
        
        Parameters:
        image_path: path to the image file
        
        Returns:
        PIL Image object
        """
        img = Image.open(image_path).convert('RGB')
        return img
    
    def predict_from_image(self, image_input, return_probabilities=False):
        """
        Make a prediction from an image (or list of images).
        
        Parameters:
        image_input: single image path (str), PIL Image, or list of either
        return_probabilities: if True, return class probabilities
        
        Returns:
        Dictionary containing predictions and optional probabilities
        """

        if isinstance(image_input, (str, Path)):
            images = [self.load_image(image_input)]
            single_image = True
        elif isinstance(image_input, Image.Image):
            images = [image_input]
            single_image = True
        elif isinstance(image_input, list):
            images = []
            for img in image_input:
                if isinstance(img, (str, Path)):
                    images.append(self.load_image(img))
                else:
                    images.append(img)
            single_image = False
        else:
            raise ValueError("image_input must be a path, PIL Image, or list of these")
        

        with torch.no_grad():
            img_embeddings = get_img_embeddings(
                self.monet_model, 
                self.monet_processor, 
                images, 
                device=self.device
            )
            

            predictions = predict(
                self.img_encoder,
                self.omics_encoder,
                self.joint_model,
                x_img_embedding=img_embeddings,
                x_omics=None
            )
            

            probabilities = torch.softmax(predictions, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        

        predicted_classes = predicted_classes.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        results = []
        for i in range(len(images)):
            pred_class_idx = predicted_classes[i]
            pred_class_name = self.classes[pred_class_idx]
            
            result = {
                'predicted_class': pred_class_name,
                'predicted_class_idx': int(pred_class_idx),
                'confidence': float(probabilities[i][pred_class_idx])
            }
            
            if return_probabilities:
                result['class_probabilities'] = {
                    self.classes[j]: float(probabilities[i][j])
                    for j in range(len(self.classes))
                }
            
            results.append(result)
        
        return results[0] if single_image else results
    
    def predict_from_image_and_omics(self, image_input, gene_expression, return_probabilities=False):
        """
        Make a prediction from both image and gene expression data.
        
        Parameters:
        image_input: single image path (str), PIL Image, or list of either
        gene_expression: numpy array or torch tensor of shape (n_samples, 663)
        return_probabilities: if True, return class probabilities
        
        Returns:
        Dictionary containing predictions and optional probabilities
        """
        # Handle single image vs batch
        if isinstance(image_input, (str, Path)):
            images = [self.load_image(image_input)]
            single_sample = True
        elif isinstance(image_input, Image.Image):
            images = [image_input]
            single_sample = True
        elif isinstance(image_input, list):
            images = []
            for img in image_input:
                if isinstance(img, (str, Path)):
                    images.append(self.load_image(img))
                else:
                    images.append(img)
            single_sample = False
        else:
            raise ValueError("image_input must be a path, PIL Image, or list of these")
        

        if isinstance(gene_expression, np.ndarray):
            gene_expression = torch.from_numpy(gene_expression).float()
        
        if single_sample and len(gene_expression.shape) == 1:
            gene_expression = gene_expression.unsqueeze(0)
        
        gene_expression = gene_expression.to(self.device)
        

        with torch.no_grad():
            img_embeddings = get_img_embeddings(
                self.monet_model, 
                self.monet_processor, 
                images, 
                device=self.device
            )
            

            predictions = predict(
                self.img_encoder,
                self.omics_encoder,
                self.joint_model,
                x_img_embedding=img_embeddings,
                x_omics=gene_expression
            )
            

            probabilities = torch.softmax(predictions, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        

        predicted_classes = predicted_classes.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        results = []
        for i in range(len(images)):
            pred_class_idx = predicted_classes[i]
            pred_class_name = self.classes[pred_class_idx]
            
            result = {
                'predicted_class': pred_class_name,
                'predicted_class_idx': int(pred_class_idx),
                'confidence': float(probabilities[i][pred_class_idx])
            }
            
            if return_probabilities:
                result['class_probabilities'] = {
                    self.classes[j]: float(probabilities[i][j])
                    for j in range(len(self.classes))
                }
            
            results.append(result)
        
        return results[0] if single_sample else results
    
    def predict_from_omics_only(self, gene_expression, return_probabilities=False):
        """
        Make a prediction from gene expression data only.
        
        Parameters:
        gene_expression: numpy array or torch tensor of shape (n_samples, 663)
        return_probabilities: if True, return class probabilities
        
        Returns:
        Dictionary containing predictions and optional probabilities
        """

        if isinstance(gene_expression, np.ndarray):
            gene_expression = torch.from_numpy(gene_expression).float()
        
        single_sample = len(gene_expression.shape) == 1
        if single_sample:
            gene_expression = gene_expression.unsqueeze(0)
        
        gene_expression = gene_expression.to(self.device)
        

        with torch.no_grad():
            predictions = predict(
                self.img_encoder,
                self.omics_encoder,
                self.joint_model,
                x_img_embedding=None,
                x_omics=gene_expression
            )
            

            probabilities = torch.softmax(predictions, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        

        predicted_classes = predicted_classes.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        results = []
        for i in range(gene_expression.shape[0]):
            pred_class_idx = predicted_classes[i]
            pred_class_name = self.classes[pred_class_idx]
            
            result = {
                'predicted_class': pred_class_name,
                'predicted_class_idx': int(pred_class_idx),
                'confidence': float(probabilities[i][pred_class_idx])
            }
            
            if return_probabilities:
                result['class_probabilities'] = {
                    self.classes[j]: float(probabilities[i][j])
                    for j in range(len(self.classes))
                }
            
            results.append(result)
        
        return results[0] if single_sample else results
    
def resolve_inputs_from_tables(args):
    """
    Returns: (image_path_str_or_None, gene_expr_row_numpy_or_None)
    """
    sample_id = getattr(args, "sample_id", None)
    if args.row_idx is None and sample_id is None:
        return None, None

    samples_df = pd.read_csv(args.samples_csv)
    expr_df = pd.read_csv(args.expr_csv, index_col=0)

    sample_id = getattr(args, "sample_id", None)
    if sample_id is not None:

        if args.samples_id_col not in samples_df.columns:
            raise ValueError(f"--samples-id-col '{args.samples_id_col}' not in samples CSV columns: {samples_df.columns.tolist()}")
        samp_match = samples_df[samples_df[args.samples_id_col].astype(str) == str(args.sample_id)]
        if len(samp_match) != 1:
            raise ValueError(f"Expected exactly 1 match for sample_id={args.sample_id} in samples CSV; got {len(samp_match)}")
        image_path = samp_match.iloc[0]['image_path']


        if str(args.sample_id) in expr_df.index.astype(str):
            expr_row = expr_df.loc[str(args.sample_id)].values
        elif args.expr_id_col in expr_df.columns:
            expr_match = expr_df[expr_df[args.expr_id_col].astype(str) == str(args.sample_id)]
            if len(expr_match) != 1:
                raise ValueError(f"Expected exactly 1 match for sample_id={args.sample_id} in expr CSV; got {len(expr_match)}")
            expr_row = expr_match.drop(columns=[args.expr_id_col]).iloc[0].values
            
        else:
            raise ValueError(
                f"Could not find sample_id={args.sample_id} in expr index and expr-id-col '{args.expr_id_col}' not present."
            )

        return str(image_path), expr_row


    idx = int(args.row_idx)

    if idx < 0 or idx >= len(samples_df):
        raise IndexError(f"--row-idx {idx} out of range for samples CSV (n={len(samples_df)})")
    if idx < 0 or idx >= len(expr_df):
        raise IndexError(f"--row-idx {idx} out of range for expr CSV (n={len(expr_df)})")

    if 'image_path' not in samples_df.columns:
        raise ValueError(f"'image_path' column not found in samples CSV: {args.samples_csv}")

    image_path = samples_df.iloc[idx]['image_path']
    expr_row = expr_df.iloc[idx].values
    
    return str(image_path), expr_row


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained LUNA model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to label mapping (.pkl file)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image file or directory of images')
    parser.add_argument('--gene-expr', type=str, default=None,
                       help='Path to gene expression CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output file for predictions')
    parser.add_argument('--probabilities', action='store_true',
                       help='Include class probabilities in output')
    parser.add_argument('--row-idx', type=int, default=None,
                        help='Row index to pull from final_expr.csv and final_sorted_samples.csv')
    parser.add_argument('--label', type = str, default = 0,
                        help='0 means true phenotype label for gene expression')
    parser.add_argument('--expr-csv', type=str,
                        default='/home/provido/provido/luna_final/LUNA-main/gene_expr_data/final_expr.csv',
                        help='Path to final expression matrix CSV (rows=samples)')

    parser.add_argument('--samples-csv', type=str,
                        default='/home/provido/provido/luna_final/LUNA-main/image_data/final_sorted_samples.csv',
                        help='Path to samples CSV containing image_path column')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = LUNAPredictor(args.checkpoint, args.labels)
    
    # Load gene expression data if provided

    gene_expr = None
    if args.gene_expr:
        gene_expr_df = pd.read_csv(args.gene_expr, index_col=0)
        gene_expr = gene_expr_df.values
        print(f"Loaded gene expression data with shape: {gene_expr.shape}")

    # If user provided --row-idx or --sample-id, resolve inputs from the big tables
    auto_image, auto_expr_row = resolve_inputs_from_tables(args)

    # Prefer explicit CLI inputs, but fill missing ones from auto resolution
    if args.image is None and auto_image is not None:
        args.image = auto_image

    # If user did not pass --gene-expr, use the single resolved row
    if args.gene_expr is None and auto_expr_row is not None:
        gene_expr = auto_expr_row  # shape (663,)
        print(f"Loaded gene expression row from tables with shape: {gene_expr.shape}")
      
    if args.labels == 0:
        print("True Label for GEP data to predict: ", pheno.iloc[args.row_idx]['disease_status'])
    if args.labels != 0:
        print("True Label for Image to predict: ", imgpheno.iloc[args.row_idx]['dermatologist_skin_condition_on_label_name'])
    # Make predictions
    if args.image:
        image_path = Path(args.image)
        
        # Handle single image
        if image_path.is_file():
            print(f"\nProcessing single image: {image_path}")
            
            try:
                if gene_expr is not None:
                    print("Making prediction with image + gene expression...")
                    result = predictor.predict_from_image_and_omics(
                        str(image_path), 
                        gene_expr[0] if len(gene_expr.shape) > 1 else gene_expr,
                        return_probabilities=args.probabilities
                    )
                else:
                    print("Making prediction with image only...")
                    result = predictor.predict_from_image(
                        str(image_path),
                        return_probabilities=args.probabilities
                    )
                if args.row_idx is not None:
                    print("\n" + "="*60)
                    #print("True Label to predict: ", pheno.iloc[args.row_idx]['disease_status'])
                    #print("True Label to predict: ", imgpheno.iloc[args.row_idx]['dermatologist_skin_condition_on_label_name'])
                    #print("Observed conditions on skin: ", imgpheno.iloc[args.row_idx]['dermatologist_skin_condition_on_label_name'])
                    print("\n" + "="*60)
                print("\n" + "="*60)
                print("DIAGNOSIS")
                print("="*60)
                print(f"Predicted Class: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print("="*60)
                
                if args.probabilities:
                    print("\nClass Probabilities:")
                    for class_name, prob in result['class_probabilities'].items():
                        print(f"  {class_name}: {prob:.2%}")
            
            except Exception as e:
                print(f"\nERROR during prediction: {e}")
                import traceback
                traceback.print_exc()
        
        # Handle directory of images
        elif image_path.is_dir():
            image_files = list(image_path.glob('*.png')) + \
                         list(image_path.glob('*.jpg')) + \
                         list(image_path.glob('*.jpeg'))
            
            print(f"Processing {len(image_files)} images from directory: {image_path}")
            
            results = []
            for img_file in image_files:
                if gene_expr is not None:
                    result = predictor.predict_from_image_and_omics(
                        str(img_file),
                        gene_expr,
                        return_probabilities=args.probabilities
                    )
                else:
                    result = predictor.predict_from_image(
                        str(img_file),
                        return_probabilities=args.probabilities
                    )
                
                result['image_file'] = img_file.name
                results.append(result)
            
            # Save to CSV
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
            print(f"\nSummary:")
            print(df['predicted_class'].value_counts())
    
    elif gene_expr is not None:
        print("Processing gene expression data only")
        results = predictor.predict_from_omics_only(
            gene_expr,
            return_probabilities=args.probabilities
        )
        
        if isinstance(results, list):
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
        else:
            print("\nPrediction Result:")
            print(f"Predicted Class: {results['predicted_class']}")
            print(f"Confidence: {results['confidence']:.4f}")
    
    else:
        print("Error: Must provide either --image or --gene-expr")


if __name__ == "__main__":
    pheno = pd.read_csv("gene_expr_data/final_pheno.csv")
    imgpheno = pd.read_csv("image_data/final_sorted_samples.csv")
    print(imgpheno.columns)
    main()
"""
True use
 python predict.py \
     --checkpoint best_model.pt \
     --labels label_mapping.pkl \
     --image image_data/1161010728027923477.png \
     --gene-expr test_data/example_test_dermatitis.csv \
     --probabilities
"""


"""
 python predict.py \
     --checkpoint /home/provido/provido/luna_final/LUNA-main/best_model_by_acc_run48_ed128_lr0.0001_wd0.0001_do0.0_lam0.1_t0.2_bs64.pt \
     --labels /home/provido/provido/luna_final/LUNA-main/label_mapping.pkl \
     --image /home/provido/provido/luna/image_data/scin/1161010728027923477.png \
     --gene-expr /labs/khatrilab/provido/luna_final/LUNA-main/example_test_dermatitis.csv \
     --probabilities
     or -6637063938529970478.png 
"""

# python predict.py \
# --checkpoint /home/provido/provido/luna_final/LUNA-main/best_model.pt \
# --labels /home/provido/provido/luna_final/LUNA-main/label_mapping.pkl \
# --image /home/provido/provido/luna/image_data/scin/1161010728027923477.png \
# --gene-expr /labs/khatrilab/provido/luna_final/LUNA-main/example_test_dermatitis.csv \
# --probabilities
