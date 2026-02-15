import scanpy as sc
import pandas as pd
import anndata as ad 
import numpy as np
import torch
# from pydeseq2.dds import DeseqDataSet
# from pydeseq2.default_inference import DefaultInference
# from pydeseq2.ds import DeseqStats

def csv_to_anndata(fpath_expr, fpath_pheno):
    """
    Convert CSV of expression and phenotype data into Anndata files.

    Parameters:
    fpath_expr: file path to expression data
    fpath_pheno: file path to pheno/metadata

    Returns:
    adata: anndata object with expression and metadata 
    """
    expr = pd.read_csv(fpath_expr).T
    pheno = pd.read_csv(fpath_pheno).T

    adata = ad.AnnData(np.array(expr))
    adata.obs = pheno
    adata.obs_names = list(expr.index)
    adata.var_names = list(expr.columns)

    return adata

def updated_expr_deseq(fpath_expr, fpath_pheno):
    """
    Run differential gene expression on the expression matrix to obtain updated expression matrix. Keep top 768 diff expressed genes.

    Parameters:
    fpath_expr: file path to expression data
    fpath_pheno: file path to pheno/metadata

    Returns:
    adata: anndata object with expression and metadata 
    torch_X: X matrix in torch
    torch_y: y matrix in torch 
    """
    adata = csv_to_anndata(fpath_expr, fpath_pheno)
    sc.tl.rank_genes_groups(adata, groupby="disease_status", method="wilcoxon", use_raw=False)
    de_df = sc.get.rank_genes_groups_df(adata, group=None) 
    
    top_genes = (
        de_df.groupby("names")["scores"]
            .max()
            .nlargest(768)
            .index.tolist()
        )
    
    adata_filtered = adata[:, top_genes].copy()

    torch_X = torch.tensor(adata_filtered.X, dtype=torch.float32) 
    torch_y = torch.tensor(
        pd.Categorical(adata_filtered.obs["disease_status"]).codes,
        dtype=torch.long
        )        

    return adata_filtered, torch_X, torch_y









# extra code
# expr = pd.read_csv(fpath_expr).T
# pheno = pd.read_csv(fpath_pheno).T

# inference = DefaultInference(n_cpus=8)
# dds = DeseqDataSet(
#     counts=expr,
#     metadata=pheno["disease_status"],
#     design="~disease_status",
#     inference=inference
# )
# dds.deseq2()
# dds.obs["case_control"] = pheno["class"]
# stats = DeseqStats(dds, contrast=["case_control", 1, 0], inference=inference)
# stats.summary()
# de_results = stats.results_df

