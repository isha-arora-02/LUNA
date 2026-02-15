# LUNA
### A multi-omics driven prediction model for skin lesions and rashes.

LUNA is a multimodal AI system that improves skin disease detection by integrating dermatologic images with blood-based transcriptomic data. Many conditions — including dermatitis, psoriasis, melanoma, and systemic lupus erythematosus — share overlapping visual features, making diagnosis challenging even for specialists. To address this, LUNA leverages MONET, a dermatology-specific vision-language foundation model built on OpenAI’s CLIP ViT-L/14 architecture, to extract clinically meaningful visual embeddings from skin images. These visual representations are combined with embeddings derived from publicly available transcriptomic datasets, which we refined through multicohort meta-analysis to isolate biologically meaningful gene expression signals and enhance downstream predictive performance.

By fusing visual phenotypes with molecular biomarkers, LUNA moves beyond single-modality skin classifiers toward a more biologically grounded diagnostic framework. Despite working with unpaired imaging and transcriptomic datasets, we designed a neural architecture capable of learning shared multimodal representations and deployed it through a web interface for clinical interaction. LUNA represents a step toward scalable, precision-driven dermatological decision-support systems.

### To run predictions:
Run predict.py, providing file paths for:
  * A skin image (e.g., .jpg or .png)
  * A transcriptomic data matrix (.csv)
The model will process both inputs and output a predicted diagnosis based on the integrated multimodal features.

Alternatively, visit our (proof-of-concept) website to upload the requisite files and obtain a diagnosis: https://luna-scan-view.base44.app/

