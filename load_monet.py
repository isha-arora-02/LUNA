from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# Load MONET model
processor = AutoProcessor.from_pretrained("suinleelab/monet")
model = AutoModelForZeroShotImageClassification.from_pretrained("suinleelab/monet")