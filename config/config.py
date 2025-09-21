# Dataset variables
COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO' # Location of dataset
IMG_SIZE = 640 # Size of the image (it will be square)
PATCH_SIZE = 16 # Patch size for the transformer embeddings
PROB_AUGMENT_TRAINING = 0.85 # Probability to perform photogrametric augmentation in training
PROB_AUGMENT_VALID = 0.0 # Probability to perform photogrametric augmentation in validation
IMG_MEAN = [0.485, 0.456, 0.406] # Mean of the image that the backbone (e.g. ResNet) expects
IMG_STD = [0.229, 0.224, 0.225] # Std of the image that the backbone (e.g. ResNet) expects

# Model parameters
DINOV3_DIR = '/home/rafa/deep_learning/projects/segmentation_dinov3/dinov3' # Directory for dinov3 code
DINO_MODEL = "dinov3_vits16plus" # Type of DINOv3 model to use
DINO_WEIGHTS = "/home/rafa/deep_learning/projects/semantic_segmentation_dinov3/dinov3_weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth" # Location of weights of DINOv3 model
MODEL_TO_NUM_LAYERS = { # Mapping from model type to number of layers
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}
MODEL_TO_EMBED_DIM = { # Mapping from model type to embedding dimension
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
}
HIDDEN_DIM=256 # Size of the hidden dim
TARGET_SIZE=320 # Size of the output of the NN (kept below the size img to prevent out of memory CUDA errors)

# TRAINING PARAMETERS
BATCH_SIZE = 8 # Batch size
WEIGHT_LOSS_DICE = 1.0 # Weight of dice loss
WEIGHT_LOSS_FOCAL = 1.0 # Weight of focal loss


LEARNING_RATE = 0.0001 # Learning rate
WEIGHT_DECAY = 0.0001 # Weight decay for regularization
NUM_EPOCHS = 20 # Number of epochs
NUM_SAMPLES_PLOT = 6 # Number of samples to plot during training or validation

LOAD_MODEL = False # Whether to load an existing model for training
SAVE_MODEL = True # Whether to save the result from the training
MODEL_PATH_TRAIN_LOAD = '/home/rafa/deep_learning/projects/semantic_segmentation_dinov3/results/2025-09-08_21-50-18/2025-09-09_23-19-30/model_3.pth' # Path of the model to load
RESULTS_PATH = '/home/rafa/deep_learning/projects/semantic_segmentation_dinov3/results' # Folder where the result will be saved

# PARAMETERS FOR INFERENCE
MODEL_PATH_INFERENCE = '/home/rafa/deep_learning/projects/semantic_segmentation_dinov3/weights/model.pth' # Path of the model to perform inference
IMG_INFERENCE_PATH = '/home/rafa/deep_learning/datasets/COCO/val2017/000000000139.jpg'
CLASS_NAMES_PATH = "/home/rafa/deep_learning/projects/semantic_segmentation_dinov3/src/class_names.txt" # Path of the file with class names