# Dataset variables
COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO'
IMG_SIZE = 640
PATCH_SIZE = 16
PROB_AUGMENT_TRAINING = 0.85 # Probability to perform photogrametric augmentation in training
PROB_AUGMENT_VALID = 0.0 # Probability to perform photogrametric augmentation in validation
IMG_MEAN = [0.485, 0.456, 0.406] # Mean of the image that the backbone (e.g. ResNet) expects
IMG_STD = [0.229, 0.224, 0.225] # Std of the image that the backbone (e.g. ResNet) expects

# Model parameters
DINOV3_DIR = '/home/rafa/deep_learning/projects/segmentation_dinov3/dinov3'
DINO_MODEL = "dinov3_vits16plus"
DINO_WEIGHTS = '/home/rafa/deep_learning/projects/segmentation_dinov3/results/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "inov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}
MODEL_TO_EMBED_DIM = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "inov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
}
HIDDEN_DIM=256
TARGET_SIZE=320

# TRAINING PARAMETERS
BATCH_SIZE = 8 # Batch size
WEIGHT_LOSS_DICE = 1.0
WEIGHT_LOSS_FOCAL = 1.0


LEARNING_RATE = 0.0001 # Learning rate
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 20 # Number of epochs
NUM_SAMPLES_PLOT = 6 # Number of samples to plot during training or validation

LOAD_MODEL = False # Whether to load an existing model for training
SAVE_MODEL = True # Whether to save the result from the training
MODEL_PATH_TRAIN_LOAD = '/home/rafa/deep_learning/projects/segmentation_dinov3/results/2025-09-07_02-36-38/model_4.pth' # Path of the model to load
RESULTS_PATH = '/home/rafa/deep_learning/projects/semantic_segmentation_dinov3/results' # Folder where the result will be saved

# PARAMETERS FOR INFERENCE
MODEL_PATH_INFERENCE = '/home/rafa/deep_learning/projects/semantic_segmentation_dinov3/results/2025-09-08_21-50-18/model_13.pth' # Path of the model to perform inference
IMG_INFERENCE_PATH = '/home/rafa/deep_learning/datasets/COCO/val2017/000000000139.jpg'

CLASS_NAMES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 
               'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
               'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
               'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 
               'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 
               'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 
               'house', 'light', 'mirror-stuff', 'net', 'pillow', 
               'platform', 'playingfield', 'railroad', 'river', 'road', 
               'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 
               'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 
               'water-other', 'window-blind', 'window-other', 'tree-merged', 
               'fence-merged', 'ceiling-merged', 'sky-other-merged', 
               'cabinet-merged', 'table-merged', 'floor-other-merged', 
               'pavement-merged', 'mountain-merged', 'grass-merged', 
               'dirt-merged', 'paper-merged', 'food-other-merged', 
               'building-other-merged', 'rock-merged', 'wall-other-merged', 
               'rug-merged']