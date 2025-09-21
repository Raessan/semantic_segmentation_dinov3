import torch
from torch import nn
import numpy as np
from src.model_head import ASPPDecoder
from src.model_backbone import DinoBackbone
from src.common import image_to_tensor
from src.utils import outputs_to_maps, visualize_maps
import config.config as cfg
import cv2
import sys
import time

COCO_ROOT = cfg.COCO_ROOT
IMG_SIZE = cfg.IMG_SIZE
PATCH_SIZE = cfg.PATCH_SIZE
IMG_MEAN = np.array(cfg.IMG_MEAN, dtype=np.float32)[:, None, None]
IMG_STD = np.array(cfg.IMG_STD, dtype=np.float32)[:, None, None]
HIDDEN_DIM = cfg.HIDDEN_DIM
TARGET_SIZE = cfg.TARGET_SIZE
DINOV3_DIR = cfg.DINOV3_DIR
DINO_MODEL = cfg.DINO_MODEL
DINO_WEIGHTS = cfg.DINO_WEIGHTS
MODEL_TO_NUM_LAYERS = cfg.MODEL_TO_NUM_LAYERS
MODEL_TO_EMBED_DIM = cfg.MODEL_TO_EMBED_DIM
MODEL_PATH_INFERENCE = cfg.MODEL_PATH_INFERENCE
IMG_INFERENCE_PATH = cfg.IMG_INFERENCE_PATH
CLASS_NAMES_PATH = cfg.CLASS_NAMES_PATH

# Get class names from COCO
with open(CLASS_NAMES_PATH) as f:
    class_names = [line.strip() for line in f]

device = "cuda" if torch.cuda.is_available() else "cpu"

n_layers_dino = MODEL_TO_NUM_LAYERS[DINO_MODEL]
dino_model = torch.hub.load(
        repo_or_dir=DINOV3_DIR,
        model=DINO_MODEL,
        source="local",
        weights=DINO_WEIGHTS
)
dino_backbone = DinoBackbone(dino_model, n_layers_dino).to(device)

embed_dim = MODEL_TO_EMBED_DIM[DINO_MODEL]

model_head = ASPPDecoder(in_ch = embed_dim,
                                 num_classes = len(class_names),
                                 target_size=(TARGET_SIZE, TARGET_SIZE)).to(device)

model_head.load_state_dict(torch.load(MODEL_PATH_INFERENCE))

image = cv2.imread(IMG_INFERENCE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
image_tensor = image_to_tensor(image, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)

# Inference
dino_backbone.eval()
model_head.eval()

with torch.no_grad():
    feat = dino_backbone(image_tensor)
    init = time.time()
    n_inference = 1000
    for i in range(n_inference):
        semantic_logits = model_head(feat)
    end = time.time()
    print("time per sample: ", (end-init)*1000/n_inference)

semantic_map = outputs_to_maps(semantic_logits, (IMG_SIZE, IMG_SIZE))
visualize_maps(image, semantic_map, class_names=class_names,
                alpha=0.6, figsize=(12, 8), draw_semantic_labels=True, semantic_label_fontsize=10,
                background_index=0)
