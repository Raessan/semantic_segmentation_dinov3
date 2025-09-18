from torch.utils.data import Dataset
import cv2
import numpy as np
import random
import os
from src.common import image_to_tensor, tensor_to_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import json
import torch
import albumentations as A

class DatasetCOCOPanoptic(Dataset):
    """
    COCO Panoptic dataset loader.

    Expects COCO-style layout:
      root_dir/
        panoptic_train2017/            # PNG segmentation maps (R,G,B encoding of segment id)
        panoptic_val2017/
        images: train2017/ val2017/
        annotations/
          panoptic_train2017.json
          panoptic_val2017.json

    Returns: image_tensor, semantic_mask (H,W), instance_mask (H,W), segments_info
    """
    def __init__(self, root_dir, mode, img_size, patch_size,
                 augment_prob=0.8,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.patch_size = patch_size

        if self.mode == "train":
            self.images_dir = os.path.join(self.root_dir, "train2017")
            self.panoptic_dir = os.path.join(self.root_dir, "annotations", "panoptic_train2017")
            self.path_annotations = os.path.join(self.root_dir, "annotations", "panoptic_train2017.json")
        elif self.mode == "val":
            self.images_dir = os.path.join(self.root_dir, "val2017")
            self.panoptic_dir = os.path.join(self.root_dir, "annotations", "panoptic_val2017")
            self.path_annotations = os.path.join(self.root_dir, "annotations", "panoptic_val2017.json")
        else:
            raise ValueError("mode must be 'train' or 'val'")

        # Load panoptic JSON (different structure than instances json)
        with open(self.path_annotations, 'r') as f:
            self.panoptic_json = json.load(f)

        # Build mapping: image_id -> panoptic annotation entry
        # The panoptic JSON has a top-level 'annotations' list (one per image)
        self.panoptic_anns = {ann['image_id']: ann for ann in self.panoptic_json['annotations']}

        # If you also want instance/semantic category info:
        # panoptic JSON includes 'categories' list with 'id', 'name', 'isthing'
        self.categories = {c['id']: c for c in self.panoptic_json.get('categories', [])}
        self.class_names = [c['name'] for c in sorted(self.categories.values(), key=lambda x: x['id'])]
        self.class_names.insert(0, 'background') # Add background
        print("Total number of panoptic categories:", len(self.class_names))

        # Let's get thing classes
        thing_ids = [cid for cid, c in self.categories.items() if c["isthing"] == 1]
        self.thing_names = [self.categories[cid]["name"] for cid in sorted(thing_ids)]
        self.thing_names.insert(0,'background')
        print("Thing names ({}):".format(len(self.thing_names)), self.thing_names[:10], "…")

        self.augment_prob = augment_prob
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

        # list of image info entries from panoptic JSON (to preserve order)
        self.images = {img['id']: img for img in self.panoptic_json['images']}
        self.ids = list(sorted(self.images.keys()))

        self.transform = A.Compose([
            # --- Geometric (apply to both image & depth) ---
            A.HorizontalFlip(p=0.5),

            # a bit more aggressive crop/ratio
            A.RandomResizedCrop(size=(480, 640),
                                scale=(0.8, 1.0),
                                ratio=(0.95, 1.05),
                                p=0.6),

            # combine small shift/scale/rotate
            A.Affine(
                scale=(0.95, 1.05),               
                translate_percent=(-0.02, 0.02),   
                rotate=(-10, 10),                  
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,        
                fill=0,                             
                fill_mask=0,                        
                p=0.5
            ),

            # --- Photometric (image only) ---
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.RandomGamma(gamma_limit=(90, 110), p=0.2),

            # small amount of blur/noise; OneOf keeps it moderate
            A.OneOf([
                A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
        ],
            additional_targets={'semantic': 'mask',
                                'instance': 'mask'}
        )

        random.seed(42)

    def __len__(self):
        return len(self.ids)

    # --- Helpers ----------------------------------------------------------
    @staticmethod
    def _rgb_to_seg_id(rgb):
        """
        Convert an RGB triplet array (H,W,3 uint8) to a single integer segment id per pixel.
        COCO panoptic encoding: id = R + 256*G + 256^2*B
        """
        if rgb.ndim == 2:
            raise ValueError("Expected 3 channel RGB image for panoptic PNG")
        r = rgb[..., 0].astype(np.int32)
        g = rgb[..., 1].astype(np.int32)
        b = rgb[..., 2].astype(np.int32)
        seg_id = r + (g << 8) + (b << 16)
        return seg_id

    # --- Core -------------------------------------------------------------
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get panoptic annotation entry for this image
        if img_id not in self.panoptic_anns:
            raise KeyError(f"No panoptic annotation for image_id {img_id}")
        pan_ann = self.panoptic_anns[img_id]
        pan_png = pan_ann['file_name']  # something like "000000000139_panoptic.png"
        pan_path = os.path.join(self.panoptic_dir, pan_png)
        pan_rgb = cv2.imread(pan_path, cv2.IMREAD_UNCHANGED)
        if pan_rgb is None:
            raise FileNotFoundError(f"Panoptic PNG not found: {pan_path}")

        # pan_rgb is read as BGR by OpenCV. Convert to RGB ordering before decoding.
        if pan_rgb.shape[2] == 3:
            pan_rgb = cv2.cvtColor(pan_rgb, cv2.COLOR_BGR2RGB)
        seg_id_map = self._rgb_to_seg_id(pan_rgb)  # H x W integers giving segment id

        # Build per-pixel semantic mask (category_id) and instance mask (unique per segment)
        semantic_mask = np.zeros_like(seg_id_map, dtype=np.int32)   # category ids
        instance_mask = np.zeros_like(seg_id_map, dtype=np.int32)   # instance ids (unique small ints)
        segments = pan_ann.get('segments_info', [])

        # map segment id -> category id and assign instance indices
        segid_to_cat = {}
        segid_to_instid = {}
        next_inst_id = 1  # reserve 0 for background/void

        # We may need to know which categories are 'things' to give instance ids only to things
        isthing_map = {c_id: bool(c.get('isthing', 0)) for c_id, c in self.categories.items()}

        # track stuff categories already assigned an id
        stuff_class_to_instid = {}

        for seg in segments:
            sid = seg.get('id')
            cid = seg.get('category_id')
            cname = self.categories[cid]['name']
            label = self.class_names.index(cname)
            segid_to_cat[sid] = label
            if isthing_map.get(cid, False):
                segid_to_instid[sid] = next_inst_id
                next_inst_id += 1
            else:
                # With this line, we consider all the "stuff" as 0 class
                segid_to_instid[sid] = 0 
                
                # With the following lines (instead of the previous one) we consider each "stuff" as a separate class, with just one instance
                # stuff -> one id per category
                # if label not in stuff_class_to_instid:
                #     stuff_class_to_instid[label] = next_inst_id
                #     next_inst_id += 1
                # segid_to_instid[sid] = stuff_class_to_instid[label]

        # create masks by mapping seg_id_map values
        # To avoid huge loops, vectorized approach:
        # For each unique seg id present in the seg_id_map, map to category and instance
        unique_seg_ids = np.unique(seg_id_map)
        for sid in unique_seg_ids:
            if sid == 0:
                # Some datasets may use 0 for background/void; keep as 0
                continue
            if sid in segid_to_cat:
                mask = (seg_id_map == sid)
                semantic_mask[mask] = segid_to_cat[sid]
                instance_mask[mask] = segid_to_instid[sid]
            else:
                # Unknown segment id -> mark as void (category 0)
                continue

        # Optional augment (only photometric; geometric transforms need masks too)
        if random.random() < self.augment_prob:
            aug = self.transform(image=image, semantic=semantic_mask, instance=instance_mask)
            image = aug['image']
            semantic_mask = aug['semantic']
            instance_mask = aug['instance']

        # Resize image using helper, then resize masks to match
        image_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # get output size
        out_h, out_w = image_resized.shape[:2]

        # resize masks with nearest neighbor to avoid interpolation
        semantic_mask_resized = cv2.resize(semantic_mask.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        instance_mask_resized = cv2.resize(instance_mask.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        # convert to tensors
        image_tensor = image_to_tensor(image_resized, self.mean, self.std)
        semantic_mask_t = torch.from_numpy(semantic_mask_resized.astype(np.int64))
        instance_mask_t = torch.from_numpy(instance_mask_resized.astype(np.int32))

        return image_tensor, semantic_mask_t, instance_mask_t

if __name__ == '__main__':
    COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO'
    MODE = "val"
    IMG_SIZE = 640
    PATCH_SIZE = 16
    AUGMENT_PROB=1.0
    dataset = DatasetCOCOPanoptic(COCO_ROOT, MODE, IMG_SIZE, PATCH_SIZE, AUGMENT_PROB)
    print("Class names:\n", dataset.class_names)
    data = dataset.__getitem__(0)
    #dataset.visualize(0, 1.0)
    image, semantic_mask, _ = data # We don't use here the instance map
    print("image: ", image.shape, image.dtype)
    print("semantic_mask: ", semantic_mask.shape, semantic_mask.dtype)
    from utils import visualize_maps
    visualize_maps(tensor_to_image(image, dataset.mean, dataset.std),
                   semantic_mask.cpu().numpy(),
                   class_names=dataset.class_names,
                   alpha=0.6)