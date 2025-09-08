import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.utils import outputs_to_maps, maps_to_outputs, tensor_to_image
from src.dataset_coco import DatasetCOCOPanoptic

# Dummy test
def test_outputs_to_maps():

    # ---------- dataset setup (user-provided) ----------
    COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO'
    MODE = "val"
    IMG_SIZE = 640
    TARGET_SIZE = 320
    MEAN_IMG=[0.485, 0.456, 0.406]
    STD_IMG=[0.229, 0.224, 0.225]
    PATCH_SIZE = 16
    AUGMENT_PROB = 0.0

    # IMPORT/CONSTRUCT your dataset class (assumes it's defined in your workspace)
    # from your_dataset_module import DatasetCOCOPanoptic
    dataset = DatasetCOCOPanoptic(COCO_ROOT, MODE, IMG_SIZE, PATCH_SIZE, AUGMENT_PROB)
    print("Class names (len={}):".format(len(dataset.class_names)))
    print(dataset.class_names)

    # load one sample
    data = dataset.__getitem__(0)
    image_tensor, semantic_mask, instance_mask = data  # image_tensor: (3,H,W), semantic_mask: (H,W), instance_mask: (H,W)
    print("Loaded sample shapes:", image_tensor.shape, semantic_mask.shape, instance_mask.shape)

    # ensure HxW are IMG_SIZE (they should)
    H, W = semantic_mask.shape
    assert (H, W) == (IMG_SIZE, IMG_SIZE), "Expected dataset to return maps at IMG_SIZE"

    # batchify for maps_to_outputs (B=1)
    semantic_batch = semantic_mask.unsqueeze(0)  # (1,H,W)
    instance_batch = instance_mask.unsqueeze(0)  # (1,H,W)

    # produce per-instance masks & labels
    out_size = (TARGET_SIZE, TARGET_SIZE)  # we want outputs at image resolution for this test
    sem_ds, inst_masks_padded, inst_labels_padded, inst_counts = maps_to_outputs(
        semantic_batch, instance_batch, out_size, background_index=0, max_instances=None)

    print("Prepared targets shapes:",
          sem_ds.shape, inst_masks_padded.shape, inst_labels_padded.shape, inst_counts)

    # ---------- synthesize logits from GT targets ----------
    torch.manual_seed(0)
    device = torch.device('cpu')

    num_semantic_classes = len(dataset.class_names)  # channels for semantic logits
    C = num_semantic_classes
    Q = inst_masks_padded.shape[1]  # maximum number of queries / padded instances
    K = C  # we'll set the instance head to predict classes in the same label space

    # semantic logits: produce a strong logit for the GT class at each pixel (plus small noise)
    semantic_logits = torch.full((C, TARGET_SIZE, TARGET_SIZE), fill_value=-6.0, dtype=torch.float32)
    sem_np = sem_ds[0].cpu().numpy()
    for c in range(C):
        semantic_logits[c][sem_np == c] = 6.0
    semantic_logits += 0.05 * torch.randn_like(semantic_logits)  # small noise

    # mask logits: for each padded query, create a high logit inside the GT mask, low outside.
    mask_logits = torch.full((Q, TARGET_SIZE, TARGET_SIZE), fill_value=-6.0, dtype=torch.float32)
    labels_for_queries = np.full((Q,), fill_value=0, dtype=np.int64)
    scores_for_queries = np.zeros((Q,), dtype=np.float32)
    # number of valid instances for this image:
    n_inst = int(inst_counts[0].item())
    for q in range(Q):
        if q < n_inst:
            mask = inst_masks_padded[0, q].cpu().numpy()  # boolean
            mask_logits[q][mask] = 6.0
            labels_for_queries[q] = int(inst_labels_padded[0, q].item())
            scores_for_queries[q] = 0.99
        else:
            # leave as very negative (no-object)
            mask_logits[q] += -0.0
            labels_for_queries[q] = 0  # background
            scores_for_queries[q] = 0.0

    # class_logits: high logit for the label (in same label space), small for others
    class_logits = torch.full((Q, K), fill_value=-6.0, dtype=torch.float32)
    for q in range(Q):
        lbl = int(labels_for_queries[q])
        class_logits[q, lbl] = 6.0

    # optional: add tiny noise
    class_logits += 0.01 * torch.randn_like(class_logits)

    # run the inference->maps pipeline
    semantic_pred, instance_map, inst_infos = outputs_to_maps(
        semantic_logits, mask_logits, class_logits, (IMG_SIZE, IMG_SIZE),
        mask_thresh=0.5, score_thresh=0.01, overlap_thresh=0.7,
        background_class_idx=0, device=device)

    print("Recovered semantic_pred unique labels:", np.unique(semantic_pred))
    print("Recovered instance ids:", np.unique(instance_map))
    print("inst_infos len:", len(inst_infos))
    for info in inst_infos[:5]:
        print(info)

    # ---------- visualization ----------
    # convert image_tensor to HxW x 3 numpy in [0,1] for plotting (assumes image_tensor is in 0..1 range; adapt if normalized)
    img = tensor_to_image(image_tensor, MEAN_IMG, STD_IMG)

    def show_overlay(ax, base_img, mask, color=(1.0, 0.0, 0.0), alpha=0.5):
        ax.imshow(base_img)
        colored = np.zeros_like(base_img)
        colored[..., 0] = color[0]
        colored[..., 1] = color[1]
        colored[..., 2] = color[2]
        ax.imshow(np.where(mask[..., None], colored, 0), alpha=alpha)
        ax.axis('off')

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()

    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[0].axis('off')

    axs[1].imshow(sem_ds[0].cpu().numpy(), cmap='tab20')
    axs[1].set_title("GT semantic map")
    axs[1].axis('off')

    axs[2].imshow(instance_batch[0].cpu().numpy(), cmap='tab20')
    axs[2].set_title("GT instance map")
    axs[2].axis('off')

    axs[3].imshow(semantic_pred, cmap='tab20')
    axs[3].set_title("Pred semantic (from logits)")
    axs[3].axis('off')

    axs[4].imshow(instance_map, cmap='tab20')
    axs[4].set_title("Pred instance map")
    axs[4].axis('off')

    # overlay a few predicted instance masks on the image
    axs[5].imshow(img)
    axs[5].set_title("Overlay predicted instance masks")
    # pick up to 6 instances to overlay
    inst_ids = [info['id'] for info in inst_infos][:6]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, iid in enumerate(inst_ids):
        mask = (instance_map == iid)
        color = colors[i % len(colors)][:3]
        axs[5].imshow(np.where(mask[..., None], color, 0), alpha=0.45)
    axs[5].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_outputs_to_maps()