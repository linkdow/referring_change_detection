import os
import numpy as np
from easydict import EasyDict as edict

# Configuration for Sentinel-2 Showcase
# Adapted from config_second.py for real-world satellite imagery

C = edict()
config = C
cfg = C  # legacy alias

# Repro
C.seed = 3407

# Dataset - Sentinel-2 Île-de-France
C.dataset_name = "sentinel_idf_showcase"
C.root_folder = os.path.abspath(os.path.join(os.getcwd(), "showcase", "data"))
C.A_format = ".png"
C.B_format = ".png"
C.gt_format = ".png"  # Not used for inference, but required by dataloader

# Classes - Same taxonomy as SECOND dataset
# This allows us to use the pre-trained SECOND model
C.num_classes = 7
C.class_names = [
    "Non-change",                     # 0
    "Low Vegetation",                 # 1 (grasslands, crops)
    "Non-vegetated Ground Surface",   # 2 (bare soil, construction sites)
    "Tree",                           # 3 (forests, urban parks)
    "Water",                          # 4 (Seine river, lakes)
    "Building",                       # 5 (urban development)
    "Playground",                     # 6 (sports fields, recreational areas)
]

# Splits - Using pairs.txt as test split for inference
C.train_split = "pairs"  # Will use pairs.txt
C.val_split = "pairs"
C.test_split = "pairs"
C.eval_class_selection = "first"

# Images - Sentinel-2 tiled to 512×512
C.image_height = 512
C.image_width = 512

# Model - Match SECOND training configuration
C.backbone = "sigma_small"  # Matches pretrained weights
C.decoder = "MambaDecoder"
C.decoder_embed_dim = 512
C.pretrained_model = None  # Will be specified at inference time
C.freeze_backbone = False
C.use_imagenet_pretrain = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# Inference settings
C.batch_size = 4  # Adjust based on GPU memory (increase for faster inference)
C.num_workers = 8

# Normalization
# Using ImageNet statistics as default
# Sentinel-2 RGB composites are already contrast-stretched to [0, 255]
C.norm_mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean (RGB)
C.norm_std = np.array([0.229, 0.224, 0.225])   # ImageNet std (RGB)
C.use_cached_norm = False
C.use_single_normalization = True

# Augmentation (disabled for inference)
C.use_color_jitter = False
C.jitter_hyper = 0.0

# Training settings (not used for inference, but required by config)
C.lr = 6e-5
C.weight_decay = 0.01
C.nepochs = 500

# Naming
C.trial_name = f"{C.dataset_name}_{C.backbone}_{C.decoder}"

# Showcase-specific settings
C.showcase = edict()
C.showcase.output_dir = os.path.abspath(os.path.join(os.getcwd(), "showcase", "results"))
C.showcase.save_predictions = True
C.showcase.save_overlays = True
C.showcase.confidence_threshold = 0.5  # Threshold for change detection
