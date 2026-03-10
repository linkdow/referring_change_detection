import os
import numpy as np
from easydict import EasyDict as edict

# Config IGN Orthophoto — générée automatiquement par download_ign_orthophoto.py
# Zone     : Saint-Denis / Village Olympique
# BEFORE   : ORTHOIMAGERY.ORTHOPHOTOS2021
# AFTER    : ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2024
# Résol.   : 0.5 m/pixel  (identical to SECOND training data)
# Patch    : 512×512 px = 256m×256m

C = edict()
config = C
cfg = C

C.seed = 3407
C.dataset_name = "ign_orthophoto_showcase"
C.root_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "showcase", "data")
C.A_format = ".png"
C.B_format = ".png"
C.gt_format = ".png"

C.num_classes = 7
C.class_names = [
    "Non-change",
    "Low Vegetation",
    "Non-vegetated Ground Surface",
    "Tree",
    "Water",
    "Building",
    "Playground",
]

C.train_split = "pairs"
C.val_split   = "pairs"
C.test_split  = "pairs"
C.eval_class_selection = "first"

C.image_height = 512
C.image_width  = 512

C.backbone = "sigma_small"
C.decoder  = "MambaDecoder"
C.decoder_embed_dim = 512
C.pretrained_model = None
C.freeze_backbone = False
C.use_imagenet_pretrain = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

C.batch_size = 4
C.num_workers = 8

# Normalisation du dataset SECOND (entraînement) — à utiliser pour inférence
# Les stats IGN calculées (std~0.14-0.17) étirent les valeurs à std~1.02
# alors que le modèle attend std~0.82 → valeurs hors distribution → 0 détections.
C.norm_mean = np.array([0.439, 0.447, 0.459])
C.norm_std  = np.array([0.193, 0.183, 0.189])
C.use_cached_norm = False
C.use_single_normalization = True

C.use_color_jitter = False
C.jitter_hyper = 0.0

C.lr = 6e-5
C.weight_decay = 0.01
C.nepochs = 500

C.trial_name = f"{C.dataset_name}_{C.backbone}_{C.decoder}"

C.showcase = edict()
C.showcase.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "showcase", "results")
C.showcase.save_predictions = True
C.showcase.save_overlays = True
C.showcase.confidence_threshold = 0.5
