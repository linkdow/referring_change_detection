#!/usr/bin/env python3
"""
Télécharge des orthophotos IGN à deux dates différentes pour la détection de changements.

Utilise le WMS IGN Géoplateforme (data.geopf.fr) pour télécharger des images
haute résolution (0.5 m/pixel) — même résolution que les données d'entraînement
du modèle (dataset SECOND).

Couverture :
    BEFORE (A) : ORTHOIMAGERY.ORTHOPHOTOS2021
    AFTER  (B) : ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2024

Zone par défaut : La Défense / Nanterre (changements urbains visibles 2021→2024)

Usage :
    python download_ign_orthophoto.py                   # zone par défaut, ~841 tuiles
    python download_ign_orthophoto.py --area saclay     # zone tech Saclay
    python download_ign_orthophoto.py --center-lon 2.35 --center-lat 48.86 --size-km 5
    python download_ign_orthophoto.py --dry-run         # affiche la grille sans télécharger
    python download_ign_orthophoto.py --workers 8       # parallélisme
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# Configuration IGN WMS
# ─────────────────────────────────────────────────────────
WMS_URL = "https://data.geopf.fr/wms-r"

# Layers testés et confirmés disponibles sur l'IDF
LAYER_BEFORE = "ORTHOIMAGERY.ORTHOPHOTOS2021"            # ~2021
LAYER_AFTER  = "ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2024"  # ~2024

# Résolution d'entraînement du modèle (SECOND dataset : ~0.5 m/pixel)
PIXEL_SIZE_M = 0.5      # mètres par pixel
PATCH_SIZE_PX = 512     # pixels (identique à l'entraînement)
PATCH_SIZE_M  = PIXEL_SIZE_M * PATCH_SIZE_PX   # = 256 m

# Zones prédéfinies (longitude, latitude WGS84)
AREAS = {
    "ladefense": {
        "name": "La Défense / Nanterre",
        "lon": 2.238, "lat": 48.892,
        "size_km": 7.5,
        "description": "Quartier d'affaires, chantiers Tour Hekla et Grand Paris Express",
    },
    "saclay": {
        "name": "Paris-Saclay",
        "lon": 2.169, "lat": 48.716,
        "size_km": 7.5,
        "description": "Campus technologique en pleine expansion",
    },
    "stdenis": {
        "name": "Saint-Denis / Village Olympique",
        "lon": 2.360, "lat": 48.929,
        "size_km": 6.0,
        "description": "Village olympique Paris 2024, chantiers majeurs 2020→2024",
    },
    "massy": {
        "name": "Massy-Palaiseau",
        "lon": 2.268, "lat": 48.724,
        "size_km": 5.0,
        "description": "Pôle technologique et gare Grand Paris Express",
    },
}

# ─────────────────────────────────────────────────────────
# Téléchargement WMS
# ─────────────────────────────────────────────────────────

def download_tile(
    layer: str,
    bbox_l93: tuple[float, float, float, float],
    size_px: int = PATCH_SIZE_PX,
    retries: int = 3,
    timeout: int = 30,
) -> Optional[np.ndarray]:
    """
    Télécharge une tuile WMS IGN en Lambert 93 (EPSG:2154).

    Args:
        layer:    Nom de la couche WMS IGN.
        bbox_l93: (xmin, ymin, xmax, ymax) en Lambert 93.
        size_px:  Taille de la tuile en pixels (carré).

    Returns:
        Array RGB (H, W, 3) uint8, ou None si la tuile est vide / erreur.
    """
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "CRS": "EPSG:2154",
        "BBOX": ",".join(f"{v:.2f}" for v in bbox_l93),
        "WIDTH": size_px,
        "HEIGHT": size_px,
        "FORMAT": "image/png",
        "STYLES": "",
    }

    for attempt in range(retries):
        try:
            r = requests.get(WMS_URL, params=params, timeout=timeout)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            arr = np.array(img)
            # Rejette les tuiles blanches (hors couverture)
            if arr.mean() > 254.0 and arr.std() < 1.0:
                return None
            return arr
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return None


def check_layer_coverage(layer: str, bbox_l93: tuple) -> bool:
    """Vérifie qu'une couche renvoie bien des données sur cette zone."""
    arr = download_tile(layer, bbox_l93, size_px=128)
    return arr is not None


# ─────────────────────────────────────────────────────────
# Grille de tuiles
# ─────────────────────────────────────────────────────────

def build_grid(center_l93: tuple[float, float], size_km: float) -> list[dict]:
    """
    Construit une grille régulière de tuiles 256m×256m centrée sur center_l93.

    Returns:
        Liste de dicts { 'tile_id': str, 'bbox': (xmin,ymin,xmax,ymax) }
    """
    cx, cy = center_l93
    half_m = (size_km * 1000) / 2

    x_start = cx - half_m
    y_start = cy - half_m

    n_tiles = int((size_km * 1000) / PATCH_SIZE_M)
    tiles = []
    for row in range(n_tiles):
        for col in range(n_tiles):
            xmin = x_start + col * PATCH_SIZE_M
            ymin = y_start + row * PATCH_SIZE_M
            xmax = xmin + PATCH_SIZE_M
            ymax = ymin + PATCH_SIZE_M
            tile_id = f"r{row:03d}_c{col:03d}"
            tiles.append({"tile_id": tile_id, "bbox": (xmin, ymin, xmax, ymax)})

    return tiles


# ─────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────

def download_all(
    tiles: list[dict],
    layer: str,
    out_dir: Path,
    workers: int = 4,
    dry_run: bool = False,
) -> list[str]:
    """
    Télécharge toutes les tuiles et les sauvegarde en PNG.

    Returns:
        Liste des tile_id sauvegardés avec succès.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    skipped = 0

    def _download_one(tile: dict) -> Optional[str]:
        path = out_dir / f"{tile['tile_id']}.png"
        if path.exists():
            return tile["tile_id"]  # déjà téléchargé
        arr = download_tile(layer, tile["bbox"])
        if arr is None:
            return None
        Image.fromarray(arr).save(path)
        return tile["tile_id"]

    if dry_run:
        print(f"[DRY-RUN] {len(tiles)} tuiles à télécharger → {out_dir}")
        return [t["tile_id"] for t in tiles]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, t): t for t in tiles}
        with tqdm(total=len(tiles), desc=f"  {out_dir.name}/", unit="tuile") as bar:
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    saved.append(result)
                else:
                    skipped += 1
                bar.update(1)
                bar.set_postfix(ok=len(saved), skip=skipped)

    return saved


def compute_norm_stats(directory: Path, sample_n: int = 200) -> dict:
    """Calcule mean/std RGB sur un échantillon d'images du répertoire."""
    files = sorted(directory.glob("*.png"))
    if not files:
        return {}
    step = max(1, len(files) // sample_n)
    samples = files[::step]

    pixels = []
    for f in tqdm(samples, desc="  Calcul stats", unit="img", leave=False):
        arr = np.array(Image.open(f).convert("RGB")).reshape(-1, 3) / 255.0
        pixels.append(arr)

    all_px = np.concatenate(pixels, axis=0)
    mean = all_px.mean(axis=0).tolist()
    std  = all_px.std(axis=0).tolist()
    return {"mean": mean, "std": std, "n_samples": len(samples)}


def write_pairs_txt(saved_ids: list[str], out_dir: Path) -> None:
    """Écrit pairs.txt (un tile_id par ligne)."""
    with open(out_dir / "pairs.txt", "w") as f:
        for tid in sorted(saved_ids):
            f.write(tid + "\n")


def write_config(stats: dict, area_name: str, out_path: Path) -> None:
    """Génère un fichier de config adapté pour le modèle."""
    mean = stats.get("mean", [0.485, 0.456, 0.406])
    std  = stats.get("std",  [0.229, 0.224, 0.225])

    config_content = f'''import os
import numpy as np
from easydict import EasyDict as edict

# Config IGN Orthophoto — générée automatiquement par download_ign_orthophoto.py
# Zone     : {area_name}
# BEFORE   : {LAYER_BEFORE}
# AFTER    : {LAYER_AFTER}
# Résol.   : {PIXEL_SIZE_M} m/pixel  (identical to SECOND training data)
# Patch    : {PATCH_SIZE_PX}×{PATCH_SIZE_PX} px = {PATCH_SIZE_M:.0f}m×{PATCH_SIZE_M:.0f}m

C = edict()
config = C
cfg = C

C.seed = 3407
C.dataset_name = "ign_orthophoto_showcase"
C.root_folder = os.path.abspath(os.path.join(os.getcwd(), "showcase", "data"))
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

C.image_height = {PATCH_SIZE_PX}
C.image_width  = {PATCH_SIZE_PX}

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

# Statistiques calculées sur les images téléchargées
C.norm_mean = np.array({[round(v, 4) for v in mean]})
C.norm_std  = np.array({[round(v, 4) for v in std]})
C.use_cached_norm = False
C.use_single_normalization = True

C.use_color_jitter = False
C.jitter_hyper = 0.0

C.lr = 6e-5
C.weight_decay = 0.01
C.nepochs = 500

C.trial_name = f"{{C.dataset_name}}_{{C.backbone}}_{{C.decoder}}"

C.showcase = edict()
C.showcase.output_dir = os.path.abspath(os.path.join(os.getcwd(), "showcase", "results"))
C.showcase.save_predictions = True
C.showcase.save_overlays = True
C.showcase.confidence_threshold = 0.5
'''
    with open(out_path, "w") as f:
        f.write(config_content)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Télécharge des orthophotos IGN à deux dates pour RCDNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Zone
    area_group = p.add_mutually_exclusive_group()
    area_group.add_argument(
        "--area",
        choices=list(AREAS.keys()),
        default="ladefense",
        help="Zone prédéfinie (défaut: ladefense)",
    )
    area_group.add_argument(
        "--center-lon", type=float, metavar="LON",
        help="Longitude WGS84 du centre de la zone",
    )
    p.add_argument(
        "--center-lat", type=float, metavar="LAT",
        help="Latitude WGS84 du centre (avec --center-lon)",
    )
    p.add_argument(
        "--size-km", type=float, default=None,
        help="Taille de la zone en km (défaut selon la zone)",
    )

    # Layers
    p.add_argument("--layer-before", default=LAYER_BEFORE, help="Couche WMS BEFORE")
    p.add_argument("--layer-after",  default=LAYER_AFTER,  help="Couche WMS AFTER")

    # Output
    p.add_argument(
        "--output", default="showcase/data",
        help="Répertoire de sortie (défaut: showcase/data)",
    )

    # Execution
    p.add_argument("--workers", type=int, default=6, help="Threads parallèles (défaut: 6)")
    p.add_argument("--dry-run", action="store_true", help="Affiche la grille sans télécharger")
    p.add_argument("--no-stats", action="store_true", help="Skip le calcul des stats de normalisation")
    p.add_argument("--list-areas", action="store_true", help="Liste les zones disponibles et quitte")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_areas:
        print("\nZones disponibles :")
        for key, info in AREAS.items():
            print(f"  {key:12s}  {info['name']} — {info['description']}")
        return

    # ── Résolution de la zone ──────────────────────────────
    if args.center_lon is not None:
        if args.center_lat is None:
            print("Erreur : --center-lat requis avec --center-lon", file=sys.stderr)
            sys.exit(1)
        area_info = {
            "name": f"Custom ({args.center_lon:.3f}, {args.center_lat:.3f})",
            "lon": args.center_lon, "lat": args.center_lat,
            "size_km": args.size_km or 5.0,
        }
    else:
        area_info = AREAS[args.area].copy()
        if args.size_km:
            area_info["size_km"] = args.size_km

    size_km = area_info["size_km"]
    n_tiles_side = int(size_km * 1000 / PATCH_SIZE_M)
    n_tiles_total = n_tiles_side ** 2

    # ── Conversion WGS84 → Lambert 93 ─────────────────────
    t = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    cx_l93, cy_l93 = t.transform(area_info["lon"], area_info["lat"])

    # ── Résumé ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  IGN Orthophoto — RCDNet Showcase")
    print("═" * 60)
    print(f"  Zone      : {area_info['name']}")
    print(f"  Centre    : {area_info['lon']:.4f}°E  {area_info['lat']:.4f}°N")
    print(f"  Taille    : {size_km:.1f} km × {size_km:.1f} km")
    print(f"  BEFORE    : {args.layer_before}")
    print(f"  AFTER     : {args.layer_after}")
    print(f"  Résol.    : {PIXEL_SIZE_M} m/pixel  (même que SECOND dataset)")
    print(f"  Patch     : {PATCH_SIZE_PX}×{PATCH_SIZE_PX} px = {PATCH_SIZE_M:.0f}m×{PATCH_SIZE_M:.0f}m")
    print(f"  Tuiles    : {n_tiles_side}×{n_tiles_side} = {n_tiles_total}")
    print(f"  Workers   : {args.workers}")
    print("═" * 60)

    # ── Vérification de couverture ─────────────────────────
    if not args.dry_run:
        print("\nVérification de la couverture WMS...")
        test_bbox = (
            cx_l93 - 500, cy_l93 - 500,
            cx_l93 + 500, cy_l93 + 500,
        )
        for label, layer in [("BEFORE", args.layer_before), ("AFTER", args.layer_after)]:
            ok = check_layer_coverage(layer, test_bbox)
            status = "✓ données disponibles" if ok else "✗ AUCUNE DONNÉE — zone hors couverture"
            print(f"  {label}: {status}")
            if not ok:
                print(f"\n  Conseil : essayez --layer-{label.lower()} avec un autre millésime.")
                print(f"  Millésimes disponibles confirmés pour l'IDF :")
                print(f"    2021 : ORTHOIMAGERY.ORTHOPHOTOS2021")
                print(f"    2024 : ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2024")
                print(f"  Ou utilisez --list-areas pour choisir une autre zone.")
                sys.exit(1)

    # ── Construction de la grille ──────────────────────────
    tiles = build_grid((cx_l93, cy_l93), size_km)

    out_root = Path(args.output)
    dir_A = out_root / "A"
    dir_B = out_root / "B"

    if args.dry_run:
        print(f"\n[DRY-RUN] Grille : {len(tiles)} tuiles")
        print(f"  → {dir_A}/  (BEFORE)")
        print(f"  → {dir_B}/  (AFTER)")
        print(f"  → {out_root}/pairs.txt")
        print(f"  → configs/config_ign_showcase.py")
        return

    # ── Téléchargement BEFORE (A) ──────────────────────────
    print(f"\nTéléchargement BEFORE ({args.layer_before})...")
    saved_A = download_all(tiles, args.layer_before, dir_A, workers=args.workers)
    print(f"  {len(saved_A)}/{len(tiles)} tuiles sauvegardées")

    # ── Téléchargement AFTER (B) ───────────────────────────
    print(f"\nTéléchargement AFTER ({args.layer_after})...")
    saved_B = download_all(tiles, args.layer_after, dir_B, workers=args.workers)
    print(f"  {len(saved_B)}/{len(tiles)} tuiles sauvegardées")

    # ── Paires valides (présentes dans A ET B) ─────────────
    valid_ids = sorted(set(saved_A) & set(saved_B))
    print(f"\nPaires valides (A ∩ B) : {len(valid_ids)}")

    write_pairs_txt(valid_ids, out_root)
    print(f"  → {out_root}/pairs.txt")

    # ── Statistiques de normalisation ─────────────────────
    norm_stats: dict = {}
    if not args.no_stats and valid_ids:
        print("\nCalcul des statistiques de normalisation...")
        stats_A = compute_norm_stats(dir_A)
        stats_B = compute_norm_stats(dir_B)
        # Moyenne A + B (comme dans les configs SECOND/CNAM)
        mean_ab = [(a + b) / 2 for a, b in zip(stats_A["mean"], stats_B["mean"])]
        std_ab  = [(a + b) / 2 for a, b in zip(stats_A["std"],  stats_B["std"])]
        norm_stats = {"mean": mean_ab, "std": std_ab}
        print(f"  mean (A+B)/2 : {[round(v, 4) for v in mean_ab]}")
        print(f"  std  (A+B)/2 : {[round(v, 4) for v in std_ab]}")

        # Sauvegarde JSON
        stats_path = out_root / "norm_stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {"before": stats_A, "after": stats_B, "combined": norm_stats},
                f, indent=2,
            )
        print(f"  → {stats_path}")

    # ── Génération de la config ────────────────────────────
    config_path = Path("configs/config_ign_showcase.py")
    write_config(norm_stats, area_info["name"], config_path)
    print(f"\nConfig générée : {config_path}")

    # ── Résumé final ───────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  ✓ {len(valid_ids)} paires prêtes dans {out_root}/")
    print(f"  Résolution : {PIXEL_SIZE_M} m/pixel  (SECOND training resolution)")
    print(f"  BEFORE : {args.layer_before}")
    print(f"  AFTER  : {args.layer_after}")
    print()
    print("  Inference :")
    print("    python3 showcase/scripts/03_run_inference.py \\")
    print("        --checkpoint weights/SECOND-model.safetensors \\")
    print("        --config configs.config_ign_showcase \\")
    print("        --device cuda")
    print("═" * 60)


if __name__ == "__main__":
    main()
