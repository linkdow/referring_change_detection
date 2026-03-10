# Changelog — IGN Orthophoto Showcase

Résumé de tous les changements apportés au projet RCDNet depuis le clone initial,
dans le cadre de l'adaptation pour l'inférence sur images IGN Orthophoto.

**Zone** : Saint-Denis / Village Olympique
**Période** : 2021 → 2024 (construction du Village Olympique)
**Résolution** : 0.5 m/pixel (IGN Géoplateforme WMS)
**Modèle** : RCDNet — VMamba backbone (sigma_small) + MambaDecoder + CLIP text encoder
**GPU** : GTX 1650 (4 Go VRAM)

---

## Modifications de fichiers existants

### `models/decoders/attention.py` — Attention par blocs (chunked attention)

**Problème** : La matrice d'attention `CrossAttention` allouait un tenseur `[8, 16384, 16384]`
en FP32 = **8 Go** → OOM sur un GPU 4 Go VRAM.

**Fix** : Remplacement de l'attention dense O(N²) par une boucle traitant les queries
par blocs de 512 tokens → pic mémoire ~128 Mo.

```python
# Avant
sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
attn = sim.softmax(dim=-1)
out = einsum('b i j, b j d -> b i d', attn, v)

# Après
q_len = q.shape[1]
out = torch.zeros_like(q)
for start in range(0, q_len, chunk_size):   # chunk_size=512
    end = min(start + chunk_size, q_len)
    sim = einsum('b i d, b j d -> b i j', q[:, start:end], k) * self.scale
    attn = sim.softmax(dim=-1)
    out[:, start:end] = einsum('b i j, b j d -> b i d', attn, v)
```

> ⚠️ La chunked attention est incompatible avec AMP (FP16) : l'accumulation sur 32 blocs
> produit des NaN. Toujours utiliser `--no-amp` à l'inférence.

---

### `dataloader/changeDataset.py` — Mode inférence sans ground truth

**Problème** : Le dataloader crashait si le dossier `gt/` n'existait pas
(normal en inférence, pas de labels disponibles).

**Fix** :

```python
# Avant
gt = self._open_image(gt_path, "L", dtype=np.uint8)

# Après
if os.path.exists(gt_path):
    gt = self._open_image(gt_path, "L", dtype=np.uint8)
else:
    # No ground truth available (inference-only mode)
    gt = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
```

---

### `.gitignore`

Ajout de `credentials.txt`.

---

## Nouveaux fichiers créés

### Scripts de téléchargement

| Fichier | Rôle |
|---------|------|
| `download_ign_orthophoto.py` | Télécharge les orthophotos IGN 2021 et 2024 via WMS Géoplateforme (0.5 m/px), construit une grille de 529 tuiles 512×512 px sur Saint-Denis, génère `pairs.txt` et `configs/config_ign_showcase.py` |
| `download_sentinel_2023.py` | (non utilisé) Ancien script Sentinel-2 |
| `download_sentinel_idf.py` | (non utilisé) Ancien script Sentinel-2 IDF |

**Paramètres clés de `download_ign_orthophoto.py`** :
- `LAYER_BEFORE = "ORTHOIMAGERY.ORTHOPHOTOS2021"`
- `LAYER_AFTER  = "ORTHOIMAGERY.ORTHO-EXPRESS.2024"`
- `PIXEL_SIZE_M = 0.5` — identique à la résolution du dataset SECOND (entraînement)
- `PATCH_SIZE_PX = 512`
- Grille : 23×23 tuiles = 529 paires, couvrant ~7.5 km × 7.5 km

---

### Configuration

**`configs/config_ign_showcase.py`** — deux points critiques :

```python
# 1. Chemin absolu basé sur __file__ (pas os.getcwd() qui change selon le CWD)
C.root_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "showcase", "data"
)

# 2. Normalisation du dataset SECOND (entraînement), PAS les stats IGN calculées.
# Les stats IGN (std~0.14-0.17) produisent des valeurs hors distribution → 0 détections.
C.norm_mean = np.array([0.439, 0.447, 0.459])  # SECOND dataset
C.norm_std  = np.array([0.193, 0.183, 0.189])  # SECOND dataset
```

---

### Scripts d'inférence et visualisation

| Fichier | Rôle |
|---------|------|
| `showcase/scripts/03_run_inference_lowmem.py` | Inférence optimisée mémoire : AMP optionnel (`--no-amp`), CLIP text encoder, traitement classe par classe pour limiter le pic VRAM |
| `showcase/visualize_changes.py` | Génère des grilles Before/After/Overlay (rouge) pour chaque classe de changement, sauvegarde en PNG dans `showcase/visualizations/` |

**Commande d'inférence complète (à utiliser systématiquement)** :
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

python3 showcase/scripts/03_run_inference_lowmem.py \
    --checkpoint weights/SECOND-model.safetensors \
    --config configs.config_ign_showcase \
    --device cuda \
    --no-amp
```

---

### Documentation

| Fichier | Rôle |
|---------|------|
| `docs/IGN_INFERENCE_TROUBLESHOOTING.md` | Guide complet (FR) des 9 problèmes rencontrés et leurs solutions |
| `docs/CHANGELOG_IGN_SHOWCASE.md` | Ce fichier |
| `CUDA_SETUP.md` | Setup CUDA pour WSL2 |
| `QUICK_START.md` | Guide de démarrage rapide |

---

## Résultats obtenus

**529 paires** traitées sur la zone Saint-Denis / Village Olympique.
**1 450 masques** de changement détectés, **1.50 km²** de surface modifiée sur 34.67 km² (4.3 %).

| Classe | Patches détectés | % des patches | Surface (km²) |
|--------|-----------------|---------------|---------------|
| Non-vegetated Ground Surface | 479 | 90.5 % | 0.72 |
| Building | 322 | 60.9 % | 0.40 |
| Low Vegetation | 324 | 61.2 % | 0.30 |
| Water | 132 | 25.0 % | 0.04 |
| Tree | 137 | 25.9 % | 0.03 |
| Playground | 56 | 10.6 % | 0.02 |

---

## Les 9 bugs résolus

| # | Symptôme | Cause | Solution |
|---|----------|-------|----------|
| 1 | `ModuleNotFoundError: selective_scan_cuda_core` | Extension CUDA compilée mais absente du path Python | Copie du `.so` dans `$CONDA_PREFIX/lib/python3.10/site-packages/` |
| 2 | `OSError: libcuda.so not found` | WSL2 place `libcuda.so` dans `/usr/lib/wsl/lib/` hors des paths standard | `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:...` avant chaque lancement |
| 3 | `FileNotFoundError` sur `gt/` | Le dataloader ouvre toujours le fichier gt même à l'inférence | Fallback `np.zeros` dans `changeDataset.py` si le fichier est absent |
| 4 | `FileNotFoundError` sur les images A/B | `root_folder` résolu depuis `os.getcwd()` qui change selon le répertoire courant | Utiliser `os.path.dirname(os.path.abspath(__file__))` dans le config |
| 5 | `torch.cuda.OutOfMemoryError` (8 Go alloués) | Matrice d'attention dense `[8, 16384, 16384]` en FP32 | Chunked attention par blocs de 512 tokens dans `attention.py` |
| 6 | `TypeError: 'NoneType' object is not callable` | `caption_embedding` passé en 3ème argument positionnel → interprété comme `label` | Utiliser l'argument nommé `captions=caption_embedding` |
| 7 | 0 détections (toutes les classes) | `--config` non spécifié → config Sentinel utilisée par défaut | Toujours passer `--config configs.config_ign_showcase` |
| 8 | 0 détections (toutes les classes) | Stats IGN calculées (std~0.14) étirent les valeurs à std~1.02 vs ~0.82 attendu → hors distribution | Utiliser les stats du dataset SECOND : `mean=[0.439, 0.447, 0.459]`, `std=[0.193, 0.183, 0.189]` |
| 9 | 0 détections / NaN dans les masques | AMP (FP16) + accumulation `torch.zeros_like(q)` sur 32 chunks → overflow → NaN → sigmoid(NaN) = 0 | `--no-amp` obligatoire ; vérifiable avec `probs.max()` qui doit être ~0.997 et non `nan` |
