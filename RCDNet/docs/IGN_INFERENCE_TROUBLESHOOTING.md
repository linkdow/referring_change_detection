# Guide de dépannage — Inférence IGN Orthophoto avec RCDNet

Ce document recense tous les problèmes rencontrés lors de la mise en place de l'inférence RCDNet sur
des orthophotos IGN, dans l'ordre chronologique où ils ont été identifiés. Chaque section décrit le
symptôme visible, la cause racine, et la correction appliquée.

**Contexte matériel et logiciel**

| Paramètre | Valeur |
|---|---|
| GPU | GTX 1650 — 4 Go VRAM |
| OS | Ubuntu/WSL2 (Windows 11) |
| CUDA | 12.6 |
| Python | 3.10 (conda) |
| Modèle | RCDNet, backbone VMamba (`sigma_small`) |
| Données d'entraînement | SECOND dataset (~0.5 m/pixel) |
| Données d'inférence | IGN Orthophoto 2021/2024 (0.5 m/pixel) |
| Pourquoi IGN | Remplace Sentinel-2 (10 m/pixel) pour correspondre à la résolution d'entraînement |

---

## Table des matières

1. [selective\_scan\_cuda\_core introuvable](#problème-1--selective_scan_cuda_core-introuvable)
2. [libcuda.so introuvable sous WSL2](#problème-2--libcudaso-introuvable-sous-wsl2)
3. [Crash au chargement du répertoire gt/](#problème-3--crash-au-chargement-du-répertoire-gt)
4. [Mauvais chemin root\_folder via os.getcwd()](#problème-4--mauvais-chemin-root_folder-via-osgetcwd)
5. [Out of Memory — matrice d'attention trop grande](#problème-5--out-of-memory--matrice-dattention-trop-grande)
6. [caption\_embedding passé comme label (argument positionnel)](#problème-6--caption_embedding-passé-comme-label-argument-positionnel)
7. [Mauvais fichier de config utilisé](#problème-7--mauvais-fichier-de-config-utilisé)
8. [0 détection — normalisation hors distribution](#problème-8--0-détection--normalisation-hors-distribution)
9. [0 détection — NaN causés par AMP en FP16](#problème-9--0-détection--nan-causés-par-amp-en-fp16)
10. [Commande finale fonctionnelle](#commande-finale-fonctionnelle)
11. [Tableau récapitulatif](#tableau-récapitulatif)

---

## Problème 1 : `selective_scan_cuda_core` introuvable

### Symptôme

```
ModuleNotFoundError: No module named 'selective_scan_cuda_core'
```

### Cause

L'extension CUDA du backbone VMamba est compilée dans
`models/encoders/selective_scan/`. Ce répertoire n'est pas dans le chemin
de recherche de modules Python (`sys.path`), donc Python ne trouve pas le
fichier `.so` compilé.

### Correction

Copier le fichier `.so` compilé dans le répertoire `site-packages` de
l'environnement conda :

```bash
cp models/encoders/selective_scan/selective_scan_cuda_core.cpython-310-x86_64-linux-gnu.so \
   $CONDA_PREFIX/lib/python3.10/site-packages/
```

Le fichier `.so` peut avoir un nom légèrement différent selon l'architecture
et la version de Python. Adapter le nom si nécessaire.

---

## Problème 2 : `libcuda.so` introuvable sous WSL2

### Symptôme

```
Could not load library libcudnn_cnn_infer.so.8.
Error: libcuda.so: cannot open shared object file: No such file or directory
```

### Cause

Sous WSL2, `libcuda.so` se trouve dans `/usr/lib/wsl/lib/` et non dans les
chemins CUDA standard. Le linker dynamique ne le trouve pas par défaut.

### Correction

Définir `LD_LIBRARY_PATH` avant chaque lancement. Cette variable doit inclure
trois chemins dans cet ordre :

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

Ajouter cette ligne au `.bashrc` ou au `.zshrc` pour ne pas avoir à la
répéter, ou l'inclure systématiquement dans les scripts de lancement.

---

## Problème 3 : Crash au chargement du répertoire `gt/`

### Symptôme

```
FileNotFoundError: Could not find split lists
```
ou crash dans `__getitem__` en tentant d'ouvrir le fichier de vérité terrain.

### Cause

`dataloader/changeDataset.py` ouvre systématiquement le fichier de vérité
terrain (`gt/`) dans `__getitem__`, même en mode inférence où aucune
annotation n'existe. Le code supposait que les images gt étaient toujours
présentes.

### Correction

Modifier `dataloader/changeDataset.py` pour retourner un masque nul quand
le fichier gt est absent :

```python
if os.path.exists(gt_path):
    gt = self._open_image(gt_path, "L", dtype=np.uint8)
else:
    gt = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
```

Cette modification est rétrocompatible : en mode entraînement et évaluation,
les fichiers gt existent et le comportement reste inchangé.

---

## Problème 4 : Mauvais chemin `root_folder` via `os.getcwd()`

### Symptôme

Le dataset est cherché dans un répertoire incorrect, par exemple
`models/encoders/selective_scan/showcase/data` au lieu de `showcase/data`.

### Cause

Le fichier de config généré automatiquement utilisait :

```python
C.root_folder = os.path.join(os.getcwd(), "showcase", "data")
```

Si le répertoire de travail du shell avait changé entre la génération de la
config et l'exécution (par exemple après un `cd` implicite), le chemin
résultait erroné.

### Correction

Construire le chemin absolu depuis l'emplacement du fichier de config
lui-même, avec `__file__` :

```python
C.root_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "showcase", "data"
)
```

Cette approche est insensible au répertoire de travail courant du shell.
C'est la méthode correcte pour tout chemin ancré dans le projet.

---

## Problème 5 : Out of Memory — matrice d'attention trop grande

### Symptôme

```
torch.cuda.OutOfMemoryError: Tried to allocate 8.00 GiB
(GPU 0; 4.00 GiB total capacity; 3.58 GiB already allocated)
```

### Cause

Le module `CrossAttention` dans `models/decoders/attention.py` calcule la
matrice d'attention complète en une seule opération :

```python
sim = einsum('b i d, b j d -> b i j', q, k)
```

Pour une entrée 512×512, la feature map du décodeur est 128×128 = 16 384
tokens. La matrice d'attention résultante a la forme
`[8 têtes, 16 384, 16 384]`, ce qui requiert **8 Go en FP32** — impossible
sur un GPU de 4 Go.

### Correction

Implémenter une attention par blocs (*chunked attention*) dans
`models/decoders/attention.py` : traiter les requêtes par blocs de 512
tokens pour limiter la mémoire de pointe à ~120 Mo :

```python
def forward(self, x, context=None, mask=None, chunk_size=512):
    ...
    out = torch.zeros_like(q)
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        sim = einsum('b i d, b j d -> b i j', q[:, start:end], k) * self.scale
        attn = sim.softmax(dim=-1)
        out[:, start:end] = einsum('b i j, b j d -> b i d', attn, v)
    ...
```

Note : cette correction rend AMP dangereux (voir Problème 9).

---

## Problème 6 : `caption_embedding` passé comme `label` (argument positionnel)

### Symptôme

```
TypeError: 'NoneType' object is not callable
  File "builder.py", line 177, in forward
    loss = self.criterion(out, label.float())
```

### Cause

La signature de `builder.py` est :

```python
def forward(self, rgb, modal_x, label=None, captions=None):
```

Le script d'inférence appelait le modèle ainsi :

```python
output = model(img_a, img_b, caption_embedding)
```

`caption_embedding` était donc reçu comme troisième argument positionnel,
c'est-à-dire comme `label`. Puisque `label is not None`, le modèle tentait
de calculer la perte avec `self.criterion` — qui vaut `None` en mode
inférence.

### Correction

Utiliser l'argument nommé dans `showcase/scripts/03_run_inference_lowmem.py` :

```python
output = model(img_a, img_b, captions=caption_embedding)
```

Règle générale : toujours passer les embeddings textuels par argument nommé
pour éviter toute confusion avec les masques de supervision.

---

## Problème 7 : Mauvais fichier de config utilisé

### Symptôme

0 détection sur les 529 patches, sans autre message d'erreur.

### Cause

La valeur par défaut du flag `--config` dans
`showcase/scripts/03_run_inference_lowmem.py` est
`configs.config_sentinel_showcase`. Sans l'option `--config`, le script
chargeait silencieusement la config Sentinel-2 avec ses statistiques de
normalisation ImageNet au lieu des statistiques SECOND.

### Correction

Toujours spécifier explicitement `--config configs.config_ign_showcase` :

```bash
python3 showcase/scripts/03_run_inference_lowmem.py \
    --checkpoint weights/SECOND-model.safetensors \
    --config configs.config_ign_showcase \
    --device cuda \
    --no-amp
```

---

## Problème 8 : 0 détection — normalisation hors distribution

### Symptôme

0 détection même avec la bonne config. Les logits de sortie sont dans la
plage `[-11.9, -1.7]`, sigmoid maximum = 0.16. Le modèle n'active aucune
classe de changement.

### Cause

La config avait été générée automatiquement avec des statistiques calculées
sur les images IGN elles-mêmes :

| Statistique | Stats IGN (calculées) | Stats SECOND (entraînement) |
|---|---|---|
| mean | [0.427, 0.438, 0.421] | [0.439, 0.447, 0.459] |
| std | [0.168, 0.148, 0.138] | [0.193, 0.183, 0.189] |

Les écarts-types IGN (~0.14–0.17) sont sensiblement plus petits que ceux du
dataset d'entraînement (~0.19). Diviser par ces petits écarts-types produit
des valeurs normalisées avec `std ~ 1.02`, alors que le modèle a été entraîné
avec `std ~ 0.82`. Les entrées sont hors distribution et le modèle répond
systématiquement "pas de changement".

### Correction

Utiliser les statistiques de normalisation du dataset d'entraînement
(SECOND) dans `configs/config_ign_showcase.py` :

```python
# Normalisation SECOND — à utiliser pour l'inférence IGN
C.norm_mean = np.array([0.439, 0.447, 0.459])
C.norm_std  = np.array([0.193, 0.183, 0.189])

# Ne pas utiliser les stats calculées sur les images IGN :
# mean=[0.427, 0.438, 0.421], std=[0.168, 0.148, 0.138]
```

**Règle absolue :** à l'inférence, utiliser toujours les statistiques de
normalisation du dataset d'entraînement, jamais celles calculées sur les
images de test.

---

## Problème 9 : 0 détection — NaN causés par AMP en FP16

### Symptôme

0 détection même avec la normalisation correcte. Toutes les valeurs de
sortie sont NaN.

Vérification des logits :
- **Avec AMP** : `max=nan, px>0.5=0, NaN=262144` — échec total
- **Sans AMP** : `max=0.997, px>0.5=2633, NaN=0` — résultat correct

### Cause

La correction de l'attention par blocs (Problème 5) initialise un tenseur
d'accumulation avec `torch.zeros_like(q)`. En mode AMP, `q` est en FP16,
donc le tenseur d'accumulation est également FP16. FP16 a une plage
dynamique limitée (~65 000 au maximum).

L'accumulation de produits scalaires sur 16 384 positions en 32 blocs
provoque un **dépassement de capacité (overflow) → inf → NaN** dans les
activations d'attention.

### Correction

Toujours utiliser le flag `--no-amp` :

```bash
python3 showcase/scripts/03_run_inference_lowmem.py \
    --checkpoint weights/SECOND-model.safetensors \
    --config configs.config_ign_showcase \
    --device cuda \
    --no-amp
```

AMP n'est de toute façon pas nécessaire ici : la correction par blocs
(Problème 5) a déjà résolu le problème mémoire. FP32 tient dans 4 Go avec
l'attention par blocs.

---

## Commande finale fonctionnelle

```bash
# Étape 1 : configurer les bibliothèques CUDA (WSL2)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# Étape 2 : lancer l'inférence
python3 showcase/scripts/03_run_inference_lowmem.py \
    --checkpoint weights/SECOND-model.safetensors \
    --config configs.config_ign_showcase \
    --device cuda \
    --no-amp
```

**Performances attendues :**
- ~9.6 s/patch
- ~1 h 25 min pour 529 patches
- Détections actives dans les classes : `Building` et `Non-vegetated Ground Surface`

---

## Tableau récapitulatif

| # | Problème | Cause racine | Correction |
|---|---|---|---|
| 1 | `selective_scan_cuda_core` introuvable | Fichier `.so` absent du `sys.path` | Copier dans `site-packages` |
| 2 | `libcuda.so` introuvable | Chemin WSL2 non standard | Ajouter `/usr/lib/wsl/lib` à `LD_LIBRARY_PATH` |
| 3 | Crash à l'ouverture de `gt/` | Le dataloader charge toujours le gt | Retourner un masque nul si le fichier est absent |
| 4 | Mauvais chemin `root_folder` | `os.getcwd()` dans la config | Utiliser `__file__` pour un chemin absolu |
| 5 | OOM — 8 Go pour la matrice d'attention | Attention O(N²) sur GPU 4 Go | Attention par blocs de 512 tokens (~120 Mo) |
| 6 | `NoneType` not callable | `caption_embedding` passé comme argument positionnel | Utiliser `captions=` comme argument nommé |
| 7 | 0 détection — mauvaise config | Flag `--config` absent, défaut Sentinel-2 | Toujours spécifier `--config configs.config_ign_showcase` |
| 8 | 0 détection — normalisation erronée | Stats IGN (std~0.14) vs SECOND (std~0.19) → hors distribution | Utiliser les stats SECOND à l'inférence |
| 9 | 0 détection — NaN | AMP + FP16 overflow dans l'attention par blocs | Utiliser `--no-amp` |
