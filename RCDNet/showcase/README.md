# Cas d'usage IGN — Village Olympique, Saint-Denis (2021 → 2024)

Détection automatique de changements sémantiques sur les orthophotos IGN Géoplateforme (0,5 m/pixel), appliquée à la construction du Village Olympique de Paris 2024.

**Résultats :** 1,50 km² de changements détectés · 529 paires de tuiles · 34,67 km² couverts

→ Rapport complet : [`docs/RAPPORT_IGN_VILLAGE_OLYMPIQUE.md`](../docs/RAPPORT_IGN_VILLAGE_OLYMPIQUE.md)

---

## Reproduction en 3 étapes

### Étape 1 — Télécharger les orthophotos IGN

```bash
# Téléchargement via le service WMS public IGN Géoplateforme
# Zone : Saint-Denis / Village Olympique (bbox en Lambert-93)
python download_ign_orthophoto.py \
    --layer ORTHOPHOTOS2021 \
    --bbox 651000 6867000 660000 6874000 \
    --output showcase/data/A \
    --resolution 0.5

python download_ign_orthophoto.py \
    --layer ORTHO-EXPRESS.2024 \
    --bbox 651000 6867000 660000 6874000 \
    --output showcase/data/B \
    --resolution 0.5
```

Les images sont automatiquement découpées en tuiles de 512 × 512 pixels.

### Étape 2 — Lancer l'inférence

```bash
# Inférence sur les paires IGN
python showcase/scripts/03_run_inference_lowmem.py \
    --config configs.config_ign_showcase \
    --checkpoint weights/SECOND-model.safetensors \
    --input_a showcase/data/A \
    --input_b showcase/data/B \
    --output showcase/results
```

Le script traite les tuiles par lots et génère un masque de changement par classe sémantique pour chaque paire.

### Étape 3 — Visualiser les résultats

```bash
# Générer les visualisations (grilles before/after/overlay)
python visualize_changes.py \
    --results showcase/results \
    --images_a showcase/data/A \
    --images_b showcase/data/B \
    --output showcase/visualizations \
    --top_k 5
```

Les visualisations sont enregistrées dans `showcase/visualizations/` :
- `all_classes_best.png` — grille multi-classes des meilleures détections
- `class_breakdown.png` — répartition surfacique par classe
- `<classe>_grid.png` — grille par classe individuelle
- `summary_report.txt` — statistiques quantitatives complètes

---

## Résultats obtenus

| Classe | Surface modifiée | % de la zone |
|---|---|---|
| Sol non végétalisé | 0,720 km² | 2,08 % |
| Bâtiment | 0,405 km² | 1,17 % |
| Végétation basse | 0,297 km² | 0,86 % |
| Eau | 0,038 km² | 0,11 % |
| Arbre | 0,028 km² | 0,08 % |
| Terrain de jeux | 0,016 km² | 0,05 % |

---

## Configuration utilisée

Fichier de configuration : [`configs/config_ign_showcase.py`](../configs/config_ign_showcase.py)

- **Classes :** 6 classes sémantiques (taxonomie SECOND)
- **Taille des tuiles :** 512 × 512 pixels
- **Normalisation :** statistiques SECOND (RGB)
- **Modèle :** backbone VMamba-Small + décodeur Mamba
- **Poids :** `weights/SECOND-model.safetensors` (fournis par les auteurs originaux)

---

## Pipeline Sentinel-2 (pour référence)

Les scripts `scripts/01_extract_sentinel.py` et `scripts/02_create_pairs.py` documentent le pipeline initial développé sur Sentinel-2 (10 m/pixel). Ce workflow a servi de point de départ avant l'adaptation aux orthophotos IGN à 0,5 m/pixel.

---

## Dépannage

Voir [`docs/IGN_INFERENCE_TROUBLESHOOTING.md`](../docs/IGN_INFERENCE_TROUBLESHOOTING.md) pour les problèmes courants :
- Dépassement mémoire GPU → résolu par le chunked attention dans le décodeur Mamba (voir `models/decoders/attention.py`)
- Décalage géométrique entre millésimes → alignement assuré par le WMS IGN (même bbox Lambert-93)
