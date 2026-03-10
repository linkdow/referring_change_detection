# Referring Change Detection in Remote Sensing Imagery (WACV 2026)

<p align="center">
  <a href="https://yilmazkorkmaz1.github.io/RCD/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2ea44f"></a>
  <a href="https://arxiv.org/pdf/2512.11719"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.11719-b31b1b"></a>
  <a href="https://huggingface.co/yilmazkorkmaz/RCDGen"><img alt="HuggingFace" src="https://img.shields.io/badge/HuggingFace-RCDGen-yellow"></a>
</p>

> **Fork du projet original** ([yilmazkorkmaz1/referring_change_detection](https://github.com/yilmazkorkmaz1/referring_change_detection)), enrichi d'un cas d'usage sur données **IGN Géoplateforme** : détection de changements sémantiques sur le Village Olympique de Saint-Denis (2021→2024), 1,50 km² détectés sur 34,67 km² via orthophotos 0,5 m/pixel.
>
> Contributions apportées : adaptation au pipeline IGN WMS · fix chunked attention (décodeur Mamba) · pipeline Sentinel-2 · documentation en français
>
> → Détails : [`RCDNet/README.md`](RCDNet/README.md) · Rapport : [`RCDNet/docs/RAPPORT_IGN_VILLAGE_OLYMPIQUE.md`](RCDNet/docs/RAPPORT_IGN_VILLAGE_OLYMPIQUE.md)

---

## Liens

- **Page projet** : [`yilmazkorkmaz1.github.io/RCD`](https://yilmazkorkmaz1.github.io/RCD/)
- **Article (arXiv)** : [`arxiv.org/pdf/2512.11719`](https://arxiv.org/pdf/2512.11719)

## Statut (dépôt original)

- ✅ **RCDGen et RCDNet publiés**
- 🤗 **Poids pré-entraînés RCDGen :** [`yilmazkorkmaz/RCDGen`](https://huggingface.co/yilmazkorkmaz/RCDGen)
- 🤗 **Poids pré-entraînés RCDNet :** [Google Drive](https://drive.google.com/drive/folders/1foXpLPz3jtaQN7l6UdlDFVSgakgm6RXP?usp=share_link) (`SECOND-model.safetensors`, `CNAM-CD-model.safetensors`)
- 🤗 **Jeux de données synthétiques :**
  - **SECOND Synthetic :** [`yilmazkorkmaz/Synthetic_RCD_1`](https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1)
  - **CNAM-CD Synthetic :** [`yilmazkorkmaz/Synthetic_RCD_2`](https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_2)

## RCDGen

Voir [`RCDGen/README.md`](RCDGen/README.md) pour la préparation des données, l'entraînement et l'inférence.

## RCDNet

Voir [`RCDNet/README.md`](RCDNet/README.md) pour la préparation des données, l'entraînement et l'évaluation.

## Utilisation et citation
Ce code est librement utilisable à des fins de recherche avec citation appropriée :

```bibtex
@article{korkmaz2025referring,
  title={Referring Change Detection in Remote Sensing Imagery},
  author={Korkmaz, Yilmaz and Paranjape, Jay N and de Melo, Celso M and Patel, Vishal M},
  journal={arXiv preprint arXiv:2512.11719},
  year={2025}
}
