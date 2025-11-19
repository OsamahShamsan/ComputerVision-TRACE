# TRACE Project Guide

## Workspace Structure

**Principle**: Method isolation - each method is completely self-contained.

```
TRACE/
├── configs/
│   └── data_config.json          # Data processing config (shared)
├── data/
│   ├── raw/                      # Raw datasets (shared)
│   ├── processed/                # Processed data (shared)
│   └── results/                  # Final comparison results (shared)
├── notebooks/
│   └── 01_data_inspection_and_setup.ipynb  # Data processing (shared)
├── src/
│   └── data/                     # Data processing utilities (shared)
├── methods/                       # Method-specific work (ISOLATED)
│   ├── sift/
│   │   ├── notebooks/            # SIFT notebooks only
│   │   ├── src/                  # SIFT code only
│   │   ├── configs/              # SIFT configs only
│   │   ├── outputs/              # SIFT outputs only
│   │   └── results/              # SIFT results only
│   ├── orb/                      # ORB isolated
│   └── deep_learning/            # DL isolated
└── venv/                         # Virtual environment
```

**Rules**: Root `notebooks/` and `src/data/` are for shared data processing only. All method-specific work goes in `methods/method_name/`. No mixing between methods.

## Method Workflow

**Order**: SIFT → ORB → Deep Learning (if needed)

Each method is isolated with its own notebooks, source code, configs, outputs, and results. Current focus: SIFT feature-based copy-move detection.

## Dataset Guide

### FAU Dataset Categories

**Essential** (start here, ~5-8GB):
- `nul`: Direct copy-paste
- `rot`: Rotated copies
- `scale`: Scaled copies

**Recommended** (add for robustness, ~10-15GB total):
- `cmb_easy1`: Combined effects
- `lnoise`: Gaussian noise
- `jpeg`: JPEG compression

**Skip**: `orig`, `*_sd` patterns, `scale_down` (not needed for training)

Configure in `configs/data_config.json`:
```json
{
  "dataset_categories": {
    "include": [],
    "exclude_patterns": ["_sd", "orig", "scale_down", "orig_jpeg"]
  }
}
```

### Dataset Recommendations

**FAU Dataset**: 48 base images. Sufficient for feature-based methods (SIFT, SURF, ORB) and transfer learning. Limited for deep learning training from scratch.

**For Deep Learning**:
- **Transfer learning**: FAU + CoMoFoD (200 pairs) or MICC-F220 (110 pairs) = ~250-300 unique images
- **Training from scratch**: CASIA v2.0 (5,000+ tampered images) or combine multiple datasets

**Alternative Datasets**:
- **CoMoFoD**: 200 forged + 200 authentic images. [http://www.vcl.fer.hr/comofod/](http://www.vcl.fer.hr/comofod/)
- **CASIA v2.0**: 12,614 images. [http://forensics.idealtest.org/](http://forensics.idealtest.org/)
- **MICC-F220/F600**: 220-600 images. [http://lci.micc.unifi.it/labd/2014/01/micc-f220-and-micc-f600-datasets/](http://lci.micc.unifi.it/labd/2014/01/micc-f220-and-micc-f600-datasets/)

**Strategy**: Start with FAU for feature-based methods. Add CoMoFoD or MICC for deep learning. Use CASIA v2.0 for large-scale training. Always apply data augmentation (geometric, photometric, noise, compression).


