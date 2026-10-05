# SeFAR + Compositional Pseudo-Label Decomposition (CPLD)

Extension of [SeFAR](https://github.com/KyleHuang9/SeFAR) (AAAI 2025) that
addresses **Gap 1**: SeFAR treats fine-grained action labels as flat, opaque
class IDs. CPLD decomposes them into compositional attribute slots and
generates per-attribute pseudo-labels independently gated by attribute
confidence — providing partial supervision even when the full class label is
too uncertain to trust.

---

## What's new

| File | Description |
|------|-------------|
| `timesformer/models/compositional_heads.py` | `CompositionalAttributeHeads` + `multi_granular_vit_comp` registered model |
| `timesformer/utils/label_parser.py` | `FineGymLabelParser`: maps class name strings → 5 attribute IDs via keyword rules |
| `timesformer/config/defaults.py` | Added `_C.COMPOSITIONAL` config block (7 keys) |
| `tools/train_net_compositional.py` | Full training loop with compositional supervised + SSL losses |
| `run_net_compositional.py` | Entry point (mirrors `run_net_emamix.py`) |
| `train_compositional.sh` | One-liner launcher for all experiment variants |
| `tools/run_baseline.sh` | Runs the original SeFAR-B for comparison numbers |
| `tools/compare_results.py` | Parses logs and generates LaTeX results tables |
| `tools/plot_figures.py` | Generates all paper figures (4 PDF figures) |
| `tools/generate_gym288_names.py` | Generates `gym288_class_names.json`; use `--official` with real FineGym annotations |
| `dataset/finegym/gym99_class_names.json` | 99 FineGym element names |
| `dataset/finegym/gym288_class_names.json` | 288 FineGym element names (programmatic taxonomy) |
| `dataset/finediving/finediving_class_names.json` | 52 FineDiving action names |
| `configs/TimeSformer_base_ssl_compositional.yaml` | Gym99 (base model) |
| `configs/TimeSformer_base_ssl_compositional_gym288.yaml` | Gym288 (base model) |
| `configs/TimeSformer_base_ssl_compositional_diving.yaml` | FineDiving (base model) |
| `configs/TimeSformer_small_ssl_compositional.yaml` | Gym99 (small model, 31M params) |
| `tests/test_label_parser.py` | 40 unit tests for the label parser |
| `paper/` | Complete AAAI 2026 paper (LaTeX) |

---

## Quick start

```bash
conda activate sefar          # existing SeFAR environment

# 1. Run baseline SeFAR-B (for comparison table)
bash tools/run_baseline.sh gym99_5

# 2. Run SeFAR-B + CPLD
bash train_compositional.sh gym99_5

# 3. Generate results table
python tools/compare_results.py --paper --root output/ --latex paper/table1.tex

# 4. Generate figures (requires matplotlib)
pip install matplotlib
python tools/plot_figures.py --all --log-dir output/ --out paper/figures/

# 5. Compile paper
pdflatex paper/main.tex && bibtex paper/main && pdflatex paper/main.tex
```

---

## Attribute slots

| Slot | Vocab | Examples |
|------|------:|---------|
| movement_type | 14 | circle, giant, salto, leap, jump, turn, tkatchev, handspring, dive … |
| direction | 4 | backward/inward, forward, reverse/sideward, unknown |
| body_shape | 6 | pike, tuck, straddle, layout, sole, unknown |
| rotation_count | 10 | 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0+, unknown |
| end_position | 3 | to_handstand, to_standing, unknown |

---

## New config keys

```yaml
COMPOSITIONAL:
  ENABLE: True                    # set False to recover original SeFAR
  CLASS_NAMES_PATH: /path/to/gym99_class_names.json
  ATTR_WEIGHT: 0.5                # λ_attr: weight of attribute loss on labeled data
  ATTR_CONFIDENCE_THRESH: 0.85    # τ_s: gate threshold for attribute pseudo-labels
  ATTR_VOCAB_SIZES: [14, 4, 6, 10, 3]
  ATTR_SSL_WEIGHT: 1.0            # λ_ssl: weight of attribute SSL loss
```

Setting `COMPOSITIONAL.ENABLE: False` with `MODEL.MODEL_NAME: multi_granular_vit`
exactly reproduces the original SeFAR-B training.

---

## Total SSL objective

```
L = L_sup_class + λ_attr · (1/5) Σ_s L_sup_attr_s        [labeled]
  + ξ · (γ₁ · L_un_class · η                              [class pseudo-label, SeFAR]
        + γ₂ · L_mix                                       [TubeMixup, SeFAR]
        + λ_ssl · (1/5) Σ_s L_un_attr_s · η^(s))          [attribute pseudo-label, CPLD]
```

Key: `η^(s)` is computed **independently** of `η` (Adaptive Regulation).
A sample with low class confidence (`η ≈ 0`) can still contribute attribute
supervision if its attribute prediction is confident (`η^(s) > τ`).

---

## Running unit tests

```bash
# Without torch (pure parsing tests)
python tests/test_label_parser.py

# With pytest
pip install pytest
pytest tests/test_label_parser.py -v
```

Expected output: **40 tests passed** (torch-dependent tests skipped without torch).

---

## Replacing gym288_class_names.json with official FineGym names

If you have the official FineGym annotation file:

```bash
python tools/generate_gym288_names.py \
    --official /path/to/finegym_annotation_info_v1.1.json \
    --out dataset/finegym/gym288_class_names.json
```

Without `--official`, the script uses a programmatic taxonomy (288 classes
covering UB, BB, FX, VT events). Attribute coverage is high for both versions.
