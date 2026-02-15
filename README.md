# Does-Confidence-Calibration-Improve-Conformal-Prediction

This repository is the official implementation of ["Does confidence calibration improve conformal prediction?"](https://openreview.net/forum?id=6DDaTwTvdE) at TMLR'2025.

## Setup

1. Install [uv](https://github.com/astral-sh/uv) (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

2. Configure data path:
```bash
cp .env.example .env
# Edit .env and set DATA_DIR to your ImageNet root directory
```

## How to Run

Results for ConfTS:
```bash
uv run python main.py --preprocess confts
```

Results for ConfPS:
```bash
uv run python main.py --preprocess confps
```

Results for ConfVS:
```bash
uv run python main.py --preprocess confvs
```

## Available Methods

- **confts**: Conformal Temperature Scaling (proposed)
- **confps**: Conformal Platt Scaling (proposed)
- **confvs**: Conformal Vector Scaling (proposed)
- **ts**: Temperature Scaling (baseline)
- **ps**: Platt Scaling (baseline)
- **vs**: Vector Scaling (baseline)
- **none**: Identity (no calibration)

## Citation
If you find this useful in your research, please consider citing:
```bash
@article{
    xi2025does,
    title={Does confidence calibration improve conformal prediction?},
    author={HuaJun Xi and Jianguo Huang and Kangdao Liu and Lei Feng and Hongxin Wei},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
}
```
