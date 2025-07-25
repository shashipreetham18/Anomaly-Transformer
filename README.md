# Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy 🚀

Implementation of *Anomaly Transformer*, from the paper [“Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy”], spotlighted at **ICLR 2022** ([GitHub](https://github.com/thuml/Anomaly-Transformer), [arXiv](https://arxiv.org/abs/2110.02642)).

Unsupervised time-series anomaly detection using a novel **Association Discrepancy** criterion and **Anomaly-Attention** mechanism for enhanced distinguishability.

---

## 📌 Key Contributions

- **Association Discrepancy**: A new anomaly scoring criterion based on contrast in self-attention patterns between normal and abnormal points.
- **Anomaly-Attention**: A transformer-based attention module geared toward detecting anomalies via pairwise associations.
- **Minimax strategy**: Optimized to amplify difference between normal and abnormal behaviors in association distributions.

---

## 🧩 Repository Structure

```
Anomaly-Transformer/
├── data_factory/    # data loading and processing utilities
├── model/           # model architecture definitions
├── scripts/         # training / evaluation scripts for each dataset
├── utils/           # helper functions
├── pics/            # illustrative or sample output images
├── LICENSE
├── main.py          # script entry point
├── solver.py        # training logic and loss implementation
└── results.txt      # stored experimental results
```

---

## ⚙️ Installation

1. Install Python **>= 3.6** and PyTorch **>= 1.4.0**.
2. Clone this repository:

   ```bash
   git clone https://github.com/shashipreetham18/Anomaly-Transformer.git
   cd Anomaly-Transformer
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Datasets

We support six popular benchmarks: **SMD**, **MSL**, **SMAP**, **PSM**, **SWaT**, **WADI**. All are preprocessed and hosted on Google Cloud. Note: you may need to apply separately for data access (e.g. for SWaT).

---

## ▶️ Training & Evaluation

Use batch scripts in `scripts/`, for example:

```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
```

Alternatively, call the Python entry:

```bash
python main.py --dataset SMD --retrain
```

Scripts handle training and testing. Model outputs include anomaly detection metrics adjusted via point-adjustment strategy from Xu et al., 2018.

---

## 📊 Results

Anomaly Transformer achieves **state-of-the-art performance** on all six benchmark datasets: service monitoring, space & earth datasets, and water treatment applications.

---
