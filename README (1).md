# ⚙️ Turbofan Engine RUL Prediction Dashboard

An end-to-end **Remaining Useful Life (RUL) prediction system** for turbofan engines, built on the NASA CMAPSS FD002 dataset. It combines classical ML, deep learning (CNN-LSTM + Temporal Attention), and an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
IDSProject/
├── app.py                  # Streamlit dashboard (EDA, training, prediction)
├── supervised_models.py    # Model training script (run this first)
├── train_FD002.txt         # NASA CMAPSS training data
├── test_FD002.txt          # NASA CMAPSS test data
├── RUL_FD002.txt           # Ground truth RUL values
├── best_model.keras        # Saved CNN-LSTM model (auto-generated)
├── scaler.pkl              # Fitted MinMaxScaler (auto-generated)
└── cnn_lstm_results.json   # Training results & history (auto-generated)
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install streamlit tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### 2. Train the Models

Run this **before** launching the dashboard. It trains all three models and saves the required artifacts (`best_model.keras`, `scaler.pkl`, `cnn_lstm_results.json`).

```bash
python supervised_models.py
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

> ⚠️ **Never run** `python app.py` directly — Streamlit apps must be launched with `streamlit run`.

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 **EDA** | Dataset overview, RUL distribution, sensor trends, correlation heatmap |
| 🤖 **Supervised Models** | Train Linear Regression, Random Forest, CNN-LSTM; view metrics & curves |
| 🔍 **Unsupervised Models** | K-Means & Hierarchical clustering with PCA visualisation |
| 🔮 **Predict RUL** | Upload new engine data and get real-time RUL predictions |

---

## 🧠 Models

### Linear Regression
Baseline model trained on preprocessed sensor features with a row-level train/test split.

### Random Forest
Ensemble model with 200 trees, tuned with `min_samples_leaf=4` and `max_features=0.7` to reduce overfitting.

### CNN-LSTM + Temporal Attention
Deep learning model designed for time-series degradation modelling:
- **CNN block** — extracts local temporal patterns
- **Bidirectional LSTM** — captures long-range dependencies in both directions
- **Temporal Attention** — weights the most degradation-relevant timesteps
- **Engine-level train/val split** — prevents sequence leakage
- **Mixed precision training** — ~2× GPU throughput on Tensor Core GPUs

---

## 🔧 Preprocessing Pipeline

All steps are applied identically in both `supervised_models.py` and `app.py`:

1. **RUL label generation** — clipped at 130 cycles (piece-wise linear target)
2. **MinMaxScaler** — fitted on training data only, saved to `scaler.pkl`
3. **Low-variance sensor removal** — drops sensors 1, 5, 6, 10, 16, 18, 19
4. **Rolling statistics** — 5-cycle rolling mean & std per engine per sensor
5. **Cluster-wise z-score normalisation** — KMeans (6 clusters) on operating conditions compensates for multi-regime flight conditions

---

## 📐 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — penalises large errors |
| **MAE** | Mean Absolute Error — average prediction deviation |
| **NASA Score** | Asymmetric score: late predictions penalised more than early ones |

**NASA Score formula:**
- Early prediction (d < 0): `exp(−d/13) − 1`
- Late prediction (d ≥ 0): `exp(d/10) − 1`

Lower is better.

---

## 🔮 RUL Prediction (New Engine Data)

Upload a space-separated `.txt` or `.csv` file with **no header** and **26 columns**:

| Columns | Content |
|---------|---------|
| 1 | `engine_id` |
| 2 | `cycle` |
| 3–5 | `op_setting_1`, `op_setting_2`, `op_setting_3` |
| 6–26 | `sensor_1` … `sensor_21` |

The model predicts RUL for the **last recorded cycle** of each engine and colour-codes health status:

| Status | Condition |
|--------|-----------|
| 🔴 Critical | RUL < 20 |
| 🟡 Warning | RUL < 50 |
| 🟢 Healthy | RUL ≥ 50 |

Optionally upload a ground truth RUL file to compute RMSE, MAE, and NASA Score against predictions.

---

## 🔍 Unsupervised Analysis

Two clustering algorithms are compared on PCA-reduced sensor features:

- **K-Means** — configurable number of clusters (2–6), evaluated with Silhouette Score
- **Hierarchical (Ward linkage)** — dendrogram visualisation on a 1000-sample subset

The better-performing algorithm is automatically highlighted based on Silhouette Score.

---

## 📋 Requirements

- Python 3.9+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- streamlit

---

## 📄 Dataset

NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) — **FD002 subset** (6 operating conditions, 1 fault mode).

Dataset source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## 👤 Author

Developed as part of an Intelligent Data Systems (IDS) project.
