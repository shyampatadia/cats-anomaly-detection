# CATS Anomaly Detection

Machine learning system for detecting anomalies in multivariate time series data using the Controlled Anomalies Time Series (CATS) dataset.

## Overview

This project implements an ensemble anomaly detection system for industrial monitoring applications. It analyzes 17 sensor channels across 5 million timestamps to identify anomalous system behavior with 97% accuracy (ROC-AUC 0.9712).

The system combines three complementary approaches:
- **Random Forest**: Fast, interpretable baseline with feature importance analysis
- **Gradient Boosting**: Sequential learning for complex pattern detection
- **Isolation Forest**: Unsupervised detection for novel anomaly types

## Dataset

The CATS dataset simulates a complex dynamical system with deliberately injected anomalies.

**Key Statistics:**
- Total timestamps: 5,000,000
- Sensor channels: 17
- Anomaly rate: 3.8%
- Anomaly segments: 200

**Channel Categories:**
- Commands (4): aimp, amud, adbr, adfl
- Environmental (3): arnd, asin1, asin2
- Telemetry (10): bed1, bed2, bfo1, bfo2, bso1, bso2, bso3, ced1, cfo1, cso1

## Installation

```bash
# Clone repository
git clone <repository-url>
cd cats-anomaly-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run the Pipeline

Train models and generate visualizations:

```bash
python cats_pipeline.py
```

This will:
- Load and preprocess the CATS dataset
- Perform exploratory data analysis
- Train Random Forest, Gradient Boosting, and Isolation Forest models
- Generate evaluation metrics and plots
- Save models and results to `outputs/`

**Output structure:**
```
outputs/
├── eda_plots/          # 10 visualization images
├── models/             # Trained models (.joblib)
├── results/            # JSON result files
├── dashboard_data.json # Dashboard data
└── dashboard.html      # Standalone HTML dashboard
```

### 2. Launch Dashboard

Run the interactive Streamlit dashboard:

```bash
streamlit run streamlit_dash.py
```

The dashboard provides three views:
- **Live Monitor**: Real-time anomaly score visualization with configurable threshold
- **EDA Analysis**: Dataset statistics, correlations, and feature importance
- **Model Comparison**: Side-by-side performance metrics

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 96.2% | 89.4% | 87.1% | 88.2% | 0.9634 |
| Gradient Boosting | 95.8% | 86.7% | 85.3% | 86.0% | 0.9521 |
| Isolation Forest | 94.1% | 78.2% | 82.6% | 80.3% | 0.9287 |
| **Ensemble** | **96.5%** | **88.1%** | **89.2%** | **88.6%** | **0.9712** |

## Key Findings

**Top Predictive Features:**
1. bfo2 (18.3%)
2. cso1 (15.7%)
3. bso3 (12.4%)
4. ced1 (9.8%)
5. bso1 (8.2%)

These telemetry channels correspond to the most common root causes of anomalies.

**Feature Correlations:**
- bso1 ↔ amud: 0.98
- cfo1 ↔ amud: 0.77
- bed1 ↔ bed2: 0.72

High correlations indicate coupled system components where anomalies can propagate.

## Interpreting Anomaly Scores

The ensemble model outputs a score between 0 and 1:

- **0.0 - 0.3**: Low risk, normal operation
- **0.3 - 0.5**: Moderate risk, increased monitoring recommended
- **0.5 - 0.7**: High risk, investigate sensor readings
- **0.7 - 1.0**: Critical, immediate attention required

The default detection threshold is 0.5, configurable in the dashboard.

## Memory Optimization

The pipeline uses stratified sampling to handle large datasets on local machines:

- Training set: 200,000 samples
- Test set: 50,000 samples
- EDA visualization: 100,000 samples

Sampling maintains the original 3.8% anomaly rate for statistical validity. Adjust `TRAIN_SAMPLE_SIZE`, `TEST_SAMPLE_SIZE`, and `EDA_SAMPLE_SIZE` in `cats_pipeline.py` based on available RAM.

## Technical Details

**Preprocessing:**
- Min-Max scaling to [0, 1] range
- Stratified train/test split (80/20)
- Balanced class weights to handle 96:4 class imbalance

**Model Hyperparameters:**

Random Forest:
```python
n_estimators=100, max_depth=15,
min_samples_split=10, min_samples_leaf=5
```

Gradient Boosting:
```python
n_estimators=50, max_depth=5,
learning_rate=0.1, subsample=0.8
```

Isolation Forest:
```python
n_estimators=100, contamination=0.038
```

## Project Structure

```
cats-anomaly-detection/
├── cats_pipeline.py      # Main training pipeline
├── streamlit_dash.py     # Interactive dashboard
├── requirements.txt      # Python dependencies
├── data.csv             # CATS dataset (not included)
├── metadata.csv         # Anomaly metadata (not included)
├── outputs/             # Generated results
└── README.md
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib

## License

This project is for educational purposes.

## References

1. Solenix Engineering GmbH. (2023). Controlled Anomalies Time Series (CATS) Dataset Description Document - Version 2.
2. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. ICDM, 413-422.
