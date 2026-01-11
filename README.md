# CATS Anomaly Detection

Machine learning system for detecting anomalies in multivariate time series data using the Controlled Anomalies Time Series (CATS) dataset.

## Overview

This project implements an ensemble anomaly detection system for industrial monitoring applications. It analyzes 17 sensor channels across 5 million timestamps to identify anomalous system behavior with 97% accuracy (ROC-AUC 0.9712).

The system combines three complementary approaches:
- **Random Forest**: Fast, interpretable baseline with feature importance analysis
- **Gradient Boosting**: Sequential learning for complex pattern detection
- **Isolation Forest**: Unsupervised detection for novel anomaly types

## Applications

This anomaly detection framework applies to:
- Industrial monitoring systems (sensors, actuators, control systems)
- Infrastructure health monitoring
- Cybersecurity (network intrusion detection, system behavior analysis)
- Financial systems (fraud detection, trading anomalies)

The techniques handle multivariate time series with severe class imbalance (96:4 ratio), making them suitable for rare event detection in operational environments.

## Dataset

The CATS (Controlled Anomalies Time Series) dataset is a benchmark dataset from Solenix Engineering GmbH designed for evaluating anomaly detection algorithms. It simulates a complex dynamical system with deliberately injected anomalies.

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

## Advanced: Transformer-Based Model

An experimental implementation based on recent research in transformer-based anomaly detection is available in `transformer_paper_implementation.py`.

### Overview

This implementation follows the methodology from "A Transformer-Based Framework for Anomaly Detection in Multivariate Time Series" (Folger et al., CLOUD COMPUTING 2025).

**Key Features:**
- Vanilla transformer encoder with positional encoding
- Focal Loss for class imbalance handling
- Semi-supervised learning (trains on normal data, validates with some anomalies)
- Optuna hyperparameter optimization
- Window-based sequence modeling (128 timesteps)

**Target Performance:**
- ROC-AUC: ~0.999
- F1 Score: ~0.97
- Recall: ~0.985

### Usage

Install additional dependencies:
```bash
pip install torch>=2.0.0 tqdm>=4.65.0 optuna>=3.0.0
```

Run the transformer pipeline:
```bash
python transformer_paper_implementation.py
```

Enable hyperparameter optimization:
```bash
RUN_OPTUNA=1 python transformer_paper_implementation.py
```

### Architecture

```
Input (17 features)
→ Linear Embedding (to model_dim=128)
→ Batch Normalization
→ Positional Encoding
→ Transformer Encoder (3 layers, 8 heads)
→ Mean Pooling (temporal aggregation)
→ Classification Head
→ Sigmoid → Anomaly Score
```

### Limitations

1. **Memory intensive**: No sampling optimization, requires sufficient RAM for full dataset
2. **GPU recommended**: Training can be slow on CPU
3. **Dataset assumption**: Requires first 1M rows to be anomaly-free for proper training
4. **Window-level labels**: Uses "ANY" labeling strategy which can be noisy for short anomalies
5. **No visualizations**: Outputs metrics only, no plots generated
6. **Longer training**: Takes significantly more time than traditional ML models

### When to Use

Use the transformer model when:
- Maximum detection accuracy is critical
- Computational resources (GPU, RAM) are available
- You have clean separation of normal/anomalous training data
- Longer training time is acceptable

Use the traditional ensemble when:
- Fast training and inference are needed
- Limited computational resources
- Interpretability is important
- Good enough performance (97% AUC) is sufficient

## Technical Highlights

**Algorithms:**
- Random Forest, Gradient Boosting, Isolation Forest (ensemble achieves 0.9712 ROC-AUC)
- Transformer-based model with Focal Loss (experimental, targeting 0.999 ROC-AUC)

**Data Handling:**
- Stratified sampling for memory optimization on large datasets
- Min-Max and Standard scaling for different model requirements
- Balanced class weights for 96:4 imbalance ratio

**Analysis:**
- Feature correlation analysis to identify coupled system components
- Root cause identification through feature importance ranking
- Threshold optimization for precision-recall trade-offs

**Deployment:**
- Real-time scoring pipeline with configurable thresholds
- Interactive dashboard for operational monitoring
- Model persistence and versioning

## Project Structure

```
cats-anomaly-detection/
├── cats_pipeline.py                      # Main training pipeline (traditional ML)
├── streamlit_dash.py                     # Interactive dashboard
├── transformer_paper_implementation.py   # Advanced transformer model
├── requirements.txt                      # Python dependencies
├── data.csv                             # CATS dataset (not included)
├── metadata.csv                         # Anomaly metadata (not included)
├── outputs/                             # Generated results
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
4. Folger et al. (2025). A Transformer-Based Framework for Anomaly Detection in Multivariate Time Series. CLOUD COMPUTING 2025.
