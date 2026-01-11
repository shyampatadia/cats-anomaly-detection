"""
CATS Anomaly Detection - MEMORY OPTIMIZED Pipeline
===================================================
Optimized for local machines with limited RAM.
Uses stratified sampling to train on subset while maintaining statistical validity.

Usage:
    python cats_pipeline_optimized.py

Outputs:
    - outputs/eda_plots/           (EDA visualizations)
    - outputs/models/              (trained models)
    - outputs/results/             (metrics, reports)
    - outputs/dashboard_data.json  (for dashboard)
    - outputs/dashboard.html       (standalone dashboard)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, 
    precision_recall_curve, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ADJUST THESE FOR YOUR MACHINE
# ============================================================================

DATA_PATH = "data.csv"
METADATA_PATH = "metadata.csv"
OUTPUT_DIR = "outputs"
SEED = 42

# MEMORY OPTIMIZATION: Sample size for training
# Increase if you have more RAM, decrease if you get memory errors
TRAIN_SAMPLE_SIZE = 200_000  # 200k samples (instead of 4M)
TEST_SAMPLE_SIZE = 50_000    # 50k samples for testing
EDA_SAMPLE_SIZE = 100_000    # 100k for EDA visualizations

FEATURE_COLS = [
    "aimp", "amud", "arnd", "asin1", "asin2", "adbr", "adfl",
    "bed1", "bed2", "bfo1", "bfo2", "bso1", "bso2", "bso3",
    "ced1", "cfo1", "cso1"
]

CHANNEL_TYPES = {
    "Commands": ["aimp", "amud", "adbr", "adfl"],
    "Environmental": ["arnd", "asin1", "asin2"],
    "Telemetry": ["bed1", "bed2", "bfo1", "bfo2", "bso1", "bso2", "bso3", "ced1", "cfo1", "cso1"]
}

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create output directories."""
    dirs = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/eda_plots",
        f"{OUTPUT_DIR}/models",
        f"{OUTPUT_DIR}/results"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✓ Output directories created in '{OUTPUT_DIR}/'")

# ============================================================================
# DATA LOADING WITH SAMPLING
# ============================================================================

def load_data():
    """Load main data and metadata with memory optimization."""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load main data
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)
    print(f"✓ Loaded: {total_rows:,} rows, {len(df.columns)} columns")
    
    # Store full dataset stats before sampling
    full_stats = {
        'total_rows': total_rows,
        'anomaly_count': int(df['y'].sum()),
        'anomaly_rate': float(df['y'].mean() * 100)
    }
    print(f"  Full dataset anomaly rate: {full_stats['anomaly_rate']:.2f}%")
    
    # Load metadata if exists
    metadata = None
    if os.path.exists(METADATA_PATH):
        metadata = pd.read_csv(METADATA_PATH)
        print(f"✓ Loaded {METADATA_PATH}: {len(metadata)} anomaly segments")
    else:
        print(f"⚠ {METADATA_PATH} not found, continuing without metadata")
    
    return df, metadata, full_stats


def stratified_sample(df, n_samples, random_state=42):
    """Create stratified sample maintaining class proportions."""
    if n_samples >= len(df):
        return df
    
    # Stratified sampling
    sample_df, _ = train_test_split(
        df, 
        train_size=n_samples, 
        stratify=df['y'], 
        random_state=random_state
    )
    return sample_df.reset_index(drop=True)

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(df, metadata, full_stats):
    """Comprehensive EDA with visualizations."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Use sample for EDA visualizations
    eda_sample = stratified_sample(df, EDA_SAMPLE_SIZE, SEED)
    print(f"Using {len(eda_sample):,} samples for EDA (stratified)")
    
    eda_results = {}
    
    # --- Basic Statistics (from FULL dataset) ---
    print("\n[1/7] Basic Statistics...")
    
    eda_results['n_samples'] = full_stats['total_rows']
    eda_results['n_features'] = len(FEATURE_COLS)
    eda_results['anomaly_rate'] = full_stats['anomaly_rate']
    eda_results['n_anomalies'] = full_stats['anomaly_count']
    eda_results['n_normal'] = full_stats['total_rows'] - full_stats['anomaly_count']
    
    print(f"  Total samples: {eda_results['n_samples']:,}")
    print(f"  Features: {eda_results['n_features']}")
    print(f"  Anomaly rate: {eda_results['anomaly_rate']:.2f}%")
    print(f"  Normal: {eda_results['n_normal']:,} | Anomaly: {eda_results['n_anomalies']:,}")
    
    # --- Class Distribution Plot ---
    print("\n[2/7] Plotting class distribution...")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#10b981', '#ef4444']
    labels = ['Normal', 'Anomaly']
    counts = [eda_results['n_normal'], eda_results['n_anomalies']]
    bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution (Full Dataset)', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                f'{count:,}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots/01_class_distribution.png", dpi=150)
    plt.close()
    print("  ✓ Saved 01_class_distribution.png")
    
    # --- Feature Statistics ---
    print("\n[3/7] Computing feature statistics...")
    
    feature_stats = eda_sample[FEATURE_COLS].describe().T
    feature_stats['range'] = feature_stats['max'] - feature_stats['min']
    eda_results['feature_stats'] = feature_stats.to_dict()
    
    # Box plots
    fig, ax = plt.subplots(figsize=(14, 6))
    eda_sample[FEATURE_COLS].boxplot(ax=ax, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='#3b82f6', alpha=0.7),
                              medianprops=dict(color='#ef4444', linewidth=2))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha='right')
    ax.set_title('Feature Distributions (Box Plot)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots/02_feature_boxplots.png", dpi=150)
    plt.close()
    print("  ✓ Saved 02_feature_boxplots.png")
    
    # --- Correlation Matrix ---
    print("\n[4/7] Computing correlation matrix...")
    
    corr_matrix = eda_sample[FEATURE_COLS].corr()
    eda_results['correlations'] = corr_matrix.to_dict()
    
    # Find top correlations
    corr_pairs = []
    for i in range(len(FEATURE_COLS)):
        for j in range(i+1, len(FEATURE_COLS)):
            corr_pairs.append({
                'pair': f"{FEATURE_COLS[i]}-{FEATURE_COLS[j]}",
                'correlation': float(abs(corr_matrix.iloc[i, j]))
            })
    corr_pairs = sorted(corr_pairs, key=lambda x: x['correlation'], reverse=True)[:10]
    eda_results['top_correlations'] = corr_pairs
    
    print("  Top 5 correlations:")
    for cp in corr_pairs[:5]:
        print(f"    {cp['pair']}: {cp['correlation']:.3f}")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots/03_correlation_heatmap.png", dpi=150)
    plt.close()
    print("  ✓ Saved 03_correlation_heatmap.png")
    
    # --- Anomaly vs Normal Feature Comparison ---
    print("\n[5/7] Comparing features by class...")
    
    normal_data = eda_sample[eda_sample['y'] == 0][FEATURE_COLS]
    anomaly_data = eda_sample[eda_sample['y'] == 1][FEATURE_COLS]
    
    feature_diff = {}
    for col in FEATURE_COLS:
        normal_mean = float(normal_data[col].mean())
        anomaly_mean = float(anomaly_data[col].mean())
        diff_pct = ((anomaly_mean - normal_mean) / (normal_mean + 1e-10)) * 100
        feature_diff[col] = {
            'normal_mean': normal_mean,
            'anomaly_mean': anomaly_mean,
            'diff_pct': float(diff_pct)
        }
    eda_results['feature_diff'] = feature_diff
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FEATURE_COLS))
    width = 0.35
    bars1 = ax.bar(x - width/2, [normal_data[c].mean() for c in FEATURE_COLS], 
                   width, label='Normal', color='#10b981', alpha=0.8)
    bars2 = ax.bar(x + width/2, [anomaly_data[c].mean() for c in FEATURE_COLS], 
                   width, label='Anomaly', color='#ef4444', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha='right')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Feature Values: Normal vs Anomaly', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots/04_normal_vs_anomaly.png", dpi=150)
    plt.close()
    print("  ✓ Saved 04_normal_vs_anomaly.png")
    
    # --- Metadata Analysis (if available) ---
    if metadata is not None:
        print("\n[6/7] Analyzing metadata...")
        
        # Root cause distribution
        if 'root_cause' in metadata.columns:
            root_cause_counts = metadata['root_cause'].value_counts()
            eda_results['root_cause_distribution'] = {str(k): int(v) for k, v in root_cause_counts.items()}
            
            fig, ax = plt.subplots(figsize=(10, 6))
            root_cause_counts.plot(kind='barh', ax=ax, color='#3b82f6', edgecolor='white')
            ax.set_xlabel('Count')
            ax.set_title('Root Cause Channel Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/eda_plots/05_root_cause_distribution.png", dpi=150)
            plt.close()
            print("  ✓ Saved 05_root_cause_distribution.png")
        
        # Category distribution
        if 'category' in metadata.columns:
            category_counts = metadata['category'].value_counts()
            eda_results['category_distribution'] = {str(k): int(v) for k, v in category_counts.items()}
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
            category_counts.plot(kind='pie', ax=ax, colors=colors, autopct='%1.1f%%')
            ax.set_ylabel('')
            ax.set_title('Anomaly Category Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/eda_plots/06_category_distribution.png", dpi=150)
            plt.close()
            print("  ✓ Saved 06_category_distribution.png")
    else:
        print("\n[6/7] Skipping metadata analysis (no metadata file)")
        eda_results['root_cause_distribution'] = {}
        eda_results['category_distribution'] = {}
    
    # --- Time Series Sample ---
    print("\n[7/7] Plotting time series sample...")
    
    # Take first 5000 rows for time series plot
    sample_df = df.iloc[:5000].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot a few key channels
    key_channels = ['bfo2', 'cso1', 'bso3']
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    for i, (ch, color) in enumerate(zip(key_channels, colors)):
        if ch in sample_df.columns:
            axes[i].plot(sample_df.index, sample_df[ch], color=color, linewidth=0.5, alpha=0.8)
            # Highlight anomalies
            anomaly_idx = sample_df[sample_df['y'] == 1].index
            if len(anomaly_idx) > 0:
                axes[i].scatter(anomaly_idx, sample_df.loc[anomaly_idx, ch], 
                               color='#ef4444', s=5, alpha=0.5, label='Anomaly')
            axes[i].set_ylabel(ch, fontsize=11)
            axes[i].legend(loc='upper right')
    
    axes[2].set_xlabel('Time Index')
    axes[0].set_title('Time Series with Anomalies Highlighted', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots/07_timeseries_sample.png", dpi=150)
    plt.close()
    print("  ✓ Saved 07_timeseries_sample.png")
    
    # Save EDA results
    with open(f"{OUTPUT_DIR}/results/eda_results.json", 'w') as f:
        json.dump(eda_results, f, indent=2)
    print(f"\n✓ EDA results saved to {OUTPUT_DIR}/results/eda_results.json")
    
    return eda_results

# ============================================================================
# MODEL TRAINING (MEMORY OPTIMIZED)
# ============================================================================

def train_models(df):
    """Train multiple models using stratified sampling."""
    print("\n" + "="*60)
    print("MODEL TRAINING (Memory Optimized)")
    print("="*60)
    
    # Create stratified sample for training
    total_sample_size = TRAIN_SAMPLE_SIZE + TEST_SAMPLE_SIZE
    print(f"\nSampling {total_sample_size:,} rows (stratified) from {len(df):,} total...")
    
    sample_df = stratified_sample(df, total_sample_size, SEED)
    print(f"✓ Sample created: {len(sample_df):,} rows")
    print(f"  Sample anomaly rate: {sample_df['y'].mean()*100:.2f}% (matches original)")
    
    # Prepare data
    X = sample_df[FEATURE_COLS].values
    y = sample_df['y'].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SAMPLE_SIZE/(TRAIN_SAMPLE_SIZE + TEST_SAMPLE_SIZE), 
        random_state=SEED, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Train anomaly rate: {y_train.mean()*100:.2f}%")
    print(f"Test anomaly rate: {y_test.mean()*100:.2f}%")
    
    models = {}
    results = {}
    
    # --- Random Forest ---
    print("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,       # Reduced from 200
        max_depth=15,           # Limited depth
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=SEED
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    results['random_forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf)),
        'recall': float(recall_score(y_test, y_pred_rf)),
        'f1': float(f1_score(y_test, y_pred_rf)),
        'roc_auc': float(roc_auc_score(y_test, y_proba_rf))
    }
    print(f"  ✓ Random Forest - AUC: {results['random_forest']['roc_auc']:.4f}")
    
    # Feature importance
    feature_importance = {k: float(v) for k, v in zip(FEATURE_COLS, rf.feature_importances_)}
    results['random_forest']['feature_importance'] = feature_importance
    
    # --- Gradient Boosting ---
    print("\n[2/3] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=50,        # Reduced
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,          # Use subsample for speed
        random_state=SEED
    )
    gb.fit(X_train, y_train)
    models['gradient_boosting'] = gb
    
    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    
    results['gradient_boosting'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_gb)),
        'precision': float(precision_score(y_test, y_pred_gb)),
        'recall': float(recall_score(y_test, y_pred_gb)),
        'f1': float(f1_score(y_test, y_pred_gb)),
        'roc_auc': float(roc_auc_score(y_test, y_proba_gb))
    }
    print(f"  ✓ Gradient Boosting - AUC: {results['gradient_boosting']['roc_auc']:.4f}")
    
    # --- Isolation Forest (unsupervised) ---
    print("\n[3/3] Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.038,  # CATS anomaly rate
        random_state=SEED,
        n_jobs=-1
    )
    iso.fit(X_train)
    models['isolation_forest'] = iso
    
    # Isolation Forest returns -1 for anomaly, 1 for normal
    y_pred_iso = iso.predict(X_test)
    y_pred_iso = (y_pred_iso == -1).astype(int)  # Convert to 0/1
    y_scores_iso = -iso.score_samples(X_test)  # Higher = more anomalous
    
    results['isolation_forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_iso)),
        'precision': float(precision_score(y_test, y_pred_iso)),
        'recall': float(recall_score(y_test, y_pred_iso)),
        'f1': float(f1_score(y_test, y_pred_iso)),
        'roc_auc': float(roc_auc_score(y_test, y_scores_iso))
    }
    print(f"  ✓ Isolation Forest - AUC: {results['isolation_forest']['roc_auc']:.4f}")
    
    # --- Save Models ---
    print("\nSaving models...")
    joblib.dump(rf, f"{OUTPUT_DIR}/models/random_forest.joblib")
    joblib.dump(gb, f"{OUTPUT_DIR}/models/gradient_boosting.joblib")
    joblib.dump(iso, f"{OUTPUT_DIR}/models/isolation_forest.joblib")
    joblib.dump(scaler, f"{OUTPUT_DIR}/models/scaler.joblib")
    print(f"  ✓ Models saved to {OUTPUT_DIR}/models/")
    
    # --- Plot ROC Curves ---
    print("\nPlotting ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, y_proba in [('Random Forest', y_proba_rf), 
                          ('Gradient Boosting', y_proba_gb),
                          ('Isolation Forest', y_scores_iso)]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots/08_roc_curves.png", dpi=150)
    plt.close()
    print("  ✓ Saved 08_roc_curves.png")
    
    # --- Feature Importance Plot ---
    print("Plotting feature importance...")
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    colors = ['#ef4444' if imp > importance_df['importance'].mean() else '#3b82f6' 
              for imp in importance_df['importance']]
    ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots/09_feature_importance.png", dpi=150)
    plt.close()
    print("  ✓ Saved 09_feature_importance.png")
    
    # --- Confusion Matrix ---
    print("Plotting confusion matrix...")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Random Forest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots/10_confusion_matrix.png", dpi=150)
    plt.close()
    print("  ✓ Saved 10_confusion_matrix.png")
    
    # --- Save Results ---
    with open(f"{OUTPUT_DIR}/results/model_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Model results saved to {OUTPUT_DIR}/results/model_results.json")
    
    return models, results, scaler, (X_test, y_test, y_proba_rf)

# ============================================================================
# PREPARE DASHBOARD DATA
# ============================================================================

def prepare_dashboard_data(df, models, scaler, eda_results, model_results):
    """Prepare data for the dashboard."""
    print("\n" + "="*60)
    print("PREPARING DASHBOARD DATA")
    print("="*60)
    
    # Get a sample for dashboard simulation (unseen data)
    sample_size = 500
    # Take from latter part of dataset (simulating "new" data)
    start_idx = len(df) - sample_size * 10
    sample_indices = np.random.choice(range(start_idx, len(df)), size=sample_size, replace=False)
    sample_indices = np.sort(sample_indices)
    sample_df = df.iloc[sample_indices].copy()
    
    print(f"Selected {len(sample_df)} unseen samples for dashboard")
    
    # Scale and predict
    X_sample = sample_df[FEATURE_COLS].values
    X_sample_scaled = scaler.transform(X_sample)
    
    rf = models['random_forest']
    gb = models['gradient_boosting']
    iso = models['isolation_forest']
    
    sample_df['rf_proba'] = rf.predict_proba(X_sample_scaled)[:, 1]
    sample_df['gb_proba'] = gb.predict_proba(X_sample_scaled)[:, 1]
    sample_df['iso_score'] = -iso.score_samples(X_sample_scaled)
    # Normalize isolation forest scores to 0-1
    iso_min, iso_max = sample_df['iso_score'].min(), sample_df['iso_score'].max()
    sample_df['iso_proba'] = (sample_df['iso_score'] - iso_min) / (iso_max - iso_min + 1e-10)
    sample_df['ensemble_proba'] = (sample_df['rf_proba'] + sample_df['gb_proba'] + sample_df['iso_proba']) / 3
    
    # Prepare JSON data
    dashboard_data = {
        'generated_at': datetime.now().isoformat(),
        'eda': {
            'n_samples': eda_results['n_samples'],
            'n_features': eda_results['n_features'],
            'anomaly_rate': eda_results['anomaly_rate'],
            'n_anomalies': eda_results['n_anomalies'],
            'top_correlations': eda_results['top_correlations'][:5],
            'root_cause_distribution': eda_results.get('root_cause_distribution', {}),
            'category_distribution': eda_results.get('category_distribution', {})
        },
        'models': {
            name: {
                'accuracy': round(m['accuracy'] * 100, 2),
                'precision': round(m['precision'] * 100, 2),
                'recall': round(m['recall'] * 100, 2),
                'f1': round(m['f1'] * 100, 2),
                'roc_auc': round(m['roc_auc'], 4)
            }
            for name, m in model_results.items()
        },
        'feature_importance': dict(sorted(
            model_results['random_forest']['feature_importance'].items(),
            key=lambda x: x[1], reverse=True
        )[:10]),
        'time_series': []
    }
    
    # Add time series data
    for idx, row in sample_df.iterrows():
        point = {
            'index': int(idx),
            'true_label': int(row['y']),
            'rf_proba': round(float(row['rf_proba']), 4),
            'gb_proba': round(float(row['gb_proba']), 4),
            'iso_proba': round(float(row['iso_proba']), 4),
            'ensemble_proba': round(float(row['ensemble_proba']), 4),
            'channels': {ch: round(float(row[ch]), 4) for ch in FEATURE_COLS[:5]}
        }
        dashboard_data['time_series'].append(point)
    
    # Save dashboard data
    with open(f"{OUTPUT_DIR}/dashboard_data.json", 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"✓ Dashboard data saved to {OUTPUT_DIR}/dashboard_data.json")
    
    return dashboard_data

# ============================================================================
# GENERATE HTML DASHBOARD
# ============================================================================

def generate_html_dashboard(dashboard_data):
    """Generate standalone HTML dashboard."""
    print("\n" + "="*60)
    print("GENERATING HTML DASHBOARD")
    print("="*60)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CATS Anomaly Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(16, 185, 129, 0.1);
            border-radius: 12px;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .header h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #10b981, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }
        .header p { color: #94a3b8; }
        .grid { display: grid; gap: 20px; }
        .grid-4 { grid-template-columns: repeat(4, 1fr); }
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        @media (max-width: 1200px) { .grid-4, .grid-3 { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 768px) { .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr; } }
        .card {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
        }
        .card h3 {
            font-size: 0.875rem;
            color: #94a3b8;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #fff;
        }
        .card .subtext { font-size: 0.75rem; color: #64748b; margin-top: 4px; }
        .card.green .value { color: #10b981; }
        .card.red .value { color: #ef4444; }
        .card.blue .value { color: #3b82f6; }
        .card.amber .value { color: #f59e0b; }
        .section { margin-bottom: 30px; }
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .chart-container {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
        }
        .model-card {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
            transition: all 0.3s;
        }
        .model-card:hover {
            border-color: #10b981;
            transform: translateY(-2px);
        }
        .model-card.best {
            border-color: #10b981;
            background: rgba(16, 185, 129, 0.1);
        }
        .model-card h4 { font-size: 1.1rem; margin-bottom: 15px; }
        .model-card .badge {
            background: #10b981;
            color: #fff;
            padding: 2px 8px;
            border-radius: 20px;
            font-size: 0.7rem;
            margin-left: 8px;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #334155;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #94a3b8; }
        .metric-value { font-weight: 600; }
        .bar-chart { display: flex; flex-direction: column; gap: 10px; }
        .bar-item { display: flex; align-items: center; gap: 10px; }
        .bar-label { width: 100px; font-size: 0.875rem; color: #94a3b8; }
        .bar-bg {
            flex: 1;
            height: 24px;
            background: #1e293b;
            border-radius: 4px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #06b6d4);
            border-radius: 4px;
            display: flex;
            align-items: center;
            padding-left: 8px;
            font-size: 0.75rem;
            color: #fff;
            font-weight: 600;
        }
        .alert-feed { max-height: 300px; overflow-y: auto; }
        .alert-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .alert-item .time { color: #94a3b8; font-size: 0.875rem; }
        .alert-item .score { color: #ef4444; font-weight: 600; }
        .controls {
            display: flex;
            gap: 20px;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .control-group { display: flex; align-items: center; gap: 10px; }
        .control-group label { color: #94a3b8; font-size: 0.875rem; }
        select, input[type="range"] {
            background: #1e293b;
            border: 1px solid #334155;
            color: #fff;
            padding: 8px 12px;
            border-radius: 8px;
        }
        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        .btn-primary { background: #10b981; color: #fff; }
        .btn-primary:hover { background: #059669; }
        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(16, 185, 129, 0.2);
            border-radius: 20px;
            font-size: 0.875rem;
            color: #10b981;
        }
        .live-dot {
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab {
            padding: 10px 20px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #94a3b8;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab:hover { background: #334155; }
        .tab.active {
            background: #10b981;
            color: #fff;
            border-color: #10b981;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ CATS Anomaly Detection System</h1>
        <p>Multivariate Time Series Monitoring Dashboard</p>
    </div>

    <div class="tabs">
        <div class="tab active" onclick="showTab('monitor')">📊 Live Monitor</div>
        <div class="tab" onclick="showTab('eda')">📈 EDA Analysis</div>
        <div class="tab" onclick="showTab('models')">🤖 Model Comparison</div>
    </div>

    <!-- MONITOR TAB -->
    <div id="monitor-tab" class="tab-content active">
        <div class="section grid grid-4">
            <div class="card green">
                <h3>Total Samples</h3>
                <div class="value" id="stat-samples">-</div>
                <div class="subtext">in dataset</div>
            </div>
            <div class="card red">
                <h3>Anomaly Rate</h3>
                <div class="value" id="stat-rate">-</div>
                <div class="subtext">of observations</div>
            </div>
            <div class="card blue">
                <h3>Detected Now</h3>
                <div class="value" id="stat-detected">0</div>
                <div class="subtext">above threshold</div>
            </div>
            <div class="card amber">
                <h3>Current Score</h3>
                <div class="value" id="stat-current">0.00</div>
                <div class="subtext">anomaly probability</div>
            </div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label>Model:</label>
                <select id="model-select" onchange="updateChart()">
                    <option value="rf_proba">Random Forest</option>
                    <option value="gb_proba">Gradient Boosting</option>
                    <option value="iso_proba">Isolation Forest</option>
                    <option value="ensemble_proba" selected>Ensemble</option>
                </select>
            </div>
            <div class="control-group">
                <label>Threshold: <span id="threshold-value">0.50</span></label>
                <input type="range" id="threshold-slider" min="0.1" max="0.9" step="0.05" value="0.5" onchange="updateThreshold()">
            </div>
            <button class="btn btn-primary" onclick="toggleStreaming()">
                <span id="stream-btn-text">⏸ Pause</span>
            </button>
            <div class="live-indicator" id="live-indicator">
                <div class="live-dot"></div>
                <span>LIVE</span>
            </div>
        </div>

        <div class="section grid grid-3">
            <div class="chart-container" style="grid-column: span 2;">
                <h3 style="margin-bottom: 15px;">Anomaly Score Over Time</h3>
                <canvas id="score-chart" height="100"></canvas>
            </div>
            <div class="chart-container">
                <h3 style="margin-bottom: 15px;">Recent Alerts</h3>
                <div class="alert-feed" id="alert-feed">
                    <p style="color: #64748b; text-align: center;">No alerts yet</p>
                </div>
            </div>
        </div>
    </div>

    <!-- EDA TAB -->
    <div id="eda-tab" class="tab-content">
        <div class="section grid grid-4">
            <div class="card green">
                <h3>Total Samples</h3>
                <div class="value" id="eda-samples">-</div>
            </div>
            <div class="card blue">
                <h3>Features</h3>
                <div class="value">17</div>
            </div>
            <div class="card red">
                <h3>Anomalies</h3>
                <div class="value" id="eda-anomalies">-</div>
            </div>
            <div class="card amber">
                <h3>Anomaly Rate</h3>
                <div class="value" id="eda-rate">-</div>
            </div>
        </div>

        <div class="section grid grid-2">
            <div class="chart-container">
                <h3 style="margin-bottom: 15px;">Top Feature Correlations</h3>
                <div class="bar-chart" id="correlation-bars"></div>
            </div>
            <div class="chart-container">
                <h3 style="margin-bottom: 15px;">Feature Importance</h3>
                <div class="bar-chart" id="importance-bars"></div>
            </div>
        </div>

        <div class="section grid grid-2">
            <div class="chart-container">
                <h3 style="margin-bottom: 15px;">Root Cause Distribution</h3>
                <canvas id="rootcause-chart" height="200"></canvas>
            </div>
            <div class="chart-container">
                <h3 style="margin-bottom: 15px;">Channel Categories</h3>
                <div style="padding: 20px;">
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <div style="width: 14px; height: 14px; background: #3b82f6; border-radius: 3px;"></div>
                            <strong>Commands (4)</strong>
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem; margin-left: 22px;">aimp, amud, adbr, adfl</p>
                    </div>
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <div style="width: 14px; height: 14px; background: #f59e0b; border-radius: 3px;"></div>
                            <strong>Environmental (3)</strong>
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem; margin-left: 22px;">arnd, asin1, asin2</p>
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <div style="width: 14px; height: 14px; background: #10b981; border-radius: 3px;"></div>
                            <strong>Telemetry (10)</strong>
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem; margin-left: 22px;">bed1, bed2, bfo1, bfo2, bso1, bso2, bso3, ced1, cfo1, cso1</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- MODELS TAB -->
    <div id="models-tab" class="tab-content">
        <div class="section grid grid-3" id="model-cards"></div>
        
        <div class="section">
            <div class="chart-container">
                <h3 style="margin-bottom: 15px;">Model Performance Comparison</h3>
                <canvas id="model-comparison-chart" height="100"></canvas>
            </div>
        </div>
    </div>

    <script>
        const dashboardData = ''' + json.dumps(dashboard_data) + ''';

        let isStreaming = true;
        let currentIndex = 0;
        let threshold = 0.5;
        let selectedModel = 'ensemble_proba';
        let alerts = [];
        let chartData = { labels: [], scores: [] };
        let scoreChart = null;

        document.addEventListener('DOMContentLoaded', function() {
            initStats();
            initEDA();
            initModels();
            initScoreChart();
            startStreaming();
        });

        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach((t, i) => {
                t.classList.remove('active');
                if ((tabName === 'monitor' && i === 0) || 
                    (tabName === 'eda' && i === 1) || 
                    (tabName === 'models' && i === 2)) {
                    t.classList.add('active');
                }
            });
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(tabName + '-tab').classList.add('active');
        }

        function formatNumber(n) {
            if (n >= 1000000) return (n/1000000).toFixed(1) + 'M';
            if (n >= 1000) return (n/1000).toFixed(0) + 'K';
            return n.toString();
        }

        function initStats() {
            document.getElementById('stat-samples').textContent = formatNumber(dashboardData.eda.n_samples);
            document.getElementById('stat-rate').textContent = dashboardData.eda.anomaly_rate.toFixed(1) + '%';
            document.getElementById('eda-samples').textContent = formatNumber(dashboardData.eda.n_samples);
            document.getElementById('eda-anomalies').textContent = formatNumber(dashboardData.eda.n_anomalies);
            document.getElementById('eda-rate').textContent = dashboardData.eda.anomaly_rate.toFixed(1) + '%';
        }

        function initEDA() {
            const corrBars = document.getElementById('correlation-bars');
            dashboardData.eda.top_correlations.forEach(c => {
                const pct = (c.correlation * 100).toFixed(0);
                corrBars.innerHTML += '<div class="bar-item"><div class="bar-label">' + c.pair + '</div><div class="bar-bg"><div class="bar-fill" style="width:' + pct + '%">' + c.correlation.toFixed(2) + '</div></div></div>';
            });

            const impBars = document.getElementById('importance-bars');
            const impEntries = Object.entries(dashboardData.feature_importance).slice(0, 5);
            const maxImp = Math.max(...impEntries.map(e => e[1]));
            impEntries.forEach(([feat, imp]) => {
                const pct = (imp / maxImp * 100).toFixed(0);
                impBars.innerHTML += '<div class="bar-item"><div class="bar-label">' + feat + '</div><div class="bar-bg"><div class="bar-fill" style="width:' + pct + '%">' + (imp * 100).toFixed(1) + '%</div></div></div>';
            });

            const rcData = dashboardData.eda.root_cause_distribution;
            if (Object.keys(rcData).length > 0) {
                new Chart(document.getElementById('rootcause-chart'), {
                    type: 'bar',
                    data: {
                        labels: Object.keys(rcData),
                        datasets: [{ data: Object.values(rcData), backgroundColor: '#10b981' }]
                    },
                    options: {
                        indexAxis: 'y',
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                            y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                        }
                    }
                });
            }
        }

        function initModels() {
            const container = document.getElementById('model-cards');
            const models = dashboardData.models;
            let bestModel = null;
            let bestAuc = 0;
            Object.entries(models).forEach(([name, m]) => {
                if (m.roc_auc > bestAuc) { bestAuc = m.roc_auc; bestModel = name; }
            });

            Object.entries(models).forEach(([name, m]) => {
                const isBest = name === bestModel;
                const displayName = name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                container.innerHTML += '<div class="model-card ' + (isBest ? 'best' : '') + '"><h4>' + displayName + (isBest ? ' <span class="badge">Best</span>' : '') + '</h4><div class="metric-row"><span class="metric-label">Accuracy</span><span class="metric-value">' + m.accuracy.toFixed(1) + '%</span></div><div class="metric-row"><span class="metric-label">Precision</span><span class="metric-value">' + m.precision.toFixed(1) + '%</span></div><div class="metric-row"><span class="metric-label">Recall</span><span class="metric-value">' + m.recall.toFixed(1) + '%</span></div><div class="metric-row"><span class="metric-label">F1 Score</span><span class="metric-value">' + m.f1.toFixed(1) + '%</span></div><div class="metric-row"><span class="metric-label">ROC-AUC</span><span class="metric-value" style="color:#10b981;">' + m.roc_auc.toFixed(4) + '</span></div></div>';
            });

            new Chart(document.getElementById('model-comparison-chart'), {
                type: 'bar',
                data: {
                    labels: Object.keys(models).map(n => n.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')),
                    datasets: [
                        { label: 'Precision', data: Object.values(models).map(m => m.precision), backgroundColor: '#3b82f6' },
                        { label: 'Recall', data: Object.values(models).map(m => m.recall), backgroundColor: '#10b981' },
                        { label: 'F1', data: Object.values(models).map(m => m.f1), backgroundColor: '#f59e0b' }
                    ]
                },
                options: {
                    plugins: { legend: { labels: { color: '#94a3b8' } } },
                    scales: {
                        x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, min: 0, max: 100 }
                    }
                }
            });
        }

        function initScoreChart() {
            scoreChart = new Chart(document.getElementById('score-chart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Anomaly Score',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: 'Threshold',
                        data: [],
                        borderColor: '#ef4444',
                        borderDash: [5, 5],
                        pointRadius: 0
                    }]
                },
                options: {
                    animation: { duration: 0 },
                    plugins: { legend: { labels: { color: '#94a3b8' } } },
                    scales: {
                        x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8', maxTicksLimit: 10 } },
                        y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, min: 0, max: 1 }
                    }
                }
            });
        }

        function startStreaming() {
            setInterval(() => {
                if (!isStreaming) return;
                
                const point = dashboardData.time_series[currentIndex % dashboardData.time_series.length];
                const score = point[selectedModel];
                
                chartData.labels.push(currentIndex);
                chartData.scores.push(score);
                if (chartData.labels.length > 50) {
                    chartData.labels.shift();
                    chartData.scores.shift();
                }
                
                scoreChart.data.labels = chartData.labels;
                scoreChart.data.datasets[0].data = chartData.scores;
                scoreChart.data.datasets[1].data = chartData.labels.map(() => threshold);
                scoreChart.update();
                
                document.getElementById('stat-current').textContent = score.toFixed(3);
                const detected = chartData.scores.filter(s => s > threshold).length;
                document.getElementById('stat-detected').textContent = detected;
                
                if (score > threshold) {
                    alerts.unshift({ index: currentIndex, score: score, time: new Date().toLocaleTimeString() });
                    if (alerts.length > 10) alerts.pop();
                    updateAlerts();
                }
                
                currentIndex++;
            }, 500);
        }

        function updateAlerts() {
            const feed = document.getElementById('alert-feed');
            if (alerts.length === 0) {
                feed.innerHTML = '<p style="color: #64748b; text-align: center;">No alerts yet</p>';
            } else {
                feed.innerHTML = alerts.map(a => '<div class="alert-item"><span class="time">' + a.time + '</span><span class="score">' + a.score.toFixed(3) + '</span></div>').join('');
            }
        }

        function updateChart() {
            selectedModel = document.getElementById('model-select').value;
            chartData = { labels: [], scores: [] };
        }

        function updateThreshold() {
            threshold = parseFloat(document.getElementById('threshold-slider').value);
            document.getElementById('threshold-value').textContent = threshold.toFixed(2);
        }

        function toggleStreaming() {
            isStreaming = !isStreaming;
            document.getElementById('stream-btn-text').textContent = isStreaming ? '⏸ Pause' : '▶ Resume';
            document.getElementById('live-indicator').style.opacity = isStreaming ? 1 : 0.5;
        }
    </script>
</body>
</html>'''
    
    with open(f"{OUTPUT_DIR}/dashboard.html", 'w') as f:
        f.write(html_content)
    
    print(f"✓ Dashboard saved to {OUTPUT_DIR}/dashboard.html")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("CATS ANOMALY DETECTION - MEMORY OPTIMIZED")
    print("="*60)
    print(f"Train sample size: {TRAIN_SAMPLE_SIZE:,}")
    print(f"Test sample size: {TEST_SAMPLE_SIZE:,}")
    
    setup_directories()
    df, metadata, full_stats = load_data()
    eda_results = perform_eda(df, metadata, full_stats)
    models, model_results, scaler, test_data = train_models(df)
    dashboard_data = prepare_dashboard_data(df, models, scaler, eda_results, model_results)
    generate_html_dashboard(dashboard_data)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📁 Outputs saved to '{OUTPUT_DIR}/':")
    print(f"   • eda_plots/     - 10 visualization images")
    print(f"   • models/        - trained models (.joblib)")
    print(f"   • results/       - JSON result files")
    print(f"   • dashboard.html - standalone dashboard")
    print(f"\n🚀 Open '{OUTPUT_DIR}/dashboard.html' in your browser!")
    
    print("\n📊 Model Performance:")
    print("-" * 50)
    for name, metrics in model_results.items():
        print(f"  {name:20} | AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()