"""
Transformer-Based Anomaly Detection for CATS Dataset
=====================================================

Implementation based on:
"A Transformer-Based Framework for Anomaly Detection in Multivariate Time Series"
Folger et al., CLOUD COMPUTING 2025

Key aspects from the paper:
1. Vanilla Transformer encoder (no decoder)
2. Semi-supervised hyperparameter tuning (validation includes some anomalies)
3. Final training on normal data only (unsupervised)
4. Focal Loss for class imbalance
5. Mean pooling for temporal aggregation
6. Optuna for hyperparameter optimization

Target metrics: ROC-AUC ~0.999, F1 ~0.97, Recall ~0.985
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, precision_recall_curve, auc
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration matching the paper's setup."""
    
    # Data paths
    DATA_CSV: str = "data.csv"
    META_CSV: str = "metadata.csv"
    TIMESTAMP_COL: str = "timestamp"
    
    # CATS dataset structure
    NOMINAL_ROWS: int = 1_000_000  # First 1M rows are purely normal
    
    # Windowing (paper uses sliding windows)
    WINDOW_SIZE: int = 128  # Sequence length
    STRIDE_TRAIN: int = 64
    STRIDE_EVAL: int = 32
    
    # Model architecture (paper's optimal values)
    MODEL_DIM: int = 128
    N_HEADS: int = 8
    N_ENCODER_LAYERS: int = 3
    DROPOUT: float = 0.2
    DIM_FEEDFORWARD: int = 512  # 4x model_dim
    
    # Training
    BATCH_SIZE: int = 128
    EPOCHS: int = 50
    LEARNING_RATE: float = 2e-5  # Paper uses very low LR
    WEIGHT_DECAY: float = 0.00036
    PATIENCE: int = 10
    
    # Focal Loss parameters
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0
    
    # Optuna
    N_OPTUNA_TRIALS: int = 50
    OPTUNA_TIMEOUT: int = 3600  # 1 hour max
    
    # Output
    OUT_DIR: str = "./outputs_transformer_paper"
    SEED: int = 42
    
    # Device
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def convert_to_native(obj: Any) -> Any:
    """Convert numpy/torch types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_metadata_intervals(meta_csv: str, timestamps: np.ndarray) -> List[Tuple[int, int]]:
    """Load anomaly intervals from metadata file."""
    meta = pd.read_csv(meta_csv)
    
    # Find start/end columns
    start_col = next((c for c in ["start_time", "start", "anomaly_start"] if c in meta.columns), None)
    end_col = next((c for c in ["end_time", "end", "anomaly_end"] if c in meta.columns), None)
    
    if not start_col or not end_col:
        raise ValueError("Metadata missing start/end time columns")
    
    intervals = []
    ts = np.array(timestamps, dtype="datetime64[ns]")
    n = len(ts)
    
    for _, row in meta.iterrows():
        try:
            s_time = pd.to_datetime(row[start_col])
            e_time = pd.to_datetime(row[end_col])
            if pd.isna(s_time) or pd.isna(e_time):
                continue
            
            s_idx = int(np.searchsorted(ts, np.datetime64(s_time)))
            e_idx = int(np.searchsorted(ts, np.datetime64(e_time), side='right'))
            
            s_idx = max(0, min(s_idx, n))
            e_idx = max(0, min(e_idx, n))
            
            if s_idx < e_idx:
                intervals.append((s_idx, e_idx))
        except Exception:
            continue
    
    return intervals


def create_anomaly_mask(n_rows: int, intervals: List[Tuple[int, int]]) -> np.ndarray:
    """Create boolean mask where True = anomalous timestep."""
    mask = np.zeros(n_rows, dtype=bool)
    for s, e in intervals:
        mask[s:e] = True
    return mask


# ============================================================================
# DATASET
# ============================================================================

class CATSDataset(Dataset):
    """
    Dataset for CATS time series with window-level labels.
    
    Label strategy (from paper):
    - A window is labeled anomalous if ANY timestep in the window is anomalous
    - This matches the paper's approach for binary classification
    """
    
    def __init__(
        self,
        X: np.ndarray,
        anomaly_mask: np.ndarray,
        window_size: int,
        indices: List[int],
        label_mode: str = "any"  # "any" or "last"
    ):
        self.X = X
        self.anomaly_mask = anomaly_mask
        self.window_size = window_size
        self.indices = indices
        self.label_mode = label_mode
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        
        x = self.X[start:end]
        
        # Label based on window content
        if self.label_mode == "any":
            # Paper approach: any anomaly in window = positive
            label = float(self.anomaly_mask[start:end].any())
        else:
            # Alternative: label based on last timestep
            label = float(self.anomaly_mask[end - 1])
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


# ============================================================================
# MODEL: TRANSFORMER ENCODER (Paper Architecture)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention Is All You Need'."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerAnomalyDetector(nn.Module):
    """
    Transformer-based anomaly detector following the paper's architecture.
    
    Architecture:
    1. Linear embedding layer
    2. Batch normalization
    3. Positional encoding
    4. Transformer encoder layers
    5. Mean pooling (temporal aggregation)
    6. Classification head with sigmoid
    """
    
    def __init__(
        self,
        n_features: int,
        model_dim: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.model_dim = model_dim
        
        # 1. Linear embedding (project features to model dimension)
        self.embedding = nn.Linear(n_features, model_dim)
        
        # 2. Batch normalization (paper mentions this for training stability)
        self.batch_norm = nn.BatchNorm1d(model_dim)
        
        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, dropout=dropout)
        
        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers
        )
        
        # 5. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
        
        Returns:
            logits: Tensor of shape (batch_size,)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Embed features
        x = self.embedding(x)  # (B, T, model_dim)
        
        # 2. Batch normalization (reshape for BatchNorm1d)
        x = x.transpose(1, 2)  # (B, model_dim, T)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # (B, T, model_dim)
        
        # 3. Add positional encoding
        x = self.pos_encoder(x)
        
        # 4. Transformer encoder
        x = self.transformer_encoder(x)  # (B, T, model_dim)
        
        # 5. Temporal aggregation (mean pooling over time)
        x = x.mean(dim=1)  # (B, model_dim)
        
        # 6. Classification
        logits = self.classifier(x).squeeze(-1)  # (B,)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability scores."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ============================================================================
# FOCAL LOSS (Paper uses this for class imbalance)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    From paper: "Focal Loss addresses this challenge by dynamically down-weighting
    the loss contribution of well-classified normal samples and up-weighting
    the misclassified or harder-to-classify anomalous examples."
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Binary labels (0 or 1)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final loss
        loss = alpha_t * focal_weight * ce_loss
        
        return loss.mean()


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    
    return total_loss / n_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate model and compute metrics."""
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = model.predict_proba(x).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    
    # Find optimal threshold
    thresholds = np.linspace(0.001, 0.999, 200)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    
    # Compute predictions at optimal threshold
    preds = (probs >= best_thr).astype(int)
    
    # Compute all metrics
    metrics = {
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        "f1": float(best_f1),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "threshold": float(best_thr),
        "n_positives": int(labels.sum()),
        "n_samples": int(len(labels)),
        "positive_rate": float(labels.mean())
    }
    
    # PR-AUC
    if len(np.unique(labels)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
        metrics["pr_auc"] = float(auc(recall_curve, precision_curve))
    else:
        metrics["pr_auc"] = 0.0
    
    return metrics


# ============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================

class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer following the paper's approach.
    
    Key insight from paper:
    "The model is mainly trained on normal data, but the validation set 
    contains a few anomalies. This enables the optimization process to 
    favor parameter combinations that effectively detect anomalies."
    """
    
    def __init__(
        self,
        X: np.ndarray,
        anomaly_mask: np.ndarray,
        train_indices: List[int],
        val_indices: List[int],
        n_features: int,
        window_size: int,
        device: torch.device,
        n_trials: int = 50,
        timeout: int = 3600
    ):
        self.X = X
        self.anomaly_mask = anomaly_mask
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.n_features = n_features
        self.window_size = window_size
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        
        self.best_params = None
        self.best_score = 0.0
    
    def objective(self, trial: Trial) -> float:
        """Optuna objective function."""
        
        # Sample hyperparameters (ranges from paper)
        params = {
            "model_dim": trial.suggest_categorical("model_dim", [64, 128, 256]),
            "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
            "n_encoder_layers": trial.suggest_int("n_encoder_layers", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "focal_alpha": trial.suggest_float("focal_alpha", 0.1, 0.5),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
        }
        
        # Ensure model_dim is divisible by n_heads
        if params["model_dim"] % params["n_heads"] != 0:
            return 0.0
        
        # Create datasets
        train_ds = CATSDataset(
            self.X, self.anomaly_mask, self.window_size,
            self.train_indices, label_mode="any"
        )
        val_ds = CATSDataset(
            self.X, self.anomaly_mask, self.window_size,
            self.val_indices, label_mode="any"
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=params["batch_size"],
            shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=params["batch_size"], shuffle=False
        )
        
        # Create model
        model = TransformerAnomalyDetector(
            n_features=self.n_features,
            model_dim=params["model_dim"],
            n_heads=params["n_heads"],
            n_encoder_layers=params["n_encoder_layers"],
            dim_feedforward=params["model_dim"] * 4,
            dropout=params["dropout"]
        ).to(self.device)
        
        # Loss and optimizer
        criterion = FocalLoss(alpha=params["focal_alpha"], gamma=params["focal_gamma"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )
        
        # Train for fewer epochs during HP search
        n_epochs = 10
        best_val_f1 = 0.0
        
        for epoch in range(n_epochs):
            train_epoch(model, train_loader, optimizer, criterion, self.device)
            val_metrics = evaluate(model, val_loader, self.device)
            
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
            
            # Report for pruning
            trial.report(val_metrics["f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_f1
    
    def optimize(self) -> Dict[str, Any]:
        """Run Optuna optimization."""
        
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\nBest F1 Score: {self.best_score:.4f}")
        print("Best Hyperparameters:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        
        return self.best_params


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline following the paper's methodology."""
    
    config = Config()
    
    # Set seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    os.makedirs(config.OUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Transformer-Based Anomaly Detection (Paper Implementation)")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/7] Loading data...")
    
    df = pd.read_csv(config.DATA_CSV, parse_dates=[config.TIMESTAMP_COL])
    timestamps = df[config.TIMESTAMP_COL].values
    
    exclude_cols = {config.TIMESTAMP_COL, "y", "label", "category"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_raw = df[feature_cols].values.astype(np.float32)
    n_rows, n_features = X_raw.shape
    
    print(f"  Total rows: {n_rows:,}")
    print(f"  Features: {n_features}")
    print(f"  Feature columns: {feature_cols}")
    
    # =========================================================================
    # 2. Load Metadata & Create Labels
    # =========================================================================
    print("\n[2/7] Processing metadata...")
    
    intervals = load_metadata_intervals(config.META_CSV, timestamps)
    anomaly_mask = create_anomaly_mask(n_rows, intervals)
    
    print(f"  Anomaly intervals: {len(intervals)}")
    print(f"  Anomalous rows: {anomaly_mask.sum():,} ({100*anomaly_mask.mean():.2f}%)")
    
    # Verify CATS structure
    nominal_anomalies = anomaly_mask[:config.NOMINAL_ROWS].sum()
    print(f"  Anomalies in nominal region (should be 0): {nominal_anomalies}")
    
    # =========================================================================
    # 3. Scale Features
    # =========================================================================
    print("\n[3/7] Scaling features...")
    
    # Fit scaler on nominal (normal) data only
    scaler = StandardScaler()
    scaler.fit(X_raw[:config.NOMINAL_ROWS])
    X = scaler.transform(X_raw).astype(np.float32)
    
    # =========================================================================
    # 4. Create Data Splits
    # =========================================================================
    print("\n[4/7] Creating data splits...")
    
    """
    Paper's approach (semi-supervised):
    - Training: Normal data only (first 1M rows)
    - Validation: Include SOME anomalies (for HP tuning to detect them)
    - Test: Mixed data (normal + anomalies)
    """
    
    # Training indices (normal region only, no anomalies)
    train_end = int(0.8 * config.NOMINAL_ROWS)
    train_indices = list(range(0, train_end - config.WINDOW_SIZE + 1, config.STRIDE_TRAIN))
    
    # Validation indices - KEY: Include some anomalies from mixed region
    # This is the semi-supervised trick from the paper
    val_normal_start = train_end
    val_normal_end = config.NOMINAL_ROWS
    val_normal_indices = list(range(val_normal_start, val_normal_end - config.WINDOW_SIZE + 1, config.STRIDE_TRAIN))
    
    # Add some anomalous windows to validation (from mixed region)
    mixed_start = config.NOMINAL_ROWS
    mixed_end = min(config.NOMINAL_ROWS + 500000, n_rows)  # Use first 500k of mixed
    mixed_indices = list(range(mixed_start, mixed_end - config.WINDOW_SIZE + 1, config.STRIDE_TRAIN))
    
    # Combine for validation (normal + some anomalous)
    val_indices = val_normal_indices + mixed_indices[:len(mixed_indices)//4]  # Add 25% of mixed
    np.random.shuffle(val_indices)
    
    # Test indices (rest of mixed region)
    test_start = mixed_end
    test_indices = list(range(test_start, n_rows - config.WINDOW_SIZE + 1, config.STRIDE_EVAL))
    
    # Count labels in each split
    train_labels = [anomaly_mask[i:i+config.WINDOW_SIZE].any() for i in train_indices]
    val_labels = [anomaly_mask[i:i+config.WINDOW_SIZE].any() for i in val_indices]
    test_labels = [anomaly_mask[i:i+config.WINDOW_SIZE].any() for i in test_indices]
    
    print(f"  Train windows: {len(train_indices):,} (anomaly rate: {np.mean(train_labels):.4f})")
    print(f"  Val windows: {len(val_indices):,} (anomaly rate: {np.mean(val_labels):.4f})")
    print(f"  Test windows: {len(test_indices):,} (anomaly rate: {np.mean(test_labels):.4f})")
    
    # =========================================================================
    # 5. Hyperparameter Optimization (Optional)
    # =========================================================================
    run_optuna = OPTUNA_AVAILABLE and os.environ.get("RUN_OPTUNA", "0") == "1"
    
    if run_optuna:
        print("\n[5/7] Running Optuna hyperparameter optimization...")
        
        optimizer = OptunaOptimizer(
            X=X,
            anomaly_mask=anomaly_mask,
            train_indices=train_indices,
            val_indices=val_indices,
            n_features=n_features,
            window_size=config.WINDOW_SIZE,
            device=config.DEVICE,
            n_trials=config.N_OPTUNA_TRIALS,
            timeout=config.OPTUNA_TIMEOUT
        )
        
        best_params = optimizer.optimize()
        
        # Update config with best params
        config.MODEL_DIM = best_params.get("model_dim", config.MODEL_DIM)
        config.N_HEADS = best_params.get("n_heads", config.N_HEADS)
        config.N_ENCODER_LAYERS = best_params.get("n_encoder_layers", config.N_ENCODER_LAYERS)
        config.DROPOUT = best_params.get("dropout", config.DROPOUT)
        config.LEARNING_RATE = best_params.get("learning_rate", config.LEARNING_RATE)
        config.WEIGHT_DECAY = best_params.get("weight_decay", config.WEIGHT_DECAY)
        config.BATCH_SIZE = best_params.get("batch_size", config.BATCH_SIZE)
        config.FOCAL_ALPHA = best_params.get("focal_alpha", config.FOCAL_ALPHA)
        config.FOCAL_GAMMA = best_params.get("focal_gamma", config.FOCAL_GAMMA)
        
        # Save best params
        with open(os.path.join(config.OUT_DIR, "best_params.json"), "w") as f:
            json.dump(convert_to_native(best_params), f, indent=2)
    else:
        print("\n[5/7] Using paper's optimal hyperparameters (skip Optuna)...")
        print("  To run Optuna: set environment variable RUN_OPTUNA=1")
    
    # =========================================================================
    # 6. Final Training
    # =========================================================================
    print("\n[6/7] Training final model...")
    print(f"  Model dim: {config.MODEL_DIM}")
    print(f"  Heads: {config.N_HEADS}")
    print(f"  Layers: {config.N_ENCODER_LAYERS}")
    print(f"  Dropout: {config.DROPOUT}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    
    # Create datasets
    train_ds = CATSDataset(X, anomaly_mask, config.WINDOW_SIZE, train_indices, label_mode="any")
    val_ds = CATSDataset(X, anomaly_mask, config.WINDOW_SIZE, val_indices, label_mode="any")
    test_ds = CATSDataset(X, anomaly_mask, config.WINDOW_SIZE, test_indices, label_mode="any")
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Create model
    model = TransformerAnomalyDetector(
        n_features=n_features,
        model_dim=config.MODEL_DIM,
        n_heads=config.N_HEADS,
        n_encoder_layers=config.N_ENCODER_LAYERS,
        dim_feedforward=config.MODEL_DIM * 4,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.LEARNING_RATE / 100
    )
    
    # Training loop
    history = {"train_loss": [], "val_f1": [], "val_auc": []}
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE, scheduler
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, config.DEVICE)
        
        # Log
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["roc_auc"])
        
        print(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"Loss: {train_loss:.6f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )
        
        # Early stopping based on F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_state, os.path.join(config.OUT_DIR, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # =========================================================================
    # 7. Final Evaluation
    # =========================================================================
    print("\n[7/7] Final evaluation on test set...")
    
    test_metrics = evaluate(model, test_loader, config.DEVICE)
    
    print("\n" + "=" * 70)
    print("FINAL TEST METRICS")
    print("=" * 70)
    
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Compute confusion matrix
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.DEVICE)
            probs = model.predict_proba(x).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= test_metrics["threshold"]).astype(int)
    
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    # Detection rates
    if cm[1,0] + cm[1,1] > 0:
        anomaly_detection_rate = cm[1,1] / (cm[1,0] + cm[1,1])
        print(f"\nAnomalies detected: {cm[1,1]}/{cm[1,0]+cm[1,1]} ({100*anomaly_detection_rate:.2f}%)")
    
    # =========================================================================
    # Save Artifacts
    # =========================================================================
    print("\nSaving artifacts...")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(config.OUT_DIR, "final_model.pt"))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(config.OUT_DIR, "scaler.pkl"))
    
    # Save metrics
    with open(os.path.join(config.OUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(convert_to_native(test_metrics), f, indent=2)
    
    # Save history
    with open(os.path.join(config.OUT_DIR, "training_history.json"), "w") as f:
        json.dump(convert_to_native(history), f, indent=2)
    
    # Save config
    config_dict = {
        "WINDOW_SIZE": config.WINDOW_SIZE,
        "MODEL_DIM": config.MODEL_DIM,
        "N_HEADS": config.N_HEADS,
        "N_ENCODER_LAYERS": config.N_ENCODER_LAYERS,
        "DROPOUT": config.DROPOUT,
        "LEARNING_RATE": config.LEARNING_RATE,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,
        "BATCH_SIZE": config.BATCH_SIZE,
        "FOCAL_ALPHA": config.FOCAL_ALPHA,
        "FOCAL_GAMMA": config.FOCAL_GAMMA,
        "OPTIMAL_THRESHOLD": test_metrics["threshold"]
    }
    with open(os.path.join(config.OUT_DIR, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\nAll outputs saved to: {config.OUT_DIR}")
    print("=" * 70)
    
    return test_metrics


if __name__ == "__main__":
    main()
