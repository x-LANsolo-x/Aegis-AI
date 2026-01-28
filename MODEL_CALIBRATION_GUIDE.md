# Model Calibration & Improvement - Implementation Guide

## Feature: D1 - Model Calibration & Threshold Tuning

**Priority:** High  
**Estimated Time:** 2-3 weeks  
**Dependencies:** Real ASVspoof training complete, validation dataset available

---

## 1. Requirements

### Functional Requirements
- [ ] Calibrate model confidence scores (match accuracy)
- [ ] Optimize decision thresholds for verdicts
- [ ] Measure and minimize false positive/negative rates
- [ ] Support multiple operating points (precision vs recall)
- [ ] Provide calibration metrics and visualizations
- [ ] Allow threshold customization per use case

### Performance Targets
- [ ] Equal Error Rate (EER) <10%
- [ ] False Positive Rate <5% at 95% True Positive Rate
- [ ] Calibration error <5% (Expected Calibration Error)
- [ ] AUC-ROC >0.95

### Technical Requirements
- [ ] Platt scaling or temperature scaling
- [ ] Threshold tuning on validation set
- [ ] Cross-validation for robustness
- [ ] Metrics: EER, AUC, Precision, Recall, F1
- [ ] Reliability diagrams

---

## 2. Calibration Concepts

### What is Calibration?

**Problem:** Raw model outputs don't represent true probabilities
- Model says 80% confidence but only correct 60% of the time
- Overconfident or underconfident predictions

**Solution:** Post-processing to align confidence with accuracy
- 80% confidence → actually correct 80% of the time
- Better risk assessment and decision making

### Calibration Methods

**1. Platt Scaling (Logistic Regression)**
- Fit sigmoid to validation set logits
- Simple, fast, works well for binary classification
- `P(y=1|x) = 1 / (1 + exp(A*logit + B))`

**2. Temperature Scaling**
- Divide logits by learned temperature T
- Single parameter, preserves model ranking
- `P(y|x) = softmax(logits / T)`

**3. Isotonic Regression**
- Non-parametric, fits monotonic function
- More flexible, can overfit on small datasets

**Recommended:** Temperature Scaling (simplest, effective)

---

## 3. Implementation Phases

### Phase 1: Evaluation Infrastructure (Week 1, Days 1-3)

**Tasks:**
- [ ] Create comprehensive evaluation script
- [ ] Implement all metrics (EER, AUC, precision, recall)
- [ ] Generate reliability diagrams
- [ ] Add ROC curve plotting

**Code:**

```python
# ml/evaluation/metrics.py (NEW)

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Comprehensive model evaluation and calibration."""
    
    def __init__(self, predictions, labels, logits=None):
        """
        Args:
            predictions: Predicted probabilities (0-1)
            labels: True labels (0 or 1)
            logits: Raw model outputs (optional)
        """
        self.predictions = np.array(predictions)
        self.labels = np.array(labels)
        self.logits = np.array(logits) if logits is not None else None
    
    def compute_eer(self):
        """Compute Equal Error Rate."""
        fpr, tpr, thresholds = roc_curve(self.labels, self.predictions)
        fnr = 1 - tpr
        
        # EER is where FPR = FNR
        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        
        return {
            "eer": float(eer),
            "eer_threshold": float(eer_threshold)
        }
    
    def compute_auc_roc(self):
        """Compute AUC-ROC."""
        fpr, tpr, _ = roc_curve(self.labels, self.predictions)
        roc_auc = auc(fpr, tpr)
        
        return {
            "auc_roc": float(roc_auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
    
    def compute_precision_recall(self):
        """Compute precision-recall metrics."""
        precision, recall, thresholds = precision_recall_curve(
            self.labels, self.predictions
        )
        
        # Find F1-optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
        
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "best_f1_threshold": float(best_threshold),
            "best_f1_score": float(np.max(f1_scores))
        }
    
    def compute_calibration_error(self, n_bins=10):
        """Compute Expected Calibration Error (ECE)."""
        
        prob_true, prob_pred = calibration_curve(
            self.labels, 
            self.predictions, 
            n_bins=n_bins,
            strategy='uniform'
        )
        
        # ECE is weighted average of |accuracy - confidence|
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            in_bin = (self.predictions >= bin_edges[i]) & \
                     (self.predictions < bin_edges[i + 1])
            bin_counts[i] = np.sum(in_bin)
        
        bin_weights = bin_counts / len(self.predictions)
        ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
        
        return {
            "ece": float(ece),
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "n_bins": n_bins
        }
    
    def plot_reliability_diagram(self, save_path=None):
        """Plot reliability diagram (calibration curve)."""
        
        prob_true, prob_pred = calibration_curve(
            self.labels,
            self.predictions,
            n_bins=10
        )
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Plot model calibration
        ax.plot(prob_pred, prob_true, 'o-', label='Model Calibration')
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability')
        ax.set_title('Reliability Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve."""
        
        fpr, tpr, _ = roc_curve(self.labels, self.predictions)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        
        report = {
            "n_samples": len(self.labels),
            "n_positive": int(np.sum(self.labels == 1)),
            "n_negative": int(np.sum(self.labels == 0)),
        }
        
        report.update(self.compute_eer())
        report.update(self.compute_auc_roc())
        report.update(self.compute_precision_recall())
        report.update(self.compute_calibration_error())
        
        return report
```

**Testing:**
```python
# ml/tests/test_metrics.py

def test_compute_eer():
    # Perfect predictions
    evaluator = ModelEvaluator([0.1, 0.9], [0, 1])
    result = evaluator.compute_eer()
    assert result["eer"] < 0.01
    
    # Random predictions
    evaluator = ModelEvaluator([0.5] * 100, [0, 1] * 50)
    result = evaluator.compute_eer()
    assert 0.4 < result["eer"] < 0.6

def test_calibration_error():
    # Perfect calibration
    evaluator = ModelEvaluator([0.2, 0.8], [0, 1])
    result = evaluator.compute_calibration_error()
    assert result["ece"] < 0.1
```

---

### Phase 2: Temperature Scaling (Week 1, Days 4-5)

**Tasks:**
- [ ] Implement temperature scaling
- [ ] Optimize temperature on validation set
- [ ] Compare before/after calibration
- [ ] Save calibrated model

**Code:**

```python
# ml/calibration/temperature_scaling.py (NEW)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    Divides logits by learned temperature parameter.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        """Apply temperature scaling."""
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Optimize temperature on validation set.
        
        Args:
            logits: Model outputs (N, num_classes)
            labels: True labels (N,)
            lr: Learning rate
            max_iter: Max optimization iterations
        """
        
        logits = torch.FloatTensor(logits)
        labels = torch.LongTensor(labels)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        return self.temperature.item()

def calibrate_model(model, val_loader, device='cpu'):
    """
    Calibrate model using temperature scaling.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to use
    
    Returns:
        temperature: Optimized temperature value
        calibrated_model: Model with temperature scaling
    """
    
    model.eval()
    
    # Collect logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            logits = model(features)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Fit temperature
    temp_scaler = TemperatureScaling()
    temperature = temp_scaler.fit(all_logits, all_labels)
    
    print(f"Optimized temperature: {temperature:.4f}")
    
    # Create calibrated model wrapper
    class CalibratedModel(nn.Module):
        def __init__(self, base_model, temperature):
            super().__init__()
            self.model = base_model
            self.temperature = temperature
        
        def forward(self, x):
            logits = self.model(x)
            return logits / self.temperature
    
    calibrated = CalibratedModel(model, temperature)
    
    return temperature, calibrated
```

---

Continue in next message...
