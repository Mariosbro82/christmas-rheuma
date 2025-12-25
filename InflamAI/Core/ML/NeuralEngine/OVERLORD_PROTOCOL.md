# OVERLORD PROTOCOL: NEURAL SUPREMACY

## GLOBAL STATUS
CURRENT_PHASE: 3
BEST_ACCURACY: 0.00
CURRENT_ITERATION: 2
LAST_CRITIQUE: Data generation failed - check vectorization
LAST_UPDATE: 2025-11-21T00:00:00Z

## ARCHITECTURE REQUIREMENTS
- **Framework:** PyTorch with MPS (Apple Silicon) acceleration
- **Data Scale:** 2.1 Million rows minimum
- **Features:** Time-series sliding window (lookback 30 days → predict 7 days)
- **Validation:** Independent Test Set (20% split, never seen during training)
- **Target Metrics:** Accuracy > 92%, F1-Score > 0.85, Latency < 50ms

## PHASES OF DOMINATION

### PHASE 1: THE FOUNDATION (DATA ENGINE) ✅ ACTIVE
- **Objective:** Generate massive, realistic synthetic dataset
- **Logic:** Mathematical correlations (Pearson), seasonality, and noise injection
- **Constraint:** Use pandas and numpy vectorization. Generating 2.1M rows must take < 2 minutes
- **Output:** `data/huge_training_set.parquet`
- **Status:** IN PROGRESS
- **Metrics:**
  - Rows Generated: 0
  - Generation Time: N/A
  - Correlation Strength: N/A

### PHASE 2: THE ARCHITECT (MODEL DESIGN)
- **Objective:** Create class-based Model structure (`NeuralFlareNet`)
- **Complexity:** LSTM → Transformer (auto-upgrade if stuck)
- **Loss Function:** Focal Loss for class imbalance
- **Optimizer:** AdamW with Cosine Annealing Scheduler
- **Status:** PENDING
- **Metrics:**
  - Model Parameters: N/A
  - Architecture: N/A
  - Memory Usage: N/A

### PHASE 3: THE GAUNTLET (TRAINING LOOP)
- **Action:** Train for epochs with self-correction
- **Self-Correction Rules:**
  - If Overfitting (Train Loss << Val Loss): Increase Dropout, Add L2
  - If Underfitting (Train Loss > 0.3): Add Layers, Increase Capacity
  - If Plateau (5 epochs no improvement): Switch Architecture
- **Stop Condition:** Validation Accuracy > 92% AND F1-Score > 0.85
- **Status:** PENDING
- **Metrics:**
  - Current Epoch: N/A
  - Train Loss: N/A
  - Val Loss: N/A
  - Val Accuracy: N/A
  - Val F1: N/A

### PHASE 4: THE INQUISITOR (RIGOROUS EVALUATION)
- **Action:** Load best model, test on new generated dataset
- **Metrics:** Confusion Matrix, ROC-AUC, Precision-Recall
- **Visualization:** Generate `evaluation_results.png`
- **Status:** PENDING
- **Metrics:**
  - Test Accuracy: N/A
  - Test F1: N/A
  - ROC-AUC: N/A
  - Inference Time: N/A

### PHASE 5: DEPLOYMENT PREP
- **Action:** Convert PyTorch to CoreML (.mlpackage)
- **Integration:** Swift wrapper for iOS app
- **Status:** PENDING
- **Metrics:**
  - Model Size (MB): N/A
  - iOS Inference Time: N/A
  - Memory Footprint: N/A

## EXECUTION RULES
1. **READ** the last 50 lines of `logs/training_log.txt`
2. **CRITIQUE** previous attempt (too slow? diverging? overfitting?)
3. **IMPROVE** code based on critique
4. **EXECUTE** the next phase or retry current
5. **UPDATE** this file with new metrics immediately

## SELF-IMPROVEMENT STRATEGIES
- **Stuck at 80-85% accuracy:** Switch from LSTM to Transformer
- **Stuck at 85-90% accuracy:** Implement attention mechanisms
- **High variance:** Add dropout, reduce model complexity
- **High bias:** Add layers, increase hidden dimensions
- **Slow convergence:** Increase learning rate, use warm restarts
- **Exploding gradients:** Gradient clipping, reduce learning rate
- **Class imbalance issues:** Adjust focal loss gamma parameter

## ITERATION HISTORY
| Iteration | Phase | Accuracy | F1 Score | Duration | Failure Reason |
|-----------|-------|----------|----------|----------|----------------|
| 0         | 1     | N/A      | N/A      | N/A      | Initial setup  |

## HYPERPARAMETER EVOLUTION
```json
{
  "current": {
    "learning_rate": 0.001,
    "batch_size": 256,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "weight_decay": 0.0001,
    "focal_gamma": 2.0,
    "architecture": "LSTM"
  },
  "best": null
}
```

## CRITICAL OBSERVATIONS
- None yet (first run)

## NEXT ACTION
Generate 2.1M rows of synthetic AS patient data with realistic correlations

---
*Protocol Version: 1.0 | Last Human Intervention: Setup*