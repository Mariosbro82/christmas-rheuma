# 🔥 OVERLORD Neural Engine

## Autonomous ML Training System for AS Flare Prediction

This is a fully autonomous machine learning system that trains neural networks to predict Ankylosing Spondylitis (AS) flares. It runs continuously, self-corrects, and automatically switches architectures until it achieves medical-grade performance (>92% accuracy, >0.85 F1 score).

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- macOS with Apple Silicon (M1/M2) or CUDA-capable GPU
- 16GB+ RAM recommended
- 10GB+ free disk space

### Installation

```bash
# Navigate to Neural Engine directory
cd /Users/fabianharnisch/Documents/Rheuma-app/InflamAI/Core/ML/NeuralEngine/

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x overlord.sh
```

### Launch Autonomous Training

```bash
# Start the OVERLORD system (runs indefinitely)
./overlord.sh &

# Monitor progress
tail -f logs/overlord_log.txt

# Check training metrics
tail -f logs/training_log.txt
```

### Manual Execution (for testing)

```bash
# Generate data
python src/data_generator.py

# Train model
python src/trainer.py

# Evaluate performance
python evaluator.py --accuracy 0.93 --f1 0.86 --latency 45

# Export to CoreML
python src/coreml_exporter.py
```

## 📊 System Architecture

### Components

1. **OVERLORD_PROTOCOL.md** - Central control file tracking phases and metrics
2. **overlord.sh** - Bash orchestrator for autonomous execution
3. **evaluator.py** - Ruthless performance grader (fail if score < 80/100)
4. **src/data_generator.py** - Generates 2.1M rows of synthetic AS patient data
5. **src/neural_flare_net.py** - LSTM/Transformer architectures with auto-switching
6. **src/trainer.py** - Self-correcting training loop with hyperparameter tuning
7. **src/coreml_exporter.py** - Converts to iOS-compatible CoreML format

### Training Phases

1. **Phase 1: Data Generation** - Create massive synthetic dataset with realistic correlations
2. **Phase 2: Architecture Design** - Initialize LSTM or Transformer model
3. **Phase 3: Training** - Self-correcting loop with automatic adjustments
4. **Phase 4: Evaluation** - Rigorous testing against hard metrics
5. **Phase 5: Deployment** - CoreML conversion for iOS integration

## 🧠 Self-Correction Mechanisms

The system automatically detects and corrects:

- **Overfitting**: Increases dropout, adds L2 regularization
- **Underfitting**: Increases model capacity, adds layers
- **Plateau**: Reduces learning rate, switches architectures
- **Divergence**: Restarts with lower learning rate
- **Class Imbalance**: Adjusts focal loss gamma parameter

## 🎯 Target Metrics

- **Accuracy**: > 92%
- **F1 Score**: > 0.85
- **Inference Latency**: < 50ms
- **Model Size**: < 50MB (after CoreML conversion)

## 📈 Monitoring

### Check Current Status
```bash
grep "CURRENT_PHASE" OVERLORD_PROTOCOL.md
grep "BEST_ACCURACY" OVERLORD_PROTOCOL.md
```

### View Training History
```bash
cat logs/training_history.json | jq '.'
```

### Evaluate Convergence
```bash
python evaluator.py --check-convergence
```

## 🍎 iOS Integration

After successful training, the system generates:

1. **NeuralFlareNet.mlpackage** - CoreML model for iOS
2. **NeuralFlarePredictionService.swift** - Ready-to-use Swift wrapper

### Using in iOS App

```swift
// Import the generated service
let predictor = NeuralFlarePredictionService()

// Prepare 30 days of features (35 features per day)
let features: [[Float]] = // ... your feature data

// Get prediction
if let prediction = await predictor.predictFlare(features: features) {
    print("Flare Risk: \(prediction.riskScore)")
    print("Will Flare: \(prediction.willFlare)")
    print("Confidence: \(prediction.confidence)")
}
```

## ⚙️ Configuration

Edit hyperparameters in `src/trainer.py`:

```python
config = {
    'architecture': 'LSTM',  # or 'Transformer'
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'batch_size': 256,
    'learning_rate': 0.001,
    'focal_gamma': 2.0
}
```

## 🔧 Troubleshooting

### Out of Memory
- Reduce `batch_size` in trainer.py
- Decrease `hidden_dim` or `num_layers`

### Training Diverges
- Lower `learning_rate`
- Increase `gradient_clip` value

### Stuck at Low Accuracy
- System will auto-switch from LSTM to Transformer
- Manually trigger switch by updating OVERLORD_PROTOCOL.md

### CoreML Export Fails
```bash
pip install --upgrade coremltools
```

## 📝 Logs

- `logs/overlord_log.txt` - Main system log
- `logs/training_log.txt` - Detailed training output
- `logs/training_history.json` - Metrics per epoch
- `logs/evaluation_log.json` - Performance evaluations

## 🛑 Stopping the System

```bash
# Find the process
ps aux | grep overlord.sh

# Kill it
kill -9 [PID]
```

## ⚠️ Important Notes

1. **This generates SYNTHETIC data** - Not for real medical use without validation
2. **Resource Intensive** - Will use significant CPU/GPU resources
3. **Autonomous** - Runs indefinitely until target metrics achieved
4. **Parallel Development** - Doesn't interfere with existing statistical approach

## 🏆 Success Criteria

The system considers itself successful when:
- Validation Accuracy > 92%
- F1 Score > 0.85
- Test set maintains similar performance
- CoreML model < 50MB
- Inference time < 50ms on iPhone

## 📊 Current Best Results

Check `OVERLORD_PROTOCOL.md` for latest metrics.

---

**OVERLORD Neural Engine v1.0** - Autonomous ML for Medical Excellence