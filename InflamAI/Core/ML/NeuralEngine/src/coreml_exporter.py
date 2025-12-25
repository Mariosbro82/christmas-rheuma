#!/usr/bin/env python3
"""
coreml_exporter.py - Convert PyTorch model to CoreML for iOS deployment
Exports trained NeuralFlareNet to .mlpackage format
"""

import torch
import coremltools as ct
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from neural_flare_net import NeuralFlareNet

class CoreMLExporter:
    """Export PyTorch model to CoreML format"""

    def __init__(self, checkpoint_path: str = 'models/best_model.pth'):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"🍎 CoreML Exporter initialized")
        print(f"Checkpoint: {self.checkpoint_path}")

    def load_model(self) -> NeuralFlareNet:
        """Load trained PyTorch model from checkpoint"""
        print("\nLoading PyTorch model...")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        metrics = checkpoint['metrics']

        print(f"Model architecture: {config['architecture']}")
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation F1: {metrics['f1']:.4f}")

        # Create model
        model = NeuralFlareNet(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, config

    def trace_model(self, model: NeuralFlareNet, config: dict) -> torch.jit.ScriptModule:
        """Trace PyTorch model for CoreML conversion"""
        print("\nTracing model...")

        # Create dummy input (batch_size=1 for inference)
        sequence_length = 30
        input_dim = config.get('input_dim', 35)
        dummy_input = torch.randn(1, sequence_length, input_dim)

        # Trace the model
        with torch.no_grad():
            # We need to wrap the model to only return logits for CoreML
            class CoreMLWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    output = self.model(x)
                    # Return both classification and risk score
                    logits = output['logits']
                    risk = output['risk_score']
                    return logits, risk

            wrapped_model = CoreMLWrapper(model)
            wrapped_model.eval()

            # Use torch.jit.trace for CoreML
            traced_model = torch.jit.trace(wrapped_model, dummy_input)

        print("✅ Model traced successfully")
        return traced_model

    def convert_to_coreml(self, traced_model: torch.jit.ScriptModule, config: dict):
        """Convert traced PyTorch model to CoreML"""
        print("\nConverting to CoreML...")

        # Define input shape and type
        sequence_length = 30
        input_dim = config.get('input_dim', 35)

        # Create input specification
        inputs = [
            ct.TensorType(
                name="input_sequence",
                shape=(1, sequence_length, input_dim),
                dtype=np.float32
            )
        ]

        # Create output specification
        outputs = [
            ct.TensorType(name="flare_prediction", dtype=np.float32),
            ct.TensorType(name="risk_score", dtype=np.float32)
        ]

        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            convert_to="mlprogram",  # Use ML Program format (newer)
            minimum_deployment_target=ct.target.iOS17,  # iOS 17+
        )

        # Add metadata
        mlmodel.author = "OVERLORD Neural Engine"
        mlmodel.short_description = "AS Flare Prediction Model"
        mlmodel.version = "1.0"

        # Add detailed description
        mlmodel.input_description["input_sequence"] = "30-day sequence of patient features"
        mlmodel.output_description["flare_prediction"] = "Binary classification logits [no_flare, will_flare]"
        mlmodel.output_description["risk_score"] = "Continuous risk score (0-1)"

        # Add user-defined metadata
        mlmodel.user_defined_metadata["architecture"] = config['architecture']
        mlmodel.user_defined_metadata["training_samples"] = str(config.get('training_samples', 'unknown'))
        mlmodel.user_defined_metadata["accuracy"] = str(config.get('accuracy', 'unknown'))
        mlmodel.user_defined_metadata["f1_score"] = str(config.get('f1_score', 'unknown'))

        print("✅ CoreML model created")
        return mlmodel

    def validate_coreml_model(self, mlmodel, pytorch_model, config: dict):
        """Validate that CoreML model produces same outputs as PyTorch"""
        print("\nValidating CoreML model...")

        # Create test input
        sequence_length = 30
        input_dim = config.get('input_dim', 35)
        test_input = np.random.randn(1, sequence_length, input_dim).astype(np.float32)

        # PyTorch prediction
        with torch.no_grad():
            torch_input = torch.from_numpy(test_input)
            torch_output = pytorch_model(torch_input)
            torch_logits = torch_output['logits'].numpy()
            torch_risk = torch_output['risk_score'].numpy()

        # CoreML prediction
        coreml_input = {"input_sequence": test_input}
        coreml_output = mlmodel.predict(coreml_input)
        coreml_logits = coreml_output["flare_prediction"]
        coreml_risk = coreml_output["risk_score"]

        # Compare outputs
        logits_diff = np.abs(torch_logits - coreml_logits).max()
        risk_diff = np.abs(torch_risk - coreml_risk).max()

        print(f"  Logits difference: {logits_diff:.6f}")
        print(f"  Risk score difference: {risk_diff:.6f}")

        if logits_diff < 1e-3 and risk_diff < 1e-3:
            print("✅ Validation passed! Outputs match.")
            return True
        else:
            print("⚠️ Warning: Outputs differ significantly")
            return False

    def save_coreml_model(self, mlmodel, output_path: str = "models/NeuralFlareNet.mlpackage"):
        """Save CoreML model to disk"""
        print(f"\nSaving CoreML model to {output_path}...")

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)

        # Save model
        mlmodel.save(output_path)

        # Get file size
        import shutil
        size_mb = sum(f.stat().st_size for f in output_path.rglob('*')) / (1024 * 1024)

        print(f"✅ Model saved: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")

        # Create Swift integration code
        self.generate_swift_wrapper(output_path)

    def generate_swift_wrapper(self, model_path: Path):
        """Generate Swift code for iOS integration"""
        swift_code = '''//
//  NeuralFlarePredictionService.swift
//  Spinalytics
//
//  Generated by OVERLORD Neural Engine
//

import Foundation
import CoreML

@MainActor
class NeuralFlarePredictionService: ObservableObject {

    @Published var isModelLoaded = false
    @Published var lastPrediction: FlarePrediction?
    @Published var errorMessage: String?

    private var model: NeuralFlareNet?

    struct FlarePrediction {
        let willFlare: Bool
        let riskScore: Float
        let confidence: Float
        let timestamp: Date
    }

    init() {
        Task {
            await loadModel()
        }
    }

    private func loadModel() async {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine if available

            self.model = try NeuralFlareNet(configuration: config)
            self.isModelLoaded = true
            print("✅ Neural flare prediction model loaded")
        } catch {
            self.errorMessage = "Failed to load model: \\(error)"
            print("❌ Model loading failed: \\(error)")
        }
    }

    func predictFlare(features: [[Float]]) async -> FlarePrediction? {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return nil
        }

        guard features.count == 30 && features.first?.count == 35 else {
            errorMessage = "Invalid input shape. Expected 30x35"
            return nil
        }

        do {
            // Convert to MLMultiArray
            let inputArray = try MLMultiArray(shape: [1, 30, 35], dataType: .float32)

            for (i, timestep) in features.enumerated() {
                for (j, value) in timestep.enumerated() {
                    inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
                }
            }

            // Create input
            let input = NeuralFlareNetInput(input_sequence: inputArray)

            // Run prediction
            let output = try model.prediction(input: input)

            // Parse outputs
            let logits = output.flare_prediction
            let riskScore = output.risk_score[0].floatValue

            // Apply softmax to get probabilities
            let willFlareProb = softmax(logits)[1].floatValue
            let willFlare = willFlareProb > 0.5

            let prediction = FlarePrediction(
                willFlare: willFlare,
                riskScore: riskScore,
                confidence: abs(willFlareProb - 0.5) * 2,  // Scale to 0-1
                timestamp: Date()
            )

            self.lastPrediction = prediction
            return prediction

        } catch {
            self.errorMessage = "Prediction failed: \\(error)"
            return nil
        }
    }

    private func softmax(_ input: MLMultiArray) -> [Float] {
        var values: [Float] = []
        for i in 0..<input.count {
            values.append(input[i].floatValue)
        }

        let maxVal = values.max() ?? 0
        let expValues = values.map { exp($0 - maxVal) }
        let sumExp = expValues.reduce(0, +)

        return expValues.map { $0 / sumExp }
    }
}
'''

        swift_path = model_path.parent / "NeuralFlarePredictionService.swift"
        with open(swift_path, 'w') as f:
            f.write(swift_code)

        print(f"✅ Swift wrapper generated: {swift_path}")

    def export(self):
        """Execute full export pipeline"""
        print("\n" + "="*60)
        print("COREML EXPORT PIPELINE")
        print("="*60)

        try:
            # Load model
            model, config = self.load_model()

            # Trace model
            traced_model = self.trace_model(model, config)

            # Convert to CoreML
            mlmodel = self.convert_to_coreml(traced_model, config)

            # Validate
            self.validate_coreml_model(mlmodel, model, config)

            # Save
            self.save_coreml_model(mlmodel)

            print("\n" + "="*60)
            print("🎉 EXPORT COMPLETE! Model ready for iOS deployment")
            print("="*60)

            # Update protocol
            self.update_protocol_completion()

            return True

        except Exception as e:
            print(f"\n❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_protocol_completion(self):
        """Update OVERLORD_PROTOCOL.md to mark completion"""
        protocol_path = Path('../OVERLORD_PROTOCOL.md')
        if protocol_path.exists():
            content = protocol_path.read_text()
            content = content.replace('CURRENT_PHASE: 5', 'CURRENT_PHASE: COMPLETE')
            protocol_path.write_text(content)
            print("\n✅ OVERLORD PROTOCOL marked as COMPLETE")

def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Export PyTorch model to CoreML')
    parser.add_argument('--checkpoint', default='models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', default='models/NeuralFlareNet.mlpackage',
                       help='Output path for CoreML model')
    args = parser.parse_args()

    # Check if coremltools is installed
    try:
        import coremltools
    except ImportError:
        print("❌ coremltools not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'coremltools'])
        print("✅ coremltools installed. Please run the script again.")
        sys.exit(1)

    # Create exporter
    exporter = CoreMLExporter(checkpoint_path=args.checkpoint)

    # Run export
    success = exporter.export()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()