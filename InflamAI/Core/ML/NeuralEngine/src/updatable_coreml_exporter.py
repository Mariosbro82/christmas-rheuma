#!/usr/bin/env python3
"""
updatable_coreml_exporter.py - Export PyTorch Model to Updatable CoreML
Creates on-device learning capable models for iOS
"""

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.neural_network.quantization_utils import quantize_weights
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from neural_flare_net import NeuralFlareNet

class UpdatableCoreMLExporter:
    """Export PyTorch model to updatable CoreML format"""
    
    def __init__(self, checkpoint_path: str = 'models/best_model.pth'):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"🍎 UPDATABLE COREML EXPORTER")
        print(f"Checkpoint: {self.checkpoint_path}")
    
    def load_pytorch_model(self):
        """Load trained PyTorch model"""
        print("\n📦 Loading PyTorch model...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        metrics = checkpoint['metrics']
        
        print(f"   Architecture: {config['architecture']}")
        print(f"   Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Validation F1: {metrics['f1']:.4f}")
        
        # Create model
        model = NeuralFlareNet(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, config, metrics
    
    def convert_to_base_coreml(self, model: nn.Module, config: dict):
        """Convert PyTorch to base CoreML model"""
        print("\n🔄 Converting to base CoreML...")
        
        # Model parameters
        sequence_length = 30
        input_dim = config.get('input_dim', 92)
        
        # Create dummy input
        dummy_input = torch.randn(1, sequence_length, input_dim)
        
        # Trace model
        print("   Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to CoreML
        print("   Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input_sequence", 
                                 shape=(1, sequence_length, input_dim))],
            convert_to="neuralnetwork",  # Use neural network (not mlprogram) for updatable models
            minimum_deployment_target=ct.target.iOS16
        )
        
        print("✅ Base CoreML model created")
        return mlmodel
    
    def make_model_updatable(self, mlmodel, config: dict, scaler_params: dict):
        """Transform CoreML model to be updatable"""
        print("\n🔧 Making model updatable...")
        
        # Get model spec
        spec = mlmodel.get_spec()
        
        # Create neural network builder
        builder = NeuralNetworkBuilder(spec=spec)
        
        # 1. Mark layers as updatable
        print("   Marking layers as updatable...")
        # Strategy: Make only the final dense layers updatable to preserve base knowledge
        # while allowing personalization
        updatable_layers = []
        
        # Find layer names (this depends on your model architecture)
        # For LSTM/GRU models, we typically update the final dense layers
        for layer in spec.neuralNetwork.layers:
            layer_name = layer.name
            # Make final 1-2 dense layers updatable
            if 'dense' in layer_name.lower() or 'linear' in layer_name.lower():
                updatable_layers.append(layer_name)
                print(f"      - {layer_name}")
        
        if not updatable_layers:
            print("      ⚠️  No dense layers found, making all layers updatable")
            updatable_layers = [layer.name for layer in spec.neuralNetwork.layers]
        
        # Make layers updatable
        builder.make_updatable(updatable_layers)
        
        # 2. Set loss function
        print("   Setting categorical cross-entropy loss...")
        builder.set_categorical_cross_entropy_loss(
            name='lossLayer',
            input='target'  # This will be the ground truth label input during training
        )
        
        # 3. Configure optimizer
        print("   Configuring SGD optimizer...")
        from coremltools.models.neural_network import SgdParams
        
        # Conservative learning rate for on-device learning
        # We don't want to destroy the pre-trained knowledge
        sgd_params = SgdParams(
            lr=0.0001,  # Very conservative learning rate
            batch=10,    # Small batch size for mobile
            momentum=0.9
        )
        builder.set_sgd_optimizer(sgd_params)
        
        # 4. Set training epochs
        print("   Setting training epochs...")
        builder.set_epochs(5)  # Default epochs for each update
        
        # 5. Embed scaler parameters in metadata
        print("   Embedding scaler parameters in metadata...")
        spec.description.metadata.userDefined['scaler_params'] = json.dumps(scaler_params)
        spec.description.metadata.userDefined['input_dim'] = str(config.get('input_dim', 92))
        spec.description.metadata.userDefined['sequence_length'] = '30'
        spec.description.metadata.userDefined['architecture'] = config['architecture']
        spec.description.metadata.userDefined['training_accuracy'] = str(config.get('accuracy', 0))
        
        # 6. Add model description
        spec.description.metadata.shortDescription = "Updatable AS Flare Prediction Model"
        spec.description.metadata.author = "OVERLORD Neural Engine"
        spec.description.metadata.license = "Proprietary"
        spec.description.metadata.versionString = "1.0.0"
        
        # Create updatable model from spec
        from coremltools.models import MLModel
        updatable_model = MLModel(spec)
        
        print("✅ Model is now updatable")
        return updatable_model
    
    def validate_updatable_model(self, mlmodel):
        """Validate that model is properly configured for updates"""
        print("\n🔍 Validating updatable model...")
        
        spec = mlmodel.get_spec()
        
        # Check if model is updatable
        is_updatable = spec.isUpdatable
        print(f"   Is updatable: {is_updatable}")
        
        if not is_updatable:
            print("   ❌ Model is NOT updatable!")
            return False
        
        # Check updatable layers
        updatable_layers = []
        for layer in spec.neuralNetwork.layers:
            if layer.isUpdatable:
                updatable_layers.append(layer.name)
        
        print(f"   Updatable layers: {len(updatable_layers)}")
        for layer_name in updatable_layers:
            print(f"      - {layer_name}")
        
        # Check loss function
        has_loss = hasattr(spec.neuralNetwork, 'lossLayers') and len(spec.neuralNetwork.lossLayers) > 0
        print(f"   Has loss function: {has_loss}")
        
        # Check optimizer
        has_optimizer = hasattr(spec.neuralNetwork, 'optimizer')
        print(f"   Has optimizer: {has_optimizer}")
        
        # Check training inputs
        has_training_inputs = hasattr(spec, 'trainingInputs') and len(spec.trainingInputs) > 0
        print(f"   Has training inputs: {has_training_inputs}")
        
        if is_updatable and has_loss and has_optimizer:
            print("✅ Model validation passed")
            return True
        else:
            print("❌ Model validation failed")
            return False
    
    def save_model(self, mlmodel, output_path: str = "models/UpdatableNeuralFlareNet.mlpackage"):
        """Save updatable CoreML model"""
        print(f"\n💾 Saving updatable model...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save model
        mlmodel.save(output_path)
        
        # Calculate size
        import shutil
        size_mb = sum(f.stat().st_size for f in output_path.rglob('*')) / (1024 * 1024)
        
        print(f"✅ Model saved: {output_path}")
        print(f"   Size: {size_mb:.2f} MB")
        
        return output_path
    
    def generate_swift_integration(self, model_path: Path):
        """Generate Swift code for using the updatable model"""
        print(f"\n📝 Generating Swift integration code...")
        
        swift_code = '''//
//  UpdatableFlarePredictor.swift
//  Spinalytics
//
//  Auto-generated by OVERLORD Neural Engine
//  Updatable CoreML Model Integration
//

import Foundation
import CoreML

@MainActor
class UpdatableFlarePredictor: ObservableObject {
    
    @Published var isModelLoaded = false
    @Published var lastPrediction: FlarePrediction?
    @Published var isUpdating = false
    @Published var updateProgress: Double = 0.0
    @Published var errorMessage: String?
    
    private var model: MLModel?
    private var modelURL: URL
    private var scaler: FeatureScaler?
    
    struct FlarePrediction {
        let willFlare: Bool
        let confidence: Float
        let riskScore: Float
        let timestamp: Date
    }
    
    init() {
        // Get model URL from bundle
        guard let url = Bundle.main.url(forResource: "UpdatableNeuralFlareNet", withExtension: "mlpackage") else {
            fatalError("Model not found in bundle")
        }
        self.modelURL = url
        
        Task {
            await loadModel()
        }
    }
    
    private func loadModel() async {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine
            
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
            
            // Load scaler from model metadata
            if let metadata = model?.modelDescription.metadata[.creatorDefinedKey] as? [String: String] {
                self.scaler = FeatureScaler(from: metadata)
            }
            
            self.isModelLoaded = true
            print("✅ Updatable model loaded successfully")
        } catch {
            self.errorMessage = "Failed to load model: \\(error)"
            print("❌ Model loading failed: \\(error)")
        }
    }
    
    func predict(features: [[Float]]) async -> FlarePrediction? {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return nil
        }
        
        guard features.count == 30 && features.first?.count == 92 else {
            errorMessage = "Invalid input shape. Expected 30x92"
            return nil
        }
        
        do {
            // Normalize features using scaler
            let normalizedFeatures = scaler?.transform(features) ?? features
            
            // Create input
            let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
            for (i, timestep) in normalizedFeatures.enumerated() {
                for (j, value) in timestep.enumerated() {
                    inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
                }
            }
            
            let input = try MLDictionaryFeatureProvider(dictionary: ["input_sequence": inputArray])
            
            // Run prediction
            let output = try model.prediction(from: input)
            
            // Parse output
            guard let logits = output.featureValue(for: "output")?.multiArrayValue else {
                errorMessage = "Failed to parse model output"
                return nil
            }
            
            // Apply softmax
            let probs = softmax(logits)
            let willFlare = probs[1] > 0.5
            let confidence = abs(probs[1] - 0.5) * 2  // Scale to 0-1
            
            let prediction = FlarePrediction(
                willFlare: willFlare,
                confidence: confidence,
                riskScore: probs[1],
                timestamp: Date()
            )
            
            self.lastPrediction = prediction
            return prediction
            
        } catch {
            self.errorMessage = "Prediction failed: \\(error)"
            return nil
        }
    }
    
    func updateModel(with trainingData: [(features: [[Float]], label: Int)]) async throws {
        guard let model = model else {
            throw UpdateError.modelNotLoaded
        }
        
        guard !trainingData.isEmpty else {
            throw UpdateError.noTrainingData
        }
        
        print("🔄 Starting on-device model update with \\(trainingData.count) samples...")
        
        self.isUpdating = true
        self.updateProgress = 0.0
        
        do {
            // Prepare training batch
            let batchProvider = try prepareTrainingBatch(trainingData)
            
            // Create update task
            let updateTask = try MLUpdateTask(
                forModelAt: modelURL,
                trainingData: batchProvider,
                configuration: nil,
                completionHandler: { [weak self] context in
                    Task { @MainActor in
                        await self?.handleUpdateCompletion(context)
                    }
                },
                progressHandler: { [weak self] context in
                    Task { @MainActor in
                        self?.updateProgress = Double(context.metrics[.epochIndex] as? Int ?? 0) / 5.0
                    }
                }
            )
            
            // Start update
            updateTask.resume()
            
        } catch {
            self.isUpdating = false
            throw UpdateError.updateFailed(error)
        }
    }
    
    private func prepareTrainingBatch(_ data: [(features: [[Float]], label: Int)]) throws -> MLBatchProvider {
        var featureProviders: [MLFeatureProvider] = []
        
        for sample in data {
            // Normalize features
            let normalizedFeatures = scaler?.transform(sample.features) ?? sample.features
            
            // Create input array
            let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
            for (i, timestep) in normalizedFeatures.enumerated() {
                for (j, value) in timestep.enumerated() {
                    inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
                }
            }
            
            // Create label array
            let labelArray = try MLMultiArray(shape: [1], dataType: .int32)
            labelArray[0] = NSNumber(value: sample.label)
            
            // Create feature provider
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_sequence": inputArray,
                "target": labelArray
            ])
            
            featureProviders.append(provider)
        }
        
        return MLArrayBatchProvider(array: featureProviders)
    }
    
    private func handleUpdateCompletion(_ context: MLUpdateContext) async {
        if let error = context.task.error {
            print("❌ Update failed: \\(error)")
            self.errorMessage = "Update failed: \\(error.localizedDescription)"
            self.isUpdating = false
            return
        }
        
        // Get updated model
        guard let updatedModel = context.model else {
            print("❌ No updated model returned")
            self.isUpdating = false
            return
        }
        
        // Save updated model
        do {
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("updated_model.mlpackage")
            try updatedModel.write(to: tempURL)
            
            // Replace old model
            try FileManager.default.removeItem(at: modelURL)
            try FileManager.default.moveItem(at: tempURL, to: modelURL)
            
            // Reload model
            self.model = updatedModel
            
            print("✅ Model updated successfully")
            self.isUpdating = false
            self.updateProgress = 1.0
            
        } catch {
            print("❌ Failed to save updated model: \\(error)")
            self.errorMessage = "Failed to save updated model"
            self.isUpdating = false
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
    
    enum UpdateError: Error {
        case modelNotLoaded
        case noTrainingData
        case updateFailed(Error)
    }
}
'''
        
        swift_path = model_path.parent / "UpdatableFlarePredictor.swift"
        with open(swift_path, 'w') as f:
            f.write(swift_code)
        
        print(f"✅ Swift integration code: {swift_path}")
        return swift_path
    
    def export(self):
        """Execute full export pipeline"""
        print("\n" + "="*70)
        print("UPDATABLE COREML EXPORT PIPELINE")
        print("="*70)
        
        try:
            # Load PyTorch model
            model, config, metrics = self.load_pytorch_model()
            
            # Load scaler parameters
            scaler_path = Path('data/scaler_params.json')
            if not scaler_path.exists():
                print("⚠️  Scaler parameters not found. Run feature_scaler.py first.")
                return False
            
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            
            # Convert to base CoreML
            base_mlmodel = self.convert_to_base_coreml(model, config)
            
            # Make model updatable
            updatable_model = self.make_model_updatable(base_mlmodel, config, scaler_params)
            
            # Validate
            is_valid = self.validate_updatable_model(updatable_model)
            if not is_valid:
                print("❌ Model validation failed. Cannot proceed.")
                return False
            
            # Save model
            model_path = self.save_model(updatable_model)
            
            # Generate Swift integration
            self.generate_swift_integration(model_path)
            
            print("\n" + "="*70)
            print("🎉 UPDATABLE COREML EXPORT COMPLETE!")
            print("="*70)
            print(f"\n✅ Your model can now learn on-device!")
            print(f"   Model: {model_path}")
            print(f"   Features: On-device learning with SGD optimizer")
            print(f"   Learning rate: 0.0001 (conservative)")
            print(f"   Batch size: 10")
            print(f"   Epochs per update: 5")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Export PyTorch model to Updatable CoreML')
    parser.add_argument('--checkpoint', default='models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', default='models/UpdatableNeuralFlareNet.mlpackage',
                       help='Output path for CoreML model')
    args = parser.parse_args()
    
    # Check if coremltools is installed
    try:
        import coremltools
        print(f"CoreML Tools version: {coremltools.__version__}")
    except ImportError:
        print("❌ coremltools not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'coremltools'])
        print("✅ coremltools installed. Please run the script again.")
        sys.exit(1)
    
    # Create exporter
    exporter = UpdatableCoreMLExporter(checkpoint_path=args.checkpoint)
    
    # Run export
    success = exporter.export()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
