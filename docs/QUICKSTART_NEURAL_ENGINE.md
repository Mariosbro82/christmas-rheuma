# 🚀 Neural Engine Quick Start Guide

**Goal**: Get your 10/10 neural engine running in the iOS app
**Time Required**: 2-4 hours for basic integration
**Status**: Model exported (2.13MB) ✅, Ready for iOS integration

---

## ✅ What's Already Done

- ✅ Model exported to CoreML (AS FlarePredictor.mlpackage, 2.13MB)
- ✅ FP16 quantization applied (27x compression)
- ✅ 92-feature architecture aligned
- ✅ Scaler parameters embedded in metadata
- ✅ Baseline performance: 84.3% accuracy on synthetic data

---

## 🔥 Critical Next Steps (Priority Order)

### **Step 1: Bundle Model in Xcode** (30 minutes)

1. **Copy model to Xcode project**:
   ```bash
   # From project root
   mkdir -p InflamAI/Resources/ML
   cp InflamAI/Core/ML/NeuralEngine/models/ASFlarePredictor.mlpackage \
      InflamAI/Resources/ML/
   ```

2. **Add to Xcode**:
   - Open `InflamAI.xcodeproj` in Xcode
   - Right-click project in navigator → "Add Files to InflamAI..."
   - Navigate to `InflamAI/Resources/ML/ASFlarePredictor.mlpackage`
   - Check: ✅ "Copy items if needed", ✅ "InflamAI" target

3. **Verify in Build Phases**:
   - Select project → Target: InflamAI → Build Phases
   - Expand "Copy Bundle Resources"
   - Confirm `ASFlarePredictor.mlpackage` is listed
   - If not, click "+", add `ASFlarePredictor.mlpackage`

4. **Test model loads**:
   ```swift
   // In InflamAIApp.swift or any test view
   @State private var modelTest: String = "Not tested"

   Button("Test Neural Engine Load") {
       do {
           let model = try ASFlarePredictor(configuration: MLModelConfiguration())
           modelTest = "✅ Model loaded successfully!"
       } catch {
           modelTest = "❌ Failed: \(error.localizedDescription)"
       }
   }
   Text(modelTest)
   ```

---

### **Step 2: Create Swift Integration Service** (2 hours)

Create file: `InflamAI/Core/ML/NeuralEnginePredictionService.swift`

```swift
//
//  NeuralEnginePredictionService.swift
//  InflamAI
//
//  Neural Engine prediction service with 92-feature support
//

import Foundation
import CoreML
import Combine

@MainActor
class NeuralEnginePredictionService: ObservableObject {

    // MARK: - Published Properties
    @Published var isModelLoaded = false
    @Published var lastPrediction: FlarePrediction?
    @Published var errorMessage: String?

    // MARK: - Private Properties
    private var model: ASFlarePredictor?
    private var scaler: FeatureScaler?
    private var metadata: ModelMetadata?

    // MARK: - Data Structures
    struct FlarePrediction {
        let willFlare: Bool
        let probability: Float  // P(flare)
        let riskScore: Float    // 0-1 continuous
        let confidence: ConfidenceLevel
        let timestamp: Date
        let daysOfDataUsed: Int

        enum ConfidenceLevel: String {
            case low = "Low"
            case medium = "Medium"
            case high = "High"

            init(probability: Float) {
                let distance = abs(probability - 0.5)
                if distance < 0.15 { self = .low }
                else if distance < 0.3 { self = .medium }
                else { self = .high }
            }

            var color: String {
                switch self {
                case .low: return "gray"
                case .medium: return "orange"
                case .high: return "green"
                }
            }
        }
    }

    struct ModelMetadata {
        let architecture: String
        let inputFeatures: Int
        let baselineAccuracy: Float
        let baselineF1: Float
        let featureNames: [String]
        let scalerMeans: [Float]
        let scalerStds: [Float]
    }

    // MARK: - Initialization
    init() {
        Task {
            await loadModel()
        }
    }

    // MARK: - Model Loading
    private func loadModel() async {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine + GPU

            self.model = try ASFlarePredictor(configuration: config)

            // Parse metadata
            if let modelDescription = model?.model.modelDescription,
               let userDefined = modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String] {
                self.metadata = parseMetadata(userDefined)
                self.scaler = FeatureScaler(
                    means: metadata?.scalerMeans ?? [],
                    stds: metadata?.scalerStds ?? []
                )
            }

            self.isModelLoaded = true
            print("✅ Neural Engine loaded successfully")
            print("   Architecture: \(metadata?.architecture ?? "unknown")")
            print("   Features: \(metadata?.inputFeatures ?? 0)")
            print("   Baseline Accuracy: \(metadata?.baselineAccuracy ?? 0)")

        } catch {
            self.errorMessage = "Failed to load neural engine: \(error.localizedDescription)"
            print("❌ Model loading failed: \(error)")
        }
    }

    private func parseMetadata(_ userDefined: [String: String]) -> ModelMetadata {
        return ModelMetadata(
            architecture: userDefined["architecture"] ?? "LSTM",
            inputFeatures: Int(userDefined["input_features"] ?? "92") ?? 92,
            baselineAccuracy: Float(userDefined["baseline_accuracy"] ?? "0") ?? 0,
            baselineF1: Float(userDefined["baseline_f1"] ?? "0") ?? 0,
            featureNames: parseJSONStringArray(userDefined["feature_names"]) ?? [],
            scalerMeans: parseJSONFloatArray(userDefined["scaler_means"]) ?? [],
            scalerStds: parseJSONFloatArray(userDefined["scaler_stds"]) ?? []
        )
    }

    private func parseJSONStringArray(_ json: String?) -> [String]? {
        guard let json = json,
              let data = json.data(using: .utf8),
              let array = try? JSONDecoder().decode([String].self, from: data) else {
            return nil
        }
        return array
    }

    private func parseJSONFloatArray(_ json: String?) -> [Float]? {
        guard let json = json,
              let data = json.data(using: .utf8),
              let array = try? JSONDecoder().decode([Float].self, from: data) else {
            return nil
        }
        return array
    }

    // MARK: - Prediction
    func predict(features: [[Float]]) async throws -> FlarePrediction {
        guard let model = model else {
            throw PredictionError.modelNotLoaded
        }

        guard features.count == 30 && features.first?.count == 92 else {
            throw PredictionError.invalidInputShape(
                expected: "(30, 92)",
                got: "(\(features.count), \(features.first?.count ?? 0))"
            )
        }

        // Normalize features
        let normalizedFeatures = scaler?.transform(features) ?? features

        // Create MLMultiArray
        let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
        for (i, timestep) in normalizedFeatures.enumerated() {
            for (j, value) in timestep.enumerated() {
                inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
            }
        }

        // Create input
        let input = ASFlarePredictorInput(features: inputArray)

        // Run prediction
        let output = try model.prediction(input: input)

        // Parse outputs
        let probabilities = output.probabilities
        let willFlareProb = probabilities[1].floatValue  // P(flare=1)
        let willFlare = willFlareProb > 0.5
        let riskScore = output.risk_score[0].floatValue
        let confidence = FlarePrediction.ConfidenceLevel(probability: willFlareProb)

        // Count non-zero days (actual data vs padding)
        let daysOfData = features.filter { timestep in
            timestep.contains { $0 != 0.0 }
        }.count

        let prediction = FlarePrediction(
            willFlare: willFlare,
            probability: willFlareProb,
            riskScore: riskScore,
            confidence: confidence,
            timestamp: Date(),
            daysOfDataUsed: daysOfData
        )

        self.lastPrediction = prediction
        return prediction
    }

    // MARK: - Errors
    enum PredictionError: LocalizedError {
        case modelNotLoaded
        case invalidInputShape(expected: String, got: String)

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "Neural engine not loaded. Please wait for initialization."
            case .invalidInputShape(let expected, let got):
                return "Invalid input shape. Expected \(expected), got \(got)"
            }
        }
    }
}

// MARK: - Feature Scaler
class FeatureScaler {
    private let means: [Float]
    private let stds: [Float]

    init(means: [Float], stds: [Float]) {
        self.means = means
        self.stds = stds
    }

    func transform(_ features: [[Float]]) -> [[Float]] {
        guard !means.isEmpty && !stds.isEmpty else {
            return features  // No scaling
        }

        return features.map { timestep in
            timestep.enumerated().map { (index, value) in
                guard index < means.count && index < stds.count else {
                    return value
                }
                let mean = means[index]
                let std = stds[index]
                return std > 0 ? (value - mean) / std : value
            }
        }
    }
}
```

---

### **Step 3: Create Test UI** (30 minutes)

Create file: `InflamAI/Features/AI/NeuralEnginePredictionView.swift`

```swift
//
//  NeuralEnginePredictionView.swift
//  InflamAI
//

import SwiftUI

struct NeuralEnginePredictionView: View {
    @StateObject private var service = NeuralEnginePredictionService()

    var body: some View {
        VStack(spacing: 20) {
            // Model Status
            HStack {
                Circle()
                    .fill(service.isModelLoaded ? Color.green : Color.red)
                    .frame(width: 12, height: 12)
                Text(service.isModelLoaded ? "Neural Engine Ready" : "Loading...")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Last Prediction (if available)
            if let prediction = service.lastPrediction {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Latest Prediction")
                        .font(.headline)

                    HStack {
                        Text(prediction.willFlare ? "⚠️ Flare Likely" : "✅ Low Risk")
                            .font(.title2)
                            .fontWeight(.bold)

                        Spacer()

                        Text("\(Int(prediction.probability * 100))%")
                            .font(.title)
                            .foregroundColor(prediction.willFlare ? .orange : .green)
                    }

                    HStack {
                        Text("Confidence:")
                        Text(prediction.confidence.rawValue)
                            .foregroundColor(Color(prediction.confidence.color))

                        Spacer()

                        Text("Risk Score: \(String(format: "%.2f", prediction.riskScore))")
                            .font(.caption)
                    }

                    Text("Based on \(prediction.daysOfDataUsed) days of data")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text("⚠️ Research feature - not medical advice")
                        .font(.caption2)
                        .foregroundColor(.orange)
                        .padding(.top, 4)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
            }

            // Test Button (for development)
            Button(action: testPrediction) {
                Label("Test Prediction (Random Data)", systemImage: "waveform.path.ecg")
            }
            .buttonStyle(.borderedProminent)

            // Error Display
            if let error = service.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
            }

            Spacer()
        }
        .padding()
        .navigationTitle("Neural Engine")
    }

    private func testPrediction() {
        Task {
            do {
                // Generate 30 days of random features (92 features each)
                var testFeatures: [[Float]] = []
                for _ in 0..<30 {
                    var dayFeatures: [Float] = []
                    for _ in 0..<92 {
                        dayFeatures.append(Float.random(in: -2...2))  // Normalized range
                    }
                    testFeatures.append(dayFeatures)
                }

                let prediction = try await service.predict(features: testFeatures)
                print("✅ Test prediction: \(prediction.willFlare ? "Flare" : "No Flare") (\(Int(prediction.probability * 100))%)")

            } catch {
                print("❌ Test prediction failed: \(error)")
            }
        }
    }
}

#Preview {
    NavigationStack {
        NeuralEnginePredictionView()
    }
}
```

---

### **Step 4: Add to Main Navigation** (15 minutes)

In your main navigation (e.g., `ContentView.swift` or tab bar):

```swift
NavigationLink(destination: NeuralEnginePredictionView()) {
    Label("Neural Engine", systemImage: "brain.head.profile")
}
```

---

### **Step 5: Build & Test** (30 minutes)

1. **Build the project** (`Cmd + B`)
   - Fix any compilation errors
   - Ensure model is in bundle

2. **Run on simulator or device**:
   - Navigate to Neural Engine view
   - Check: "Neural Engine Ready" indicator
   - Click "Test Prediction" button
   - Should see prediction with percentage and confidence

3. **Verify model outputs**:
   - Check console for: "✅ Neural Engine loaded successfully"
   - Prediction should show:
     - ✅/⚠️ icon
     - Percentage (0-100%)
     - Confidence level (Low/Medium/High)
     - Risk score (0-1)

4. **Common issues**:
   - **"Model not found"**: Check Build Phases → Copy Bundle Resources
   - **"Invalid input shape"**: Ensure exactly 92 features per timestep
   - **Crash on load**: Check iOS deployment target is ≥17.0

---

## 🎯 What You'll Have After These Steps

- ✅ Neural engine integrated into your app
- ✅ Model loads on app launch
- ✅ Can run predictions (test mode with random data)
- ✅ See confidence levels and risk scores
- ✅ Foundation for real feature extraction

---

## 📝 Next Steps (After Integration)

Once basic integration works, you'll need to:

1. **Connect to Real Data** (Next Priority)
   - Extract 92 features from Core Data + HealthKit
   - Replace random test data with actual patient history
   - Handle missing values gracefully

2. **Add Explainability** (Phase 1, Day 3)
   - SHAP value computation
   - Display top 5 contributing factors
   - "Why this prediction?" explanations

3. **Implement Continuous Learning** (Phase 1, Day 4-5)
   - MLUpdateTask for on-device training
   - Personalization adapter
   - Progressive blending (synthetic → personal)

4. **Enhance UI** (Throughout testing)
   - Bootstrap progress indicator
   - Weekly accuracy reports
   - Recommendations engine

---

## 🆘 Troubleshooting

### Model Won't Load
```
❌ Failed to load neural engine: The model could not be read because it is missing.
```
**Fix**: Model not in bundle. Re-do Step 1, ensure "Copy Bundle Resources" includes `.mlpackage`

### Invalid Input Shape
```
❌ Invalid input shape. Expected (30, 92), got (30, 0)
```
**Fix**: Feature extraction incomplete. Check that all 92 features are populated, not zeros.

### Slow Predictions (>100ms)
**Fix**:
- Check model config: `computeUnits = .all` (uses Neural Engine)
- Reduce batch size (should be 1 for inference)
- Profile with Instruments

### Crashes on Simulator
**Note**: Neural Engine unavailable on simulator. Some features may not work. Test on physical device for best results.

---

## 📚 Reference: 92 Feature List

Copy this to your code for documentation:

```swift
/// 92 features expected by ASFlarePredictor model
/// Order matters - must match scaler_params.json!
enum FeatureIndex: Int, CaseIterable {
    // Demographics (6)
    case age = 0, gender, hla_b27, disease_duration, bmi, smoking

    // Clinical Assessment (15)
    case basdai_score = 6, asdas_crp, basfi, basmi, patient_global
    case physician_global, tender_joint_count, swollen_joint_count
    case enthesitis, dactylitis, spinal_mobility, disease_activity_composite

    // Pain (12)
    case pain_current = 21, pain_avg_24h, pain_max_24h, nocturnal_pain
    case morning_stiffness_duration, morning_stiffness_severity
    case pain_location_count, pain_burning, pain_aching, pain_sharp
    case pain_interference_sleep, pain_interference_activity

    // And so on... (see NEURAL_ENGINE_10_10_ROADMAP.md for full list)
}
```

---

## ⏱️ Time Estimates

- **Step 1** (Bundle Model): 30 minutes
- **Step 2** (Swift Service): 2 hours
- **Step 3** (Test UI): 30 minutes
- **Step 4** (Navigation): 15 minutes
- **Step 5** (Build & Test): 30 minutes

**Total**: 3-4 hours for complete basic integration

---

## 🎉 Success Criteria

You'll know it's working when:
- ✅ App builds without errors
- ✅ "Neural Engine Ready" shows green indicator
- ✅ Test prediction returns a percentage (0-100%)
- ✅ Confidence level displays (Low/Medium/High)
- ✅ No crashes when navigating to Neural Engine view

---

**Questions?** Refer to `NEURAL_ENGINE_10_10_ROADMAP.md` for detailed architecture and next phases.

**Ready to start?** Begin with Step 1 - bundle the model in Xcode!
