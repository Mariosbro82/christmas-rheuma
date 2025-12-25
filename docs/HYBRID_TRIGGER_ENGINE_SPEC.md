# Hybrid Trigger Detection Engine - Deep Technical Specification

## Document Version: 2.0 - Deep Dive
## Date: December 7, 2025
## Status: Detailed Design Phase

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Three Engines](#2-the-three-engines)
3. [Ensemble System](#3-ensemble-system)
4. [Progressive Activation](#4-progressive-activation)
5. [Data Schema](#5-data-schema)
6. [Feature Engineering Pipeline](#6-feature-engineering-pipeline)
7. [Confidence & Uncertainty System](#7-confidence--uncertainty-system)
8. [Explainability Framework](#8-explainability-framework)
9. [Implementation Architecture](#9-implementation-architecture)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Architecture Overview

### 1.1 The Hybrid Philosophy

The core insight: **No single approach works for all users at all stages.**

```
User Journey:
┌────────────────────────────────────────────────────────────────────┐
│ Day 1          Day 30         Day 90         Day 180              │
│   │              │              │              │                   │
│   ▼              ▼              ▼              ▼                   │
│ ┌────────┐  ┌─────────┐   ┌──────────┐   ┌──────────────┐        │
│ │ Stats  │→ │ Stats + │ → │ Stats +  │ → │ Full Hybrid  │        │
│ │ Only   │  │ k-NN    │   │ k-NN +   │   │ Ensemble     │        │
│ │        │  │         │   │ Neural   │   │              │        │
│ └────────┘  └─────────┘   └──────────┘   └──────────────┘        │
│                                                                   │
│ Confidence: Low → Medium → High → Very High                       │
└────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Hybrid?

| Approach | Strengths | Weaknesses | When to Use |
|----------|-----------|------------|-------------|
| **Statistical** | 100% explainable, no overfitting | Linear only, no interactions | Always (foundation) |
| **k-NN** | Non-parametric, intuitive | Memory grows, no generalization | 30+ days |
| **Neural** | Complex patterns, predictive | Overfitting risk, black-box | 90+ days, opt-in |

**Hybrid Solution**: Combine all three, weight by confidence, explain everything.

### 1.3 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID TRIGGER ENGINE                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        DATA COLLECTION LAYER                         │    │
│  │                                                                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │ Symptom  │  │ Trigger  │  │HealthKit │  │ OpenMeteo│            │    │
│  │  │ Logs     │  │ Logs     │  │ Data     │  │ Weather  │            │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │    │
│  │       │              │              │              │                │    │
│  │       └──────────────┴──────────────┴──────────────┘                │    │
│  │                              │                                       │    │
│  └──────────────────────────────┼───────────────────────────────────────┘    │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    FEATURE ENGINEERING LAYER                         │    │
│  │                                                                      │    │
│  │  TriggerFeatureExtractor                                            │    │
│  │  ├── Raw Features (92 from existing FeatureExtractor)               │    │
│  │  ├── Lag Features (t-1, t-2, t-3)                                   │    │
│  │  ├── Rolling Features (3d, 7d, 14d averages)                        │    │
│  │  ├── Interaction Features (coffee×stress, sleep×activity)          │    │
│  │  ├── Delta Features (day-over-day changes)                          │    │
│  │  └── Temporal Features (day of week, weekend, season)               │    │
│  │                                                                      │    │
│  │  Output: 150+ engineered features per day                           │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       ANALYSIS ENGINE LAYER                          │    │
│  │                                                                      │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │    │
│  │  │  STATISTICAL     │  │  K-NN            │  │  NEURAL          │   │    │
│  │  │  ENGINE          │  │  ENGINE          │  │  ENGINE          │   │    │
│  │  │                  │  │                  │  │                  │   │    │
│  │  │ • Pearson        │  │ • Core ML KNN    │  │ • GRU-Attention  │   │    │
│  │  │ • Spearman       │  │ • k=5 neighbors  │  │ • Transfer learn │   │    │
│  │  │ • Lagged corr    │  │ • Euclidean dist │  │ • MC Dropout     │   │    │
│  │  │ • Effect size    │  │ • On-device      │  │ • On-device      │   │    │
│  │  │ • Bonferroni     │  │   updatable      │  │   updatable      │   │    │
│  │  │                  │  │                  │  │                  │   │    │
│  │  │ Activation: 7d   │  │ Activation: 30d  │  │ Activation: 90d  │   │    │
│  │  │ Weight: 0.5-1.0  │  │ Weight: 0-0.3    │  │ Weight: 0-0.3    │   │    │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │    │
│  │           │                     │                     │              │    │
│  └───────────┼─────────────────────┼─────────────────────┼──────────────┘    │
│              │                     │                     │                   │
│              └─────────────────────┼─────────────────────┘                   │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       ENSEMBLE LAYER                                 │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  EnsembleWeightCalculator                                     │   │    │
│  │  │  ├── Data availability weights                                │   │    │
│  │  │  ├── Historical accuracy weights                              │   │    │
│  │  │  ├── Confidence calibration                                   │   │    │
│  │  │  └── Dynamic reweighting                                      │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  ConflictResolver                                             │   │    │
│  │  │  ├── When engines disagree                                    │   │    │
│  │  │  ├── Confidence-weighted voting                               │   │    │
│  │  │  └── User feedback integration                                │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     EXPLAINABILITY LAYER                             │    │
│  │                                                                      │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │    │
│  │  │ Statistical     │  │ Similar Days    │  │ Feature         │      │    │
│  │  │ Report          │  │ Explanation     │  │ Attribution     │      │    │
│  │  │ Generator       │  │ Generator       │  │ Generator       │      │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  UnifiedExplanationGenerator                                  │   │    │
│  │  │  Combines all explanations into user-friendly format          │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         OUTPUT LAYER                                 │    │
│  │                                                                      │    │
│  │  • TriggerAnalysisResult (ranked triggers with confidence)          │    │
│  │  • PainPrediction (tomorrow's pain with uncertainty)                │    │
│  │  • PersonalizedRecommendations (actionable advice)                  │    │
│  │  • ExplanationBundle (multi-level explanations)                     │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Three Engines

### 2.1 Statistical Engine (Always Active)

**Purpose**: Foundation layer providing interpretable, clinically-validated correlations.

#### 2.1.1 Core Algorithms

```swift
// File: Core/ML/TriggerAnalysis/StatisticalTriggerEngine.swift

@MainActor
public final class StatisticalTriggerEngine: ObservableObject {

    // MARK: - Published State
    @Published public private(set) var analyzedTriggers: [StatisticalTrigger] = []
    @Published public private(set) var isAnalyzing: Bool = false
    @Published public private(set) var analysisDate: Date?

    // MARK: - Configuration
    private let minimumDays: Int = 7
    private let significanceLevel: Double = 0.05  // α
    private let minimumEffectSize: Double = 0.3   // |r| threshold
    private let maxLagDays: Int = 3

    // MARK: - Core Analysis

    /// Comprehensive trigger analysis with lagged correlations
    public func analyzeAllTriggers(
        symptomLogs: [SymptomLog],
        triggerLogs: [TriggerLog]
    ) async -> StatisticalAnalysisResult {

        isAnalyzing = true
        defer { isAnalyzing = false }

        var allTriggers: [StatisticalTrigger] = []

        // 1. Analyze each trigger category
        for category in TriggerCategory.allCases {
            let categoryTriggers = await analyzeCategory(
                category: category,
                symptomLogs: symptomLogs,
                triggerLogs: triggerLogs
            )
            allTriggers.append(contentsOf: categoryTriggers)
        }

        // 2. Apply Bonferroni correction
        let correctedTriggers = applyBonferroniCorrection(allTriggers)

        // 3. Calculate effect sizes
        let triggersWithEffects = await calculateEffectSizes(
            triggers: correctedTriggers,
            symptomLogs: symptomLogs,
            triggerLogs: triggerLogs
        )

        // 4. Generate confidence classifications
        let classifiedTriggers = classifyConfidence(triggersWithEffects)

        // 5. Sort by impact
        analyzedTriggers = classifiedTriggers.sorted {
            abs($0.effectSize.cohenD) > abs($1.effectSize.cohenD)
        }

        analysisDate = Date()

        return StatisticalAnalysisResult(
            triggers: analyzedTriggers,
            analysisDate: analysisDate!,
            daysAnalyzed: symptomLogs.count,
            bonferroniAlpha: 0.05 / Double(allTriggers.count)
        )
    }

    // MARK: - Lagged Correlation Analysis

    /// Test correlation at multiple lag offsets
    private func analyzeLaggedCorrelation(
        triggerValues: [Double],
        symptomValues: [Double],
        triggerName: String,
        maxLag: Int = 3
    ) -> [LaggedCorrelationResult] {

        var results: [LaggedCorrelationResult] = []

        for lag in 0...maxLag {
            guard triggerValues.count > lag, symptomValues.count > lag else { continue }

            // Align data: trigger[t] correlates with symptom[t+lag]
            let alignedTrigger = Array(triggerValues.dropLast(lag))
            let alignedSymptom = Array(symptomValues.dropFirst(lag))

            guard alignedTrigger.count == alignedSymptom.count,
                  alignedTrigger.count >= minimumDays else { continue }

            // Calculate Pearson correlation
            if let r = pearsonCorrelation(alignedTrigger, alignedSymptom) {
                let n = alignedTrigger.count
                let pValue = calculatePValue(r: r, n: n)

                results.append(LaggedCorrelationResult(
                    lag: lag,
                    lagDescription: lagDescription(lag),
                    correlation: r,
                    pValue: pValue,
                    sampleSize: n,
                    isSignificant: pValue < significanceLevel
                ))
            }
        }

        return results
    }

    // MARK: - Effect Size Calculation

    /// Calculate Cohen's d and other effect metrics
    private func calculateEffectSize(
        triggerDays: [Double],   // Pain on trigger days
        nonTriggerDays: [Double] // Pain on non-trigger days
    ) -> EffectSize {

        let meanTrigger = triggerDays.mean()
        let meanNonTrigger = nonTriggerDays.mean()
        let meanDifference = meanTrigger - meanNonTrigger

        let pooledSD = sqrt(
            (triggerDays.variance() + nonTriggerDays.variance()) / 2
        )

        let cohenD = pooledSD > 0 ? meanDifference / pooledSD : 0

        return EffectSize(
            meanWithTrigger: meanTrigger,
            meanWithoutTrigger: meanNonTrigger,
            meanDifference: meanDifference,
            pooledStandardDeviation: pooledSD,
            cohenD: cohenD,
            percentChange: meanNonTrigger > 0
                ? (meanDifference / meanNonTrigger) * 100
                : 0,
            clinicallySignificant: abs(cohenD) >= 0.5 && abs(meanDifference) >= 1.0
        )
    }

    // MARK: - Partial Correlation (Confound Control)

    /// Calculate correlation controlling for confounders
    public func partialCorrelation(
        x: [Double],
        y: [Double],
        controlling z: [Double]
    ) -> Double? {

        guard x.count == y.count, y.count == z.count, x.count >= 3 else {
            return nil
        }

        // Calculate residuals of x regressed on z
        let residualsX = calculateResiduals(y: x, x: z)
        // Calculate residuals of y regressed on z
        let residualsY = calculateResiduals(y: y, x: z)

        // Correlation of residuals
        return pearsonCorrelation(residualsX, residualsY)
    }

    private func calculateResiduals(y: [Double], x: [Double]) -> [Double] {
        // Simple linear regression: y = a + bx + residual
        let n = Double(y.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)

        let b = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        let a = (sumY - b * sumX) / n

        return zip(x, y).map { (xi, yi) in
            yi - (a + b * xi)  // residual
        }
    }

    // MARK: - Spearman Rank Correlation (Non-parametric)

    /// For non-normal distributions
    public func spearmanCorrelation(_ x: [Double], _ y: [Double]) -> Double? {
        guard x.count == y.count, x.count >= 3 else { return nil }

        let ranksX = assignRanks(x)
        let ranksY = assignRanks(y)

        return pearsonCorrelation(ranksX, ranksY)
    }

    private func assignRanks(_ values: [Double]) -> [Double] {
        let sorted = values.enumerated().sorted { $0.element < $1.element }
        var ranks = Array(repeating: 0.0, count: values.count)

        for (rank, item) in sorted.enumerated() {
            ranks[item.offset] = Double(rank + 1)
        }

        return ranks
    }
}
```

#### 2.1.2 Statistical Trigger Output

```swift
public struct StatisticalTrigger: Identifiable, Codable {
    public let id: UUID
    public let name: String
    public let category: TriggerCategory
    public let icon: String

    // Correlation results
    public let laggedResults: [LaggedCorrelationResult]
    public let bestLag: LaggedCorrelationResult?

    // Effect size
    public let effectSize: EffectSize

    // Statistical significance
    public let rawPValue: Double
    public let correctedPValue: Double  // After Bonferroni
    public let isSignificant: Bool

    // Confidence
    public let confidence: TriggerConfidence
    public let sampleSize: Int
    public let triggerDaysCount: Int
    public let nonTriggerDaysCount: Int

    // User-facing
    public var impactDescription: String {
        let direction = effectSize.meanDifference > 0 ? "increases" : "decreases"
        let amount = abs(effectSize.meanDifference)
        let timing = bestLag?.lagDescription ?? "same day"

        return "\(name) \(direction) pain by \(String(format: "%.1f", amount)) points (\(timing))"
    }

    public var confidenceDescription: String {
        switch confidence {
        case .high:
            return "High confidence (p=\(String(format: "%.3f", correctedPValue)), n=\(sampleSize))"
        case .medium:
            return "Moderate confidence - track \(max(0, 60 - sampleSize)) more days to confirm"
        case .low:
            return "Low confidence - early indication only"
        case .insufficient:
            return "Insufficient data - need \(max(0, 14 - triggerDaysCount)) more trigger days"
        }
    }
}

public struct LaggedCorrelationResult: Codable {
    public let lag: Int
    public let lagDescription: String  // "Same day", "Next day", etc.
    public let correlation: Double
    public let pValue: Double
    public let sampleSize: Int
    public let isSignificant: Bool

    public var strengthDescription: String {
        let absR = abs(correlation)
        if absR >= 0.7 { return "Strong" }
        if absR >= 0.5 { return "Moderate" }
        if absR >= 0.3 { return "Weak" }
        return "Negligible"
    }
}

public struct EffectSize: Codable {
    public let meanWithTrigger: Double
    public let meanWithoutTrigger: Double
    public let meanDifference: Double
    public let pooledStandardDeviation: Double
    public let cohenD: Double
    public let percentChange: Double
    public let clinicallySignificant: Bool

    public var cohenDInterpretation: String {
        let absD = abs(cohenD)
        if absD >= 0.8 { return "Large effect" }
        if absD >= 0.5 { return "Medium effect" }
        if absD >= 0.2 { return "Small effect" }
        return "Negligible effect"
    }
}

public enum TriggerConfidence: String, Codable {
    case high = "high"           // n >= 60, p < 0.01, |r| > 0.5, |d| > 0.5
    case medium = "medium"       // n >= 30, p < 0.05, |r| > 0.3
    case low = "low"             // Significant but weak
    case insufficient = "insufficient"  // Not enough trigger days

    public var color: String {
        switch self {
        case .high: return "green"
        case .medium: return "yellow"
        case .low: return "orange"
        case .insufficient: return "gray"
        }
    }

    public var icon: String {
        switch self {
        case .high: return "checkmark.seal.fill"
        case .medium: return "exclamationmark.triangle.fill"
        case .low: return "questionmark.circle.fill"
        case .insufficient: return "ellipsis.circle"
        }
    }
}
```

---

### 2.2 k-NN Engine (30+ Days)

**Purpose**: Instance-based learning that finds similar historical days for prediction and explanation.

#### 2.2.1 Architecture

```swift
// File: Core/ML/TriggerAnalysis/KNNTriggerEngine.swift

@MainActor
public final class KNNTriggerEngine: ObservableObject {

    // MARK: - Published State
    @Published public private(set) var isReady: Bool = false
    @Published public private(set) var trainingDays: Int = 0
    @Published public private(set) var lastUpdate: Date?

    // MARK: - Core ML Model
    private var knnModel: MLModel?
    private let modelURL: URL

    // MARK: - Configuration
    private let k: Int = 5                    // Number of neighbors
    private let minimumDays: Int = 30
    private let maxStoredDays: Int = 365      // Memory management

    // MARK: - Training Data Storage (for explanations)
    private var trainingData: [TrainingDay] = []

    // MARK: - Initialization

    public init() {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        modelURL = appSupport.appendingPathComponent("TriggerKNN.mlmodelc")

        Task {
            await initialize()
        }
    }

    // MARK: - Prediction with Explanation

    /// Predict pain and return similar days for explanation
    public func predictWithExplanation(
        features: [String: Double]
    ) async -> KNNPredictionResult? {

        guard isReady, let model = knnModel else { return nil }

        // Prepare feature vector
        let featureVector = prepareFeatureVector(features)

        // Find k nearest neighbors from stored training data
        let neighbors = findNearestNeighbors(features: featureVector, k: k)

        // Calculate prediction
        let predictedPain = neighbors.map { $0.painLevel }.mean()
        let painStd = neighbors.map { $0.painLevel }.standardDeviation()

        // Extract common patterns from neighbors
        let commonTriggers = identifyCommonTriggers(in: neighbors)

        return KNNPredictionResult(
            predictedPain: predictedPain,
            uncertainty: painStd,
            neighbors: neighbors,
            commonTriggers: commonTriggers,
            confidence: calculateConfidence(neighbors: neighbors)
        )
    }

    // MARK: - On-Device Update

    /// Add new day to k-NN training set
    public func addTrainingDay(_ day: TrainingDay) async {
        trainingData.append(day)

        // Memory management: Keep only recent days
        if trainingData.count > maxStoredDays {
            trainingData.removeFirst(trainingData.count - maxStoredDays)
        }

        // Update Core ML model via MLUpdateTask
        await updateCoreMLModel()

        trainingDays = trainingData.count
        lastUpdate = Date()
    }

    // MARK: - Neighbor Finding

    private func findNearestNeighbors(
        features: [Double],
        k: Int
    ) -> [SimilarDay] {

        // Calculate distances to all training days
        var distances: [(day: TrainingDay, distance: Double)] = []

        for day in trainingData {
            let dayFeatures = day.featureVector
            let distance = euclideanDistance(features, dayFeatures)
            distances.append((day, distance))
        }

        // Sort by distance and take k nearest
        distances.sort { $0.distance < $1.distance }
        let nearest = distances.prefix(k)

        return nearest.map { item in
            SimilarDay(
                date: item.day.date,
                painLevel: item.day.painLevel,
                distance: item.distance,
                triggers: item.day.activeTriggers,
                keyFeatures: extractKeyFeatures(item.day)
            )
        }
    }

    private func euclideanDistance(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count else { return Double.infinity }

        let squaredSum = zip(a, b).map { pow($0 - $1, 2) }.reduce(0, +)
        return sqrt(squaredSum)
    }

    // MARK: - Common Trigger Identification

    private func identifyCommonTriggers(in neighbors: [SimilarDay]) -> [CommonTrigger] {
        var triggerCounts: [String: Int] = [:]
        var triggerValues: [String: [Double]] = [:]

        for neighbor in neighbors {
            for trigger in neighbor.triggers {
                triggerCounts[trigger.name, default: 0] += 1
                triggerValues[trigger.name, default: []].append(trigger.value)
            }
        }

        return triggerCounts.compactMap { (name, count) in
            let frequency = Double(count) / Double(neighbors.count)
            guard frequency >= 0.6 else { return nil }  // At least 3/5 neighbors

            return CommonTrigger(
                name: name,
                frequency: frequency,
                averageValue: triggerValues[name]?.mean() ?? 0
            )
        }.sorted { $0.frequency > $1.frequency }
    }

    // MARK: - Core ML Update

    private func updateCoreMLModel() async {
        guard trainingData.count >= minimumDays else { return }

        // Prepare batch provider
        let batchProvider = createBatchProvider()

        // Run MLUpdateTask
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU

            let updateTask = try MLUpdateTask(
                forModelAt: modelURL,
                trainingData: batchProvider,
                configuration: config,
                completionHandler: { [weak self] context in
                    if let updatedModel = context.model {
                        try? updatedModel.write(to: self?.modelURL ?? URL(fileURLWithPath: ""))
                        self?.knnModel = updatedModel
                    }
                }
            )

            updateTask.resume()

        } catch {
            print("k-NN update failed: \(error)")
        }
    }
}
```

#### 2.2.2 k-NN Output Structures

```swift
public struct KNNPredictionResult {
    public let predictedPain: Double
    public let uncertainty: Double  // Standard deviation of neighbor pains
    public let neighbors: [SimilarDay]
    public let commonTriggers: [CommonTrigger]
    public let confidence: KNNConfidence

    public var explanation: String {
        let neighborDates = neighbors.prefix(3).map {
            DateFormatter.shortDate.string(from: $0.date)
        }.joined(separator: ", ")

        let triggerList = commonTriggers.prefix(3).map { $0.name }.joined(separator: ", ")

        return """
        Based on your 5 most similar days (\(neighborDates)...):
        • Predicted pain: \(String(format: "%.1f", predictedPain)) ± \(String(format: "%.1f", uncertainty))
        • Common factors: \(triggerList.isEmpty ? "No clear pattern" : triggerList)
        """
    }
}

public struct SimilarDay: Identifiable {
    public let id = UUID()
    public let date: Date
    public let painLevel: Double
    public let distance: Double  // Feature space distance
    public let triggers: [TriggerValue]
    public let keyFeatures: [String: Double]

    public var similarityScore: Double {
        // Convert distance to similarity (0-100%)
        max(0, 100 - distance * 10)
    }
}

public struct TriggerValue: Codable {
    public let name: String
    public let value: Double
    public let unit: String?
}

public struct CommonTrigger {
    public let name: String
    public let frequency: Double  // 0-1, how many neighbors had this
    public let averageValue: Double

    public var description: String {
        let percent = Int(frequency * 100)
        return "\(percent)% of similar days had \(name)"
    }
}

public enum KNNConfidence {
    case high      // Low distance variance, consistent pain levels
    case medium    // Moderate variance
    case low       // High variance, neighbors disagree

    public init(neighbors: [SimilarDay]) {
        let painStd = neighbors.map { $0.painLevel }.standardDeviation()
        let distanceStd = neighbors.map { $0.distance }.standardDeviation()

        if painStd < 1.0 && distanceStd < 0.5 {
            self = .high
        } else if painStd < 2.0 {
            self = .medium
        } else {
            self = .low
        }
    }
}

public struct TrainingDay: Codable {
    public let date: Date
    public let painLevel: Double
    public let featureVector: [Double]
    public let activeTriggers: [TriggerValue]
}
```

---

### 2.3 Neural Engine (90+ Days, Opt-In)

**Purpose**: Deep learning for complex non-linear patterns, with uncertainty quantification.

#### 2.3.1 Model Architecture

```
Input (7 days × 40 features = 280 values):
┌─────────────────────────────────────────────────────────┐
│ Day 1: [coffee, sleep, stress, pressure, ...] (40 feat) │
│ Day 2: [coffee, sleep, stress, pressure, ...] (40 feat) │
│ ...                                                      │
│ Day 7: [coffee, sleep, stress, pressure, ...] (40 feat) │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              GRU Layer (32 units)                        │
│              Dropout: 30%                                │
│              Returns: Sequences                          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Self-Attention Layer                           │
│           • Learns which days/features matter            │
│           • Outputs attention weights (explainability)   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Global Average Pooling                         │
│           Reduces sequence to single vector              │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Dense Layer (16 units)                         │
│           Activation: ReLU                               │
│           L2 Regularization: 0.001                       │
│           Dropout: 30%                                   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Output Layer (1 unit)                          │
│           Activation: Linear                             │
│           Output: Pain prediction (0-10)                 │
└─────────────────────────────────────────────────────────┘

Total Parameters: ~12,000 (manageable for 90+ samples with transfer learning)
```

#### 2.3.2 Implementation

```swift
// File: Core/ML/TriggerAnalysis/NeuralTriggerEngine.swift

@MainActor
public final class NeuralTriggerEngine: ObservableObject {

    // MARK: - Published State
    @Published public private(set) var isReady: Bool = false
    @Published public private(set) var isTraining: Bool = false
    @Published public private(set) var trainingProgress: Float = 0
    @Published public private(set) var lastTrainingDate: Date?
    @Published public private(set) var modelVersion: Int = 0

    // MARK: - Models
    private var model: MLModel?
    private let baseModelURL: URL  // Pre-trained population model
    private let personalizedModelURL: URL  // User-specific fine-tuned model

    // MARK: - Configuration
    private let minimumDays: Int = 90
    private let sequenceLength: Int = 7
    private let featureCount: Int = 40
    private let mcDropoutSamples: Int = 50  // For uncertainty

    // MARK: - Prediction with Uncertainty

    /// Predict with Monte Carlo Dropout for uncertainty estimation
    public func predictWithUncertainty(
        features: [[Double]]  // 7 days × 40 features
    ) async -> NeuralPredictionResult? {

        guard isReady, let model = model else { return nil }

        // Monte Carlo Dropout: Multiple forward passes with dropout active
        var predictions: [Double] = []

        for _ in 0..<mcDropoutSamples {
            if let pred = runInference(model: model, features: features, withDropout: true) {
                predictions.append(pred)
            }
        }

        guard !predictions.isEmpty else { return nil }

        // Calculate statistics
        let mean = predictions.mean()
        let std = predictions.standardDeviation()

        // Calculate confidence interval
        let sortedPreds = predictions.sorted()
        let lowerIdx = Int(Double(sortedPreds.count) * 0.025)
        let upperIdx = Int(Double(sortedPreds.count) * 0.975)
        let ci95Lower = sortedPreds[lowerIdx]
        let ci95Upper = sortedPreds[upperIdx]

        // Extract attention weights for explainability
        let attentionWeights = extractAttentionWeights(features: features)

        return NeuralPredictionResult(
            predictedPain: mean,
            uncertainty: std,
            ci95Lower: ci95Lower,
            ci95Upper: ci95Upper,
            attentionWeights: attentionWeights,
            confidence: NeuralConfidence(uncertainty: std)
        )
    }

    // MARK: - Feature Attribution (SHAP-like)

    /// Approximate feature importance via permutation
    public func calculateFeatureAttribution(
        baseFeatures: [[Double]],
        featureNames: [String]
    ) async -> [FeatureAttribution] {

        guard let basePrediction = await predictWithUncertainty(features: baseFeatures)?.predictedPain else {
            return []
        }

        var attributions: [FeatureAttribution] = []

        // For each feature, permute and measure impact
        for (featureIdx, featureName) in featureNames.enumerated() {
            var permutedFeatures = baseFeatures

            // Set feature to mean value across all days
            let meanValue = baseFeatures.map { $0[featureIdx] }.mean()
            for dayIdx in 0..<permutedFeatures.count {
                permutedFeatures[dayIdx][featureIdx] = meanValue
            }

            // Predict with permuted feature
            if let permutedPrediction = await predictWithUncertainty(features: permutedFeatures)?.predictedPain {
                let attribution = basePrediction - permutedPrediction

                attributions.append(FeatureAttribution(
                    feature: featureName,
                    attribution: attribution,
                    direction: attribution > 0 ? .increases : .decreases,
                    importance: abs(attribution)
                ))
            }
        }

        return attributions.sorted { $0.importance > $1.importance }
    }

    // MARK: - On-Device Training

    /// Fine-tune model on user's personal data
    public func trainOnUserData(
        trainingData: [NeuralTrainingExample],
        epochs: Int = 5
    ) async throws {

        guard trainingData.count >= minimumDays else {
            throw NeuralEngineError.insufficientData(
                required: minimumDays,
                available: trainingData.count
            )
        }

        isTraining = true
        trainingProgress = 0
        defer {
            isTraining = false
            trainingProgress = 1.0
        }

        // Prepare batch provider
        let batchProvider = try createBatchProvider(from: trainingData)

        // Configure training
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        // Create update task
        let updateTask = try MLUpdateTask(
            forModelAt: personalizedModelURL,
            trainingData: batchProvider,
            configuration: config,
            completionHandler: { [weak self] context in
                guard let self = self else { return }

                if let updatedModel = context.model {
                    try? updatedModel.write(to: self.personalizedModelURL)
                    self.model = updatedModel
                    self.modelVersion += 1
                    self.lastTrainingDate = Date()
                }
            }
        )

        // Progress monitoring
        updateTask.progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.epochEnd],
            progressHandler: { [weak self] context in
                let currentEpoch = context.metrics[.epochIndex] as? Int ?? 0
                self?.trainingProgress = Float(currentEpoch + 1) / Float(epochs)

                let loss = context.metrics[.lossValue] as? Double ?? -1
                print("Epoch \(currentEpoch + 1)/\(epochs): Loss = \(String(format: "%.4f", loss))")
            }
        )

        // Start training
        updateTask.resume()

        // Wait for completion (simplified - in production use proper async handling)
        try await Task.sleep(nanoseconds: UInt64(epochs * 10_000_000_000)) // 10s per epoch max
    }

    // MARK: - Attention Weight Extraction

    private func extractAttentionWeights(features: [[Double]]) -> AttentionWeights {
        // Extract from model's attention layer output
        // This requires the model to expose attention weights as an output

        // Placeholder - actual implementation depends on model architecture
        return AttentionWeights(
            dayWeights: [0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.2],  // Recent days matter more
            featureWeights: [:],  // Feature-level attention
            topFeatures: []
        )
    }
}
```

#### 2.3.3 Neural Output Structures

```swift
public struct NeuralPredictionResult {
    public let predictedPain: Double
    public let uncertainty: Double  // Standard deviation from MC Dropout
    public let ci95Lower: Double    // 95% confidence interval lower bound
    public let ci95Upper: Double    // 95% confidence interval upper bound
    public let attentionWeights: AttentionWeights
    public let confidence: NeuralConfidence

    public var explanation: String {
        let confRange = "\(String(format: "%.1f", ci95Lower))-\(String(format: "%.1f", ci95Upper))"
        return """
        Predicted pain: \(String(format: "%.1f", predictedPain))
        95% confidence range: \(confRange)
        Model confidence: \(confidence.rawValue)
        """
    }
}

public struct FeatureAttribution: Identifiable {
    public let id = UUID()
    public let feature: String
    public let attribution: Double  // Positive = increases pain
    public let direction: AttributionDirection
    public let importance: Double   // Absolute value

    public enum AttributionDirection {
        case increases, decreases
    }

    public var description: String {
        let verb = direction == .increases ? "increased" : "decreased"
        return "\(feature) \(verb) prediction by \(String(format: "%.1f", abs(attribution))) points"
    }
}

public struct AttentionWeights {
    public let dayWeights: [Double]  // Which days mattered most (length 7)
    public let featureWeights: [String: Double]  // Which features mattered
    public let topFeatures: [String]  // Top 5 features by attention
}

public enum NeuralConfidence: String {
    case high = "High"
    case medium = "Medium"
    case low = "Low"

    public init(uncertainty: Double) {
        if uncertainty < 0.8 {
            self = .high
        } else if uncertainty < 1.5 {
            self = .medium
        } else {
            self = .low
        }
    }
}

public struct NeuralTrainingExample {
    public let features: [[Double]]  // 7 days × 40 features
    public let label: Double         // Pain level on day 8
}

public enum NeuralEngineError: Error {
    case insufficientData(required: Int, available: Int)
    case modelNotLoaded
    case trainingFailed(String)
}
```

---

## 3. Ensemble System

### 3.1 Ensemble Architecture

```swift
// File: Core/ML/TriggerAnalysis/TriggerEnsemble.swift

@MainActor
public final class TriggerEnsemble: ObservableObject {

    // MARK: - Engines
    private let statisticalEngine: StatisticalTriggerEngine
    private let knnEngine: KNNTriggerEngine
    private let neuralEngine: NeuralTriggerEngine?  // Optional, opt-in

    // MARK: - Published State
    @Published public private(set) var activeEngines: Set<EngineType> = [.statistical]
    @Published public private(set) var lastEnsembleResult: EnsembleResult?

    // MARK: - Configuration
    private var daysOfData: Int = 0
    private var neuralOptIn: Bool = false

    // MARK: - Engine Activation Rules

    public func updateActiveEngines(daysOfData: Int, neuralOptIn: Bool) {
        self.daysOfData = daysOfData
        self.neuralOptIn = neuralOptIn

        var engines: Set<EngineType> = [.statistical]  // Always active

        if daysOfData >= 30 {
            engines.insert(.knn)
        }

        if daysOfData >= 90 && neuralOptIn {
            engines.insert(.neural)
        }

        activeEngines = engines
    }

    // MARK: - Ensemble Prediction

    public func predict(features: TriggerFeatures) async -> EnsembleResult {
        var predictions: [EnginePrediction] = []

        // 1. Statistical prediction (always)
        if let statResult = await statisticalEngine.predictPain(features: features) {
            predictions.append(EnginePrediction(
                engine: .statistical,
                prediction: statResult.prediction,
                uncertainty: statResult.uncertainty,
                confidence: statResult.confidence,
                explanation: statResult.explanation
            ))
        }

        // 2. k-NN prediction (30+ days)
        if activeEngines.contains(.knn),
           let knnResult = await knnEngine.predictWithExplanation(features: features.featureDict) {
            predictions.append(EnginePrediction(
                engine: .knn,
                prediction: knnResult.predictedPain,
                uncertainty: knnResult.uncertainty,
                confidence: knnResult.confidence.rawValue,
                explanation: knnResult.explanation
            ))
        }

        // 3. Neural prediction (90+ days, opt-in)
        if activeEngines.contains(.neural),
           let neuralEngine = neuralEngine,
           let neuralResult = await neuralEngine.predictWithUncertainty(features: features.sequentialFeatures) {
            predictions.append(EnginePrediction(
                engine: .neural,
                prediction: neuralResult.predictedPain,
                uncertainty: neuralResult.uncertainty,
                confidence: neuralResult.confidence.rawValue,
                explanation: neuralResult.explanation
            ))
        }

        // 4. Calculate ensemble weights
        let weights = calculateWeights(predictions: predictions)

        // 5. Combine predictions
        let ensemblePrediction = combinePredictoins(predictions: predictions, weights: weights)

        // 6. Generate unified explanation
        let explanation = generateUnifiedExplanation(predictions: predictions, weights: weights)

        let result = EnsembleResult(
            prediction: ensemblePrediction.value,
            uncertainty: ensemblePrediction.uncertainty,
            ci95: ensemblePrediction.ci95,
            enginePredictions: predictions,
            weights: weights,
            explanation: explanation,
            confidence: calculateOverallConfidence(predictions: predictions)
        )

        lastEnsembleResult = result
        return result
    }

    // MARK: - Weight Calculation

    private func calculateWeights(predictions: [EnginePrediction]) -> [EngineType: Double] {
        var weights: [EngineType: Double] = [:]

        // Base weights by data availability
        let baseWeights: [EngineType: Double]

        switch daysOfData {
        case 0..<30:
            baseWeights = [.statistical: 1.0, .knn: 0.0, .neural: 0.0]
        case 30..<60:
            baseWeights = [.statistical: 0.6, .knn: 0.4, .neural: 0.0]
        case 60..<90:
            baseWeights = [.statistical: 0.5, .knn: 0.5, .neural: 0.0]
        case 90..<180:
            if neuralOptIn {
                baseWeights = [.statistical: 0.4, .knn: 0.3, .neural: 0.3]
            } else {
                baseWeights = [.statistical: 0.5, .knn: 0.5, .neural: 0.0]
            }
        default: // 180+ days
            if neuralOptIn {
                baseWeights = [.statistical: 0.3, .knn: 0.3, .neural: 0.4]
            } else {
                baseWeights = [.statistical: 0.5, .knn: 0.5, .neural: 0.0]
            }
        }

        // Adjust weights by confidence
        for prediction in predictions {
            let baseWeight = baseWeights[prediction.engine] ?? 0
            let confidenceMultiplier = confidenceToMultiplier(prediction.confidence)
            weights[prediction.engine] = baseWeight * confidenceMultiplier
        }

        // Normalize weights to sum to 1.0
        let totalWeight = weights.values.reduce(0, +)
        if totalWeight > 0 {
            for (engine, weight) in weights {
                weights[engine] = weight / totalWeight
            }
        }

        return weights
    }

    private func confidenceToMultiplier(_ confidence: String) -> Double {
        switch confidence.lowercased() {
        case "high": return 1.0
        case "medium": return 0.8
        case "low": return 0.5
        default: return 0.3
        }
    }

    // MARK: - Prediction Combination

    private func combinePredictoins(
        predictions: [EnginePrediction],
        weights: [EngineType: Double]
    ) -> (value: Double, uncertainty: Double, ci95: (lower: Double, upper: Double)) {

        // Weighted average of predictions
        var weightedSum = 0.0
        var totalWeight = 0.0

        for prediction in predictions {
            let weight = weights[prediction.engine] ?? 0
            weightedSum += prediction.prediction * weight
            totalWeight += weight
        }

        let ensemblePrediction = totalWeight > 0 ? weightedSum / totalWeight : 5.0

        // Combine uncertainties (propagate errors)
        var uncertaintySquaredSum = 0.0
        for prediction in predictions {
            let weight = weights[prediction.engine] ?? 0
            uncertaintySquaredSum += pow(prediction.uncertainty * weight, 2)
        }
        let combinedUncertainty = sqrt(uncertaintySquaredSum)

        // 95% CI
        let ci95Lower = ensemblePrediction - 1.96 * combinedUncertainty
        let ci95Upper = ensemblePrediction + 1.96 * combinedUncertainty

        return (
            value: ensemblePrediction,
            uncertainty: combinedUncertainty,
            ci95: (lower: max(0, ci95Lower), upper: min(10, ci95Upper))
        )
    }

    // MARK: - Conflict Resolution

    /// When engines significantly disagree
    private func resolveConflict(predictions: [EnginePrediction]) -> ConflictResolution? {
        guard predictions.count >= 2 else { return nil }

        let values = predictions.map { $0.prediction }
        let range = (values.max() ?? 0) - (values.min() ?? 0)

        // Significant disagreement: range > 2 points
        if range > 2.0 {
            // Find majority agreement
            let median = values.sorted()[values.count / 2]
            let agreeing = predictions.filter { abs($0.prediction - median) < 1.0 }
            let disagreeing = predictions.filter { abs($0.prediction - median) >= 1.0 }

            return ConflictResolution(
                hasConflict: true,
                range: range,
                agreeingEngines: agreeing.map { $0.engine },
                disagreeingEngines: disagreeing.map { $0.engine },
                resolution: "Using weighted average - individual engines show different patterns",
                recommendation: "Track more data to improve agreement"
            )
        }

        return nil
    }
}
```

### 3.2 Ensemble Output

```swift
public struct EnsembleResult {
    public let prediction: Double
    public let uncertainty: Double
    public let ci95: (lower: Double, upper: Double)
    public let enginePredictions: [EnginePrediction]
    public let weights: [EngineType: Double]
    public let explanation: UnifiedExplanation
    public let confidence: EnsembleConfidence

    public var predictionDescription: String {
        return """
        Pain prediction: \(String(format: "%.1f", prediction)) ± \(String(format: "%.1f", uncertainty))
        95% confidence: \(String(format: "%.1f", ci95.lower)) - \(String(format: "%.1f", ci95.upper))
        """
    }
}

public struct EnginePrediction {
    public let engine: EngineType
    public let prediction: Double
    public let uncertainty: Double
    public let confidence: String
    public let explanation: String
}

public enum EngineType: String, CaseIterable {
    case statistical = "Statistical Analysis"
    case knn = "Similar Days (k-NN)"
    case neural = "Neural Network"

    public var icon: String {
        switch self {
        case .statistical: return "chart.bar.xaxis"
        case .knn: return "person.3.fill"
        case .neural: return "brain.head.profile"
        }
    }
}

public struct UnifiedExplanation {
    public let summary: String
    public let topFactors: [ExplanationFactor]
    public let engineBreakdown: [EngineExplanation]
    public let recommendations: [String]
}

public struct ExplanationFactor {
    public let factor: String
    public let impact: Double
    public let source: EngineType
    public let confidence: String
}

public struct EngineExplanation {
    public let engine: EngineType
    public let weight: Double
    public let explanation: String
}

public struct ConflictResolution {
    public let hasConflict: Bool
    public let range: Double
    public let agreeingEngines: [EngineType]
    public let disagreeingEngines: [EngineType]
    public let resolution: String
    public let recommendation: String
}

public enum EnsembleConfidence: String {
    case veryHigh = "Very High"   // All engines agree, low uncertainty
    case high = "High"            // Most engines agree
    case medium = "Medium"        // Some disagreement
    case low = "Low"              // Significant disagreement or high uncertainty
}
```

---

## 4. Progressive Activation

### 4.1 Activation Timeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      PROGRESSIVE ENGINE ACTIVATION                          │
│                                                                             │
│  Days:   0    7    14    21    30    60    90    120    180    365         │
│          │    │     │     │     │     │     │      │      │      │         │
│          ▼    ▼     ▼     ▼     ▼     ▼     ▼      ▼      ▼      ▼         │
│                                                                             │
│  STATISTICAL ENGINE ════════════════════════════════════════════════════   │
│  [Always Active - Foundation Layer]                                         │
│                                                                             │
│  Day 7:  Basic correlations (same-day only)                                │
│  Day 14: Add lag analysis (0h, 24h)                                        │
│  Day 21: Add 48h lag, effect sizes                                         │
│  Day 30: Full lag suite (0h-72h), partial correlations                     │
│  Day 60: Bonferroni correction meaningful, high confidence possible        │
│  Day 90+: Robust long-term patterns                                        │
│                                                                             │
│  K-NN ENGINE ───────────────────────────────────────────────────────────   │
│  [Activates at 30 days]                                                     │
│                                                                             │
│  Day 30: Initial activation (limited neighbor pool)                        │
│  Day 60: Better coverage of trigger combinations                           │
│  Day 90: Good seasonal coverage begins                                     │
│  Day 180: Full seasonal cycle captured                                     │
│  Day 365: Complete year of patterns                                        │
│                                                                             │
│  NEURAL ENGINE ────────────────────────────────────────────────────────────│
│  [Activates at 90 days, requires opt-in]                                   │
│                                                                             │
│  Day 90:  Initial activation (transfer learning only)                      │
│  Day 120: First on-device training possible                                │
│  Day 180: Personalization deepens                                          │
│  Day 365: Full personalized model                                          │
│                                                                             │
│  ENSEMBLE WEIGHTS ═════════════════════════════════════════════════════    │
│                                                                             │
│  Days 0-29:   Statistical: 100%                                            │
│  Days 30-59:  Statistical: 60%  |  k-NN: 40%                              │
│  Days 60-89:  Statistical: 50%  |  k-NN: 50%                              │
│  Days 90-179: Statistical: 40%  |  k-NN: 30%  |  Neural: 30%              │
│  Days 180+:   Statistical: 30%  |  k-NN: 30%  |  Neural: 40%              │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Activation Manager

```swift
// File: Core/ML/TriggerAnalysis/EngineActivationManager.swift

@MainActor
public final class EngineActivationManager: ObservableObject {

    // MARK: - Published State
    @Published public private(set) var currentPhase: ActivationPhase = .bootstrap
    @Published public private(set) var daysOfData: Int = 0
    @Published public private(set) var activeCapabilities: Set<Capability> = []
    @Published public private(set) var nextMilestone: Milestone?

    // MARK: - Phases

    public enum ActivationPhase: String, CaseIterable {
        case bootstrap = "Bootstrap"           // 0-6 days
        case earlyLearning = "Early Learning"  // 7-29 days
        case personalization = "Personalization" // 30-89 days
        case advanced = "Advanced"             // 90+ days

        public var description: String {
            switch self {
            case .bootstrap:
                return "Collecting baseline data"
            case .earlyLearning:
                return "Identifying initial patterns"
            case .personalization:
                return "Building your personal profile"
            case .advanced:
                return "Deep pattern analysis active"
            }
        }
    }

    // MARK: - Capabilities

    public enum Capability: String, CaseIterable {
        case basicCorrelation = "Basic Correlations"
        case lagAnalysis = "Lag Analysis (Delayed Effects)"
        case effectSize = "Effect Size Calculation"
        case bonferroniCorrection = "Multiple Comparison Correction"
        case partialCorrelation = "Confound Control"
        case similarDays = "Similar Day Matching"
        case tomorrowPrediction = "Tomorrow's Pain Prediction"
        case neuralPatterns = "Complex Pattern Detection"
        case uncertaintyQuantification = "Prediction Confidence Ranges"

        public var requiredDays: Int {
            switch self {
            case .basicCorrelation: return 7
            case .lagAnalysis: return 14
            case .effectSize: return 21
            case .bonferroniCorrection: return 30
            case .partialCorrelation: return 30
            case .similarDays: return 30
            case .tomorrowPrediction: return 30
            case .neuralPatterns: return 90
            case .uncertaintyQuantification: return 90
            }
        }
    }

    // MARK: - Milestones

    public struct Milestone {
        public let daysRequired: Int
        public let name: String
        public let capabilities: [Capability]
        public let daysRemaining: Int
    }

    // MARK: - Update

    public func updateActivation(daysOfData: Int) {
        self.daysOfData = daysOfData

        // Update phase
        currentPhase = determinePhase(daysOfData: daysOfData)

        // Update capabilities
        activeCapabilities = Set(Capability.allCases.filter {
            $0.requiredDays <= daysOfData
        })

        // Calculate next milestone
        nextMilestone = calculateNextMilestone(daysOfData: daysOfData)
    }

    private func determinePhase(daysOfData: Int) -> ActivationPhase {
        switch daysOfData {
        case 0..<7: return .bootstrap
        case 7..<30: return .earlyLearning
        case 30..<90: return .personalization
        default: return .advanced
        }
    }

    private func calculateNextMilestone(daysOfData: Int) -> Milestone? {
        let milestones: [(days: Int, name: String, capabilities: [Capability])] = [
            (7, "First Correlations", [.basicCorrelation]),
            (14, "Delayed Effect Detection", [.lagAnalysis]),
            (21, "Clinical Effect Sizes", [.effectSize]),
            (30, "Personal Trigger Profile", [.bonferroniCorrection, .partialCorrelation, .similarDays, .tomorrowPrediction]),
            (90, "Advanced AI Analysis", [.neuralPatterns, .uncertaintyQuantification])
        ]

        for milestone in milestones {
            if milestone.days > daysOfData {
                return Milestone(
                    daysRequired: milestone.days,
                    name: milestone.name,
                    capabilities: milestone.capabilities,
                    daysRemaining: milestone.days - daysOfData
                )
            }
        }

        return nil  // All milestones achieved
    }

    // MARK: - User Messaging

    public var progressMessage: String {
        if let milestone = nextMilestone {
            return "Track \(milestone.daysRemaining) more days to unlock: \(milestone.name)"
        } else {
            return "All analysis capabilities unlocked!"
        }
    }

    public var phaseMessage: String {
        switch currentPhase {
        case .bootstrap:
            return "Keep logging daily to build your baseline"
        case .earlyLearning:
            return "We're starting to see patterns in your data"
        case .personalization:
            return "Your personal trigger profile is taking shape"
        case .advanced:
            return "Full analysis suite active - deep insights available"
        }
    }
}
```

---

## 5. Data Schema

### 5.1 Core Data Entities

```swift
// MARK: - TriggerLog Entity

/// Tracks specific trigger events (coffee, alcohol, stress, etc.)
@objc(TriggerLog)
public class TriggerLog: NSManagedObject {

    // MARK: - Attributes
    @NSManaged public var id: UUID
    @NSManaged public var timestamp: Date
    @NSManaged public var triggerCategory: String     // food, sleep, activity, weather, stress, medication
    @NSManaged public var triggerName: String         // coffee, alcohol, dairy, etc.
    @NSManaged public var triggerValue: Double        // Quantity or intensity (0-10)
    @NSManaged public var triggerUnit: String?        // cups, hours, steps, etc.
    @NSManaged public var isBinary: Bool              // true = yes/no, false = continuous
    @NSManaged public var notes: String?
    @NSManaged public var source: String              // manual, healthkit, weather

    // MARK: - Relationships
    @NSManaged public var symptomLog: SymptomLog?     // Link to same-day symptoms

    // MARK: - Computed Properties
    public var category: TriggerCategory {
        TriggerCategory(rawValue: triggerCategory) ?? .other
    }

    public var isPresent: Bool {
        isBinary ? triggerValue > 0 : triggerValue > 0
    }
}

// MARK: - TriggerAnalysisCache Entity

/// Caches expensive analysis results
@objc(TriggerAnalysisCache)
public class TriggerAnalysisCache: NSManagedObject {

    @NSManaged public var id: UUID
    @NSManaged public var analysisDate: Date
    @NSManaged public var triggerName: String
    @NSManaged public var correlationData: Data       // Encoded [LaggedCorrelationResult]
    @NSManaged public var effectSizeData: Data        // Encoded EffectSize
    @NSManaged public var confidence: String
    @NSManaged public var daysAnalyzed: Int32
    @NSManaged public var isValid: Bool               // Invalidate when new data arrives
}

// MARK: - KNNTrainingDay Entity

/// Stores training data for k-NN model
@objc(KNNTrainingDay)
public class KNNTrainingDay: NSManagedObject {

    @NSManaged public var id: UUID
    @NSManaged public var date: Date
    @NSManaged public var painLevel: Double
    @NSManaged public var featureVector: Data         // Encoded [Double]
    @NSManaged public var activeTriggers: Data        // Encoded [TriggerValue]
    @NSManaged public var usedInTraining: Bool
}

// MARK: - NeuralModelVersion Entity

/// Tracks neural model versions and training history
@objc(NeuralModelVersion)
public class NeuralModelVersion: NSManagedObject {

    @NSManaged public var id: UUID
    @NSManaged public var version: Int32
    @NSManaged public var trainedDate: Date
    @NSManaged public var trainingDays: Int32
    @NSManaged public var trainingLoss: Double
    @NSManaged public var validationLoss: Double
    @NSManaged public var modelPath: String
    @NSManaged public var isActive: Bool
}
```

### 5.2 Trigger Categories & Types

```swift
// File: Core/ML/TriggerAnalysis/TriggerTypes.swift

public enum TriggerCategory: String, CaseIterable, Codable {
    case food = "food"
    case sleep = "sleep"
    case activity = "activity"
    case weather = "weather"
    case stress = "stress"
    case medication = "medication"
    case other = "other"

    public var icon: String {
        switch self {
        case .food: return "fork.knife"
        case .sleep: return "bed.double.fill"
        case .activity: return "figure.run"
        case .weather: return "cloud.sun.fill"
        case .stress: return "brain.head.profile"
        case .medication: return "pills.fill"
        case .other: return "ellipsis.circle"
        }
    }

    public var displayName: String {
        switch self {
        case .food: return "Food & Drink"
        case .sleep: return "Sleep"
        case .activity: return "Physical Activity"
        case .weather: return "Weather"
        case .stress: return "Stress & Mental"
        case .medication: return "Medication"
        case .other: return "Other"
        }
    }
}

public struct TriggerDefinition: Identifiable, Codable {
    public let id: String
    public let name: String
    public let category: TriggerCategory
    public let icon: String
    public let unit: String?
    public let isBinary: Bool
    public let minValue: Double
    public let maxValue: Double
    public let defaultValue: Double
    public let expectedLag: Int  // Hours
    public let dataSource: DataSource
    public let clinicalRelevance: String

    public enum DataSource: String, Codable {
        case manual = "manual"
        case healthKit = "healthkit"
        case weather = "weather"
        case medication = "medication"
    }
}

// MARK: - Default Trigger Definitions

public let defaultTriggerDefinitions: [TriggerDefinition] = [
    // Food & Drink
    TriggerDefinition(
        id: "coffee",
        name: "Coffee",
        category: .food,
        icon: "cup.and.saucer.fill",
        unit: "cups",
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 0,
        expectedLag: 24,
        dataSource: .manual,
        clinicalRelevance: "Caffeine may affect sleep quality and inflammation"
    ),
    TriggerDefinition(
        id: "alcohol",
        name: "Alcohol",
        category: .food,
        icon: "wineglass.fill",
        unit: "drinks",
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 0,
        expectedLag: 24,
        dataSource: .manual,
        clinicalRelevance: "Alcohol can affect inflammation and sleep"
    ),
    TriggerDefinition(
        id: "sugar",
        name: "Sugar (High)",
        category: .food,
        icon: "birthday.cake.fill",
        unit: nil,
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLag: 24,
        dataSource: .manual,
        clinicalRelevance: "High sugar intake may promote inflammation"
    ),
    TriggerDefinition(
        id: "dairy",
        name: "Dairy",
        category: .food,
        icon: "mug.fill",
        unit: nil,
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLag: 48,
        dataSource: .manual,
        clinicalRelevance: "Some AS patients report dairy sensitivity"
    ),
    TriggerDefinition(
        id: "gluten",
        name: "Gluten",
        category: .food,
        icon: "leaf.fill",
        unit: nil,
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLag: 48,
        dataSource: .manual,
        clinicalRelevance: "Gluten sensitivity is common in autoimmune conditions"
    ),

    // Sleep
    TriggerDefinition(
        id: "sleep_duration",
        name: "Sleep Duration",
        category: .sleep,
        icon: "moon.zzz.fill",
        unit: "hours",
        isBinary: false,
        minValue: 0,
        maxValue: 14,
        defaultValue: 7,
        expectedLag: 0,
        dataSource: .healthKit,
        clinicalRelevance: "Sleep deprivation increases inflammation markers"
    ),
    TriggerDefinition(
        id: "sleep_quality",
        name: "Sleep Quality",
        category: .sleep,
        icon: "bed.double.fill",
        unit: nil,
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 5,
        expectedLag: 0,
        dataSource: .manual,
        clinicalRelevance: "Poor sleep quality correlates with pain intensity"
    ),

    // Activity
    TriggerDefinition(
        id: "steps",
        name: "Daily Steps",
        category: .activity,
        icon: "figure.walk",
        unit: "steps",
        isBinary: false,
        minValue: 0,
        maxValue: 50000,
        defaultValue: 0,
        expectedLag: 0,
        dataSource: .healthKit,
        clinicalRelevance: "Activity levels affect joint mobility"
    ),
    TriggerDefinition(
        id: "prolonged_sitting",
        name: "Prolonged Sitting",
        category: .activity,
        icon: "chair.fill",
        unit: nil,
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLag: 24,
        dataSource: .manual,
        clinicalRelevance: "Prolonged sitting worsens spinal stiffness"
    ),
    TriggerDefinition(
        id: "exercise",
        name: "Exercise",
        category: .activity,
        icon: "figure.strengthtraining.traditional",
        unit: "minutes",
        isBinary: false,
        minValue: 0,
        maxValue: 300,
        defaultValue: 0,
        expectedLag: 24,
        dataSource: .healthKit,
        clinicalRelevance: "Regular exercise improves AS symptoms"
    ),

    // Weather
    TriggerDefinition(
        id: "pressure_drop",
        name: "Pressure Drop",
        category: .weather,
        icon: "barometer",
        unit: "hPa",
        isBinary: false,
        minValue: -30,
        maxValue: 30,
        defaultValue: 0,
        expectedLag: 12,
        dataSource: .weather,
        clinicalRelevance: "Barometric drops strongly correlate with AS flares"
    ),
    TriggerDefinition(
        id: "humidity",
        name: "High Humidity",
        category: .weather,
        icon: "humidity.fill",
        unit: "%",
        isBinary: false,
        minValue: 0,
        maxValue: 100,
        defaultValue: 50,
        expectedLag: 0,
        dataSource: .weather,
        clinicalRelevance: "High humidity may increase joint stiffness"
    ),
    TriggerDefinition(
        id: "cold_temperature",
        name: "Cold Temperature",
        category: .weather,
        icon: "thermometer.snowflake",
        unit: "°C",
        isBinary: false,
        minValue: -30,
        maxValue: 50,
        defaultValue: 20,
        expectedLag: 0,
        dataSource: .weather,
        clinicalRelevance: "Cold weather can worsen joint pain"
    ),

    // Stress
    TriggerDefinition(
        id: "stress",
        name: "Stress Level",
        category: .stress,
        icon: "brain.head.profile",
        unit: nil,
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 0,
        expectedLag: 24,
        dataSource: .manual,
        clinicalRelevance: "Stress triggers inflammatory pathways"
    ),
    TriggerDefinition(
        id: "work_hours",
        name: "Long Work Day",
        category: .stress,
        icon: "briefcase.fill",
        unit: "hours",
        isBinary: false,
        minValue: 0,
        maxValue: 24,
        defaultValue: 8,
        expectedLag: 24,
        dataSource: .manual,
        clinicalRelevance: "Overwork can trigger fatigue and flares"
    ),

    // Medication
    TriggerDefinition(
        id: "missed_medication",
        name: "Missed Medication",
        category: .medication,
        icon: "pills.fill",
        unit: nil,
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLag: 24,
        dataSource: .medication,
        clinicalRelevance: "Medication adherence is crucial for symptom control"
    )
]
```

---

## 6. Feature Engineering Pipeline

### 6.1 Feature Categories

```swift
// File: Core/ML/TriggerAnalysis/TriggerFeatureExtractor.swift

@MainActor
public final class TriggerFeatureExtractor {

    // MARK: - Feature Extraction

    /// Extract all features for trigger analysis
    public func extractFeatures(
        symptomLogs: [SymptomLog],
        triggerLogs: [TriggerLog],
        healthKitData: HealthKitDataBundle,
        weatherData: WeatherDataBundle
    ) -> [DayFeatures] {

        var allFeatures: [DayFeatures] = []

        for (index, log) in symptomLogs.enumerated() {
            let date = log.timestamp ?? Date()
            var features = DayFeatures(date: date)

            // 1. RAW FEATURES (from existing 92-feature extractor)
            features.rawFeatures = extractRawFeatures(log: log)

            // 2. TRIGGER FEATURES
            features.triggerFeatures = extractTriggerFeatures(
                date: date,
                triggerLogs: triggerLogs
            )

            // 3. LAG FEATURES (t-1, t-2, t-3)
            features.lagFeatures = extractLagFeatures(
                index: index,
                logs: symptomLogs,
                triggerLogs: triggerLogs
            )

            // 4. ROLLING FEATURES (3d, 7d, 14d averages)
            features.rollingFeatures = extractRollingFeatures(
                index: index,
                logs: symptomLogs
            )

            // 5. DELTA FEATURES (day-over-day changes)
            features.deltaFeatures = extractDeltaFeatures(
                index: index,
                logs: symptomLogs
            )

            // 6. INTERACTION FEATURES
            features.interactionFeatures = extractInteractionFeatures(
                log: log,
                triggerLogs: triggerLogs.filter { Calendar.current.isDate($0.timestamp, inSameDayAs: date) }
            )

            // 7. TEMPORAL FEATURES
            features.temporalFeatures = extractTemporalFeatures(date: date)

            // 8. TARGET (pain level)
            features.painLevel = log.basdaiScore

            allFeatures.append(features)
        }

        return allFeatures
    }

    // MARK: - Raw Features

    private func extractRawFeatures(log: SymptomLog) -> [String: Double] {
        var features: [String: Double] = [:]

        // Pain metrics
        features["current_pain"] = log.basdaiScore
        features["fatigue"] = Double(log.fatigueLevel)
        features["morning_stiffness"] = Double(log.morningStiffnessMinutes)
        features["sleep_quality"] = Double(log.sleepQuality)

        // Biometrics (from context snapshot)
        if let context = log.contextSnapshot {
            features["hrv"] = context.hrvValue
            features["resting_hr"] = Double(context.restingHeartRate)
            features["steps"] = Double(context.stepCount)
            features["sleep_hours"] = context.sleepDuration / 3600

            // Weather
            features["pressure"] = context.barometricPressure
            features["pressure_change"] = context.pressureChange12h
            features["humidity"] = Double(context.humidity)
            features["temperature"] = context.temperature
        }

        return features
    }

    // MARK: - Trigger Features

    private func extractTriggerFeatures(
        date: Date,
        triggerLogs: [TriggerLog]
    ) -> [String: Double] {

        var features: [String: Double] = [:]

        // Get triggers for this day
        let dayTriggers = triggerLogs.filter {
            Calendar.current.isDate($0.timestamp, inSameDayAs: date)
        }

        // Aggregate by trigger name
        for trigger in dayTriggers {
            let key = "trigger_\(trigger.triggerName)"
            features[key] = trigger.triggerValue
        }

        // Set missing triggers to 0
        for definition in defaultTriggerDefinitions {
            let key = "trigger_\(definition.id)"
            if features[key] == nil {
                features[key] = definition.isBinary ? 0 : definition.defaultValue
            }
        }

        return features
    }

    // MARK: - Lag Features

    private func extractLagFeatures(
        index: Int,
        logs: [SymptomLog],
        triggerLogs: [TriggerLog]
    ) -> [String: Double] {

        var features: [String: Double] = [:]

        for lag in 1...3 {
            guard index >= lag else { continue }

            let lagLog = logs[index - lag]
            let lagDate = lagLog.timestamp ?? Date()

            // Lagged pain
            features["pain_lag\(lag)"] = lagLog.basdaiScore

            // Lagged sleep
            features["sleep_lag\(lag)"] = lagLog.sleepDurationHours

            // Lagged triggers
            let lagTriggers = triggerLogs.filter {
                Calendar.current.isDate($0.timestamp, inSameDayAs: lagDate)
            }
            for trigger in lagTriggers {
                features["trigger_\(trigger.triggerName)_lag\(lag)"] = trigger.triggerValue
            }
        }

        return features
    }

    // MARK: - Rolling Features

    private func extractRollingFeatures(
        index: Int,
        logs: [SymptomLog]
    ) -> [String: Double] {

        var features: [String: Double] = [:]

        // 3-day rolling
        if index >= 2 {
            let window3 = Array(logs[(index-2)...index])
            features["pain_3d_avg"] = window3.map { $0.basdaiScore }.mean()
            features["pain_3d_std"] = window3.map { $0.basdaiScore }.standardDeviation()
        }

        // 7-day rolling
        if index >= 6 {
            let window7 = Array(logs[(index-6)...index])
            features["pain_7d_avg"] = window7.map { $0.basdaiScore }.mean()
            features["pain_7d_std"] = window7.map { $0.basdaiScore }.standardDeviation()
            features["pain_7d_min"] = window7.map { $0.basdaiScore }.min() ?? 0
            features["pain_7d_max"] = window7.map { $0.basdaiScore }.max() ?? 0

            features["fatigue_7d_avg"] = window7.map { Double($0.fatigueLevel) }.mean()
            features["sleep_7d_avg"] = window7.map { $0.sleepDurationHours }.mean()
        }

        // 14-day rolling
        if index >= 13 {
            let window14 = Array(logs[(index-13)...index])
            features["pain_14d_avg"] = window14.map { $0.basdaiScore }.mean()
            features["pain_14d_trend"] = calculateTrend(window14.map { $0.basdaiScore })
        }

        return features
    }

    // MARK: - Delta Features

    private func extractDeltaFeatures(
        index: Int,
        logs: [SymptomLog]
    ) -> [String: Double] {

        var features: [String: Double] = [:]

        guard index >= 1 else { return features }

        let today = logs[index]
        let yesterday = logs[index - 1]

        // Day-over-day changes
        features["pain_delta"] = today.basdaiScore - yesterday.basdaiScore
        features["fatigue_delta"] = Double(today.fatigueLevel - yesterday.fatigueLevel)
        features["sleep_delta"] = today.sleepDurationHours - yesterday.sleepDurationHours

        // Weather changes
        if let todayContext = today.contextSnapshot,
           let yesterdayContext = yesterday.contextSnapshot {
            features["pressure_delta"] = todayContext.barometricPressure - yesterdayContext.barometricPressure
            features["humidity_delta"] = Double(todayContext.humidity - yesterdayContext.humidity)
            features["temp_delta"] = todayContext.temperature - yesterdayContext.temperature
        }

        return features
    }

    // MARK: - Interaction Features

    private func extractInteractionFeatures(
        log: SymptomLog,
        triggerLogs: [TriggerLog]
    ) -> [String: Double] {

        var features: [String: Double] = [:]

        // Get trigger values
        let coffee = triggerLogs.first { $0.triggerName == "coffee" }?.triggerValue ?? 0
        let stress = Double(log.stressLevel ?? 0)
        let sleepHours = log.sleepDurationHours
        let steps = Double(log.contextSnapshot?.stepCount ?? 0)

        // Interaction terms
        features["coffee_x_stress"] = coffee * stress / 10.0
        features["coffee_x_sleep_deficit"] = coffee * max(0, 7 - sleepHours)
        features["stress_x_sleep_deficit"] = stress * max(0, 7 - sleepHours) / 10.0
        features["low_sleep_x_low_activity"] = (sleepHours < 6 ? 1.0 : 0.0) * (steps < 3000 ? 1.0 : 0.0)
        features["high_stress_x_low_sleep"] = (stress > 7 ? 1.0 : 0.0) * (sleepHours < 6 ? 1.0 : 0.0)

        return features
    }

    // MARK: - Temporal Features

    private func extractTemporalFeatures(date: Date) -> [String: Double] {
        var features: [String: Double] = [:]

        let calendar = Calendar.current
        let weekday = calendar.component(.weekday, from: date)
        let month = calendar.component(.month, from: date)
        let day = calendar.component(.day, from: date)

        // Day of week (one-hot)
        for i in 1...7 {
            features["dow_\(i)"] = weekday == i ? 1.0 : 0.0
        }

        // Weekend flag
        features["is_weekend"] = (weekday == 1 || weekday == 7) ? 1.0 : 0.0

        // Month (cyclical encoding)
        features["month_sin"] = sin(2 * .pi * Double(month) / 12.0)
        features["month_cos"] = cos(2 * .pi * Double(month) / 12.0)

        // Day of month (cyclical encoding)
        features["day_sin"] = sin(2 * .pi * Double(day) / 31.0)
        features["day_cos"] = cos(2 * .pi * Double(day) / 31.0)

        return features
    }

    // MARK: - Utility

    private func calculateTrend(_ values: [Double]) -> Double {
        guard values.count >= 2 else { return 0 }

        let n = Double(values.count)
        let x = Array(0..<values.count).map { Double($0) }
        let sumX = x.reduce(0, +)
        let sumY = values.reduce(0, +)
        let sumXY = zip(x, values).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)

        let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        return slope
    }
}

// MARK: - Day Features Structure

public struct DayFeatures {
    public let date: Date
    public var rawFeatures: [String: Double] = [:]
    public var triggerFeatures: [String: Double] = [:]
    public var lagFeatures: [String: Double] = [:]
    public var rollingFeatures: [String: Double] = [:]
    public var deltaFeatures: [String: Double] = [:]
    public var interactionFeatures: [String: Double] = [:]
    public var temporalFeatures: [String: Double] = [:]
    public var painLevel: Double = 0

    /// All features combined
    public var allFeatures: [String: Double] {
        var all: [String: Double] = [:]
        all.merge(rawFeatures) { $1 }
        all.merge(triggerFeatures) { $1 }
        all.merge(lagFeatures) { $1 }
        all.merge(rollingFeatures) { $1 }
        all.merge(deltaFeatures) { $1 }
        all.merge(interactionFeatures) { $1 }
        all.merge(temporalFeatures) { $1 }
        return all
    }

    /// Feature vector for ML models
    public func toVector(featureOrder: [String]) -> [Double] {
        let all = allFeatures
        return featureOrder.map { all[$0] ?? 0 }
    }
}
```

---

## 7. Confidence & Uncertainty System

### 7.1 Multi-Level Confidence

```swift
// File: Core/ML/TriggerAnalysis/ConfidenceSystem.swift

/// Unified confidence assessment across all engines
public struct ConfidenceAssessment {

    // MARK: - Per-Trigger Confidence

    public struct TriggerConfidenceMetrics {
        public let triggerName: String
        public let sampleSize: Int
        public let triggerDays: Int
        public let statisticalPValue: Double
        public let correlationStrength: Double
        public let effectSize: Double
        public let dataQuality: Double  // 0-1
        public let overallConfidence: TriggerConfidence

        public init(
            triggerName: String,
            sampleSize: Int,
            triggerDays: Int,
            statisticalPValue: Double,
            correlationStrength: Double,
            effectSize: Double,
            dataQuality: Double
        ) {
            self.triggerName = triggerName
            self.sampleSize = sampleSize
            self.triggerDays = triggerDays
            self.statisticalPValue = statisticalPValue
            self.correlationStrength = correlationStrength
            self.effectSize = effectSize
            self.dataQuality = dataQuality

            // Calculate overall confidence
            self.overallConfidence = Self.calculateConfidence(
                sampleSize: sampleSize,
                triggerDays: triggerDays,
                pValue: statisticalPValue,
                correlation: correlationStrength,
                effectSize: effectSize,
                dataQuality: dataQuality
            )
        }

        private static func calculateConfidence(
            sampleSize: Int,
            triggerDays: Int,
            pValue: Double,
            correlation: Double,
            effectSize: Double,
            dataQuality: Double
        ) -> TriggerConfidence {

            // Insufficient data
            if triggerDays < 7 || sampleSize < 14 {
                return .insufficient
            }

            // Calculate confidence score (0-100)
            var score = 0.0

            // Sample size contribution (0-25 points)
            score += min(25, Double(sampleSize) / 4)

            // Trigger days contribution (0-15 points)
            score += min(15, Double(triggerDays) / 2)

            // P-value contribution (0-25 points)
            if pValue < 0.001 {
                score += 25
            } else if pValue < 0.01 {
                score += 20
            } else if pValue < 0.05 {
                score += 15
            }

            // Correlation strength contribution (0-20 points)
            score += min(20, abs(correlation) * 25)

            // Effect size contribution (0-10 points)
            score += min(10, abs(effectSize) * 12.5)

            // Data quality contribution (0-5 points)
            score += dataQuality * 5

            // Map to confidence level
            if score >= 70 && pValue < 0.01 && abs(correlation) > 0.5 {
                return .high
            } else if score >= 45 && pValue < 0.05 && abs(correlation) > 0.3 {
                return .medium
            } else if pValue < 0.05 {
                return .low
            } else {
                return .insufficient
            }
        }
    }

    // MARK: - Prediction Confidence

    public struct PredictionConfidenceMetrics {
        public let engineAgreement: Double        // 0-1, how much engines agree
        public let uncertaintyLevel: Double       // Combined uncertainty
        public let dataRecency: Double            // 0-1, how recent is training data
        public let featureAvailability: Double    // 0-1, how many features have data
        public let historicalAccuracy: Double     // 0-1, past prediction accuracy
        public let overallConfidence: PredictionConfidence

        public enum PredictionConfidence: String {
            case veryHigh = "Very High"
            case high = "High"
            case medium = "Medium"
            case low = "Low"
            case veryLow = "Very Low"

            public var color: String {
                switch self {
                case .veryHigh: return "green"
                case .high: return "teal"
                case .medium: return "yellow"
                case .low: return "orange"
                case .veryLow: return "red"
                }
            }
        }
    }

    // MARK: - Confidence Display

    public static func confidenceExplanation(
        for triggerMetrics: TriggerConfidenceMetrics
    ) -> String {

        var explanations: [String] = []

        // Sample size
        if triggerMetrics.sampleSize < 30 {
            explanations.append("Limited data (\(triggerMetrics.sampleSize) days)")
        } else if triggerMetrics.sampleSize >= 90 {
            explanations.append("Strong data foundation (\(triggerMetrics.sampleSize) days)")
        }

        // Trigger frequency
        if triggerMetrics.triggerDays < 14 {
            explanations.append("Need more \(triggerMetrics.triggerName) days (\(triggerMetrics.triggerDays) logged)")
        }

        // Statistical significance
        if triggerMetrics.statisticalPValue >= 0.05 {
            explanations.append("Not yet statistically significant (p=\(String(format: "%.3f", triggerMetrics.statisticalPValue)))")
        } else if triggerMetrics.statisticalPValue < 0.01 {
            explanations.append("Highly significant (p<0.01)")
        }

        // Effect size
        let absEffect = abs(triggerMetrics.effectSize)
        if absEffect >= 0.8 {
            explanations.append("Large effect on symptoms")
        } else if absEffect >= 0.5 {
            explanations.append("Moderate effect on symptoms")
        } else if absEffect >= 0.2 {
            explanations.append("Small but measurable effect")
        }

        return explanations.joined(separator: " | ")
    }
}
```

---

## 8. Explainability Framework

### 8.1 Unified Explanation Generator

```swift
// File: Core/ML/TriggerAnalysis/ExplanationGenerator.swift

@MainActor
public final class UnifiedExplanationGenerator {

    // MARK: - Generate Full Explanation

    public func generateExplanation(
        ensembleResult: EnsembleResult,
        statisticalTriggers: [StatisticalTrigger],
        knnResult: KNNPredictionResult?,
        neuralResult: NeuralPredictionResult?
    ) -> TriggerExplanationBundle {

        // 1. Summary
        let summary = generateSummary(
            prediction: ensembleResult.prediction,
            topTriggers: statisticalTriggers.prefix(3).map { $0.name }
        )

        // 2. Top factors from all sources
        let factors = mergeAndRankFactors(
            statisticalTriggers: statisticalTriggers,
            knnCommonTriggers: knnResult?.commonTriggers ?? [],
            neuralAttributions: neuralResult?.attentionWeights.topFeatures ?? []
        )

        // 3. Per-engine explanations
        let engineExplanations = generateEngineExplanations(
            statisticalTriggers: statisticalTriggers,
            knnResult: knnResult,
            neuralResult: neuralResult,
            weights: ensembleResult.weights
        )

        // 4. Recommendations
        let recommendations = generateRecommendations(
            topTriggers: statisticalTriggers.filter { $0.confidence == .high || $0.confidence == .medium }
        )

        // 5. Uncertainty explanation
        let uncertaintyExplanation = explainUncertainty(
            ci95: ensembleResult.ci95,
            enginePredictions: ensembleResult.enginePredictions
        )

        return TriggerExplanationBundle(
            summary: summary,
            topFactors: factors,
            engineExplanations: engineExplanations,
            recommendations: recommendations,
            uncertaintyExplanation: uncertaintyExplanation,
            dataQualityNote: generateDataQualityNote(ensembleResult: ensembleResult)
        )
    }

    // MARK: - Summary Generation

    private func generateSummary(prediction: Double, topTriggers: [String]) -> String {
        let painLevel = painLevelDescription(prediction)
        let triggerList = topTriggers.isEmpty
            ? "No clear triggers identified yet"
            : topTriggers.joined(separator: ", ")

        return """
        Tomorrow's expected pain: \(String(format: "%.1f", prediction))/10 (\(painLevel))

        Top contributing factors: \(triggerList)
        """
    }

    private func painLevelDescription(_ level: Double) -> String {
        switch level {
        case 0..<2: return "Very Low"
        case 2..<4: return "Low"
        case 4..<6: return "Moderate"
        case 6..<8: return "High"
        default: return "Very High"
        }
    }

    // MARK: - Factor Merging

    private func mergeAndRankFactors(
        statisticalTriggers: [StatisticalTrigger],
        knnCommonTriggers: [CommonTrigger],
        neuralAttributions: [String]
    ) -> [RankedFactor] {

        var factorScores: [String: Double] = [:]
        var factorSources: [String: Set<String>] = [:]

        // Statistical factors (weight: 1.0 for high confidence, 0.6 for medium)
        for trigger in statisticalTriggers {
            let weight = trigger.confidence == .high ? 1.0 : 0.6
            let score = abs(trigger.effectSize.cohenD) * weight
            factorScores[trigger.name, default: 0] += score
            factorSources[trigger.name, default: []].insert("Statistical")
        }

        // k-NN factors (weight by frequency)
        for trigger in knnCommonTriggers {
            let score = trigger.frequency * 0.8
            factorScores[trigger.name, default: 0] += score
            factorSources[trigger.name, default: []].insert("Similar Days")
        }

        // Neural factors (top 5 get decreasing weights)
        for (index, feature) in neuralAttributions.prefix(5).enumerated() {
            let weight = 0.8 - Double(index) * 0.1
            factorScores[feature, default: 0] += weight
            factorSources[feature, default: []].insert("Neural Network")
        }

        // Sort and return top factors
        return factorScores
            .sorted { $0.value > $1.value }
            .prefix(5)
            .map { name, score in
                RankedFactor(
                    name: name,
                    combinedScore: score,
                    sources: Array(factorSources[name] ?? []),
                    description: factorDescription(name: name, triggers: statisticalTriggers)
                )
            }
    }

    // MARK: - Recommendations

    private func generateRecommendations(
        topTriggers: [StatisticalTrigger]
    ) -> [Recommendation] {

        return topTriggers.compactMap { trigger in
            guard trigger.effectSize.meanDifference > 0 else {
                // Protective factor - don't recommend avoiding
                return nil
            }

            let action = recommendedAction(for: trigger.name)
            let impact = "Could reduce pain by ~\(String(format: "%.1f", abs(trigger.effectSize.meanDifference))) points"

            return Recommendation(
                trigger: trigger.name,
                action: action,
                expectedImpact: impact,
                confidence: trigger.confidence,
                timing: "Effect typically seen \(trigger.bestLag?.lagDescription ?? "same day")"
            )
        }
    }

    private func recommendedAction(for trigger: String) -> String {
        switch trigger.lowercased() {
        case "coffee": return "Consider reducing coffee intake"
        case "alcohol": return "Try limiting alcohol consumption"
        case "sleep duration", "sleep_duration": return "Aim for 7+ hours of sleep"
        case "stress", "stress level": return "Practice stress management techniques"
        case "prolonged sitting": return "Take regular movement breaks"
        case "pressure drop": return "Plan for lower activity on pressure drop days"
        default: return "Monitor and consider reducing exposure"
        }
    }
}

// MARK: - Explanation Structures

public struct TriggerExplanationBundle {
    public let summary: String
    public let topFactors: [RankedFactor]
    public let engineExplanations: [EngineExplanation]
    public let recommendations: [Recommendation]
    public let uncertaintyExplanation: String
    public let dataQualityNote: String
}

public struct RankedFactor {
    public let name: String
    public let combinedScore: Double
    public let sources: [String]  // Which engines identified this
    public let description: String
}

public struct Recommendation {
    public let trigger: String
    public let action: String
    public let expectedImpact: String
    public let confidence: TriggerConfidence
    public let timing: String
}

public struct EngineExplanation {
    public let engine: EngineType
    public let weight: Double
    public let explanation: String
    public let keyFindings: [String]
}
```

---

## 9. Implementation Architecture

### 9.1 File Structure

```
Core/ML/TriggerAnalysis/
├── README.md
│
├── Engines/
│   ├── StatisticalTriggerEngine.swift      # Phase 1: Correlation analysis
│   ├── KNNTriggerEngine.swift              # Phase 2: Similar day matching
│   └── NeuralTriggerEngine.swift           # Phase 3: Deep learning
│
├── Ensemble/
│   ├── TriggerEnsemble.swift               # Combines all engines
│   ├── EnsembleWeightCalculator.swift      # Dynamic weight calculation
│   └── ConflictResolver.swift              # Handles engine disagreements
│
├── Features/
│   ├── TriggerFeatureExtractor.swift       # Feature engineering
│   ├── FeatureScaler.swift                 # Min-max normalization
│   └── FeatureDefinitions.swift            # Feature metadata
│
├── Activation/
│   ├── EngineActivationManager.swift       # Progressive activation
│   └── ActivationRules.swift               # Activation criteria
│
├── Confidence/
│   ├── ConfidenceSystem.swift              # Confidence calculation
│   └── UncertaintyQuantification.swift     # MC Dropout, intervals
│
├── Explainability/
│   ├── UnifiedExplanationGenerator.swift   # Combined explanations
│   ├── StatisticalExplainer.swift          # Correlation-based
│   ├── KNNExplainer.swift                  # Similar days
│   └── NeuralExplainer.swift               # Feature attribution
│
├── Types/
│   ├── TriggerTypes.swift                  # Enums, categories
│   ├── TriggerDefinitions.swift            # Default triggers
│   ├── StatisticalTypes.swift              # Correlation results
│   ├── KNNTypes.swift                      # Similar day types
│   ├── NeuralTypes.swift                   # Neural output types
│   └── EnsembleTypes.swift                 # Combined results
│
├── Persistence/
│   ├── TriggerLog+CoreData.swift           # Core Data entity
│   ├── AnalysisCache+CoreData.swift        # Cached results
│   └── ModelVersionManager.swift           # Neural model versions
│
├── Models/
│   ├── PopulationTriggerModel.mlpackage    # Pre-trained base model
│   └── TriggerKNN.mlmodel                  # k-NN template
│
└── Tests/
    ├── StatisticalEngineTests.swift
    ├── KNNEngineTests.swift
    ├── NeuralEngineTests.swift
    ├── EnsembleTests.swift
    └── FeatureExtractorTests.swift
```

### 9.2 Integration Points

```swift
// Integration with existing UnifiedNeuralEngine

// In UnifiedNeuralEngine.swift, add:

extension UnifiedNeuralEngine {

    /// Access to trigger analysis (new hybrid system)
    public var triggerAnalysis: TriggerEnsemble {
        TriggerEnsemble.shared
    }

    /// Run comprehensive trigger analysis
    public func analyzeTriggers() async -> TriggerAnalysisResult {
        let symptomLogs = await fetchSymptomLogs()
        let triggerLogs = await fetchTriggerLogs()

        return await triggerAnalysis.analyzeAll(
            symptomLogs: symptomLogs,
            triggerLogs: triggerLogs
        )
    }
}
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```swift
// Example: StatisticalEngineTests.swift

@MainActor
final class StatisticalEngineTests: XCTestCase {

    var engine: StatisticalTriggerEngine!

    override func setUp() async throws {
        engine = StatisticalTriggerEngine()
    }

    // MARK: - Correlation Tests

    func testPearsonCorrelation_perfectPositive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [2.0, 4.0, 6.0, 8.0, 10.0]

        let r = engine.pearsonCorrelation(x, y)

        XCTAssertNotNil(r)
        XCTAssertEqual(r!, 1.0, accuracy: 0.001)
    }

    func testPearsonCorrelation_perfectNegative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [10.0, 8.0, 6.0, 4.0, 2.0]

        let r = engine.pearsonCorrelation(x, y)

        XCTAssertNotNil(r)
        XCTAssertEqual(r!, -1.0, accuracy: 0.001)
    }

    func testLaggedCorrelation_detectsDelayedEffect() async {
        // Trigger on day t affects symptom on day t+1
        let trigger = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        let symptom = [5.0, 8.0, 5.0, 8.0, 5.0, 8.0, 5.0]  // Pain high day after trigger

        let results = engine.analyzeLaggedCorrelation(
            triggerValues: Array(trigger.dropLast()),
            symptomValues: Array(symptom.dropFirst()),
            triggerName: "test",
            maxLag: 2
        )

        // Lag 1 should show strongest correlation
        let lag1 = results.first { $0.lag == 1 }
        XCTAssertNotNil(lag1)
        XCTAssertGreaterThan(abs(lag1!.correlation), 0.8)
    }

    // MARK: - Effect Size Tests

    func testEffectSize_largeEffect() {
        let withTrigger = [7.0, 8.0, 7.5, 8.5, 7.0]      // Mean ~7.6
        let withoutTrigger = [3.0, 4.0, 3.5, 4.5, 3.0]   // Mean ~3.6

        let effect = engine.calculateEffectSize(
            triggerDays: withTrigger,
            nonTriggerDays: withoutTrigger
        )

        XCTAssertGreaterThan(effect.cohenD, 0.8)  // Large effect
        XCTAssertTrue(effect.clinicallySignificant)
    }

    // MARK: - Bonferroni Correction Tests

    func testBonferroniCorrection_reducesAlpha() {
        let triggers = (0..<10).map { i in
            StatisticalTrigger(
                id: UUID(),
                name: "trigger_\(i)",
                category: .food,
                icon: "test",
                laggedResults: [],
                bestLag: nil,
                effectSize: EffectSize(
                    meanWithTrigger: 6.0,
                    meanWithoutTrigger: 5.0,
                    meanDifference: 1.0,
                    pooledStandardDeviation: 1.5,
                    cohenD: 0.67,
                    percentChange: 20,
                    clinicallySignificant: true
                ),
                rawPValue: 0.04,  // Would be significant at α=0.05
                correctedPValue: 0.04,
                isSignificant: true,
                confidence: .medium,
                sampleSize: 30,
                triggerDaysCount: 15,
                nonTriggerDaysCount: 15
            )
        }

        let corrected = engine.applyBonferroniCorrection(triggers)

        // With 10 tests, α becomes 0.005
        // p=0.04 should no longer be significant
        let firstTrigger = corrected.first!
        XCTAssertFalse(firstTrigger.isSignificant)
    }
}
```

### 10.2 Integration Tests

```swift
// Example: EnsembleIntegrationTests.swift

@MainActor
final class EnsembleIntegrationTests: XCTestCase {

    func testEnsemble_progressiveActivation() async {
        let ensemble = TriggerEnsemble()

        // Day 7: Only statistical
        ensemble.updateActiveEngines(daysOfData: 7, neuralOptIn: false)
        XCTAssertEqual(ensemble.activeEngines, [.statistical])

        // Day 30: Statistical + k-NN
        ensemble.updateActiveEngines(daysOfData: 30, neuralOptIn: false)
        XCTAssertEqual(ensemble.activeEngines, [.statistical, .knn])

        // Day 90 without opt-in: Still no neural
        ensemble.updateActiveEngines(daysOfData: 90, neuralOptIn: false)
        XCTAssertEqual(ensemble.activeEngines, [.statistical, .knn])

        // Day 90 with opt-in: All three
        ensemble.updateActiveEngines(daysOfData: 90, neuralOptIn: true)
        XCTAssertEqual(ensemble.activeEngines, [.statistical, .knn, .neural])
    }

    func testEnsemble_weightsNormalizeToOne() async {
        let ensemble = TriggerEnsemble()
        ensemble.updateActiveEngines(daysOfData: 120, neuralOptIn: true)

        let features = TriggerFeatures.mock()
        let result = await ensemble.predict(features: features)

        let totalWeight = result.weights.values.reduce(0, +)
        XCTAssertEqual(totalWeight, 1.0, accuracy: 0.001)
    }
}
```

---

## Summary

This deep technical specification provides:

1. **Three-engine hybrid architecture** with clear responsibilities
2. **Progressive activation system** that adds capabilities as data grows
3. **Comprehensive data schema** for trigger logging
4. **150+ engineered features** including lags, rolling stats, and interactions
5. **Ensemble weighting system** that adapts to data availability
6. **Confidence quantification** at trigger and prediction levels
7. **Explainability framework** combining all engine outputs
8. **Complete file structure** and integration points
9. **Testing strategy** with example test cases

**Next Steps**:
1. Implement Core Data entities (TriggerLog, AnalysisCache)
2. Build StatisticalTriggerEngine with lagged analysis
3. Create trigger logging UI
4. Add k-NN engine at day 30 milestone
5. Add neural engine at day 90 milestone (opt-in)

---

**Document Version**: 2.0
**Last Updated**: December 7, 2025
**Author**: Claude Code
**Status**: Ready for Implementation
