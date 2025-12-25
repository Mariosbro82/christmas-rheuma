//
//  AIInsightsView.swift
//  InflamAI
//
//  Local AI predictions and insights
//  Uses MLPredictionService for hybrid predictions (Neural Engine + Statistical)
//
//  Phase 2: Updated to use unified MLPredictionService
//

import SwiftUI
import CoreData

struct AIInsightsView: View {
    // MARK: - Unified ML Prediction Service (Phase 2)
    @ObservedObject private var mlService = MLPredictionService.shared

    // MARK: - Individual Engines (for backward compatibility)
    @ObservedObject private var neuralEngine = UnifiedNeuralEngine.shared
    @StateObject private var predictor: FlarePredictor

    // MARK: - Trigger Analysis Service
    @ObservedObject private var triggerService = TriggerAnalysisService.shared

    @State private var isTraining = false
    @State private var isRefreshing = false
    @State private var trainingError: String?
    @State private var showTrainingInfo = false
    @State private var showNeuralEngineDetail = false

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _predictor = StateObject(wrappedValue: FlarePredictor(context: context))
    }

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from MoreView,
        // which is already wrapped in NavigationView in MainTabView.
        ScrollView {
            VStack(spacing: Spacing.xl) {
                // Compliance Disclaimer Banner
                betaDisclaimerBanner

                // Hybrid Prediction Card (NEW - Phase 2)
                hybridPredictionCard

                // Data Sources Comparison (NEW - Phase 2)
                dataSourcesComparisonCard

                // Neural Engine Card
                neuralEngineCard

                // Statistical Risk Card (renamed from riskOverviewCard)
                if predictor.isModelTrained {
                    statisticalRiskCard
                }

                // Train Model Section
                if !predictor.isModelTrained {
                    trainModelSection
                }

                // Contributing Factors (from hybrid sources)
                if let prediction = mlService.currentPrediction, !prediction.allFactors.isEmpty {
                    hybridFactorsSection(factors: prediction.allFactors)
                } else if predictor.isModelTrained && !predictor.contributingFactors.isEmpty {
                    contributingFactorsSection
                }

                // Recommendations
                if predictor.isModelTrained || neuralEngine.currentPrediction != nil {
                    recommendationsSection
                }

                // Trigger Analysis Section (NEW)
                triggerAnalysisCard

                // Model Info
                modelInfoSection
            }
            .padding()
        }
        .navigationTitle("Pattern Insights")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    Task {
                        isRefreshing = true
                        await mlService.refresh()
                        isRefreshing = false
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .rotationEffect(.degrees(isRefreshing ? 360 : 0))
                        .animation(isRefreshing ? .linear(duration: 1).repeatForever(autoreverses: false) : .default, value: isRefreshing)
                }
                .disabled(isRefreshing)
            }
        }
        .alert("Training Error", isPresented: .constant(trainingError != nil)) {
            Button("OK") {
                trainingError = nil
            }
        } message: {
            Text(trainingError ?? "")
        }
        .task {
            // Get initial hybrid prediction
            if mlService.currentPrediction == nil {
                _ = await mlService.getPrediction()
            }
        }
    }

    // MARK: - Hybrid Prediction Card (NEW - Phase 2)

    private var hybridPredictionCard: some View {
        VStack(spacing: Spacing.lg) {
            // Header with source badge
            HStack {
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    HStack(spacing: Spacing.sm) {
                        Text("Pattern Overview")
                            .font(.system(size: Typography.xl, weight: .bold))
                            .foregroundColor(Colors.Gray.g900)

                        // Source badge
                        HStack(spacing: Spacing.xxs) {
                            Image(systemName: mlService.primarySource.icon)
                                .font(.system(size: Typography.xxs))
                            Text(mlService.primarySource.rawValue)
                                .font(.system(size: Typography.xxs, weight: .medium))
                        }
                        .padding(.horizontal, Spacing.sm)
                        .padding(.vertical, Spacing.xxs)
                        .background(Colors.Accent.purple.opacity(0.2))
                        .foregroundColor(Colors.Accent.purple)
                        .cornerRadius(Radii.md)
                    }

                    Text(mlService.primarySource.description)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                }

                Spacer()
            }

            if let prediction = mlService.currentPrediction {
                // Main risk display
                HStack(alignment: .center, spacing: 20) {
                    // Risk circle
                    ZStack {
                        Circle()
                            .stroke(Colors.Gray.g500.opacity(0.2), lineWidth: 8)
                            .frame(width: 100, height: 100)

                        Circle()
                            .trim(from: 0, to: CGFloat(prediction.combinedRiskScore))
                            .stroke(
                                LinearGradient(
                                    colors: [riskGradientStart(prediction.riskLevel), riskGradientEnd(prediction.riskLevel)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                ),
                                style: StrokeStyle(lineWidth: 8, lineCap: .round)
                            )
                            .frame(width: 100, height: 100)
                            .rotationEffect(.degrees(-90))

                        VStack(spacing: 2) {
                            Text("\(Int(prediction.combinedRiskScore * 100))")
                                .font(.system(size: 32, weight: .bold))
                                .foregroundColor(colorForHybridRisk(prediction.riskLevel))

                            Text("%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text(prediction.riskLevel.rawValue.uppercased() + " PATTERN")
                            .font(.headline)
                            .foregroundColor(colorForHybridRisk(prediction.riskLevel))

                        Text(prediction.summary)
                            .font(.subheadline)
                            .foregroundColor(.secondary)

                        // Data quality indicator
                        HStack(spacing: 4) {
                            Image(systemName: "gauge.medium")
                                .font(.caption)
                            Text(prediction.combinedConfidence.rawValue)
                                .font(.caption)
                                .fontWeight(.medium)
                            Text("data quality")
                                .font(.caption)
                        }
                        .foregroundColor(.secondary)
                    }

                    Spacer()
                }

                // Data quality bar
                if let quality = prediction.dataQuality {
                    HStack(spacing: 8) {
                        Image(systemName: quality.isHealthKitConnected ? "heart.fill" : "heart.slash")
                            .foregroundColor(quality.isHealthKitConnected ? Colors.Semantic.error : Colors.Gray.g500)
                            .font(.caption)

                        Text("Data Quality: \(quality.qualityLevel)")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Spacer()

                        Text("\(Int(quality.overallScore * 100))%")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(quality.overallScore >= 0.6 ? Colors.Semantic.success : Colors.Semantic.warning)
                    }
                    .padding(.top, 8)
                }

            } else {
                // No prediction available - clean empty state matching weather design
                MLEmptyStateView(
                    icon: "chart.bar.doc.horizontal",
                    iconColor: Colors.Primary.p500,
                    title: "Pattern Analysis Unavailable",
                    message: mlService.getDataReadiness().nextMilestone,
                    tips: [
                        "Log symptoms daily",
                        "Complete daily check-ins",
                        "Connect HealthKit for biometrics"
                    ]
                )
            }
        }
        .padding(Spacing.md)
        .background(
            LinearGradient(
                gradient: Gradient(colors: [Color(.systemBackground), Color(.secondarySystemBackground)]),
                startPoint: .top,
                endPoint: .bottom
            )
        )
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.md)
    }

    // MARK: - Data Sources Comparison Card (NEW - Phase 2)

    private var dataSourcesComparisonCard: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            HStack {
                Image(systemName: "arrow.triangle.branch")
                    .foregroundColor(Colors.Accent.purple)
                Text("Prediction Sources")
                    .font(.system(size: Typography.sm, weight: .medium))
                    .foregroundColor(Colors.Gray.g900)
                Spacer()
            }

            if let prediction = mlService.currentPrediction {
                HStack(spacing: 12) {
                    // Neural Engine source
                    predictionSourceCard(
                        title: "Neural Engine",
                        icon: "brain.head.profile",
                        probability: prediction.neuralEnginePrediction?.probability,
                        isAvailable: prediction.neuralEnginePrediction != nil,
                        isPrimary: mlService.primarySource == .neuralEngine,
                        detail: prediction.neuralEnginePrediction?.isPersonalized == true ? "Personalized" : "Baseline"
                    )

                    // Statistical source
                    predictionSourceCard(
                        title: "Statistical",
                        icon: "chart.xyaxis.line",
                        probability: prediction.statisticalPrediction.map { Float($0.riskPercentage / 100) },
                        isAvailable: prediction.statisticalPrediction != nil,
                        isPrimary: mlService.primarySource == .statistical,
                        detail: prediction.statisticalPrediction?.weatherRisk != nil ? "Weather+Patterns" : "Patterns Only"
                    )
                }
            } else {
                HStack(spacing: 12) {
                    predictionSourceCard(
                        title: "Neural Engine",
                        icon: "brain.head.profile",
                        probability: nil,
                        isAvailable: neuralEngine.daysOfUserData >= 7,
                        isPrimary: false,
                        detail: neuralEngine.daysOfUserData >= 7 ? "Ready" : "Need \(7 - neuralEngine.daysOfUserData) days"
                    )

                    predictionSourceCard(
                        title: "Statistical",
                        icon: "chart.xyaxis.line",
                        probability: nil,
                        isAvailable: predictor.isModelTrained,
                        isPrimary: false,
                        detail: predictor.isModelTrained ? "Trained" : "Need 30 days"
                    )
                }
            }
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }

    private func predictionSourceCard(
        title: String,
        icon: String,
        probability: Float?,
        isAvailable: Bool,
        isPrimary: Bool,
        detail: String
    ) -> some View {
        VStack(spacing: Spacing.sm) {
            HStack(spacing: Spacing.xxs) {
                Image(systemName: icon)
                    .font(.system(size: Typography.xs))
                Text(title)
                    .font(.system(size: Typography.xs, weight: .medium))
            }
            .foregroundColor(isPrimary ? Colors.Accent.purple : (isAvailable ? Colors.Gray.g900 : Colors.Gray.g500))

            if let prob = probability {
                Text("\(Int(prob * 100))%")
                    .font(.system(size: Typography.lg, weight: .bold))
                    .foregroundColor(isPrimary ? Colors.Accent.purple : Colors.Gray.g900)
            } else {
                Text("--")
                    .font(.system(size: Typography.lg, weight: .bold))
                    .foregroundColor(Colors.Gray.g500)
            }

            Text(detail)
                .font(.system(size: Typography.xxs))
                .foregroundColor(Colors.Gray.g500)

            if isPrimary {
                Text("PRIMARY")
                    .font(.system(size: Typography.xxs, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, Spacing.xs)
                    .padding(.vertical, Spacing.xxs)
                    .background(Colors.Accent.purple)
                    .cornerRadius(Radii.xs)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(Spacing.md)
        .background(isPrimary ? Colors.Accent.purple.opacity(0.1) : Color(.systemBackground))
        .cornerRadius(Radii.lg)
        .overlay(
            RoundedRectangle(cornerRadius: Radii.lg)
                .stroke(isPrimary ? Colors.Accent.purple : Color.clear, lineWidth: 2)
        )
    }

    // MARK: - Hybrid Factors Section

    private func hybridFactorsSection(factors: [PredictionFactor]) -> some View {
        VStack(alignment: .leading, spacing: Spacing.lg) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(Colors.Semantic.warning)
                Text("Contributing Factors")
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)
                Spacer()
                Text("\(factors.count) factors")
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)
            }

            ForEach(factors.prefix(5)) { factor in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Circle()
                            .fill(colorForPredictionImpact(factor.impact))
                            .frame(width: 8, height: 8)

                        Text(factor.name)
                            .font(.subheadline)
                            .fontWeight(.semibold)

                        Spacer()

                        Text(factor.impact == .high ? "High" : factor.impact == .medium ? "Medium" : "Low")
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(colorForPredictionImpact(factor.impact).opacity(0.2))
                            .foregroundColor(colorForPredictionImpact(factor.impact))
                            .cornerRadius(8)
                    }

                    Text(factor.recommendation)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.md)
    }

    // MARK: - Statistical Risk Card (renamed)

    private var statisticalRiskCard: some View {
        VStack(spacing: Spacing.md) {
            HStack {
                Image(systemName: "chart.xyaxis.line")
                    .foregroundColor(Colors.Primary.p500)
                Text("Statistical Analysis")
                    .font(.system(size: Typography.sm, weight: .medium))
                    .foregroundColor(Colors.Gray.g900)
                Spacer()
                if mlService.primarySource == .statistical {
                    Text("PRIMARY")
                        .font(.system(size: Typography.xxs, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, Spacing.xs)
                        .padding(.vertical, Spacing.xxs)
                        .background(Colors.Primary.p500)
                        .cornerRadius(Radii.xs)
                }
            }

            HStack {
                Text("\(Int(predictor.riskPercentage))%")
                    .font(.system(size: Typography.xxl, weight: .bold))
                    .foregroundColor(colorForRisk(predictor.flareRiskLevel))

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(riskLevelText)
                        .font(.system(size: Typography.sm, weight: .medium))
                        .foregroundColor(Colors.Gray.g900)

                    if let days = predictor.daysUntilLikelyFlare {
                        Text("~\(days) days until likely flare")
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }

                Spacer()
            }
        }
        .padding(Spacing.md)
        .background(mlService.primarySource == .statistical ? Colors.Primary.p500.opacity(0.1) : Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }

    // MARK: - Helper Functions for Hybrid UI

    private func colorForHybridRisk(_ risk: HybridRiskLevel) -> Color {
        switch risk {
        case .unknown: return Colors.Gray.g500
        case .low: return Colors.Semantic.success
        case .moderate: return Color.yellow
        case .high: return Colors.Semantic.warning
        case .critical: return Colors.Semantic.error
        }
    }

    private func riskGradientStart(_ risk: HybridRiskLevel) -> Color {
        switch risk {
        case .unknown: return Colors.Gray.g500.opacity(0.5)
        case .low: return Colors.Semantic.success.opacity(0.7)
        case .moderate: return Color.yellow.opacity(0.7)
        case .high: return Colors.Semantic.warning.opacity(0.7)
        case .critical: return Colors.Semantic.error.opacity(0.7)
        }
    }

    private func riskGradientEnd(_ risk: HybridRiskLevel) -> Color {
        switch risk {
        case .unknown: return Colors.Gray.g500
        case .low: return Colors.Semantic.success
        case .moderate: return Colors.Semantic.warning
        case .high: return Colors.Semantic.error
        case .critical: return Colors.Semantic.error
        }
    }

    private func colorForPredictionImpact(_ impact: PredictionFactor.Impact) -> Color {
        switch impact {
        case .low: return Colors.Primary.p500
        case .medium: return Colors.Semantic.warning
        case .high: return Colors.Semantic.error
        }
    }

    // MARK: - Neural Engine Card (Primary Prediction Source)

    private var neuralEngineCard: some View {
        NavigationLink(destination: UnifiedNeuralEngineView()) {
            VStack(spacing: 16) {
                // Header with status
                HStack {
                    ZStack {
                        Circle()
                            .fill(neuralEngineStatusColor.opacity(0.2))
                            .frame(width: 60, height: 60)

                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 28))
                            .foregroundColor(neuralEngineStatusColor)
                            .symbolEffect(.pulse, isActive: neuralEngine.engineStatus == .learning)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Neural Engine")
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(.primary)

                            // Status badge
                            Text(neuralEngineStatusBadge)
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(neuralEngineStatusColor)
                                .cornerRadius(4)
                        }

                        Text(neuralEngine.engineStatus.displayMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }

                // Current prediction summary
                if let prediction = neuralEngine.currentPrediction {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 4) {
                                Text(prediction.willFlare ? "NOTABLE PATTERN" : "STABLE PATTERN")
                                    .font(.headline)
                                    .foregroundColor(prediction.willFlare ? Colors.Semantic.warning : Colors.Semantic.success)

                                // Removed percentage display per compliance requirements
                            }

                            Text(prediction.summary)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                        }

                        Spacer()

                        // Mini probability indicator
                        ZStack {
                            Circle()
                                .stroke(Colors.Gray.g500.opacity(0.2), lineWidth: 4)
                                .frame(width: 50, height: 50)

                            Circle()
                                .trim(from: 0, to: CGFloat(prediction.probability))
                                .stroke(
                                    prediction.willFlare ? Colors.Semantic.warning : Colors.Semantic.success,
                                    style: StrokeStyle(lineWidth: 4, lineCap: .round)
                                )
                                .frame(width: 50, height: 50)
                                .rotationEffect(.degrees(-90))

                            Text("\(Int(prediction.probability * 100))")
                                .font(.caption2)
                                .fontWeight(.bold)
                        }
                    }
                    .padding(.vertical, 8)
                } else {
                    // No prediction yet - clean progress state
                    if neuralEngine.daysOfUserData < 7 {
                        MLProgressStateView(
                            title: "Learning Your Patterns",
                            progress: Double(neuralEngine.daysOfUserData) / 7.0,
                            subtitle: "\(neuralEngine.daysOfUserData) of 7 days collected",
                            iconColor: Colors.Accent.purple
                        )
                    } else {
                        MLEmptyStateView(
                            icon: "waveform.path.ecg",
                            iconColor: Colors.Accent.purple,
                            title: "Ready to Analyze",
                            message: "Tap to view your data patterns"
                        )
                    }
                }

                // Personalization progress
                VStack(spacing: 8) {
                    HStack {
                        Text("Personalization: \(neuralEngine.personalizationPhase.rawValue)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(neuralEngine.learningProgress * 100))%")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(Colors.Accent.purple)
                    }

                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color(.systemGray5))

                            RoundedRectangle(cornerRadius: 4)
                                .fill(
                                    LinearGradient(
                                        colors: [Colors.Accent.purple, Colors.Primary.p500],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: geometry.size.width * CGFloat(neuralEngine.learningProgress))
                        }
                    }
                    .frame(height: 6)

                    HStack {
                        Label("\(neuralEngine.daysOfUserData) days", systemImage: "calendar")
                        Spacer()
                        if neuralEngine.isPersonalized {
                            Label("Personalized", systemImage: "checkmark.seal.fill")
                                .foregroundColor(Colors.Semantic.success)
                        } else {
                            Label("v\(neuralEngine.modelVersion)", systemImage: "tag")
                        }
                    }
                    .font(.caption2)
                    .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(
                LinearGradient(
                    gradient: Gradient(colors: [Colors.Accent.purple.opacity(0.15), Colors.Primary.p500.opacity(0.1)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .cornerRadius(16)
            .shadow(color: Colors.Accent.purple.opacity(0.2), radius: 10)
        }
        .buttonStyle(PlainButtonStyle())
    }

    // MARK: - Neural Engine Helpers

    private var neuralEngineStatusColor: Color {
        switch neuralEngine.engineStatus {
        case .ready: return Colors.Semantic.success
        case .initializing: return Colors.Semantic.warning
        case .learning: return Colors.Primary.p500
        case .error: return Colors.Semantic.error
        }
    }

    private var neuralEngineStatusBadge: String {
        if neuralEngine.isPersonalized {
            return "PERSONALIZED"
        } else {
            switch neuralEngine.engineStatus {
            case .ready: return "READY"
            case .initializing: return "STARTING"
            case .learning: return "LEARNING"
            case .error: return "ERROR"
            }
        }
    }

    // MARK: - Trigger Analysis Card (NEW)

    private var triggerAnalysisCard: some View {
        NavigationLink(destination: TriggerInsightsView()) {
            VStack(spacing: 16) {
                // Header with phase indicator
                HStack {
                    ZStack {
                        Circle()
                            .fill(triggerPhaseColor.opacity(0.2))
                            .frame(width: 60, height: 60)

                        Image(systemName: "waveform.path.ecg.rectangle")
                            .font(.system(size: 28))
                            .foregroundColor(triggerPhaseColor)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Trigger Analysis")
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(.primary)

                            // Phase badge
                            Text(triggerService.currentPhase.displayName)
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(triggerPhaseColor)
                                .cornerRadius(4)
                        }

                        Text("\(triggerService.daysOfData) days of data analyzed")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }

                // Top triggers summary
                if !triggerService.topTriggers.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Frequently Observed Correlations")
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(.secondary)

                        HStack(spacing: 8) {
                            ForEach(triggerService.topTriggers.prefix(3)) { trigger in
                                HStack(spacing: 4) {
                                    Image(systemName: trigger.icon)
                                        .font(.caption2)
                                    // CRIT-003 FIX: Apply displayName to convert snake_case to Title Case
                                    Text(trigger.triggerName.displayName)
                                        .font(.caption)
                                        .lineLimit(1)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(trigger.triggerCategory.color.opacity(0.2))
                                .foregroundColor(trigger.triggerCategory.color)
                                .cornerRadius(8)
                            }
                        }
                    }
                } else {
                    // No triggers yet - clean progress/empty state
                    if triggerService.daysOfData < 7 {
                        MLProgressStateView(
                            title: "Analyzing Triggers",
                            progress: Double(triggerService.daysOfData) / 7.0,
                            subtitle: "\(triggerService.daysOfData) of 7 days analyzed",
                            iconColor: Colors.Accent.teal
                        )
                    } else {
                        MLEmptyStateView(
                            icon: "chart.bar.doc.horizontal",
                            iconColor: Colors.Accent.teal,
                            title: "Analysis Ready",
                            message: "Tap to view your pattern analysis"
                        )
                    }
                }

                // Active engines indicator
                VStack(spacing: 8) {
                    HStack {
                        Text("Active Engines")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                    }

                    HStack(spacing: 8) {
                        ForEach(triggerService.currentPhase.activeEngines, id: \.self) { engine in
                            HStack(spacing: 4) {
                                Image(systemName: engine.icon)
                                    .font(.caption2)
                                Text(engine.displayName)
                                    .font(.caption2)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Colors.Primary.p500.opacity(0.1))
                            .foregroundColor(Colors.Primary.p500)
                            .cornerRadius(8)
                        }
                        Spacer()
                    }
                }
            }
            .padding()
            .background(
                LinearGradient(
                    gradient: Gradient(colors: [Colors.Accent.teal.opacity(0.15), Colors.Primary.p500.opacity(0.1)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .cornerRadius(16)
            .shadow(color: Colors.Accent.teal.opacity(0.2), radius: 10)
        }
        .buttonStyle(PlainButtonStyle())
    }

    // MARK: - Trigger Analysis Helpers

    private var triggerPhaseColor: Color {
        switch triggerService.currentPhase {
        case .statistical: return Colors.Primary.p500
        case .knn: return Colors.Accent.purple
        case .neural: return Colors.Semantic.warning
        }
    }

    // MARK: - Risk Overview

    private var riskOverviewCard: some View {
        VStack(spacing: 16) {
            // Pattern Level Badge
            HStack {
                Text(predictor.flareRiskLevel.emoji)
                    .font(.system(size: 60))

                VStack(alignment: .leading, spacing: 4) {
                    Text("Pattern Status")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text(predictor.flareRiskLevel == .unknown ? "Not Available" : riskLevelText)
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(colorForRisk(predictor.flareRiskLevel))

                    Text("Based on your logged data")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Spacer()
            }

            // Progress Bar
            if predictor.isModelTrained {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Rectangle()
                            .fill(Colors.Gray.g500.opacity(0.2))
                            .frame(height: 8)
                            .cornerRadius(4)

                        Rectangle()
                            .fill(colorForRisk(predictor.flareRiskLevel))
                            .frame(width: geometry.size.width * (predictor.riskPercentage / 100), height: 8)
                            .cornerRadius(4)
                    }
                }
                .frame(height: 8)
            }

            // Pattern info - removed time estimates per compliance
            HStack {
                Image(systemName: "info.circle")
                    .foregroundColor(Colors.Primary.p500)
                Text("Patterns based on your logged data. Discuss with your doctor.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
            }
            .padding(.top, 4)

            // Last Updated
            if let lastPrediction = predictor.lastPrediction {
                HStack {
                    Image(systemName: "clock")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("Updated \(lastPrediction, style: .relative) ago")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 10)
    }

    // MARK: - Training Section (Redesigned to match weather design language)

    private var trainModelSection: some View {
        VStack(spacing: Spacing.lg) {
            // Header with icon in soft circle
            MLInsightCardHeader(
                icon: "brain.head.profile",
                iconColor: Colors.Primary.p500,
                title: "Analyze Your Patterns",
                subtitle: "Build your personal insights"
            )

            // Progress indicator
            MLProgressStateView(
                title: "Collecting Data",
                progress: min(Double(UserDefaults.standard.integer(forKey: "trainingDataPointCount")) / 30.0, 1.0),
                subtitle: "30 days of data needed for analysis",
                iconColor: Colors.Primary.p500
            )

            // Action button
            Button {
                Task {
                    await trainModel()
                }
            } label: {
                HStack(spacing: Spacing.sm) {
                    if isTraining {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Image(systemName: "chart.bar.doc.horizontal")
                            .font(.system(size: 14, weight: .semibold))
                        Text("Analyze Patterns")
                            .font(.system(size: Typography.sm, weight: .semibold))
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, Spacing.md)
                .background(Colors.Primary.p500)
                .foregroundColor(.white)
                .cornerRadius(Radii.lg)
            }
            .disabled(isTraining)

            // Info link
            Button {
                showTrainingInfo = true
            } label: {
                HStack(spacing: Spacing.xs) {
                    Image(systemName: "info.circle")
                    Text("How it works")
                }
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Primary.p500)
            }
        }
        .padding(Spacing.lg)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.md)
        .sheet(isPresented: $showTrainingInfo) {
            trainingInfoSheet
        }
    }

    // MARK: - Contributing Factors

    private var contributingFactorsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(Colors.Semantic.warning)
                Text("Contributing Factors")
                    .font(.headline)
                Spacer()
            }

            ForEach(predictor.contributingFactors) { factor in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Circle()
                            .fill(colorForImpact(factor.impact))
                            .frame(width: 8, height: 8)

                        Text(factor.name)
                            .font(.subheadline)
                            .fontWeight(.semibold)

                        Spacer()

                        Text(impactText(factor.impact))
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(colorForImpact(factor.impact).opacity(0.2))
                            .foregroundColor(colorForImpact(factor.impact))
                            .cornerRadius(8)
                    }

                    Text(factor.recommendation)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 10)
    }

    // MARK: - Recommendations

    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(Color.yellow)
                Text("Personalized Suggestions")
                    .font(.headline)
                Spacer()
            }

            VStack(alignment: .leading, spacing: 8) {
                recommendationItem(
                    icon: "checkmark.circle.fill",
                    text: "Continue daily symptom logging for better predictions",
                    color: Colors.Semantic.success
                )

                if predictor.flareRiskLevel == .high || predictor.flareRiskLevel == .critical {
                    recommendationItem(
                        icon: "cross.case.fill",
                        text: "Consider contacting your rheumatologist",
                        color: Colors.Semantic.error
                    )

                    recommendationItem(
                        icon: "bed.double.fill",
                        text: "Prioritize rest and reduce physical stress",
                        color: Colors.Primary.p500
                    )
                }

                if predictor.riskPercentage > 30 {
                    recommendationItem(
                        icon: "pills.fill",
                        text: "Ensure medication adherence",
                        color: Colors.Accent.purple
                    )
                }

                recommendationItem(
                    icon: "moon.fill",
                    text: "Maintain good sleep hygiene",
                    color: Color.indigo
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 10)
    }

    private func recommendationItem(icon: String, text: String, color: Color) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 24)

            Text(text)
                .font(.subheadline)
                .foregroundColor(.primary)

            Spacer()
        }
    }

    // MARK: - Compliance Banner

    private var betaDisclaimerBanner: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(Colors.Semantic.warning)
                .font(.subheadline)

            VStack(alignment: .leading, spacing: 2) {
                Text("Beta Feature")
                    .font(.system(size: Typography.xs, weight: .semibold))
                    .foregroundColor(Colors.Gray.g700)

                Text("Pattern insights are based on your logged data, not clinical analysis. Discuss any findings with your healthcare provider.")
                    .font(.system(size: Typography.xxs))
                    .foregroundColor(Colors.Gray.g500)
                    .lineLimit(2)
            }

            Spacer()
        }
        .padding(Spacing.md)
        .background(Colors.Semantic.warning.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: Radii.md))
    }

    // MARK: - Model Info

    private var modelInfoSection: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "info.circle.fill")
                    .foregroundColor(Colors.Primary.p500)
                Text("About Pattern Analysis")
                    .font(.headline)
                Spacer()
            }

            VStack(alignment: .leading, spacing: 8) {
                infoRow(label: "Status", value: predictor.isModelTrained ? "Trained ✓" : "Not Trained")
                infoRow(label: "Privacy", value: "100% On-Device")
                infoRow(label: "Method", value: "Statistical Pattern Analysis")

                if let lastTraining = UserDefaults.standard.object(forKey: "lastModelTrainingDate") as? Date {
                    infoRow(label: "Last Training", value: lastTraining.formatted(date: .abbreviated, time: .omitted))
                }

                if let dataPoints = UserDefaults.standard.object(forKey: "trainingDataPointCount") as? Int {
                    infoRow(label: "Training Data", value: "\(dataPoints) days")
                }
            }

            VStack(spacing: 8) {
                Text("All predictions run locally on your device. No data is sent to servers.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)

                Text("⚠️ Not Medical Advice")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(Colors.Semantic.warning)

                Text("These insights are based on statistical patterns in your data. Always consult your rheumatologist for medical decisions.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.top, 4)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 10)
    }

    private func infoRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
    }

    // MARK: - Training Info Sheet

    private var trainingInfoSheet: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("How Pattern Analysis Works")
                        .font(.title2)
                        .fontWeight(.bold)

                    VStack(alignment: .leading, spacing: 12) {
                        Text("1. Data Collection")
                            .font(.headline)
                        Text("The system analyzes your symptom logs, including pain levels, BASDAI scores, weather data, sleep quality, and medication adherence.")
                            .foregroundColor(.secondary)

                        Text("2. Pattern Analysis")
                            .font(.headline)
                        Text("Using statistical analysis, the system identifies patterns that preceded your past flares by comparing averages and trends.")
                            .foregroundColor(.secondary)

                        Text("3. Risk Calculation")
                            .font(.headline)
                        Text("Based on current symptoms and environmental factors, it estimates flare probability within the next 7 days using pattern matching.")
                            .foregroundColor(.secondary)

                        Text("4. Privacy")
                            .font(.headline)
                        Text("Everything runs on your device. No data ever leaves your iPhone. Your patterns stay private.")
                            .foregroundColor(.secondary)

                        Text("5. Limitations")
                            .font(.headline)
                        Text("This is not a medical diagnostic tool. Predictions are based solely on your historical data patterns and should not replace professional medical advice.")
                            .foregroundColor(Colors.Semantic.error)
                            .fontWeight(.semibold)
                    }

                    Divider()

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Requirements")
                            .font(.headline)
                        Text("• Minimum 30 days of symptom data")
                        Text("• Regular daily logging")
                        Text("• At least one recorded flare event")
                    }
                    .font(.subheadline)
                }
                .padding()
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        showTrainingInfo = false
                    }
                }
            }
        }
    }

    // MARK: - Actions

    private func trainModel() async {
        isTraining = true
        trainingError = nil

        do {
            try await predictor.trainModel()
        } catch {
            trainingError = error.localizedDescription
        }

        isTraining = false
    }

    private func retrainModel() async {
        await trainModel()
    }

    // MARK: - Helpers

    private var riskLevelText: String {
        switch predictor.flareRiskLevel {
        case .unknown: return "Insufficient Data"
        case .low: return "Stable Patterns"
        case .moderate: return "Some Changes"
        case .high: return "Notable Patterns"
        case .critical: return "Discuss With Doctor"
        }
    }

    private func colorForRisk(_ risk: FlarePredictorRiskLevel) -> Color {
        switch risk {
        case .unknown: return Colors.Gray.g500
        case .low: return Colors.Semantic.success
        case .moderate: return Color.yellow
        case .high: return Colors.Semantic.warning
        case .critical: return Colors.Semantic.error
        }
    }

    private func colorForImpact(_ impact: FlarePredictorFactor.Impact) -> Color {
        switch impact {
        case .low: return Colors.Primary.p500
        case .medium: return Colors.Semantic.warning
        case .high: return Colors.Semantic.error
        }
    }

    private func impactText(_ impact: FlarePredictorFactor.Impact) -> String {
        switch impact {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        }
    }
}

// MARK: - Preview

struct AIInsightsView_Previews: PreviewProvider {
    static var previews: some View {
        AIInsightsView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}

// MARK: - ML Insight Components (Matches Weather Design Language)

/// Clean empty state for when ML data isn't available
private struct MLEmptyStateView: View {
    let icon: String
    let iconColor: Color
    let title: String
    let message: String
    var tips: [String]? = nil

    var body: some View {
        VStack(spacing: Spacing.lg) {
            // Icon in soft circle (matches weather error state)
            ZStack {
                Circle()
                    .fill(iconColor.opacity(0.15))
                    .frame(width: 72, height: 72)

                Image(systemName: icon)
                    .font(.system(size: 32))
                    .foregroundColor(iconColor)
            }

            // Title and message
            VStack(spacing: Spacing.xs) {
                Text(title)
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                Text(message)
                    .font(.system(size: Typography.sm))
                    .foregroundColor(Colors.Gray.g500)
                    .multilineTextAlignment(.center)
            }

            // Optional tips
            if let tips = tips, !tips.isEmpty {
                VStack(alignment: .leading, spacing: Spacing.xs) {
                    ForEach(tips, id: \.self) { tip in
                        HStack(spacing: Spacing.sm) {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 12))
                                .foregroundColor(Colors.Primary.p500)

                            Text(tip)
                                .font(.system(size: Typography.xs))
                                .foregroundColor(Colors.Gray.g600)
                        }
                    }
                }
                .padding(.top, Spacing.sm)
            }
        }
        .padding(Spacing.xl)
        .frame(maxWidth: .infinity)
    }
}

/// Clean loading/progress state matching weather design
private struct MLProgressStateView: View {
    let title: String
    let progress: Double
    let subtitle: String
    let iconColor: Color

    var body: some View {
        VStack(spacing: Spacing.lg) {
            // Icon in soft circle
            ZStack {
                Circle()
                    .fill(iconColor.opacity(0.15))
                    .frame(width: 72, height: 72)

                Image(systemName: "brain.head.profile")
                    .font(.system(size: 32))
                    .foregroundColor(iconColor)
                    .symbolEffect(.pulse)
            }

            // Title
            Text(title)
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            // Progress bar
            VStack(spacing: Spacing.xs) {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: Radii.xs)
                            .fill(Colors.Gray.g200)

                        RoundedRectangle(cornerRadius: Radii.xs)
                            .fill(
                                LinearGradient(
                                    colors: [iconColor, iconColor.opacity(0.7)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: geometry.size.width * progress)
                    }
                }
                .frame(height: 8)

                HStack {
                    Text(subtitle)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)

                    Spacer()

                    Text("\(Int(progress * 100))%")
                        .font(.system(size: Typography.xs, weight: .semibold))
                        .foregroundColor(iconColor)
                }
            }
        }
        .padding(Spacing.xl)
        .frame(maxWidth: .infinity)
    }
}

/// Consistent header for ML insight cards
private struct MLInsightCardHeader: View {
    let icon: String
    let iconColor: Color
    let title: String
    var subtitle: String? = nil

    var body: some View {
        HStack(spacing: Spacing.md) {
            // Icon in soft circle
            ZStack {
                Circle()
                    .fill(iconColor.opacity(0.15))
                    .frame(width: 48, height: 48)

                Image(systemName: icon)
                    .font(.system(size: 22))
                    .foregroundColor(iconColor)
            }

            // Title and subtitle
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(title)
                    .font(.system(size: Typography.md, weight: .bold))
                    .foregroundColor(Colors.Gray.g900)

                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                }
            }

            Spacer()
        }
    }
}
