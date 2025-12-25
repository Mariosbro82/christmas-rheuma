//
//  UnifiedNeuralEngineView.swift
//  InflamAI
//
//  Single unified view for all Neural Engine functionality
//  Self-learning ML predictions with full personalization status
//

import SwiftUI
import Charts

/// The main Neural Engine view - use this instead of fragmented views
/// All flare prediction UI should go through this view
struct UnifiedNeuralEngineView: View {
    @ObservedObject private var engine = UnifiedNeuralEngine.shared
    @State private var isRefreshing = false
    @State private var showPersonalizationSheet = false
    @State private var showFactorsDetail = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Engine Status Header
                engineStatusHeader

                // Main Prediction Card
                if let prediction = engine.currentPrediction {
                    predictionCard(prediction)
                } else {
                    noPredictionCard
                }

                // Personalization Progress
                personalizationCard

                // Contributing Factors
                if !engine.topFactors.isEmpty {
                    factorsCard
                }

                // Data Collection Status
                dataStatusCard

                // Data Quality Indicators (NEW - Phase 1)
                dataQualityCard

                // High Confidence Mode Explanation
                highConfidenceModeCard

                // Medical Disclaimer
                disclaimerCard

                Spacer(minLength: 40)
            }
            .padding()
        }
        .navigationTitle("Neural Engine")
        .navigationBarTitleDisplayMode(.large)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                refreshButton
            }
        }
        .task {
            if engine.currentPrediction == nil {
                await engine.refresh()
            }
        }
        .sheet(isPresented: $showPersonalizationSheet) {
            PersonalizationDetailSheet(engine: engine)
        }
    }

    // MARK: - Engine Status Header

    private var engineStatusHeader: some View {
        HStack(spacing: 12) {
            // Status indicator
            Circle()
                .fill(statusColor)
                .frame(width: 12, height: 12)
                .overlay(
                    Circle()
                        .stroke(statusColor.opacity(0.3), lineWidth: 4)
                )

            VStack(alignment: .leading, spacing: 2) {
                Text(engine.engineStatus.displayMessage)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                if engine.isPersonalized {
                    Text("Model v\(engine.modelVersion) - Personalized")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Text("Baseline model - Not yet personalized")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            // Brain icon with status
            if #available(iOS 17.0, *) {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(statusColor)
                    .symbolEffect(.pulse, isActive: engine.engineStatus == .learning)
            } else {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(statusColor)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var statusColor: Color {
        switch engine.engineStatus {
        case .ready: return .green
        case .initializing: return .orange
        case .learning: return .blue
        case .error: return .red
        }
    }

    // MARK: - Prediction Card

    private func predictionCard(_ prediction: FlareRiskPrediction) -> some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                Text("3-7 Day Outlook")
                    .font(.headline)

                Spacer()

                // Conservative mode indicator
                Text("High Confidence Mode")
                    .font(.caption2)
                    .foregroundColor(.blue)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(4)

                Text(prediction.timestamp, style: .relative)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Main result - User friendly, not scary
            HStack(alignment: .center, spacing: 16) {
                // Risk icon
                ZStack {
                    Circle()
                        .fill(riskColor(prediction.riskLevel).opacity(0.2))
                        .frame(width: 70, height: 70)

                    Image(systemName: prediction.willFlare ? "exclamationmark.triangle.fill" : "checkmark.shield.fill")
                        .font(.system(size: 30))
                        .foregroundColor(riskColor(prediction.riskLevel))
                }

                VStack(alignment: .leading, spacing: 4) {
                    // Friendly status message instead of scary percentage
                    if prediction.willFlare {
                        Text("HEADS UP")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.orange)
                        Text("Changes observed in your data")
                            .font(.callout)
                            .foregroundColor(.secondary)
                    } else {
                        Text("STABLE")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.green)
                        Text("No notable changes in your data")
                            .font(.callout)
                            .foregroundColor(.secondary)
                    }

                    // Pattern level badge
                    HStack(spacing: 4) {
                        Circle()
                            .fill(riskColor(prediction.riskLevel))
                            .frame(width: 8, height: 8)
                        Text(prediction.riskLevel.rawValue + " Pattern")
                            .font(.caption)
                    }
                    .foregroundColor(riskColor(prediction.riskLevel))
                }

                Spacer()
            }

            // Confidence indicator (simpler than probability gauge)
            HStack {
                Text("Confidence:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(prediction.confidence.rawValue)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(Color(prediction.confidence.color))
                Spacer()
            }

            Divider()

            // Recommended action
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(.yellow)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Recommended Action")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text(prediction.recommendedAction.rawValue)
                        .font(.callout)
                        .fontWeight(.medium)
                }

                Spacer()
            }

            // Personalization badge
            if prediction.isPersonalized {
                HStack(spacing: 6) {
                    Image(systemName: "person.crop.circle.badge.checkmark")
                        .foregroundColor(.green)
                    Text("Personalized to your patterns (v\(prediction.modelVersion))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 4)
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundColor(.orange)
                    Text("Using baseline model - Keep logging for personalization")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
                .padding(.top, 4)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.1), radius: 8, x: 0, y: 2)
    }

    private func riskColor(_ level: RiskLevel) -> Color {
        switch level {
        case .low: return .green
        case .moderate: return .yellow
        case .high: return .orange
        case .critical: return .red
        }
    }

    // MARK: - No Prediction Card

    private var noPredictionCard: some View {
        VStack(spacing: 16) {
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            Text("No Predictions Yet")
                .font(.headline)

            if engine.daysOfUserData < 7 {
                Text("Need at least 7 days of symptom data to generate predictions. Keep logging!")
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            } else {
                Text("Tap refresh to generate your first AI-powered prediction")
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(32)
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    // MARK: - Personalization Card

    private var personalizationCard: some View {
        Button(action: { showPersonalizationSheet = true }) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "brain")
                        .font(.title2)
                        .foregroundColor(.purple)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Self-Learning Status")
                            .font(.headline)
                            .foregroundColor(.primary)

                        Text(engine.personalizationPhase.description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                // Progress bar
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(.systemGray5))

                        RoundedRectangle(cornerRadius: 4)
                            .fill(
                                LinearGradient(
                                    colors: [.purple, .blue],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: geometry.size.width * CGFloat(engine.learningProgress))
                    }
                }
                .frame(height: 8)

                // Stats row
                HStack {
                    Label("\(engine.daysOfUserData) days", systemImage: "calendar")
                    Spacer()
                    Label("\(Int(engine.learningProgress * 100))%", systemImage: "chart.line.uptrend.xyaxis")
                    Spacer()
                    Label("v\(engine.modelVersion)", systemImage: "tag")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
            .padding()
            .background(Color.purple.opacity(0.1))
            .cornerRadius(12)
        }
        .buttonStyle(PlainButtonStyle())
    }

    // MARK: - Contributing Factors Card

    private var factorsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Contributing Factors")
                    .font(.headline)

                Spacer()

                Button("See All") {
                    showFactorsDetail = true
                }
                .font(.caption)
            }

            ForEach(engine.topFactors.prefix(3)) { factor in
                FactorRowView(factor: factor)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
        .sheet(isPresented: $showFactorsDetail) {
            FactorsDetailSheet(factors: engine.topFactors)
        }
    }

    // MARK: - Data Status Card

    private var dataStatusCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "chart.bar.doc.horizontal")
                    .foregroundColor(.blue)

                Text("Data Collection")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()

                Text("\(engine.daysOfUserData) days")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if engine.daysOfUserData < 37 {
                let remaining = 37 - engine.daysOfUserData
                Text("Need \(remaining) more days of logging to enable on-device personalization")
                    .font(.caption)
                    .foregroundColor(.secondary)

                // Progress indicator
                ProgressView(value: Float(engine.daysOfUserData), total: 37)
                    .tint(.blue)
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Ready for personalization!")
                        .font(.caption)
                        .foregroundColor(.green)
                }
            }
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }

    // MARK: - Data Quality Card (NEW - Phase 1)

    private var dataQualityCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "waveform.path.ecg.rectangle")
                    .foregroundColor(.cyan)

                Text("Data Quality")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()

                // Data quality score badge
                if let metrics = engine.featureExtractor.lastExtractionMetrics {
                    let score = Int(metrics.dataQualityScore * 100)
                    Text("\(score)%")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(score >= 70 ? .green : score >= 40 ? .orange : .red)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(score >= 70 ? Color.green.opacity(0.2) :
                                      score >= 40 ? Color.orange.opacity(0.2) : Color.red.opacity(0.2))
                        )
                }
            }

            // Feature category breakdown
            if let metrics = engine.featureExtractor.lastExtractionMetrics {
                VStack(spacing: 8) {
                    DataSourceRow(
                        name: "HealthKit",
                        available: metrics.healthKitFeatures,
                        total: metrics.healthKitExpected,
                        icon: "heart.fill",
                        color: .red
                    )

                    DataSourceRow(
                        name: "Core Data",
                        available: metrics.coreDataFeatures,
                        total: 61,  // Demographics + Clinical + Pain + Mental + Adherence + Universal
                        icon: "cylinder.split.1x2.fill",
                        color: .blue
                    )

                    DataSourceRow(
                        name: "Weather",
                        available: metrics.weatherFeatures,
                        total: metrics.weatherExpected,
                        icon: "cloud.fill",
                        color: .cyan
                    )
                }

                // Missing features warning
                if !metrics.missingFeatureNames.isEmpty && metrics.missingFeatureNames.count <= 5 {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Missing data:")
                            .font(.caption2)
                            .foregroundColor(.secondary)

                        Text(metrics.missingFeatureNames.joined(separator: ", "))
                            .font(.caption2)
                            .foregroundColor(.orange)
                    }
                    .padding(.top, 4)
                }
            } else {
                Text("Run a prediction to see data quality metrics")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.cyan.opacity(0.1))
        .cornerRadius(12)
    }

    // MARK: - High Confidence Mode Explanation

    private var highConfidenceModeCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "shield.checkered")
                    .foregroundColor(.blue)

                Text("High Confidence Mode")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.caption)
                    Text("Alerts only trigger when confidence is high (~65% accurate)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "bell.badge")
                        .foregroundColor(.orange)
                        .font(.caption)
                    Text("Fewer false alarms means alerts are more meaningful")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "info.circle")
                        .foregroundColor(.blue)
                        .font(.caption)
                    Text("May miss some flares - stay aware of your symptoms")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }

    // MARK: - Disclaimer Card

    private var disclaimerCard: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)

            VStack(alignment: .leading, spacing: 4) {
                Text("Research Feature")
                    .font(.caption)
                    .fontWeight(.semibold)

                Text("This is statistical pattern analysis, not medical advice. Always consult your rheumatologist for medical decisions.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
    }

    // MARK: - Refresh Button

    private var refreshButton: some View {
        Button {
            Task {
                isRefreshing = true
                await engine.refresh()
                isRefreshing = false
            }
        } label: {
            Image(systemName: "arrow.clockwise")
                .rotationEffect(.degrees(isRefreshing ? 360 : 0))
                .animation(
                    isRefreshing ? .linear(duration: 1).repeatForever(autoreverses: false) : .default,
                    value: isRefreshing
                )
        }
        .disabled(isRefreshing)
    }
}

// MARK: - Supporting Views

struct ProbabilityGaugeView: View {
    let probability: Float

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color(.systemGray5))

                // Gradient fill
                RoundedRectangle(cornerRadius: 6)
                    .fill(
                        LinearGradient(
                            colors: [.green, .yellow, .orange, .red],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: geometry.size.width * CGFloat(probability))

                // Threshold marker at 50%
                Rectangle()
                    .fill(Color.primary.opacity(0.3))
                    .frame(width: 2)
                    .offset(x: geometry.size.width * 0.5)

                // Current value indicator
                Circle()
                    .fill(Color.white)
                    .frame(width: 16, height: 16)
                    .overlay(
                        Circle()
                            .stroke(Color.primary.opacity(0.5), lineWidth: 2)
                    )
                    .offset(x: geometry.size.width * CGFloat(probability) - 8)
            }
        }
    }
}

struct FactorRowView: View {
    let factor: ContributingFactor

    var body: some View {
        HStack(spacing: 12) {
            // Impact indicator
            Circle()
                .fill(impactColor)
                .frame(width: 8, height: 8)

            // Factor name
            Text(factor.name)
                .font(.subheadline)

            Spacer()

            // Trend indicator
            Image(systemName: trendIcon)
                .font(.caption)
                .foregroundColor(trendColor)

            // Impact level
            Text(impactText)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }

    private var impactColor: Color {
        switch factor.impact {
        case .low: return .green
        case .medium: return .orange
        case .high: return .red
        }
    }

    private var impactText: String {
        switch factor.impact {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        }
    }

    private var trendIcon: String {
        switch factor.trend {
        case .increasing: return "arrow.up.right"
        case .stable: return "arrow.right"
        case .decreasing: return "arrow.down.right"
        }
    }

    private var trendColor: Color {
        switch factor.trend {
        case .increasing: return .red
        case .stable: return .gray
        case .decreasing: return .green
        }
    }
}

// MARK: - Sheets

struct PersonalizationDetailSheet: View {
    @ObservedObject var engine: UnifiedNeuralEngine
    @Environment(\.dismiss) var dismiss
    @State private var isPersonalizing = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Phase visualization
                    phaseVisualization

                    // Stats grid
                    statsGrid

                    // Personalize button
                    if engine.daysOfUserData >= 37 && !engine.isPersonalized {
                        personalizeButton
                    }

                    // Learning explanation
                    learningExplanation
                }
                .padding()
            }
            .navigationTitle("Self-Learning")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var phaseVisualization: some View {
        VStack(spacing: 16) {
            Text("Learning Phase")
                .font(.headline)

            // Phase circles
            HStack(spacing: 4) {
                ForEach(PersonalizationPhase.allCases, id: \.self) { phase in
                    VStack(spacing: 4) {
                        Circle()
                            .fill(phase == engine.personalizationPhase ? Color.purple : Color(.systemGray4))
                            .frame(width: 12, height: 12)

                        Text(phase.rawValue)
                            .font(.caption2)
                            .foregroundColor(phase == engine.personalizationPhase ? .purple : .secondary)
                    }

                    if phase != .expert {
                        Rectangle()
                            .fill(phase.progressPercentage <= engine.learningProgress ? Color.purple : Color(.systemGray4))
                            .frame(height: 2)
                    }
                }
            }
            .padding(.horizontal)

            Text(engine.personalizationPhase.description)
                .font(.callout)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    private var statsGrid: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
            NeuralEngineStatBox(title: "Days Logged", value: "\(engine.daysOfUserData)", icon: "calendar")
            NeuralEngineStatBox(title: "Model Version", value: "v\(engine.modelVersion)", icon: "tag")
            NeuralEngineStatBox(title: "Progress", value: "\(Int(engine.learningProgress * 100))%", icon: "chart.line.uptrend.xyaxis")
            NeuralEngineStatBox(title: "Status", value: engine.isPersonalized ? "Personal" : "Baseline", icon: "brain")
        }
    }

    private var personalizeButton: some View {
        Button {
            Task {
                isPersonalizing = true
                try? await engine.triggerPersonalization()
                isPersonalizing = false
            }
        } label: {
            HStack {
                if isPersonalizing {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "brain.head.profile")
                }
                Text(isPersonalizing ? "Personalizing..." : "Start Personalization")
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.purple)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(isPersonalizing)
    }

    private var learningExplanation: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("How Self-Learning Works")
                .font(.headline)

            VStack(alignment: .leading, spacing: 8) {
                ExplanationRow(
                    icon: "1.circle.fill",
                    title: "Data Collection",
                    description: "Log symptoms daily to build your personal dataset"
                )
                ExplanationRow(
                    icon: "2.circle.fill",
                    title: "Pattern Recognition",
                    description: "Neural Engine identifies YOUR unique flare triggers"
                )
                ExplanationRow(
                    icon: "3.circle.fill",
                    title: "On-Device Training",
                    description: "Model updates locally - your data never leaves your phone"
                )
                ExplanationRow(
                    icon: "4.circle.fill",
                    title: "Personalized Predictions",
                    description: "Get forecasts tailored to your condition"
                )
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }
}

struct NeuralEngineStatBox: View {
    let title: String
    let value: String
    let icon: String

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.purple)

            Text(value)
                .font(.title3)
                .fontWeight(.bold)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }
}

struct ExplanationRow: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.purple)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}

/// Data source row for feature quality display
struct DataSourceRow: View {
    let name: String
    let available: Int
    let total: Int
    let icon: String
    let color: Color

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundColor(color)
                .frame(width: 16)

            Text(name)
                .font(.caption)
                .frame(width: 70, alignment: .leading)

            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color(.systemGray5))
                        .frame(height: 6)

                    RoundedRectangle(cornerRadius: 2)
                        .fill(progressColor)
                        .frame(width: geometry.size.width * progress, height: 6)
                }
            }
            .frame(height: 6)

            Text("\(available)/\(total)")
                .font(.caption2)
                .foregroundColor(.secondary)
                .frame(width: 35, alignment: .trailing)
        }
    }

    private var progress: CGFloat {
        guard total > 0 else { return 0 }
        return CGFloat(available) / CGFloat(total)
    }

    private var progressColor: Color {
        let ratio = Float(available) / Float(max(total, 1))
        if ratio >= 0.7 { return .green }
        if ratio >= 0.4 { return .orange }
        return .red
    }
}

struct FactorsDetailSheet: View {
    let factors: [ContributingFactor]
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(factors) { factor in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(factor.name)
                                .font(.headline)

                            Spacer()

                            Text(factor.category.rawValue)
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color(.systemGray5))
                                .cornerRadius(4)
                        }

                        if !factor.recommendation.isEmpty {
                            Text(factor.recommendation)
                                .font(.callout)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("Risk Factors")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        UnifiedNeuralEngineView()
    }
}
