//
//  NeuralEngineStatusView.swift
//  InflamAI
//
//  Compact status widget for the Unified Neural Engine
//  Can be embedded in any view to show current ML status
//

import SwiftUI

/// Compact Neural Engine status widget
/// Embed this in HomeView, DashboardView, or any screen where
/// you want to show the current prediction status
struct NeuralEngineStatusView: View {
    @ObservedObject private var engine = UnifiedNeuralEngine.shared
    @State private var showDetailSheet = false

    enum Style {
        case compact      // Single line with icon
        case card         // Card with prediction details
        case minimal      // Just the icon indicator
        case inline       // Inline text only
    }

    let style: Style

    init(style: Style = .compact) {
        self.style = style
    }

    var body: some View {
        switch style {
        case .compact:
            compactView
        case .card:
            cardView
        case .minimal:
            minimalView
        case .inline:
            inlineView
        }
    }

    // MARK: - Compact View

    private var compactView: some View {
        Button {
            showDetailSheet = true
        } label: {
            HStack(spacing: 12) {
                // Status indicator
                statusIndicator

                // Text info
                VStack(alignment: .leading, spacing: 2) {
                    Text("Neural Engine")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)

                    if let prediction = engine.currentPrediction {
                        Text("\(Int(prediction.probability * 100))% flare risk")
                            .font(.caption)
                            .foregroundColor(prediction.willFlare ? .orange : .green)
                    } else {
                        Text(engine.engineStatus.displayMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Spacer()

                // Chevron
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $showDetailSheet) {
            NavigationStack {
                UnifiedNeuralEngineView()
            }
        }
    }

    // MARK: - Card View

    private var cardView: some View {
        VStack(spacing: 12) {
            // Header
            HStack {
                statusIndicator

                Text("Neural Engine")
                    .font(.headline)

                Spacer()

                Button {
                    showDetailSheet = true
                } label: {
                    Image(systemName: "arrow.up.right.square")
                        .foregroundColor(.accentColor)
                }
            }

            // Prediction info
            if let prediction = engine.currentPrediction {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 4) {
                            Text(prediction.willFlare ? "Flare Likely" : "Low Risk")
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(prediction.willFlare ? .orange : .green)
                        }

                        Text(prediction.summary)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    // Risk gauge
                    riskGauge(probability: CGFloat(prediction.probability))
                }

                // Personalization status
                HStack {
                    Image(systemName: engine.isPersonalized ? "person.crop.circle.badge.checkmark" : "person.crop.circle.badge.clock")
                        .foregroundColor(engine.isPersonalized ? .green : .orange)

                    Text(engine.isPersonalized
                         ? "Personalized to your patterns"
                         : "Learning your patterns...")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Spacer()

                    Text("v\(engine.modelVersion)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            } else {
                // No prediction - clean empty state
                VStack(spacing: 16) {
                    ZStack {
                        Circle()
                            .fill(Color.purple.opacity(0.15))
                            .frame(width: 64, height: 64)

                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 28))
                            .foregroundColor(.purple)
                    }

                    VStack(spacing: 4) {
                        Text("No Predictions Yet")
                            .font(.subheadline)
                            .fontWeight(.semibold)

                        Text(engine.daysOfUserData < 7
                             ? "Need \(7 - engine.daysOfUserData) more days of data"
                             : "Continue logging to enable predictions")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 8)
        .sheet(isPresented: $showDetailSheet) {
            NavigationStack {
                UnifiedNeuralEngineView()
            }
        }
    }

    // MARK: - Minimal View

    private var minimalView: some View {
        Button {
            showDetailSheet = true
        } label: {
            statusIndicator
        }
        .sheet(isPresented: $showDetailSheet) {
            NavigationStack {
                UnifiedNeuralEngineView()
            }
        }
    }

    // MARK: - Inline View

    private var inlineView: some View {
        HStack(spacing: 6) {
            statusDot

            if let prediction = engine.currentPrediction {
                Text("\(Int(prediction.probability * 100))% risk")
                    .font(.caption)
                    .foregroundColor(prediction.willFlare ? .orange : .green)
            } else {
                Text(statusText)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Supporting Views

    private var statusIndicator: some View {
        ZStack {
            Circle()
                .fill(statusColor.opacity(0.2))
                .frame(width: 40, height: 40)

            Image(systemName: "brain.head.profile")
                .font(.system(size: 18))
                .foregroundColor(statusColor)
                .symbolEffect(.pulse, isActive: engine.engineStatus == .learning)
        }
    }

    private var statusDot: some View {
        Circle()
            .fill(statusColor)
            .frame(width: 8, height: 8)
    }

    private func riskGauge(probability: CGFloat) -> some View {
        ZStack {
            Circle()
                .stroke(Color.gray.opacity(0.2), lineWidth: 6)
                .frame(width: 60, height: 60)

            Circle()
                .trim(from: 0, to: probability)
                .stroke(
                    probability > 0.5 ? Color.orange : Color.green,
                    style: StrokeStyle(lineWidth: 6, lineCap: .round)
                )
                .frame(width: 60, height: 60)
                .rotationEffect(.degrees(-90))

            Text("\(Int(probability * 100))%")
                .font(.caption)
                .fontWeight(.bold)
        }
    }

    // MARK: - Computed Properties

    private var statusColor: Color {
        switch engine.engineStatus {
        case .ready:
            if let prediction = engine.currentPrediction {
                return prediction.willFlare ? .orange : .green
            }
            return .green
        case .initializing:
            return .orange
        case .learning:
            return .blue
        case .error:
            return .red
        }
    }

    private var statusText: String {
        switch engine.engineStatus {
        case .ready: return "Ready"
        case .initializing: return "Starting..."
        case .learning: return "Learning..."
        case .error: return "Error"
        }
    }
}

// MARK: - Home Widget Variant

/// A larger widget designed for the home screen dashboard
struct NeuralEngineHomeWidget: View {
    @ObservedObject private var engine = UnifiedNeuralEngine.shared
    @State private var showDetailView = false

    var body: some View {
        Button {
            showDetailView = true
        } label: {
            VStack(spacing: 16) {
                // Header
                HStack {
                    Image(systemName: "brain.head.profile")
                        .font(.title2)
                        .foregroundColor(.purple)

                    Text("AI Predictions")
                        .font(.headline)

                    Spacer()

                    statusBadge
                }

                // Main content
                if let prediction = engine.currentPrediction {
                    HStack(alignment: .center, spacing: 16) {
                        // Risk level
                        VStack(alignment: .leading, spacing: 4) {
                            Text("3-7 Day Flare Risk")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text("\(Int(prediction.probability * 100))%")
                                .font(.system(size: 36, weight: .bold))
                                .foregroundColor(prediction.willFlare ? .orange : .green)

                            Text(prediction.riskLevel.rawValue)
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(prediction.willFlare ? .orange : .green)
                        }

                        Spacer()

                        // Visual indicator
                        ZStack {
                            Circle()
                                .fill(prediction.willFlare ? Color.orange.opacity(0.2) : Color.green.opacity(0.2))
                                .frame(width: 80, height: 80)

                            Image(systemName: prediction.riskLevel.icon)
                                .font(.system(size: 32))
                                .foregroundColor(prediction.willFlare ? .orange : .green)
                        }
                    }

                    // Confidence bar
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Confidence")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(prediction.confidence.rawValue)
                                .font(.caption)
                                .fontWeight(.medium)
                        }

                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Color(.systemGray5))

                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Color.purple)
                                    .frame(width: geometry.size.width * confidenceValue(prediction.confidence))
                            }
                        }
                        .frame(height: 6)
                    }
                } else {
                    // Empty state - clean design matching weather
                    VStack(spacing: 16) {
                        ZStack {
                            Circle()
                                .fill(Color.purple.opacity(0.15))
                                .frame(width: 64, height: 64)

                            Image(systemName: "waveform.path.ecg")
                                .font(.system(size: 28))
                                .foregroundColor(.purple)
                        }

                        VStack(spacing: 4) {
                            Text("AI Predictions")
                                .font(.subheadline)
                                .fontWeight(.semibold)

                            Text("Start logging to enable predictions")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        if engine.daysOfUserData > 0 {
                            VStack(spacing: 8) {
                                GeometryReader { geometry in
                                    ZStack(alignment: .leading) {
                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(Color(.systemGray5))

                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(Color.purple)
                                            .frame(width: geometry.size.width * (CGFloat(engine.daysOfUserData) / 7.0))
                                    }
                                }
                                .frame(height: 6)

                                Text("\(engine.daysOfUserData) of 7 days")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(16)
            .shadow(color: Color.black.opacity(0.1), radius: 8)
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $showDetailView) {
            NavigationStack {
                UnifiedNeuralEngineView()
            }
        }
    }

    private var statusBadge: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(badgeColor)
                .frame(width: 6, height: 6)

            Text(badgeText)
                .font(.caption2)
                .fontWeight(.medium)
                .foregroundColor(badgeColor)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(badgeColor.opacity(0.1))
        .cornerRadius(8)
    }

    private var badgeColor: Color {
        switch engine.engineStatus {
        case .ready: return .green
        case .initializing: return .orange
        case .learning: return .blue
        case .error: return .red
        }
    }

    private var badgeText: String {
        if engine.isPersonalized {
            return "Personalized"
        }
        switch engine.engineStatus {
        case .ready: return "Ready"
        case .initializing: return "Starting"
        case .learning: return "Learning"
        case .error: return "Error"
        }
    }

    private func confidenceValue(_ level: ConfidenceLevel) -> CGFloat {
        switch level {
        case .low: return 0.25
        case .moderate: return 0.50
        case .high: return 0.75
        case .veryHigh: return 1.0
        }
    }
}

// MARK: - Quick Check-In Integration

/// View to show neural engine feedback after completing a check-in
struct NeuralEngineFeedbackView: View {
    @ObservedObject private var engine = UnifiedNeuralEngine.shared
    @Environment(\.dismiss) var dismiss
    let onComplete: () -> Void

    @State private var isUpdating = false
    @State private var updateComplete = false

    var body: some View {
        VStack(spacing: 24) {
            // Header
            Image(systemName: updateComplete ? "checkmark.circle.fill" : "brain.head.profile")
                .font(.system(size: 60))
                .foregroundColor(updateComplete ? .green : .purple)
                .symbolEffect(.bounce, value: updateComplete)

            Text(updateComplete ? "Neural Engine Updated" : "Updating Neural Engine")
                .font(.title2)
                .fontWeight(.bold)

            Text(updateComplete
                 ? "Your data has been processed. The model will continue learning from your patterns."
                 : "Processing your check-in data to improve predictions...")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            // Progress or completion
            if !updateComplete {
                ProgressView()
                    .scaleEffect(1.2)
            } else if let prediction = engine.currentPrediction {
                // Show updated prediction
                VStack(spacing: 8) {
                    Text("Current Risk Assessment")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 4) {
                        Image(systemName: prediction.riskLevel.icon)
                        Text("\(Int(prediction.probability * 100))%")
                            .font(.title)
                            .fontWeight(.bold)
                        Text(prediction.riskLevel.rawValue)
                    }
                    .foregroundColor(prediction.willFlare ? .orange : .green)
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)
            }

            Spacer()

            // Continue button
            if updateComplete {
                Button {
                    onComplete()
                    dismiss()
                } label: {
                    Text("Continue")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.accentColor)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
            }
        }
        .padding()
        .task {
            await updateEngine()
        }
    }

    private func updateEngine() async {
        isUpdating = true

        // Refresh the prediction with new data
        await engine.refresh()

        // Small delay for UX
        try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second

        withAnimation {
            updateComplete = true
        }

        isUpdating = false
    }
}

// MARK: - Preview

#Preview("Compact") {
    VStack(spacing: 16) {
        NeuralEngineStatusView(style: .compact)
        NeuralEngineStatusView(style: .card)
        NeuralEngineStatusView(style: .minimal)
        NeuralEngineStatusView(style: .inline)
    }
    .padding()
}

#Preview("Home Widget") {
    NeuralEngineHomeWidget()
        .padding()
}
