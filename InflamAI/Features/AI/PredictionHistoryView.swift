//
//  PredictionHistoryView.swift
//  InflamAI
//
//  Shows prediction history and accuracy metrics
//

import SwiftUI
import Charts

struct PredictionHistoryView: View {
    @ObservedObject private var mlIntegration = MLIntegrationService.shared
    @ObservedObject private var neuralEngine = UnifiedNeuralEngine.shared

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Accuracy Summary Card
                accuracySummaryCard

                // Current Prediction
                if let prediction = neuralEngine.currentPrediction {
                    currentPredictionCard(prediction)
                }

                // Learning Progress
                learningProgressCard

                // Prediction Factors
                if let prediction = neuralEngine.currentPrediction {
                    factorsCard(prediction)
                }

                // Stats Grid
                statsGrid
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Prediction History")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Accuracy Summary

    private var accuracySummaryCard: some View {
        let metrics = mlIntegration.getAccuracyMetrics()

        return VStack(spacing: 16) {
            HStack {
                Image(systemName: "chart.pie.fill")
                    .font(.title2)
                    .foregroundColor(.purple)

                Text("Model Accuracy")
                    .font(.headline)

                Spacer()

                if metrics.hasEnoughData {
                    Text("\(Int(metrics.accuracy * 100))%")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(accuracyColor(metrics.accuracy))
                }
            }

            if metrics.hasEnoughData {
                // Accuracy breakdown
                HStack(spacing: 20) {
                    metricItem(title: "Precision", value: metrics.precision)
                    metricItem(title: "Recall", value: metrics.recall)
                    metricItem(title: "F1 Score", value: metrics.f1Score)
                }

                Text("Based on \(metrics.validatedPredictions) validated predictions")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                VStack(spacing: 8) {
                    ProgressView(value: Float(metrics.validatedPredictions), total: 10)
                        .tint(.purple)

                    Text(metrics.summary)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 5, x: 0, y: 2)
    }

    private func metricItem(title: String, value: Float) -> some View {
        VStack(spacing: 4) {
            Text("\(Int(value * 100))%")
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundColor(.primary)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Current Prediction

    private func currentPredictionCard(_ prediction: NeuralPrediction) -> some View {
        VStack(spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(.purple)

                Text("Current Prediction")
                    .font(.headline)

                Spacer()

                Text(prediction.timestamp, style: .relative)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            HStack(alignment: .center, spacing: 20) {
                // Risk gauge
                ZStack {
                    Circle()
                        .stroke(Color.gray.opacity(0.2), lineWidth: 10)
                        .frame(width: 100, height: 100)

                    Circle()
                        .trim(from: 0, to: CGFloat(prediction.probability))
                        .stroke(
                            riskColor(Double(prediction.probability)),
                            style: StrokeStyle(lineWidth: 10, lineCap: .round)
                        )
                        .frame(width: 100, height: 100)
                        .rotationEffect(.degrees(-90))

                    VStack(spacing: 2) {
                        Text("\(Int(prediction.probability * 100))")
                            .font(.system(size: 28, weight: .bold, design: .rounded))
                            .foregroundColor(riskColor(Double(prediction.probability)))

                        Text("%")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Label(prediction.riskLevel.rawValue, systemImage: prediction.riskLevel.icon)
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(riskColor(Double(prediction.probability)))

                    Text("Confidence: \(prediction.confidence.rawValue)")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    if prediction.willFlare {
                        Text(prediction.recommendedAction.rawValue)
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                }

                Spacer()
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 5, x: 0, y: 2)
    }

    // MARK: - Learning Progress

    private var learningProgressCard: some View {
        VStack(spacing: 16) {
            HStack {
                Image(systemName: "graduationcap.fill")
                    .font(.title2)
                    .foregroundColor(.blue)

                Text("Learning Progress")
                    .font(.headline)

                Spacer()

                Text(neuralEngine.personalizationPhase.rawValue)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(phaseColor)
                    .cornerRadius(8)
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("\(neuralEngine.daysOfUserData) days of data")
                        .font(.subheadline)

                    Spacer()

                    Text("\(Int(neuralEngine.learningProgress * 100))%")
                        .font(.subheadline)
                        .fontWeight(.medium)
                }

                ProgressView(value: neuralEngine.learningProgress)
                    .tint(phaseColor)

                Text(phaseDescription)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 5, x: 0, y: 2)
    }

    // MARK: - Factors Card

    private func factorsCard(_ prediction: NeuralPrediction) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "list.bullet.clipboard")
                    .font(.title2)
                    .foregroundColor(.orange)

                Text("Contributing Factors")
                    .font(.headline)

                Spacer()
            }

            ForEach(prediction.topFactors, id: \.id) { factor in
                HStack(spacing: 12) {
                    Image(systemName: factorIcon(factor.name))
                        .foregroundColor(factorColor(factor.name))
                        .frame(width: 24)

                    Text(factor.name)
                        .font(.subheadline)

                    Spacer()
                }
                .padding(.vertical, 4)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 5, x: 0, y: 2)
    }

    // MARK: - Stats Grid

    private var statsGrid: some View {
        let metrics = mlIntegration.getAccuracyMetrics()

        return VStack(spacing: 16) {
            HStack {
                Image(systemName: "square.grid.2x2.fill")
                    .font(.title2)
                    .foregroundColor(.green)

                Text("Statistics")
                    .font(.headline)

                Spacer()
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                statCard(title: "Total Samples", value: "\(metrics.totalPredictions)", icon: "doc.text.fill", color: .blue)
                statCard(title: "Validated", value: "\(metrics.validatedPredictions)", icon: "checkmark.seal.fill", color: .green)
                statCard(title: "Model Version", value: "v\(neuralEngine.modelVersion)", icon: "cpu.fill", color: .purple)
                statCard(title: "Days Tracked", value: "\(neuralEngine.daysOfUserData)", icon: "calendar", color: .orange)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 5, x: 0, y: 2)
    }

    private func statCard(title: String, value: String, icon: String, color: Color) -> some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)

            Text(value)
                .font(.title2)
                .fontWeight(.bold)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }

    // MARK: - Helpers

    private func riskColor(_ probability: Double) -> Color {
        switch probability {
        case 0..<0.25: return .green
        case 0.25..<0.5: return .yellow
        case 0.5..<0.75: return .orange
        default: return .red
        }
    }

    private func accuracyColor(_ accuracy: Float) -> Color {
        switch accuracy {
        case 0.8...: return .green
        case 0.6..<0.8: return .yellow
        default: return .orange
        }
    }

    private var phaseColor: Color {
        switch neuralEngine.personalizationPhase {
        case .bootstrap: return .gray
        case .earlyLearning: return .blue
        case .adapting: return .cyan
        case .personalized: return .green
        case .expert: return .purple
        }
    }

    private var phaseDescription: String {
        switch neuralEngine.personalizationPhase {
        case .bootstrap:
            return "Building baseline predictions with initial data"
        case .earlyLearning:
            return "Learning your unique symptom patterns"
        case .adapting:
            return "Refining predictions based on your feedback"
        case .personalized:
            return "Predictions tailored to your history"
        case .expert:
            return "Fully personalized with high confidence"
        }
    }

    private func factorIcon(_ factor: String) -> String {
        // Disease activity
        if factor.contains("BASDAI") || factor.contains("activity") {
            return "waveform.path.ecg"
        } else if factor.contains("trend") || factor.contains("Worsening") {
            return "arrow.up.right"
        }
        // Weather factors
        else if factor.contains("pressure") {
            return "barometer"
        } else if factor.contains("humidity") {
            return "humidity.fill"
        } else if factor.contains("Cold") || factor.contains("weather") || factor.contains("temperature") {
            return "thermometer.snowflake"
        }
        // HealthKit biometrics
        else if factor.contains("HRV") || factor.contains("stress") {
            return "heart.text.square"
        } else if factor.contains("sleep") || factor.contains("Sleep") {
            return "moon.zzz.fill"
        } else if factor.contains("steps") || factor.contains("activity") {
            return "figure.walk"
        } else if factor.contains("resting HR") || factor.contains("heart rate") {
            return "heart.fill"
        }
        // Status
        else if factor.contains("normal") {
            return "checkmark.circle.fill"
        }
        return "circle.fill"
    }

    private func factorColor(_ factor: String) -> Color {
        // Positive/normal
        if factor.contains("normal") {
            return .green
        }
        // High severity
        else if factor.contains("High") || factor.contains("Rapid") || factor.contains("Poor") || factor.contains("Low HRV") {
            return .red
        }
        // Moderate severity
        else if factor.contains("Elevated") || factor.contains("dropping") || factor.contains("Insufficient") || factor.contains("Below") {
            return .orange
        }
        // HealthKit factors (blue theme)
        else if factor.contains("HRV") || factor.contains("sleep") || factor.contains("steps") || factor.contains("HR") {
            return .purple
        }
        // Weather factors
        else if factor.contains("pressure") || factor.contains("humidity") || factor.contains("Cold") {
            return .cyan
        }
        return .blue
    }
}

#Preview {
    NavigationStack {
        PredictionHistoryView()
    }
}
