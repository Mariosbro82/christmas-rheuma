//
//  NeuralOptInView.swift
//  InflamAI
//
//  View for opting into neural network analysis
//  Explains benefits and privacy considerations
//

import SwiftUI

struct NeuralOptInView: View {
    @ObservedObject var viewModel: TriggerInsightsViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var isEnabling: Bool = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Hero Image
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 80))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.purple, .blue],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .padding(.top, 32)

                    // Title
                    VStack(spacing: 8) {
                        Text("Enhanced Pattern Analysis")
                            .font(.title)
                            .fontWeight(.bold)

                        Text("Advanced Data Visualization")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }

                    // Benefits Section
                    VStack(alignment: .leading, spacing: 16) {
                        Text("What You'll Get")
                            .font(.headline)

                        BenefitRow(
                            icon: "waveform.path.ecg",
                            title: "Pattern Visualization",
                            description: "Explore correlations between your logged factors and symptoms"
                        )

                        BenefitRow(
                            icon: "clock.arrow.2.circlepath",
                            title: "Temporal Patterns",
                            description: "See how factor combinations appear in your data over time"
                        )

                        BenefitRow(
                            icon: "chart.line.uptrend.xyaxis",
                            title: "Tomorrow's Outlook",
                            description: "View patterns based on today's logged factors (not medical advice)"
                        )

                        BenefitRow(
                            icon: "sparkles",
                            title: "Factor Attribution",
                            description: "See which factors most correlate with your logged symptoms"
                        )
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    // Privacy Section
                    VStack(alignment: .leading, spacing: 12) {
                        Label("Privacy First", systemImage: "lock.shield.fill")
                            .font(.headline)
                            .foregroundStyle(.green)

                        Text("All neural network training and inference happens entirely on your device. Your data never leaves your phone.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                            Text("100% on-device processing")
                                .font(.caption)
                        }

                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                            Text("No cloud uploads")
                                .font(.caption)
                        }

                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                            Text("You can disable anytime")
                                .font(.caption)
                        }
                    }
                    .padding()
                    .background(Color.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    // Requirements
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Requirements")
                            .font(.headline)

                        HStack {
                            Image(systemName: viewModel.daysOfData >= 90 ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(viewModel.daysOfData >= 90 ? .green : .red)
                            Text("90+ days of data (\(viewModel.daysOfData) days)")
                                .font(.subheadline)
                        }

                        Text("Neural networks require substantial data to learn your personal patterns accurately.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    // Enable Button
                    Button {
                        Task {
                            isEnabling = true
                            await viewModel.enableNeuralEngine()
                            isEnabling = false
                            dismiss()
                        }
                    } label: {
                        if isEnabling {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .tint(.white)
                        } else {
                            Text("Enable Neural Network")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                    .disabled(viewModel.daysOfData < 90 || isEnabling)
                    .frame(maxWidth: .infinity)
                    .padding(.top)

                    // Skip Button
                    Button("Maybe Later") {
                        dismiss()
                    }
                    .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Neural Network")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }
}

struct BenefitRow: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.purple)
                .frame(width: 32)

            VStack(alignment: .leading) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

#Preview {
    NeuralOptInView(viewModel: TriggerInsightsViewModel())
}
