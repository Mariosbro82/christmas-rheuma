//
//  MeditationPlayerView.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import SwiftUI

struct MeditationPlayerView: View {
    let session: MeditationSessionModel
    @ObservedObject var viewModel: MeditationViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var showingBeforeMetrics = true
    @State private var showingAfterMetrics = false
    @State private var stressBefore: Double = 5
    @State private var painBefore: Double = 5
    @State private var moodBefore: Double = 5

    @State private var stressAfter: Double = 5
    @State private var painAfter: Double = 5
    @State private var moodAfter: Double = 5
    @State private var energyAfter: Double = 5
    @State private var notes: String = ""

    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    gradient: Gradient(colors: [Color.purple.opacity(0.3), Color.blue.opacity(0.3)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                if showingBeforeMetrics {
                    beforeMetricsView
                } else if showingAfterMetrics {
                    afterMetricsView
                } else {
                    playerView
                }
            }
            .navigationTitle(session.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        viewModel.stopSession()
                        dismiss()
                    }
                }
            }
        }
    }

    // MARK: - Before Metrics View

    private var beforeMetricsView: some View {
        VStack(spacing: 32) {
            Spacer()

            VStack(spacing: 16) {
                // Custom dino character for starting meditation
                Image(MeditationAssets.startingImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 100, height: 100)

                Text("How are you feeling right now?")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .multilineTextAlignment(.center)

                Text("This helps us track meditation effectiveness")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding()

            VStack(spacing: 24) {
                MetricSlider(
                    title: "Stress Level",
                    value: $stressBefore,
                    icon: "brain.head.profile",
                    lowLabel: "Calm",
                    highLabel: "Stressed"
                )

                MetricSlider(
                    title: "Pain Level",
                    value: $painBefore,
                    icon: "heart.circle",
                    lowLabel: "No Pain",
                    highLabel: "High Pain"
                )

                MetricSlider(
                    title: "Mood",
                    value: $moodBefore,
                    icon: "face.smiling",
                    lowLabel: "Low",
                    highLabel: "Great"
                )
            }
            .padding()

            Button(action: {
                viewModel.stressLevelBefore = Int(stressBefore)
                viewModel.painLevelBefore = Int(painBefore)
                viewModel.moodBefore = Int(moodBefore)
                viewModel.startSession(session)
                withAnimation {
                    showingBeforeMetrics = false
                }
            }) {
                Text("Start Meditation")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.purple)
                    .cornerRadius(12)
            }
            .padding()

            Spacer()
        }
    }

    // MARK: - Player View

    private var playerView: some View {
        VStack(spacing: 40) {
            Spacer()

            // Session info
            VStack(spacing: 8) {
                Text(session.title)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .multilineTextAlignment(.center)

                Text(session.category.displayName)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            // Circular progress
            ZStack {
                // Background circle
                Circle()
                    .stroke(Color.white.opacity(0.3), lineWidth: 20)
                    .frame(width: 250, height: 250)

                // Progress circle
                Circle()
                    .trim(from: 0, to: viewModel.progress)
                    .stroke(Color.purple, style: StrokeStyle(lineWidth: 20, lineCap: .round))
                    .frame(width: 250, height: 250)
                    .rotationEffect(.degrees(-90))
                    .animation(.linear, value: viewModel.progress)

                // Time display
                VStack(spacing: 8) {
                    Text(viewModel.currentTimeFormatted)
                        .font(.system(size: 48, weight: .bold, design: .rounded))

                    Text(viewModel.remainingTimeFormatted + " remaining")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // Breathing guide (if applicable)
            if session.hasBreathingGuide {
                breathingGuide
            }

            // Controls
            HStack(spacing: 40) {
                Button(action: {
                    viewModel.stopSession()
                    dismiss()
                }) {
                    Image(systemName: "stop.fill")
                        .font(.title)
                        .foregroundColor(.white)
                        .frame(width: 60, height: 60)
                        .background(Color.red.opacity(0.8))
                        .clipShape(Circle())
                }

                Button(action: {
                    if viewModel.isPaused {
                        viewModel.resumeSession()
                    } else {
                        viewModel.pauseSession()
                    }
                }) {
                    Image(systemName: viewModel.isPaused ? "play.fill" : "pause.fill")
                        .font(.title)
                        .foregroundColor(.white)
                        .frame(width: 80, height: 80)
                        .background(Color.purple)
                        .clipShape(Circle())
                }

                Button(action: {
                    showingAfterMetrics = true
                }) {
                    Image(systemName: "checkmark")
                        .font(.title)
                        .foregroundColor(.white)
                        .frame(width: 60, height: 60)
                        .background(Color.green.opacity(0.8))
                        .clipShape(Circle())
                }
            }

            Spacer()
        }
        .padding()
        .onChange(of: viewModel.isPlaying) { _, newValue in
            if !newValue && viewModel.currentTime >= session.duration * 0.95 {
                // Session auto-completed
                showingAfterMetrics = true
            }
        }
    }

    // MARK: - Breathing Guide

    private var breathingGuide: some View {
        VStack(spacing: 8) {
            if let pattern = session.breathingPattern {
                Text(pattern.displayName)
                    .font(.headline)

                Text("Inhale: \(pattern.inhaleCount)s • Hold: \(pattern.holdCount)s • Exhale: \(pattern.exhaleCount)s")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.white.opacity(0.1))
        .cornerRadius(12)
    }

    // MARK: - After Metrics View

    private var afterMetricsView: some View {
        ScrollView {
            VStack(spacing: 32) {
                VStack(spacing: 16) {
                    // Happy dino for completed session
                    Image(MeditationAssets.completedImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 120, height: 120)

                    Text("Great job!")
                        .font(.title2)
                        .fontWeight(.semibold)

                    Text("How are you feeling now?")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding()

                VStack(spacing: 24) {
                    MetricSlider(
                        title: "Stress Level",
                        value: $stressAfter,
                        icon: "brain.head.profile",
                        lowLabel: "Calm",
                        highLabel: "Stressed"
                    )

                    MetricSlider(
                        title: "Pain Level",
                        value: $painAfter,
                        icon: "heart.circle",
                        lowLabel: "No Pain",
                        highLabel: "High Pain"
                    )

                    MetricSlider(
                        title: "Mood",
                        value: $moodAfter,
                        icon: "face.smiling",
                        lowLabel: "Low",
                        highLabel: "Great"
                    )

                    MetricSlider(
                        title: "Energy",
                        value: $energyAfter,
                        icon: "bolt.fill",
                        lowLabel: "Tired",
                        highLabel: "Energized"
                    )
                }
                .padding()

                // Notes
                VStack(alignment: .leading, spacing: 8) {
                    Text("Notes (optional)")
                        .font(.headline)

                    TextEditor(text: $notes)
                        .frame(height: 100)
                        .padding(8)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                }
                .padding()

                Button(action: {
                    Task {
                        do {
                            try await viewModel.completeSession(
                                stressAfter: Int(stressAfter),
                                painAfter: Int(painAfter),
                                moodAfter: Int(moodAfter),
                                energyAfter: Int(energyAfter),
                                notes: notes.isEmpty ? nil : notes
                            )
                            dismiss()
                        } catch {
                            // Handle error
                            print("Error completing session: \(error)")
                        }
                    }
                }) {
                    Text("Complete Session")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.purple)
                        .cornerRadius(12)
                }
                .padding()
            }
        }
    }
}

// MARK: - Metric Slider Component

struct MetricSlider: View {
    let title: String
    @Binding var value: Double
    let icon: String
    let lowLabel: String
    let highLabel: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                // Dynamic dino image based on value
                Image(dinoImageForValue)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 30, height: 30)

                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Spacer()
                Text("\(Int(value))/10")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Slider(value: $value, in: 0...10, step: 1)
                .tint(.purple)

            HStack {
                Text(lowLabel)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text(highLabel)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    /// Get appropriate dino image based on metric type and value
    private var dinoImageForValue: String {
        // For stress, pain, and other negative metrics (higher value = worse)
        if title.contains("Stress") || title.contains("Pain") {
            return MeditationAssets.image(forMoodLevel: Int(value))
        } else if title.contains("Mood") || title.contains("Energy") {
            // For positive metrics, invert the scale (higher value = better)
            return MeditationAssets.image(forMoodLevel: 10 - Int(value))
        } else {
            return "dino-casual"
        }
    }
}

// MARK: - Preview

#Preview {
    MeditationPlayerView(
        session: MeditationSessionModel.asSpecificSessions[0],
        viewModel: MeditationViewModel()
    )
}
