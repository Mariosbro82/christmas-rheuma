//
//  MeditationProgressView.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import SwiftUI
import Charts

struct MeditationProgressView: View {
    @ObservedObject var viewModel: MeditationViewModel
    @State private var impactAnalysis: MeditationImpactAnalysis?
    @State private var isLoadingAnalysis = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Streak card
                streakCard

                // Weekly/Monthly stats
                statsCards

                // Impact analysis
                impactAnalysisSection

                // Recent sessions
                recentSessionsSection

                // Favorite types
                favoriteTypesSection
            }
            .padding()
        }
        .navigationTitle("Progress")
        .onAppear {
            loadImpactAnalysis()
        }
    }

    // MARK: - Streak Card

    private var streakCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                // Strong dino for active streak
                Image(MeditationAssets.streakActiveImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 60, height: 60)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Current Streak")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(viewModel.streak?.currentStreak ?? 0) days")
                        .font(.title2)
                        .fontWeight(.bold)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    Text("Longest Streak")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(viewModel.streak?.longestStreak ?? 0) days")
                        .font(.title3)
                        .fontWeight(.semibold)
                }
            }

            Divider()

            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Total Sessions")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(viewModel.streak?.totalSessions ?? 0)")
                        .font(.headline)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    Text("Total Minutes")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(Int(viewModel.streak?.totalMinutes ?? 0))")
                        .font(.headline)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 4)
    }

    // MARK: - Stats Cards

    private var statsCards: some View {
        HStack(spacing: 16) {
            MeditationStatCard(
                title: "This Week",
                value: "\(Int(viewModel.getTotalMinutesThisWeek())) min",
                icon: "calendar.badge.clock",
                color: .blue,
                goal: Int(viewModel.streak?.weeklyGoal ?? 7),
                progress: Double(viewModel.streak?.weeklyProgress ?? 0)
            )

            MeditationStatCard(
                title: "This Month",
                value: "\(Int(viewModel.getTotalMinutesThisMonth())) min",
                icon: "calendar",
                color: .purple,
                goal: Int(viewModel.streak?.monthlyGoal ?? 30),
                progress: Double(viewModel.streak?.monthlyProgress ?? 0)
            )
        }
    }

    // MARK: - Impact Analysis Section

    private var impactAnalysisSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Meditation Impact")
                    .font(.headline)

                Spacer()

                if isLoadingAnalysis {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }

            if let analysis = impactAnalysis {
                if analysis.hasSignificantImpact {
                    significantImpactView(analysis)
                } else {
                    insufficientDataView(analysis)
                }
            } else {
                insufficientDataPlaceholder
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 4)
    }

    private func significantImpactView(_ analysis: MeditationImpactAnalysis) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            // Summary
            Text(analysis.summaryDescription)
                .font(.subheadline)
                .foregroundColor(.secondary)

            // Pain comparison
            HStack(spacing: 40) {
                VStack(spacing: 8) {
                    Text("With Meditation")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.1f", analysis.avgPainWithMeditation))
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(.green)
                    Text("/ 10")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Image(systemName: "arrow.right")
                    .font(.title2)
                    .foregroundColor(.secondary)

                VStack(spacing: 8) {
                    Text("Without Meditation")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.1f", analysis.avgPainWithoutMeditation))
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(.orange)
                    Text("/ 10")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)

            // Reduction percentage
            HStack {
                Image(systemName: "arrow.down.circle.fill")
                    .foregroundColor(.green)
                Text("\(Int(analysis.painReductionPercentage))% pain reduction")
                    .font(.headline)
                    .foregroundColor(.green)
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.green.opacity(0.1))
            .cornerRadius(12)

            // Confidence level
            HStack {
                Image(systemName: "chart.bar.fill")
                    .foregroundColor(.purple)
                Text("Confidence: \(analysis.confidence.displayName)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Additional metrics
            if analysis.stressReduction > 0 {
                HStack {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(.blue)
                    Text("Stress reduced by \(String(format: "%.1f", analysis.stressReduction)) points")
                        .font(.caption)
                }
            }

            if let sleepImprovement = analysis.sleepImprovement, sleepImprovement > 0 {
                HStack {
                    Image(systemName: "moon.stars.fill")
                        .foregroundColor(.indigo)
                    Text("Sleep quality improved by \(String(format: "%.1f", sleepImprovement)) points")
                        .font(.caption)
                }
            }
        }
    }

    private func insufficientDataView(_ analysis: MeditationImpactAnalysis) -> some View {
        VStack(spacing: 12) {
            // Encouraging dino
            Image(MeditationAssets.progressImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 80, height: 80)

            Text("Keep Meditating!")
                .font(.headline)

            Text("We need more data to show your personalized meditation impact. Complete at least 7 days with meditation and 7 days without to see your analytics.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            HStack(spacing: 24) {
                VStack(spacing: 4) {
                    Text("\(analysis.daysWithMeditation)")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("With Meditation")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                VStack(spacing: 4) {
                    Text("\(analysis.daysWithoutMeditation)")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Without")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
        }
    }

    private var insufficientDataPlaceholder: some View {
        VStack(spacing: 12) {
            // Meditating dino for starting journey
            Image(MeditationAssets.meditatingImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 80, height: 80)

            Text("No data yet")
                .font(.headline)

            Text("Start your meditation journey to see personalized insights!")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    // MARK: - Recent Sessions

    private var recentSessionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Sessions")
                .font(.headline)

            ForEach(viewModel.recentSessions.prefix(5), id: \.id) { session in
                RecentSessionRow(session: session)
            }

            if viewModel.recentSessions.isEmpty {
                Text("No sessions yet. Start your first meditation!")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding()
            }
        }
    }

    // MARK: - Favorite Types

    private var favoriteTypesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Favorite Types")
                .font(.headline)

            ForEach(viewModel.getFavoriteTypes(), id: \.type) { item in
                HStack {
                    Text(MeditationType(rawValue: item.type)?.displayName ?? item.type)
                        .font(.subheadline)

                    Spacer()

                    Text("\(item.count) sessions")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
            }

            if viewModel.getFavoriteTypes().isEmpty {
                Text("Complete more sessions to see your favorites!")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding()
            }
        }
    }

    // MARK: - Helper Functions

    private func loadImpactAnalysis() {
        isLoadingAnalysis = true
        Task {
            let engine = MeditationCorrelationEngine()
            do {
                let analysis = try engine.analyzeMeditationImpact(days: 30)
                await MainActor.run {
                    self.impactAnalysis = analysis
                    self.isLoadingAnalysis = false
                }
            } catch {
                print("Error analyzing meditation impact: \(error)")
                await MainActor.run {
                    self.isLoadingAnalysis = false
                }
            }
        }
    }
}

// MARK: - Meditation Stat Card Component

struct MeditationStatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    let goal: Int
    let progress: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                // Progress dino
                Image(MeditationAssets.progressImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 40, height: 40)
                Spacer()
            }

            Text(value)
                .font(.title2)
                .fontWeight(.bold)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)

            ProgressView(value: progress, total: Double(goal))
                .tint(color)

            Text("\(Int(progress))/\(goal) days")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 2)
    }
}

// MARK: - Recent Session Row

struct RecentSessionRow: View {
    let session: MeditationSession

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                if let title = session.title {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.medium)
                }

                if let timestamp = session.timestamp {
                    Text(timestamp, style: .date)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                Text("\(session.completedDuration / 60) min")
                    .font(.caption)
                    .foregroundColor(.secondary)

                let painBefore = session.painLevelBefore
                let painAfter = session.painLevelAfter

                if painBefore != 0 && painAfter != 0 {
                    let reduction = painBefore - painAfter
                    HStack(spacing: 4) {
                        Image(systemName: reduction > 0 ? "arrow.down" : "arrow.up")
                            .font(.caption2)
                        Text("\(abs(reduction))")
                            .font(.caption2)
                    }
                    .foregroundColor(reduction > 0 ? .green : .red)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Preview

#Preview {
    NavigationView {
        MeditationProgressView(viewModel: MeditationViewModel())
    }
}
