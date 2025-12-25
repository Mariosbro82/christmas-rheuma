//
//  ContentView.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//  Redesigned with beautiful tab navigation
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var connectivityManager: WatchConnectivityManager
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            // Home Tab - Quick Overview
            HomeTabView()
                .tag(0)

            // Quick Log Tab
            QuickLogView()
                .tag(1)

            // Medications Tab
            MedicationTrackerView()
                .tag(2)

            // Exercises Tab
            ExercisesWatchView()
                .tag(3)

            // Flare Tab
            FlareLogView()
                .tag(4)
        }
        .tabViewStyle(.page)
    }
}

// MARK: - Home Tab View

struct HomeTabView: View {
    @EnvironmentObject var connectivityManager: WatchConnectivityManager
    @EnvironmentObject var healthViewModel: WatchHealthViewModel
    @State private var currentTime = Date()

    let timer = Timer.publish(every: 60, on: .main, in: .common).autoconnect()

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Header with Time
                VStack(spacing: 4) {
                    Text("InflamAI")
                        .font(.headline)
                        .fontWeight(.bold)

                    Text(currentTime, style: .time)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 8)

                // Connection Status Card
                ConnectionStatusCard()

                // Flare Risk Prediction (ML)
                if connectivityManager.flareRiskPrediction != nil {
                    FlareRiskCard()
                }

                // Today's Summary
                TodaySummaryCard()

                // Quick Actions
                QuickActionsGrid()

                // Health Metrics
                HealthMetricsCard()
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 8)
        }
        .onReceive(timer) { _ in
            currentTime = Date()
        }
    }
}

// MARK: - Connection Status Card

struct ConnectionStatusCard: View {
    @EnvironmentObject var connectivityManager: WatchConnectivityManager

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(statusColor.opacity(0.2))
                    .frame(width: 40, height: 40)

                Image(systemName: statusIcon)
                    .foregroundColor(statusColor)
                    .font(.system(size: 16))
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(statusText)
                    .font(.system(size: 14, weight: .semibold))

                Text("iPhone")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            if connectivityManager.isReachable {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.system(size: 18))
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(white: 0.15))
        )
    }

    private var statusColor: Color {
        connectivityManager.isReachable ? .green : .gray
    }

    private var statusIcon: String {
        connectivityManager.isReachable ? "iphone.radiowaves.left.and.right" : "iphone.slash"
    }

    private var statusText: String {
        connectivityManager.isReachable ? "Connected" : "Disconnected"
    }
}

// MARK: - Today's Summary Card

struct TodaySummaryCard: View {
    @EnvironmentObject var connectivityManager: WatchConnectivityManager

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "calendar")
                    .foregroundColor(.blue)
                Text("Today's Activity")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
            }

            if let summary = connectivityManager.activitySummary {
                VStack(spacing: 8) {
                    if let med = summary.lastMedication {
                        SummaryRow(
                            icon: "pills.fill",
                            color: .blue,
                            title: "Last Med",
                            subtitle: med.name,
                            time: med.timestamp
                        )
                    }

                    if let flare = summary.lastFlare {
                        SummaryRow(
                            icon: "flame.fill",
                            color: .red,
                            title: "Last Flare",
                            subtitle: "Severity \(flare.severity)",
                            time: flare.timestamp
                        )
                    }

                    if let log = summary.lastQuickLog {
                        SummaryRow(
                            icon: "heart.text.square.fill",
                            color: .pink,
                            title: "Last Log",
                            subtitle: "Pain: \(log.painScore)/10",
                            time: log.timestamp
                        )
                    }
                }
            } else {
                Text("No recent activity")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.vertical, 8)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(white: 0.15))
        )
    }
}

struct SummaryRow: View {
    let icon: String
    let color: Color
    let title: String
    let subtitle: String
    let time: Date

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.system(size: 14))
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(subtitle)
                    .font(.system(size: 13, weight: .medium))
            }

            Spacer()

            Text(time, style: .relative)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Quick Actions Grid

struct QuickActionsGrid: View {
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Quick Actions")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                NavigationLink(destination: QuickLogView()) {
                    QuickActionButton(
                        icon: "heart.text.square.fill",
                        title: "Quick Log",
                        color: .pink,
                        gradient: [.pink, .pink.opacity(0.7)]
                    )
                }
                .buttonStyle(.plain)

                NavigationLink(destination: FlareLogView()) {
                    QuickActionButton(
                        icon: "flame.fill",
                        title: "Log Flare",
                        color: .red,
                        gradient: [.red, .orange]
                    )
                }
                .buttonStyle(.plain)

                NavigationLink(destination: MedicationTrackerView()) {
                    QuickActionButton(
                        icon: "pills.fill",
                        title: "Meds",
                        color: .blue,
                        gradient: [.blue, .cyan]
                    )
                }
                .buttonStyle(.plain)

                NavigationLink(destination: ExercisesWatchView()) {
                    QuickActionButton(
                        icon: "figure.walk",
                        title: "Exercise",
                        color: .green,
                        gradient: [.green, .mint]
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }
}

struct QuickActionButton: View {
    let icon: String
    let title: String
    let color: Color
    let gradient: [Color]

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundColor(.white)

            Text(title)
                .font(.caption2)
                .fontWeight(.medium)
                .foregroundColor(.white)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 70)
        .background(
            LinearGradient(
                colors: gradient,
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .cornerRadius(12)
    }
}

// MARK: - Health Metrics Card

struct HealthMetricsCard: View {
    @EnvironmentObject var healthViewModel: WatchHealthViewModel

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "heart.fill")
                    .foregroundColor(.red)
                Text("Health Metrics")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
            }

            HStack(spacing: 8) {
                MetricPill(
                    icon: "figure.walk",
                    value: healthViewModel.steps.map { "\(Int($0))" } ?? "--",
                    label: "Steps",
                    color: .green
                )

                MetricPill(
                    icon: "heart.fill",
                    value: healthViewModel.heartRate.map { "\(Int($0))" } ?? "--",
                    label: "BPM",
                    color: .red
                )
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(white: 0.15))
        )
    }
}

struct MetricPill: View {
    let icon: String
    let value: String
    let label: String
    let color: Color

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.system(size: 12))

            VStack(alignment: .leading, spacing: 0) {
                Text(value)
                    .font(.system(size: 13, weight: .bold))
                Text(label)
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .padding(.horizontal, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(color.opacity(0.1))
        )
    }
}

// MARK: - Flare Risk Card

struct FlareRiskCard: View {
    @EnvironmentObject var connectivityManager: WatchConnectivityManager

    var body: some View {
        if let prediction = connectivityManager.flareRiskPrediction {
            VStack(spacing: 12) {
                HStack {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(.purple)
                    Text("Flare Risk")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Text(prediction.updatedAt, style: .relative)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                HStack(spacing: 12) {
                    // Risk gauge
                    ZStack {
                        Circle()
                            .stroke(riskColor(for: prediction).opacity(0.3), lineWidth: 6)
                            .frame(width: 50, height: 50)

                        Circle()
                            .trim(from: 0, to: CGFloat(prediction.riskPercentage) / 100)
                            .stroke(riskColor(for: prediction), style: StrokeStyle(lineWidth: 6, lineCap: .round))
                            .frame(width: 50, height: 50)
                            .rotationEffect(.degrees(-90))

                        Text("\(prediction.riskPercentage)%")
                            .font(.system(size: 12, weight: .bold))
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text(prediction.riskLevel)
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(riskColor(for: prediction))

                        Text("Confidence: \(prediction.confidence)")
                            .font(.caption2)
                            .foregroundColor(.secondary)

                        if let topFactor = prediction.topFactors.first {
                            Text(topFactor)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                        }
                    }

                    Spacer()
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(white: 0.15))
            )
        }
    }

    private func riskColor(for prediction: WatchFlareRiskPrediction) -> Color {
        switch prediction.riskColor {
        case "green": return .green
        case "yellow": return .yellow
        case "orange": return .orange
        case "red": return .red
        default: return .gray
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(WatchConnectivityManager.shared)
        .environmentObject(WatchHealthViewModel())
}
