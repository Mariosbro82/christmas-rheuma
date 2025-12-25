//
//  BiometricsView.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//

import SwiftUI
import HealthKit

struct BiometricsView: View {
    @EnvironmentObject var viewModel: WatchHealthViewModel

    var body: some View {
        List {
            Section("Current") {
                MetricRow(
                    icon: "heart.fill",
                    color: .red,
                    title: "Heart Rate",
                    value: viewModel.heartRate.map { "\(Int($0)) bpm" } ?? "--"
                )

                MetricRow(
                    icon: "waveform.path.ecg",
                    color: .blue,
                    title: "HRV",
                    value: viewModel.hrv.map { "\(Int($0)) ms" } ?? "--"
                )
            }

            Section("Today") {
                MetricRow(
                    icon: "figure.walk",
                    color: .green,
                    title: "Steps",
                    value: viewModel.steps.map { "\(Int($0))" } ?? "--"
                )

                MetricRow(
                    icon: "flame.fill",
                    color: .orange,
                    title: "Active Energy",
                    value: viewModel.activeEnergy.map { "\(Int($0)) cal" } ?? "--"
                )
            }

            Button {
                Task {
                    await viewModel.refresh()
                }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
        }
        .navigationTitle("Biometrics")
        .onAppear {
            Task {
                await viewModel.startMonitoring()
            }
        }
    }
}

struct MetricRow: View {
    let icon: String
    let color: Color
    let title: String
    let value: String

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 24)

            VStack(alignment: .leading) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.body)
                    .bold()
            }

            Spacer()
        }
    }
}

#Preview {
    BiometricsView()
        .environmentObject(WatchHealthViewModel())
}
