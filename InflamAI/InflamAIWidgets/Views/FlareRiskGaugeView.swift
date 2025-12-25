//
//  FlareRiskGaugeView.swift
//  InflamAIWidgetExtension
//
//  Circular gauge view for flare risk display
//

import SwiftUI
import WidgetKit

struct FlareRiskGaugeView: View {
    let percentage: Int
    let riskLevel: WidgetFlareData.RiskLevel
    let size: CGFloat

    var body: some View {
        ZStack {
            // Background circle
            Circle()
                .stroke(Color.gray.opacity(0.2), lineWidth: size * 0.1)

            // Progress arc
            Circle()
                .trim(from: 0, to: CGFloat(percentage) / 100.0)
                .stroke(
                    riskLevel.color,
                    style: StrokeStyle(lineWidth: size * 0.1, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))

            // Center content
            VStack(spacing: 2) {
                Text("\(percentage)")
                    .font(.system(size: size * 0.35, weight: .bold, design: .rounded))
                    .foregroundColor(riskLevel.color)

                Text("%")
                    .font(.system(size: size * 0.12, weight: .medium))
                    .foregroundColor(.secondary)
            }
        }
        .frame(width: size, height: size)
    }
}

// MARK: - Lock Screen Circular Gauge

struct LockScreenFlareGaugeView: View {
    let percentage: Int
    let riskLevel: WidgetFlareData.RiskLevel

    var body: some View {
        Gauge(value: Double(percentage), in: 0...100) {
            Image(systemName: riskLevel.icon)
        } currentValueLabel: {
            Text("\(percentage)")
                .font(.system(.title3, design: .rounded).weight(.bold))
        }
        .gaugeStyle(.accessoryCircular)
        .tint(riskLevel.color)
    }
}

// MARK: - Small Widget View

struct FlareRiskSmallView: View {
    let entry: FlareRiskEntry

    var body: some View {
        VStack(spacing: 8) {
            FlareRiskGaugeView(
                percentage: entry.data.riskPercentage,
                riskLevel: entry.data.riskLevel,
                size: 70
            )

            Text(entry.data.riskLevel.displayName)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(entry.data.riskLevel.color)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Medium Widget View

struct FlareRiskMediumView: View {
    let entry: FlareRiskEntry

    var body: some View {
        HStack(spacing: 16) {
            // Left: Gauge
            FlareRiskGaugeView(
                percentage: entry.data.riskPercentage,
                riskLevel: entry.data.riskLevel,
                size: 80
            )

            // Right: Details
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: entry.data.riskLevel.icon)
                        .foregroundColor(entry.data.riskLevel.color)
                    Text("Pattern Status")
                        .font(.headline)
                }

                Text(entry.data.riskLevel.displayName + " Risk")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                if !entry.data.topFactors.isEmpty {
                    Divider()

                    HStack(spacing: 12) {
                        ForEach(entry.data.topFactors.prefix(2), id: \.self) { factor in
                            Label(factor, systemImage: factorIcon(for: factor))
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }

            Spacer()
        }
        .padding()
    }

    private func factorIcon(for factor: String) -> String {
        switch factor.lowercased() {
        case "weather", "pressure": return "cloud.fill"
        case "sleep": return "bed.double.fill"
        case "stress": return "brain.head.profile"
        case "activity": return "figure.walk"
        case "medication": return "pills.fill"
        default: return "circle.fill"
        }
    }
}

// MARK: - Lock Screen Rectangular View

struct FlareRiskRectangularView: View {
    let entry: FlareRiskEntry

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: entry.data.riskLevel.icon)
                .font(.title2)

            VStack(alignment: .leading) {
                Text("Pattern Status")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Text(entry.data.riskLevel.displayName)
                    .font(.headline)
            }
        }
    }
}

// MARK: - Lock Screen Inline View

struct FlareRiskInlineView: View {
    let entry: FlareRiskEntry

    var body: some View {
        Text("Pattern: \(entry.data.riskLevel.displayName)")
    }
}

#Preview("Small", as: .systemSmall) {
    FlareRiskWidget()
} timeline: {
    FlareRiskEntry(date: Date(), data: .placeholder)
}
