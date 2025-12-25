//
//  BASDAISeverityView.swift
//  InflamAIWidgetExtension
//
//  BASDAI score display views for widgets
//

import SwiftUI
import WidgetKit

// MARK: - Small Widget View

struct BASDAISmallView: View {
    let entry: BASDAIEntry

    var body: some View {
        VStack(spacing: 8) {
            // Score display
            Text(String(format: "%.1f", entry.data.score))
                .font(.system(size: 44, weight: .bold, design: .rounded))
                .foregroundColor(entry.data.severityColor)

            // Severity indicator dots
            HStack(spacing: 4) {
                ForEach(0..<5) { index in
                    Circle()
                        .fill(index < severityDots ? entry.data.severityColor : Color.gray.opacity(0.3))
                        .frame(width: 8, height: 8)
                }
            }

            // Category label
            Text(entry.data.category)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var severityDots: Int {
        switch entry.data.score {
        case 0..<2: return 1
        case 2..<4: return 2
        case 4..<6: return 3
        case 6..<8: return 4
        default: return 5
        }
    }
}

// MARK: - Medium Widget View

struct BASDAIMediumView: View {
    let entry: BASDAIEntry

    var body: some View {
        HStack(spacing: 20) {
            // Left: Score card
            VStack(spacing: 4) {
                Text(String(format: "%.1f", entry.data.score))
                    .font(.system(size: 48, weight: .bold, design: .rounded))
                    .foregroundColor(entry.data.severityColor)

                Text("BASDAI")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(entry.data.severityColor.opacity(0.1))
            .cornerRadius(12)

            // Right: Details
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(entry.data.category)
                        .font(.headline)

                    Spacer()

                    // Trend indicator
                    HStack(spacing: 4) {
                        Image(systemName: entry.data.trend.icon)
                        Text(entry.data.trend == .improving ? "Better" :
                                entry.data.trend == .worsening ? "Worse" : "Stable")
                            .font(.caption)
                    }
                    .foregroundColor(entry.data.trend.color)
                }

                Text("Disease Activity")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Spacer()

                HStack {
                    Image(systemName: "clock")
                        .font(.caption2)
                    Text(entry.data.lastAssessed, style: .relative)
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding()
    }
}

// MARK: - Lock Screen Circular View

struct BASDAICircularView: View {
    let entry: BASDAIEntry

    var body: some View {
        Gauge(value: entry.data.score, in: 0...10) {
            Text("BASDAI")
        } currentValueLabel: {
            Text(String(format: "%.1f", entry.data.score))
                .font(.system(.body, design: .rounded).weight(.bold))
        }
        .gaugeStyle(.accessoryCircular)
        .tint(entry.data.severityColor)
    }
}

// MARK: - Lock Screen Rectangular View

struct BASDAIRectangularView: View {
    let entry: BASDAIEntry

    var body: some View {
        HStack(spacing: 8) {
            // Score with color indicator
            ZStack {
                RoundedRectangle(cornerRadius: 6)
                    .fill(entry.data.severityColor.opacity(0.2))
                    .frame(width: 44, height: 44)

                Text(String(format: "%.1f", entry.data.score))
                    .font(.system(.title3, design: .rounded).weight(.bold))
                    .foregroundColor(entry.data.severityColor)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("BASDAI Score")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                HStack(spacing: 4) {
                    Text(entry.data.category)
                        .font(.headline)

                    Image(systemName: entry.data.trend.icon)
                        .font(.caption)
                        .foregroundColor(entry.data.trend.color)
                }
            }
        }
    }
}

// MARK: - Lock Screen Inline View

struct BASDAIInlineView: View {
    let entry: BASDAIEntry

    var body: some View {
        Text("BASDAI: \(String(format: "%.1f", entry.data.score)) \(entry.data.category)")
    }
}

#Preview("Small BASDAI", as: .systemSmall) {
    BASDAIWidget()
} timeline: {
    BASDAIEntry(date: Date(), data: .placeholder)
}
