//
//  StreakView.swift
//  InflamAIWidgetExtension
//
//  Logging streak display views for widgets
//

import SwiftUI
import WidgetKit

// MARK: - Small Widget View

struct StreakSmallView: View {
    let entry: StreakEntry

    var body: some View {
        VStack(spacing: 8) {
            // Flame icon
            Image(systemName: entry.data.streakDays > 0 ? "flame.fill" : "flame")
                .font(.system(size: 36))
                .foregroundColor(flameColor)
                .symbolEffect(.pulse, options: .repeating, value: entry.data.streakDays > 7)

            // Streak number
            Text("\(entry.data.streakDays)")
                .font(.system(size: 40, weight: .bold, design: .rounded))
                .foregroundColor(.primary)

            // Label
            Text(entry.data.streakDays == 1 ? "day" : "days")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var flameColor: Color {
        switch entry.data.streakDays {
        case 0: return .gray
        case 1...6: return .orange
        case 7...29: return Color(red: 1.0, green: 0.5, blue: 0.0)
        default: return .red
        }
    }
}

// MARK: - Lock Screen Circular View

struct StreakCircularView: View {
    let entry: StreakEntry

    var body: some View {
        ZStack {
            AccessoryWidgetBackground()

            VStack(spacing: 0) {
                Image(systemName: "flame.fill")
                    .font(.system(size: 16))
                    .foregroundColor(.orange)

                Text("\(entry.data.streakDays)")
                    .font(.system(.title3, design: .rounded).weight(.bold))
            }
        }
    }
}

// MARK: - Lock Screen Rectangular View

struct StreakRectangularView: View {
    let entry: StreakEntry

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "flame.fill")
                .font(.title2)
                .foregroundColor(.orange)

            VStack(alignment: .leading, spacing: 2) {
                Text("Logging Streak")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Text("\(entry.data.streakDays) \(entry.data.streakDays == 1 ? "day" : "days")")
                    .font(.headline)
            }
        }
    }
}

// MARK: - Lock Screen Inline View

struct StreakInlineView: View {
    let entry: StreakEntry

    var body: some View {
        Label("\(entry.data.streakDays) day streak", systemImage: "flame.fill")
    }
}

#Preview("Streak Small", as: .systemSmall) {
    StreakWidget()
} timeline: {
    StreakEntry(date: Date(), data: .placeholder)
}
