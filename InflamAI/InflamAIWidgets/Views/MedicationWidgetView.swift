//
//  MedicationWidgetView.swift
//  InflamAIWidgetExtension
//
//  Medication reminder display views for widgets
//

import SwiftUI
import WidgetKit

// MARK: - Small Widget View

struct MedicationSmallView: View {
    let entry: MedicationEntry

    var body: some View {
        VStack(spacing: 8) {
            if let nextMed = entry.nextMedication {
                // Pill icon
                Image(systemName: "pills.fill")
                    .font(.system(size: 28))
                    .foregroundColor(.blue)

                // Medication name
                Text(nextMed.name)
                    .font(.headline)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)

                // Time
                Text(nextMed.relativeTimeString)
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                // Dosage
                Text(nextMed.dosage)
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                // No upcoming medications
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 32))
                    .foregroundColor(.green)

                Text("All done!")
                    .font(.headline)

                Text("No medications due")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Medium Widget View

struct MedicationMediumView: View {
    let entry: MedicationEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: "pills.fill")
                    .foregroundColor(.blue)
                Text("Medications")
                    .font(.headline)
                Spacer()
            }

            if entry.data.medications.isEmpty {
                HStack {
                    Spacer()
                    VStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.green)
                        Text("No medications scheduled")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                }
            } else {
                // Medication list
                ForEach(entry.data.medications.prefix(3)) { med in
                    MedicationRowView(medication: med)
                }
            }
        }
        .padding()
    }
}

// MARK: - Medication Row

struct MedicationRowView: View {
    let medication: WidgetMedicationData.MedicationReminder

    var body: some View {
        HStack(spacing: 12) {
            // Time indicator
            VStack {
                Text(medication.timeString)
                    .font(.caption)
                    .fontWeight(.medium)
            }
            .frame(width: 50)

            // Medication info
            VStack(alignment: .leading, spacing: 2) {
                Text(medication.name)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(medication.dosage)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Status indicator
            if medication.isOverdue {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else if medication.isDueSoon {
                Text("Due soon")
                    .font(.caption2)
                    .foregroundColor(.orange)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Lock Screen Rectangular View

struct MedicationRectangularView: View {
    let entry: MedicationEntry

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "pills.fill")
                .font(.title2)
                .foregroundColor(.blue)

            if let nextMed = entry.nextMedication {
                VStack(alignment: .leading, spacing: 2) {
                    Text(nextMed.name)
                        .font(.headline)
                        .lineLimit(1)

                    Text(nextMed.relativeTimeString)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    Text("All done")
                        .font(.headline)

                    Text("No meds due")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }
}

// MARK: - Lock Screen Circular View

struct MedicationCircularView: View {
    let entry: MedicationEntry

    var body: some View {
        ZStack {
            AccessoryWidgetBackground()

            if let nextMed = entry.nextMedication {
                VStack(spacing: 0) {
                    Image(systemName: "pills.fill")
                        .font(.system(size: 14))

                    Text(nextMed.timeString)
                        .font(.system(.caption2, design: .rounded).weight(.bold))
                }
            } else {
                Image(systemName: "checkmark.circle.fill")
                    .font(.title2)
            }
        }
    }
}

// MARK: - Lock Screen Inline View

struct MedicationInlineView: View {
    let entry: MedicationEntry

    var body: some View {
        if let nextMed = entry.nextMedication {
            Label("\(nextMed.name) \(nextMed.relativeTimeString)", systemImage: "pills.fill")
        } else {
            Label("No medications due", systemImage: "checkmark.circle")
        }
    }
}

#Preview("Medication Small", as: .systemSmall) {
    MedicationWidget()
} timeline: {
    MedicationEntry(date: Date(), data: .placeholder)
}
