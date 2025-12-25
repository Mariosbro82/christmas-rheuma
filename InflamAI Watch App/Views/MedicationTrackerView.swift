//
//  MedicationTrackerView.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//  Enhanced with beautiful medication tracking UI
//

import SwiftUI
import UserNotifications

struct MedicationTrackerView: View {
    @StateObject private var viewModel = WatchMedicationViewModel()
    @EnvironmentObject var connectivityManager: WatchConnectivityManager

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "pills.circle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.blue)
                        .symbolRenderingMode(.hierarchical)

                    Text("Medications")
                        .font(.headline)
                        .fontWeight(.bold)

                    // Today's Progress
                    if !connectivityManager.medications.isEmpty {
                        TodayMedicationProgress(medications: connectivityManager.medications)
                    }
                }
                .padding(.top, 8)

                // Due Now Section
                if !dueMedications.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "exclamationmark.circle.fill")
                                .foregroundColor(.red)
                            Text("Due Now")
                                .font(.system(size: 14, weight: .semibold))
                        }
                        .padding(.horizontal, 4)

                        ForEach(dueMedications) { medication in
                            MedicationCardView(
                                medication: medication,
                                onTaken: {
                                    markTaken(medication)
                                },
                                onSkip: {
                                    markSkipped(medication)
                                }
                            )
                        }
                    }
                }

                // Upcoming Section
                if !upcomingMedications.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "clock")
                                .foregroundColor(.blue)
                            Text("Upcoming")
                                .font(.system(size: 14, weight: .semibold))
                        }
                        .padding(.horizontal, 4)

                        ForEach(upcomingMedications) { medication in
                            UpcomingMedicationRow(medication: medication)
                        }
                    }
                }

                // Empty State
                if connectivityManager.medications.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "pills")
                            .font(.system(size: 50))
                            .foregroundColor(.secondary)

                        Text("No medications")
                            .font(.subheadline)
                            .foregroundColor(.secondary)

                        Text("Add medications on your iPhone")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.vertical, 30)
                }
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 8)
        }
        .onAppear {
            Task {
                await connectivityManager.requestMedicationSync()
            }
        }
    }

    private var dueMedications: [WatchMedicationSummary] {
        connectivityManager.medications.filter { med in
            med.status == .due || med.status == .overdue || med.isOverdue
        }
    }

    private var upcomingMedications: [WatchMedicationSummary] {
        connectivityManager.medications.filter { med in
            med.status == .upcoming && !med.isOverdue
        }
    }

    private func markTaken(_ medication: WatchMedicationSummary) {
        WKInterfaceDevice.current().play(.success)
        Task {
            await connectivityManager.sendMessage([
                "type": "medication_taken",
                "id": medication.id.uuidString,
                "timestamp": Date().timeIntervalSince1970
            ])
        }
    }

    private func markSkipped(_ medication: WatchMedicationSummary) {
        WKInterfaceDevice.current().play(.click)
        Task {
            await connectivityManager.sendMessage([
                "type": "medication_skipped",
                "id": medication.id.uuidString,
                "timestamp": Date().timeIntervalSince1970
            ])
        }
    }
}

// MARK: - Today Medication Progress

struct TodayMedicationProgress: View {
    let medications: [WatchMedicationSummary]

    private var completedCount: Int {
        medications.filter { $0.status == .due || $0.status == .overdue }.count
    }

    private var totalCount: Int {
        medications.count
    }

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(progressColor)
                .frame(width: 8, height: 8)

            Text("\(totalCount - completedCount) of \(totalCount) taken")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(progressColor.opacity(0.1))
        )
    }

    private var progressColor: Color {
        if completedCount == 0 {
            return .green
        } else if completedCount <= totalCount / 2 {
            return .orange
        } else {
            return .red
        }
    }
}

// MARK: - Medication Card View

struct MedicationCardView: View {
    let medication: WatchMedicationSummary
    let onTaken: () -> Void
    let onSkip: () -> Void

    @State private var showActions = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(medication.name)
                        .font(.system(size: 15, weight: .semibold))

                    Text(medication.dosage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Status Badge
                statusBadge
            }

            // Time Info
            HStack(spacing: 4) {
                Image(systemName: "clock")
                    .font(.system(size: 11))
                Text("Due:")
                Text(medication.scheduledTime, style: .time)
            }
            .font(.caption2)
            .foregroundColor(.secondary)

            Divider()

            // Action Buttons
            HStack(spacing: 8) {
                Button(action: onTaken) {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark")
                        Text("Taken")
                    }
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(
                        LinearGradient(
                            colors: [.green, .green.opacity(0.8)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(10)
                }
                .buttonStyle(.plain)

                Button(action: { showActions = true }) {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 16))
                        .foregroundColor(.secondary)
                        .frame(width: 44, height: 44)
                        .background(Color(white: 0.15))
                        .cornerRadius(10)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(medication.isOverdue ? Color.red.opacity(0.15) : Color(white: 0.15))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(medication.isOverdue ? Color.red.opacity(0.3) : Color.clear, lineWidth: 1)
        )
        .confirmationDialog("Medication Actions", isPresented: $showActions) {
            Button("Skip this dose") {
                onSkip()
            }
            Button("Cancel", role: .cancel) {}
        }
    }

    private var statusBadge: some View {
        Group {
            if medication.isOverdue {
                Text("OVERDUE")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.red)
                    .cornerRadius(8)
            } else {
                Text("DUE")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.orange)
                    .cornerRadius(8)
            }
        }
    }
}

// MARK: - Upcoming Medication Row

struct UpcomingMedicationRow: View {
    let medication: WatchMedicationSummary

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.blue.opacity(0.2))
                    .frame(width: 40, height: 40)

                Image(systemName: "pills")
                    .font(.system(size: 16))
                    .foregroundColor(.blue)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(medication.name)
                    .font(.system(size: 13, weight: .medium))

                Text(medication.dosage)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text(medication.scheduledTime, style: .time)
                    .font(.system(size: 12, weight: .medium))

                Text(medication.frequency)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(white: 0.15))
        )
    }
}

struct MedicationRowView: View {
    let medication: WatchMedicationSummary
    let onTaken: () -> Void
    let onSkip: () -> Void
    let onSnooze: () -> Void

    @State private var showActions = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(medication.name)
                .font(.headline)

            HStack {
                Text("\(medication.dosage) • \(medication.frequency)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
            }

            HStack {
                Text("Due: \(medication.scheduledTime, style: .time)")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                if medication.isOverdue {
                    Text("OVERDUE")
                        .font(.caption2)
                        .foregroundColor(.red)
                        .bold()
                } else if medication.status == .due {
                    Text("DUE NOW")
                        .font(.caption2)
                        .foregroundColor(.orange)
                        .bold()
                }
            }

            HStack(spacing: 8) {
                Button {
                    onTaken()
                    WKInterfaceDevice.current().play(.success)
                } label: {
                    Label("Taken", systemImage: "checkmark.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)

                Button {
                    showActions = true
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(.vertical, 4)
        .confirmationDialog("Medication Actions", isPresented: $showActions) {
            Button("Skip this dose") {
                onSkip()
            }
            Button("Remind in 15 min") {
                onSnooze()
            }
            Button("Cancel", role: .cancel) {}
        }
    }
}

#Preview {
    MedicationTrackerView()
}
