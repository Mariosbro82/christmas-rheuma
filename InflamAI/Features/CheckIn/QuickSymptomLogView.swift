//
//  QuickSymptomLogView.swift
//  InflamAI
//
//  Quick symptom entry for busy days
//  Simplified BASDAI-style questions for rapid logging
//

import SwiftUI
import CoreData

struct QuickSymptomLogView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.managedObjectContext) private var context

    @State private var overallPain: Double = 5
    @State private var morningStiffnessDuration: Double = 30 // minutes
    @State private var fatigueLevel: Double = 5
    @State private var notes: String = ""
    @State private var isSaving = false

    var body: some View {
        NavigationView {
            Form {
                Section {
                    Text("Quick 3-Question Log")
                        .font(.headline)
                        .foregroundColor(.secondary)
                } header: {
                    Text("Takes 30 seconds")
                }

                // Question 1: Overall Pain
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Overall Pain Level")
                                .font(.subheadline)
                                .fontWeight(.semibold)

                            Spacer()

                            Text(String(format: "%.0f/10", overallPain))
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(colorForPain(overallPain))
                        }

                        Slider(value: $overallPain, in: 0...10, step: 1)
                            .tint(colorForPain(overallPain))

                        HStack {
                            Text("No pain")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("Worst pain")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                } header: {
                    Text("1. Pain")
                }

                // Question 2: Morning Stiffness
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Morning Stiffness Duration")
                                .font(.subheadline)
                                .fontWeight(.semibold)

                            Spacer()

                            Text("\(Int(morningStiffnessDuration)) min")
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(.orange)
                        }

                        Slider(value: $morningStiffnessDuration, in: 0...120, step: 5)
                            .tint(.orange)

                        HStack {
                            Text("None")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("2+ hours")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                } header: {
                    Text("2. Stiffness")
                }

                // Question 3: Fatigue
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Fatigue Level")
                                .font(.subheadline)
                                .fontWeight(.semibold)

                            Spacer()

                            Text(String(format: "%.0f/10", fatigueLevel))
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(.purple)
                        }

                        Slider(value: $fatigueLevel, in: 0...10, step: 1)
                            .tint(.purple)

                        HStack {
                            Text("Energized")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("Exhausted")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                } header: {
                    Text("3. Energy")
                }

                // Optional Notes
                Section {
                    TextEditor(text: $notes)
                        .frame(minHeight: 80)
                } header: {
                    Text("Optional Notes")
                } footer: {
                    Text("Any triggers, events, or observations?")
                        .font(.caption)
                }

                // Summary
                Section {
                    VStack(spacing: 12) {
                        HStack {
                            Text("Estimated BASDAI:")
                                .font(.subheadline)
                            Spacer()
                            Text(String(format: "%.1f", estimatedBASDAI))
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(colorForBASDAI(estimatedBASDAI))
                        }

                        Text("Based on quick assessment")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text("Summary")
                }
            }
            .navigationTitle("Quick Log")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        Task {
                            await saveLog()
                        }
                    } label: {
                        if isSaving {
                            ProgressView()
                        } else {
                            Text("Save")
                                .fontWeight(.semibold)
                        }
                    }
                    .disabled(isSaving)
                }
            }
        }
    }

    // MARK: - Computed Properties

    private var estimatedBASDAI: Double {
        // Simplified BASDAI calculation (normally 6 questions)
        // Using 3 questions as proxies
        let q1 = overallPain // Overall discomfort
        let q2 = min(10, morningStiffnessDuration / 12) // Convert minutes to 0-10 scale
        let q3 = fatigueLevel

        // BASDAI = (Q1 + Q2 + Q3 + Q4 + Q5 + (Q6a + Q6b)/2) / 6
        // Simplified: average of 3 questions
        return (q1 + q2 + q3) / 3.0
    }

    // MARK: - Helpers

    private func colorForPain(_ value: Double) -> Color {
        switch value {
        case 0..<3: return .green
        case 3..<5: return .yellow
        case 5..<7: return .orange
        default: return .red
        }
    }

    private func colorForBASDAI(_ value: Double) -> Color {
        switch value {
        case 0..<2: return .green
        case 2..<4: return .yellow
        case 4..<6: return .orange
        default: return .red
        }
    }

    // MARK: - Save

    private func saveLog() async {
        isSaving = true

        await context.perform {
            let log = SymptomLog(context: context)
            log.id = UUID()
            log.timestamp = Date()
            log.source = "quick_log"
            log.basdaiScore = estimatedBASDAI
            log.morningStiffnessMinutes = Int16(morningStiffnessDuration)
            log.fatigueLevel = Int16(fatigueLevel)
            log.notes = notes.isEmpty ? nil : notes

            // Create context snapshot
            let snapshot = ContextSnapshot(context: context)
            snapshot.id = UUID()
            snapshot.timestamp = Date()
            log.contextSnapshot = snapshot

            do {
                try context.save()
                print("✅ Quick symptom log saved")
            } catch {
                print("❌ Failed to save quick log: \(error)")
            }
        }

        isSaving = false

        // Haptic feedback
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)

        // Dismiss after short delay
        try? await Task.sleep(nanoseconds: 300_000_000) // 0.3 seconds
        dismiss()
    }
}

// MARK: - Preview

struct QuickSymptomLogView_Previews: PreviewProvider {
    static var previews: some View {
        QuickSymptomLogView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}
