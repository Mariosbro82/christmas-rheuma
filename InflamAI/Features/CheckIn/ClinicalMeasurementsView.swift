//
//  ClinicalMeasurementsView.swift
//  InflamAI
//
//  Clinical Measurements Entry Screen
//  For periodic input of clinical data that feeds ML prediction:
//  - BASMI (Bath AS Metrology Index)
//  - Physician Global Assessment
//  - Spinal Mobility Self-Assessment
//

import SwiftUI
import CoreData

struct ClinicalMeasurementsView: View {
    @StateObject private var viewModel: ClinicalMeasurementsViewModel
    @Environment(\.dismiss) private var dismiss

    init(context: NSManagedObjectContext) {
        _viewModel = StateObject(wrappedValue: ClinicalMeasurementsViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            Form {
                // MARK: - Spinal Mobility Self-Assessment
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "figure.walk")
                                .foregroundColor(.blue)
                            Text("Spinal Mobility")
                                .font(.headline)
                        }

                        Text("Rate your current spinal flexibility (0 = very stiff, 10 = fully flexible)")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        HStack {
                            Text("Stiff")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: $viewModel.spinalMobility, in: 0...10, step: 0.5)
                                .accessibilityLabel("Spinal mobility rating")
                            Text("Flexible")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Text(String(format: "%.1f / 10", viewModel.spinalMobility))
                            .font(.title2)
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                    }
                    .padding(.vertical, 8)
                } header: {
                    Text("Self-Assessment")
                } footer: {
                    Text("Update this weekly based on how your spine feels during daily activities like bending and turning.")
                }

                // MARK: - BASMI Score
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "ruler")
                                .foregroundColor(.orange)
                            Text("BASMI Score")
                                .font(.headline)
                            Spacer()
                            Button {
                                viewModel.showingBASMIGuide = true
                            } label: {
                                Image(systemName: "info.circle")
                            }
                        }

                        Text("Bath AS Metrology Index measures spinal mobility through 5 tests")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        // Direct score entry
                        HStack {
                            Text("Score")
                            Spacer()
                            Slider(value: $viewModel.basmiScore, in: 0...10, step: 0.5)
                                .frame(width: 200)
                                .accessibilityLabel("BASMI score")
                            Text(String(format: "%.1f", viewModel.basmiScore))
                                .frame(width: 40)
                        }

                        // Interpretation
                        HStack {
                            Circle()
                                .fill(colorForBASMI(viewModel.basmiScore))
                                .frame(width: 12, height: 12)
                            Text(viewModel.basmiInterpretation)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 8)

                    // Expandable component measurements
                    DisclosureGroup("Enter Individual Measurements") {
                        VStack(spacing: 16) {
                            MeasurementRow(
                                title: "Tragus-to-Wall",
                                subtitle: "Stand with back to wall, measure distance from ear to wall",
                                value: $viewModel.tragusToWall,
                                unit: "cm",
                                range: 0...50
                            )

                            MeasurementRow(
                                title: "Lumbar Side Flexion",
                                subtitle: "Slide hand down thigh, measure how far fingers reach",
                                value: $viewModel.lumbarSideFlexion,
                                unit: "cm",
                                range: 0...25
                            )

                            MeasurementRow(
                                title: "Lumbar Flexion (Schober's)",
                                subtitle: "Increase in distance between marks when bending forward",
                                value: $viewModel.lumbarFlexion,
                                unit: "cm",
                                range: 0...8
                            )

                            MeasurementRow(
                                title: "Intermalleolar Distance",
                                subtitle: "Lie down, spread legs as wide as possible",
                                value: $viewModel.intermalleolarDistance,
                                unit: "cm",
                                range: 30...150
                            )

                            MeasurementRow(
                                title: "Cervical Rotation",
                                subtitle: "Turn head left/right, measure angle from center",
                                value: $viewModel.cervicalRotation,
                                unit: "degrees",
                                range: 0...90
                            )

                            Button {
                                viewModel.updateBASMIFromComponents()
                            } label: {
                                HStack {
                                    Image(systemName: "function")
                                    Text("Calculate BASMI")
                                }
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(Colors.Primary.p500)
                        }
                        .padding(.vertical, 8)
                    }
                } header: {
                    Text("Clinical Measurement")
                } footer: {
                    Text("Enter your BASMI score from your last rheumatology appointment, or use individual measurements to calculate it.")
                }

                // MARK: - Physician Global Assessment
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "stethoscope")
                                .foregroundColor(.green)
                            Text("Physician Global Assessment")
                                .font(.headline)
                        }

                        Toggle("I have a recent doctor assessment", isOn: $viewModel.hasRecentDoctorVisit)

                        if viewModel.hasRecentDoctorVisit {
                            DatePicker(
                                "Visit Date",
                                selection: $viewModel.lastDoctorVisit,
                                in: ...Date(),
                                displayedComponents: .date
                            )

                            VStack(alignment: .leading, spacing: 8) {
                                Text("Doctor's Disease Activity Rating")
                                    .font(.subheadline)

                                HStack {
                                    Text("None")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Slider(value: $viewModel.physicianGlobal, in: 0...10, step: 0.5)
                                        .accessibilityLabel("Physician global assessment")
                                    Text("Severe")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }

                                Text(String(format: "%.1f / 10", viewModel.physicianGlobal))
                                    .font(.title3)
                                    .fontWeight(.semibold)
                                    .frame(maxWidth: .infinity)
                            }
                        }
                    }
                    .padding(.vertical, 8)
                } header: {
                    Text("Doctor's Assessment")
                } footer: {
                    Text("If your rheumatologist provided a global disease activity score, enter it here. This helps improve prediction accuracy.")
                }

                // MARK: - Save Button
                Section {
                    Button {
                        Task {
                            await viewModel.saveMeasurements()
                        }
                    } label: {
                        HStack {
                            if viewModel.isSaving {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle())
                            } else {
                                Image(systemName: "checkmark.circle.fill")
                            }
                            Text("Save Measurements")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .disabled(viewModel.isSaving)
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                }
            }
            .navigationTitle("Clinical Measurements")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .alert("Measurements Saved", isPresented: $viewModel.showingSaveConfirmation) {
                Button("OK") {
                    dismiss()
                }
            } message: {
                Text("Your clinical measurements have been recorded and will be used to improve prediction accuracy.")
            }
            .alert("Error", isPresented: $viewModel.showingError) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(viewModel.errorMessage)
            }
            .sheet(isPresented: $viewModel.showingBASMIGuide) {
                BASMIGuideView()
            }
        }
    }

    private func colorForBASMI(_ score: Double) -> Color {
        switch score {
        case 0..<2: return .green
        case 2..<4: return .yellow
        case 4..<7: return .orange
        default: return .red
        }
    }
}

// MARK: - Measurement Row Component

struct MeasurementRow: View {
    let title: String
    let subtitle: String
    @Binding var value: Double
    let unit: String
    let range: ClosedRange<Double>

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
            Text(subtitle)
                .font(.caption2)
                .foregroundColor(.secondary)
            HStack {
                Slider(value: $value, in: range)
                    .accessibilityLabel(title)
                Text("\(Int(value)) \(unit)")
                    .frame(width: 70, alignment: .trailing)
                    .font(.caption)
                    .monospacedDigit()
            }
        }
    }
}

// MARK: - BASMI Guide View

struct BASMIGuideView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("BASMI Measurement Guide")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("The Bath AS Metrology Index measures spinal mobility through 5 standardized tests. Each test scores 0-2 points (total 0-10).")
                        .foregroundColor(.secondary)

                    ForEach(basmiTests) { test in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: test.icon)
                                    .foregroundColor(.blue)
                                    .frame(width: 24)
                                Text(test.name)
                                    .font(.headline)
                            }
                            Text(test.description)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Text("Normal: \(test.normalRange)")
                                .font(.caption)
                                .foregroundColor(.green)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }

                    Text("For accurate measurements, ask your physiotherapist or rheumatologist to help you perform these tests.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .padding(.top)
                }
                .padding()
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }

    private var basmiTests: [BASMITest] {
        [
            BASMITest(
                name: "Tragus-to-Wall",
                icon: "figure.stand",
                description: "Stand with heels and back against a wall. Measure the distance from your tragus (ear) to the wall.",
                normalRange: "< 15 cm"
            ),
            BASMITest(
                name: "Lumbar Side Flexion",
                icon: "arrow.left.and.right",
                description: "Stand straight, slide your hand down your thigh as far as possible. Measure the distance your fingers travel.",
                normalRange: "> 10 cm"
            ),
            BASMITest(
                name: "Modified Schober's",
                icon: "arrow.up.and.down",
                description: "Mark 10cm above the dimples of Venus. Bend forward and measure the increase in distance.",
                normalRange: "> 4 cm increase"
            ),
            BASMITest(
                name: "Intermalleolar Distance",
                icon: "figure.walk",
                description: "Lie on your back and spread your legs as wide as possible. Measure the distance between your ankles.",
                normalRange: "> 120 cm"
            ),
            BASMITest(
                name: "Cervical Rotation",
                icon: "arrow.triangle.2.circlepath",
                description: "Sit looking forward. Rotate your head to each side and measure the angle from center.",
                normalRange: "> 70 degrees"
            )
        ]
    }
}

struct BASMITest: Identifiable {
    let id = UUID()
    let name: String
    let icon: String
    let description: String
    let normalRange: String
}

#Preview {
    ClinicalMeasurementsView(context: InflamAIPersistenceController.preview.container.viewContext)
}
