//
//  LabResultsView.swift
//  InflamAI
//
//  Lab Results Entry Screen for ML Feature Collection
//  Phase 3: Clinical Inputs
//
//  Enables user to enter lab values from blood tests:
//  - CRP (C-Reactive Protein) - required for ASDAS-CRP calculation
//  - ESR (optional - Erythrocyte Sedimentation Rate)
//
//  ML Features enabled:
//  - asdas_crp (index 7) - when combined with BASDAI
//

import SwiftUI
import CoreData

struct LabResultsView: View {
    @StateObject private var viewModel: LabResultsViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) var reduceMotion

    init(context: NSManagedObjectContext) {
        _viewModel = StateObject(wrappedValue: LabResultsViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    headerSection

                    // CRP Entry Card
                    crpEntryCard

                    // Info Card
                    infoCard

                    // ASDAS Preview (if BASDAI available)
                    if viewModel.canCalculateASDAS {
                        asdasPreviewCard
                    }

                    // Save Button
                    saveButton
                        .padding(.horizontal)
                        .padding(.bottom, 32)
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Lab Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .alert("Saved!", isPresented: $viewModel.showingSaveConfirmation) {
                Button("OK") {
                    dismiss()
                }
            } message: {
                Text("Your lab results have been recorded.")
            }
            .alert("Error", isPresented: $viewModel.showingError) {
                Button("OK") {}
            } message: {
                Text(viewModel.errorMessage)
            }
        }
    }

    // MARK: - Header Section

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "cross.vial.fill")
                .font(.system(size: 48))
                .foregroundColor(.blue)
                .padding(.top, 16)

            Text("Enter Lab Results")
                .font(.title2)
                .fontWeight(.bold)

            Text("From your recent blood test")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.bottom, 8)
    }

    // MARK: - CRP Entry Card

    private var crpEntryCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "drop.fill")
                    .foregroundColor(.red)
                    .font(.title2)
                VStack(alignment: .leading) {
                    Text("CRP (C-Reactive Protein)")
                        .font(.headline)
                    Text("Entzündungswert")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // Date picker for lab test date
            VStack(alignment: .leading, spacing: 8) {
                Text("Test Date")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                DatePicker(
                    "",
                    selection: $viewModel.labDate,
                    in: ...Date(),
                    displayedComponents: .date
                )
                .datePickerStyle(.compact)
                .labelsHidden()
            }

            Divider()

            // CRP Value Input
            VStack(alignment: .leading, spacing: 12) {
                Text("CRP Value")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                HStack(alignment: .bottom, spacing: 8) {
                    TextField("0.0", text: $viewModel.crpInput)
                        .keyboardType(.decimalPad)
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .foregroundColor(crpColor)
                        .multilineTextAlignment(.trailing)
                        .frame(maxWidth: 150)

                    Text("mg/L")
                        .font(.title2)
                        .foregroundColor(.secondary)
                        .padding(.bottom, 8)
                }
                .frame(maxWidth: .infinity)

                // CRP interpretation
                if let crp = viewModel.crpValue {
                    HStack {
                        Circle()
                            .fill(crpColor)
                            .frame(width: 12, height: 12)
                        Text(crpInterpretation(crp))
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }

                // Reference range
                Text("Normal range: < 5 mg/L")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Quick value buttons
            VStack(alignment: .leading, spacing: 8) {
                Text("Quick Select")
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack(spacing: 8) {
                    ForEach([1.0, 5.0, 10.0, 20.0, 50.0], id: \.self) { value in
                        Button {
                            viewModel.crpInput = String(format: "%.1f", value)
                            if !reduceMotion {
                                UIImpactFeedbackGenerator(style: .light).impactOccurred()
                            }
                        } label: {
                            Text(String(format: "%.0f", value))
                                .font(.caption)
                                .fontWeight(.medium)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(
                                    viewModel.crpValue == value ?
                                    Color.blue.opacity(0.2) :
                                    Color(.systemGray5)
                                )
                                .cornerRadius(8)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
        .padding(.horizontal)
    }

    // MARK: - Info Card

    private var infoCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "info.circle.fill")
                    .foregroundColor(.blue)
                Text("What is CRP?")
                    .font(.headline)
            }

            Text("C-Reactive Protein (CRP) is a marker of inflammation in your body. It's measured through a blood test and helps assess disease activity in Ankylosing Spondylitis.")
                .font(.subheadline)
                .foregroundColor(.secondary)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Circle().fill(Color.green).frame(width: 8, height: 8)
                    Text("< 5 mg/L: Normal")
                        .font(.caption)
                }
                HStack {
                    Circle().fill(Color.yellow).frame(width: 8, height: 8)
                    Text("5-10 mg/L: Mild inflammation")
                        .font(.caption)
                }
                HStack {
                    Circle().fill(Color.orange).frame(width: 8, height: 8)
                    Text("10-50 mg/L: Moderate inflammation")
                        .font(.caption)
                }
                HStack {
                    Circle().fill(Color.red).frame(width: 8, height: 8)
                    Text("> 50 mg/L: Significant inflammation")
                        .font(.caption)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
        .padding(.horizontal)
    }

    // MARK: - ASDAS Preview Card

    private var asdasPreviewCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "function")
                    .foregroundColor(.purple)
                    .font(.title2)
                Text("ASDAS-CRP Calculation")
                    .font(.headline)
            }

            if let asdas = viewModel.calculatedASDAS {
                let interpretation = ASDACalculator.interpretation(score: asdas)

                HStack {
                    VStack(alignment: .leading) {
                        Text(String(format: "%.2f", asdas))
                            .font(.system(size: 36, weight: .bold, design: .rounded))
                            .foregroundColor(interpretation.color)

                        Text(interpretation.category)
                            .font(.subheadline)
                            .foregroundColor(interpretation.color)
                    }

                    Spacer()

                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Based on:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("BASDAI: \(String(format: "%.1f", viewModel.latestBASDAI))")
                            .font(.caption)
                        Text("CRP: \(viewModel.crpInput) mg/L")
                            .font(.caption)
                    }
                }
            } else {
                Text("Enter CRP value to calculate ASDAS")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
        .padding(.horizontal)
    }

    // MARK: - Save Button

    private var saveButton: some View {
        Button {
            Task {
                await viewModel.saveLabResults()
            }
        } label: {
            HStack {
                if viewModel.isSaving {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Image(systemName: "checkmark.circle.fill")
                    Text("Save Lab Results")
                        .fontWeight(.semibold)
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(viewModel.isValid ? Color.accentColor : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(16)
        }
        .disabled(viewModel.isSaving || !viewModel.isValid)
        .accessibilityLabel("Save lab results")
    }

    // MARK: - Helpers

    private var crpColor: Color {
        guard let crp = viewModel.crpValue else { return .primary }
        switch crp {
        case ..<5: return .green
        case 5..<10: return .yellow
        case 10..<50: return .orange
        default: return .red
        }
    }

    private func crpInterpretation(_ value: Double) -> String {
        switch value {
        case ..<5: return "Normal"
        case 5..<10: return "Mild elevation"
        case 10..<50: return "Moderate elevation"
        default: return "Significant elevation"
        }
    }
}

// MARK: - Preview

struct LabResultsView_Previews: PreviewProvider {
    static var previews: some View {
        LabResultsView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
