//
//  SOSFlareView.swift
//  InflamAI-Swift
//
//  Created by Claude Code on 2025-01-25.
//  Quick SOS flare capture interface
//

import SwiftUI
import CoreData

struct SOSFlareView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewModel: SOSFlareViewModel

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: SOSFlareViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // SOS Header
                    sosHeader

                    // Severity Selector
                    severitySection

                    // Affected Areas
                    bodyPartsSection

                    // Possible Triggers
                    triggersSection

                    // Quick Notes
                    notesSection

                    // Save Button
                    saveButton
                }
                .padding()
            }
            .navigationTitle("SOS Flare")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .alert("Flare Logged", isPresented: $viewModel.showSuccessAlert) {
                Button("OK") {
                    dismiss()
                }
            } message: {
                Text("Your flare has been recorded. Take care of yourself!")
            }
            .alert("Error", isPresented: $viewModel.showErrorAlert) {
                Button("OK") {}
            } message: {
                if let error = viewModel.errorMessage {
                    Text(error)
                }
            }
        }
    }

    // MARK: - SOS Header

    private var sosHeader: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.red.opacity(0.15))
                    .frame(width: 80, height: 80)

                Image(systemName: "flame.fill")
                    .font(.system(size: 40))
                    .foregroundColor(.red)
            }

            Text("Quick Flare Capture")
                .font(.title2)
                .fontWeight(.bold)

            Text("Record this flare event quickly so you can rest")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.bottom, 8)
    }

    // MARK: - Severity Section

    private var severitySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(severityColor)
                Text("How severe is this flare?")
                    .font(.headline)
            }

            VStack(spacing: 8) {
                HStack {
                    Text(severityLabel)
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(severityColor)
                    Spacer()
                    Text("\(Int(viewModel.severity))/10")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(severityColor)
                }

                Slider(value: $viewModel.severity, in: 1...10, step: 1)
                    .accentColor(severityColor)

                HStack {
                    Text("Mild")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("Severe")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }

    // MARK: - Body Parts Section

    private var bodyPartsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "figure.stand")
                    .foregroundColor(.blue)
                Text("Affected Areas")
                    .font(.headline)

                Spacer()

                Text("\(viewModel.selectedBodyParts.count) selected")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ForEach(viewModel.commonBodyParts, id: \.self) { part in
                    Button(action: { viewModel.toggleBodyPart(part) }) {
                        Text(part)
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(viewModel.selectedBodyParts.contains(part) ? .white : .primary)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(viewModel.selectedBodyParts.contains(part) ? Color.blue : Color(.systemGray6))
                            .cornerRadius(10)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }

    // MARK: - Triggers Section

    private var triggersSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "wind")
                    .foregroundColor(.orange)
                Text("Possible Triggers (Optional)")
                    .font(.headline)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ForEach(viewModel.availableTriggers, id: \.self) { trigger in
                    Button(action: { viewModel.toggleTrigger(trigger) }) {
                        HStack {
                            Image(systemName: viewModel.selectedTriggers.contains(trigger) ? "checkmark.circle.fill" : "circle")
                                .foregroundColor(viewModel.selectedTriggers.contains(trigger) ? .orange : .gray)
                            // CRIT-003 FIX: Apply displayName to convert snake_case to Title Case
                            Text(trigger.displayName)
                                .font(.subheadline)
                            Spacer()
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(viewModel.selectedTriggers.contains(trigger) ? Color.orange.opacity(0.1) : Color(.systemGray6))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }

    // MARK: - Notes Section

    private var notesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "note.text")
                    .foregroundColor(.green)
                Text("Quick Notes (Optional)")
                    .font(.headline)
            }

            TextEditor(text: $viewModel.notes)
                .frame(height: 80)
                .padding(8)
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }

    // MARK: - Save Button

    private var saveButton: some View {
        Button {
            Task {
                await viewModel.saveFlare()
            }
        } label: {
            HStack(spacing: 12) {
                if viewModel.isSaving {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    Text("Saving...")
                        .fontWeight(.semibold)
                } else {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.title3)
                    Text("Log This Flare")
                        .fontWeight(.semibold)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(viewModel.canSave && !viewModel.isSaving ? Color.red : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(!viewModel.canSave || viewModel.isSaving)
    }

    // MARK: - Helper Properties

    private var severityColor: Color {
        switch viewModel.severity {
        case 1...3: return .yellow
        case 4...6: return .orange
        case 7...10: return .red
        default: return .gray
        }
    }

    private var severityLabel: String {
        switch viewModel.severity {
        case 1...3: return "Mild"
        case 4...6: return "Moderate"
        case 7...8: return "Severe"
        case 9...10: return "Critical"
        default: return "Unknown"
        }
    }
}

// MARK: - Preview

#Preview {
    SOSFlareView(context: InflamAIPersistenceController.preview.container.viewContext)
}
