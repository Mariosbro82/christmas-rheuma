//
//  RegionDetailView.swift
//  InflamAI
//
//  Detailed region pain logging with photo capture
//

import SwiftUI
import PhotosUI

struct RegionDetailView: View {
    let region: BodyRegion
    @ObservedObject var viewModel: BodyMapViewModel

    @Environment(\.dismiss) private var dismiss

    // Form state
    @State private var painLevel: Double = 0
    @State private var stiffnessDuration: Double = 0
    @State private var hasSwelling = false
    @State private var hasWarmth = false
    @State private var notes = ""
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var photoData: Data?

    @State private var isSaving = false
    @State private var showingError = false
    @State private var errorMessage = ""

    // FIXED: Track focus state to enable keyboard dismissal
    @FocusState private var isNotesFocused: Bool

    var body: some View {
        NavigationView {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(region.displayName)
                            .font(.title2)
                            .fontWeight(.bold)

                        Text(region.category.rawValue)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                }

                // Pain Level
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Pain Level")
                                .font(.headline)
                            Spacer()
                            Text("\(Int(painLevel))/10")
                                .font(.title3)
                                .fontWeight(.semibold)
                                .foregroundColor(painColor(painLevel))
                        }

                        Slider(value: $painLevel, in: 0...10, step: 1) { editing in
                            if !editing {
                                // Haptic feedback at milestones
                                if [0, 5, 10].contains(Int(painLevel)) {
                                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                                }
                            }
                        }
                        .accessibilityLabel("Pain level")
                        .accessibilityValue("\(Int(painLevel)) out of 10")

                        HStack {
                            Text("😊 None")
                                .font(.caption)
                            Spacer()
                            Text("😣 Severe")
                                .font(.caption)
                        }
                        .foregroundColor(.secondary)
                    }
                } header: {
                    Text("Pain Intensity")
                }

                // Stiffness Duration
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Stiffness Duration")
                                .font(.headline)
                            Spacer()
                            Text("\(Int(stiffnessDuration)) min")
                                .font(.title3)
                                .fontWeight(.semibold)
                        }

                        Slider(value: $stiffnessDuration, in: 0...120, step: 5)
                            .accessibilityLabel("Stiffness duration")
                            .accessibilityValue("\(Int(stiffnessDuration)) minutes")

                        HStack {
                            Text("0 min")
                                .font(.caption)
                            Spacer()
                            Text("2+ hours")
                                .font(.caption)
                        }
                        .foregroundColor(.secondary)
                    }
                } header: {
                    Text("Morning Stiffness")
                }

                // Clinical Signs
                Section {
                    Toggle(isOn: $hasSwelling) {
                        HStack {
                            Image(systemName: "drop.fill")
                                .foregroundColor(.blue)
                            Text("Swelling present")
                        }
                    }
                    .accessibilityLabel("Swelling")

                    Toggle(isOn: $hasWarmth) {
                        HStack {
                            Image(systemName: "flame.fill")
                                .foregroundColor(.orange)
                            Text("Warmth/heat present")
                        }
                    }
                    .accessibilityLabel("Warmth")
                } header: {
                    Text("Clinical Signs")
                }

                // Photo
                Section {
                    PhotosPicker(selection: $selectedPhoto, matching: .images) {
                        HStack {
                            Image(systemName: "camera")
                            Text(photoData == nil ? "Add Photo" : "Change Photo")
                        }
                    }
                    .onChange(of: selectedPhoto) { newItem in
                        Task {
                            if let data = try? await newItem?.loadTransferable(type: Data.self) {
                                photoData = data
                            }
                        }
                    }

                    if photoData != nil {
                        Label("Photo attached", systemImage: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                } header: {
                    Text("Visual Documentation")
                } footer: {
                    Text("Photos are stored locally and never uploaded to cloud services.")
                        .font(.caption)
                }

                // Notes
                Section {
                    TextEditor(text: $notes)
                        .frame(minHeight: 80)
                        .focused($isNotesFocused)
                        .accessibilityLabel("Notes")
                } header: {
                    Text("Notes")
                } footer: {
                    Text("Describe any triggers, activities, or additional symptoms.")
                        .font(.caption)
                }
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("Log Pain")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                // FIXED: Keyboard toolbar with Done button to dismiss keyboard
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        isNotesFocused = false
                    }
                    .fontWeight(.semibold)
                }

                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveLog()
                    }
                    .disabled(isSaving)
                    .fontWeight(.semibold)
                }
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") {}
            } message: {
                Text(errorMessage)
            }
        }
    }

    // MARK: - Helpers

    private func painColor(_ level: Double) -> Color {
        switch level {
        case 0..<3: return .green
        case 3..<6: return .yellow
        case 6..<8: return .orange
        default: return .red
        }
    }

    private func saveLog() {
        isSaving = true

        Task {
            do {
                try await viewModel.logPain(
                    region: region,
                    painLevel: Int16(painLevel),
                    stiffness: Int16(stiffnessDuration),
                    swelling: hasSwelling,
                    warmth: hasWarmth,
                    notes: notes.isEmpty ? nil : notes
                )

                // Haptic success feedback
                UINotificationFeedbackGenerator().notificationOccurred(.success)

                dismiss()
            } catch {
                errorMessage = "Failed to save: \(error.localizedDescription)"
                showingError = true
                isSaving = false
            }
        }
    }
}

// MARK: - Preview

struct RegionDetailView_Previews: PreviewProvider {
    static var previews: some View {
        RegionDetailView(
            region: .l5,
            viewModel: BodyMapViewModel(context: InflamAIPersistenceController.preview.container.viewContext)
        )
    }
}
