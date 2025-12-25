//
//  QuickLogView.swift
//  InflamAI-Swift
//
//  Created by Claude Code on 2025-01-25.
//  Quick symptom logging from home screen
//

import SwiftUI
import CoreData

struct QuickLogView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewModel: QuickLogViewModel

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: QuickLogViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Quick Status Section
                    quickStatusSection

                    // Pain Level
                    painLevelSection

                    // Morning Stiffness
                    stiffnessSection

                    // Fatigue
                    fatigueSection

                    // Optional: Quick Body Map
                    if viewModel.showBodyMap {
                        bodyMapSection
                    }

                    // Optional: Quick Note
                    quickNoteSection

                    // Save Button
                    saveButton
                }
                .padding()
            }
            .navigationTitle("Quick Log")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .alert("Success", isPresented: $viewModel.showSuccessAlert) {
                Button("OK") {
                    dismiss()
                }
            } message: {
                Text("Symptom log saved successfully!")
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

    // MARK: - Quick Status Section

    private var quickStatusSection: some View {
        VStack(spacing: 16) {
            Text("How are you feeling today?")
                .font(.headline)

            HStack(spacing: 12) {
                QuickStatusButton(
                    emoji: "😊",
                    label: "Great",
                    isSelected: viewModel.quickStatus == .great
                ) {
                    viewModel.quickStatus = .great
                    viewModel.applyQuickStatus(.great)
                }

                QuickStatusButton(
                    emoji: "🙂",
                    label: "Good",
                    isSelected: viewModel.quickStatus == .good
                ) {
                    viewModel.quickStatus = .good
                    viewModel.applyQuickStatus(.good)
                }

                QuickStatusButton(
                    emoji: "😐",
                    label: "Okay",
                    isSelected: viewModel.quickStatus == .okay
                ) {
                    viewModel.quickStatus = .okay
                    viewModel.applyQuickStatus(.okay)
                }

                QuickStatusButton(
                    emoji: "😣",
                    label: "Poor",
                    isSelected: viewModel.quickStatus == .poor
                ) {
                    viewModel.quickStatus = .poor
                    viewModel.applyQuickStatus(.poor)
                }

                QuickStatusButton(
                    emoji: "😰",
                    label: "Bad",
                    isSelected: viewModel.quickStatus == .bad
                ) {
                    viewModel.quickStatus = .bad
                    viewModel.applyQuickStatus(.bad)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    // MARK: - Pain Level Section

    private var painLevelSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "bolt.circle.fill")
                    .foregroundColor(.red)
                Text("Pain Level")
                    .font(.headline)

                Spacer()

                Text("\(Int(viewModel.painLevel))/10")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(painColor(for: viewModel.painLevel))
            }

            Slider(value: $viewModel.painLevel, in: 0...10, step: 1)
                .accentColor(painColor(for: viewModel.painLevel))

            HStack {
                Text("No Pain")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Worst Pain")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    // MARK: - Stiffness Section

    private var stiffnessSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "figure.walk.motion")
                    .foregroundColor(.orange)
                Text("Morning Stiffness")
                    .font(.headline)

                Spacer()

                Text("\(viewModel.morningStiffness) min")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.orange)
            }

            Slider(value: Binding(
                get: { Double(viewModel.morningStiffness) },
                set: { viewModel.morningStiffness = Int($0) }
            ), in: 0...120, step: 5)
                .accentColor(.orange)

            HStack {
                Text("None")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("2+ Hours")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    // MARK: - Fatigue Section

    private var fatigueSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "battery.25")
                    .foregroundColor(.purple)
                Text("Fatigue Level")
                    .font(.headline)

                Spacer()

                Text("\(Int(viewModel.fatigueLevel))/10")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.purple)
            }

            Slider(value: $viewModel.fatigueLevel, in: 0...10, step: 1)
                .accentColor(.purple)

            HStack {
                Text("Energized")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Exhausted")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    // MARK: - Body Map Section

    private var bodyMapSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "figure.stand")
                    .foregroundColor(.blue)
                Text("Pain Locations (Optional)")
                    .font(.headline)

                Spacer()

                Text("\(viewModel.selectedBodyParts.count) selected")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Simple body part selector
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                ForEach(BodyPart.commonParts, id: \.self) { part in
                    Button(action: { viewModel.toggleBodyPart(part) }) {
                        VStack(spacing: 4) {
                            Image(systemName: part.icon)
                                .font(.title3)
                            Text(part.name)
                                .font(.caption2)
                        }
                        .foregroundColor(viewModel.selectedBodyParts.contains(part) ? .white : .primary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(viewModel.selectedBodyParts.contains(part) ? Color.blue : Color(.systemGray6))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                }
            }

            Button {
                viewModel.showBodyMap.toggle()
            } label: {
                HStack {
                    Image(systemName: viewModel.showBodyMap ? "chevron.up" : "chevron.down")
                    Text(viewModel.showBodyMap ? "Hide" : "Show Body Map")
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    // MARK: - Quick Note Section

    private var quickNoteSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "note.text")
                    .foregroundColor(.green)
                Text("Quick Note (Optional)")
                    .font(.headline)
            }

            TextEditor(text: $viewModel.notes)
                .frame(height: 100)
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
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    // MARK: - Save Button

    private var saveButton: some View {
        Button {
            Task {
                await viewModel.saveLog()
            }
        } label: {
            HStack {
                if viewModel.isSaving {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Image(systemName: "checkmark.circle.fill")
                    Text("Save Log")
                }
            }
            .font(.headline)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding()
            .background(viewModel.canSave ? Color.blue : Color.gray)
            .cornerRadius(12)
        }
        .disabled(!viewModel.canSave || viewModel.isSaving)
    }

    // MARK: - Helper Functions

    private func painColor(for level: Double) -> Color {
        switch level {
        case 0...2: return .green
        case 3...4: return .yellow
        case 5...6: return .orange
        case 7...8: return .red
        default: return .purple
        }
    }
}

// MARK: - Supporting Views

struct QuickStatusButton: View {
    let emoji: String
    let label: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Text(emoji)
                    .font(.title)
                Text(label)
                    .font(.caption2)
                    .fontWeight(isSelected ? .semibold : .regular)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(isSelected ? Color.blue.opacity(0.1) : Color(.systemGray6))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Data Models

enum QuickStatus {
    case great, good, okay, poor, bad
}

enum BodyPart: String, CaseIterable {
    case neck, shoulders, upperBack, lowerBack
    case leftHip, rightHip, leftKnee, rightKnee
    case leftAnkle, rightAnkle, hands, feet

    var name: String {
        rawValue.replacingOccurrences(of: "([A-Z])", with: " $1", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
            .capitalized
    }

    var icon: String {
        switch self {
        case .neck: return "figure.stand"
        case .shoulders: return "figure.arms.open"
        case .upperBack, .lowerBack: return "figure.walk"
        case .leftHip, .rightHip: return "figure.walk"
        case .leftKnee, .rightKnee: return "figure.walk.motion"
        case .leftAnkle, .rightAnkle: return "figure.step.training"
        case .hands: return "hand.raised.fill"
        case .feet: return "shoe.fill"
        }
    }

    static var commonParts: [BodyPart] {
        [.neck, .shoulders, .upperBack, .lowerBack, .leftHip, .rightHip,
         .leftKnee, .rightKnee, .hands, .feet]
    }
}

// MARK: - Preview

struct QuickLogView_Previews: PreviewProvider {
    static var previews: some View {
        QuickLogView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
