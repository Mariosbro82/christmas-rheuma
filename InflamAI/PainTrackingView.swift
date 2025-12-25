//
//  PainTrackingView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData
import UIKit

struct PainTrackingView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss
    @StateObject private var errorHandler = CoreDataErrorHandler.shared
    
    @State private var isLoading = false
    @State private var showingHistory = false
    @State private var showingSuccessAlert = false
    @State private var selectedBodyRegions: Set<BodyDiagramRegion> = []
    @State private var regionPainLevels: [BodyDiagramRegion: Int] = [:]  // Per-region pain levels
    @State private var notes = ""
    @State private var selectedPainType = "Aching"
    @State private var selectedTrigger = "Unknown"

    let painTypes = ["Aching", "Sharp", "Burning", "Throbbing", "Stabbing", "Cramping", "Tingling", "Dull", "Shooting", "Electric", "Pressure", "Stiff", "Tender", "Radiating", "Pulsing", "Gnawing", "Piercing", "Crushing", "Squeezing", "Tearing", "Searing", "Numbing"]
    let triggers = ["Weather", "Movement", "Rest", "Stress", "Morning", "Evening", "Unknown"]

    // Computed property for completion tracking
    private var completionPercentage: Int {
        var completed = 0
        let totalSteps = 3  // Body regions, pain type, trigger

        if !selectedBodyRegions.isEmpty { completed += 1 }
        if !selectedPainType.isEmpty { completed += 1 }
        if !selectedTrigger.isEmpty { completed += 1 }

        return Int((Double(completed) / Double(totalSteps)) * 100)
    }

    // Can save as long as at least one region is selected (unset pain levels = 0)
    private var canSave: Bool {
        !selectedBodyRegions.isEmpty
    }

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from TrackHubView,
        // which is already wrapped in NavigationView in MainTabView.
        ScrollView {
            VStack(spacing: Spacing.lg) {
                    // Enhanced Progress Indicator
                    VStack(spacing: Spacing.sm) {
                        HStack {
                            Text("Pain Entry Progress")
                                .font(.system(size: Typography.md, weight: .semibold))
                                .foregroundColor(Colors.Gray.g500)
                            Spacer()
                            Text("\(completionPercentage)% Complete")
                                .font(.system(size: Typography.base, weight: .medium))
                                .foregroundColor(Colors.Primary.p500)
                        }

                        ProgressView(value: Double(completionPercentage) / 100.0)
                            .progressViewStyle(LinearProgressViewStyle(tint: Colors.Primary.p500))
                            .scaleEffect(x: 1, y: 1.5, anchor: .center)
                    }
                    .padding(.horizontal, Spacing.lg)
                    .padding(.top, Spacing.xs)

                    // Body Diagram Section - Select regions and set individual pain levels
                    VStack(alignment: .leading, spacing: Spacing.lg) {
                        HStack {
                            Image(systemName: "figure.walk")
                                .font(.title)
                                .fontWeight(.semibold)
                                .foregroundColor(Colors.Primary.p500)
                            Text("Select Pain Locations")
                                .font(.system(size: Typography.xl, weight: .bold))
                                .foregroundColor(Colors.Gray.g900)
                            Spacer()
                            if !selectedBodyRegions.isEmpty {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.title2)
                                    .foregroundColor(Colors.Semantic.success)
                            }
                        }
                        .accessibilityAddTraits(.isHeader)

                        // Instructional text for better UX
                        if selectedBodyRegions.isEmpty {
                            HStack {
                                Image(systemName: "hand.tap")
                                    .font(.callout)
                                    .foregroundColor(Colors.Primary.p500)
                                Text("Tap on the body diagram to select areas where you feel pain")
                                    .font(.system(size: Typography.sm))
                                    .foregroundColor(Colors.Gray.g500)
                                    .multilineTextAlignment(.leading)
                            }
                            .padding(.vertical, Spacing.xs)
                            .padding(.horizontal, Spacing.md)
                            .background(
                                RoundedRectangle(cornerRadius: Radii.lg)
                                    .fill(Colors.Primary.p50)
                            )
                        }

                        BodyDiagramView(selectedRegions: $selectedBodyRegions, regionPainLevels: $regionPainLevels)

                        // Show selected regions summary (pain levels set via body diagram popup)
                        if !selectedBodyRegions.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Text("Selected Regions (\(selectedBodyRegions.count)):")
                                        .font(.headline)
                                        .fontWeight(.semibold)
                                        .foregroundColor(.primary)
                                    Spacer()
                                }

                                // Summary of selected regions with pain levels
                                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                                    ForEach(Array(selectedBodyRegions).sorted { $0.rawValue < $1.rawValue }, id: \.self) { region in
                                        HStack(spacing: 6) {
                                            Circle()
                                                .fill(painLevelColorFor(regionPainLevels[region] ?? 0))
                                                .frame(width: 12, height: 12)
                                            Text(region.rawValue)
                                                .font(.caption)
                                                .lineLimit(1)
                                            Spacer()
                                            Text("\(regionPainLevels[region] ?? 0)")
                                                .font(.caption)
                                                .fontWeight(.bold)
                                                .foregroundColor(painLevelColorFor(regionPainLevels[region] ?? 0))
                                        }
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 6)
                                        .background(
                                            RoundedRectangle(cornerRadius: 8)
                                                .fill(Color(.systemGray6))
                                        )
                                    }
                                }
                            }
                            .padding(.top, 8)
                            .transition(.opacity.combined(with: .scale))
                        }
                    }
                    .padding(Spacing.lg)
                    .background(
                        RoundedRectangle(cornerRadius: Radii.xxl)
                            .fill(Color(.systemBackground))
                    )
                    .dshadow(Shadows.sm)
                    .animation(Animations.easeOut, value: selectedBodyRegions.isEmpty)

                    // Enhanced Pain Characteristics
                    VStack(alignment: .leading, spacing: Spacing.lg) {
                        HStack {
                            Image(systemName: "waveform.path.ecg")
                                .font(.title)
                                .fontWeight(.semibold)
                                .foregroundColor(Colors.Primary.p500)
                            Text("Pain Characteristics")
                                .font(.system(size: Typography.xl, weight: .bold))
                                .foregroundColor(Colors.Gray.g900)
                            Spacer()
                            if !selectedPainType.isEmpty || !selectedTrigger.isEmpty {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.title2)
                                    .foregroundColor(Colors.Semantic.success)
                            }
                        }
                        .accessibilityAddTraits(.isHeader)
                        
                        VStack(alignment: .leading, spacing: 24) {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: "bolt.fill")
                                        .font(.title3)
                                        .foregroundColor(.orange)
                                    Text("Pain Type")
                                        .font(.title2)
                                        .fontWeight(.bold)
                                        .foregroundColor(.primary)
                                    Spacer()
                                }
                                
                                Picker("Pain Type", selection: $selectedPainType) {
                                    ForEach(painTypes, id: \.self) { type in
                                        Text(type)
                                            .font(.callout)
                                            .fontWeight(.semibold)
                                            .tag(type)
                                    }
                                }
                                .pickerStyle(MenuPickerStyle())
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color(.systemGray6))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(Color(.systemGray4), lineWidth: 1)
                                        )
                                )
                                .padding(.vertical, 4)
                            }
                            
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: "exclamationmark.triangle")
                                        .font(.title2)
                                        .fontWeight(.semibold)
                                        .foregroundColor(.orange)
                                    Text("Trigger")
                                        .font(.title2)
                                        .fontWeight(.bold)
                                        .foregroundColor(.primary)
                                    Spacer()
                                }

                                Picker("Trigger", selection: $selectedTrigger) {
                                    ForEach(triggers, id: \.self) { trigger in
                                        Text(trigger)
                                            .font(.callout)
                                            .fontWeight(.semibold)
                                            .tag(trigger)
                                    }
                                }
                                .pickerStyle(MenuPickerStyle())
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color(.systemGray6))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(Color(.systemGray4), lineWidth: 1)
                                        )
                                )
                                .padding(.vertical, 4)
                            }
                        }
                    }
                    .padding(Spacing.lg)
                    .background(
                        RoundedRectangle(cornerRadius: Radii.xxl)
                            .fill(Color(.systemBackground))
                    )
                    .dshadow(Shadows.sm)

                    // Enhanced Notes Section
                    VStack(alignment: .leading, spacing: Spacing.lg) {
                        HStack {
                            Image(systemName: "note.text")
                                .font(.title)
                                .fontWeight(.semibold)
                                .foregroundColor(Colors.Primary.p500)
                            Text("Additional Notes")
                                .font(.system(size: Typography.xl, weight: .bold))
                                .foregroundColor(Colors.Gray.g900)
                            Spacer()
                            if !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.title2)
                                    .foregroundColor(Colors.Semantic.success)
                            }
                        }
                        .accessibilityAddTraits(.isHeader)
                        
                        VStack(alignment: .leading, spacing: 12) {
                            if notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                HStack {
                                    Image(systemName: "pencil")
                                        .font(.callout)
                                        .foregroundColor(.blue)
                                    Text("Add any additional details about your pain experience")
                                        .font(.body)
                                        .fontWeight(.medium)
                                        .foregroundColor(.primary)
                                        .multilineTextAlignment(.leading)
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 16)
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color.blue.opacity(0.1))
                                )
                            }
                            
                            ZStack(alignment: .topLeading) {
                                if notes.isEmpty {
                                    Text("Describe your pain, what makes it better or worse, how it affects your daily activities...")
                                        .font(.callout)
                                        .fontWeight(.medium)
                                        .foregroundColor(.secondary)
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 16)
                                        .allowsHitTesting(false)
                                }
                                
                                TextEditor(text: $notes)
                                    .frame(minHeight: 140)
                                    .padding(16)
                                    .background(Color(.systemGray6))
                                    .cornerRadius(14)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 14)
                                            .stroke(!notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.blue : Color(.systemGray3), lineWidth: 2)
                                    )
                                    .font(.callout)
                                    .fontWeight(.medium)
                                    .lineSpacing(4)
                                    .accessibilityLabel("Additional notes text field")
                            }
                            
                            if !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                HStack {
                                    Image(systemName: "character.cursor.ibeam")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text("\(notes.count) characters")
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .foregroundColor(.primary)
                                    Spacer()
                                }
                                .padding(.top, 4)
                            }
                        }
                    }
                    .padding(Spacing.lg)
                    .background(
                        RoundedRectangle(cornerRadius: Radii.xxl)
                            .fill(Color(.systemBackground))
                    )
                    .dshadow(Shadows.sm)
                    .animation(Animations.easeOut, value: notes.isEmpty)

                    // Enhanced Save Button
                    VStack(spacing: Spacing.md) {
                        // Progress summary before save
                        if !selectedBodyRegions.isEmpty {
                            HStack {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.callout)
                                    .foregroundColor(Colors.Semantic.success)
                                Text("Ready to save: \(selectedBodyRegions.count) region(s) selected")
                                    .font(.system(size: Typography.base, weight: .semibold))
                                    .foregroundColor(Colors.Gray.g900)
                                Spacer()
                            }
                            .padding(.horizontal, Spacing.lg)
                            .padding(.vertical, Spacing.sm)
                            .background(
                                RoundedRectangle(cornerRadius: Radii.lg)
                                    .fill(Colors.Semantic.success.opacity(0.1))
                            )
                            .transition(.opacity.combined(with: .scale))
                        }

                        Button(action: {
                            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                            savePainEntry()
                        }) {
                            HStack(spacing: Spacing.sm) {
                                if isLoading {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        .scaleEffect(0.8)
                                } else {
                                    Image(systemName: canSave ? "checkmark.circle.fill" : "exclamationmark.circle")
                                        .font(.title2)
                                        .fontWeight(.semibold)
                                }
                                Text(isLoading ? "Saving..." : "Save Pain Entry")
                                    .font(.system(size: Typography.lg, weight: .bold))
                            }
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .frame(height: 56)
                            .background(
                                RoundedRectangle(cornerRadius: Radii.xxl)
                                    .fill(canSave ? Colors.Primary.p500 : Colors.Gray.g300)
                            )
                            .dshadow(canSave ? Shadows.md : Shadows.xs)
                            .scaleEffect(canSave ? 1.0 : 0.98)
                        }
                        .disabled(!canSave || isLoading)
                        .accessibilityLabel("Save pain entry button")
                        .accessibilityHint(!canSave ? "Select body regions and set pain levels for each to enable" : "Tap to save your pain entry")
                        .animation(Animations.easeOut, value: canSave)
                        .animation(Animations.easeOut, value: isLoading)

                        // Helpful hint when no regions selected
                        if selectedBodyRegions.isEmpty {
                            HStack {
                                Image(systemName: "info.circle")
                                    .font(.callout)
                                    .foregroundColor(Colors.Semantic.warning)
                                Text("Please select at least one pain location to continue")
                                    .font(.system(size: Typography.base, weight: .medium))
                                    .foregroundColor(Colors.Gray.g900)
                                    .multilineTextAlignment(.center)
                            }
                            .padding(.horizontal, Spacing.lg)
                            .transition(.opacity.combined(with: .scale))
                        }
                    }
                .animation(Animations.easeOut, value: selectedBodyRegions.isEmpty)
            }
            .padding()
        }
        .navigationTitle("Pain Tracking")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button("History") {
                    showingHistory = true
                }
            }

            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Save") {
                    savePainEntry()
                }
                .disabled(isLoading)
            }
        }
        .sheet(isPresented: $showingHistory) {
            PainTrackingHistoryView()
                .environment(\.managedObjectContext, viewContext)
        }
        .coreDataErrorAlert()
        .alert("Success", isPresented: $showingSuccessAlert) {
            Button("OK") {
                resetForm()
            }
        } message: {
            Text("Pain entry has been saved successfully. You can add another entry.")
        }
    }
    
    // Helper function for pain level color (used by per-region pickers)
    private func painLevelColorFor(_ level: Int) -> Color {
        switch level {
        case 0...2:
            return .green
        case 3...5:
            return .yellow
        case 6...7:
            return .orange
        case 8...10:
            return .red
        default:
            return .gray
        }
    }

    // Helper function for pain level description (used by per-region pickers)
    private func painLevelDescriptionFor(_ level: Int) -> String {
        switch level {
        case 0:
            return "No Pain"
        case 1...2:
            return "Mild"
        case 3...4:
            return "Moderate"
        case 5...6:
            return "Moderately Severe"
        case 7...8:
            return "Severe"
        case 9...10:
            return "Very Severe"
        default:
            return "Select pain level"
        }
    }

    private func savePainEntry() {
        // Validate input - only need at least one region selected
        guard !selectedBodyRegions.isEmpty else {
            errorHandler.handle(.invalidData("Please select at least one body region"))
            return
        }

        isLoading = true

        // Create SymptomLog with BodyRegionLogs for per-region pain tracking
        let symptomLogResult: Result<SymptomLog, CoreDataError> = CoreDataOperations.createEntity(
            entityName: "SymptomLog",
            context: viewContext
        )

        switch symptomLogResult {
        case .success(let symptomLog):
            // Set symptom log properties
            symptomLog.id = UUID()
            symptomLog.timestamp = Date()
            symptomLog.source = "pain_tracking"
            symptomLog.painTriggers = selectedTrigger
            symptomLog.painType = selectedPainType
            symptomLog.notes = notes.isEmpty ? nil : notes
            symptomLog.painLocationCount = Int16(selectedBodyRegions.count)

            // Calculate max pain level for pain type indicators
            let maxPainLevel = regionPainLevels.values.max() ?? 0

            // Set pain type indicators based on selected type
            switch selectedPainType {
            case "Aching": symptomLog.painAching = Float(maxPainLevel)
            case "Sharp": symptomLog.painSharp = Float(maxPainLevel)
            case "Burning": symptomLog.painBurning = Float(maxPainLevel)
            default: break
            }

            // Create BodyRegionLog for each selected region with INDIVIDUAL pain levels
            for region in selectedBodyRegions {
                let regionLogResult: Result<BodyRegionLog, CoreDataError> = CoreDataOperations.createEntity(
                    entityName: "BodyRegionLog",
                    context: viewContext
                )

                if case .success(let regionLog) = regionLogResult {
                    regionLog.id = UUID()
                    regionLog.regionID = region.rawValue
                    // Use the individual pain level for this specific region (default 0 if not set)
                    regionLog.painLevel = Int16(regionPainLevels[region] ?? 0)
                    regionLog.symptomLog = symptomLog
                }
            }

            // Save with proper error handling
            CoreDataOperations.safeSave(context: viewContext) { saveResult in
                DispatchQueue.main.async {
                    self.isLoading = false

                    switch saveResult {
                    case .success:
                        self.showingSuccessAlert = true

                    case .failure:
                        // Error already handled by CoreDataOperations
                        break
                    }
                }
            }

        case .failure:
            isLoading = false
            // Error already handled by CoreDataOperations
        }
    }

    private func resetForm() {
        selectedBodyRegions.removeAll()
        regionPainLevels.removeAll()
        notes = ""
        selectedPainType = "Aching"
        selectedTrigger = "Unknown"
    }
}

struct PainTrackingView_Previews: PreviewProvider {
    static var previews: some View {
        PainTrackingView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}