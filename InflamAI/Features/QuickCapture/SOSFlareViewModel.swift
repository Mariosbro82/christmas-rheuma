//
//  SOSFlareViewModel.swift
//  InflamAI-Swift
//
//  Created by Claude Code on 2025-01-25.
//  Quick flare capture ViewModel with Core Data integration
//

import Foundation
import CoreData
import SwiftUI

@MainActor
class SOSFlareViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var severity: Double = 5
    @Published var selectedBodyParts: Set<String> = []
    @Published var notes: String = ""
    @Published var selectedTriggers: Set<String> = []

    // UI State
    @Published var isSaving: Bool = false
    @Published var showSuccessAlert: Bool = false
    @Published var showErrorAlert: Bool = false
    @Published var errorMessage: String? = nil

    // MARK: - Dependencies

    private let context: NSManagedObjectContext

    // MARK: - Computed Properties

    var canSave: Bool {
        return severity > 0 || !selectedBodyParts.isEmpty
    }

    // MARK: - Available Triggers

    let availableTriggers = [
        "Stress", "Weather", "Poor Sleep", "Overexertion",
        "Medication Skip", "Diet", "Unknown"
    ]

    // MARK: - Common Body Parts

    let commonBodyParts = [
        "Neck", "Upper Back", "Lower Back",
        "Left Hip", "Right Hip",
        "Left Knee", "Right Knee",
        "Shoulders", "Hands", "Feet"
    ]

    // MARK: - Initialization

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        self.context = context
    }

    // MARK: - Toggle Actions

    func toggleBodyPart(_ part: String) {
        if selectedBodyParts.contains(part) {
            selectedBodyParts.remove(part)
        } else {
            selectedBodyParts.insert(part)
        }
    }

    func toggleTrigger(_ trigger: String) {
        if selectedTriggers.contains(trigger) {
            selectedTriggers.remove(trigger)
        } else {
            selectedTriggers.insert(trigger)
        }
    }

    // MARK: - Save Logic

    func saveFlare() async {
        guard canSave else { return }

        isSaving = true
        errorMessage = nil

        do {
            // Create new FlareEvent
            let flare = FlareEvent(context: context)
            flare.id = UUID()
            flare.startDate = Date()
            flare.severity = Int16(severity)
            flare.isResolved = false
            flare.notes = notes.isEmpty ? nil : notes

            // Add triggers as binary data (array of strings)
            if !selectedTriggers.isEmpty {
                let triggersArray = Array(selectedTriggers)
                if let triggersData = try? JSONEncoder().encode(triggersArray) {
                    flare.suspectedTriggers = triggersData
                }
            }

            // Add primary regions as binary data (array of strings)
            if !selectedBodyParts.isEmpty {
                let regionsArray = Array(selectedBodyParts)
                if let regionsData = try? JSONEncoder().encode(regionsArray) {
                    flare.primaryRegions = regionsData
                }
            }

            // Create associated SymptomLog for tracking
            let symptomLog = SymptomLog(context: context)
            symptomLog.id = UUID()
            symptomLog.timestamp = Date()
            symptomLog.isFlareEvent = true
            symptomLog.source = "sos_flare"

            // Add body region logs
            if !selectedBodyParts.isEmpty {
                for bodyPart in selectedBodyParts {
                    let regionLog = BodyRegionLog(context: context)
                    regionLog.id = UUID()
                    regionLog.regionID = bodyPart
                    regionLog.painLevel = Int16(severity)
                    regionLog.symptomLog = symptomLog
                }
            }

            // Save context
            try context.save()

            isSaving = false
            showSuccessAlert = true

        } catch {
            isSaving = false
            errorMessage = "Failed to save flare: \(error.localizedDescription)"
            showErrorAlert = true
            print("Error saving SOS flare: \(error)")
        }
    }

    // MARK: - Reset

    func reset() {
        severity = 5
        selectedBodyParts.removeAll()
        notes = ""
        selectedTriggers.removeAll()
        errorMessage = nil
    }
}
