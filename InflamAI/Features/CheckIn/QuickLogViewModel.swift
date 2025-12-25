//
//  QuickLogViewModel.swift
//  InflamAI-Swift
//
//  Created by Claude Code on 2025-01-25.
//  Quick symptom logging ViewModel with Core Data integration
//

import Foundation
import CoreData
import SwiftUI
import Combine

@MainActor
class QuickLogViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var quickStatus: QuickStatus? = nil
    @Published var painLevel: Double = 0
    @Published var morningStiffness: Int = 0
    @Published var fatigueLevel: Double = 0
    @Published var selectedBodyParts: Set<BodyPart> = []
    @Published var notes: String = ""
    @Published var showBodyMap: Bool = false

    // UI State
    @Published var isSaving: Bool = false
    @Published var showSuccessAlert: Bool = false
    @Published var showErrorAlert: Bool = false
    @Published var errorMessage: String? = nil

    // MARK: - Dependencies

    private let context: NSManagedObjectContext

    // MARK: - Computed Properties

    var canSave: Bool {
        // Require at least one of: quick status, pain level > 0, or stiffness > 0
        return quickStatus != nil || painLevel > 0 || morningStiffness > 0 || fatigueLevel > 0
    }

    // MARK: - Initialization

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        self.context = context
    }

    // MARK: - Quick Status Logic

    func applyQuickStatus(_ status: QuickStatus) {
        switch status {
        case .great:
            painLevel = 0
            morningStiffness = 0
            fatigueLevel = 0
        case .good:
            painLevel = 2
            morningStiffness = 10
            fatigueLevel = 2
        case .okay:
            painLevel = 4
            morningStiffness = 30
            fatigueLevel = 5
        case .poor:
            painLevel = 6
            morningStiffness = 60
            fatigueLevel = 7
        case .bad:
            painLevel = 8
            morningStiffness = 90
            fatigueLevel = 9
        }
    }

    // MARK: - Body Part Selection

    func toggleBodyPart(_ part: BodyPart) {
        if selectedBodyParts.contains(part) {
            selectedBodyParts.remove(part)
        } else {
            selectedBodyParts.insert(part)
        }
    }

    // MARK: - Save Logic

    func saveLog() async {
        guard canSave else { return }

        isSaving = true
        errorMessage = nil

        do {
            // Create new SymptomLog
            let symptomLog = SymptomLog(context: context)
            symptomLog.id = UUID()
            symptomLog.timestamp = Date()
            symptomLog.morningStiffnessMinutes = Int16(morningStiffness)
            symptomLog.fatigueLevel = Int16(fatigueLevel)
            symptomLog.source = "quick_log"

            // Calculate BASDAI score
            let basDAI = calculateBASDAI()
            symptomLog.basdaiScore = basDAI

            #if DEBUG
            let timestampStr = {
                let df = DateFormatter()
                df.dateFormat = "dd.MM.yyyy HH:mm:ss"
                return df.string(from: symptomLog.timestamp ?? Date())
            }()
            print("💾 [QuickLog] Creating SymptomLog: timestamp=\(timestampStr), basdaiScore=\(basDAI), source=quick_log")
            #endif

            // Add notes if present
            if !notes.isEmpty {
                // Store notes in a context snapshot or separate notes field if available
                // For now, we'll skip notes storage as it's not in the main schema
            }

            // Add body region logs if any selected
            if !selectedBodyParts.isEmpty {
                for bodyPart in selectedBodyParts {
                    let regionLog = BodyRegionLog(context: context)
                    regionLog.id = UUID()
                    regionLog.regionID = bodyPart.rawValue
                    regionLog.painLevel = Int16(painLevel)
                    regionLog.symptomLog = symptomLog
                }
            }

            // FIXED: Create and attach ContextSnapshot for ML feature extraction
            await attachContextData(to: symptomLog)

            // Save context
            try context.save()

            #if DEBUG
            print("✅ [QuickLog] SymptomLog SAVED successfully to Core Data")
            #endif

            isSaving = false
            showSuccessAlert = true

        } catch {
            isSaving = false
            errorMessage = "Failed to save symptom log: \(error.localizedDescription)"
            showErrorAlert = true
            print("Error saving quick log: \(error)")
        }
    }

    // MARK: - Helper Methods

    private func calculateBASDAI() -> Double {
        // Simplified BASDAI calculation for quick log
        // Full BASDAI would require all 6 questions, but we can estimate from what we have
        let painScore = painLevel / 10.0
        let stiffnessScore = min(Double(morningStiffness) / 120.0, 1.0)
        let fatigueScore = fatigueLevel / 10.0

        // Average the available metrics (simplified version)
        let basDAI = (painScore + stiffnessScore + fatigueScore) / 3.0 * 10.0
        return basDAI
    }

    // MARK: - Context Data (HealthKit + Weather)

    /// FIXED: Attach environmental and biometric context to symptom log
    private func attachContextData(to log: SymptomLog) async {
        let snapshot = ContextSnapshot(context: context)
        snapshot.id = UUID()
        snapshot.timestamp = Date()

        let today = Date()

        // Fetch weather data (Open-Meteo - FREE, no API key)
        do {
            let weather = try await OpenMeteoService.shared.fetchCurrentWeather()
            snapshot.barometricPressure = weather.pressure
            snapshot.humidity = Int16(weather.humidity)
            snapshot.temperature = weather.temperature
            snapshot.pressureChange12h = weather.pressureChange12h
        } catch {
            print("⚠️ QuickLog: Weather fetch failed - \(error.localizedDescription)")
        }

        // Fetch HealthKit data
        do {
            let biometrics = try await HealthKitService.shared.fetchAllBiometrics(for: today)
            snapshot.restingHeartRate = Int16(biometrics.restingHeartRate)
            snapshot.stepCount = Int32(biometrics.stepCount)
            snapshot.hrvValue = biometrics.hrvValue
            snapshot.sleepEfficiency = biometrics.sleep.efficiency
            log.sleepDurationHours = biometrics.sleep.durationHours
            log.sleepQuality = Int16(biometrics.sleep.quality)
            log.exerciseMinutesToday = Int16(biometrics.exerciseMinutes)
            print("✅ QuickLog: HealthKit data attached")
        } catch {
            print("⚠️ QuickLog: HealthKit fetch failed - \(error.localizedDescription)")
        }

        log.contextSnapshot = snapshot
    }

    // MARK: - Reset

    func reset() {
        quickStatus = nil
        painLevel = 0
        morningStiffness = 0
        fatigueLevel = 0
        selectedBodyParts.removeAll()
        notes = ""
        showBodyMap = false
        errorMessage = nil
    }
}

// MARK: - Preview Support

extension QuickLogViewModel {
    static var preview: QuickLogViewModel {
        let viewModel = QuickLogViewModel(context: InflamAIPersistenceController.preview.container.viewContext)
        viewModel.painLevel = 4
        viewModel.morningStiffness = 30
        viewModel.fatigueLevel = 5
        viewModel.notes = "Sample note for preview"
        return viewModel
    }
}
