//
//  LabResultsViewModel.swift
//  InflamAI
//
//  ViewModel for Lab Results Entry
//  Handles CRP data entry and ASDAS calculation
//
//  ML Features enabled:
//  - asdas_crp (index 7) - calculated from CRP + BASDAI
//

import Foundation
import CoreData
import Combine
import UIKit

@MainActor
class LabResultsViewModel: ObservableObject {

    // MARK: - Published Properties

    /// CRP value input as string (for TextField binding)
    @Published var crpInput: String = ""

    /// Date of the lab test
    @Published var labDate: Date = Date()

    // MARK: - UI State

    @Published var isSaving = false
    @Published var showingSaveConfirmation = false
    @Published var showingError = false
    @Published var errorMessage = ""

    // MARK: - Cached Data

    @Published var latestBASDAI: Double = 0.0
    @Published var canCalculateASDAS: Bool = false

    // MARK: - Computed Properties

    /// Parsed CRP value
    var crpValue: Double? {
        // Handle both comma and period as decimal separator
        let normalized = crpInput.replacingOccurrences(of: ",", with: ".")
        return Double(normalized)
    }

    /// Check if input is valid
    var isValid: Bool {
        guard let crp = crpValue else { return false }
        return crp >= 0 && crp <= 500  // Reasonable range for CRP
    }

    /// Calculate ASDAS if we have both CRP and BASDAI
    var calculatedASDAS: Double? {
        guard let crp = crpValue, crp > 0, latestBASDAI > 0 else { return nil }

        // Use ASDACalculator for accurate calculation
        // For simplified calculation when we don't have individual BASDAI components:
        // ASDAS-CRP (simplified) = 0.121 * BASDAI + 0.579 * ln(CRP + 1)
        return ASDACalculator.calculate(
            backPain: latestBASDAI,  // Using BASDAI as proxy for back pain
            duration: latestBASDAI,  // Proxy for morning stiffness duration
            patientGlobal: latestBASDAI,  // Proxy
            peripheralPain: latestBASDAI,  // Proxy
            crp: crp
        )
    }

    // MARK: - Dependencies

    private let context: NSManagedObjectContext

    // MARK: - Initialization

    init(context: NSManagedObjectContext) {
        self.context = context
        loadLatestBASDAI()
    }

    // MARK: - Load Latest BASDAI

    private func loadLatestBASDAI() {
        let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        request.predicate = NSPredicate(format: "basdaiScore > 0")
        request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
        request.fetchLimit = 1

        do {
            if let latestLog = try context.fetch(request).first {
                latestBASDAI = latestLog.basdaiScore
                canCalculateASDAS = latestBASDAI > 0
                print("📊 Found latest BASDAI: \(latestBASDAI)")
            } else {
                print("⚠️ No BASDAI scores found - ASDAS calculation will not be available")
            }
        } catch {
            print("❌ Failed to fetch latest BASDAI: \(error)")
        }
    }

    // MARK: - Save Lab Results

    func saveLabResults() async {
        guard isValid else {
            errorMessage = "Please enter a valid CRP value"
            showingError = true
            return
        }

        isSaving = true
        defer { isSaving = false }

        do {
            // Create new SymptomLog with lab results
            let log = SymptomLog(context: context)
            log.id = UUID()
            log.timestamp = labDate
            log.source = "lab_results"

            // Store CRP value
            if let crp = crpValue {
                log.crpValue = crp
                log.crpLevel = Float(crp)
                print("💉 CRP value stored: \(crp) mg/L")
            }

            // If we have BASDAI, copy it and calculate ASDAS
            if latestBASDAI > 0 {
                log.basdaiScore = latestBASDAI

                // Calculate ASDAS-CRP
                if let asdas = calculatedASDAS {
                    log.asdasScore = asdas
                    print("📊 ASDAS-CRP calculated: \(asdas)")
                }
            }

            // Create context snapshot (weather/health data)
            await attachContextData(to: log)

            // Auto-populate ML properties
            log.populateMLProperties(context: context)

            // Save
            try context.save()

            print("✅ Lab results saved successfully")
            print("   - CRP: \(crpInput) mg/L")
            print("   - Date: \(labDate)")
            if let asdas = calculatedASDAS {
                print("   - ASDAS-CRP: \(String(format: "%.2f", asdas))")
            }

            // Success feedback
            UINotificationFeedbackGenerator().notificationOccurred(.success)

            showingSaveConfirmation = true

        } catch {
            print("❌ Failed to save lab results: \(error)")
            errorMessage = "Failed to save: \(error.localizedDescription)"
            showingError = true
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        }
    }

    // MARK: - Context Data

    private func attachContextData(to log: SymptomLog) async {
        let snapshot = ContextSnapshot(context: context)
        snapshot.id = UUID()
        snapshot.timestamp = labDate

        // Fetch weather data
        do {
            let weather = try await OpenMeteoService.shared.fetchCurrentWeather()
            snapshot.barometricPressure = weather.pressure
            snapshot.humidity = Int16(weather.humidity)
            snapshot.temperature = weather.temperature
            snapshot.pressureChange12h = weather.pressureChange12h
        } catch {
            print("⚠️ Weather fetch failed: \(error.localizedDescription)")
        }

        // Fetch HealthKit data for the lab date
        do {
            let biometrics = try await HealthKitService.shared.fetchAllBiometrics(for: labDate)
            snapshot.restingHeartRate = Int16(biometrics.restingHeartRate)
            snapshot.stepCount = Int32(biometrics.stepCount)
            snapshot.hrvValue = biometrics.hrvValue
            snapshot.sleepEfficiency = biometrics.sleep.efficiency
        } catch {
            print("⚠️ HealthKit fetch failed: \(error.localizedDescription)")
        }

        log.contextSnapshot = snapshot
    }
}
