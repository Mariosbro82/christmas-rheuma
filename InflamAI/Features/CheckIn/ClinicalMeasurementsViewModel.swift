//
//  ClinicalMeasurementsViewModel.swift
//  InflamAI
//
//  ViewModel for Clinical Measurements Entry
//  Handles periodic clinical data that feeds ML features:
//  - BASMI (index 9): Bath AS Metrology Index
//  - Physician Global (index 11): Doctor's assessment
//  - Spinal Mobility (index 16): Self-measured flexibility
//

import Foundation
import CoreData
import Combine
import UIKit

@MainActor
class ClinicalMeasurementsViewModel: ObservableObject {

    // MARK: - Published Properties

    /// BASMI Score (0-10): Bath AS Metrology Index
    /// Measures spinal mobility through 5 clinical tests
    /// ML feature index 9
    @Published var basmiScore: Double = 0.0

    /// Physician Global Assessment (0-10)
    /// Doctor's overall disease activity assessment
    /// ML feature index 11
    @Published var physicianGlobal: Double = 0.0

    /// Spinal Mobility Self-Assessment (0-10, higher = better)
    /// User's perceived spinal flexibility
    /// ML feature index 16
    @Published var spinalMobility: Double = 5.0

    /// Date of last doctor visit (for context)
    @Published var lastDoctorVisit: Date = Date()

    /// Whether user has seen a doctor recently
    @Published var hasRecentDoctorVisit: Bool = false

    // MARK: - BASMI Component Measurements

    /// Tragus-to-wall distance (cm) - cervical rotation
    @Published var tragusToWall: Double = 15.0

    /// Lumbar side flexion (cm)
    @Published var lumbarSideFlexion: Double = 10.0

    /// Lumbar flexion (Schober's modification, cm)
    @Published var lumbarFlexion: Double = 4.0

    /// Intermalleolar distance (cm) - hip abduction
    @Published var intermalleolarDistance: Double = 100.0

    /// Cervical rotation (degrees)
    @Published var cervicalRotation: Double = 70.0

    // MARK: - UI State

    @Published var isSaving = false
    @Published var showingSaveConfirmation = false
    @Published var showingError = false
    @Published var errorMessage = ""
    @Published var showingBASMIGuide = false

    // MARK: - Computed Properties

    /// Calculate BASMI from component measurements
    /// Each component scores 0-2, total 0-10
    var calculatedBASMI: Double {
        let tragusScore = scoreTragusToWall(tragusToWall)
        let sideFlexionScore = scoreLumbarSideFlexion(lumbarSideFlexion)
        let lumbarScore = scoreLumbarFlexion(lumbarFlexion)
        let intermalleolarScore = scoreIntermalleolarDistance(intermalleolarDistance)
        let cervicalScore = scoreCervicalRotation(cervicalRotation)

        return tragusScore + sideFlexionScore + lumbarScore + intermalleolarScore + cervicalScore
    }

    /// Interpretation of BASMI score
    var basmiInterpretation: String {
        switch basmiScore {
        case 0..<2: return "Mild limitation"
        case 2..<4: return "Moderate limitation"
        case 4..<7: return "Significant limitation"
        default: return "Severe limitation"
        }
    }

    /// Color for BASMI severity
    var basmiColor: String {
        switch basmiScore {
        case 0..<2: return "green"
        case 2..<4: return "yellow"
        case 4..<7: return "orange"
        default: return "red"
        }
    }

    // MARK: - Dependencies

    private let context: NSManagedObjectContext

    // MARK: - Initialization

    init(context: NSManagedObjectContext) {
        self.context = context
        loadLastMeasurements()
    }

    // MARK: - BASMI Scoring Functions

    /// Score tragus-to-wall distance (0-2)
    private func scoreTragusToWall(_ cm: Double) -> Double {
        switch cm {
        case ..<15: return 0
        case 15..<30: return 1
        default: return 2
        }
    }

    /// Score lumbar side flexion (0-2)
    private func scoreLumbarSideFlexion(_ cm: Double) -> Double {
        switch cm {
        case 10...: return 0
        case 5..<10: return 1
        default: return 2
        }
    }

    /// Score lumbar flexion - Schober's test (0-2)
    private func scoreLumbarFlexion(_ cm: Double) -> Double {
        switch cm {
        case 4...: return 0
        case 2..<4: return 1
        default: return 2
        }
    }

    /// Score intermalleolar distance (0-2)
    private func scoreIntermalleolarDistance(_ cm: Double) -> Double {
        switch cm {
        case 120...: return 0
        case 70..<120: return 1
        default: return 2
        }
    }

    /// Score cervical rotation (0-2)
    private func scoreCervicalRotation(_ degrees: Double) -> Double {
        switch degrees {
        case 70...: return 0
        case 20..<70: return 1
        default: return 2
        }
    }

    // MARK: - Data Loading

    private func loadLastMeasurements() {
        let fetchRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        fetchRequest.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
        fetchRequest.predicate = NSPredicate(format: "basmi > 0 OR physicianGlobal > 0 OR spinalMobility > 0")
        fetchRequest.fetchLimit = 1

        do {
            if let lastLog = try context.fetch(fetchRequest).first {
                if lastLog.basmi > 0 {
                    basmiScore = Double(lastLog.basmi)
                }
                if lastLog.physicianGlobal > 0 {
                    physicianGlobal = Double(lastLog.physicianGlobal)
                    hasRecentDoctorVisit = true
                }
                if lastLog.spinalMobility > 0 {
                    spinalMobility = Double(lastLog.spinalMobility)
                }
            }
        } catch {
            print("Failed to load last clinical measurements: \(error)")
        }
    }

    // MARK: - Update BASMI from Components

    func updateBASMIFromComponents() {
        basmiScore = calculatedBASMI
    }

    // MARK: - Save Measurements

    func saveMeasurements() async {
        isSaving = true
        defer { isSaving = false }

        do {
            // Find today's SymptomLog or create a new one
            let today = Calendar.current.startOfDay(for: Date())
            let fetchRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            fetchRequest.predicate = NSPredicate(
                format: "timestamp >= %@ AND timestamp < %@",
                today as NSDate,
                Calendar.current.date(byAdding: .day, value: 1, to: today)! as NSDate
            )
            fetchRequest.fetchLimit = 1

            let existingLog = try context.fetch(fetchRequest).first
            let log = existingLog ?? SymptomLog(context: context)

            if existingLog == nil {
                log.id = UUID()
                log.timestamp = Date()
                log.source = "clinical_measurements"
            }

            // Store clinical measurements
            log.basmi = Float(basmiScore)
            log.physicianGlobal = Float(physicianGlobal)
            log.spinalMobility = Float(spinalMobility)

            // If this is a doctor visit update, store the date
            if hasRecentDoctorVisit {
                // Store in notes or a dedicated field if available
                print("Doctor visit recorded: \(lastDoctorVisit)")
            }

            try context.save()

            print("Clinical measurements saved successfully")
            print("   - BASMI: \(basmiScore)")
            print("   - Physician Global: \(physicianGlobal)")
            print("   - Spinal Mobility: \(spinalMobility)")

            UINotificationFeedbackGenerator().notificationOccurred(.success)
            showingSaveConfirmation = true

        } catch {
            print("Failed to save clinical measurements: \(error)")
            errorMessage = "Failed to save: \(error.localizedDescription)"
            showingError = true
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        }
    }
}
