//
//  HealthKitServiceStub.swift
//  InflamAI
//
//  Lightweight stub for HealthKitService to allow app to compile
//  The full HealthKitService is excluded due to type conflicts
//  This stub provides the interface needed by QuickLogViewModel, DailyCheckInViewModel, etc.
//

import Foundation
import HealthKit

// MARK: - HealthKit Service (Stub)

@MainActor
final class HealthKitService: ObservableObject {

    // MARK: - Singleton

    static let shared = HealthKitService()

    // MARK: - Properties

    @Published var isAuthorized = false
    @Published var authorizationError: Error?

    private let healthStore = HKHealthStore()

    private init() {
        // Check if already authorized
        if HKHealthStore.isHealthDataAvailable() {
            // Start with false, will update after authorization check
            isAuthorized = false
        }
    }

    // MARK: - Authorization

    func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw HealthKitError.notAvailable
        }

        // Define minimal read types for basic functionality
        var readTypes = Set<HKObjectType>()
        if let hrvType = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN) {
            readTypes.insert(hrvType)
        }
        if let restingHRType = HKObjectType.quantityType(forIdentifier: .restingHeartRate) {
            readTypes.insert(restingHRType)
        }
        if let stepsType = HKObjectType.quantityType(forIdentifier: .stepCount) {
            readTypes.insert(stepsType)
        }
        if let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) {
            readTypes.insert(sleepType)
        }

        do {
            try await healthStore.requestAuthorization(toShare: [], read: readTypes)
            isAuthorized = true
        } catch {
            authorizationError = error
            throw HealthKitError.authorizationFailed(error)
        }
    }

    // MARK: - Fetch All Biometrics (Stub)

    func fetchAllBiometrics(for date: Date = Date()) async throws -> BiometricSnapshot {
        // Return empty/default biometric data
        // Full implementation is in the excluded HealthKitService.swift
        return BiometricSnapshot(
            date: date,
            sleep: SleepData(
                durationHours: 0,
                efficiency: 0,
                quality: 0,
                remMinutes: 0,
                deepMinutes: 0,
                coreMinutes: 0,
                awakeMinutes: 0
            ),
            hrvValue: 0,
            averageHeartRate: 0,
            restingHeartRate: 0,
            vo2Max: 0,
            stepCount: 0,
            distanceKm: 0,
            flightsClimbed: 0,
            exerciseMinutes: 0,
            standHours: 0,
            activeEnergyKcal: 0,
            basalEnergyKcal: 0,
            walkingSpeedMps: 0,
            walkingStepLengthCm: 0,
            walkingDoubleSupportPct: 0,
            walkingAsymmetryPct: 0,
            oxygenSaturationPct: 0,
            respiratoryRate: 0,
            mindfulMinutes: 0
        )
    }
}

// MARK: - Models

struct SleepData {
    let durationHours: Double
    let efficiency: Double
    let quality: Int
    let remMinutes: Double
    let deepMinutes: Double
    let coreMinutes: Double
    let awakeMinutes: Double

    var totalSleepMinutes: Double {
        remMinutes + deepMinutes + coreMinutes
    }
}

struct BiometricSnapshot {
    let date: Date
    let sleep: SleepData
    let hrvValue: Double
    let averageHeartRate: Double
    let restingHeartRate: Int
    let vo2Max: Double
    let stepCount: Int
    let distanceKm: Double
    let flightsClimbed: Int
    let exerciseMinutes: Int
    let standHours: Int
    let activeEnergyKcal: Double
    let basalEnergyKcal: Double
    let walkingSpeedMps: Double
    let walkingStepLengthCm: Double
    let walkingDoubleSupportPct: Double
    let walkingAsymmetryPct: Double
    let oxygenSaturationPct: Double
    let respiratoryRate: Double
    let mindfulMinutes: Int

    var totalEnergyKcal: Double {
        activeEnergyKcal + basalEnergyKcal
    }
}

struct WorkoutData {
    let type: String
    let durationMinutes: Double
    let calories: Double
    let distance: Double
    let startDate: Date
    let endDate: Date
}

// MARK: - Errors

enum HealthKitError: LocalizedError {
    case notAvailable
    case notAuthorized
    case authorizationFailed(Error)

    var errorDescription: String? {
        switch self {
        case .notAvailable:
            return "HealthKit is not available on this device"
        case .notAuthorized:
            return "HealthKit access not authorized. Please enable in Settings."
        case .authorizationFailed(let error):
            return "HealthKit authorization failed: \(error.localizedDescription)"
        }
    }
}
