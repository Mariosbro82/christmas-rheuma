//
//  HealthKitService.swift
//  InflamAI
//
//  Comprehensive HealthKit integration for 25+ biometric data streams
//  Optimized for AS/rheumatic disease flare prediction ML models
//  Zero cloud uploads, all data stays on-device
//

import Foundation
import HealthKit

/// HealthKit service for collecting comprehensive biometric context data
/// Supports 25+ data types for ML-based flare prediction
@MainActor
final class HealthKitService: ObservableObject {

    // MARK: - Singleton

    static let shared = HealthKitService()

    // MARK: - Properties

    @Published var isAuthorized = false
    @Published var authorizationError: Error?

    private let healthStore = HKHealthStore()

    // MARK: - Comprehensive Data Types (25+ streams)

    private var readTypes: Set<HKObjectType> {
        var types = Set<HKObjectType>()

        // ===== TIER 1: CRITICAL FOR AS/FLARE PREDICTION =====

        // Heart & Cardiovascular (inflammation markers)
        if let type = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .restingHeartRate) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .heartRate) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .vo2Max) { types.insert(type) }

        // Activity & Movement
        if let type = HKObjectType.quantityType(forIdentifier: .stepCount) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .distanceWalkingRunning) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .flightsClimbed) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .appleExerciseTime) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .appleStandTime) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .activeEnergyBurned) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .basalEnergyBurned) { types.insert(type) }

        // Sleep (critical for inflammation)
        if let type = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) { types.insert(type) }

        // ===== TIER 2: HIGH VALUE FOR AS TRACKING =====

        // Mobility Metrics (objective functional assessment)
        if let type = HKObjectType.quantityType(forIdentifier: .walkingSpeed) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .walkingStepLength) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .walkingDoubleSupportPercentage) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .walkingAsymmetryPercentage) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .stairAscentSpeed) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .stairDescentSpeed) { types.insert(type) }

        // Vital Signs (Apple Watch enhanced)
        if let type = HKObjectType.quantityType(forIdentifier: .oxygenSaturation) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .respiratoryRate) { types.insert(type) }

        // Cardio Recovery (Line 7: Cardioerholung) - iOS 16+
        if #available(iOS 16.0, *) {
            if let type = HKObjectType.quantityType(forIdentifier: .heartRateRecoveryOneMinute) { types.insert(type) }
        }

        // Environmental Audio Exposure (Line 20: Umgebungslautstärke)
        if let type = HKObjectType.quantityType(forIdentifier: .environmentalAudioExposure) { types.insert(type) }
        if let type = HKObjectType.categoryType(forIdentifier: .audioExposureEvent) { types.insert(type) }

        // Body Measurements
        if let type = HKObjectType.quantityType(forIdentifier: .bodyMass) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .bodyMassIndex) { types.insert(type) }
        if let type = HKObjectType.quantityType(forIdentifier: .height) { types.insert(type) }

        // ===== TIER 3: ENHANCED METRICS =====

        // Workout Data
        types.insert(HKObjectType.workoutType())

        // Mindfulness (stress indicator)
        if let type = HKObjectType.categoryType(forIdentifier: .mindfulSession) { types.insert(type) }

        // Distance cycling (alternative exercise)
        if let type = HKObjectType.quantityType(forIdentifier: .distanceCycling) { types.insert(type) }

        // Swimming (joint-friendly exercise)
        if let type = HKObjectType.quantityType(forIdentifier: .distanceSwimming) { types.insert(type) }

        // Six Minute Walk Test (clinical functional test)
        if let type = HKObjectType.quantityType(forIdentifier: .sixMinuteWalkTestDistance) { types.insert(type) }

        return types
    }

    // MARK: - Authorization

    /// Check if we already have authorization for critical types (without prompting)
    /// Returns true if we have at least partial authorization
    func checkExistingAuthorization() -> Bool {
        guard HKHealthStore.isHealthDataAvailable() else { return false }

        // Check a few critical types to determine if we have some authorization
        // Note: authorizationStatus only tells us if we've requested, not if granted for READ
        // For read types, we need to try fetching to know if authorized
        let criticalTypes: [HKQuantityTypeIdentifier] = [
            .stepCount,
            .heartRateVariabilitySDNN,
            .restingHeartRate
        ]

        for identifier in criticalTypes {
            if let type = HKQuantityType.quantityType(forIdentifier: identifier) {
                let status = healthStore.authorizationStatus(for: type)
                // For read-only, we can't definitively know, but sharingDenied means we asked
                if status != .notDetermined {
                    // We've at least asked before
                    isAuthorized = true
                    return true
                }
            }
        }

        return false
    }

    /// Request HealthKit authorization for all 25+ data types
    /// This should be called from a user-visible context (onboarding, settings)
    func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw HealthKitError.notAvailable
        }

        // Don't re-request if already authorized
        if isAuthorized {
            print("✅ HealthKit already authorized, skipping request")
            return
        }

        do {
            print("🔄 [HealthKit] Requesting authorization for \(readTypes.count) data types...")
            try await healthStore.requestAuthorization(toShare: [], read: readTypes)
            isAuthorized = true
            authorizationError = nil
            print("✅ [HealthKit] Authorization granted for \(readTypes.count) data types")

            // CRITICAL: Post notification so UnifiedNeuralEngine can invalidate its cache
            // This prevents stale 0% data from being used after authorization succeeds
            NotificationCenter.default.post(name: .healthKitAuthorizationDidChange, object: nil)
            print("🔄 [HealthKit] Posted authorization change notification")
        } catch {
            authorizationError = error
            print("❌ [HealthKit] Authorization failed: \(error.localizedDescription)")
            throw HealthKitError.authorizationFailed(error)
        }
    }

    /// Ensure authorization with retry logic
    /// Returns true if authorized, false if user needs to enable in Settings
    /// - Parameter maxRetries: Number of retry attempts (default 2)
    func ensureAuthorization(maxRetries: Int = 2) async -> Bool {
        // Already authorized?
        if isAuthorized {
            return true
        }

        // Check if we have existing authorization (from previous app session)
        if checkExistingAuthorization() {
            print("✅ [HealthKit] Found existing authorization")
            return true
        }

        // Try to request authorization with retries
        for attempt in 1...maxRetries {
            do {
                try await requestAuthorization()
                return true
            } catch {
                print("⚠️ [HealthKit] Authorization attempt \(attempt)/\(maxRetries) failed: \(error.localizedDescription)")

                if attempt < maxRetries {
                    // Wait before retry (exponential backoff)
                    let delay = UInt64(attempt) * 1_000_000_000 // 1s, 2s, etc.
                    try? await Task.sleep(nanoseconds: delay)
                }
            }
        }

        // All retries failed - user needs to enable in Settings
        print("❌ [HealthKit] All authorization attempts failed. User must enable in Settings.")
        return false
    }

    /// Check if HealthKit is available on this device
    static var isAvailable: Bool {
        HKHealthStore.isHealthDataAvailable()
    }

    // MARK: - Sleep Data (with stages)

    /// Fetch comprehensive sleep data including stages
    func fetchSleepData(for date: Date) async throws -> SleepData {
        guard isAuthorized else {
            throw HealthKitError.notAuthorized
        }

        let sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis)!

        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

        let samples = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKCategorySample], Error>) in
            let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: samples as? [HKCategorySample] ?? [])
                }
            }
            healthStore.execute(query)
        }

        // Calculate sleep stages
        var totalSleepSeconds: TimeInterval = 0
        var inBedSeconds: TimeInterval = 0
        var remSeconds: TimeInterval = 0
        var deepSeconds: TimeInterval = 0
        var coreSeconds: TimeInterval = 0
        var awakeSeconds: TimeInterval = 0

        for sample in samples {
            let duration = sample.endDate.timeIntervalSince(sample.startDate)

            switch sample.value {
            case HKCategoryValueSleepAnalysis.asleepREM.rawValue:
                remSeconds += duration
                totalSleepSeconds += duration
            case HKCategoryValueSleepAnalysis.asleepDeep.rawValue:
                deepSeconds += duration
                totalSleepSeconds += duration
            case HKCategoryValueSleepAnalysis.asleepCore.rawValue:
                coreSeconds += duration
                totalSleepSeconds += duration
            case HKCategoryValueSleepAnalysis.awake.rawValue:
                awakeSeconds += duration
            case HKCategoryValueSleepAnalysis.inBed.rawValue:
                inBedSeconds += duration
            default:
                // Legacy asleep values (iOS 15 and earlier)
                if sample.value == HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue {
                    totalSleepSeconds += duration
                }
            }
        }

        // If no in-bed time recorded, use total sleep + awake as approximation
        if inBedSeconds == 0 {
            inBedSeconds = totalSleepSeconds + awakeSeconds
        }

        let sleepHours = totalSleepSeconds / 3600
        let efficiency = inBedSeconds > 0 ? (totalSleepSeconds / inBedSeconds) * 100 : 0

        return SleepData(
            durationHours: sleepHours,
            efficiency: efficiency,
            quality: calculateSleepQuality(efficiency: efficiency, duration: sleepHours),
            remMinutes: remSeconds / 60,
            deepMinutes: deepSeconds / 60,
            coreMinutes: coreSeconds / 60,
            awakeMinutes: awakeSeconds / 60
        )
    }

    private func calculateSleepQuality(efficiency: Double, duration: Double) -> Int {
        let efficiencyScore = efficiency / 10
        let durationScore = min(duration / 0.8, 10)
        return Int((efficiencyScore + durationScore) / 2)
    }

    // MARK: - HRV (Heart Rate Variability)

    /// Fetch HRV using HKStatisticsQuery for proper averaging across sources
    func fetchHRV(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        return try await fetchDiscreteAverage(type: hrvType, unit: HKUnit.secondUnit(with: .milli), for: date)
    }

    // MARK: - Heart Rate (General)

    /// Fetch average heart rate using HKStatisticsQuery for proper averaging
    func fetchAverageHeartRate(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let hrType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        return try await fetchDiscreteAverage(type: hrType, unit: HKUnit.count().unitDivided(by: .minute()), for: date)
    }

    // MARK: - Resting Heart Rate

    func fetchRestingHeartRate(for date: Date) async throws -> Int {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let hrType = HKQuantityType.quantityType(forIdentifier: .restingHeartRate)!
        let samples = try await fetchQuantitySamples(type: hrType, for: date, limit: 1)

        guard let sample = samples.first else { return 0 }
        return Int(sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute())))
    }

    // MARK: - VO2 Max (Cardio Fitness)

    func fetchVO2Max(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let vo2Type = HKQuantityType.quantityType(forIdentifier: .vo2Max)!
        let samples = try await fetchQuantitySamples(type: vo2Type, for: date, limit: 1)

        guard let sample = samples.first else { return 0 }
        // mL/kg·min
        let unit = HKUnit.literUnit(with: .milli).unitDivided(by: HKUnit.gramUnit(with: .kilo).unitMultiplied(by: .minute()))
        return sample.quantity.doubleValue(for: unit)
    }

    // MARK: - Step Count

    /// Fetch step count using HKStatisticsQuery for proper deduplication
    /// This prevents double-counting from iPhone + Apple Watch + third-party apps
    func fetchStepCount(for date: Date) async throws -> Int {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount)!
        let steps = try await fetchCumulativeSum(type: stepType, unit: .count(), for: date)
        return Int(steps)
    }

    // MARK: - Distance Walking/Running

    /// Fetch walking/running distance using HKStatisticsQuery for proper deduplication
    func fetchDistanceWalking(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let distanceType = HKQuantityType.quantityType(forIdentifier: .distanceWalkingRunning)!
        return try await fetchCumulativeSum(type: distanceType, unit: .meterUnit(with: .kilo), for: date) // kilometers
    }

    // MARK: - Flights Climbed

    /// Fetch flights climbed using HKStatisticsQuery for proper deduplication
    func fetchFlightsClimbed(for date: Date) async throws -> Int {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let flightsType = HKQuantityType.quantityType(forIdentifier: .flightsClimbed)!
        let flights = try await fetchCumulativeSum(type: flightsType, unit: .count(), for: date)
        return Int(flights)
    }

    // MARK: - Exercise Time

    /// Fetch exercise minutes using HKStatisticsQuery for proper deduplication
    func fetchExerciseMinutes(for date: Date) async throws -> Int {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let exerciseType = HKQuantityType.quantityType(forIdentifier: .appleExerciseTime)!
        let minutes = try await fetchCumulativeSum(type: exerciseType, unit: .minute(), for: date)
        return Int(minutes)
    }

    // MARK: - Stand Time

    /// Fetch stand time using HKStatisticsQuery for proper deduplication
    func fetchStandHours(for date: Date) async throws -> Int {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let standType = HKQuantityType.quantityType(forIdentifier: .appleStandTime)!
        let totalMinutes = try await fetchCumulativeSum(type: standType, unit: .minute(), for: date)
        return Int(totalMinutes / 60) // Convert to hours
    }

    // MARK: - Active & Basal Energy

    /// Fetch active energy using HKStatisticsQuery for proper deduplication
    func fetchActiveEnergy(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let energyType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!
        return try await fetchCumulativeSum(type: energyType, unit: .kilocalorie(), for: date)
    }

    /// Fetch basal energy using HKStatisticsQuery for proper deduplication
    func fetchBasalEnergy(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let energyType = HKQuantityType.quantityType(forIdentifier: .basalEnergyBurned)!
        return try await fetchCumulativeSum(type: energyType, unit: .kilocalorie(), for: date)
    }

    // MARK: - Mobility Metrics (iOS 14+)

    /// Fetch walking speed using HKStatisticsQuery for proper averaging
    func fetchWalkingSpeed(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let speedType = HKQuantityType.quantityType(forIdentifier: .walkingSpeed)!
        return try await fetchDiscreteAverage(type: speedType, unit: HKUnit.meter().unitDivided(by: .second()), for: date) // m/s
    }

    /// Fetch walking step length using HKStatisticsQuery for proper averaging
    func fetchWalkingStepLength(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let lengthType = HKQuantityType.quantityType(forIdentifier: .walkingStepLength)!
        return try await fetchDiscreteAverage(type: lengthType, unit: .meterUnit(with: .centi), for: date) // cm
    }

    /// Fetch walking double support percentage using HKStatisticsQuery for proper averaging
    func fetchWalkingDoubleSupportPercentage(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let supportType = HKQuantityType.quantityType(forIdentifier: .walkingDoubleSupportPercentage)!
        let value = try await fetchDiscreteAverage(type: supportType, unit: .percent(), for: date)
        return value * 100 // percentage
    }

    /// Fetch walking asymmetry using HKStatisticsQuery for proper averaging
    func fetchWalkingAsymmetry(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let asymmetryType = HKQuantityType.quantityType(forIdentifier: .walkingAsymmetryPercentage)!
        let value = try await fetchDiscreteAverage(type: asymmetryType, unit: .percent(), for: date)
        return value * 100 // percentage
    }

    // MARK: - Stair Metrics

    /// Fetch stair ascent speed using HKStatisticsQuery for proper averaging
    func fetchStairAscentSpeed(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let ascentType = HKQuantityType.quantityType(forIdentifier: .stairAscentSpeed)!
        return try await fetchDiscreteAverage(type: ascentType, unit: HKUnit.meter().unitDivided(by: .second()), for: date) // m/s
    }

    /// Fetch stair descent speed using HKStatisticsQuery for proper averaging
    func fetchStairDescentSpeed(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let descentType = HKQuantityType.quantityType(forIdentifier: .stairDescentSpeed)!
        return try await fetchDiscreteAverage(type: descentType, unit: HKUnit.meter().unitDivided(by: .second()), for: date) // m/s
    }

    // MARK: - Vital Signs (Apple Watch)

    /// Fetch oxygen saturation using HKStatisticsQuery for proper averaging
    func fetchOxygenSaturation(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let oxygenType = HKQuantityType.quantityType(forIdentifier: .oxygenSaturation)!
        let value = try await fetchDiscreteAverage(type: oxygenType, unit: .percent(), for: date)
        return value * 100 // percentage (e.g., 98%)
    }

    /// Fetch respiratory rate using HKStatisticsQuery for proper averaging
    func fetchRespiratoryRate(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let respType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate)!
        return try await fetchDiscreteAverage(type: respType, unit: HKUnit.count().unitDivided(by: .minute()), for: date) // breaths per minute
    }

    // MARK: - Cardio Recovery (Line 7: Cardioerholung)

    /// Fetch heart rate recovery 1 minute after exercise (Cardioerholung)
    /// HKQuantityTypeIdentifierHeartRateRecoveryOneMinute - available iOS 16+
    /// Returns the drop in BPM after workout completion
    func fetchHeartRateRecovery(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        // heartRateRecoveryOneMinute is iOS 16.0+
        if #available(iOS 16.0, *) {
            let recoveryType = HKQuantityType.quantityType(forIdentifier: .heartRateRecoveryOneMinute)!
            let samples = try await fetchQuantitySamples(type: recoveryType, for: date, limit: 1)

            guard let sample = samples.first else { return 0 }
            // Returns BPM drop (positive number = good recovery)
            return sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
        } else {
            // Not available on iOS < 16
            return 0
        }
    }

    // MARK: - Environmental Audio Exposure (Line 20: Umgebungslautstärke)

    /// Fetch environmental audio exposure level (average dB for the day)
    /// HKQuantityTypeIdentifierEnvironmentalAudioExposure
    func fetchEnvironmentalAudioExposure(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let audioType = HKQuantityType.quantityType(forIdentifier: .environmentalAudioExposure)!
        // Use discrete average for audio levels
        return try await fetchDiscreteAverage(type: audioType, unit: .decibelAWeightedSoundPressureLevel(), for: date)
    }

    /// Fetch count of audio exposure events (significant noise warnings)
    /// HKCategoryTypeIdentifierAudioExposureEvent - triggered when exposure > 80dB for extended time
    func fetchAudioExposureEventCount(for date: Date) async throws -> Int {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let eventType = HKCategoryType.categoryType(forIdentifier: .audioExposureEvent)!

        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

        let events = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKCategorySample], Error>) in
            let query = HKSampleQuery(sampleType: eventType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: samples as? [HKCategorySample] ?? [])
                }
            }
            healthStore.execute(query)
        }

        return events.count
    }

    // MARK: - Body Measurements

    func fetchBodyMass(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let massType = HKQuantityType.quantityType(forIdentifier: .bodyMass)!
        let samples = try await fetchQuantitySamples(type: massType, for: date, limit: 1)

        guard let sample = samples.first else { return 0 }
        return sample.quantity.doubleValue(for: .gramUnit(with: .kilo)) // kg
    }

    func fetchBMI(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let bmiType = HKQuantityType.quantityType(forIdentifier: .bodyMassIndex)!
        let samples = try await fetchQuantitySamples(type: bmiType, for: date, limit: 1)

        guard let sample = samples.first else { return 0 }
        return sample.quantity.doubleValue(for: .count())
    }

    func fetchHeight() async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let heightType = HKQuantityType.quantityType(forIdentifier: .height)!

        let samples = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKQuantitySample], Error>) in
            let query = HKSampleQuery(
                sampleType: heightType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: samples as? [HKQuantitySample] ?? [])
                }
            }
            healthStore.execute(query)
        }

        guard let sample = samples.first else { return 0 }
        return sample.quantity.doubleValue(for: .meterUnit(with: .centi)) // cm
    }

    // MARK: - Mindfulness Sessions

    func fetchMindfulMinutes(for date: Date) async throws -> Int {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let mindfulType = HKCategoryType.categoryType(forIdentifier: .mindfulSession)!

        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

        let samples = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKCategorySample], Error>) in
            let query = HKSampleQuery(sampleType: mindfulType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: samples as? [HKCategorySample] ?? [])
                }
            }
            healthStore.execute(query)
        }

        let totalSeconds = samples.reduce(0.0) { $0 + $1.endDate.timeIntervalSince($1.startDate) }
        return Int(totalSeconds / 60)
    }

    // MARK: - Workouts

    func fetchWorkouts(for date: Date) async throws -> [WorkoutData] {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

        let workouts = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKWorkout], Error>) in
            let query = HKSampleQuery(
                sampleType: HKObjectType.workoutType(),
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: samples as? [HKWorkout] ?? [])
                }
            }
            healthStore.execute(query)
        }

        return workouts.map { workout in
            WorkoutData(
                type: workout.workoutActivityType.name,
                durationMinutes: workout.duration / 60,
                calories: workout.totalEnergyBurned?.doubleValue(for: .kilocalorie()) ?? 0,
                distance: workout.totalDistance?.doubleValue(for: .meterUnit(with: .kilo)) ?? 0,
                startDate: workout.startDate,
                endDate: workout.endDate
            )
        }
    }

    // MARK: - Six Minute Walk Test

    func fetchSixMinuteWalkDistance(for date: Date) async throws -> Double {
        guard isAuthorized else { throw HealthKitError.notAuthorized }

        let walkTestType = HKQuantityType.quantityType(forIdentifier: .sixMinuteWalkTestDistance)!
        let samples = try await fetchQuantitySamples(type: walkTestType, for: date, limit: 1)

        guard let sample = samples.first else { return 0 }
        return sample.quantity.doubleValue(for: .meter()) // meters
    }

    // MARK: - Comprehensive Aggregate Fetch

    /// Fetch ALL biometric data for date - returns comprehensive snapshot
    func fetchAllBiometrics(for date: Date = Date()) async throws -> BiometricSnapshot {
        // Fetch all data in parallel for performance
        async let sleep = fetchSleepData(for: date)
        async let hrv = fetchHRV(for: date)
        async let avgHR = fetchAverageHeartRate(for: date)
        async let restingHR = fetchRestingHeartRate(for: date)
        async let vo2Max = fetchVO2Max(for: date)
        async let steps = fetchStepCount(for: date)
        async let distance = fetchDistanceWalking(for: date)
        async let flights = fetchFlightsClimbed(for: date)
        async let exerciseMin = fetchExerciseMinutes(for: date)
        async let standHrs = fetchStandHours(for: date)
        async let activeEnergy = fetchActiveEnergy(for: date)
        async let basalEnergy = fetchBasalEnergy(for: date)
        async let walkSpeed = fetchWalkingSpeed(for: date)
        async let stepLength = fetchWalkingStepLength(for: date)
        async let doubleSupport = fetchWalkingDoubleSupportPercentage(for: date)
        async let asymmetry = fetchWalkingAsymmetry(for: date)
        async let oxygenSat = fetchOxygenSaturation(for: date)
        async let respRate = fetchRespiratoryRate(for: date)
        async let mindful = fetchMindfulMinutes(for: date)

        return try await BiometricSnapshot(
            date: date,
            sleep: sleep,
            hrvValue: hrv,
            averageHeartRate: avgHR,
            restingHeartRate: restingHR,
            vo2Max: vo2Max,
            stepCount: steps,
            distanceKm: distance,
            flightsClimbed: flights,
            exerciseMinutes: exerciseMin,
            standHours: standHrs,
            activeEnergyKcal: activeEnergy,
            basalEnergyKcal: basalEnergy,
            walkingSpeedMps: walkSpeed,
            walkingStepLengthCm: stepLength,
            walkingDoubleSupportPct: doubleSupport,
            walkingAsymmetryPct: asymmetry,
            oxygenSaturationPct: oxygenSat,
            respiratoryRate: respRate,
            mindfulMinutes: mindful
        )
    }

    // MARK: - Helper Methods

    private func fetchQuantitySamples(type: HKQuantityType, for date: Date, limit: Int = HKObjectQueryNoLimit) async throws -> [HKQuantitySample] {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKQuantitySample], Error>) in
            let query = HKSampleQuery(
                sampleType: type,
                predicate: predicate,
                limit: limit,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: samples as? [HKQuantitySample] ?? [])
                }
            }
            healthStore.execute(query)
        }
    }

    // MARK: - Statistics Query (Properly Deduplicated)

    /// Fetch cumulative sum for a quantity type using HKStatisticsQuery
    /// This properly de-duplicates data from multiple sources (iPhone + Watch + apps)
    /// IMPORTANT: Use this for steps, distance, energy, flights - NOT sample queries
    private func fetchCumulativeSum(type: HKQuantityType, unit: HKUnit, for date: Date) async throws -> Double {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Double, Error>) in
            let query = HKStatisticsQuery(
                quantityType: type,
                quantitySamplePredicate: predicate,
                options: .cumulativeSum
            ) { _, statistics, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else if let sum = statistics?.sumQuantity() {
                    continuation.resume(returning: sum.doubleValue(for: unit))
                } else {
                    continuation.resume(returning: 0)
                }
            }
            healthStore.execute(query)
        }
    }

    /// Fetch discrete average for a quantity type using HKStatisticsQuery
    /// Use this for HRV, heart rate, etc. where we want the average, not sum
    private func fetchDiscreteAverage(type: HKQuantityType, unit: HKUnit, for date: Date) async throws -> Double {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Double, Error>) in
            let query = HKStatisticsQuery(
                quantityType: type,
                quantitySamplePredicate: predicate,
                options: .discreteAverage
            ) { _, statistics, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else if let avg = statistics?.averageQuantity() {
                    continuation.resume(returning: avg.doubleValue(for: unit))
                } else {
                    continuation.resume(returning: 0)
                }
            }
            healthStore.execute(query)
        }
    }
}

// MARK: - Models

struct SleepData {
    let durationHours: Double
    let efficiency: Double // 0-100%
    let quality: Int // 1-10 scale
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

    // Sleep
    let sleep: SleepData

    // Heart & Cardiovascular
    let hrvValue: Double // ms
    let averageHeartRate: Double // bpm
    let restingHeartRate: Int // bpm
    let vo2Max: Double // mL/kg·min

    // Activity
    let stepCount: Int
    let distanceKm: Double
    let flightsClimbed: Int
    let exerciseMinutes: Int
    let standHours: Int
    let activeEnergyKcal: Double
    let basalEnergyKcal: Double

    // Mobility Metrics
    let walkingSpeedMps: Double
    let walkingStepLengthCm: Double
    let walkingDoubleSupportPct: Double
    let walkingAsymmetryPct: Double

    // Vital Signs
    let oxygenSaturationPct: Double
    let respiratoryRate: Double // breaths/min

    // Mindfulness
    let mindfulMinutes: Int

    var totalEnergyKcal: Double {
        activeEnergyKcal + basalEnergyKcal
    }
}

struct WorkoutData {
    let type: String
    let durationMinutes: Double
    let calories: Double
    let distance: Double // km
    let startDate: Date
    let endDate: Date
}

// MARK: - Workout Activity Type Extension

extension HKWorkoutActivityType {
    var name: String {
        switch self {
        case .walking: return "Walking"
        case .running: return "Running"
        case .cycling: return "Cycling"
        case .swimming: return "Swimming"
        case .yoga: return "Yoga"
        case .functionalStrengthTraining: return "Strength Training"
        case .traditionalStrengthTraining: return "Weight Training"
        case .coreTraining: return "Core Training"
        case .flexibility: return "Flexibility"
        case .pilates: return "Pilates"
        case .dance: return "Dance"
        case .elliptical: return "Elliptical"
        case .rowing: return "Rowing"
        case .stairClimbing: return "Stair Climbing"
        case .hiking: return "Hiking"
        case .cooldown: return "Cooldown"
        case .mindAndBody: return "Mind & Body"
        default: return "Other"
        }
    }
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

// MARK: - Notifications

extension Notification.Name {
    /// Posted when HealthKit authorization status changes
    /// Observers should invalidate any cached health data and re-fetch
    static let healthKitAuthorizationDidChange = Notification.Name("healthKitAuthorizationDidChange")
}
