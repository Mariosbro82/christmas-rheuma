//
//  AppleWatchManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import HealthKit
import WatchConnectivity
import Combine
import CoreLocation

// MARK: - Apple Watch Manager
@MainActor
class AppleWatchManager: NSObject, ObservableObject {
    
    // MARK: - Published Properties
    @Published var isWatchConnected = false
    @Published var isHealthKitAuthorized = false
    @Published var currentHeartRate: Double = 0
    @Published var heartRateVariability: Double = 0
    @Published var restingHeartRate: Double = 0
    @Published var activeEnergyBurned: Double = 0
    @Published var stepCount: Int = 0
    @Published var sleepData: SleepAnalysis?
    @Published var workoutSessions: [WorkoutSession] = []
    @Published var stressLevel: StressLevel = .normal
    @Published var isMonitoring = false
    @Published var lastSyncDate: Date?
    
    // MARK: - Private Properties
    private let healthStore = HKHealthStore()
    private var watchSession: WCSession?
    private var healthQueries: [HKQuery] = []
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Health Data Types
    private let healthDataTypes: Set<HKSampleType> = [
        HKQuantityType.quantityType(forIdentifier: .heartRate)!,
        HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
        HKQuantityType.quantityType(forIdentifier: .restingHeartRate)!,
        HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!,
        HKQuantityType.quantityType(forIdentifier: .stepCount)!,
        HKQuantityType.quantityType(forIdentifier: .respiratoryRate)!,
        HKQuantityType.quantityType(forIdentifier: .oxygenSaturation)!,
        HKCategoryType.categoryType(forIdentifier: .sleepAnalysis)!,
        HKWorkoutType.workoutType()
    ]
    
    // MARK: - Initialization
    override init() {
        super.init()
        setupWatchConnectivity()
        setupHealthKit()
    }
    
    deinit {
        stopAllQueries()
    }
    
    // MARK: - Public Methods
    
    func requestHealthKitAuthorization() async -> Bool {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("HealthKit is not available on this device")
            return false
        }
        
        do {
            try await healthStore.requestAuthorization(toShare: [], read: healthDataTypes)
            await MainActor.run {
                self.isHealthKitAuthorized = true
            }
            return true
        } catch {
            print("HealthKit authorization failed: \(error)")
            return false
        }
    }
    
    func startMonitoring() async {
        guard isHealthKitAuthorized else {
            print("HealthKit not authorized")
            return
        }
        
        isMonitoring = true
        
        await startHeartRateMonitoring()
        await startHRVMonitoring()
        await startActivityMonitoring()
        await startSleepMonitoring()
        await startWorkoutMonitoring()
        
        // Send monitoring command to watch
        sendMessageToWatch(["command": "startMonitoring"])
    }
    
    func stopMonitoring() {
        isMonitoring = false
        stopAllQueries()
        
        // Send stop command to watch
        sendMessageToWatch(["command": "stopMonitoring"])
    }
    
    func syncHealthData() async {
        guard isHealthKitAuthorized else { return }
        
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await self.fetchLatestHeartRate() }
            group.addTask { await self.fetchLatestHRV() }
            group.addTask { await self.fetchLatestActivity() }
            group.addTask { await self.fetchLatestSleep() }
            group.addTask { await self.fetchLatestWorkouts() }
        }
        
        lastSyncDate = Date()
    }
    
    func getHealthSummary(for date: Date) async -> HealthSummary {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!
        
        async let heartRateData = fetchHeartRateData(from: startOfDay, to: endOfDay)
        async let activityData = fetchActivityData(from: startOfDay, to: endOfDay)
        async let sleepData = fetchSleepData(from: startOfDay, to: endOfDay)
        
        let (heartRate, activity, sleep) = await (heartRateData, activityData, sleepData)
        
        return HealthSummary(
            date: date,
            averageHeartRate: heartRate.average,
            maxHeartRate: heartRate.max,
            minHeartRate: heartRate.min,
            heartRateVariability: heartRate.hrv,
            steps: activity.steps,
            activeCalories: activity.calories,
            sleepDuration: sleep?.totalDuration ?? 0,
            sleepQuality: sleep?.quality ?? 0,
            stressScore: calculateStressScore(heartRate: heartRate, activity: activity)
        )
    }
    
    func detectStressLevel() async -> StressLevel {
        let recentHRV = await fetchRecentHRV()
        let recentHeartRate = await fetchRecentHeartRate()
        
        let stressScore = calculateStressFromHRV(hrv: recentHRV, heartRate: recentHeartRate)
        
        let level: StressLevel
        if stressScore > 0.7 {
            level = .high
        } else if stressScore > 0.4 {
            level = .moderate
        } else {
            level = .normal
        }
        
        await MainActor.run {
            self.stressLevel = level
        }
        
        return level
    }
    
    func triggerWorkoutDetection() {
        sendMessageToWatch(["command": "detectWorkout"])
    }
    
    func requestEmergencyData() async -> EmergencyHealthData {
        async let heartRate = fetchLatestHeartRate()
        async let location = getCurrentLocation()
        
        let (hr, loc) = await (heartRate, location)
        
        return EmergencyHealthData(
            timestamp: Date(),
            heartRate: hr,
            location: loc,
            medicalID: await fetchMedicalID()
        )
    }
    
    // MARK: - Private Health Monitoring Methods
    
    private func startHeartRateMonitoring() async {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                if let latestSample = samples.last {
                    self?.currentHeartRate = latestSample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                }
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                if let latestSample = samples.last {
                    self?.currentHeartRate = latestSample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                }
            }
        }
        
        healthStore.execute(query)
        healthQueries.append(query)
    }
    
    private func startHRVMonitoring() async {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: hrvType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                if let latestSample = samples.last {
                    self?.heartRateVariability = latestSample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                }
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                if let latestSample = samples.last {
                    self?.heartRateVariability = latestSample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                }
            }
        }
        
        healthStore.execute(query)
        healthQueries.append(query)
    }
    
    private func startActivityMonitoring() async {
        // Monitor step count
        guard let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount) else { return }
        
        let stepQuery = HKAnchoredObjectQuery(
            type: stepType,
            predicate: HKQuery.predicateForSamples(withStart: Calendar.current.startOfDay(for: Date()), end: nil, options: .strictStartDate),
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            let totalSteps = samples.reduce(0) { $0 + Int($1.quantity.doubleValue(for: .count())) }
            
            Task { @MainActor in
                self?.stepCount = totalSteps
            }
        }
        
        healthStore.execute(stepQuery)
        healthQueries.append(stepQuery)
        
        // Monitor active energy
        guard let energyType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned) else { return }
        
        let energyQuery = HKAnchoredObjectQuery(
            type: energyType,
            predicate: HKQuery.predicateForSamples(withStart: Calendar.current.startOfDay(for: Date()), end: nil, options: .strictStartDate),
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            let totalEnergy = samples.reduce(0) { $0 + $1.quantity.doubleValue(for: .kilocalorie()) }
            
            Task { @MainActor in
                self?.activeEnergyBurned = totalEnergy
            }
        }
        
        healthStore.execute(energyQuery)
        healthQueries.append(energyQuery)
    }
    
    private func startSleepMonitoring() async {
        guard let sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: sleepType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKCategorySample] else { return }
            
            Task {
                let sleepAnalysis = await self?.analyzeSleepData(samples)
                await MainActor.run {
                    self?.sleepData = sleepAnalysis
                }
            }
        }
        
        healthStore.execute(query)
        healthQueries.append(query)
    }
    
    private func startWorkoutMonitoring() async {
        let workoutType = HKWorkoutType.workoutType()
        
        let query = HKAnchoredObjectQuery(
            type: workoutType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let workouts = samples as? [HKWorkout] else { return }
            
            let sessions = workouts.map { workout in
                WorkoutSession(
                    id: workout.uuid.uuidString,
                    type: workout.workoutActivityType,
                    startDate: workout.startDate,
                    endDate: workout.endDate,
                    duration: workout.duration,
                    totalEnergyBurned: workout.totalEnergyBurned?.doubleValue(for: .kilocalorie()) ?? 0,
                    totalDistance: workout.totalDistance?.doubleValue(for: .meter()) ?? 0
                )
            }
            
            Task { @MainActor in
                self?.workoutSessions = sessions
            }
        }
        
        healthStore.execute(query)
        healthQueries.append(query)
    }
    
    // MARK: - Data Fetching Methods
    
    private func fetchLatestHeartRate() async -> Double {
        return await withCheckedContinuation { continuation in
            guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
                continuation.resume(returning: 0)
                return
            }
            
            let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
            let query = HKSampleQuery(
                sampleType: heartRateType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [sortDescriptor]
            ) { query, samples, error in
                guard let sample = samples?.first as? HKQuantitySample else {
                    continuation.resume(returning: 0)
                    return
                }
                
                let heartRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                continuation.resume(returning: heartRate)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func fetchLatestHRV() async -> Double {
        return await withCheckedContinuation { continuation in
            guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
                continuation.resume(returning: 0)
                return
            }
            
            let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
            let query = HKSampleQuery(
                sampleType: hrvType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [sortDescriptor]
            ) { query, samples, error in
                guard let sample = samples?.first as? HKQuantitySample else {
                    continuation.resume(returning: 0)
                    return
                }
                
                let hrv = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                continuation.resume(returning: hrv)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func fetchLatestActivity() async {
        // Implementation for fetching latest activity data
    }
    
    private func fetchLatestSleep() async {
        // Implementation for fetching latest sleep data
    }
    
    private func fetchLatestWorkouts() async {
        // Implementation for fetching latest workout data
    }
    
    private func fetchHeartRateData(from startDate: Date, to endDate: Date) async -> HeartRateData {
        return await withCheckedContinuation { continuation in
            guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
                continuation.resume(returning: HeartRateData(average: 0, max: 0, min: 0, hrv: 0))
                return
            }
            
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
            let query = HKSampleQuery(
                sampleType: heartRateType,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: nil
            ) { query, samples, error in
                guard let samples = samples as? [HKQuantitySample], !samples.isEmpty else {
                    continuation.resume(returning: HeartRateData(average: 0, max: 0, min: 0, hrv: 0))
                    return
                }
                
                let heartRates = samples.map { $0.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute())) }
                let average = heartRates.reduce(0, +) / Double(heartRates.count)
                let max = heartRates.max() ?? 0
                let min = heartRates.min() ?? 0
                
                continuation.resume(returning: HeartRateData(average: average, max: max, min: min, hrv: 0))
            }
            
            healthStore.execute(query)
        }
    }
    
    private func fetchActivityData(from startDate: Date, to endDate: Date) async -> ActivityData {
        return await withCheckedContinuation { continuation in
            let group = DispatchGroup()
            var steps = 0
            var calories = 0.0
            
            // Fetch steps
            group.enter()
            if let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount) {
                let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
                let query = HKSampleQuery(
                    sampleType: stepType,
                    predicate: predicate,
                    limit: HKObjectQueryNoLimit,
                    sortDescriptors: nil
                ) { query, samples, error in
                    if let samples = samples as? [HKQuantitySample] {
                        steps = samples.reduce(0) { $0 + Int($1.quantity.doubleValue(for: .count())) }
                    }
                    group.leave()
                }
                healthStore.execute(query)
            } else {
                group.leave()
            }
            
            // Fetch calories
            group.enter()
            if let energyType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned) {
                let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
                let query = HKSampleQuery(
                    sampleType: energyType,
                    predicate: predicate,
                    limit: HKObjectQueryNoLimit,
                    sortDescriptors: nil
                ) { query, samples, error in
                    if let samples = samples as? [HKQuantitySample] {
                        calories = samples.reduce(0) { $0 + $1.quantity.doubleValue(for: .kilocalorie()) }
                    }
                    group.leave()
                }
                healthStore.execute(query)
            } else {
                group.leave()
            }
            
            group.notify(queue: .main) {
                continuation.resume(returning: ActivityData(steps: steps, calories: calories))
            }
        }
    }
    
    private func fetchSleepData(from startDate: Date, to endDate: Date) async -> SleepAnalysis? {
        return await withCheckedContinuation { continuation in
            guard let sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis) else {
                continuation.resume(returning: nil)
                return
            }
            
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
            let query = HKSampleQuery(
                sampleType: sleepType,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: nil
            ) { query, samples, error in
                guard let samples = samples as? [HKCategorySample] else {
                    continuation.resume(returning: nil)
                    return
                }
                
                Task {
                    let analysis = await self.analyzeSleepData(samples)
                    continuation.resume(returning: analysis)
                }
            }
            
            healthStore.execute(query)
        }
    }
    
    // MARK: - Analysis Methods
    
    private func analyzeSleepData(_ samples: [HKCategorySample]) async -> SleepAnalysis {
        var totalSleepTime: TimeInterval = 0
        var deepSleepTime: TimeInterval = 0
        var remSleepTime: TimeInterval = 0
        var lightSleepTime: TimeInterval = 0
        
        for sample in samples {
            let duration = sample.endDate.timeIntervalSince(sample.startDate)
            totalSleepTime += duration
            
            switch sample.value {
            case HKCategoryValueSleepAnalysis.asleepDeep.rawValue:
                deepSleepTime += duration
            case HKCategoryValueSleepAnalysis.asleepREM.rawValue:
                remSleepTime += duration
            default:
                lightSleepTime += duration
            }
        }
        
        let efficiency = totalSleepTime > 0 ? (deepSleepTime + remSleepTime) / totalSleepTime : 0
        let quality = calculateSleepQuality(efficiency: efficiency, totalSleep: totalSleepTime)
        
        return SleepAnalysis(
            totalDuration: totalSleepTime,
            deepSleepDuration: deepSleepTime,
            remSleepDuration: remSleepTime,
            lightSleepDuration: lightSleepTime,
            efficiency: efficiency,
            quality: quality,
            bedtime: samples.first?.startDate,
            wakeTime: samples.last?.endDate
        )
    }
    
    private func calculateSleepQuality(efficiency: Double, totalSleep: TimeInterval) -> Double {
        let optimalSleep: TimeInterval = 8 * 3600 // 8 hours
        let sleepScore = min(1.0, totalSleep / optimalSleep)
        let efficiencyScore = efficiency
        
        return (sleepScore * 0.6 + efficiencyScore * 0.4) * 10 // Scale to 0-10
    }
    
    private func calculateStressScore(heartRate: HeartRateData, activity: ActivityData) -> Double {
        // Simplified stress calculation based on heart rate and activity
        let restingHRNorm = max(0, min(1, (100 - heartRate.average) / 40)) // Normalize resting HR
        let activityNorm = min(1, Double(activity.steps) / 10000) // Normalize steps
        
        return (restingHRNorm * 0.7 + activityNorm * 0.3) * 10
    }
    
    private func calculateStressFromHRV(hrv: Double, heartRate: Double) -> Double {
        // Lower HRV typically indicates higher stress
        let normalHRV = 50.0 // Baseline HRV
        let hrvStress = max(0, (normalHRV - hrv) / normalHRV)
        
        // Higher resting heart rate can indicate stress
        let normalHR = 70.0 // Baseline resting HR
        let hrStress = max(0, (heartRate - normalHR) / 30)
        
        return min(1.0, (hrvStress * 0.7 + hrStress * 0.3))
    }
    
    private func fetchRecentHRV() async -> Double {
        // Fetch HRV from last hour
        let endDate = Date()
        let startDate = endDate.addingTimeInterval(-3600) // 1 hour ago
        
        return await withCheckedContinuation { continuation in
            guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
                continuation.resume(returning: 0)
                return
            }
            
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
            let query = HKSampleQuery(
                sampleType: hrvType,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: nil
            ) { query, samples, error in
                guard let samples = samples as? [HKQuantitySample], !samples.isEmpty else {
                    continuation.resume(returning: 0)
                    return
                }
                
                let avgHRV = samples.reduce(0) { $0 + $1.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli)) } / Double(samples.count)
                continuation.resume(returning: avgHRV)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func fetchRecentHeartRate() async -> Double {
        // Fetch heart rate from last 10 minutes
        let endDate = Date()
        let startDate = endDate.addingTimeInterval(-600) // 10 minutes ago
        
        return await withCheckedContinuation { continuation in
            guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
                continuation.resume(returning: 0)
                return
            }
            
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
            let query = HKSampleQuery(
                sampleType: heartRateType,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: nil
            ) { query, samples, error in
                guard let samples = samples as? [HKQuantitySample], !samples.isEmpty else {
                    continuation.resume(returning: 0)
                    return
                }
                
                let avgHR = samples.reduce(0) { $0 + $1.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute())) } / Double(samples.count)
                continuation.resume(returning: avgHR)
            }
            
            healthStore.execute(query)
        }
    }
    
    // MARK: - Watch Connectivity
    
    private func setupWatchConnectivity() {
        guard WCSession.isSupported() else { return }
        
        watchSession = WCSession.default
        watchSession?.delegate = self
        watchSession?.activate()
    }
    
    private func sendMessageToWatch(_ message: [String: Any]) {
        guard let session = watchSession, session.isReachable else { return }
        
        session.sendMessage(message, replyHandler: nil) { error in
            print("Failed to send message to watch: \(error)")
        }
    }
    
    // MARK: - Utility Methods
    
    private func setupHealthKit() {
        // Initial setup for HealthKit
    }
    
    private func stopAllQueries() {
        healthQueries.forEach { healthStore.stop($0) }
        healthQueries.removeAll()
    }
    
    private func getCurrentLocation() async -> CLLocation? {
        // Implementation for getting current location
        return nil
    }
    
    private func fetchMedicalID() async -> MedicalID? {
        // Implementation for fetching medical ID
        return nil
    }
}

// MARK: - WCSessionDelegate

extension AppleWatchManager: WCSessionDelegate {
    
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.isWatchConnected = activationState == .activated
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isWatchConnected = false
        }
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isWatchConnected = false
        }
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        // Handle messages from watch
        if let command = message["command"] as? String {
            switch command {
            case "heartRateUpdate":
                if let heartRate = message["heartRate"] as? Double {
                    DispatchQueue.main.async {
                        self.currentHeartRate = heartRate
                    }
                }
            case "workoutStarted":
                // Handle workout started
                break
            case "emergencyAlert":
                // Handle emergency alert from watch
                break
            default:
                break
            }
        }
    }
}

// MARK: - Supporting Types

struct HealthSummary {
    let date: Date
    let averageHeartRate: Double
    let maxHeartRate: Double
    let minHeartRate: Double
    let heartRateVariability: Double
    let steps: Int
    let activeCalories: Double
    let sleepDuration: TimeInterval
    let sleepQuality: Double
    let stressScore: Double
}

struct SleepAnalysis {
    let totalDuration: TimeInterval
    let deepSleepDuration: TimeInterval
    let remSleepDuration: TimeInterval
    let lightSleepDuration: TimeInterval
    let efficiency: Double
    let quality: Double
    let bedtime: Date?
    let wakeTime: Date?
}

struct WorkoutSession {
    let id: String
    let type: HKWorkoutActivityType
    let startDate: Date
    let endDate: Date
    let duration: TimeInterval
    let totalEnergyBurned: Double
    let totalDistance: Double
}

enum StressLevel: String, CaseIterable {
    case normal = "Normal"
    case moderate = "Moderate"
    case high = "High"
    
    var color: String {
        switch self {
        case .normal: return "green"
        case .moderate: return "orange"
        case .high: return "red"
        }
    }
}

struct EmergencyHealthData {
    let timestamp: Date
    let heartRate: Double
    let location: CLLocation?
    let medicalID: MedicalID?
}

struct MedicalID {
    let name: String
    let dateOfBirth: Date
    let medicalConditions: [String]
    let medications: [String]
    let allergies: [String]
    let emergencyContacts: [EmergencyContact]
}

struct EmergencyContact {
    let name: String
    let phoneNumber: String
    let relationship: String
}

struct HeartRateData {
    let average: Double
    let max: Double
    let min: Double
    let hrv: Double
}

struct ActivityData {
    let steps: Int
    let calories: Double
}