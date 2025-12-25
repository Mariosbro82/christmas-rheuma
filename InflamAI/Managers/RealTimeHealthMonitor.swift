//
//  RealTimeHealthMonitor.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-20.
//

import Foundation
import HealthKit
import Combine
import SwiftUI
import BackgroundTasks

@MainActor
class RealTimeHealthMonitor: ObservableObject {
    static let shared = RealTimeHealthMonitor()
    
    // MARK: - Published Properties
    
    @Published var isMonitoring = false
    @Published var currentVitals = VitalSigns()
    @Published var realtimeMetrics = RealtimeHealthMetrics()
    @Published var healthAlerts: [HealthAlert] = []
    @Published var connectionStatus: HealthKitConnectionStatus = .disconnected
    @Published var lastUpdateTime: Date?
    
    // MARK: - Private Properties
    
    private let healthStore = HKHealthStore()
    private var cancellables = Set<AnyCancellable>()
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    private var streamingQueries: [HKAnchoredObjectQuery] = []
    private var workoutSession: HKWorkoutSession?
    
    // Background processing
    private let backgroundQueue = DispatchQueue(label: "health.monitoring.background", qos: .utility)
    private let processingQueue = DispatchQueue(label: "health.processing", qos: .userInitiated)
    
    // Data buffers for real-time processing
    private var heartRateBuffer: CircularBuffer<Double> = CircularBuffer(capacity: 100)
    private var stepsBuffer: CircularBuffer<Double> = CircularBuffer(capacity: 1000)
    private var painLevelBuffer: CircularBuffer<Double> = CircularBuffer(capacity: 50)
    
    // Timers for periodic updates
    private var vitalsUpdateTimer: Timer?
    private var metricsUpdateTimer: Timer?
    
    // MARK: - Initialization
    
    private init() {
        setupHealthKitAuthorization()
        setupBackgroundProcessing()
    }
    
    // MARK: - Public Methods
    
    func startRealTimeMonitoring() async {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("HealthKit not available")
            return
        }
        
        do {
            try await requestHealthKitPermissions()
            await startContinuousDataStreaming()
            startVitalsMonitoring()
            startBackgroundProcessing()
            
            isMonitoring = true
            connectionStatus = .connected
            
            print("Real-time health monitoring started")
        } catch {
            print("Failed to start monitoring: \(error)")
            connectionStatus = .error(error.localizedDescription)
        }
    }
    
    func stopRealTimeMonitoring() {
        stopContinuousDataStreaming()
        stopVitalsMonitoring()
        stopBackgroundProcessing()
        
        isMonitoring = false
        connectionStatus = .disconnected
        
        print("Real-time health monitoring stopped")
    }
    
    func refreshVitals() async {
        await updateCurrentVitals()
        lastUpdateTime = Date()
    }
    
    // MARK: - HealthKit Setup
    
    private func setupHealthKitAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        
        connectionStatus = .connecting
    }
    
    private func requestHealthKitPermissions() async throws {
        let readTypes: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .distanceWalkingRunning)!,
            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!,
            HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureDiastolic)!,
            HKObjectType.quantityType(forIdentifier: .oxygenSaturation)!,
            HKObjectType.quantityType(forIdentifier: .bodyTemperature)!,
            HKObjectType.quantityType(forIdentifier: .respiratoryRate)!,
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
            HKObjectType.workoutType()
        ]
        
        try await healthStore.requestAuthorization(toShare: [], read: readTypes)
    }
    
    // MARK: - Continuous Data Streaming
    
    private func startContinuousDataStreaming() async {
        await startHeartRateStreaming()
        await startStepCountStreaming()
        await startActivityStreaming()
        await startVitalSignsStreaming()
    }
    
    private func stopContinuousDataStreaming() {
        streamingQueries.forEach { healthStore.stop($0) }
        streamingQueries.removeAll()
    }
    
    private func startHeartRateStreaming() async {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processHeartRateData(samples)
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processHeartRateData(samples)
            }
        }
        
        streamingQueries.append(query)
        healthStore.execute(query)
    }
    
    private func startStepCountStreaming() async {
        guard let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: stepType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processStepCountData(samples)
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processStepCountData(samples)
            }
        }
        
        streamingQueries.append(query)
        healthStore.execute(query)
    }
    
    private func startActivityStreaming() async {
        guard let energyType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: energyType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processActivityData(samples)
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processActivityData(samples)
            }
        }
        
        streamingQueries.append(query)
        healthStore.execute(query)
    }
    
    private func startVitalSignsStreaming() async {
        // Blood pressure monitoring
        await startBloodPressureStreaming()
        
        // Oxygen saturation monitoring
        await startOxygenSaturationStreaming()
        
        // Body temperature monitoring
        await startBodyTemperatureStreaming()
        
        // Respiratory rate monitoring
        await startRespiratoryRateStreaming()
    }
    
    private func startBloodPressureStreaming() async {
        guard let systolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureSystolic),
              let diastolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureDiastolic) else { return }
        
        // Systolic pressure
        let systolicQuery = HKAnchoredObjectQuery(
            type: systolicType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processBloodPressureData(samples, type: .systolic)
            }
        }
        
        systolicQuery.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processBloodPressureData(samples, type: .systolic)
            }
        }
        
        streamingQueries.append(systolicQuery)
        healthStore.execute(systolicQuery)
        
        // Diastolic pressure
        let diastolicQuery = HKAnchoredObjectQuery(
            type: diastolicType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processBloodPressureData(samples, type: .diastolic)
            }
        }
        
        diastolicQuery.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processBloodPressureData(samples, type: .diastolic)
            }
        }
        
        streamingQueries.append(diastolicQuery)
        healthStore.execute(diastolicQuery)
    }
    
    private func startOxygenSaturationStreaming() async {
        guard let oxygenType = HKQuantityType.quantityType(forIdentifier: .oxygenSaturation) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: oxygenType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processOxygenSaturationData(samples)
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processOxygenSaturationData(samples)
            }
        }
        
        streamingQueries.append(query)
        healthStore.execute(query)
    }
    
    private func startBodyTemperatureStreaming() async {
        guard let temperatureType = HKQuantityType.quantityType(forIdentifier: .bodyTemperature) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: temperatureType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processBodyTemperatureData(samples)
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processBodyTemperatureData(samples)
            }
        }
        
        streamingQueries.append(query)
        healthStore.execute(query)
    }
    
    private func startRespiratoryRateStreaming() async {
        guard let respiratoryType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: respiratoryType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processRespiratoryRateData(samples)
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let self = self, let samples = samples as? [HKQuantitySample] else { return }
            
            Task { @MainActor in
                await self.processRespiratoryRateData(samples)
            }
        }
        
        streamingQueries.append(query)
        healthStore.execute(query)
    }
    
    // MARK: - Data Processing
    
    private func processHeartRateData(_ samples: [HKQuantitySample]) async {
        for sample in samples {
            let heartRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            heartRateBuffer.append(heartRate)
            
            await updateCurrentVitals(heartRate: heartRate)
            await analyzeHeartRatePattern(heartRate)
        }
    }
    
    private func processStepCountData(_ samples: [HKQuantitySample]) async {
        for sample in samples {
            let steps = sample.quantity.doubleValue(for: HKUnit.count())
            stepsBuffer.append(steps)
            
            await updateRealtimeMetrics(steps: steps)
        }
    }
    
    private func processActivityData(_ samples: [HKQuantitySample]) async {
        for sample in samples {
            let calories = sample.quantity.doubleValue(for: HKUnit.kilocalorie())
            await updateRealtimeMetrics(calories: calories)
        }
    }
    
    private func processBloodPressureData(_ samples: [HKQuantitySample], type: BloodPressureType) async {
        for sample in samples {
            let pressure = sample.quantity.doubleValue(for: HKUnit.millimeterOfMercury())
            
            switch type {
            case .systolic:
                await updateCurrentVitals(systolicBP: pressure)
            case .diastolic:
                await updateCurrentVitals(diastolicBP: pressure)
            }
            
            await analyzeBloodPressure(pressure, type: type)
        }
    }
    
    private func processOxygenSaturationData(_ samples: [HKQuantitySample]) async {
        for sample in samples {
            let oxygenSat = sample.quantity.doubleValue(for: HKUnit.percent()) * 100
            await updateCurrentVitals(oxygenSaturation: oxygenSat)
            await analyzeOxygenSaturation(oxygenSat)
        }
    }
    
    private func processBodyTemperatureData(_ samples: [HKQuantitySample]) async {
        for sample in samples {
            let temperature = sample.quantity.doubleValue(for: HKUnit.degreeFahrenheit())
            await updateCurrentVitals(bodyTemperature: temperature)
            await analyzeBodyTemperature(temperature)
        }
    }
    
    private func processRespiratoryRateData(_ samples: [HKQuantitySample]) async {
        for sample in samples {
            let respiratoryRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            await updateCurrentVitals(respiratoryRate: respiratoryRate)
            await analyzeRespiratoryRate(respiratoryRate)
        }
    }
    
    // MARK: - Vitals Monitoring
    
    private func startVitalsMonitoring() {
        vitalsUpdateTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.updateCurrentVitals()
            }
        }
        
        metricsUpdateTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.updateRealtimeMetrics()
            }
        }
    }
    
    private func stopVitalsMonitoring() {
        vitalsUpdateTimer?.invalidate()
        vitalsUpdateTimer = nil
        
        metricsUpdateTimer?.invalidate()
        metricsUpdateTimer = nil
    }
    
    private func updateCurrentVitals(
        heartRate: Double? = nil,
        systolicBP: Double? = nil,
        diastolicBP: Double? = nil,
        oxygenSaturation: Double? = nil,
        bodyTemperature: Double? = nil,
        respiratoryRate: Double? = nil
    ) async {
        if let heartRate = heartRate {
            currentVitals.heartRate = heartRate
        }
        
        if let systolicBP = systolicBP {
            currentVitals.bloodPressure.systolic = systolicBP
        }
        
        if let diastolicBP = diastolicBP {
            currentVitals.bloodPressure.diastolic = diastolicBP
        }
        
        if let oxygenSaturation = oxygenSaturation {
            currentVitals.oxygenSaturation = oxygenSaturation
        }
        
        if let bodyTemperature = bodyTemperature {
            currentVitals.bodyTemperature = bodyTemperature
        }
        
        if let respiratoryRate = respiratoryRate {
            currentVitals.respiratoryRate = respiratoryRate
        }
        
        currentVitals.lastUpdated = Date()
        lastUpdateTime = Date()
    }
    
    private func updateRealtimeMetrics(
        steps: Double? = nil,
        calories: Double? = nil
    ) async {
        if let steps = steps {
            realtimeMetrics.dailySteps += Int(steps)
        }
        
        if let calories = calories {
            realtimeMetrics.caloriesBurned += calories
        }
        
        // Calculate averages from buffers
        if !heartRateBuffer.isEmpty {
            realtimeMetrics.averageHeartRate = heartRateBuffer.average
        }
        
        realtimeMetrics.lastUpdated = Date()
    }
    
    // MARK: - Health Analysis
    
    private func analyzeHeartRatePattern(_ heartRate: Double) async {
        // Check for abnormal heart rate
        if heartRate > 100 {
            await createHealthAlert(
                type: .highHeartRate,
                message: "Heart rate elevated: \(Int(heartRate)) BPM",
                severity: heartRate > 120 ? .high : .medium
            )
        } else if heartRate < 60 {
            await createHealthAlert(
                type: .lowHeartRate,
                message: "Heart rate low: \(Int(heartRate)) BPM",
                severity: heartRate < 50 ? .high : .medium
            )
        }
        
        // Analyze heart rate variability
        if heartRateBuffer.count >= 10 {
            let variability = heartRateBuffer.standardDeviation
            if variability > 20 {
                await createHealthAlert(
                    type: .heartRateVariability,
                    message: "High heart rate variability detected",
                    severity: .medium
                )
            }
        }
    }
    
    private func analyzeBloodPressure(_ pressure: Double, type: BloodPressureType) async {
        switch type {
        case .systolic:
            if pressure > 140 {
                await createHealthAlert(
                    type: .highBloodPressure,
                    message: "High systolic blood pressure: \(Int(pressure)) mmHg",
                    severity: pressure > 160 ? .high : .medium
                )
            }
        case .diastolic:
            if pressure > 90 {
                await createHealthAlert(
                    type: .highBloodPressure,
                    message: "High diastolic blood pressure: \(Int(pressure)) mmHg",
                    severity: pressure > 100 ? .high : .medium
                )
            }
        }
    }
    
    private func analyzeOxygenSaturation(_ oxygenSat: Double) async {
        if oxygenSat < 95 {
            await createHealthAlert(
                type: .lowOxygenSaturation,
                message: "Low oxygen saturation: \(String(format: "%.1f", oxygenSat))%",
                severity: oxygenSat < 90 ? .high : .medium
            )
        }
    }
    
    private func analyzeBodyTemperature(_ temperature: Double) async {
        if temperature > 100.4 {
            await createHealthAlert(
                type: .fever,
                message: "Elevated body temperature: \(String(format: "%.1f", temperature))°F",
                severity: temperature > 102 ? .high : .medium
            )
        }
    }
    
    private func analyzeRespiratoryRate(_ rate: Double) async {
        if rate > 20 {
            await createHealthAlert(
                type: .highRespiratoryRate,
                message: "Elevated respiratory rate: \(Int(rate)) breaths/min",
                severity: rate > 25 ? .high : .medium
            )
        } else if rate < 12 {
            await createHealthAlert(
                type: .lowRespiratoryRate,
                message: "Low respiratory rate: \(Int(rate)) breaths/min",
                severity: .medium
            )
        }
    }
    
    // MARK: - Health Alerts
    
    private func createHealthAlert(
        type: HealthAlertType,
        message: String,
        severity: HealthAlertSeverity
    ) async {
        let alert = HealthAlert(
            id: UUID().uuidString,
            type: type,
            message: message,
            severity: severity,
            timestamp: Date(),
            isRead: false
        )
        
        healthAlerts.insert(alert, at: 0)
        
        // Limit alerts to prevent memory issues
        if healthAlerts.count > 50 {
            healthAlerts = Array(healthAlerts.prefix(50))
        }
        
        // Send notification for high severity alerts
        if severity == .high {
            await sendHealthNotification(alert)
        }
    }
    
    private func sendHealthNotification(_ alert: HealthAlert) async {
        // Implementation would send local notification
        print("Health Alert: \(alert.message)")
    }
    
    // MARK: - Background Processing
    
    private func setupBackgroundProcessing() {
        // Register background task
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.inflamai.health.monitoring",
            using: backgroundQueue
        ) { [weak self] task in
            self?.handleBackgroundHealthProcessing(task as! BGProcessingTask)
        }
    }
    
    private func startBackgroundProcessing() {
        backgroundTask = UIApplication.shared.beginBackgroundTask {
            self.endBackgroundTask()
        }
    }
    
    private func stopBackgroundProcessing() {
        endBackgroundTask()
    }
    
    private func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }
    
    private func handleBackgroundHealthProcessing(_ task: BGProcessingTask) {
        task.expirationHandler = {
            task.setTaskCompleted(success: false)
        }
        
        Task {
            await processBackgroundHealthData()
            task.setTaskCompleted(success: true)
        }
    }
    
    private func processBackgroundHealthData() async {
        // Process accumulated health data in background
        await analyzeHealthTrends()
        await generateHealthInsights()
        await cleanupOldData()
    }
    
    private func analyzeHealthTrends() async {
        // Analyze trends in vital signs and activity
        // Implementation would include statistical analysis
    }
    
    private func generateHealthInsights() async {
        // Generate insights based on collected data
        // Implementation would include ML-based analysis
    }
    
    private func cleanupOldData() async {
        // Clean up old alerts and data
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        healthAlerts.removeAll { $0.timestamp < cutoffDate }
    }
}

// MARK: - Supporting Types

struct VitalSigns {
    var heartRate: Double = 0
    var bloodPressure = BloodPressure()
    var oxygenSaturation: Double = 0
    var bodyTemperature: Double = 0
    var respiratoryRate: Double = 0
    var lastUpdated: Date?
}

struct BloodPressure {
    var systolic: Double = 0
    var diastolic: Double = 0
}

struct RealtimeHealthMetrics {
    var dailySteps: Int = 0
    var caloriesBurned: Double = 0
    var averageHeartRate: Double = 0
    var activeMinutes: Int = 0
    var lastUpdated: Date?
}

struct HealthAlert {
    let id: String
    let type: HealthAlertType
    let message: String
    let severity: HealthAlertSeverity
    let timestamp: Date
    var isRead: Bool
}

enum HealthAlertType {
    case highHeartRate
    case lowHeartRate
    case heartRateVariability
    case highBloodPressure
    case lowOxygenSaturation
    case fever
    case highRespiratoryRate
    case lowRespiratoryRate
    case inactivity
    case abnormalPattern
}

enum HealthAlertSeverity {
    case low
    case medium
    case high
}

enum HealthKitConnectionStatus {
    case disconnected
    case connecting
    case connected
    case error(String)
}

enum BloodPressureType {
    case systolic
    case diastolic
}

// MARK: - Circular Buffer

class CircularBuffer<T: Numeric> {
    private var buffer: [T]
    private var head = 0
    private var tail = 0
    private var count = 0
    private let capacity: Int
    
    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: T.zero, count: capacity)
    }
    
    func append(_ element: T) {
        buffer[tail] = element
        tail = (tail + 1) % capacity
        
        if count < capacity {
            count += 1
        } else {
            head = (head + 1) % capacity
        }
    }
    
    var isEmpty: Bool {
        return count == 0
    }
    
    var average: Double {
        guard count > 0 else { return 0 }
        
        let sum = buffer.prefix(count).reduce(T.zero, +)
        return Double(sum as! Double) / Double(count)
    }
    
    var standardDeviation: Double {
        guard count > 1 else { return 0 }
        
        let avg = average
        let variance = buffer.prefix(count)
            .map { pow(Double($0 as! Double) - avg, 2) }
            .reduce(0, +) / Double(count - 1)
        
        return sqrt(variance)
    }
}