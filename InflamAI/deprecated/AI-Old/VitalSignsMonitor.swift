//
//  VitalSignsMonitor.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-20.
//

import Foundation
import HealthKit
import CoreLocation
import WatchConnectivity
import Combine
import SwiftUI
import UserNotifications

// MARK: - Real-time Vital Signs Monitor

@MainActor
class VitalSignsMonitor: NSObject, ObservableObject {
    static let shared = VitalSignsMonitor()
    
    @Published var isMonitoring = false
    @Published var currentVitalSigns: VitalSigns?
    @Published var emergencyAlerts: [EmergencyAlert] = []
    @Published var healthTrends: [HealthTrend] = []
    @Published var wearableConnected = false
    @Published var backgroundProcessingEnabled = false
    
    private let healthStore = HKHealthStore()
    private let emergencyDetector = EmergencyDetectionEngine()
    private let trendAnalyzer = HealthTrendAnalyzer()
    private let wearableManager = WearableDeviceManager()
    private let notificationManager = HealthNotificationManager()
    
    private var cancellables = Set<AnyCancellable>()
    private var monitoringTimer: Timer?
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    
    private override init() {
        super.init()
        setupHealthKit()
        setupWearableConnection()
        setupBackgroundProcessing()
        setupEmergencyDetection()
    }
    
    // MARK: - Setup Methods
    
    private func setupHealthKit() {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("HealthKit not available")
            return
        }
        
        let typesToRead: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureDiastolic)!,
            HKObjectType.quantityType(forIdentifier: .oxygenSaturation)!,
            HKObjectType.quantityType(forIdentifier: .respiratoryRate)!,
            HKObjectType.quantityType(forIdentifier: .bodyTemperature)!,
            HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            if success {
                DispatchQueue.main.async {
                    self?.setupRealTimeQueries()
                }
            } else {
                print("HealthKit authorization failed: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }
    
    private func setupRealTimeQueries() {
        setupHeartRateQuery()
        setupBloodPressureQuery()
        setupOxygenSaturationQuery()
        setupRespiratoryRateQuery()
        setupBodyTemperatureQuery()
        setupActivityQuery()
    }
    
    private func setupWearableConnection() {
        wearableManager.connectionStatusPublisher
            .receive(on: DispatchQueue.main)
            .assign(to: &$wearableConnected)
        
        wearableManager.vitalSignsPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] vitalSigns in
                self?.processVitalSigns(vitalSigns)
            }
            .store(in: &cancellables)
    }
    
    private func setupBackgroundProcessing() {
        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .sink { [weak self] _ in
                self?.startBackgroundProcessing()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .sink { [weak self] _ in
                self?.stopBackgroundProcessing()
            }
            .store(in: &cancellables)
    }
    
    private func setupEmergencyDetection() {
        emergencyDetector.emergencyAlertPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] alert in
                self?.handleEmergencyAlert(alert)
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Monitoring Control
    
    func startMonitoring() {
        guard !isMonitoring else { return }
        
        isMonitoring = true
        
        // Start real-time monitoring timer
        monitoringTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            Task {
                await self?.performMonitoringCycle()
            }
        }
        
        // Start wearable monitoring
        wearableManager.startMonitoring()
        
        // Enable background processing
        backgroundProcessingEnabled = true
        
        print("Vital signs monitoring started")
    }
    
    func stopMonitoring() {
        guard isMonitoring else { return }
        
        isMonitoring = false
        
        // Stop timer
        monitoringTimer?.invalidate()
        monitoringTimer = nil
        
        // Stop wearable monitoring
        wearableManager.stopMonitoring()
        
        // Disable background processing
        backgroundProcessingEnabled = false
        
        print("Vital signs monitoring stopped")
    }
    
    // MARK: - Monitoring Cycle
    
    private func performMonitoringCycle() async {
        // Collect latest vital signs
        let vitalSigns = await collectCurrentVitalSigns()
        
        // Update current readings
        await MainActor.run {
            self.currentVitalSigns = vitalSigns
        }
        
        // Analyze for emergencies
        await emergencyDetector.analyzeVitalSigns(vitalSigns)
        
        // Update health trends
        let trends = await trendAnalyzer.analyzeTrends(vitalSigns: vitalSigns)
        await MainActor.run {
            self.healthTrends = trends
        }
        
        // Send to AI/ML engine for pattern analysis
        await AIMLEngine.shared.monitorRealTimeAnomalies(
            dataStream: AsyncStream { continuation in
                let healthDataPoints = vitalSigns.toHealthDataPoints()
                for point in healthDataPoints {
                    continuation.yield(point)
                }
                continuation.finish()
            }
        )
    }
    
    private func collectCurrentVitalSigns() async -> VitalSigns {
        async let heartRate = getLatestHeartRate()
        async let bloodPressure = getLatestBloodPressure()
        async let oxygenSaturation = getLatestOxygenSaturation()
        async let respiratoryRate = getLatestRespiratoryRate()
        async let bodyTemperature = getLatestBodyTemperature()
        async let activity = getLatestActivityData()
        
        return await VitalSigns(
            heartRate: heartRate,
            bloodPressure: bloodPressure,
            oxygenSaturation: oxygenSaturation,
            respiratoryRate: respiratoryRate,
            bodyTemperature: bodyTemperature,
            activity: activity,
            timestamp: Date()
        )
    }
    
    // MARK: - HealthKit Queries
    
    private func setupHeartRateQuery() {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processHeartRateSamples(samples)
            }
        }
        
        query.updateHandler = { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processHeartRateSamples(samples)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func setupBloodPressureQuery() {
        guard let systolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureSystolic),
              let diastolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureDiastolic) else { return }
        
        let correlationType = HKCorrelationType.correlationType(forIdentifier: .bloodPressure)!
        
        let query = HKAnchoredObjectQuery(
            type: correlationType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKCorrelation] else { return }
            
            Task {
                await self?.processBloodPressureSamples(samples)
            }
        }
        
        query.updateHandler = { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKCorrelation] else { return }
            
            Task {
                await self?.processBloodPressureSamples(samples)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func setupOxygenSaturationQuery() {
        guard let oxygenType = HKQuantityType.quantityType(forIdentifier: .oxygenSaturation) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: oxygenType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processOxygenSaturationSamples(samples)
            }
        }
        
        query.updateHandler = { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processOxygenSaturationSamples(samples)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func setupRespiratoryRateQuery() {
        guard let respiratoryType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: respiratoryType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processRespiratoryRateSamples(samples)
            }
        }
        
        query.updateHandler = { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processRespiratoryRateSamples(samples)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func setupBodyTemperatureQuery() {
        guard let temperatureType = HKQuantityType.quantityType(forIdentifier: .bodyTemperature) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: temperatureType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processBodyTemperatureSamples(samples)
            }
        }
        
        query.updateHandler = { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processBodyTemperatureSamples(samples)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func setupActivityQuery() {
        guard let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: stepType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processActivitySamples(samples)
            }
        }
        
        query.updateHandler = { [weak self] _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            Task {
                await self?.processActivitySamples(samples)
            }
        }
        
        healthStore.execute(query)
    }
    
    // MARK: - Sample Processing
    
    private func processHeartRateSamples(_ samples: [HKQuantitySample]) async {
        guard let latestSample = samples.last else { return }
        
        let heartRate = latestSample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
        
        await MainActor.run {
            self.currentVitalSigns?.heartRate = HeartRateReading(
                bpm: heartRate,
                timestamp: latestSample.endDate,
                source: latestSample.sourceRevision.source.name
            )
        }
        
        // Check for heart rate anomalies
        await emergencyDetector.checkHeartRateAnomaly(heartRate: heartRate, timestamp: latestSample.endDate)
    }
    
    private func processBloodPressureSamples(_ samples: [HKCorrelation]) async {
        guard let latestSample = samples.last else { return }
        
        var systolic: Double = 0
        var diastolic: Double = 0
        
        for object in latestSample.objects {
            if let sample = object as? HKQuantitySample {
                if sample.quantityType == HKQuantityType.quantityType(forIdentifier: .bloodPressureSystolic) {
                    systolic = sample.quantity.doubleValue(for: .millimeterOfMercury())
                } else if sample.quantityType == HKQuantityType.quantityType(forIdentifier: .bloodPressureDiastolic) {
                    diastolic = sample.quantity.doubleValue(for: .millimeterOfMercury())
                }
            }
        }
        
        await MainActor.run {
            self.currentVitalSigns?.bloodPressure = BloodPressureReading(
                systolic: systolic,
                diastolic: diastolic,
                timestamp: latestSample.endDate,
                source: latestSample.sourceRevision.source.name
            )
        }
        
        // Check for blood pressure anomalies
        await emergencyDetector.checkBloodPressureAnomaly(
            systolic: systolic,
            diastolic: diastolic,
            timestamp: latestSample.endDate
        )
    }
    
    private func processOxygenSaturationSamples(_ samples: [HKQuantitySample]) async {
        guard let latestSample = samples.last else { return }
        
        let oxygenSaturation = latestSample.quantity.doubleValue(for: .percent()) * 100
        
        await MainActor.run {
            self.currentVitalSigns?.oxygenSaturation = OxygenSaturationReading(
                percentage: oxygenSaturation,
                timestamp: latestSample.endDate,
                source: latestSample.sourceRevision.source.name
            )
        }
        
        // Check for oxygen saturation anomalies
        await emergencyDetector.checkOxygenSaturationAnomaly(
            saturation: oxygenSaturation,
            timestamp: latestSample.endDate
        )
    }
    
    private func processRespiratoryRateSamples(_ samples: [HKQuantitySample]) async {
        guard let latestSample = samples.last else { return }
        
        let respiratoryRate = latestSample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
        
        await MainActor.run {
            self.currentVitalSigns?.respiratoryRate = RespiratoryRateReading(
                breathsPerMinute: respiratoryRate,
                timestamp: latestSample.endDate,
                source: latestSample.sourceRevision.source.name
            )
        }
        
        // Check for respiratory rate anomalies
        await emergencyDetector.checkRespiratoryRateAnomaly(
            rate: respiratoryRate,
            timestamp: latestSample.endDate
        )
    }
    
    private func processBodyTemperatureSamples(_ samples: [HKQuantitySample]) async {
        guard let latestSample = samples.last else { return }
        
        let temperature = latestSample.quantity.doubleValue(for: .degreeFahrenheit())
        
        await MainActor.run {
            self.currentVitalSigns?.bodyTemperature = BodyTemperatureReading(
                fahrenheit: temperature,
                timestamp: latestSample.endDate,
                source: latestSample.sourceRevision.source.name
            )
        }
        
        // Check for temperature anomalies
        await emergencyDetector.checkBodyTemperatureAnomaly(
            temperature: temperature,
            timestamp: latestSample.endDate
        )
    }
    
    private func processActivitySamples(_ samples: [HKQuantitySample]) async {
        guard let latestSample = samples.last else { return }
        
        let steps = latestSample.quantity.doubleValue(for: .count())
        
        await MainActor.run {
            self.currentVitalSigns?.activity = ActivityReading(
                steps: Int(steps),
                timestamp: latestSample.endDate,
                source: latestSample.sourceRevision.source.name
            )
        }
    }
    
    // MARK: - Latest Data Retrieval
    
    private func getLatestHeartRate() async -> HeartRateReading? {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return nil }
        
        return await withCheckedContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: heartRateType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
            ) { _, samples, _ in
                guard let sample = samples?.first as? HKQuantitySample else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let heartRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                let reading = HeartRateReading(
                    bpm: heartRate,
                    timestamp: sample.endDate,
                    source: sample.sourceRevision.source.name
                )
                
                continuation.resume(returning: reading)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func getLatestBloodPressure() async -> BloodPressureReading? {
        guard let correlationType = HKCorrelationType.correlationType(forIdentifier: .bloodPressure) else { return nil }
        
        return await withCheckedContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: correlationType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
            ) { _, samples, _ in
                guard let correlation = samples?.first as? HKCorrelation else {
                    continuation.resume(returning: nil)
                    return
                }
                
                var systolic: Double = 0
                var diastolic: Double = 0
                
                for object in correlation.objects {
                    if let sample = object as? HKQuantitySample {
                        if sample.quantityType == HKQuantityType.quantityType(forIdentifier: .bloodPressureSystolic) {
                            systolic = sample.quantity.doubleValue(for: .millimeterOfMercury())
                        } else if sample.quantityType == HKQuantityType.quantityType(forIdentifier: .bloodPressureDiastolic) {
                            diastolic = sample.quantity.doubleValue(for: .millimeterOfMercury())
                        }
                    }
                }
                
                let reading = BloodPressureReading(
                    systolic: systolic,
                    diastolic: diastolic,
                    timestamp: correlation.endDate,
                    source: correlation.sourceRevision.source.name
                )
                
                continuation.resume(returning: reading)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func getLatestOxygenSaturation() async -> OxygenSaturationReading? {
        guard let oxygenType = HKQuantityType.quantityType(forIdentifier: .oxygenSaturation) else { return nil }
        
        return await withCheckedContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: oxygenType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
            ) { _, samples, _ in
                guard let sample = samples?.first as? HKQuantitySample else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let oxygenSaturation = sample.quantity.doubleValue(for: .percent()) * 100
                let reading = OxygenSaturationReading(
                    percentage: oxygenSaturation,
                    timestamp: sample.endDate,
                    source: sample.sourceRevision.source.name
                )
                
                continuation.resume(returning: reading)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func getLatestRespiratoryRate() async -> RespiratoryRateReading? {
        guard let respiratoryType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate) else { return nil }
        
        return await withCheckedContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: respiratoryType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
            ) { _, samples, _ in
                guard let sample = samples?.first as? HKQuantitySample else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let respiratoryRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                let reading = RespiratoryRateReading(
                    breathsPerMinute: respiratoryRate,
                    timestamp: sample.endDate,
                    source: sample.sourceRevision.source.name
                )
                
                continuation.resume(returning: reading)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func getLatestBodyTemperature() async -> BodyTemperatureReading? {
        guard let temperatureType = HKQuantityType.quantityType(forIdentifier: .bodyTemperature) else { return nil }
        
        return await withCheckedContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: temperatureType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
            ) { _, samples, _ in
                guard let sample = samples?.first as? HKQuantitySample else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let temperature = sample.quantity.doubleValue(for: .degreeFahrenheit())
                let reading = BodyTemperatureReading(
                    fahrenheit: temperature,
                    timestamp: sample.endDate,
                    source: sample.sourceRevision.source.name
                )
                
                continuation.resume(returning: reading)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func getLatestActivityData() async -> ActivityReading? {
        guard let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount) else { return nil }
        
        return await withCheckedContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: stepType,
                predicate: nil,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
            ) { _, samples, _ in
                guard let sample = samples?.first as? HKQuantitySample else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let steps = sample.quantity.doubleValue(for: .count())
                let reading = ActivityReading(
                    steps: Int(steps),
                    timestamp: sample.endDate,
                    source: sample.sourceRevision.source.name
                )
                
                continuation.resume(returning: reading)
            }
            
            healthStore.execute(query)
        }
    }
    
    // MARK: - Background Processing
    
    private func startBackgroundProcessing() {
        guard backgroundProcessingEnabled else { return }
        
        backgroundTask = UIApplication.shared.beginBackgroundTask { [weak self] in
            self?.stopBackgroundProcessing()
        }
        
        // Continue monitoring in background with reduced frequency
        monitoringTimer?.invalidate()
        monitoringTimer = Timer.scheduledTimer(withTimeInterval: 120.0, repeats: true) { [weak self] _ in
            Task {
                await self?.performMonitoringCycle()
            }
        }
    }
    
    private func stopBackgroundProcessing() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
        
        // Resume normal monitoring frequency
        if isMonitoring {
            monitoringTimer?.invalidate()
            monitoringTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
                Task {
                    await self?.performMonitoringCycle()
                }
            }
        }
    }
    
    // MARK: - Emergency Handling
    
    private func handleEmergencyAlert(_ alert: EmergencyAlert) {
        emergencyAlerts.append(alert)
        
        // Send immediate notification
        notificationManager.sendEmergencyNotification(alert)
        
        // Trigger haptic feedback
        let impactFeedback = UIImpactFeedbackGenerator(style: .heavy)
        impactFeedback.impactOccurred()
        
        // Log emergency event
        print("EMERGENCY ALERT: \(alert.type.displayName) - \(alert.description)")
        
        // Auto-contact emergency services if critical
        if alert.severity == .critical {
            handleCriticalEmergency(alert)
        }
    }
    
    private func handleCriticalEmergency(_ alert: EmergencyAlert) {
        // In a real app, this would integrate with emergency services
        // For now, we'll just log and notify
        print("CRITICAL EMERGENCY: Immediate medical attention required")
        
        // Send to emergency contacts
        notificationManager.notifyEmergencyContacts(alert)
        
        // Store emergency event for medical records
        storeEmergencyEvent(alert)
    }
    
    private func storeEmergencyEvent(_ alert: EmergencyAlert) {
        // Store emergency event in health records
        // This would integrate with the app's data storage system
    }
    
    // MARK: - Vital Signs Processing
    
    private func processVitalSigns(_ vitalSigns: VitalSigns) {
        currentVitalSigns = vitalSigns
        
        // Analyze trends
        Task {
            let trends = await trendAnalyzer.analyzeTrends(vitalSigns: vitalSigns)
            await MainActor.run {
                self.healthTrends = trends
            }
        }
        
        // Check for emergencies
        Task {
            await emergencyDetector.analyzeVitalSigns(vitalSigns)
        }
    }
}

// MARK: - Emergency Detection Engine

class EmergencyDetectionEngine {
    private let emergencyAlertSubject = PassthroughSubject<EmergencyAlert, Never>()
    
    var emergencyAlertPublisher: AnyPublisher<EmergencyAlert, Never> {
        emergencyAlertSubject.eraseToAnyPublisher()
    }
    
    func analyzeVitalSigns(_ vitalSigns: VitalSigns) async {
        // Check each vital sign for emergency conditions
        
        if let heartRate = vitalSigns.heartRate {
            await checkHeartRateAnomaly(heartRate: heartRate.bpm, timestamp: heartRate.timestamp)
        }
        
        if let bloodPressure = vitalSigns.bloodPressure {
            await checkBloodPressureAnomaly(
                systolic: bloodPressure.systolic,
                diastolic: bloodPressure.diastolic,
                timestamp: bloodPressure.timestamp
            )
        }
        
        if let oxygenSaturation = vitalSigns.oxygenSaturation {
            await checkOxygenSaturationAnomaly(
                saturation: oxygenSaturation.percentage,
                timestamp: oxygenSaturation.timestamp
            )
        }
        
        if let respiratoryRate = vitalSigns.respiratoryRate {
            await checkRespiratoryRateAnomaly(
                rate: respiratoryRate.breathsPerMinute,
                timestamp: respiratoryRate.timestamp
            )
        }
        
        if let bodyTemperature = vitalSigns.bodyTemperature {
            await checkBodyTemperatureAnomaly(
                temperature: bodyTemperature.fahrenheit,
                timestamp: bodyTemperature.timestamp
            )
        }
    }
    
    func checkHeartRateAnomaly(heartRate: Double, timestamp: Date) async {
        let severity: EmergencyAlertSeverity
        let description: String
        
        if heartRate > 150 {
            severity = .critical
            description = "Extremely high heart rate detected: \(Int(heartRate)) BPM"
        } else if heartRate > 120 {
            severity = .high
            description = "High heart rate detected: \(Int(heartRate)) BPM"
        } else if heartRate < 40 {
            severity = .critical
            description = "Extremely low heart rate detected: \(Int(heartRate)) BPM"
        } else if heartRate < 50 {
            severity = .high
            description = "Low heart rate detected: \(Int(heartRate)) BPM"
        } else {
            return // Normal range
        }
        
        let alert = EmergencyAlert(
            id: UUID(),
            type: .heartRateAnomaly,
            severity: severity,
            description: description,
            timestamp: timestamp,
            vitalSigns: ["heartRate": heartRate],
            recommendations: getHeartRateRecommendations(heartRate: heartRate)
        )
        
        emergencyAlertSubject.send(alert)
    }
    
    func checkBloodPressureAnomaly(systolic: Double, diastolic: Double, timestamp: Date) async {
        let severity: EmergencyAlertSeverity
        let description: String
        
        if systolic > 180 || diastolic > 120 {
            severity = .critical
            description = "Hypertensive crisis detected: \(Int(systolic))/\(Int(diastolic)) mmHg"
        } else if systolic > 160 || diastolic > 100 {
            severity = .high
            description = "Severe hypertension detected: \(Int(systolic))/\(Int(diastolic)) mmHg"
        } else if systolic < 90 || diastolic < 60 {
            severity = .high
            description = "Hypotension detected: \(Int(systolic))/\(Int(diastolic)) mmHg"
        } else {
            return // Normal range
        }
        
        let alert = EmergencyAlert(
            id: UUID(),
            type: .bloodPressureAnomaly,
            severity: severity,
            description: description,
            timestamp: timestamp,
            vitalSigns: ["systolic": systolic, "diastolic": diastolic],
            recommendations: getBloodPressureRecommendations(systolic: systolic, diastolic: diastolic)
        )
        
        emergencyAlertSubject.send(alert)
    }
    
    func checkOxygenSaturationAnomaly(saturation: Double, timestamp: Date) async {
        let severity: EmergencyAlertSeverity
        let description: String
        
        if saturation < 85 {
            severity = .critical
            description = "Critical oxygen saturation: \(String(format: "%.1f", saturation))%"
        } else if saturation < 90 {
            severity = .high
            description = "Low oxygen saturation: \(String(format: "%.1f", saturation))%"
        } else if saturation < 95 {
            severity = .medium
            description = "Borderline oxygen saturation: \(String(format: "%.1f", saturation))%"
        } else {
            return // Normal range
        }
        
        let alert = EmergencyAlert(
            id: UUID(),
            type: .oxygenSaturationAnomaly,
            severity: severity,
            description: description,
            timestamp: timestamp,
            vitalSigns: ["oxygenSaturation": saturation],
            recommendations: getOxygenSaturationRecommendations(saturation: saturation)
        )
        
        emergencyAlertSubject.send(alert)
    }
    
    func checkRespiratoryRateAnomaly(rate: Double, timestamp: Date) async {
        let severity: EmergencyAlertSeverity
        let description: String
        
        if rate > 30 {
            severity = .high
            description = "High respiratory rate: \(Int(rate)) breaths/min"
        } else if rate < 8 {
            severity = .critical
            description = "Critically low respiratory rate: \(Int(rate)) breaths/min"
        } else if rate < 12 {
            severity = .medium
            description = "Low respiratory rate: \(Int(rate)) breaths/min"
        } else {
            return // Normal range
        }
        
        let alert = EmergencyAlert(
            id: UUID(),
            type: .respiratoryRateAnomaly,
            severity: severity,
            description: description,
            timestamp: timestamp,
            vitalSigns: ["respiratoryRate": rate],
            recommendations: getRespiratoryRateRecommendations(rate: rate)
        )
        
        emergencyAlertSubject.send(alert)
    }
    
    func checkBodyTemperatureAnomaly(temperature: Double, timestamp: Date) async {
        let severity: EmergencyAlertSeverity
        let description: String
        
        if temperature > 104 {
            severity = .critical
            description = "Dangerous fever: \(String(format: "%.1f", temperature))°F"
        } else if temperature > 101 {
            severity = .medium
            description = "Fever detected: \(String(format: "%.1f", temperature))°F"
        } else if temperature < 95 {
            severity = .high
            description = "Hypothermia risk: \(String(format: "%.1f", temperature))°F"
        } else {
            return // Normal range
        }
        
        let alert = EmergencyAlert(
            id: UUID(),
            type: .bodyTemperatureAnomaly,
            severity: severity,
            description: description,
            timestamp: timestamp,
            vitalSigns: ["bodyTemperature": temperature],
            recommendations: getBodyTemperatureRecommendations(temperature: temperature)
        )
        
        emergencyAlertSubject.send(alert)
    }
    
    // MARK: - Recommendation Generators
    
    private func getHeartRateRecommendations(heartRate: Double) -> [String] {
        if heartRate > 150 {
            return [
                "Seek immediate medical attention",
                "Stop all physical activity",
                "Sit down and rest",
                "Call emergency services if symptoms persist"
            ]
        } else if heartRate > 120 {
            return [
                "Rest and monitor closely",
                "Avoid strenuous activity",
                "Consider contacting your doctor"
            ]
        } else if heartRate < 40 {
            return [
                "Seek immediate medical attention",
                "Do not drive",
                "Call emergency services"
            ]
        } else {
            return [
                "Monitor heart rate",
                "Rest if feeling unwell",
                "Contact doctor if symptoms persist"
            ]
        }
    }
    
    private func getBloodPressureRecommendations(systolic: Double, diastolic: Double) -> [String] {
        if systolic > 180 || diastolic > 120 {
            return [
                "Seek immediate emergency care",
                "Do not wait - call 911",
                "Sit down and rest",
                "Take prescribed emergency medication if available"
            ]
        } else if systolic > 160 || diastolic > 100 {
            return [
                "Contact your doctor immediately",
                "Rest and avoid stress",
                "Monitor blood pressure closely",
                "Take prescribed medication as directed"
            ]
        } else {
            return [
                "Rest and monitor",
                "Avoid sudden movements",
                "Stay hydrated",
                "Contact doctor if symptoms persist"
            ]
        }
    }
    
    private func getOxygenSaturationRecommendations(saturation: Double) -> [String] {
        if saturation < 85 {
            return [
                "Seek immediate emergency care",
                "Call 911 immediately",
                "Sit upright",
                "Use supplemental oxygen if available"
            ]
        } else if saturation < 90 {
            return [
                "Contact your doctor immediately",
                "Sit upright and breathe slowly",
                "Avoid physical exertion",
                "Monitor closely"
            ]
        } else {
            return [
                "Practice deep breathing",
                "Sit upright",
                "Monitor oxygen levels",
                "Contact doctor if levels don't improve"
            ]
        }
    }
    
    private func getRespiratoryRateRecommendations(rate: Double) -> [String] {
        if rate > 30 {
            return [
                "Try to slow your breathing",
                "Sit upright",
                "Contact your doctor",
                "Avoid physical exertion"
            ]
        } else if rate < 8 {
            return [
                "Seek immediate medical attention",
                "Call emergency services",
                "Stay awake and alert"
            ]
        } else {
            return [
                "Monitor breathing",
                "Practice relaxation techniques",
                "Contact doctor if concerned"
            ]
        }
    }
    
    private func getBodyTemperatureRecommendations(temperature: Double) -> [String] {
        if temperature > 104 {
            return [
                "Seek immediate emergency care",
                "Cool down with cold water",
                "Remove excess clothing",
                "Call 911"
            ]
        } else if temperature > 101 {
            return [
                "Rest and stay hydrated",
                "Take fever reducer as directed",
                "Monitor temperature",
                "Contact doctor if fever persists"
            ]
        } else {
            return [
                "Warm up gradually",
                "Seek warm environment",
                "Monitor temperature",
                "Contact doctor if temperature doesn't normalize"
            ]
        }
    }
}

// MARK: - Health Trend Analyzer

class HealthTrendAnalyzer {
    private var historicalData: [VitalSigns] = []
    
    func analyzeTrends(vitalSigns: VitalSigns) async -> [HealthTrend] {
        historicalData.append(vitalSigns)
        
        // Keep only last 100 readings
        if historicalData.count > 100 {
            historicalData = Array(historicalData.suffix(100))
        }
        
        var trends: [HealthTrend] = []
        
        // Analyze heart rate trends
        if let heartRateTrend = analyzeHeartRateTrend() {
            trends.append(heartRateTrend)
        }
        
        // Analyze blood pressure trends
        if let bloodPressureTrend = analyzeBloodPressureTrend() {
            trends.append(bloodPressureTrend)
        }
        
        // Analyze oxygen saturation trends
        if let oxygenTrend = analyzeOxygenSaturationTrend() {
            trends.append(oxygenTrend)
        }
        
        return trends
    }
    
    private func analyzeHeartRateTrend() -> HealthTrend? {
        let heartRates = historicalData.compactMap { $0.heartRate?.bpm }
        guard heartRates.count >= 5 else { return nil }
        
        let trend = calculateTrend(values: heartRates)
        let direction: TrendDirection = trend > 0.5 ? .increasing : trend < -0.5 ? .decreasing : .stable
        
        return HealthTrend(
            id: UUID(),
            type: .heartRate,
            direction: direction,
            magnitude: abs(trend),
            timeframe: .recent,
            description: generateHeartRateTrendDescription(direction: direction, magnitude: abs(trend)),
            significance: calculateSignificance(magnitude: abs(trend))
        )
    }
    
    private func analyzeBloodPressureTrend() -> HealthTrend? {
        let systolicValues = historicalData.compactMap { $0.bloodPressure?.systolic }
        guard systolicValues.count >= 5 else { return nil }
        
        let trend = calculateTrend(values: systolicValues)
        let direction: TrendDirection = trend > 2.0 ? .increasing : trend < -2.0 ? .decreasing : .stable
        
        return HealthTrend(
            id: UUID(),
            type: .bloodPressure,
            direction: direction,
            magnitude: abs(trend),
            timeframe: .recent,
            description: generateBloodPressureTrendDescription(direction: direction, magnitude: abs(trend)),
            significance: calculateSignificance(magnitude: abs(trend) / 10.0) // Normalize for BP
        )
    }
    
    private func analyzeOxygenSaturationTrend() -> HealthTrend? {
        let oxygenValues = historicalData.compactMap { $0.oxygenSaturation?.percentage }
        guard oxygenValues.count >= 5 else { return nil }
        
        let trend = calculateTrend(values: oxygenValues)
        let direction: TrendDirection = trend > 0.5 ? .increasing : trend < -0.5 ? .decreasing : .stable
        
        return HealthTrend(
            id: UUID(),
            type: .oxygenSaturation,
            direction: direction,
            magnitude: abs(trend),
            timeframe: .recent,
            description: generateOxygenSaturationTrendDescription(direction: direction, magnitude: abs(trend)),
            significance: calculateSignificance(magnitude: abs(trend))
        )
    }
    
    private func calculateTrend(values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        
        let n = Double(values.count)
        let x = Array(0..<values.count).map { Double($0) }
        
        let sumX = x.reduce(0, +)
        let sumY = values.reduce(0, +)
        let sumXY = zip(x, values).map(*).reduce(0, +)
        let sumXX = x.map { $0 * $0 }.reduce(0, +)
        
        return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
    }
    
    private func calculateSignificance(magnitude: Double) -> TrendSignificance {
        switch magnitude {
        case 0..<0.3:
            return .low
        case 0.3..<0.6:
            return .moderate
        case 0.6..<0.8:
            return .high
        default:
            return .critical
        }
    }
    
    private func generateHeartRateTrendDescription(direction: TrendDirection, magnitude: Double) -> String {
        switch direction {
        case .increasing:
            return "Heart rate showing upward trend (\(String(format: "%.1f", magnitude)) BPM/reading)"
        case .decreasing:
            return "Heart rate showing downward trend (\(String(format: "%.1f", magnitude)) BPM/reading)"
        case .stable:
            return "Heart rate remains stable"
        }
    }
    
    private func generateBloodPressureTrendDescription(direction: TrendDirection, magnitude: Double) -> String {
        switch direction {
        case .increasing:
            return "Blood pressure showing upward trend (\(String(format: "%.1f", magnitude)) mmHg/reading)"
        case .decreasing:
            return "Blood pressure showing downward trend (\(String(format: "%.1f", magnitude)) mmHg/reading)"
        case .stable:
            return "Blood pressure remains stable"
        }
    }
    
    private func generateOxygenSaturationTrendDescription(direction: TrendDirection, magnitude: Double) -> String {
        switch direction {
        case .increasing:
            return "Oxygen saturation improving (\(String(format: "%.1f", magnitude))%/reading)"
        case .decreasing:
            return "Oxygen saturation declining (\(String(format: "%.1f", magnitude))%/reading)"
        case .stable:
            return "Oxygen saturation remains stable"
        }
    }
}

// MARK: - Wearable Device Manager

class WearableDeviceManager: NSObject, ObservableObject {
    private let connectionStatusSubject = CurrentValueSubject<Bool, Never>(false)
    private let vitalSignsSubject = PassthroughSubject<VitalSigns, Never>()
    
    var connectionStatusPublisher: AnyPublisher<Bool, Never> {
        connectionStatusSubject.eraseToAnyPublisher()
    }
    
    var vitalSignsPublisher: AnyPublisher<VitalSigns, Never> {
        vitalSignsSubject.eraseToAnyPublisher()
    }
    
    override init() {
        super.init()
        setupWatchConnectivity()
    }
    
    private func setupWatchConnectivity() {
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }
    
    func startMonitoring() {
        // Start monitoring from connected wearable devices
        if WCSession.default.isReachable {
            WCSession.default.sendMessage(["command": "startMonitoring"], replyHandler: nil)
        }
    }
    
    func stopMonitoring() {
        // Stop monitoring from connected wearable devices
        if WCSession.default.isReachable {
            WCSession.default.sendMessage(["command": "stopMonitoring"], replyHandler: nil)
        }
    }
}

// MARK: - WCSessionDelegate

extension WearableDeviceManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.connectionStatusSubject.send(activationState == .activated)
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        DispatchQueue.main.async {
            self.connectionStatusSubject.send(false)
        }
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        DispatchQueue.main.async {
            self.connectionStatusSubject.send(false)
        }
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        // Process vital signs data from wearable
        if let vitalSignsData = message["vitalSigns"] as? [String: Any] {
            let vitalSigns = parseVitalSignsFromWearable(vitalSignsData)
            DispatchQueue.main.async {
                self.vitalSignsSubject.send(vitalSigns)
            }
        }
    }
    
    private func parseVitalSignsFromWearable(_ data: [String: Any]) -> VitalSigns {
        var heartRate: HeartRateReading?
        var bloodPressure: BloodPressureReading?
        var oxygenSaturation: OxygenSaturationReading?
        
        if let hr = data["heartRate"] as? Double {
            heartRate = HeartRateReading(bpm: hr, timestamp: Date(), source: "Apple Watch")
        }
        
        if let systolic = data["systolic"] as? Double,
           let diastolic = data["diastolic"] as? Double {
            bloodPressure = BloodPressureReading(
                systolic: systolic,
                diastolic: diastolic,
                timestamp: Date(),
                source: "Apple Watch"
            )
        }
        
        if let oxygen = data["oxygenSaturation"] as? Double {
            oxygenSaturation = OxygenSaturationReading(
                percentage: oxygen,
                timestamp: Date(),
                source: "Apple Watch"
            )
        }
        
        return VitalSigns(
            heartRate: heartRate,
            bloodPressure: bloodPressure,
            oxygenSaturation: oxygenSaturation,
            respiratoryRate: nil,
            bodyTemperature: nil,
            activity: nil,
            timestamp: Date()
        )
    }
}

// MARK: - Health Notification Manager

class HealthNotificationManager {
    init() {
        requestNotificationPermissions()
    }
    
    private func requestNotificationPermissions() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if granted {
                print("Notification permissions granted")
            } else {
                print("Notification permissions denied: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }
    
    func sendEmergencyNotification(_ alert: EmergencyAlert) {
        let content = UNMutableNotificationContent()
        content.title = "Health Emergency Alert"
        content.body = alert.description
        content.sound = .default
        content.categoryIdentifier = "EMERGENCY_ALERT"
        
        // Add custom data
        content.userInfo = [
            "alertId": alert.id.uuidString,
            "severity": alert.severity.rawValue,
            "type": alert.type.rawValue
        ]
        
        let request = UNNotificationRequest(
            identifier: alert.id.uuidString,
            content: content,
            trigger: nil // Immediate delivery
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Failed to send emergency notification: \(error)")
            }
        }
    }
    
    func notifyEmergencyContacts(_ alert: EmergencyAlert) {
        // In a real app, this would send notifications to emergency contacts
        // via SMS, email, or push notifications
        print("Notifying emergency contacts about: \(alert.description)")
    }
}

// MARK: - Supporting Types

struct VitalSigns {
    var heartRate: HeartRateReading?
    var bloodPressure: BloodPressureReading?
    var oxygenSaturation: OxygenSaturationReading?
    var respiratoryRate: RespiratoryRateReading?
    var bodyTemperature: BodyTemperatureReading?
    var activity: ActivityReading?
    let timestamp: Date
    
    func toHealthDataPoints() -> [HealthDataPoint] {
        var dataPoints: [HealthDataPoint] = []
        
        if let heartRate = heartRate {
            dataPoints.append(HealthDataPoint(
                type: .heartRate,
                value: heartRate.bpm,
                timestamp: heartRate.timestamp
            ))
        }
        
        if let bloodPressure = bloodPressure {
            dataPoints.append(HealthDataPoint(
                type: .bloodPressure,
                value: bloodPressure.systolic,
                timestamp: bloodPressure.timestamp
            ))
        }
        
        if let oxygenSaturation = oxygenSaturation {
            dataPoints.append(HealthDataPoint(
                type: .oxygenSaturation,
                value: oxygenSaturation.percentage,
                timestamp: oxygenSaturation.timestamp
            ))
        }
        
        return dataPoints
    }
}

struct HeartRateReading {
    let bpm: Double
    let timestamp: Date
    let source: String
}

struct BloodPressureReading {
    let systolic: Double
    let diastolic: Double
    let timestamp: Date
    let source: String
}

struct OxygenSaturationReading {
    let percentage: Double
    let timestamp: Date
    let source: String
}

struct RespiratoryRateReading {
    let breathsPerMinute: Double
    let timestamp: Date
    let source: String
}

struct BodyTemperatureReading {
    let fahrenheit: Double
    let timestamp: Date
    let source: String
    
    var celsius: Double {
        (fahrenheit - 32) * 5/9
    }
}

struct ActivityReading {
    let steps: Int
    let timestamp: Date
    let source: String
}

struct EmergencyAlert {
    let id: UUID
    let type: EmergencyAlertType
    let severity: EmergencyAlertSeverity
    let description: String
    let timestamp: Date
    let vitalSigns: [String: Double]
    let recommendations: [String]
}

enum EmergencyAlertType: String, CaseIterable {
    case heartRateAnomaly
    case bloodPressureAnomaly
    case oxygenSaturationAnomaly
    case respiratoryRateAnomaly
    case bodyTemperatureAnomaly
    case generalHealthCrisis
    
    var displayName: String {
        switch self {
        case .heartRateAnomaly:
            return "Heart Rate Alert"
        case .bloodPressureAnomaly:
            return "Blood Pressure Alert"
        case .oxygenSaturationAnomaly:
            return "Oxygen Saturation Alert"
        case .respiratoryRateAnomaly:
            return "Respiratory Rate Alert"
        case .bodyTemperatureAnomaly:
            return "Body Temperature Alert"
        case .generalHealthCrisis:
            return "Health Crisis Alert"
        }
    }
}

enum EmergencyAlertSeverity: String, CaseIterable {
    case low
    case medium
    case high
    case critical
    
    var color: Color {
        switch self {
        case .low:
            return .blue
        case .medium:
            return .yellow
        case .high:
            return .orange
        case .critical:
            return .red
        }
    }
    
    var displayName: String {
        switch self {
        case .low:
            return "Low Priority"
        case .medium:
            return "Medium Priority"
        case .high:
            return "High Priority"
        case .critical:
            return "Critical"
        }
    }
}

struct HealthTrend {
    let id: UUID
    let type: HealthTrendType
    let direction: TrendDirection
    let magnitude: Double
    let timeframe: TrendTimeframe
    let description: String
    let significance: TrendSignificance
}

enum HealthTrendType: String, CaseIterable {
    case heartRate
    case bloodPressure
    case oxygenSaturation
    case respiratoryRate
    case bodyTemperature
    case activity
    
    var displayName: String {
        switch self {
        case .heartRate:
            return "Heart Rate"
        case .bloodPressure:
            return "Blood Pressure"
        case .oxygenSaturation:
            return "Oxygen Saturation"
        case .respiratoryRate:
            return "Respiratory Rate"
        case .bodyTemperature:
            return "Body Temperature"
        case .activity:
            return "Activity Level"
        }
    }
}

enum TrendDirection: String, CaseIterable {
    case increasing
    case decreasing
    case stable
    
    var icon: String {
        switch self {
        case .increasing:
            return "arrow.up"
        case .decreasing:
            return "arrow.down"
        case .stable:
            return "minus"
        }
    }
    
    var color: Color {
        switch self {
        case .increasing:
            return .red
        case .decreasing:
            return .blue
        case .stable:
            return .green
        }
    }
}

enum TrendTimeframe: String, CaseIterable {
    case recent
    case hourly
    case daily
    case weekly
    
    var displayName: String {
        switch self {
        case .recent:
            return "Recent"
        case .hourly:
            return "Last Hour"
        case .daily:
            return "Last Day"
        case .weekly:
            return "Last Week"
        }
    }
}

enum TrendSignificance: String, CaseIterable {
    case low
    case moderate
    case high
    case critical
    
    var color: Color {
        switch self {
        case .low:
            return .gray
        case .moderate:
            return .yellow
        case .high:
            return .orange
        case .critical:
            return .red
        }
    }
    
    var displayName: String {
        switch self {
        case .low:
            return "Low Significance"
        case .moderate:
            return "Moderate Significance"
        case .high:
            return "High Significance"
        case .critical:
            return "Critical Significance"
        }
    }
}