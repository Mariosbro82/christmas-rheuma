//
//  HealthKitManager.swift
//  InflamAI-Swift
//
//  DEPRECATED: This manager is being consolidated into HealthKitService.
//  Use HealthKitService.shared for all new code.
//  This file is kept for backward compatibility with legacy views.
//
//  Created by AI Assistant
//

import Foundation
import HealthKit
import Combine
import WatchConnectivity

// MARK: - HealthKit Manager (DEPRECATED - Use HealthKitService)

/// @available(*, deprecated, message: "Use HealthKitService.shared instead")
class HealthKitManager: NSObject, ObservableObject {
    static let shared = HealthKitManager()

    /// Synced with HealthKitService.shared.isAuthorized
    @Published var isAuthorized = false
    @Published var heartRate: Double = 0
    @Published var heartRateVariability: Double = 0
    @Published var sleepData: [SleepData] = []
    @Published var stressLevel: StressLevel = .normal
    @Published var activityData: ActivityData?
    @Published var vitalsData: VitalsData = VitalsData()
    @Published var errorMessage: String?

    private let healthStore = HKHealthStore()
    private var cancellables = Set<AnyCancellable>()
    private var heartRateQuery: HKAnchoredObjectQuery?
    private var hrvQuery: HKAnchoredObjectQuery?

    // Watch connectivity
    private var watchSession: WCSession?

    // Data types we want to read (subset - full list is in HealthKitService)
    private let readTypes: Set<HKObjectType> = [
        HKObjectType.quantityType(forIdentifier: .heartRate)!,
        HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
        HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
        HKObjectType.quantityType(forIdentifier: .walkingHeartRateAverage)!,
        HKObjectType.quantityType(forIdentifier: .stepCount)!,
        HKObjectType.quantityType(forIdentifier: .distanceWalkingRunning)!,
        HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!,
        HKObjectType.quantityType(forIdentifier: .basalEnergyBurned)!,
        HKObjectType.quantityType(forIdentifier: .bodyTemperature)!,
        HKObjectType.quantityType(forIdentifier: .oxygenSaturation)!,
        HKObjectType.quantityType(forIdentifier: .respiratoryRate)!,
        HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
        HKObjectType.categoryType(forIdentifier: .mindfulSession)!,
        HKObjectType.workoutType()
    ]

    // Data types we want to write
    private let writeTypes: Set<HKSampleType> = [
        HKObjectType.quantityType(forIdentifier: .heartRate)!,
        HKObjectType.categoryType(forIdentifier: .mindfulSession)!,
        HKObjectType.workoutType()
    ]

    override init() {
        super.init()
        setupWatchConnectivity()

        // CRITICAL FIX: Do NOT auto-request authorization here!
        // Authorization should be requested from OnboardingFlow or Settings
        // via HealthKitService.shared.requestAuthorization()

        // Instead, sync our isAuthorized state with HealthKitService
        syncAuthorizationState()

        print("⚠️ [HealthKitManager] DEPRECATED - Use HealthKitService.shared instead")
    }

    /// Sync authorization state with HealthKitService (the source of truth)
    private func syncAuthorizationState() {
        Task { @MainActor in
            // Check if HealthKitService already has authorization
            self.isAuthorized = HealthKitService.shared.isAuthorized

            if self.isAuthorized {
                self.startMonitoring()
            }
        }
    }

    // MARK: - Authorization

    /// Request authorization - delegates to HealthKitService for unified flow
    /// DEPRECATED: Call HealthKitService.shared.requestAuthorization() directly
    func requestAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else {
            errorMessage = "HealthKit is not available on this device"
            return
        }

        // Delegate to HealthKitService for unified authorization
        Task { @MainActor in
            let authorized = await HealthKitService.shared.ensureAuthorization()
            self.isAuthorized = authorized

            if authorized {
                self.startMonitoring()
            } else {
                self.errorMessage = "HealthKit authorization required. Please enable in Settings."
            }
        }
    }
    
    // MARK: - Real-time Monitoring
    
    func startMonitoring() {
        guard isAuthorized else { return }
        
        startHeartRateMonitoring()
        startHRVMonitoring()
        fetchRecentSleepData()
        fetchRecentActivityData()
        fetchVitalsData()
        
        // Update stress level based on HRV and heart rate
        updateStressLevel()
    }
    
    func stopMonitoring() {
        heartRateQuery?.stop()
        hrvQuery?.stop()
        heartRateQuery = nil
        hrvQuery = nil
    }
    
    private func startHeartRateMonitoring() {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return }
        
        let predicate = HKQuery.predicateForSamples(withStart: Date().addingTimeInterval(-3600), end: nil, options: .strictEndDate)
        
        heartRateQuery = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: predicate,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            self?.processHeartRateSamples(samples)
        }
        
        heartRateQuery?.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            self?.processHeartRateSamples(samples)
        }
        
        healthStore.execute(heartRateQuery!)
    }
    
    private func startHRVMonitoring() {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else { return }
        
        let predicate = HKQuery.predicateForSamples(withStart: Date().addingTimeInterval(-3600), end: nil, options: .strictEndDate)
        
        hrvQuery = HKAnchoredObjectQuery(
            type: hrvType,
            predicate: predicate,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            self?.processHRVSamples(samples)
        }
        
        hrvQuery?.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            self?.processHRVSamples(samples)
        }
        
        healthStore.execute(hrvQuery!)
    }
    
    private func processHeartRateSamples(_ samples: [HKSample]?) {
        guard let samples = samples as? [HKQuantitySample], let latest = samples.last else { return }
        
        DispatchQueue.main.async {
            self.heartRate = latest.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            self.updateStressLevel()
        }
    }
    
    private func processHRVSamples(_ samples: [HKSample]?) {
        guard let samples = samples as? [HKQuantitySample], let latest = samples.last else { return }
        
        DispatchQueue.main.async {
            self.heartRateVariability = latest.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
            self.updateStressLevel()
        }
    }
    
    // MARK: - Sleep Data
    
    func fetchRecentSleepData() {
        guard let sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis) else { return }
        
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -7, to: endDate)!
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]) { [weak self] query, samples, error in
            
            guard let samples = samples as? [HKCategorySample] else { return }
            
            let sleepData = self?.processSleepSamples(samples) ?? []
            
            DispatchQueue.main.async {
                self?.sleepData = sleepData
            }
        }
        
        healthStore.execute(query)
    }
    
    private func processSleepSamples(_ samples: [HKCategorySample]) -> [SleepData] {
        var sleepDataByDate: [Date: SleepData] = [:]
        
        for sample in samples {
            let calendar = Calendar.current
            let date = calendar.startOfDay(for: sample.startDate)
            
            if sleepDataByDate[date] == nil {
                sleepDataByDate[date] = SleepData(date: date)
            }
            
            let duration = sample.endDate.timeIntervalSince(sample.startDate) / 3600 // hours
            
            switch sample.value {
            case HKCategoryValueSleepAnalysis.inBed.rawValue:
                sleepDataByDate[date]?.timeInBed += duration
            case HKCategoryValueSleepAnalysis.asleepCore.rawValue,
                 HKCategoryValueSleepAnalysis.asleepDeep.rawValue,
                 HKCategoryValueSleepAnalysis.asleepREM.rawValue:
                sleepDataByDate[date]?.totalSleep += duration
                
                if sample.value == HKCategoryValueSleepAnalysis.asleepDeep.rawValue {
                    sleepDataByDate[date]?.deepSleep += duration
                } else if sample.value == HKCategoryValueSleepAnalysis.asleepREM.rawValue {
                    sleepDataByDate[date]?.remSleep += duration
                }
            case HKCategoryValueSleepAnalysis.awake.rawValue:
                sleepDataByDate[date]?.awakeTime += duration
            default:
                break
            }
        }
        
        return Array(sleepDataByDate.values).sorted { $0.date > $1.date }
    }
    
    // MARK: - Activity Data
    
    func fetchRecentActivityData() {
        let group = DispatchGroup()
        var activityData = ActivityData()
        
        // Fetch steps
        group.enter()
        fetchSteps { steps in
            activityData.steps = steps
            group.leave()
        }
        
        // Fetch distance
        group.enter()
        fetchDistance { distance in
            activityData.distance = distance
            group.leave()
        }
        
        // Fetch calories
        group.enter()
        fetchCalories { calories in
            activityData.activeCalories = calories
            group.leave()
        }
        
        group.notify(queue: .main) {
            self.activityData = activityData
        }
    }
    
    private func fetchSteps(completion: @escaping (Int) -> Void) {
        guard let stepsType = HKQuantityType.quantityType(forIdentifier: .stepCount) else {
            completion(0)
            return
        }
        
        let calendar = Calendar.current
        let startDate = calendar.startOfDay(for: Date())
        let endDate = Date()
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKStatisticsQuery(quantityType: stepsType, quantitySamplePredicate: predicate, options: .cumulativeSum) { query, statistics, error in
            let steps = statistics?.sumQuantity()?.doubleValue(for: .count()) ?? 0
            completion(Int(steps))
        }
        
        healthStore.execute(query)
    }
    
    private func fetchDistance(completion: @escaping (Double) -> Void) {
        guard let distanceType = HKQuantityType.quantityType(forIdentifier: .distanceWalkingRunning) else {
            completion(0)
            return
        }
        
        let calendar = Calendar.current
        let startDate = calendar.startOfDay(for: Date())
        let endDate = Date()
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKStatisticsQuery(quantityType: distanceType, quantitySamplePredicate: predicate, options: .cumulativeSum) { query, statistics, error in
            let distance = statistics?.sumQuantity()?.doubleValue(for: .meter()) ?? 0
            completion(distance / 1000) // Convert to kilometers
        }
        
        healthStore.execute(query)
    }
    
    private func fetchCalories(completion: @escaping (Int) -> Void) {
        guard let caloriesType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned) else {
            completion(0)
            return
        }
        
        let calendar = Calendar.current
        let startDate = calendar.startOfDay(for: Date())
        let endDate = Date()
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKStatisticsQuery(quantityType: caloriesType, quantitySamplePredicate: predicate, options: .cumulativeSum) { query, statistics, error in
            let calories = statistics?.sumQuantity()?.doubleValue(for: .kilocalorie()) ?? 0
            completion(Int(calories))
        }
        
        healthStore.execute(query)
    }
    
    // MARK: - Vitals Data
    
    func fetchVitalsData() {
        let group = DispatchGroup()
        var vitals = VitalsData()
        
        // Fetch resting heart rate
        group.enter()
        fetchRestingHeartRate { rate in
            vitals.restingHeartRate = rate
            group.leave()
        }
        
        // Fetch oxygen saturation
        group.enter()
        fetchOxygenSaturation { saturation in
            vitals.oxygenSaturation = saturation
            group.leave()
        }
        
        // Fetch respiratory rate
        group.enter()
        fetchRespiratoryRate { rate in
            vitals.respiratoryRate = rate
            group.leave()
        }
        
        // Fetch body temperature
        group.enter()
        fetchBodyTemperature { temperature in
            vitals.bodyTemperature = temperature
            group.leave()
        }
        
        group.notify(queue: .main) {
            self.vitalsData = vitals
        }
    }
    
    private func fetchRestingHeartRate(completion: @escaping (Double) -> Void) {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .restingHeartRate) else {
            completion(0)
            return
        }
        
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -1, to: endDate)!
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictEndDate)
        
        let query = HKSampleQuery(sampleType: heartRateType, predicate: predicate, limit: 1, sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]) { query, samples, error in
            
            guard let sample = samples?.first as? HKQuantitySample else {
                completion(0)
                return
            }
            
            let rate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            completion(rate)
        }
        
        healthStore.execute(query)
    }
    
    private func fetchOxygenSaturation(completion: @escaping (Double) -> Void) {
        guard let oxygenType = HKQuantityType.quantityType(forIdentifier: .oxygenSaturation) else {
            completion(0)
            return
        }
        
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .hour, value: -24, to: endDate)!
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictEndDate)
        
        let query = HKSampleQuery(sampleType: oxygenType, predicate: predicate, limit: 1, sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]) { query, samples, error in
            
            guard let sample = samples?.first as? HKQuantitySample else {
                completion(0)
                return
            }
            
            let saturation = sample.quantity.doubleValue(for: HKUnit.percent()) * 100
            completion(saturation)
        }
        
        healthStore.execute(query)
    }
    
    private func fetchRespiratoryRate(completion: @escaping (Double) -> Void) {
        guard let respiratoryType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate) else {
            completion(0)
            return
        }
        
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .hour, value: -24, to: endDate)!
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictEndDate)
        
        let query = HKSampleQuery(sampleType: respiratoryType, predicate: predicate, limit: 1, sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]) { query, samples, error in
            
            guard let sample = samples?.first as? HKQuantitySample else {
                completion(0)
                return
            }
            
            let rate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            completion(rate)
        }
        
        healthStore.execute(query)
    }
    
    private func fetchBodyTemperature(completion: @escaping (Double) -> Void) {
        guard let temperatureType = HKQuantityType.quantityType(forIdentifier: .bodyTemperature) else {
            completion(0)
            return
        }
        
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .hour, value: -24, to: endDate)!
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictEndDate)
        
        let query = HKSampleQuery(sampleType: temperatureType, predicate: predicate, limit: 1, sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]) { query, samples, error in
            
            guard let sample = samples?.first as? HKQuantitySample else {
                completion(0)
                return
            }
            
            let temperature = sample.quantity.doubleValue(for: HKUnit.degreeCelsius())
            completion(temperature)
        }
        
        healthStore.execute(query)
    }
    
    // MARK: - Stress Level Calculation
    
    private func updateStressLevel() {
        // Calculate stress level based on HRV and heart rate
        let stressScore = calculateStressScore(heartRate: heartRate, hrv: heartRateVariability)
        
        DispatchQueue.main.async {
            self.stressLevel = StressLevel.fromScore(stressScore)
        }
    }
    
    private func calculateStressScore(heartRate: Double, hrv: Double) -> Double {
        // Simplified stress calculation
        // In a real app, this would use more sophisticated algorithms
        
        var score = 0.0
        
        // Heart rate component (higher HR = higher stress)
        if heartRate > 100 {
            score += 0.4
        } else if heartRate > 80 {
            score += 0.2
        }
        
        // HRV component (lower HRV = higher stress)
        if hrv < 20 {
            score += 0.4
        } else if hrv < 40 {
            score += 0.2
        }
        
        // Time of day adjustment
        let hour = Calendar.current.component(.hour, from: Date())
        if hour >= 22 || hour <= 6 {
            score *= 0.8 // Lower stress expectations during sleep hours
        }
        
        return min(score, 1.0)
    }
    
    // MARK: - Watch Connectivity
    
    private func setupWatchConnectivity() {
        if WCSession.isSupported() {
            watchSession = WCSession.default
            watchSession?.delegate = self
            watchSession?.activate()
        }
    }
    
    func sendDataToWatch(_ data: [String: Any]) {
        guard let session = watchSession, session.isReachable else { return }
        
        session.sendMessage(data, replyHandler: nil) { error in
            print("Failed to send data to watch: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Data Writing
    
    func saveMindfulSession(duration: TimeInterval) {
        guard let mindfulType = HKCategoryType.categoryType(forIdentifier: .mindfulSession) else { return }
        
        let startDate = Date().addingTimeInterval(-duration)
        let endDate = Date()
        
        let sample = HKCategorySample(
            type: mindfulType,
            value: HKCategoryValue.notApplicable.rawValue,
            start: startDate,
            end: endDate
        )
        
        healthStore.save(sample) { success, error in
            if let error = error {
                print("Failed to save mindful session: \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Watch Connectivity Delegate

extension HealthKitManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        if let error = error {
            print("Watch session activation failed: \(error.localizedDescription)")
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {}
    
    func sessionDidDeactivate(_ session: WCSession) {
        session.activate()
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        // Handle messages from watch
        DispatchQueue.main.async {
            if let heartRate = message["heartRate"] as? Double {
                self.heartRate = heartRate
            }
            
            if let hrv = message["hrv"] as? Double {
                self.heartRateVariability = hrv
            }
            
            self.updateStressLevel()
        }
    }
}

// MARK: - Supporting Data Models

struct SleepData: Identifiable {
    let id = UUID()
    let date: Date
    var totalSleep: Double = 0
    var deepSleep: Double = 0
    var remSleep: Double = 0
    var timeInBed: Double = 0
    var awakeTime: Double = 0
    
    var sleepEfficiency: Double {
        guard timeInBed > 0 else { return 0 }
        return (totalSleep / timeInBed) * 100
    }
    
    var sleepQuality: SleepQuality {
        if sleepEfficiency >= 85 && totalSleep >= 7 {
            return .excellent
        } else if sleepEfficiency >= 75 && totalSleep >= 6 {
            return .good
        } else if sleepEfficiency >= 65 && totalSleep >= 5 {
            return .fair
        } else {
            return .poor
        }
    }
}

enum SleepQuality: String, CaseIterable {
    case excellent = "Excellent"
    case good = "Good"
    case fair = "Fair"
    case poor = "Poor"
    
    var color: String {
        switch self {
        case .excellent: return "green"
        case .good: return "blue"
        case .fair: return "orange"
        case .poor: return "red"
        }
    }
}

struct ActivityData {
    var steps: Int = 0
    var distance: Double = 0 // kilometers
    var activeCalories: Int = 0
    var workouts: [WorkoutData] = []
    
    var isActiveDay: Bool {
        steps >= 8000 || distance >= 5.0 || activeCalories >= 300
    }
}

struct WorkoutData: Identifiable {
    let id = UUID()
    let type: String
    let duration: TimeInterval
    let calories: Int
    let date: Date
}

struct VitalsData {
    var restingHeartRate: Double = 0
    var oxygenSaturation: Double = 0
    var respiratoryRate: Double = 0
    var bodyTemperature: Double = 0
    
    var isNormal: Bool {
        restingHeartRate >= 60 && restingHeartRate <= 100 &&
        oxygenSaturation >= 95 &&
        respiratoryRate >= 12 && respiratoryRate <= 20 &&
        bodyTemperature >= 36.1 && bodyTemperature <= 37.2
    }
}

enum StressLevel: String, CaseIterable {
    case low = "Low"
    case normal = "Normal"
    case moderate = "Moderate"
    case high = "High"
    case veryHigh = "Very High"
    
    static func fromScore(_ score: Double) -> StressLevel {
        switch score {
        case 0.0..<0.2: return .low
        case 0.2..<0.4: return .normal
        case 0.4..<0.6: return .moderate
        case 0.6..<0.8: return .high
        default: return .veryHigh
        }
    }
    
    var color: String {
        switch self {
        case .low: return "green"
        case .normal: return "blue"
        case .moderate: return "yellow"
        case .high: return "orange"
        case .veryHigh: return "red"
        }
    }
    
    var recommendation: String {
        switch self {
        case .low:
            return "Great! Your stress levels are low. Keep up the good work."
        case .normal:
            return "Your stress levels are normal. Continue with your current routine."
        case .moderate:
            return "Consider taking some time to relax or practice mindfulness."
        case .high:
            return "Your stress levels are elevated. Try deep breathing or meditation."
        case .veryHigh:
            return "Your stress levels are very high. Consider speaking with a healthcare provider."
        }
    }
}

// MARK: - Health Metrics View

struct HealthMetricsView: View {
    @StateObject private var healthManager = HealthKitManager.shared
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                // Real-time vitals
                VitalSignsCard()
                
                // Activity summary
                ActivitySummaryCard()
                
                // Sleep quality
                SleepQualityCard()
                
                // Stress level
                StressLevelCard()
            }
            .padding()
        }
        .navigationTitle("Health Metrics")
        .themedBackground()
        .onAppear {
            if healthManager.isAuthorized {
                healthManager.startMonitoring()
            } else {
                healthManager.requestAuthorization()
            }
        }
        .onDisappear {
            healthManager.stopMonitoring()
        }
    }
}

struct VitalSignsCard: View {
    @StateObject private var healthManager = HealthKitManager.shared
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Vital Signs")
                .font(themeManager.typography.headline)
                .foregroundColor(themeManager.colors.textPrimary)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                VitalMetric(
                    title: "Heart Rate",
                    value: "\(Int(healthManager.heartRate))",
                    unit: "BPM",
                    icon: "heart.fill",
                    color: .red
                )
                
                VitalMetric(
                    title: "HRV",
                    value: "\(Int(healthManager.heartRateVariability))",
                    unit: "ms",
                    icon: "waveform.path.ecg",
                    color: .blue
                )
                
                VitalMetric(
                    title: "Oxygen",
                    value: "\(Int(healthManager.vitalsData.oxygenSaturation))",
                    unit: "%",
                    icon: "lungs.fill",
                    color: .cyan
                )
                
                VitalMetric(
                    title: "Temperature",
                    value: String(format: "%.1f", healthManager.vitalsData.bodyTemperature),
                    unit: "°C",
                    icon: "thermometer",
                    color: .orange
                )
            }
        }
        .themedCard()
    }
}

struct VitalMetric: View {
    let title: String
    let value: String
    let unit: String
    let icon: String
    let color: Color
    
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title3)
                
                Spacer()
            }
            
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(value)
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    Text("\(unit) \(title)")
                        .font(themeManager.typography.caption)
                        .foregroundColor(themeManager.colors.textSecondary)
                }
                
                Spacer()
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(themeManager.colors.surface)
        )
    }
}

struct ActivitySummaryCard: View {
    @StateObject private var healthManager = HealthKitManager.shared
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Today's Activity")
                .font(themeManager.typography.headline)
                .foregroundColor(themeManager.colors.textPrimary)
            
            if let activity = healthManager.activityData {
                HStack(spacing: 20) {
                    ActivityMetric(
                        title: "Steps",
                        value: "\(activity.steps)",
                        icon: "figure.walk",
                        color: .green
                    )
                    
                    ActivityMetric(
                        title: "Distance",
                        value: String(format: "%.1f km", activity.distance),
                        icon: "location",
                        color: .blue
                    )
                    
                    ActivityMetric(
                        title: "Calories",
                        value: "\(activity.activeCalories)",
                        icon: "flame.fill",
                        color: .orange
                    )
                }
            } else {
                Text("Loading activity data...")
                    .foregroundColor(themeManager.colors.textSecondary)
            }
        }
        .themedCard()
    }
}

struct ActivityMetric: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.title2)
            
            Text(value)
                .font(themeManager.typography.body)
                .fontWeight(.semibold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            Text(title)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct SleepQualityCard: View {
    @StateObject private var healthManager = HealthKitManager.shared
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Sleep Quality")
                .font(themeManager.typography.headline)
                .foregroundColor(themeManager.colors.textPrimary)
            
            if let latestSleep = healthManager.sleepData.first {
                VStack(spacing: 8) {
                    HStack {
                        Text("\(String(format: "%.1f", latestSleep.totalSleep)) hours")
                            .font(themeManager.typography.title2)
                            .fontWeight(.bold)
                            .foregroundColor(themeManager.colors.textPrimary)
                        
                        Spacer()
                        
                        Text(latestSleep.sleepQuality.rawValue)
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(Color(latestSleep.sleepQuality.color))
                    }
                    
                    HStack {
                        Text("Efficiency: \(Int(latestSleep.sleepEfficiency))%")
                        Spacer()
                        Text("Deep: \(String(format: "%.1f", latestSleep.deepSleep))h")
                    }
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                }
            } else {
                Text("No recent sleep data")
                    .foregroundColor(themeManager.colors.textSecondary)
            }
        }
        .themedCard()
    }
}

struct StressLevelCard: View {
    @StateObject private var healthManager = HealthKitManager.shared
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Stress Level")
                .font(themeManager.typography.headline)
                .foregroundColor(themeManager.colors.textPrimary)
            
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(healthManager.stressLevel.rawValue)
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(Color(healthManager.stressLevel.color))
                    
                    Text(healthManager.stressLevel.recommendation)
                        .font(themeManager.typography.caption)
                        .foregroundColor(themeManager.colors.textSecondary)
                        .multilineTextAlignment(.leading)
                }
                
                Spacer()
                
                Image(systemName: "brain.head.profile")
                    .font(.title)
                    .foregroundColor(Color(healthManager.stressLevel.color))
            }
        }
        .themedCard()
    }
}

#Preview {
    HealthMetricsView()
        .environmentObject(ThemeManager.shared)
}