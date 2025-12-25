//
//  MedicalFeaturesManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import HealthKit
import Combine
import os.log

// MARK: - Medical Features Manager
class MedicalFeaturesManager: ObservableObject {
    static let shared = MedicalFeaturesManager()
    
    // MARK: - Properties
    @Published var medicationInteractions: [MedicationInteraction] = []
    @Published var symptomScores: [SymptomScore] = []
    @Published var treatmentEffectiveness: [TreatmentEffectiveness] = []
    @Published var clinicalTrials: [ClinicalTrial] = []
    @Published var drugDatabase: [DrugInfo] = []
    @Published var isLoadingInteractions = false
    @Published var isLoadingTrials = false
    @Published var lastInteractionCheck: Date?
    @Published var lastTrialSearch: Date?
    
    private let healthStore = HKHealthStore()
    private let interactionChecker = DrugInteractionChecker()
    private let symptomAnalyzer = SymptomAnalyzer()
    private let treatmentTracker = TreatmentTracker()
    private let trialMatcher = ClinicalTrialMatcher()
    private let logger = Logger(subsystem: "com.inflamai.medical", category: "MedicalFeaturesManager")
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    private init() {
        loadDrugDatabase()
        setupHealthKitObservers()
    }
    
    // MARK: - Setup
    private func loadDrugDatabase() {
        // Load comprehensive drug database
        drugDatabase = DrugDatabase.shared.getAllDrugs()
        logger.info("Loaded \(drugDatabase.count) drugs in database")
    }
    
    private func setupHealthKitObservers() {
        // Set up HealthKit observers for medication and symptom data
        if HKHealthStore.isHealthDataAvailable() {
            requestHealthKitPermissions()
        }
    }
    
    private func requestHealthKitPermissions() {
        let typesToRead: Set<HKObjectType> = [
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureDiastolic)!,
            HKObjectType.quantityType(forIdentifier: .bodyTemperature)!,
            HKObjectType.categoryType(forIdentifier: .mindfulSession)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            if success {
                self?.logger.info("HealthKit permissions granted")
            } else if let error = error {
                self?.logger.error("HealthKit permissions denied: \(error.localizedDescription)")
            }
        }
    }
    
    // MARK: - Medication Interaction Checking
    func checkMedicationInteractions(medications: [Medication]) async throws -> [MedicationInteraction] {
        await MainActor.run {
            isLoadingInteractions = true
        }
        
        logger.info("Checking interactions for \(medications.count) medications")
        
        do {
            let interactions = try await interactionChecker.checkInteractions(medications: medications)
            
            await MainActor.run {
                self.medicationInteractions = interactions
                self.lastInteractionCheck = Date()
                self.isLoadingInteractions = false
            }
            
            logger.info("Found \(interactions.count) potential interactions")
            return interactions
            
        } catch {
            await MainActor.run {
                self.isLoadingInteractions = false
            }
            logger.error("Failed to check interactions: \(error.localizedDescription)")
            throw error
        }
    }
    
    func checkSingleMedicationInteraction(newMedication: Medication, existingMedications: [Medication]) async throws -> [MedicationInteraction] {
        let allMedications = existingMedications + [newMedication]
        return try await checkMedicationInteractions(medications: allMedications)
    }
    
    func getDrugInfo(for medication: Medication) -> DrugInfo? {
        return drugDatabase.first { drug in
            drug.name.lowercased() == medication.name.lowercased() ||
            drug.genericName?.lowercased() == medication.name.lowercased() ||
            drug.brandNames.contains { $0.lowercased() == medication.name.lowercased() }
        }
    }
    
    func searchDrugs(query: String) -> [DrugInfo] {
        let lowercaseQuery = query.lowercased()
        return drugDatabase.filter { drug in
            drug.name.lowercased().contains(lowercaseQuery) ||
            drug.genericName?.lowercased().contains(lowercaseQuery) ?? false ||
            drug.brandNames.contains { $0.lowercased().contains(lowercaseQuery) }
        }
    }
    
    func checkDrugAllergies(medication: Medication, patientAllergies: [String]) -> AllergyAssessment {
        let drugInfo = drugDatabase.getDrugInfo(name: medication.name)
        var allergyRisks: [AllergyRisk] = []
        
        for allergy in patientAllergies {
            if let risk = drugDatabase.checkAllergyRisk(drug: medication.name, allergy: allergy) {
                allergyRisks.append(risk)
            }
        }
        
        let overallRisk: AllergyRiskLevel
        if allergyRisks.contains(where: { $0.severity == .severe }) {
            overallRisk = .high
        } else if allergyRisks.contains(where: { $0.severity == .moderate }) {
            overallRisk = .moderate
        } else if !allergyRisks.isEmpty {
            overallRisk = .low
        } else {
            overallRisk = .none
        }
        
        return AllergyAssessment(
            medication: medication,
            overallRisk: overallRisk,
            allergyRisks: allergyRisks,
            recommendations: generateAllergyRecommendations(risks: allergyRisks)
        )
    }
    
    func generatePersonalizedDosing(medication: Medication, patientProfile: PatientProfile, labResults: LabResults?) -> DosingRecommendation {
        let baseRecommendation = drugDatabase.getStandardDosing(drug: medication.name)
        var adjustedDose = baseRecommendation.standardDose
        var adjustmentFactors: [String] = []
        
        // Age-based adjustments
        if patientProfile.age >= 65 {
            adjustedDose *= 0.8 // Reduce dose for elderly
            adjustmentFactors.append("Age-related dose reduction")
        }
        
        // Kidney function adjustments
        if let creatinine = labResults?.creatinine, creatinine > 1.5 {
            adjustedDose *= 0.7 // Reduce dose for impaired kidney function
            adjustmentFactors.append("Kidney function adjustment")
        }
        
        // Liver function adjustments
        if let alt = labResults?.alt, alt > 40 {
            adjustedDose *= 0.8 // Reduce dose for impaired liver function
            adjustmentFactors.append("Liver function adjustment")
        }
        
        // Weight-based adjustments for certain medications
        if medication.requiresWeightBasedDosing {
            let weightFactor = patientProfile.weight / 70.0 // 70kg reference weight
            adjustedDose *= weightFactor
            adjustmentFactors.append("Weight-based dosing")
        }
        
        return DosingRecommendation(
            medication: medication,
            recommendedDose: adjustedDose,
            frequency: baseRecommendation.frequency,
            route: baseRecommendation.route,
            adjustmentFactors: adjustmentFactors,
            monitoringParameters: generateMonitoringParameters(medication: medication, patientProfile: patientProfile),
            duration: baseRecommendation.duration
        )
    }
    
    // MARK: - Symptom Severity Scoring
    func calculateSymptomScore(painEntries: [PainEntry], timeframe: TimeInterval = 7 * 24 * 60 * 60) -> SymptomScore {
        let cutoffDate = Date().addingTimeInterval(-timeframe)
        let recentEntries = painEntries.filter { $0.date >= cutoffDate }
        
        let score = symptomAnalyzer.calculateScore(entries: recentEntries)
        
        let symptomScore = SymptomScore(
            overallScore: score.overall,
            painScore: score.pain,
            stiffnessScore: score.stiffness,
            fatigueScore: score.fatigue,
            functionalScore: score.functional,
            timeframe: timeframe,
            entryCount: recentEntries.count,
            calculatedDate: Date(),
            trend: calculateTrend(entries: recentEntries),
            severity: determineSeverity(score: score.overall)
        )
        
        DispatchQueue.main.async {
            self.symptomScores.append(symptomScore)
            
            // Keep only last 30 scores
            if self.symptomScores.count > 30 {
                self.symptomScores.removeFirst(self.symptomScores.count - 30)
            }
        }
        
        logger.info("Calculated symptom score: \(score.overall) (\(symptomScore.severity.rawValue))")
        return symptomScore
    }
    
    func getSymptomTrend(days: Int = 30) -> SymptomTrend {
        let recentScores = symptomScores.suffix(days)
        
        guard recentScores.count >= 2 else {
            return SymptomTrend(direction: .stable, magnitude: 0, confidence: 0)
        }
        
        let scores = recentScores.map { $0.overallScore }
        let trend = symptomAnalyzer.calculateTrend(scores: scores)
        
        return trend
    }
    
    func predictFlareRisk(painEntries: [PainEntry], medications: [Medication]) -> FlareRiskAssessment {
        let riskFactors = symptomAnalyzer.analyzeRiskFactors(
            painEntries: painEntries,
            medications: medications
        )
        
        let assessment = FlareRiskAssessment(
            riskLevel: riskFactors.overallRisk,
            confidence: riskFactors.confidence,
            factors: riskFactors.factors,
            recommendations: generateFlareRecommendations(riskFactors: riskFactors),
            timeframe: "Next 7 days",
            calculatedDate: Date()
        )
        
        logger.info("Flare risk assessment: \(assessment.riskLevel.rawValue) (\(Int(assessment.confidence * 100))% confidence)")
        return assessment
    }
    
    // MARK: - Treatment Effectiveness Tracking
    func trackTreatmentEffectiveness(medication: Medication, painEntries: [PainEntry], startDate: Date) -> TreatmentEffectiveness {
        let effectiveness = treatmentTracker.analyzeTreatmentEffectiveness(
            medication: medication,
            painEntries: painEntries,
            startDate: startDate
        )
        
        DispatchQueue.main.async {
            self.treatmentEffectiveness.append(effectiveness)
            
            // Keep only last 50 effectiveness records
            if self.treatmentEffectiveness.count > 50 {
                self.treatmentEffectiveness.removeFirst(self.treatmentEffectiveness.count - 50)
            }
        }
        
        logger.info("Treatment effectiveness for \(medication.name): \(effectiveness.effectivenessScore)")
        return effectiveness
    }
    
    func compareTreatmentEffectiveness(medications: [Medication], painEntries: [PainEntry]) -> [TreatmentComparison] {
        return treatmentTracker.compareTreatments(
            medications: medications,
            painEntries: painEntries
        )
    }
    
    func generateTreatmentRecommendations(patientProfile: PatientProfile, currentMedications: [Medication]) -> [TreatmentRecommendation] {
        return treatmentTracker.generateRecommendations(
            profile: patientProfile,
            currentMedications: currentMedications
        )
    }
    
    // MARK: - Clinical Trial Matching
    func findMatchingClinicalTrials(patientProfile: PatientProfile, location: String? = nil) async throws -> [ClinicalTrial] {
        await MainActor.run {
            isLoadingTrials = true
        }
        
        logger.info("Searching for clinical trials matching patient profile")
        
        do {
            let trials = try await trialMatcher.findMatchingTrials(
                profile: patientProfile,
                location: location
            )
            
            await MainActor.run {
                self.clinicalTrials = trials
                self.lastTrialSearch = Date()
                self.isLoadingTrials = false
            }
            
            logger.info("Found \(trials.count) matching clinical trials")
            return trials
            
        } catch {
            await MainActor.run {
                self.isLoadingTrials = false
            }
            logger.error("Failed to search clinical trials: \(error.localizedDescription)")
            throw error
        }
    }
    
    func getClinicalTrialDetails(trialId: String) async throws -> ClinicalTrialDetails {
        return try await trialMatcher.getTrialDetails(trialId: trialId)
    }
    
    func checkTrialEligibility(trial: ClinicalTrial, patientProfile: PatientProfile) -> EligibilityAssessment {
        return trialMatcher.checkEligibility(trial: trial, profile: patientProfile)
    }
    
    // MARK: - Health Data Integration
    func syncWithHealthKit() async throws {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw MedicalError.healthKitUnavailable
        }
        
        // Sync heart rate data
        try await syncHeartRateData()
        
        // Sync sleep data
        try await syncSleepData()
        
        // Sync blood pressure data
        try await syncBloodPressureData()
        
        logger.info("Successfully synced with HealthKit")
    }
    
    private func syncHeartRateData() async throws {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            throw MedicalError.invalidHealthKitType
        }
        
        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: HKQuery.predicateForSamples(
                withStart: Calendar.current.date(byAdding: .day, value: -7, to: Date()),
                end: Date(),
                options: .strictStartDate
            ),
            limit: HKObjectQueryNoLimit,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
        ) { query, samples, error in
            if let error = error {
                self.logger.error("Failed to fetch heart rate data: \(error.localizedDescription)")
                return
            }
            
            if let heartRateSamples = samples as? [HKQuantitySample] {
                self.processHeartRateData(heartRateSamples)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func syncSleepData() async throws {
        guard let sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis) else {
            throw MedicalError.invalidHealthKitType
        }
        
        let query = HKSampleQuery(
            sampleType: sleepType,
            predicate: HKQuery.predicateForSamples(
                withStart: Calendar.current.date(byAdding: .day, value: -7, to: Date()),
                end: Date(),
                options: .strictStartDate
            ),
            limit: HKObjectQueryNoLimit,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
        ) { query, samples, error in
            if let error = error {
                self.logger.error("Failed to fetch sleep data: \(error.localizedDescription)")
                return
            }
            
            if let sleepSamples = samples as? [HKCategorySample] {
                self.processSleepData(sleepSamples)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func syncBloodPressureData() async throws {
        guard let systolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureSystolic),
              let diastolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureDiastolic) else {
            throw MedicalError.invalidHealthKitType
        }
        
        // Fetch systolic readings
        let systolicQuery = HKSampleQuery(
            sampleType: systolicType,
            predicate: HKQuery.predicateForSamples(
                withStart: Calendar.current.date(byAdding: .day, value: -30, to: Date()),
                end: Date(),
                options: .strictStartDate
            ),
            limit: HKObjectQueryNoLimit,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
        ) { query, samples, error in
            if let bloodPressureSamples = samples as? [HKQuantitySample] {
                self.processBloodPressureData(bloodPressureSamples, type: .systolic)
            }
        }
        
        healthStore.execute(systolicQuery)
    }
    
    private func processHeartRateData(_ samples: [HKQuantitySample]) {
        // Process heart rate data for correlation with symptoms
        let heartRateData = samples.map { sample in
            HeartRateData(
                value: sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute())),
                date: sample.startDate
            )
        }
        
        // Analyze correlation with pain levels
        analyzeHeartRateCorrelation(heartRateData)
    }
    
    private func processSleepData(_ samples: [HKCategorySample]) {
        // Process sleep data for correlation with symptoms
        let sleepData = samples.map { sample in
            SleepData(
                value: sample.value,
                startDate: sample.startDate,
                endDate: sample.endDate,
                duration: sample.endDate.timeIntervalSince(sample.startDate)
            )
        }
        
        // Analyze correlation with pain levels
        analyzeSleepCorrelation(sleepData)
    }
    
    private func processBloodPressureData(_ samples: [HKQuantitySample], type: BloodPressureType) {
        // Process blood pressure data
        let bpData = samples.map { sample in
            BloodPressureReading(
                value: sample.quantity.doubleValue(for: HKUnit.millimeterOfMercury()),
                date: sample.startDate,
                type: type
            )
        }
        
        // Store for correlation analysis
        storeBloodPressureData(bpData)
    }
    
    // MARK: - Private Helper Methods
    private func calculateTrend(entries: [PainEntry]) -> SymptomTrend {
        guard entries.count >= 2 else {
            return SymptomTrend(direction: .stable, magnitude: 0, confidence: 0)
        }
        
        let sortedEntries = entries.sorted { $0.date < $1.date }
        let painLevels = sortedEntries.map { Double($0.painLevel) }
        
        return symptomAnalyzer.calculateTrend(scores: painLevels)
    }
    
    private func determineSeverity(score: Double) -> SymptomSeverity {
        switch score {
        case 0..<3:
            return .mild
        case 3..<6:
            return .moderate
        case 6..<8:
            return .severe
        default:
            return .critical
        }
    }
    
    private func generateFlareRecommendations(riskFactors: RiskFactors) -> [String] {
        var recommendations: [String] = []
        
        if riskFactors.factors.contains(.highStress) {
            recommendations.append("Consider stress reduction techniques such as meditation or deep breathing")
        }
        
        if riskFactors.factors.contains(.poorSleep) {
            recommendations.append("Focus on improving sleep quality and maintaining regular sleep schedule")
        }
        
        if riskFactors.factors.contains(.medicationMissed) {
            recommendations.append("Ensure consistent medication adherence")
        }
        
        if riskFactors.factors.contains(.weatherChanges) {
            recommendations.append("Monitor weather changes and prepare accordingly")
        }
        
        if riskFactors.factors.contains(.increasedActivity) {
            recommendations.append("Consider reducing activity level temporarily")
        }
        
        return recommendations
    }
    
    private func generateAllergyRecommendations(risks: [AllergyRisk]) -> [String] {
        var recommendations: [String] = []
        
        for risk in risks {
            switch risk.severity {
            case .severe:
                recommendations.append("AVOID: \(risk.allergen) - High risk of severe allergic reaction")
            case .moderate:
                recommendations.append("CAUTION: \(risk.allergen) - Monitor for allergic reactions")
            case .mild:
                recommendations.append("MONITOR: \(risk.allergen) - Low risk, watch for symptoms")
            }
        }
        
        if risks.contains(where: { $0.severity == .severe }) {
            recommendations.append("Consider alternative medications")
            recommendations.append("Consult with allergist before starting treatment")
        }
        
        return recommendations
    }
    
    private func generateMonitoringParameters(medication: Medication, patientProfile: PatientProfile) -> [String] {
        var parameters: [String] = []
        
        // Standard monitoring for rheumatoid arthritis medications
        if medication.drugClass == "DMARD" {
            parameters.append("Complete blood count every 4-8 weeks")
            parameters.append("Liver function tests every 4-8 weeks")
            parameters.append("Kidney function tests every 12 weeks")
        }
        
        if medication.drugClass == "Biologic" {
            parameters.append("Infection screening before treatment")
            parameters.append("Tuberculosis screening")
            parameters.append("Hepatitis B/C screening")
            parameters.append("Complete blood count every 4-12 weeks")
        }
        
        if medication.drugClass == "Corticosteroid" {
            parameters.append("Blood glucose monitoring")
            parameters.append("Blood pressure monitoring")
            parameters.append("Bone density screening")
            parameters.append("Eye examination for cataracts/glaucoma")
        }
        
        // Age-specific monitoring
        if patientProfile.age >= 65 {
            parameters.append("Enhanced kidney function monitoring")
            parameters.append("Cardiovascular risk assessment")
        }
        
        return parameters
    }
    
    private func analyzeHeartRateCorrelation(_ heartRateData: [HeartRateData]) {
        // Analyze correlation between heart rate and pain levels
        // This would integrate with your pain tracking data
    }
    
    private func analyzeSleepCorrelation(_ sleepData: [SleepData]) {
        // Analyze correlation between sleep quality and pain levels
        // This would integrate with your pain tracking data
    }
    
    private func storeBloodPressureData(_ bpData: [BloodPressureReading]) {
        // Store blood pressure data for analysis
        // This would integrate with your health data storage
    }
}

// MARK: - Supporting Classes
class DrugInteractionChecker {
    private let drugDatabase = DrugDatabase.shared
    
    func checkInteractions(medications: [Medication]) async throws -> [MedicationInteraction] {
        var interactions: [MedicationInteraction] = []
        
        for i in 0..<medications.count {
            for j in (i+1)..<medications.count {
                let med1 = medications[i]
                let med2 = medications[j]
                
                if let interaction = await checkInteractionBetween(med1: med1, med2: med2) {
                    interactions.append(interaction)
                }
            }
        }
        
        return interactions
    }
    
    private func checkInteractionBetween(med1: Medication, med2: Medication) async -> MedicationInteraction? {
        // Check drug database for known interactions
        let interaction = drugDatabase.getInteraction(drug1: med1.name, drug2: med2.name)
        
        guard let interactionData = interaction else { return nil }
        
        return MedicationInteraction(
            medication1: med1,
            medication2: med2,
            severity: interactionData.severity,
            description: interactionData.description,
            mechanism: interactionData.mechanism,
            recommendations: interactionData.recommendations,
            sources: interactionData.sources
        )
    }
}

class SymptomAnalyzer {
    func calculateScore(entries: [PainEntry]) -> (overall: Double, pain: Double, stiffness: Double, fatigue: Double, functional: Double) {
        guard !entries.isEmpty else {
            return (0, 0, 0, 0, 0)
        }
        
        let painScore = entries.map { Double($0.painLevel) }.reduce(0, +) / Double(entries.count)
        let stiffnessScore = entries.compactMap { $0.stiffnessLevel }.map { Double($0) }.reduce(0, +) / Double(entries.count)
        let fatigueScore = entries.compactMap { $0.fatigueLevel }.map { Double($0) }.reduce(0, +) / Double(entries.count)
        
        // Calculate functional score based on affected joints and activities
        let functionalScore = calculateFunctionalScore(entries: entries)
        
        let overallScore = (painScore + stiffnessScore + fatigueScore + functionalScore) / 4.0
        
        return (overallScore, painScore, stiffnessScore, fatigueScore, functionalScore)
    }
    
    func calculateTrend(scores: [Double]) -> SymptomTrend {
        guard scores.count >= 2 else {
            return SymptomTrend(direction: .stable, magnitude: 0, confidence: 0)
        }
        
        // Simple linear regression to determine trend
        let n = Double(scores.count)
        let x = Array(0..<scores.count).map { Double($0) }
        let y = scores
        
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumXX = x.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
        
        let direction: TrendDirection
        let magnitude = abs(slope)
        
        if slope > 0.1 {
            direction = .increasing
        } else if slope < -0.1 {
            direction = .decreasing
        } else {
            direction = .stable
        }
        
        // Calculate confidence based on R-squared
        let meanY = sumY / n
        let ssTotal = y.map { pow($0 - meanY, 2) }.reduce(0, +)
        let ssResidual = zip(x, y).map { x, y in
            let predicted = slope * x + (sumY - slope * sumX) / n
            return pow(y - predicted, 2)
        }.reduce(0, +)
        
        let rSquared = 1 - (ssResidual / ssTotal)
        let confidence = max(0, min(1, rSquared))
        
        return SymptomTrend(direction: direction, magnitude: magnitude, confidence: confidence)
    }
    
    func analyzeRiskFactors(painEntries: [PainEntry], medications: [Medication]) -> RiskFactors {
        var factors: [RiskFactor] = []
        var overallRisk: RiskLevel = .low
        var confidence: Double = 0.5
        
        // Analyze recent pain trend
        let recentEntries = painEntries.suffix(7) // Last 7 entries
        if recentEntries.count >= 3 {
            let painLevels = recentEntries.map { Double($0.painLevel) }
            let trend = calculateTrend(scores: painLevels)
            
            if trend.direction == .increasing && trend.magnitude > 0.5 {
                factors.append(.increasingPain)
                overallRisk = .moderate
            }
        }
        
        // Check medication adherence
        let missedMedications = medications.filter { medication in
            // This would check if medication was missed recently
            // Simplified logic for now
            return false
        }
        
        if !missedMedications.isEmpty {
            factors.append(.medicationMissed)
            overallRisk = .high
        }
        
        // Check for stress indicators
        let highStressEntries = recentEntries.filter { $0.stressLevel ?? 0 > 7 }
        if highStressEntries.count > recentEntries.count / 2 {
            factors.append(.highStress)
            if overallRisk == .low {
                overallRisk = .moderate
            }
        }
        
        // Calculate confidence based on data availability
        confidence = min(1.0, Double(recentEntries.count) / 7.0)
        
        return RiskFactors(
            factors: factors,
            overallRisk: overallRisk,
            confidence: confidence
        )
    }
    
    private func calculateFunctionalScore(entries: [PainEntry]) -> Double {
        // Calculate functional impairment based on affected joints and activities
        let affectedJoints = entries.flatMap { $0.affectedJoints ?? [] }
        let uniqueJoints = Set(affectedJoints)
        
        // More affected joints = higher functional impairment
        let jointScore = min(10.0, Double(uniqueJoints.count) * 0.5)
        
        return jointScore
    }
}

class TreatmentTracker {
    func analyzeTreatmentEffectiveness(medication: Medication, painEntries: [PainEntry], startDate: Date) -> TreatmentEffectiveness {
        let beforeEntries = painEntries.filter { $0.date < startDate }
        let afterEntries = painEntries.filter { $0.date >= startDate }
        
        guard !beforeEntries.isEmpty && !afterEntries.isEmpty else {
            return TreatmentEffectiveness(
                medication: medication,
                effectivenessScore: 0,
                painReduction: 0,
                sideEffects: [],
                adherenceRate: 0,
                startDate: startDate,
                evaluationPeriod: 0,
                recommendation: .insufficient_data
            )
        }
        
        let beforeAverage = beforeEntries.map { Double($0.painLevel) }.reduce(0, +) / Double(beforeEntries.count)
        let afterAverage = afterEntries.map { Double($0.painLevel) }.reduce(0, +) / Double(afterEntries.count)
        
        let painReduction = beforeAverage - afterAverage
        let effectivenessScore = max(0, min(10, painReduction + 5)) // Scale to 0-10
        
        let recommendation: TreatmentRecommendation.RecommendationType
        if effectivenessScore >= 7 {
            recommendation = .continue
        } else if effectivenessScore >= 4 {
            recommendation = .adjust_dose
        } else {
            recommendation = .consider_alternative
        }
        
        return TreatmentEffectiveness(
            medication: medication,
            effectivenessScore: effectivenessScore,
            painReduction: painReduction,
            sideEffects: [], // Would be populated from user reports
            adherenceRate: 0.9, // Would be calculated from medication logs
            startDate: startDate,
            evaluationPeriod: Date().timeIntervalSince(startDate),
            recommendation: recommendation
        )
    }
    
    func compareTreatments(medications: [Medication], painEntries: [PainEntry]) -> [TreatmentComparison] {
        var comparisons: [TreatmentComparison] = []
        
        for i in 0..<medications.count {
            for j in (i+1)..<medications.count {
                let med1 = medications[i]
                let med2 = medications[j]
                
                // This would compare effectiveness between medications
                let comparison = TreatmentComparison(
                    medication1: med1,
                    medication2: med2,
                    effectiveness1: 7.5, // Would be calculated
                    effectiveness2: 6.2, // Would be calculated
                    sideEffects1: [],
                    sideEffects2: [],
                    recommendation: med1.name // Would be determined by analysis
                )
                
                comparisons.append(comparison)
            }
        }
        
        return comparisons
    }
    
    func generateRecommendations(profile: PatientProfile, currentMedications: [Medication]) -> [TreatmentRecommendation] {
        var recommendations: [TreatmentRecommendation] = []
        
        // Generate recommendations based on patient profile and current treatments
        // This would use clinical guidelines and evidence-based medicine
        
        return recommendations
    }
}

class ClinicalTrialMatcher {
    private let trialDatabase = ClinicalTrialDatabase.shared
    
    func findMatchingTrials(profile: PatientProfile, location: String?) async throws -> [ClinicalTrial] {
        // Search clinical trials database (e.g., ClinicalTrials.gov API)
        let trials = try await trialDatabase.searchTrials(
            condition: "Rheumatoid Arthritis",
            location: location,
            ageRange: profile.ageRange,
            gender: profile.gender
        )
        
        // Filter based on patient profile
        let matchingTrials = trials.filter { trial in
            checkBasicEligibility(trial: trial, profile: profile)
        }
        
        return matchingTrials
    }
    
    func getTrialDetails(trialId: String) async throws -> ClinicalTrialDetails {
        return try await trialDatabase.getTrialDetails(trialId: trialId)
    }
    
    func checkEligibility(trial: ClinicalTrial, profile: PatientProfile) -> EligibilityAssessment {
        var eligibleCriteria: [String] = []
        var ineligibleCriteria: [String] = []
        var unknownCriteria: [String] = []
        
        // Check age criteria
        if let ageRange = trial.ageRange {
            if profile.age >= ageRange.min && profile.age <= ageRange.max {
                eligibleCriteria.append("Age requirement met")
            } else {
                ineligibleCriteria.append("Age requirement not met")
            }
        }
        
        // Check gender criteria
        if let genderRequirement = trial.genderRequirement {
            if genderRequirement == .any || genderRequirement.rawValue == profile.gender {
                eligibleCriteria.append("Gender requirement met")
            } else {
                ineligibleCriteria.append("Gender requirement not met")
            }
        }
        
        // Check disease duration
        if let durationRequirement = trial.diseaseDurationRequirement {
            if profile.diseaseDuration >= durationRequirement.min && profile.diseaseDuration <= durationRequirement.max {
                eligibleCriteria.append("Disease duration requirement met")
            } else {
                ineligibleCriteria.append("Disease duration requirement not met")
            }
        }
        
        let overallEligibility: EligibilityStatus
        if !ineligibleCriteria.isEmpty {
            overallEligibility = .ineligible
        } else if !unknownCriteria.isEmpty {
            overallEligibility = .needsReview
        } else {
            overallEligibility = .eligible
        }
        
        return EligibilityAssessment(
            status: overallEligibility,
            eligibleCriteria: eligibleCriteria,
            ineligibleCriteria: ineligibleCriteria,
            unknownCriteria: unknownCriteria,
            matchScore: calculateMatchScore(eligible: eligibleCriteria.count, ineligible: ineligibleCriteria.count, unknown: unknownCriteria.count)
        )
    }
    
    private func checkBasicEligibility(trial: ClinicalTrial, profile: PatientProfile) -> Bool {
        // Basic eligibility check
        if let ageRange = trial.ageRange {
            if profile.age < ageRange.min || profile.age > ageRange.max {
                return false
            }
        }
        
        if let genderReq = trial.genderRequirement, genderReq != .any {
            if genderReq.rawValue != profile.gender {
                return false
            }
        }
        
        return true
    }
    
    private func calculateMatchScore(eligible: Int, ineligible: Int, unknown: Int) -> Double {
        let total = eligible + ineligible + unknown
        guard total > 0 else { return 0 }
        
        return Double(eligible) / Double(total)
    }
}

// MARK: - Supporting Types
struct MedicationInteraction {
    let id = UUID()
    let medication1: Medication
    let medication2: Medication
    let severity: InteractionSeverity
    let description: String
    let mechanism: String
    let recommendations: [String]
    let sources: [String]
}

enum InteractionSeverity: String, CaseIterable {
    case minor = "Minor"
    case moderate = "Moderate"
    case major = "Major"
    case contraindicated = "Contraindicated"
    
    var color: String {
        switch self {
        case .minor: return "green"
        case .moderate: return "yellow"
        case .major: return "orange"
        case .contraindicated: return "red"
        }
    }
}

struct SymptomScore {
    let id = UUID()
    let overallScore: Double
    let painScore: Double
    let stiffnessScore: Double
    let fatigueScore: Double
    let functionalScore: Double
    let timeframe: TimeInterval
    let entryCount: Int
    let calculatedDate: Date
    let trend: SymptomTrend
    let severity: SymptomSeverity
}

enum SymptomSeverity: String, CaseIterable {
    case mild = "Mild"
    case moderate = "Moderate"
    case severe = "Severe"
    case critical = "Critical"
}

struct SymptomTrend {
    let direction: TrendDirection
    let magnitude: Double
    let confidence: Double
}

enum TrendDirection: String {
    case increasing = "Increasing"
    case decreasing = "Decreasing"
    case stable = "Stable"
}

struct FlareRiskAssessment {
    let id = UUID()
    let riskLevel: RiskLevel
    let confidence: Double
    let factors: [RiskFactor]
    let recommendations: [String]
    let timeframe: String
    let calculatedDate: Date
}

enum RiskLevel: String, CaseIterable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
    case critical = "Critical"
}

struct RiskFactors {
    let factors: [RiskFactor]
    let overallRisk: RiskLevel
    let confidence: Double
}

enum RiskFactor: String, CaseIterable {
    case increasingPain = "Increasing Pain"
    case highStress = "High Stress"
    case poorSleep = "Poor Sleep"
    case medicationMissed = "Missed Medication"
    case weatherChanges = "Weather Changes"
    case increasedActivity = "Increased Activity"
    case infection = "Recent Infection"
    case hormonalChanges = "Hormonal Changes"
}

struct TreatmentEffectiveness {
    let id = UUID()
    let medication: Medication
    let effectivenessScore: Double
    let painReduction: Double
    let sideEffects: [String]
    let adherenceRate: Double
    let startDate: Date
    let evaluationPeriod: TimeInterval
    let recommendation: TreatmentRecommendation.RecommendationType
}

struct TreatmentComparison {
    let id = UUID()
    let medication1: Medication
    let medication2: Medication
    let effectiveness1: Double
    let effectiveness2: Double
    let sideEffects1: [String]
    let sideEffects2: [String]
    let recommendation: String
}

struct TreatmentRecommendation {
    let id = UUID()
    let type: RecommendationType
    let medication: String
    let dosage: String?
    let frequency: String?
    let duration: String?
    let rationale: String
    let evidenceLevel: EvidenceLevel
    let contraindications: [String]
    let monitoring: [String]
    
    enum RecommendationType: String, CaseIterable {
        case continue = "Continue Current Treatment"
        case adjust_dose = "Adjust Dosage"
        case add_medication = "Add Medication"
        case switch_medication = "Switch Medication"
        case consider_alternative = "Consider Alternative"
        case insufficient_data = "Insufficient Data"
    }
    
    enum EvidenceLevel: String, CaseIterable {
        case high = "High"
        case moderate = "Moderate"
        case low = "Low"
        case expert_opinion = "Expert Opinion"
    }
}

struct ClinicalTrial {
    let id: String
    let title: String
    let description: String
    let phase: TrialPhase
    let status: TrialStatus
    let condition: String
    let intervention: String
    let sponsor: String
    let locations: [TrialLocation]
    let ageRange: AgeRange?
    let genderRequirement: GenderRequirement?
    let diseaseDurationRequirement: DurationRange?
    let estimatedEnrollment: Int
    let startDate: Date?
    let completionDate: Date?
    let contactInfo: ContactInfo?
}

enum TrialPhase: String, CaseIterable {
    case phase1 = "Phase 1"
    case phase2 = "Phase 2"
    case phase3 = "Phase 3"
    case phase4 = "Phase 4"
    case notApplicable = "Not Applicable"
}

enum TrialStatus: String, CaseIterable {
    case recruiting = "Recruiting"
    case notYetRecruiting = "Not Yet Recruiting"
    case enrollingByInvitation = "Enrolling by Invitation"
    case active = "Active, Not Recruiting"
    case completed = "Completed"
    case suspended = "Suspended"
    case terminated = "Terminated"
    case withdrawn = "Withdrawn"
}

enum GenderRequirement: String, CaseIterable {
    case male = "Male"
    case female = "Female"
    case any = "All"
}

struct AgeRange {
    let min: Int
    let max: Int
}

struct DurationRange {
    let min: TimeInterval
    let max: TimeInterval
}

struct TrialLocation {
    let facility: String
    let city: String
    let state: String
    let country: String
    let zipCode: String?
    let coordinates: (latitude: Double, longitude: Double)?
}

struct ContactInfo {
    let name: String
    let phone: String?
    let email: String?
}

struct ClinicalTrialDetails {
    let trial: ClinicalTrial
    let detailedDescription: String
    let inclusionCriteria: [String]
    let exclusionCriteria: [String]
    let primaryOutcomes: [String]
    let secondaryOutcomes: [String]
    let eligibilityDetails: String
    let studyDesign: String
    let interventionDetails: String
}

struct EligibilityAssessment {
    let status: EligibilityStatus
    let eligibleCriteria: [String]
    let ineligibleCriteria: [String]
    let unknownCriteria: [String]
    let matchScore: Double
}

enum EligibilityStatus: String, CaseIterable {
    case eligible = "Eligible"
    case ineligible = "Ineligible"
    case needsReview = "Needs Review"
}

struct PatientProfile {
    let age: Int
    let gender: String
    let diseaseDuration: TimeInterval
    let currentMedications: [String]
    let allergies: [String]
    let comorbidities: [String]
    let previousTreatments: [String]
    let diseaseActivity: String
    let functionalStatus: String
    
    var ageRange: AgeRange {
        AgeRange(min: max(18, age - 5), max: age + 5)
    }
}

struct DrugInfo {
    let id: String
    let name: String
    let genericName: String?
    let brandNames: [String]
    let drugClass: String
    let mechanism: String
    let indications: [String]
    let contraindications: [String]
    let sideEffects: [String]
    let interactions: [String]
    let dosageInfo: DosageInfo
    let warnings: [String]
    let pregnancyCategory: String?
}

struct DosageInfo {
    let standardDose: String
    let maxDose: String
    let frequency: String
    let route: String
    let adjustments: [String]
}

struct HeartRateData {
    let value: Double
    let date: Date
}

struct SleepData {
    let value: Int
    let startDate: Date
    let endDate: Date
    let duration: TimeInterval
}

struct BloodPressureReading {
    let value: Double
    let date: Date
    let type: BloodPressureType
}

enum BloodPressureType {
    case systolic
    case diastolic
}

struct AllergyAssessment {
    let id = UUID()
    let medication: Medication
    let overallRisk: AllergyRiskLevel
    let allergyRisks: [AllergyRisk]
    let recommendations: [String]
}

struct AllergyRisk {
    let allergen: String
    let severity: AllergySeverity
    let description: String
}

enum AllergyRiskLevel: String, CaseIterable {
    case none = "No Risk"
    case low = "Low Risk"
    case moderate = "Moderate Risk"
    case high = "High Risk"
}

enum AllergySeverity: String, CaseIterable {
    case mild = "Mild"
    case moderate = "Moderate"
    case severe = "Severe"
}

struct DosingRecommendation {
    let id = UUID()
    let medication: Medication
    let recommendedDose: Double
    let frequency: String
    let route: String
    let adjustmentFactors: [String]
    let monitoringParameters: [String]
    let duration: String
}

struct LabResults {
    let creatinine: Double?
    let alt: Double?
    let ast: Double?
    let hemoglobin: Double?
    let whiteBloodCount: Double?
    let platelets: Double?
    let esr: Double?
    let crp: Double?
    let date: Date
}

extension PatientProfile {
    var weight: Double {
        // Default weight if not specified
        return 70.0 // kg
    }
}

extension Medication {
    var drugClass: String {
        // This would be determined from the drug database
        return "Unknown"
    }
    
    var requiresWeightBasedDosing: Bool {
        // Medications that require weight-based dosing
        let weightBasedMeds = ["methotrexate", "cyclophosphamide", "azathioprine"]
        return weightBasedMeds.contains(name.lowercased())
    }
}

enum MedicalError: LocalizedError {
    case healthKitUnavailable
    case invalidHealthKitType
    case drugDatabaseUnavailable
    case interactionCheckFailed
    case trialSearchFailed
    case insufficientData
    
    var errorDescription: String? {
        switch self {
        case .healthKitUnavailable:
            return "HealthKit is not available on this device"
        case .invalidHealthKitType:
            return "Invalid HealthKit data type"
        case .drugDatabaseUnavailable:
            return "Drug database is not available"
        case .interactionCheckFailed:
            return "Failed to check medication interactions"
        case .trialSearchFailed:
            return "Failed to search clinical trials"
        case .insufficientData:
            return "Insufficient data for analysis"
        }
    }
}

// MARK: - Database Classes (Simplified)
class DrugDatabase {
    static let shared = DrugDatabase()
    private var drugCache: [String: DrugInfo] = [:]
    private var interactionCache: [String: (severity: InteractionSeverity, description: String, mechanism: String, recommendations: [String], sources: [String])] = [:]
    
    private init() {
        loadDrugDatabase()
    }
    
    func getAllDrugs() -> [DrugInfo] {
        return Array(drugCache.values)
    }
    
    func getDrugInfo(name: String) -> DrugInfo? {
        return drugCache[name.lowercased()]
    }
    
    func getStandardDosing(drug: String) -> (standardDose: Double, frequency: String, route: String, duration: String) {
        // Return standard dosing information
        // This would be loaded from a real database
        return (standardDose: 1.0, frequency: "Once daily", route: "Oral", duration: "Ongoing")
    }
    
    func checkAllergyRisk(drug: String, allergy: String) -> AllergyRisk? {
        // Check for allergy cross-reactivity
        let key = "\(drug.lowercased())-\(allergy.lowercased())"
        
        // Common allergy cross-reactions for rheumatoid arthritis medications
        let allergyDatabase: [String: AllergyRisk] = [
            "methotrexate-sulfa": AllergyRisk(allergen: "Sulfa", severity: .moderate, description: "Potential cross-reactivity"),
            "sulfasalazine-sulfa": AllergyRisk(allergen: "Sulfa", severity: .severe, description: "Contains sulfa component"),
            "penicillamine-penicillin": AllergyRisk(allergen: "Penicillin", severity: .mild, description: "Different chemical structure but monitor"),
            "adalimumab-latex": AllergyRisk(allergen: "Latex", severity: .mild, description: "Injection device may contain latex")
        ]
        
        return allergyDatabase[key]
    }
    
    func getInteraction(drug1: String, drug2: String) -> (severity: InteractionSeverity, description: String, mechanism: String, recommendations: [String], sources: [String])? {
        let key = [drug1.lowercased(), drug2.lowercased()].sorted().joined(separator: "-")
        
        if let cached = interactionCache[key] {
            return cached
        }
        
        // Common drug interactions for rheumatoid arthritis medications
        let interactionDatabase: [String: (severity: InteractionSeverity, description: String, mechanism: String, recommendations: [String], sources: [String])] = [
            "methotrexate-warfarin": (
                severity: .major,
                description: "Increased risk of bleeding",
                mechanism: "Methotrexate displaces warfarin from protein binding",
                recommendations: ["Monitor INR closely", "Consider dose adjustment"],
                sources: ["Drug Interaction Database", "Clinical Studies"]
            ),
            "methotrexate-trimethoprim": (
                severity: .major,
                description: "Increased methotrexate toxicity",
                mechanism: "Both drugs inhibit folate metabolism",
                recommendations: ["Avoid combination", "Use alternative antibiotic"],
                sources: ["FDA Drug Safety Communication"]
            ),
            "prednisone-nsaids": (
                severity: .moderate,
                description: "Increased risk of GI bleeding",
                mechanism: "Additive effects on gastric mucosa",
                recommendations: ["Use gastroprotective agent", "Monitor for GI symptoms"],
                sources: ["Clinical Guidelines"]
            )
        ]
        
        let interaction = interactionDatabase[key]
        if let interaction = interaction {
            interactionCache[key] = interaction
        }
        
        return interaction
    }
    
    private func loadDrugDatabase() {
        // Load common rheumatoid arthritis medications
        let commonRADrugs = [
            DrugInfo(
                id: "mtx-001",
                name: "Methotrexate",
                genericName: "Methotrexate",
                brandNames: ["Rheumatrex", "Trexall"],
                drugClass: "DMARD",
                mechanism: "Folate antagonist",
                indications: ["Rheumatoid Arthritis", "Psoriatic Arthritis"],
                contraindications: ["Pregnancy", "Severe kidney disease", "Active infection"],
                sideEffects: ["Nausea", "Fatigue", "Liver toxicity", "Bone marrow suppression"],
                interactions: ["Warfarin", "Trimethoprim", "NSAIDs"],
                dosageInfo: DosageInfo(
                    standardDose: "7.5-25 mg weekly",
                    maxDose: "25 mg weekly",
                    frequency: "Once weekly",
                    route: "Oral or injection",
                    adjustments: ["Reduce dose in kidney impairment", "Monitor liver function"]
                ),
                warnings: ["Requires folic acid supplementation", "Regular blood monitoring required"],
                pregnancyCategory: "X"
            ),
            DrugInfo(
                id: "ada-001",
                name: "Adalimumab",
                genericName: "Adalimumab",
                brandNames: ["Humira"],
                drugClass: "Biologic",
                mechanism: "TNF-alpha inhibitor",
                indications: ["Rheumatoid Arthritis", "Psoriatic Arthritis", "Ankylosing Spondylitis"],
                contraindications: ["Active infection", "Live vaccines"],
                sideEffects: ["Injection site reactions", "Increased infection risk", "Headache"],
                interactions: ["Live vaccines", "Immunosuppressants"],
                dosageInfo: DosageInfo(
                    standardDose: "40 mg every other week",
                    maxDose: "40 mg weekly",
                    frequency: "Every other week",
                    route: "Subcutaneous injection",
                    adjustments: ["No dose adjustment for kidney/liver impairment"]
                ),
                warnings: ["Infection screening required", "TB screening required"],
                pregnancyCategory: "B"
            )
        ]
        
        for drug in commonRADrugs {
            drugCache[drug.name.lowercased()] = drug
            if let genericName = drug.genericName {
                drugCache[genericName.lowercased()] = drug
            }
            for brandName in drug.brandNames {
                drugCache[brandName.lowercased()] = drug
            }
        }
    }
}

class ClinicalTrialDatabase {
    static let shared = ClinicalTrialDatabase()
    private var trialCache: [String: ClinicalTrial] = [:]
    
    private init() {
        loadSampleTrials()
    }
    
    func searchTrials(condition: String, location: String?, ageRange: AgeRange, gender: String) async throws -> [ClinicalTrial] {
        // Simulate API delay
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        let allTrials = Array(trialCache.values)
        
        // Filter trials based on criteria
        let filteredTrials = allTrials.filter { trial in
            // Check condition
            guard trial.condition.localizedCaseInsensitiveContains(condition) else { return false }
            
            // Check age range
            if let trialAgeRange = trial.ageRange {
                guard ageRange.min <= trialAgeRange.max && ageRange.max >= trialAgeRange.min else { return false }
            }
            
            // Check gender requirement
            if let genderReq = trial.genderRequirement, genderReq != .any {
                guard genderReq.rawValue.lowercased() == gender.lowercased() else { return false }
            }
            
            // Check location if specified
            if let location = location {
                let hasMatchingLocation = trial.locations.contains { trialLocation in
                    trialLocation.city.localizedCaseInsensitiveContains(location) ||
                    trialLocation.state.localizedCaseInsensitiveContains(location) ||
                    trialLocation.country.localizedCaseInsensitiveContains(location)
                }
                guard hasMatchingLocation else { return false }
            }
            
            return true
        }
        
        return filteredTrials
    }
    
    func getTrialDetails(trialId: String) async throws -> ClinicalTrialDetails {
        // Simulate API delay
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        
        guard let trial = trialCache[trialId] else {
            throw MedicalError.trialSearchFailed
        }
        
        // Return detailed information
        return ClinicalTrialDetails(
            trial: trial,
            detailedDescription: "Detailed description of \(trial.title). This study aims to evaluate the safety and efficacy of the intervention in patients with \(trial.condition).",
            inclusionCriteria: [
                "Diagnosis of \(trial.condition)",
                "Age \(trial.ageRange?.min ?? 18) to \(trial.ageRange?.max ?? 65) years",
                "Stable disease for at least 3 months",
                "Adequate organ function"
            ],
            exclusionCriteria: [
                "Pregnancy or nursing",
                "Active infection",
                "Severe comorbidities",
                "Recent participation in another clinical trial"
            ],
            primaryOutcomes: [
                "Change in disease activity score",
                "Safety and tolerability"
            ],
            secondaryOutcomes: [
                "Quality of life measures",
                "Functional assessment",
                "Biomarker analysis"
            ],
            eligibilityDetails: "Participants must meet all inclusion criteria and none of the exclusion criteria. Detailed medical history and physical examination required.",
            studyDesign: "Randomized, double-blind, placebo-controlled trial",
            interventionDetails: "Participants will receive either the study drug or placebo for the duration of the study period."
        )
    }
    
    private func loadSampleTrials() {
        let sampleTrials = [
            ClinicalTrial(
                id: "NCT12345678",
                title: "Efficacy and Safety of Novel JAK Inhibitor in Rheumatoid Arthritis",
                description: "A Phase 3 study evaluating a new JAK inhibitor for moderate to severe rheumatoid arthritis",
                phase: .phase3,
                status: .recruiting,
                condition: "Rheumatoid Arthritis",
                intervention: "JAK Inhibitor vs Placebo",
                sponsor: "Pharmaceutical Research Institute",
                locations: [
                    TrialLocation(facility: "University Medical Center", city: "Boston", state: "MA", country: "USA", zipCode: "02115", coordinates: (42.3601, -71.0589)),
                    TrialLocation(facility: "Research Hospital", city: "New York", state: "NY", country: "USA", zipCode: "10021", coordinates: (40.7589, -73.9441))
                ],
                ageRange: AgeRange(min: 18, max: 75),
                genderRequirement: .any,
                diseaseDurationRequirement: DurationRange(min: 6*30*24*3600, max: 20*365*24*3600), // 6 months to 20 years
                estimatedEnrollment: 500,
                startDate: Calendar.current.date(byAdding: .month, value: -6, to: Date()),
                completionDate: Calendar.current.date(byAdding: .year, value: 2, to: Date()),
                contactInfo: ContactInfo(name: "Dr. Sarah Johnson", phone: "+1-617-555-0123", email: "sarah.johnson@research.edu")
            ),
            ClinicalTrial(
                id: "NCT87654321",
                title: "Personalized Medicine Approach for Rheumatoid Arthritis Treatment",
                description: "A study using genetic markers to personalize treatment selection",
                phase: .phase2,
                status: .recruiting,
                condition: "Rheumatoid Arthritis",
                intervention: "Personalized Treatment Algorithm",
                sponsor: "Genomic Medicine Consortium",
                locations: [
                    TrialLocation(facility: "Genomic Research Center", city: "San Francisco", state: "CA", country: "USA", zipCode: "94143", coordinates: (37.7749, -122.4194))
                ],
                ageRange: AgeRange(min: 21, max: 70),
                genderRequirement: .any,
                diseaseDurationRequirement: DurationRange(min: 3*30*24*3600, max: 15*365*24*3600), // 3 months to 15 years
                estimatedEnrollment: 200,
                startDate: Calendar.current.date(byAdding: .month, value: -3, to: Date()),
                completionDate: Calendar.current.date(byAdding: .year, value: 3, to: Date()),
                contactInfo: ContactInfo(name: "Dr. Michael Chen", phone: "+1-415-555-0456", email: "m.chen@genomics.org")
            )
        ]
        
        for trial in sampleTrials {
            trialCache[trial.id] = trial
        }
    }
}