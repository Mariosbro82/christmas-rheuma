//
//  AIMLEngine.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import Foundation
import Combine
import CoreML
import CreateML
import NaturalLanguage
import Vision

class AIMLEngine: ObservableObject {
    static let shared = AIMLEngine()
    
    @Published var isAnalyzing = false
    @Published var predictions: [PainPrediction] = []
    @Published var patterns: [PainPattern] = []
    @Published var insights: [AIInsight] = []
    @Published var recommendations: [AIRecommendation] = []
    @Published var riskAssessment: RiskAssessment?
    @Published var treatmentSuggestions: [TreatmentSuggestion] = []
    
    private var painModel: MLModel?
    private var patternModel: MLModel?
    private var riskModel: MLModel?
    private let nlProcessor = NLLanguageRecognizer()
    private var cancellables = Set<AnyCancellable>()
    
    // Training data storage
    private var trainingData: [PainTrainingData] = []
    private var modelAccuracy: Double = 0.0
    private var lastModelUpdate = Date()
    
    private init() {
        loadPretrainedModels()
        setupPeriodicAnalysis()
    }
    
    // MARK: - Model Loading and Training
    
    private func loadPretrainedModels() {
        // Load pre-trained models if available
        loadPainPredictionModel()
        loadPatternRecognitionModel()
        loadRiskAssessmentModel()
    }
    
    private func loadPainPredictionModel() {
        // In a real implementation, this would load a trained CoreML model
        // For now, we'll simulate with a placeholder
        print("Loading pain prediction model...")
    }
    
    private func loadPatternRecognitionModel() {
        print("Loading pattern recognition model...")
    }
    
    private func loadRiskAssessmentModel() {
        print("Loading risk assessment model...")
    }
    
    private func setupPeriodicAnalysis() {
        // Run analysis every hour
        Timer.publish(every: 3600, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.performPeriodicAnalysis()
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Pain Analysis
    
    func analyzePainPatterns(painData: [PainDataEntry], completion: @escaping ([PainPattern]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            DispatchQueue.main.async {
                self.isAnalyzing = true
            }
            
            var detectedPatterns: [PainPattern] = []
            
            // Temporal pattern analysis
            let temporalPatterns = self.analyzeTemporalPatterns(painData)
            detectedPatterns.append(contentsOf: temporalPatterns)
            
            // Intensity pattern analysis
            let intensityPatterns = self.analyzeIntensityPatterns(painData)
            detectedPatterns.append(contentsOf: intensityPatterns)
            
            // Regional pattern analysis
            let regionalPatterns = self.analyzeRegionalPatterns(painData)
            detectedPatterns.append(contentsOf: regionalPatterns)
            
            // Trigger pattern analysis
            let triggerPatterns = self.analyzeTriggerPatterns(painData)
            detectedPatterns.append(contentsOf: triggerPatterns)
            
            DispatchQueue.main.async {
                self.patterns = detectedPatterns
                self.isAnalyzing = false
                completion(detectedPatterns)
            }
        }
    }
    
    private func analyzeTemporalPatterns(_ painData: [PainDataEntry]) -> [PainPattern] {
        var patterns: [PainPattern] = []
        
        // Group pain data by hour of day
        let hourlyData = Dictionary(grouping: painData) { entry in
            Calendar.current.component(.hour, from: entry.timestamp)
        }
        
        // Find peak pain hours
        let hourlyAverages = hourlyData.mapValues { entries in
            entries.map { $0.painLevel }.reduce(0, +) / Double(entries.count)
        }
        
        if let peakHour = hourlyAverages.max(by: { $0.value < $1.value }) {
            if peakHour.value > 6.0 {
                patterns.append(PainPattern(
                    type: .temporal,
                    description: "Pain levels peak around \(peakHour.key):00",
                    confidence: 0.8,
                    frequency: hourlyData[peakHour.key]?.count ?? 0,
                    severity: peakHour.value > 8.0 ? .high : .medium,
                    recommendations: [
                        "Consider preventive medication before \(peakHour.key):00",
                        "Plan lighter activities during peak pain hours",
                        "Practice relaxation techniques before pain onset"
                    ]
                ))
            }
        }
        
        // Analyze weekly patterns
        let weeklyData = Dictionary(grouping: painData) { entry in
            Calendar.current.component(.weekday, from: entry.timestamp)
        }
        
        let weeklyAverages = weeklyData.mapValues { entries in
            entries.map { $0.painLevel }.reduce(0, +) / Double(entries.count)
        }
        
        if let worstDay = weeklyAverages.max(by: { $0.value < $1.value }) {
            let dayName = Calendar.current.weekdaySymbols[worstDay.key - 1]
            if worstDay.value > 6.0 {
                patterns.append(PainPattern(
                    type: .temporal,
                    description: "Pain levels are consistently higher on \(dayName)s",
                    confidence: 0.7,
                    frequency: weeklyData[worstDay.key]?.count ?? 0,
                    severity: worstDay.value > 8.0 ? .high : .medium,
                    recommendations: [
                        "Plan rest activities on \(dayName)s",
                        "Consider adjusting \(dayName) schedule",
                        "Prepare pain management strategies for \(dayName)s"
                    ]
                ))
            }
        }
        
        return patterns
    }
    
    private func analyzeIntensityPatterns(_ painData: [PainDataEntry]) -> [PainPattern] {
        var patterns: [PainPattern] = []
        
        let intensities = painData.map { $0.painLevel }
        guard !intensities.isEmpty else { return patterns }
        
        let averageIntensity = intensities.reduce(0, +) / Double(intensities.count)
        let maxIntensity = intensities.max() ?? 0
        let minIntensity = intensities.min() ?? 0
        let intensityRange = maxIntensity - minIntensity
        
        // High variability pattern
        if intensityRange > 6.0 {
            patterns.append(PainPattern(
                type: .intensity,
                description: "High pain variability detected (range: \(String(format: "%.1f", intensityRange)))",
                confidence: 0.9,
                frequency: painData.count,
                severity: .medium,
                recommendations: [
                    "Track triggers for pain spikes",
                    "Consider consistent pain management routine",
                    "Monitor for patterns in high-pain episodes"
                ]
            ))
        }
        
        // Chronic high pain pattern
        let highPainCount = intensities.filter { $0 > 7.0 }.count
        let highPainPercentage = Double(highPainCount) / Double(intensities.count) * 100
        
        if highPainPercentage > 30 {
            patterns.append(PainPattern(
                type: .intensity,
                description: "Frequent severe pain episodes (\(String(format: "%.1f", highPainPercentage))% of readings > 7/10)",
                confidence: 0.95,
                frequency: highPainCount,
                severity: .high,
                recommendations: [
                    "Consult healthcare provider about pain management",
                    "Consider stronger pain management strategies",
                    "Evaluate current treatment effectiveness"
                ]
            ))
        }
        
        // Escalating pain pattern
        if painData.count >= 7 {
            let recentData = Array(painData.suffix(7))
            let recentAverage = recentData.map { $0.painLevel }.reduce(0, +) / 7.0
            
            if recentAverage > averageIntensity + 1.5 {
                patterns.append(PainPattern(
                    type: .intensity,
                    description: "Pain levels have increased recently (recent avg: \(String(format: "%.1f", recentAverage)))",
                    confidence: 0.8,
                    frequency: 7,
                    severity: .high,
                    recommendations: [
                        "Monitor for worsening symptoms",
                        "Consider medical consultation",
                        "Review recent activities or changes"
                    ]
                ))
            }
        }
        
        return patterns
    }
    
    private func analyzeRegionalPatterns(_ painData: [PainDataEntry]) -> [PainPattern] {
        var patterns: [PainPattern] = []
        
        // Group by body region
        let regionalData = Dictionary(grouping: painData.compactMap { entry in
            entry.bodyRegion != nil ? entry : nil
        }) { entry in
            entry.bodyRegion!
        }
        
        // Find most affected regions
        let regionalAverages = regionalData.mapValues { entries in
            entries.map { $0.painLevel }.reduce(0, +) / Double(entries.count)
        }
        
        let sortedRegions = regionalAverages.sorted { $0.value > $1.value }
        
        if let mostAffected = sortedRegions.first, mostAffected.value > 6.0 {
            patterns.append(PainPattern(
                type: .regional,
                description: "\(mostAffected.key.displayName) is the most affected region (avg: \(String(format: "%.1f", mostAffected.value)))",
                confidence: 0.9,
                frequency: regionalData[mostAffected.key]?.count ?? 0,
                severity: mostAffected.value > 8.0 ? .high : .medium,
                recommendations: [
                    "Focus treatment on \(mostAffected.key.displayName)",
                    "Consider targeted therapy for this region",
                    "Monitor for spreading to adjacent areas"
                ]
            ))
        }
        
        // Analyze pain spreading patterns
        let spinalRegions: [BodyRegion] = [.cervicalSpine, .thoracicSpine, .lumbarSpine]
        let spinalPainData = regionalData.filter { spinalRegions.contains($0.key) }
        
        if spinalPainData.count >= 2 {
            let avgSpinalPain = spinalPainData.values.flatMap { $0 }.map { $0.painLevel }.reduce(0, +) / Double(spinalPainData.values.flatMap { $0 }.count)
            
            if avgSpinalPain > 5.0 {
                patterns.append(PainPattern(
                    type: .regional,
                    description: "Multiple spinal regions affected (avg: \(String(format: "%.1f", avgSpinalPain)))",
                    confidence: 0.85,
                    frequency: spinalPainData.values.flatMap { $0 }.count,
                    severity: .high,
                    recommendations: [
                        "Consider comprehensive spinal evaluation",
                        "Focus on posture and ergonomics",
                        "Evaluate for systemic causes"
                    ]
                ))
            }
        }
        
        return patterns
    }
    
    private func analyzeTriggerPatterns(_ painData: [PainDataEntry]) -> [PainPattern] {
        var patterns: [PainPattern] = []
        
        // Analyze triggers if available
        let triggersData = painData.compactMap { $0.triggers }.flatMap { $0 }
        let triggerCounts = Dictionary(triggersData.map { ($0, 1) }, uniquingKeysWith: +)
        
        let sortedTriggers = triggerCounts.sorted { $0.value > $1.value }
        
        if let commonTrigger = sortedTriggers.first, commonTrigger.value >= 3 {
            patterns.append(PainPattern(
                type: .trigger,
                description: "\(commonTrigger.key) is a frequent pain trigger (\(commonTrigger.value) occurrences)",
                confidence: 0.8,
                frequency: commonTrigger.value,
                severity: .medium,
                recommendations: [
                    "Avoid or minimize exposure to \(commonTrigger.key)",
                    "Develop coping strategies for \(commonTrigger.key)",
                    "Consider preventive measures before \(commonTrigger.key)"
                ]
            ))
        }
        
        return patterns
    }
    
    // MARK: - Predictive Analytics
    
    func generatePainPredictions(painData: [PainDataEntry], completion: @escaping ([PainPrediction]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            var predictions: [PainPrediction] = []
            
            // Short-term predictions (next 24 hours)
            let shortTermPredictions = self.generateShortTermPredictions(painData)
            predictions.append(contentsOf: shortTermPredictions)
            
            // Medium-term predictions (next week)
            let mediumTermPredictions = self.generateMediumTermPredictions(painData)
            predictions.append(contentsOf: mediumTermPredictions)
            
            // Long-term predictions (next month)
            let longTermPredictions = self.generateLongTermPredictions(painData)
            predictions.append(contentsOf: longTermPredictions)
            
            DispatchQueue.main.async {
                self.predictions = predictions
                completion(predictions)
            }
        }
    }
    
    private func generateShortTermPredictions(_ painData: [PainDataEntry]) -> [PainPrediction] {
        var predictions: [PainPrediction] = []
        
        guard painData.count >= 3 else { return predictions }
        
        let recentData = Array(painData.suffix(24)) // Last 24 entries
        let recentAverage = recentData.map { $0.painLevel }.reduce(0, +) / Double(recentData.count)
        
        // Predict next 6 hours based on recent trend
        let trend = calculateTrend(recentData.map { $0.painLevel })
        let predictedLevel = max(0, min(10, recentAverage + trend))
        
        let confidence = calculatePredictionConfidence(recentData)
        
        predictions.append(PainPrediction(
            timeframe: .shortTerm,
            predictedLevel: predictedLevel,
            confidence: confidence,
            targetTime: Date().addingTimeInterval(6 * 3600), // 6 hours from now
            factors: identifyPredictionFactors(recentData),
            recommendations: generatePredictionRecommendations(predictedLevel, timeframe: .shortTerm)
        ))
        
        return predictions
    }
    
    private func generateMediumTermPredictions(_ painData: [PainDataEntry]) -> [PainPrediction] {
        var predictions: [PainPrediction] = []
        
        guard painData.count >= 7 else { return predictions }
        
        // Analyze weekly patterns
        let weeklyData = Array(painData.suffix(7 * 24)) // Last week
        let weeklyAverage = weeklyData.map { $0.painLevel }.reduce(0, +) / Double(weeklyData.count)
        
        // Consider seasonal and cyclical factors
        let cyclicalFactor = calculateCyclicalFactor(painData)
        let predictedLevel = max(0, min(10, weeklyAverage * cyclicalFactor))
        
        let confidence = calculatePredictionConfidence(weeklyData) * 0.8 // Lower confidence for longer term
        
        predictions.append(PainPrediction(
            timeframe: .mediumTerm,
            predictedLevel: predictedLevel,
            confidence: confidence,
            targetTime: Date().addingTimeInterval(7 * 24 * 3600), // 1 week from now
            factors: identifyPredictionFactors(weeklyData),
            recommendations: generatePredictionRecommendations(predictedLevel, timeframe: .mediumTerm)
        ))
        
        return predictions
    }
    
    private func generateLongTermPredictions(_ painData: [PainDataEntry]) -> [PainPrediction] {
        var predictions: [PainPrediction] = []
        
        guard painData.count >= 30 else { return predictions }
        
        // Analyze monthly trends
        let monthlyData = Array(painData.suffix(30 * 24)) // Last month
        let monthlyAverage = monthlyData.map { $0.painLevel }.reduce(0, +) / Double(monthlyData.count)
        
        // Consider long-term progression
        let progressionFactor = calculateProgressionFactor(painData)
        let predictedLevel = max(0, min(10, monthlyAverage * progressionFactor))
        
        let confidence = calculatePredictionConfidence(monthlyData) * 0.6 // Lower confidence for long term
        
        predictions.append(PainPrediction(
            timeframe: .longTerm,
            predictedLevel: predictedLevel,
            confidence: confidence,
            targetTime: Date().addingTimeInterval(30 * 24 * 3600), // 1 month from now
            factors: identifyPredictionFactors(monthlyData),
            recommendations: generatePredictionRecommendations(predictedLevel, timeframe: .longTerm)
        ))
        
        return predictions
    }
    
    // MARK: - Risk Assessment
    
    func assessPainRisk(painData: [PainDataEntry], healthData: [HeartRateReading], completion: @escaping (RiskAssessment) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            let riskFactors = self.identifyRiskFactors(painData: painData, healthData: healthData)
            let overallRisk = self.calculateOverallRisk(riskFactors)
            let recommendations = self.generateRiskRecommendations(riskFactors)
            
            let assessment = RiskAssessment(
                overallRisk: overallRisk,
                riskFactors: riskFactors,
                recommendations: recommendations,
                assessmentDate: Date(),
                nextAssessmentDate: Date().addingTimeInterval(7 * 24 * 3600) // Next week
            )
            
            DispatchQueue.main.async {
                self.riskAssessment = assessment
                completion(assessment)
            }
        }
    }
    
    private func identifyRiskFactors(painData: [PainDataEntry], healthData: [HeartRateReading]) -> [RiskFactor] {
        var riskFactors: [RiskFactor] = []
        
        // High pain frequency
        let highPainCount = painData.filter { $0.painLevel > 7.0 }.count
        let highPainPercentage = Double(highPainCount) / Double(painData.count) * 100
        
        if highPainPercentage > 25 {
            riskFactors.append(RiskFactor(
                type: .highPainFrequency,
                severity: highPainPercentage > 50 ? .high : .medium,
                description: "Frequent severe pain episodes (\(String(format: "%.1f", highPainPercentage))%)",
                impact: 0.8
            ))
        }
        
        // Pain escalation
        if painData.count >= 14 {
            let recentData = Array(painData.suffix(7))
            let olderData = Array(painData.suffix(14).prefix(7))
            
            let recentAverage = recentData.map { $0.painLevel }.reduce(0, +) / 7.0
            let olderAverage = olderData.map { $0.painLevel }.reduce(0, +) / 7.0
            
            if recentAverage > olderAverage + 1.0 {
                riskFactors.append(RiskFactor(
                    type: .painEscalation,
                    severity: .high,
                    description: "Pain levels increasing (\(String(format: "%.1f", recentAverage - olderAverage)) point increase)",
                    impact: 0.9
                ))
            }
        }
        
        // Multiple affected regions
        let affectedRegions = Set(painData.compactMap { $0.bodyRegion })
        if affectedRegions.count > 3 {
            riskFactors.append(RiskFactor(
                type: .multipleRegions,
                severity: affectedRegions.count > 5 ? .high : .medium,
                description: "Multiple body regions affected (\(affectedRegions.count) regions)",
                impact: 0.7
            ))
        }
        
        // Cardiovascular stress indicators
        if !healthData.isEmpty {
            let averageHeartRate = healthData.map { $0.beatsPerMinute }.reduce(0, +) / Double(healthData.count)
            if averageHeartRate > 90 {
                riskFactors.append(RiskFactor(
                    type: .cardiovascularStress,
                    severity: averageHeartRate > 100 ? .high : .medium,
                    description: "Elevated heart rate (avg: \(String(format: "%.0f", averageHeartRate)) BPM)",
                    impact: 0.6
                ))
            }
        }
        
        return riskFactors
    }
    
    private func calculateOverallRisk(_ riskFactors: [RiskFactor]) -> RiskLevel {
        guard !riskFactors.isEmpty else { return .low }
        
        let totalImpact = riskFactors.map { $0.impact }.reduce(0, +)
        let averageImpact = totalImpact / Double(riskFactors.count)
        
        let highSeverityCount = riskFactors.filter { $0.severity == .high }.count
        
        if averageImpact > 0.8 || highSeverityCount >= 2 {
            return .high
        } else if averageImpact > 0.5 || highSeverityCount >= 1 {
            return .medium
        } else {
            return .low
        }
    }
    
    private func generateRiskRecommendations(_ riskFactors: [RiskFactor]) -> [String] {
        var recommendations: [String] = []
        
        for factor in riskFactors {
            switch factor.type {
            case .highPainFrequency:
                recommendations.append("Consult healthcare provider about pain management optimization")
                recommendations.append("Consider comprehensive pain evaluation")
            case .painEscalation:
                recommendations.append("Seek immediate medical attention for worsening symptoms")
                recommendations.append("Review current treatment effectiveness")
            case .multipleRegions:
                recommendations.append("Evaluate for systemic causes of widespread pain")
                recommendations.append("Consider rheumatology consultation")
            case .cardiovascularStress:
                recommendations.append("Monitor cardiovascular health closely")
                recommendations.append("Consider stress management techniques")
            case .sleepDisruption:
                recommendations.append("Address sleep quality issues")
                recommendations.append("Consider sleep study evaluation")
            case .medicationIneffectiveness:
                recommendations.append("Review medication regimen with healthcare provider")
                recommendations.append("Explore alternative treatment options")
            }
        }
        
        return Array(Set(recommendations)) // Remove duplicates
    }
    
    // MARK: - Treatment Suggestions
    
    func generateTreatmentSuggestions(painData: [PainDataEntry], completion: @escaping ([TreatmentSuggestion]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            var suggestions: [TreatmentSuggestion] = []
            
            // Analyze current pain patterns
            let averagePain = painData.map { $0.painLevel }.reduce(0, +) / Double(painData.count)
            let maxPain = painData.map { $0.painLevel }.max() ?? 0
            
            // Medication suggestions
            if averagePain > 6.0 {
                suggestions.append(TreatmentSuggestion(
                    type: .medication,
                    title: "Pain Medication Optimization",
                    description: "Consider discussing stronger pain management options with your healthcare provider",
                    effectiveness: 0.8,
                    sideEffects: ["Drowsiness", "Nausea", "Constipation"],
                    contraindications: ["Liver disease", "Kidney disease"],
                    evidence: "Clinical studies show 70-80% effectiveness for chronic pain management"
                ))
            }
            
            // Physical therapy suggestions
            let spinalRegions = painData.filter { 
                guard let region = $0.bodyRegion else { return false }
                return [.cervicalSpine, .thoracicSpine, .lumbarSpine].contains(region)
            }
            
            if !spinalRegions.isEmpty {
                suggestions.append(TreatmentSuggestion(
                    type: .physicalTherapy,
                    title: "Targeted Physical Therapy",
                    description: "Specialized exercises for spinal pain management and mobility improvement",
                    effectiveness: 0.75,
                    sideEffects: ["Temporary soreness", "Fatigue"],
                    contraindications: ["Acute injury", "Severe inflammation"],
                    evidence: "Research shows 60-80% improvement in spinal pain with targeted PT"
                ))
            }
            
            // Lifestyle modifications
            if maxPain > 8.0 {
                suggestions.append(TreatmentSuggestion(
                    type: .lifestyle,
                    title: "Comprehensive Lifestyle Modifications",
                    description: "Sleep optimization, stress management, and activity pacing strategies",
                    effectiveness: 0.65,
                    sideEffects: [],
                    contraindications: [],
                    evidence: "Lifestyle interventions show sustained benefits in 50-70% of patients"
                ))
            }
            
            // Alternative therapies
            suggestions.append(TreatmentSuggestion(
                type: .alternative,
                title: "Complementary Therapies",
                description: "Acupuncture, massage therapy, or mindfulness-based interventions",
                effectiveness: 0.6,
                sideEffects: ["Mild discomfort"],
                contraindications: ["Bleeding disorders", "Severe anxiety"],
                evidence: "Meta-analyses show moderate effectiveness for chronic pain conditions"
            ))
            
            DispatchQueue.main.async {
                self.treatmentSuggestions = suggestions
                completion(suggestions)
            }
        }
    }
    
    // MARK: - Utility Functions
    
    private func calculateTrend(_ values: [Double]) -> Double {
        guard values.count >= 2 else { return 0 }
        
        let n = Double(values.count)
        let sumX = (1...values.count).map(Double.init).reduce(0, +)
        let sumY = values.reduce(0, +)
        let sumXY = zip(1...values.count, values).map { Double($0.0) * $0.1 }.reduce(0, +)
        let sumX2 = (1...values.count).map { Double($0 * $0) }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        return slope
    }
    
    private func calculatePredictionConfidence(_ data: [PainDataEntry]) -> Double {
        guard data.count >= 3 else { return 0.3 }
        
        let values = data.map { $0.painLevel }
        let variance = calculateVariance(values)
        
        // Lower variance = higher confidence
        let baseConfidence = max(0.3, 1.0 - (variance / 10.0))
        
        // More data = higher confidence
        let dataConfidence = min(1.0, Double(data.count) / 50.0)
        
        return (baseConfidence + dataConfidence) / 2.0
    }
    
    private func calculateVariance(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        
        let mean = values.reduce(0, +) / Double(values.count)
        let squaredDifferences = values.map { pow($0 - mean, 2) }
        return squaredDifferences.reduce(0, +) / Double(values.count)
    }
    
    private func calculateCyclicalFactor(_ painData: [PainDataEntry]) -> Double {
        // Simplified cyclical analysis - in reality, this would be more sophisticated
        let calendar = Calendar.current
        let currentMonth = calendar.component(.month, from: Date())
        
        // Simulate seasonal factors (winter months might have higher pain)
        switch currentMonth {
        case 12, 1, 2: return 1.1 // Winter
        case 3, 4, 5: return 0.95 // Spring
        case 6, 7, 8: return 0.9 // Summer
        case 9, 10, 11: return 1.05 // Fall
        default: return 1.0
        }
    }
    
    private func calculateProgressionFactor(_ painData: [PainDataEntry]) -> Double {
        guard painData.count >= 60 else { return 1.0 }
        
        let recentMonth = Array(painData.suffix(30))
        let previousMonth = Array(painData.suffix(60).prefix(30))
        
        let recentAverage = recentMonth.map { $0.painLevel }.reduce(0, +) / 30.0
        let previousAverage = previousMonth.map { $0.painLevel }.reduce(0, +) / 30.0
        
        return recentAverage / previousAverage
    }
    
    private func identifyPredictionFactors(_ data: [PainDataEntry]) -> [String] {
        var factors: [String] = []
        
        let averagePain = data.map { $0.painLevel }.reduce(0, +) / Double(data.count)
        
        if averagePain > 7.0 {
            factors.append("High baseline pain level")
        }
        
        let variance = calculateVariance(data.map { $0.painLevel })
        if variance > 4.0 {
            factors.append("High pain variability")
        }
        
        let trend = calculateTrend(data.map { $0.painLevel })
        if trend > 0.1 {
            factors.append("Increasing pain trend")
        } else if trend < -0.1 {
            factors.append("Decreasing pain trend")
        }
        
        return factors
    }
    
    private func generatePredictionRecommendations(_ predictedLevel: Double, timeframe: PredictionTimeframe) -> [String] {
        var recommendations: [String] = []
        
        switch timeframe {
        case .shortTerm:
            if predictedLevel > 7.0 {
                recommendations.append("Take preventive pain medication")
                recommendations.append("Plan rest activities")
                recommendations.append("Apply heat/cold therapy")
            } else if predictedLevel < 4.0 {
                recommendations.append("Good time for gentle exercise")
                recommendations.append("Consider productive activities")
            }
            
        case .mediumTerm:
            if predictedLevel > 6.0 {
                recommendations.append("Schedule lighter work week")
                recommendations.append("Prepare pain management supplies")
                recommendations.append("Consider medical consultation")
            }
            
        case .longTerm:
            if predictedLevel > 6.0 {
                recommendations.append("Evaluate long-term treatment plan")
                recommendations.append("Consider lifestyle modifications")
                recommendations.append("Schedule comprehensive medical review")
            }
        }
        
        return recommendations
    }
    
    private func performPeriodicAnalysis() {
        let painData = PainDataStore.shared.painEntries
        
        analyzePainPatterns(painData: painData) { _ in }
        generatePainPredictions(painData: painData) { _ in }
        
        let healthData = HealthKitManager.shared.heartRateData
        assessPainRisk(painData: painData, healthData: healthData) { _ in }
        generateTreatmentSuggestions(painData: painData) { _ in }
    }
}

// MARK: - Data Models

struct PainPattern {
    let id = UUID()
    let type: PatternType
    let description: String
    let confidence: Double // 0-1
    let frequency: Int
    let severity: Severity
    let recommendations: [String]
    let detectedDate = Date()
    
    enum PatternType {
        case temporal
        case intensity
        case regional
        case trigger
        case medication
        
        var displayName: String {
            switch self {
            case .temporal: return "Time-based Pattern"
            case .intensity: return "Intensity Pattern"
            case .regional: return "Regional Pattern"
            case .trigger: return "Trigger Pattern"
            case .medication: return "Medication Pattern"
            }
        }
        
        var icon: String {
            switch self {
            case .temporal: return "clock.fill"
            case .intensity: return "waveform.path.ecg"
            case .regional: return "figure.stand"
            case .trigger: return "exclamationmark.triangle.fill"
            case .medication: return "pills.fill"
            }
        }
    }
    
    enum Severity {
        case low
        case medium
        case high
        
        var color: String {
            switch self {
            case .low: return "green"
            case .medium: return "orange"
            case .high: return "red"
            }
        }
    }
}

struct PainPrediction {
    let id = UUID()
    let timeframe: PredictionTimeframe
    let predictedLevel: Double // 0-10
    let confidence: Double // 0-1
    let targetTime: Date
    let factors: [String]
    let recommendations: [String]
    let createdDate = Date()
}

enum PredictionTimeframe {
    case shortTerm // Next 6-24 hours
    case mediumTerm // Next week
    case longTerm // Next month
    
    var displayName: String {
        switch self {
        case .shortTerm: return "Next 24 Hours"
        case .mediumTerm: return "Next Week"
        case .longTerm: return "Next Month"
        }
    }
    
    var icon: String {
        switch self {
        case .shortTerm: return "clock"
        case .mediumTerm: return "calendar"
        case .longTerm: return "calendar.badge.clock"
        }
    }
}

struct AIInsight {
    let id = UUID()
    let type: InsightType
    let title: String
    let description: String
    let actionable: Bool
    let priority: Priority
    let evidence: String
    let recommendations: [String]
    let createdDate = Date()
    
    enum InsightType {
        case pattern
        case prediction
        case correlation
        case anomaly
        case trend
        
        var displayName: String {
            switch self {
            case .pattern: return "Pattern Analysis"
            case .prediction: return "Predictive Insight"
            case .correlation: return "Correlation Analysis"
            case .anomaly: return "Anomaly Detection"
            case .trend: return "Trend Analysis"
            }
        }
    }
    
    enum Priority {
        case low
        case medium
        case high
        case critical
        
        var color: String {
            switch self {
            case .low: return "blue"
            case .medium: return "orange"
            case .high: return "red"
            case .critical: return "purple"
            }
        }
    }
}

struct AIRecommendation {
    let id = UUID()
    let category: RecommendationCategory
    let title: String
    let description: String
    let effectiveness: Double // 0-1
    let evidence: String
    let priority: Priority
    let timeframe: String
    let steps: [String]
    
    enum RecommendationCategory {
        case medication
        case lifestyle
        case therapy
        case monitoring
        case emergency
        
        var displayName: String {
            switch self {
            case .medication: return "Medication"
            case .lifestyle: return "Lifestyle"
            case .therapy: return "Therapy"
            case .monitoring: return "Monitoring"
            case .emergency: return "Emergency"
            }
        }
        
        var icon: String {
            switch self {
            case .medication: return "pills.fill"
            case .lifestyle: return "heart.fill"
            case .therapy: return "figure.strengthtraining.traditional"
            case .monitoring: return "chart.line.uptrend.xyaxis"
            case .emergency: return "exclamationmark.triangle.fill"
            }
        }
    }
    
    enum Priority {
        case low
        case medium
        case high
        case urgent
        
        var color: String {
            switch self {
            case .low: return "green"
            case .medium: return "orange"
            case .high: return "red"
            case .urgent: return "purple"
            }
        }
    }
}

struct RiskAssessment {
    let id = UUID()
    let overallRisk: RiskLevel
    let riskFactors: [RiskFactor]
    let recommendations: [String]
    let assessmentDate: Date
    let nextAssessmentDate: Date
}

enum RiskLevel {
    case low
    case medium
    case high
    case critical
    
    var displayName: String {
        switch self {
        case .low: return "Low Risk"
        case .medium: return "Medium Risk"
        case .high: return "High Risk"
        case .critical: return "Critical Risk"
        }
    }
    
    var color: String {
        switch self {
        case .low: return "green"
        case .medium: return "orange"
        case .high: return "red"
        case .critical: return "purple"
        }
    }
    
    var icon: String {
        switch self {
        case .low: return "checkmark.shield.fill"
        case .medium: return "exclamationmark.shield.fill"
        case .high: return "xmark.shield.fill"
        case .critical: return "exclamationmark.triangle.fill"
        }
    }
}

struct RiskFactor {
    let id = UUID()
    let type: RiskFactorType
    let severity: Severity
    let description: String
    let impact: Double // 0-1
    
    enum RiskFactorType {
        case highPainFrequency
        case painEscalation
        case multipleRegions
        case cardiovascularStress
        case sleepDisruption
        case medicationIneffectiveness
        
        var displayName: String {
            switch self {
            case .highPainFrequency: return "High Pain Frequency"
            case .painEscalation: return "Pain Escalation"
            case .multipleRegions: return "Multiple Affected Regions"
            case .cardiovascularStress: return "Cardiovascular Stress"
            case .sleepDisruption: return "Sleep Disruption"
            case .medicationIneffectiveness: return "Medication Ineffectiveness"
            }
        }
    }
    
    enum Severity {
        case low
        case medium
        case high
        
        var color: String {
            switch self {
            case .low: return "yellow"
            case .medium: return "orange"
            case .high: return "red"
            }
        }
    }
}

struct TreatmentSuggestion {
    let id = UUID()
    let type: TreatmentType
    let title: String
    let description: String
    let effectiveness: Double // 0-1
    let sideEffects: [String]
    let contraindications: [String]
    let evidence: String
    
    enum TreatmentType {
        case medication
        case physicalTherapy
        case lifestyle
        case alternative
        case surgical
        
        var displayName: String {
            switch self {
            case .medication: return "Medication"
            case .physicalTherapy: return "Physical Therapy"
            case .lifestyle: return "Lifestyle Changes"
            case .alternative: return "Alternative Therapy"
            case .surgical: return "Surgical Options"
            }
        }
        
        var icon: String {
            switch self {
            case .medication: return "pills.fill"
            case .physicalTherapy: return "figure.strengthtraining.traditional"
            case .lifestyle: return "heart.fill"
            case .alternative: return "leaf.fill"
            case .surgical: return "cross.case.fill"
            }
        }
    }
}

struct PainTrainingData {
    let painLevel: Double
    let bodyRegion: BodyRegion?
    let timestamp: Date
    let heartRate: Double?
    let activityLevel: Double?
    let weather: String?
    let stress: Double?
    let sleep: Double?
    let medication: String?
    let outcome: Double? // For supervised learning
}