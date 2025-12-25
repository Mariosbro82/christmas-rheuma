//
//  PredictiveAnalyticsEngine.swift
//  InflamAI-Swift
//
//  Advanced AI-powered predictive analytics for rheumatic disease management
//

import Foundation
import CoreData
import CoreML
import NaturalLanguage
import Accelerate

// MARK: - Prediction Models
struct FlareUpPrediction {
    let probability: Double
    let confidence: Double
    let timeframe: TimeInterval
    let contributingFactors: [String]
    let recommendedActions: [String]
    let severity: FlareUpSeverity
}

enum FlareUpSeverity: String, CaseIterable {
    case mild = "Mild"
    case moderate = "Moderate"
    case severe = "Severe"
    case critical = "Critical"
}

struct TreatmentRecommendation {
    let medicationAdjustments: [MedicationAdjustment]
    let lifestyleChanges: [LifestyleRecommendation]
    let confidence: Double
    let reasoning: String
}

struct MedicationAdjustment {
    let medicationName: String
    let currentDosage: String
    let recommendedDosage: String
    let reason: String
}

struct LifestyleRecommendation {
    let category: String
    let recommendation: String
    let priority: Int
    let expectedBenefit: String
}

// MARK: - Advanced Analytics Engine
class PredictiveAnalyticsEngine: ObservableObject {
    private let context: NSManagedObjectContext
    private let sentimentAnalyzer = NLSentimentPredictor()
    private var mlModel: MLModel?
    
    // Time series analysis parameters
    private let windowSize = 14 // days
    private let predictionHorizon = 7 // days ahead
    
    init(context: NSManagedObjectContext) {
        self.context = context
        loadMLModel()
    }
    
    // MARK: - Core Prediction Functions
    
    func predictFlareUp() async -> FlareUpPrediction {
        let healthData = await gatherHealthData()
        let features = extractFeatures(from: healthData)
        
        // Advanced time series analysis
        let trendAnalysis = performTrendAnalysis(features)
        let seasonalPatterns = analyzeSeasonalPatterns(features)
        let correlationMatrix = calculateCorrelationMatrix(features)
        
        // Machine learning prediction
        let mlPrediction = await performMLPrediction(features: features)
        
        // Ensemble prediction combining multiple models
        let ensemblePrediction = combinepredictions(
            trendPrediction: trendAnalysis.flareRisk,
            seasonalPrediction: seasonalPatterns.flareRisk,
            mlPrediction: mlPrediction
        )
        
        let contributingFactors = identifyContributingFactors(
            features: features,
            correlations: correlationMatrix
        )
        
        let recommendedActions = generateRecommendations(
            prediction: ensemblePrediction,
            factors: contributingFactors
        )
        
        return FlareUpPrediction(
            probability: ensemblePrediction.probability,
            confidence: ensemblePrediction.confidence,
            timeframe: TimeInterval(predictionHorizon * 24 * 3600),
            contributingFactors: contributingFactors,
            recommendedActions: recommendedActions,
            severity: determineSeverity(probability: ensemblePrediction.probability)
        )
    }
    
    func generateTreatmentRecommendations() async -> TreatmentRecommendation {
        let healthData = await gatherHealthData()
        let medicationHistory = await getMedicationEffectiveness()
        let personalPatterns = await analyzePersonalPatterns()
        
        // AI-powered treatment optimization
        let medicationAdjustments = optimizeMedicationRegimen(
            currentMedications: healthData.medications,
            effectiveness: medicationHistory,
            patterns: personalPatterns
        )
        
        let lifestyleChanges = generateLifestyleRecommendations(
            healthData: healthData,
            patterns: personalPatterns
        )
        
        let confidence = calculateRecommendationConfidence(
            dataQuality: healthData.quality,
            historyLength: healthData.historyDays
        )
        
        return TreatmentRecommendation(
            medicationAdjustments: medicationAdjustments,
            lifestyleChanges: lifestyleChanges,
            confidence: confidence,
            reasoning: generateReasoningExplanation(
                adjustments: medicationAdjustments,
                lifestyle: lifestyleChanges
            )
        )
    }
    
    // MARK: - Natural Language Processing
    
    func analyzeJournalSentiment() async -> SentimentAnalysis {
        let journalEntries = await fetchRecentJournalEntries()
        var sentimentScores: [Double] = []
        var emotionalTrends: [String: Double] = [:]
        var keyTopics: [String] = []
        
        for entry in journalEntries {
            // Sentiment analysis
            let sentiment = try? sentimentAnalyzer.predictSentiment(for: entry.content ?? "")
            if let sentiment = sentiment {
                sentimentScores.append(sentiment)
            }
            
            // Topic extraction
            let topics = extractTopics(from: entry.content ?? "")
            keyTopics.append(contentsOf: topics)
            
            // Emotional pattern recognition
            let emotions = detectEmotions(in: entry.content ?? "")
            for (emotion, intensity) in emotions {
                emotionalTrends[emotion, default: 0.0] += intensity
            }
        }
        
        return SentimentAnalysis(
            averageSentiment: sentimentScores.isEmpty ? 0 : sentimentScores.reduce(0, +) / Double(sentimentScores.count),
            sentimentTrend: calculateSentimentTrend(sentimentScores),
            emotionalPatterns: emotionalTrends,
            keyTopics: Array(Set(keyTopics)),
            mentalHealthScore: calculateMentalHealthScore(sentimentScores, emotionalTrends)
        )
    }
    
    // MARK: - Advanced Pattern Recognition
    
    private func performTrendAnalysis(_ features: HealthFeatures) -> TrendAnalysis {
        let painTrend = calculateTrend(features.painLevels)
        let energyTrend = calculateTrend(features.energyLevels)
        let sleepTrend = calculateTrend(features.sleepQuality)
        let medicationTrend = calculateMedicationAdherence(features.medicationIntakes)
        
        // Advanced statistical analysis
        let volatility = calculateVolatility(features.painLevels)
        let autocorrelation = calculateAutocorrelation(features.painLevels)
        
        let flareRisk = calculateFlareRisk(
            painTrend: painTrend,
            energyTrend: energyTrend,
            sleepTrend: sleepTrend,
            volatility: volatility
        )
        
        return TrendAnalysis(
            painTrend: painTrend,
            energyTrend: energyTrend,
            sleepTrend: sleepTrend,
            medicationAdherence: medicationTrend,
            volatility: volatility,
            autocorrelation: autocorrelation,
            flareRisk: flareRisk
        )
    }
    
    private func analyzeSeasonalPatterns(_ features: HealthFeatures) -> SeasonalAnalysis {
        let monthlyPatterns = groupByMonth(features)
        let weeklyPatterns = groupByWeekday(features)
        let weatherCorrelations = analyzeWeatherCorrelations(features)
        
        return SeasonalAnalysis(
            monthlyPatterns: monthlyPatterns,
            weeklyPatterns: weeklyPatterns,
            weatherCorrelations: weatherCorrelations,
            flareRisk: calculateSeasonalFlareRisk(monthlyPatterns, weeklyPatterns)
        )
    }
    
    // MARK: - Machine Learning Integration
    
    private func loadMLModel() {
        // Load pre-trained Core ML model for flare prediction
        // In a real implementation, this would load a trained model
        // For now, we'll use statistical methods
    }
    
    private func performMLPrediction(features: HealthFeatures) async -> MLPredictionResult {
        // Simulate ML prediction using statistical methods
        // In production, this would use the loaded Core ML model
        
        let featureVector = createFeatureVector(from: features)
        let prediction = simulateMLPrediction(featureVector)
        
        return MLPredictionResult(
            probability: prediction.probability,
            confidence: prediction.confidence,
            featureImportance: prediction.featureImportance
        )
    }
    
    // MARK: - Helper Functions
    
    private func gatherHealthData() async -> HealthDataCollection {
        let painEntries = await fetchPainEntries()
        let journalEntries = await fetchJournalEntries()
        let bassdaiAssessments = await fetchBASSDAIAssessments()
        let medications = await fetchMedications()
        let medicationIntakes = await fetchMedicationIntakes()
        
        return HealthDataCollection(
            painEntries: painEntries,
            journalEntries: journalEntries,
            bassdaiAssessments: bassdaiAssessments,
            medications: medications,
            medicationIntakes: medicationIntakes,
            quality: calculateDataQuality(painEntries, journalEntries, bassdaiAssessments),
            historyDays: calculateHistoryLength(painEntries)
        )
    }
    
    private func extractFeatures(from healthData: HealthDataCollection) -> HealthFeatures {
        return HealthFeatures(
            painLevels: healthData.painEntries.map { Double($0.painLevel) },
            energyLevels: healthData.journalEntries.compactMap { Double($0.energyLevel) },
            sleepQuality: healthData.journalEntries.compactMap { Double($0.sleepQuality) },
            bassdaiScores: healthData.bassdaiAssessments.map { $0.totalScore },
            medicationIntakes: healthData.medicationIntakes,
            timestamps: healthData.painEntries.compactMap { $0.timestamp }
        )
    }
    
    private func calculateTrend(_ values: [Double]) -> Double {
        guard values.count >= 2 else { return 0 }
        
        let n = Double(values.count)
        let sumX = (0..<values.count).reduce(0) { $0 + $1 }
        let sumY = values.reduce(0, +)
        let sumXY = zip(0..<values.count, values).reduce(0) { $0 + Double($1.0) * $1.1 }
        let sumX2 = (0..<values.count).reduce(0) { $0 + $1 * $1 }
        
        let slope = (n * sumXY - Double(sumX) * sumY) / (n * Double(sumX2) - Double(sumX * sumX))
        return slope
    }
    
    private func calculateVolatility(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count - 1)
        return sqrt(variance)
    }
    
    private func calculateAutocorrelation(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        
        let mean = values.reduce(0, +) / Double(values.count)
        let numerator = zip(values.dropLast(), values.dropFirst())
            .map { ($0 - mean) * ($1 - mean) }
            .reduce(0, +)
        
        let denominator = values.map { pow($0 - mean, 2) }.reduce(0, +)
        
        return denominator == 0 ? 0 : numerator / denominator
    }
}

// MARK: - Supporting Data Structures

struct HealthDataCollection {
    let painEntries: [PainEntry]
    let journalEntries: [JournalEntry]
    let bassdaiAssessments: [BASSDAIAssessment]
    let medications: [Medication]
    let medicationIntakes: [MedicationIntake]
    let quality: Double
    let historyDays: Int
}

struct HealthFeatures {
    let painLevels: [Double]
    let energyLevels: [Double]
    let sleepQuality: [Double]
    let bassdaiScores: [Double]
    let medicationIntakes: [MedicationIntake]
    let timestamps: [Date]
}

struct TrendAnalysis {
    let painTrend: Double
    let energyTrend: Double
    let sleepTrend: Double
    let medicationAdherence: Double
    let volatility: Double
    let autocorrelation: Double
    let flareRisk: Double
}

struct SeasonalAnalysis {
    let monthlyPatterns: [Int: Double]
    let weeklyPatterns: [Int: Double]
    let weatherCorrelations: [String: Double]
    let flareRisk: Double
}

struct MLPredictionResult {
    let probability: Double
    let confidence: Double
    let featureImportance: [String: Double]
}

struct SentimentAnalysis {
    let averageSentiment: Double
    let sentimentTrend: Double
    let emotionalPatterns: [String: Double]
    let keyTopics: [String]
    let mentalHealthScore: Double
}

// MARK: - Extensions for Core Data Fetching

extension PredictiveAnalyticsEngine {
    private func fetchPainEntries() async -> [PainEntry] {
        let request: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: false)]
        request.fetchLimit = 100
        
        do {
            return try context.fetch(request)
        } catch {
            print("Error fetching pain entries: \(error)")
            return []
        }
    }
    
    private func fetchJournalEntries() async -> [JournalEntry] {
        let request: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.timestamp, ascending: false)]
        request.fetchLimit = 100
        
        do {
            return try context.fetch(request)
        } catch {
            print("Error fetching journal entries: \(error)")
            return []
        }
    }
    
    private func fetchBASSDAIAssessments() async -> [BASSDAIAssessment] {
        let request: NSFetchRequest<BASSDAIAssessment> = BASSDAIAssessment.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \BASSDAIAssessment.timestamp, ascending: false)]
        request.fetchLimit = 50
        
        do {
            return try context.fetch(request)
        } catch {
            print("Error fetching BASDAI assessments: \(error)")
            return []
        }
    }
    
    private func fetchMedications() async -> [Medication] {
        let request: NSFetchRequest<Medication> = Medication.fetchRequest()
        
        do {
            return try context.fetch(request)
        } catch {
            print("Error fetching medications: \(error)")
            return []
        }
    }
    
    private func fetchMedicationIntakes() async -> [MedicationIntake] {
        let request: NSFetchRequest<MedicationIntake> = MedicationIntake.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \MedicationIntake.timestamp, ascending: false)]
        request.fetchLimit = 200
        
        do {
            return try context.fetch(request)
        } catch {
            print("Error fetching medication intakes: \(error)")
            return []
        }
    }
}