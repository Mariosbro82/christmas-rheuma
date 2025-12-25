//
//  SentimentAnalysisEngine.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import NaturalLanguage
import CoreML
import CreateML
import os.log

// MARK: - Sentiment Analysis Engine
@MainActor
class SentimentAnalysisEngine: ObservableObject {
    
    private let logger = Logger(subsystem: "InflamAI", category: "SentimentAnalysisEngine")
    
    // Published properties
    @Published var currentMoodTrend: MoodTrend?
    @Published var emotionalInsights: [EmotionalInsight] = []
    @Published var moodHistory: [MoodEntry] = []
    @Published var isAnalyzing = false
    @Published var lastAnalysisDate: Date?
    
    // ML Models
    private var sentimentModel: MLModel?
    private var emotionClassifier: MLModel?
    private var moodPredictionModel: MLModel?
    
    // Natural Language Processing
    private let sentimentPredictor = NLModel(mlModel: try! NLModel.sentimentClassifier(revision: NLModelRevision.sentimentClassifier_1))
    private let languageRecognizer = NLLanguageRecognizer()
    private let tokenizer = NLTokenizer(unit: .word)
    private let tagger = NLTagger(tagSchemes: [.sentimentScore, .language, .tokenType])
    
    // Analysis parameters
    private let analysisWindow = 30 // Days to analyze
    private let minimumEntries = 5
    private let emotionKeywords: [String: EmotionType] = [
        // Positive emotions
        "happy": .joy, "joy": .joy, "excited": .joy, "cheerful": .joy, "optimistic": .joy,
        "grateful": .gratitude, "thankful": .gratitude, "blessed": .gratitude,
        "calm": .calm, "peaceful": .calm, "relaxed": .calm, "serene": .calm,
        "hopeful": .hope, "confident": .hope, "positive": .hope,
        
        // Negative emotions
        "sad": .sadness, "depressed": .sadness, "down": .sadness, "blue": .sadness,
        "angry": .anger, "frustrated": .anger, "irritated": .anger, "mad": .anger,
        "anxious": .anxiety, "worried": .anxiety, "nervous": .anxiety, "stressed": .anxiety,
        "afraid": .fear, "scared": .fear, "terrified": .fear, "worried": .fear,
        "tired": .fatigue, "exhausted": .fatigue, "drained": .fatigue, "weary": .fatigue,
        
        // Pain-related emotions
        "painful": .pain, "hurt": .pain, "aching": .pain, "sore": .pain,
        "overwhelmed": .overwhelm, "helpless": .overwhelm, "defeated": .overwhelm
    ]
    
    init() {
        loadHistoricalData()
        loadMLModels()
        setupNaturalLanguageProcessing()
    }
    
    // MARK: - Public Methods
    
    func analyzeJournalEntry(_ text: String) async -> JournalAnalysis {
        logger.info("Analyzing journal entry")
        isAnalyzing = true
        
        defer { isAnalyzing = false }
        
        // Basic sentiment analysis
        let sentiment = analyzeSentiment(text)
        
        // Emotion detection
        let emotions = detectEmotions(text)
        
        // Mood assessment
        let mood = assessMood(text: text, sentiment: sentiment, emotions: emotions)
        
        // Pain indicators
        let painIndicators = detectPainIndicators(text)
        
        // Coping strategies mentioned
        let copingStrategies = detectCopingStrategies(text)
        
        // Stress indicators
        let stressIndicators = detectStressIndicators(text)
        
        // Generate insights
        let insights = generateInsights(
            text: text,
            sentiment: sentiment,
            emotions: emotions,
            mood: mood,
            painIndicators: painIndicators
        )
        
        let analysis = JournalAnalysis(
            date: Date(),
            text: text,
            sentiment: sentiment,
            emotions: emotions,
            mood: mood,
            painIndicators: painIndicators,
            copingStrategies: copingStrategies,
            stressIndicators: stressIndicators,
            insights: insights,
            confidence: calculateConfidence(text: text)
        )
        
        // Add to mood history
        let moodEntry = MoodEntry(
            date: Date(),
            mood: mood,
            sentiment: sentiment,
            dominantEmotion: emotions.max(by: { $0.confidence < $1.confidence })?.type ?? .neutral,
            journalText: text,
            painLevel: extractPainLevel(from: painIndicators),
            stressLevel: extractStressLevel(from: stressIndicators)
        )
        
        moodHistory.append(moodEntry)
        saveMoodHistory()
        
        // Update trends and insights
        await updateMoodTrends()
        await updateEmotionalInsights()
        
        lastAnalysisDate = Date()
        
        return analysis
    }
    
    func generateMoodReport(for period: AnalysisPeriod) async -> MoodReport {
        logger.info("Generating mood report for \(period)")
        
        let startDate = getStartDate(for: period)
        let relevantEntries = moodHistory.filter { $0.date >= startDate }
        
        guard !relevantEntries.isEmpty else {
            return MoodReport(
                period: period,
                averageMood: .neutral,
                moodVariability: 0.0,
                dominantEmotions: [],
                sentimentTrend: .stable,
                painMoodCorrelation: 0.0,
                stressMoodCorrelation: 0.0,
                insights: ["Insufficient data for analysis"],
                recommendations: ["Continue journaling to build insights"]
            )
        }
        
        // Calculate average mood
        let averageMoodScore = relevantEntries.map { $0.mood.rawValue }.reduce(0, +) / Double(relevantEntries.count)
        let averageMood = MoodLevel(rawValue: averageMoodScore) ?? .neutral
        
        // Calculate mood variability
        let moodScores = relevantEntries.map { $0.mood.rawValue }
        let moodVariability = calculateVariability(moodScores)
        
        // Find dominant emotions
        let emotionCounts = Dictionary(grouping: relevantEntries, by: { $0.dominantEmotion })
            .mapValues { $0.count }
        let dominantEmotions = emotionCounts.sorted { $0.value > $1.value }
            .prefix(3)
            .map { $0.key }
        
        // Calculate sentiment trend
        let sentimentTrend = calculateSentimentTrend(relevantEntries)
        
        // Calculate correlations
        let painMoodCorrelation = calculatePainMoodCorrelation(relevantEntries)
        let stressMoodCorrelation = calculateStressMoodCorrelation(relevantEntries)
        
        // Generate insights and recommendations
        let insights = generateMoodInsights(entries: relevantEntries, averageMood: averageMood)
        let recommendations = generateMoodRecommendations(entries: relevantEntries, averageMood: averageMood)
        
        return MoodReport(
            period: period,
            averageMood: averageMood,
            moodVariability: moodVariability,
            dominantEmotions: Array(dominantEmotions),
            sentimentTrend: sentimentTrend,
            painMoodCorrelation: painMoodCorrelation,
            stressMoodCorrelation: stressMoodCorrelation,
            insights: insights,
            recommendations: recommendations
        )
    }
    
    func predictMoodTrend() async -> MoodPrediction {
        logger.info("Predicting mood trend")
        
        guard moodHistory.count >= minimumEntries else {
            return MoodPrediction(
                predictedMood: .neutral,
                confidence: 0.1,
                timeframe: .week,
                factors: [],
                recommendations: ["Continue journaling to improve predictions"]
            )
        }
        
        // Use ML model if available, otherwise use pattern analysis
        if let model = moodPredictionModel {
            return await predictWithMLModel(model)
        } else {
            return await predictWithPatternAnalysis()
        }
    }
    
    func detectEmotionalPatterns() async -> [EmotionalPattern] {
        logger.info("Detecting emotional patterns")
        
        var patterns: [EmotionalPattern] = []
        
        // Temporal patterns
        patterns.append(contentsOf: detectTemporalEmotionalPatterns())
        
        // Cyclical patterns
        patterns.append(contentsOf: detectCyclicalPatterns())
        
        // Trigger patterns
        patterns.append(contentsOf: detectTriggerPatterns())
        
        return patterns
    }
    
    // MARK: - Private Methods
    
    private func loadHistoricalData() {
        if let data = UserDefaults.standard.data(forKey: "mood_history"),
           let history = try? JSONDecoder().decode([MoodEntry].self, from: data) {
            moodHistory = history
            logger.info("Loaded \(history.count) mood entries")
        }
    }
    
    private func saveMoodHistory() {
        if let data = try? JSONEncoder().encode(moodHistory) {
            UserDefaults.standard.set(data, forKey: "mood_history")
        }
    }
    
    private func loadMLModels() {
        Task {
            // Load sentiment model
            if let modelURL = Bundle.main.url(forResource: "SentimentModel", withExtension: "mlmodel") {
                do {
                    sentimentModel = try MLModel(contentsOf: modelURL)
                    logger.info("Loaded sentiment model")
                } catch {
                    logger.error("Failed to load sentiment model: \(error.localizedDescription)")
                }
            }
            
            // Load emotion classifier
            if let modelURL = Bundle.main.url(forResource: "EmotionClassifier", withExtension: "mlmodel") {
                do {
                    emotionClassifier = try MLModel(contentsOf: modelURL)
                    logger.info("Loaded emotion classifier")
                } catch {
                    logger.error("Failed to load emotion classifier: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func setupNaturalLanguageProcessing() {
        tagger.string = ""
        tokenizer.string = ""
    }
    
    private func analyzeSentiment(_ text: String) -> SentimentScore {
        // Use Core ML model if available
        if let model = sentimentModel {
            return analyzeSentimentWithML(text, model: model)
        }
        
        // Fallback to Natural Language framework
        tagger.string = text
        let range = text.startIndex..<text.endIndex
        
        var sentimentScore: Double = 0.0
        var tokenCount = 0
        
        tagger.enumerateTags(in: range, unit: .word, scheme: .sentimentScore) { tag, tokenRange in
            if let tag = tag, let score = Double(tag.rawValue) {
                sentimentScore += score
                tokenCount += 1
            }
            return true
        }
        
        let averageScore = tokenCount > 0 ? sentimentScore / Double(tokenCount) : 0.0
        
        // Convert to sentiment classification
        let polarity: SentimentPolarity
        if averageScore > 0.1 {
            polarity = .positive
        } else if averageScore < -0.1 {
            polarity = .negative
        } else {
            polarity = .neutral
        }
        
        return SentimentScore(
            polarity: polarity,
            score: averageScore,
            confidence: min(abs(averageScore) * 2, 1.0)
        )
    }
    
    private func analyzeSentimentWithML(_ text: String, model: MLModel) -> SentimentScore {
        do {
            // Prepare input features
            let features = prepareSentimentFeatures(text)
            let input = try MLDictionaryFeatureProvider(dictionary: features)
            
            // Make prediction
            let prediction = try model.prediction(from: input)
            
            // Extract results
            let score = prediction.featureValue(for: "sentiment_score")?.doubleValue ?? 0.0
            let confidence = prediction.featureValue(for: "confidence")?.doubleValue ?? 0.5
            
            let polarity: SentimentPolarity
            if score > 0.1 {
                polarity = .positive
            } else if score < -0.1 {
                polarity = .negative
            } else {
                polarity = .neutral
            }
            
            return SentimentScore(
                polarity: polarity,
                score: score,
                confidence: confidence
            )
            
        } catch {
            logger.error("ML sentiment analysis failed: \(error.localizedDescription)")
            return analyzeSentiment(text) // Fallback
        }
    }
    
    private func detectEmotions(_ text: String) -> [EmotionDetection] {
        var emotions: [EmotionDetection] = []
        let lowercaseText = text.lowercased()
        
        // Keyword-based emotion detection
        for (keyword, emotionType) in emotionKeywords {
            if lowercaseText.contains(keyword) {
                let confidence = calculateEmotionConfidence(keyword: keyword, text: lowercaseText)
                emotions.append(EmotionDetection(
                    type: emotionType,
                    confidence: confidence,
                    triggers: [keyword]
                ))
            }
        }
        
        // Use ML model if available
        if let model = emotionClassifier {
            let mlEmotions = detectEmotionsWithML(text, model: model)
            emotions.append(contentsOf: mlEmotions)
        }
        
        // Merge and deduplicate emotions
        let mergedEmotions = mergeEmotions(emotions)
        
        return mergedEmotions.sorted { $0.confidence > $1.confidence }
    }
    
    private func detectEmotionsWithML(_ text: String, model: MLModel) -> [EmotionDetection] {
        do {
            let features = prepareEmotionFeatures(text)
            let input = try MLDictionaryFeatureProvider(dictionary: features)
            let prediction = try model.prediction(from: input)
            
            var emotions: [EmotionDetection] = []
            
            // Extract emotion probabilities
            for emotionType in EmotionType.allCases {
                if let probability = prediction.featureValue(for: emotionType.rawValue)?.doubleValue {
                    if probability > 0.3 { // Threshold for detection
                        emotions.append(EmotionDetection(
                            type: emotionType,
                            confidence: probability,
                            triggers: []
                        ))
                    }
                }
            }
            
            return emotions
            
        } catch {
            logger.error("ML emotion detection failed: \(error.localizedDescription)")
            return []
        }
    }
    
    private func assessMood(text: String, sentiment: SentimentScore, emotions: [EmotionDetection]) -> MoodLevel {
        var moodScore = 5.0 // Neutral baseline
        
        // Adjust based on sentiment
        moodScore += sentiment.score * 3.0
        
        // Adjust based on emotions
        for emotion in emotions {
            let emotionImpact = getEmotionMoodImpact(emotion.type) * emotion.confidence
            moodScore += emotionImpact
        }
        
        // Clamp to valid range
        moodScore = max(1.0, min(10.0, moodScore))
        
        return MoodLevel(rawValue: moodScore) ?? .neutral
    }
    
    private func getEmotionMoodImpact(_ emotion: EmotionType) -> Double {
        switch emotion {
        case .joy: return 2.0
        case .gratitude: return 1.5
        case .calm: return 1.0
        case .hope: return 1.5
        case .sadness: return -2.0
        case .anger: return -1.5
        case .anxiety: return -1.5
        case .fear: return -1.0
        case .fatigue: return -1.0
        case .pain: return -1.5
        case .overwhelm: return -2.0
        case .neutral: return 0.0
        }
    }
    
    private func detectPainIndicators(_ text: String) -> [PainIndicator] {
        var indicators: [PainIndicator] = []
        let lowercaseText = text.lowercased()
        
        let painKeywords = [
            "pain": 0.8, "hurt": 0.7, "ache": 0.6, "sore": 0.5,
            "stiff": 0.6, "swollen": 0.7, "tender": 0.5, "throbbing": 0.8,
            "burning": 0.7, "sharp": 0.8, "dull": 0.5, "chronic": 0.9
        ]
        
        for (keyword, intensity) in painKeywords {
            if lowercaseText.contains(keyword) {
                indicators.append(PainIndicator(
                    type: .physical,
                    intensity: intensity,
                    location: extractPainLocation(text, keyword: keyword),
                    description: keyword
                ))
            }
        }
        
        // Emotional pain indicators
        let emotionalPainKeywords = [
            "heartache": 0.7, "emotional pain": 0.8, "hurt feelings": 0.6
        ]
        
        for (keyword, intensity) in emotionalPainKeywords {
            if lowercaseText.contains(keyword) {
                indicators.append(PainIndicator(
                    type: .emotional,
                    intensity: intensity,
                    location: nil,
                    description: keyword
                ))
            }
        }
        
        return indicators
    }
    
    private func detectCopingStrategies(_ text: String) -> [CopingStrategy] {
        var strategies: [CopingStrategy] = []
        let lowercaseText = text.lowercased()
        
        let copingKeywords: [String: CopingStrategyType] = [
            "meditation": .mindfulness, "mindfulness": .mindfulness, "breathing": .mindfulness,
            "exercise": .physical, "walk": .physical, "yoga": .physical, "stretch": .physical,
            "friend": .social, "family": .social, "support": .social, "talk": .social,
            "music": .creative, "art": .creative, "write": .creative, "journal": .creative,
            "medication": .medical, "doctor": .medical, "therapy": .medical, "treatment": .medical
        ]
        
        for (keyword, type) in copingKeywords {
            if lowercaseText.contains(keyword) {
                strategies.append(CopingStrategy(
                    type: type,
                    description: keyword,
                    effectiveness: estimateCopingEffectiveness(keyword, in: text)
                ))
            }
        }
        
        return strategies
    }
    
    private func detectStressIndicators(_ text: String) -> [StressIndicator] {
        var indicators: [StressIndicator] = []
        let lowercaseText = text.lowercased()
        
        let stressKeywords = [
            "stressed": 0.8, "overwhelmed": 0.9, "anxious": 0.7, "worried": 0.6,
            "pressure": 0.6, "deadline": 0.7, "busy": 0.5, "exhausted": 0.8
        ]
        
        for (keyword, intensity) in stressKeywords {
            if lowercaseText.contains(keyword) {
                indicators.append(StressIndicator(
                    type: .psychological,
                    intensity: intensity,
                    source: extractStressSource(text, keyword: keyword),
                    description: keyword
                ))
            }
        }
        
        return indicators
    }
    
    private func generateInsights(
        text: String,
        sentiment: SentimentScore,
        emotions: [EmotionDetection],
        mood: MoodLevel,
        painIndicators: [PainIndicator]
    ) -> [String] {
        var insights: [String] = []
        
        // Sentiment insights
        if sentiment.confidence > 0.7 {
            switch sentiment.polarity {
            case .positive:
                insights.append("Your writing shows a positive outlook today")
            case .negative:
                insights.append("Your writing reflects some challenging emotions")
            case .neutral:
                insights.append("Your emotional tone appears balanced today")
            }
        }
        
        // Emotion insights
        if let dominantEmotion = emotions.first {
            insights.append("The dominant emotion detected is \(dominantEmotion.type.rawValue)")
        }
        
        // Pain insights
        if !painIndicators.isEmpty {
            let avgPainIntensity = painIndicators.map { $0.intensity }.reduce(0, +) / Double(painIndicators.count)
            if avgPainIntensity > 0.7 {
                insights.append("High pain levels are evident in your writing")
            } else if avgPainIntensity > 0.4 {
                insights.append("Moderate pain levels are mentioned")
            }
        }
        
        // Mood insights
        switch mood {
        case .veryLow, .low:
            insights.append("Your mood appears to be lower than usual")
        case .high, .veryHigh:
            insights.append("Your mood seems elevated and positive")
        case .neutral:
            insights.append("Your mood appears stable and balanced")
        }
        
        return insights
    }
    
    private func calculateConfidence(text: String) -> Double {
        var confidence = 0.5 // Base confidence
        
        // Increase confidence based on text length
        let wordCount = text.components(separatedBy: .whitespacesAndNewlines).count
        confidence += min(Double(wordCount) / 100.0, 0.3)
        
        // Increase confidence based on emotional keywords
        let emotionalWords = emotionKeywords.keys.filter { text.lowercased().contains($0) }
        confidence += min(Double(emotionalWords.count) / 10.0, 0.2)
        
        return min(confidence, 1.0)
    }
    
    private func updateMoodTrends() async {
        guard moodHistory.count >= 3 else { return }
        
        let recentEntries = Array(moodHistory.suffix(7)) // Last 7 entries
        let moodScores = recentEntries.map { $0.mood.rawValue }
        
        let trend = calculateTrend(moodScores)
        let direction: TrendDirection
        
        if trend > 0.2 {
            direction = .improving
        } else if trend < -0.2 {
            direction = .declining
        } else {
            direction = .stable
        }
        
        currentMoodTrend = MoodTrend(
            direction: direction,
            strength: abs(trend),
            period: .week,
            confidence: min(Double(recentEntries.count) / 7.0, 1.0)
        )
    }
    
    private func updateEmotionalInsights() async {
        guard moodHistory.count >= minimumEntries else { return }
        
        var insights: [EmotionalInsight] = []
        
        // Analyze emotion frequency
        let emotionCounts = Dictionary(grouping: moodHistory, by: { $0.dominantEmotion })
            .mapValues { $0.count }
        
        if let mostFrequentEmotion = emotionCounts.max(by: { $0.value < $1.value }) {
            insights.append(EmotionalInsight(
                type: .pattern,
                title: "Most Frequent Emotion",
                description: "\(mostFrequentEmotion.key.rawValue.capitalized) appears most often in your entries",
                confidence: 0.8,
                actionable: true,
                recommendations: generateEmotionRecommendations(mostFrequentEmotion.key)
            ))
        }
        
        // Analyze mood-pain correlation
        let painMoodCorrelation = calculatePainMoodCorrelation(moodHistory)
        if abs(painMoodCorrelation) > 0.5 {
            insights.append(EmotionalInsight(
                type: .correlation,
                title: "Pain-Mood Connection",
                description: painMoodCorrelation > 0 ? "Higher pain levels correlate with better mood" : "Higher pain levels correlate with lower mood",
                confidence: abs(painMoodCorrelation),
                actionable: true,
                recommendations: ["Consider pain management strategies to improve mood"]
            ))
        }
        
        emotionalInsights = insights
    }
    
    private func generateEmotionRecommendations(_ emotion: EmotionType) -> [String] {
        switch emotion {
        case .sadness:
            return ["Consider reaching out to friends or family", "Engage in activities you enjoy", "Practice self-compassion"]
        case .anxiety:
            return ["Try deep breathing exercises", "Practice mindfulness meditation", "Consider talking to a counselor"]
        case .anger:
            return ["Use physical exercise to release tension", "Practice anger management techniques", "Identify triggers"]
        case .joy:
            return ["Savor these positive moments", "Share your joy with others", "Note what brings you happiness"]
        default:
            return ["Continue monitoring your emotional patterns", "Practice emotional awareness"]
        }
    }
    
    // MARK: - Helper Methods
    
    private func prepareSentimentFeatures(_ text: String) -> [String: Any] {
        return [
            "text": text,
            "word_count": text.components(separatedBy: .whitespacesAndNewlines).count,
            "character_count": text.count
        ]
    }
    
    private func prepareEmotionFeatures(_ text: String) -> [String: Any] {
        return [
            "text": text,
            "word_count": text.components(separatedBy: .whitespacesAndNewlines).count,
            "emotional_word_count": emotionKeywords.keys.filter { text.lowercased().contains($0) }.count
        ]
    }
    
    private func calculateEmotionConfidence(keyword: String, text: String) -> Double {
        let occurrences = text.components(separatedBy: keyword).count - 1
        return min(0.3 + Double(occurrences) * 0.2, 1.0)
    }
    
    private func mergeEmotions(_ emotions: [EmotionDetection]) -> [EmotionDetection] {
        let grouped = Dictionary(grouping: emotions, by: { $0.type })
        
        return grouped.compactMap { (type, detections) in
            let maxConfidence = detections.map { $0.confidence }.max() ?? 0.0
            let allTriggers = detections.flatMap { $0.triggers }
            
            return EmotionDetection(
                type: type,
                confidence: maxConfidence,
                triggers: Array(Set(allTriggers))
            )
        }
    }
    
    private func extractPainLocation(_ text: String, keyword: String) -> String? {
        let bodyParts = ["head", "neck", "shoulder", "arm", "hand", "back", "chest", "hip", "leg", "knee", "foot", "joint"]
        let lowercaseText = text.lowercased()
        
        for bodyPart in bodyParts {
            if lowercaseText.contains(bodyPart) {
                return bodyPart
            }
        }
        
        return nil
    }
    
    private func extractStressSource(_ text: String, keyword: String) -> String? {
        let stressSources = ["work", "family", "health", "money", "relationship", "school"]
        let lowercaseText = text.lowercased()
        
        for source in stressSources {
            if lowercaseText.contains(source) {
                return source
            }
        }
        
        return nil
    }
    
    private func estimateCopingEffectiveness(_ strategy: String, in text: String) -> Double {
        // Simple heuristic based on context
        let positiveWords = ["helped", "better", "relief", "good", "effective"]
        let negativeWords = ["didn't help", "worse", "failed", "useless"]
        
        let lowercaseText = text.lowercased()
        
        for word in positiveWords {
            if lowercaseText.contains(word) {
                return 0.8
            }
        }
        
        for word in negativeWords {
            if lowercaseText.contains(word) {
                return 0.2
            }
        }
        
        return 0.5 // Neutral effectiveness
    }
    
    private func extractPainLevel(from indicators: [PainIndicator]) -> Double {
        guard !indicators.isEmpty else { return 0.0 }
        return indicators.map { $0.intensity }.reduce(0, +) / Double(indicators.count)
    }
    
    private func extractStressLevel(from indicators: [StressIndicator]) -> Double {
        guard !indicators.isEmpty else { return 0.0 }
        return indicators.map { $0.intensity }.reduce(0, +) / Double(indicators.count)
    }
    
    private func getStartDate(for period: AnalysisPeriod) -> Date {
        let calendar = Calendar.current
        let now = Date()
        
        switch period {
        case .week:
            return calendar.date(byAdding: .day, value: -7, to: now) ?? now
        case .month:
            return calendar.date(byAdding: .month, value: -1, to: now) ?? now
        case .quarter:
            return calendar.date(byAdding: .month, value: -3, to: now) ?? now
        case .year:
            return calendar.date(byAdding: .year, value: -1, to: now) ?? now
        }
    }
    
    private func calculateVariability(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0.0 }
        
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count)
        
        return sqrt(variance)
    }
    
    private func calculateSentimentTrend(_ entries: [MoodEntry]) -> SentimentTrend {
        guard entries.count >= 3 else { return .stable }
        
        let sentimentScores = entries.map { $0.sentiment.score }
        let trend = calculateTrend(sentimentScores)
        
        if trend > 0.1 {
            return .improving
        } else if trend < -0.1 {
            return .declining
        } else {
            return .stable
        }
    }
    
    private func calculatePainMoodCorrelation(_ entries: [MoodEntry]) -> Double {
        guard entries.count >= 3 else { return 0.0 }
        
        let painLevels = entries.map { $0.painLevel }
        let moodScores = entries.map { $0.mood.rawValue }
        
        return calculateCorrelation(painLevels, moodScores)
    }
    
    private func calculateStressMoodCorrelation(_ entries: [MoodEntry]) -> Double {
        guard entries.count >= 3 else { return 0.0 }
        
        let stressLevels = entries.map { $0.stressLevel }
        let moodScores = entries.map { $0.mood.rawValue }
        
        return calculateCorrelation(stressLevels, moodScores)
    }
    
    private func calculateTrend(_ values: [Double]) -> Double {
        guard values.count >= 2 else { return 0.0 }
        
        let n = Double(values.count)
        let x = Array(0..<values.count).map(Double.init)
        let y = values
        
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumXX = x.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
        
        return slope
    }
    
    private func calculateCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count && x.count > 1 else { return 0.0 }
        
        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumXX = x.map { $0 * $0 }.reduce(0, +)
        let sumYY = y.map { $0 * $0 }.reduce(0, +)
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY))
        
        return denominator != 0 ? numerator / denominator : 0.0
    }
    
    private func generateMoodInsights(entries: [MoodEntry], averageMood: MoodLevel) -> [String] {
        var insights: [String] = []
        
        // Average mood insight
        switch averageMood {
        case .veryLow:
            insights.append("Your mood has been consistently low during this period")
        case .low:
            insights.append("Your mood has been below average during this period")
        case .neutral:
            insights.append("Your mood has been stable and balanced")
        case .high:
            insights.append("Your mood has been above average during this period")
        case .veryHigh:
            insights.append("Your mood has been consistently high during this period")
        }
        
        // Emotion pattern insights
        let emotionCounts = Dictionary(grouping: entries, by: { $0.dominantEmotion })
            .mapValues { $0.count }
        
        if let mostCommon = emotionCounts.max(by: { $0.value < $1.value }) {
            insights.append("\(mostCommon.key.rawValue.capitalized) was your most frequent emotion")
        }
        
        return insights
    }
    
    private func generateMoodRecommendations(entries: [MoodEntry], averageMood: MoodLevel) -> [String] {
        var recommendations: [String] = []
        
        switch averageMood {
        case .veryLow, .low:
            recommendations.append("Consider reaching out to a mental health professional")
            recommendations.append("Engage in activities that typically bring you joy")
            recommendations.append("Practice self-care and be gentle with yourself")
        case .neutral:
            recommendations.append("Continue your current coping strategies")
            recommendations.append("Look for opportunities to enhance your well-being")
        case .high, .veryHigh:
            recommendations.append("Maintain the positive practices that are working for you")
            recommendations.append("Consider sharing your strategies with others")
        }
        
        return recommendations
    }
    
    private func predictWithMLModel(_ model: MLModel) async -> MoodPrediction {
        // Implementation for ML-based mood prediction
        return MoodPrediction(
            predictedMood: .neutral,
            confidence: 0.7,
            timeframe: .week,
            factors: [],
            recommendations: []
        )
    }
    
    private func predictWithPatternAnalysis() async -> MoodPrediction {
        let recentEntries = Array(moodHistory.suffix(7))
        let moodScores = recentEntries.map { $0.mood.rawValue }
        
        let trend = calculateTrend(moodScores)
        let currentMood = recentEntries.last?.mood ?? .neutral
        
        var predictedMoodValue = currentMood.rawValue + trend
        predictedMoodValue = max(1.0, min(10.0, predictedMoodValue))
        
        let predictedMood = MoodLevel(rawValue: predictedMoodValue) ?? .neutral
        
        return MoodPrediction(
            predictedMood: predictedMood,
            confidence: 0.6,
            timeframe: .week,
            factors: ["Recent mood trend", "Historical patterns"],
            recommendations: generateMoodRecommendations(entries: recentEntries, averageMood: predictedMood)
        )
    }
    
    private func detectTemporalEmotionalPatterns() -> [EmotionalPattern] {
        // Analyze patterns by time of day, day of week, etc.
        return []
    }
    
    private func detectCyclicalPatterns() -> [EmotionalPattern] {
        // Analyze monthly or seasonal patterns
        return []
    }
    
    private func detectTriggerPatterns() -> [EmotionalPattern] {
        // Analyze patterns related to specific triggers
        return []
    }
}

// MARK: - Supporting Types

struct JournalAnalysis {
    let date: Date
    let text: String
    let sentiment: SentimentScore
    let emotions: [EmotionDetection]
    let mood: MoodLevel
    let painIndicators: [PainIndicator]
    let copingStrategies: [CopingStrategy]
    let stressIndicators: [StressIndicator]
    let insights: [String]
    let confidence: Double
}

struct SentimentScore {
    let polarity: SentimentPolarity
    let score: Double // -1.0 to 1.0
    let confidence: Double
}

enum SentimentPolarity: String, CaseIterable {
    case positive = "positive"
    case negative = "negative"
    case neutral = "neutral"
}

struct EmotionDetection {
    let type: EmotionType
    let confidence: Double
    let triggers: [String]
}

enum EmotionType: String, CaseIterable {
    case joy = "joy"
    case sadness = "sadness"
    case anger = "anger"
    case fear = "fear"
    case anxiety = "anxiety"
    case calm = "calm"
    case gratitude = "gratitude"
    case hope = "hope"
    case fatigue = "fatigue"
    case pain = "pain"
    case overwhelm = "overwhelm"
    case neutral = "neutral"
}

enum MoodLevel: Double, CaseIterable {
    case veryLow = 1.0
    case low = 3.0
    case neutral = 5.0
    case high = 7.0
    case veryHigh = 9.0
}

struct PainIndicator {
    let type: PainType
    let intensity: Double
    let location: String?
    let description: String
}

enum PainType {
    case physical
    case emotional
}

struct CopingStrategy {
    let type: CopingStrategyType
    let description: String
    let effectiveness: Double
}

enum CopingStrategyType {
    case mindfulness
    case physical
    case social
    case creative
    case medical
}

struct StressIndicator {
    let type: StressType
    let intensity: Double
    let source: String?
    let description: String
}

enum StressType {
    case psychological
    case physical
    case social
}

struct MoodEntry: Codable {
    let date: Date
    let mood: MoodLevel
    let sentiment: SentimentScore
    let dominantEmotion: EmotionType
    let journalText: String
    let painLevel: Double
    let stressLevel: Double
}

struct MoodTrend {
    let direction: TrendDirection
    let strength: Double
    let period: AnalysisPeriod
    let confidence: Double
}

enum TrendDirection {
    case improving
    case declining
    case stable
}

enum AnalysisPeriod {
    case week
    case month
    case quarter
    case year
}

struct EmotionalInsight {
    let type: InsightType
    let title: String
    let description: String
    let confidence: Double
    let actionable: Bool
    let recommendations: [String]
}

enum InsightType {
    case pattern
    case correlation
    case trend
    case anomaly
}

struct MoodReport {
    let period: AnalysisPeriod
    let averageMood: MoodLevel
    let moodVariability: Double
    let dominantEmotions: [EmotionType]
    let sentimentTrend: SentimentTrend
    let painMoodCorrelation: Double
    let stressMoodCorrelation: Double
    let insights: [String]
    let recommendations: [String]
}

enum SentimentTrend {
    case improving
    case declining
    case stable
}

struct MoodPrediction {
    let predictedMood: MoodLevel
    let confidence: Double
    let timeframe: PredictionTimeframe
    let factors: [String]
    let recommendations: [String]
}

enum PredictionTimeframe {
    case day
    case week
    case month
}

struct EmotionalPattern {
    let type: PatternType
    let description: String
    let frequency: Double
    let confidence: Double
}

enum PatternType {
    case temporal
    case cyclical
    case trigger
    case correlation
}

// MARK: - Extensions

extension SentimentScore: Codable {}
extension EmotionType: Codable {}
extension MoodLevel: Codable {}