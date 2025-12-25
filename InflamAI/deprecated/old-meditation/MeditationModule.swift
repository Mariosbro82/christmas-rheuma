//
//  MeditationModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import AVFoundation
import Combine
import HealthKit
import CoreLocation

// MARK: - Data Models

struct MeditationSession: Identifiable, Codable {
    let id = UUID()
    let type: MeditationType
    let title: String
    let description: String
    let duration: TimeInterval
    let audioURL: URL?
    let imageURL: URL?
    let difficulty: DifficultyLevel
    let tags: [String]
    let instructor: String?
    let category: MeditationCategory
    let benefits: [String]
    let prerequisites: [String]
    let isGuided: Bool
    let backgroundSounds: [BackgroundSound]
    let breathingPattern: BreathingPattern?
    let visualizations: [Visualization]
    let affirmations: [String]
    let createdAt: Date
    let updatedAt: Date
    
    init(type: MeditationType, title: String, description: String, duration: TimeInterval, category: MeditationCategory) {
        self.type = type
        self.title = title
        self.description = description
        self.duration = duration
        self.category = category
        self.audioURL = nil
        self.imageURL = nil
        self.difficulty = .beginner
        self.tags = []
        self.instructor = nil
        self.benefits = []
        self.prerequisites = []
        self.isGuided = true
        self.backgroundSounds = []
        self.breathingPattern = nil
        self.visualizations = []
        self.affirmations = []
        self.createdAt = Date()
        self.updatedAt = Date()
    }
}

enum MeditationType: String, CaseIterable, Codable {
    case mindfulness = "mindfulness"
    case breathing = "breathing"
    case bodyScanning = "body_scanning"
    case lovingKindness = "loving_kindness"
    case visualization = "visualization"
    case mantra = "mantra"
    case walking = "walking"
    case sleep = "sleep"
    case painRelief = "pain_relief"
    case stressReduction = "stress_reduction"
    case anxietyRelief = "anxiety_relief"
    case focusConcentration = "focus_concentration"
    case emotionalHealing = "emotional_healing"
    case gratitude = "gratitude"
    case selfCompassion = "self_compassion"
    
    var displayName: String {
        switch self {
        case .mindfulness: return "Mindfulness"
        case .breathing: return "Breathing"
        case .bodyScanning: return "Body Scanning"
        case .lovingKindness: return "Loving Kindness"
        case .visualization: return "Visualization"
        case .mantra: return "Mantra"
        case .walking: return "Walking"
        case .sleep: return "Sleep"
        case .painRelief: return "Pain Relief"
        case .stressReduction: return "Stress Reduction"
        case .anxietyRelief: return "Anxiety Relief"
        case .focusConcentration: return "Focus & Concentration"
        case .emotionalHealing: return "Emotional Healing"
        case .gratitude: return "Gratitude"
        case .selfCompassion: return "Self-Compassion"
        }
    }
    
    var systemImage: String {
        switch self {
        case .mindfulness: return "brain.head.profile"
        case .breathing: return "lungs.fill"
        case .bodyScanning: return "figure.walk"
        case .lovingKindness: return "heart.fill"
        case .visualization: return "eye.fill"
        case .mantra: return "speaker.wave.3.fill"
        case .walking: return "figure.walk.motion"
        case .sleep: return "moon.fill"
        case .painRelief: return "cross.case.fill"
        case .stressReduction: return "leaf.fill"
        case .anxietyRelief: return "calm"
        case .focusConcentration: return "target"
        case .emotionalHealing: return "heart.text.square.fill"
        case .gratitude: return "hands.sparkles.fill"
        case .selfCompassion: return "person.fill.checkmark"
        }
    }
    
    var color: String {
        switch self {
        case .mindfulness: return "blue"
        case .breathing: return "cyan"
        case .bodyScanning: return "green"
        case .lovingKindness: return "pink"
        case .visualization: return "purple"
        case .mantra: return "orange"
        case .walking: return "brown"
        case .sleep: return "indigo"
        case .painRelief: return "red"
        case .stressReduction: return "mint"
        case .anxietyRelief: return "teal"
        case .focusConcentration: return "yellow"
        case .emotionalHealing: return "pink"
        case .gratitude: return "orange"
        case .selfCompassion: return "green"
        }
    }
}

enum MeditationCategory: String, CaseIterable, Codable {
    case beginner = "beginner"
    case intermediate = "intermediate"
    case advanced = "advanced"
    case therapeutic = "therapeutic"
    case sleep = "sleep"
    case work = "work"
    case relationships = "relationships"
    case health = "health"
    case spirituality = "spirituality"
    case daily = "daily"
    
    var displayName: String {
        return rawValue.capitalized
    }
}

enum DifficultyLevel: String, CaseIterable, Codable {
    case beginner = "beginner"
    case intermediate = "intermediate"
    case advanced = "advanced"
    
    var displayName: String {
        return rawValue.capitalized
    }
    
    var color: String {
        switch self {
        case .beginner: return "green"
        case .intermediate: return "orange"
        case .advanced: return "red"
        }
    }
}

struct BackgroundSound: Identifiable, Codable {
    let id = UUID()
    let name: String
    let audioURL: URL
    let volume: Float
    let loop: Bool
    let fadeIn: TimeInterval
    let fadeOut: TimeInterval
    
    static let presets: [BackgroundSound] = [
        BackgroundSound(name: "Ocean Waves", audioURL: URL(string: "ocean_waves.mp3")!, volume: 0.5, loop: true, fadeIn: 2.0, fadeOut: 2.0),
        BackgroundSound(name: "Forest Rain", audioURL: URL(string: "forest_rain.mp3")!, volume: 0.4, loop: true, fadeIn: 3.0, fadeOut: 3.0),
        BackgroundSound(name: "Tibetan Bowls", audioURL: URL(string: "tibetan_bowls.mp3")!, volume: 0.3, loop: true, fadeIn: 1.0, fadeOut: 1.0),
        BackgroundSound(name: "White Noise", audioURL: URL(string: "white_noise.mp3")!, volume: 0.2, loop: true, fadeIn: 1.0, fadeOut: 1.0),
        BackgroundSound(name: "Birds Chirping", audioURL: URL(string: "birds_chirping.mp3")!, volume: 0.6, loop: true, fadeIn: 2.0, fadeOut: 2.0)
    ]
}

struct BreathingPattern: Codable {
    let inhaleCount: Int
    let holdCount: Int
    let exhaleCount: Int
    let pauseCount: Int
    let cycles: Int
    let name: String
    let description: String
    
    static let patterns: [BreathingPattern] = [
        BreathingPattern(inhaleCount: 4, holdCount: 4, exhaleCount: 4, pauseCount: 4, cycles: 10, name: "Box Breathing", description: "Equal counts for inhale, hold, exhale, and pause"),
        BreathingPattern(inhaleCount: 4, holdCount: 7, exhaleCount: 8, pauseCount: 0, cycles: 8, name: "4-7-8 Breathing", description: "Relaxing pattern for stress relief"),
        BreathingPattern(inhaleCount: 6, holdCount: 0, exhaleCount: 6, pauseCount: 0, cycles: 12, name: "Equal Breathing", description: "Simple equal inhale and exhale"),
        BreathingPattern(inhaleCount: 3, holdCount: 0, exhaleCount: 6, pauseCount: 0, cycles: 15, name: "Calming Breath", description: "Longer exhale for relaxation")
    ]
}

struct Visualization: Identifiable, Codable {
    let id = UUID()
    let title: String
    let description: String
    let script: String
    let duration: TimeInterval
    let imagePrompts: [String]
    let colorTheme: String
    let soundCues: [String]
}

struct MeditationProgress: Codable {
    let sessionId: UUID
    let userId: String
    let startTime: Date
    let endTime: Date?
    let duration: TimeInterval
    let completionPercentage: Float
    let heartRateData: [HeartRateReading]
    let stressLevel: StressLevel?
    let moodBefore: MoodLevel?
    let moodAfter: MoodLevel?
    let notes: String
    let interruptions: Int
    let qualityRating: Int // 1-5 stars
    let benefits: [String]
    let challenges: [String]
    let environment: MeditationEnvironment
    let deviceUsed: String
    let location: CLLocation?
}

struct HeartRateReading: Codable {
    let timestamp: Date
    let bpm: Double
    let hrv: Double?
}

enum StressLevel: String, CaseIterable, Codable {
    case veryLow = "very_low"
    case low = "low"
    case moderate = "moderate"
    case high = "high"
    case veryHigh = "very_high"
    
    var displayName: String {
        switch self {
        case .veryLow: return "Very Low"
        case .low: return "Low"
        case .moderate: return "Moderate"
        case .high: return "High"
        case .veryHigh: return "Very High"
        }
    }
    
    var color: String {
        switch self {
        case .veryLow: return "green"
        case .low: return "mint"
        case .moderate: return "yellow"
        case .high: return "orange"
        case .veryHigh: return "red"
        }
    }
}

enum MoodLevel: String, CaseIterable, Codable {
    case veryPoor = "very_poor"
    case poor = "poor"
    case neutral = "neutral"
    case good = "good"
    case excellent = "excellent"
    
    var displayName: String {
        switch self {
        case .veryPoor: return "Very Poor"
        case .poor: return "Poor"
        case .neutral: return "Neutral"
        case .good: return "Good"
        case .excellent: return "Excellent"
        }
    }
    
    var emoji: String {
        switch self {
        case .veryPoor: return "😢"
        case .poor: return "😕"
        case .neutral: return "😐"
        case .good: return "😊"
        case .excellent: return "😄"
        }
    }
    
    var color: String {
        switch self {
        case .veryPoor: return "red"
        case .poor: return "orange"
        case .neutral: return "gray"
        case .good: return "blue"
        case .excellent: return "green"
        }
    }
}

struct MeditationEnvironment: Codable {
    let location: String // "home", "office", "outdoor", "other"
    let noiseLevel: String // "silent", "quiet", "moderate", "noisy"
    let lighting: String // "dark", "dim", "natural", "bright"
    let temperature: String // "cold", "cool", "comfortable", "warm", "hot"
    let distractions: [String]
    let companions: Int // number of people present
}

struct MeditationStreak: Codable {
    let currentStreak: Int
    let longestStreak: Int
    let lastMeditationDate: Date?
    let totalSessions: Int
    let totalMinutes: TimeInterval
    let averageSessionLength: TimeInterval
    let favoriteType: MeditationType?
    let weeklyGoal: Int
    let monthlyGoal: Int
    let achievements: [MeditationAchievement]
}

struct MeditationAchievement: Identifiable, Codable {
    let id = UUID()
    let title: String
    let description: String
    let icon: String
    let unlockedDate: Date?
    let requirement: AchievementRequirement
    let category: AchievementCategory
    let points: Int
    
    var isUnlocked: Bool {
        return unlockedDate != nil
    }
}

enum AchievementRequirement: Codable {
    case sessionsCompleted(Int)
    case minutesMeditated(Int)
    case streakDays(Int)
    case typesExplored(Int)
    case consecutiveDays(Int)
    case perfectWeek
    case perfectMonth
    case earlyBird // meditate before 8 AM
    case nightOwl // meditate after 10 PM
    case mindfulMoments(Int) // short sessions
    case deepDive(Int) // long sessions
}

enum AchievementCategory: String, CaseIterable, Codable {
    case consistency = "consistency"
    case exploration = "exploration"
    case dedication = "dedication"
    case milestone = "milestone"
    case special = "special"
    
    var displayName: String {
        return rawValue.capitalized
    }
}

struct MeditationSettings: Codable {
    var reminderEnabled: Bool
    var reminderTimes: [Date]
    var preferredDuration: TimeInterval
    var favoriteTypes: [MeditationType]
    var backgroundSoundsEnabled: Bool
    var defaultBackgroundSound: BackgroundSound?
    var vibrationEnabled: Bool
    var voiceGuidanceEnabled: Bool
    var autoStartNext: Bool
    var trackHeartRate: Bool
    var shareProgress: Bool
    var offlineMode: Bool
    var downloadQuality: AudioQuality
    var interfaceTheme: InterfaceTheme
    var accessibilityOptions: AccessibilityOptions
    
    static let `default` = MeditationSettings(
        reminderEnabled: true,
        reminderTimes: [Calendar.current.date(bySettingHour: 8, minute: 0, second: 0, of: Date()) ?? Date()],
        preferredDuration: 600, // 10 minutes
        favoriteTypes: [.mindfulness, .breathing],
        backgroundSoundsEnabled: true,
        defaultBackgroundSound: BackgroundSound.presets.first,
        vibrationEnabled: true,
        voiceGuidanceEnabled: true,
        autoStartNext: false,
        trackHeartRate: true,
        shareProgress: false,
        offlineMode: false,
        downloadQuality: .high,
        interfaceTheme: .auto,
        accessibilityOptions: AccessibilityOptions.default
    )
}

enum AudioQuality: String, CaseIterable, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case lossless = "lossless"
    
    var displayName: String {
        return rawValue.capitalized
    }
    
    var bitrate: Int {
        switch self {
        case .low: return 64
        case .medium: return 128
        case .high: return 256
        case .lossless: return 1411
        }
    }
}

enum InterfaceTheme: String, CaseIterable, Codable {
    case light = "light"
    case dark = "dark"
    case auto = "auto"
    
    var displayName: String {
        return rawValue.capitalized
    }
}

struct AccessibilityOptions: Codable {
    var largeText: Bool
    var highContrast: Bool
    var reduceMotion: Bool
    var voiceOver: Bool
    var hapticFeedback: Bool
    var audioDescriptions: Bool
    var subtitles: Bool
    
    static let `default` = AccessibilityOptions(
        largeText: false,
        highContrast: false,
        reduceMotion: false,
        voiceOver: false,
        hapticFeedback: true,
        audioDescriptions: false,
        subtitles: false
    )
}

struct MeditationInsight: Identifiable, Codable {
    let id = UUID()
    let title: String
    let description: String
    let category: InsightCategory
    let data: InsightData
    let recommendations: [String]
    let generatedDate: Date
    let period: InsightPeriod
}

enum InsightCategory: String, CaseIterable, Codable {
    case progress = "progress"
    case patterns = "patterns"
    case health = "health"
    case mood = "mood"
    case stress = "stress"
    case sleep = "sleep"
    case focus = "focus"
    case consistency = "consistency"
    
    var displayName: String {
        return rawValue.capitalized
    }
    
    var systemImage: String {
        switch self {
        case .progress: return "chart.line.uptrend.xyaxis"
        case .patterns: return "waveform.path.ecg"
        case .health: return "heart.fill"
        case .mood: return "face.smiling.fill"
        case .stress: return "exclamationmark.triangle.fill"
        case .sleep: return "moon.fill"
        case .focus: return "target"
        case .consistency: return "calendar.badge.checkmark"
        }
    }
}

enum InsightPeriod: String, CaseIterable, Codable {
    case daily = "daily"
    case weekly = "weekly"
    case monthly = "monthly"
    case quarterly = "quarterly"
    case yearly = "yearly"
    
    var displayName: String {
        return rawValue.capitalized
    }
}

struct InsightData: Codable {
    let metrics: [String: Double]
    let trends: [String: TrendDirection]
    let comparisons: [String: ComparisonResult]
    let correlations: [String: Double]
}

enum TrendDirection: String, Codable {
    case increasing = "increasing"
    case decreasing = "decreasing"
    case stable = "stable"
    case fluctuating = "fluctuating"
    
    var systemImage: String {
        switch self {
        case .increasing: return "arrow.up.right"
        case .decreasing: return "arrow.down.right"
        case .stable: return "arrow.right"
        case .fluctuating: return "waveform"
        }
    }
    
    var color: String {
        switch self {
        case .increasing: return "green"
        case .decreasing: return "red"
        case .stable: return "blue"
        case .fluctuating: return "orange"
        }
    }
}

struct ComparisonResult: Codable {
    let currentValue: Double
    let previousValue: Double
    let percentageChange: Double
    let isImprovement: Bool
}

// MARK: - Error Types

enum MeditationError: LocalizedError {
    case audioSessionSetupFailed
    case audioFileNotFound(String)
    case audioPlaybackFailed
    case healthKitNotAuthorized
    case healthKitDataUnavailable
    case sessionNotFound
    case invalidDuration
    case networkUnavailable
    case downloadFailed(String)
    case storageError
    case permissionDenied
    case deviceNotSupported
    
    var errorDescription: String? {
        switch self {
        case .audioSessionSetupFailed:
            return "Failed to set up audio session"
        case .audioFileNotFound(let filename):
            return "Audio file not found: \(filename)"
        case .audioPlaybackFailed:
            return "Audio playback failed"
        case .healthKitNotAuthorized:
            return "HealthKit access not authorized"
        case .healthKitDataUnavailable:
            return "HealthKit data unavailable"
        case .sessionNotFound:
            return "Meditation session not found"
        case .invalidDuration:
            return "Invalid session duration"
        case .networkUnavailable:
            return "Network connection unavailable"
        case .downloadFailed(let session):
            return "Failed to download session: \(session)"
        case .storageError:
            return "Storage error occurred"
        case .permissionDenied:
            return "Permission denied"
        case .deviceNotSupported:
            return "Device not supported"
        }
    }
}

// MARK: - Main Manager Class

@MainActor
class MeditationManager: ObservableObject {
    static let shared = MeditationManager()
    
    // MARK: - Published Properties
    @Published var availableSessions: [MeditationSession] = []
    @Published var currentSession: MeditationSession?
    @Published var isPlaying = false
    @Published var isPaused = false
    @Published var currentTime: TimeInterval = 0
    @Published var progress: Float = 0
    @Published var volume: Float = 1.0
    @Published var playbackRate: Float = 1.0
    @Published var currentProgress: MeditationProgress?
    @Published var streak: MeditationStreak
    @Published var settings: MeditationSettings
    @Published var insights: [MeditationInsight] = []
    @Published var achievements: [MeditationAchievement] = []
    @Published var error: MeditationError?
    @Published var isLoading = false
    @Published var downloadProgress: [UUID: Float] = [:]
    
    // MARK: - Private Properties
    private var audioPlayer: AVAudioPlayer?
    private var backgroundAudioPlayer: AVAudioPlayer?
    private var timer: Timer?
    private var healthStore: HKHealthStore?
    private var heartRateQuery: HKAnchoredObjectQuery?
    private var cancellables = Set<AnyCancellable>()
    private let fileManager = FileManager.default
    private let userDefaults = UserDefaults.standard
    
    // MARK: - Managers
    private let audioManager: MeditationAudioManager
    private let progressTracker: MeditationProgressTracker
    private let insightGenerator: MeditationInsightGenerator
    private let achievementManager: MeditationAchievementManager
    private let reminderManager: MeditationReminderManager
    private let downloadManager: MeditationDownloadManager
    
    // MARK: - Initialization
    
    private init() {
        self.streak = MeditationStreak(
            currentStreak: 0,
            longestStreak: 0,
            lastMeditationDate: nil,
            totalSessions: 0,
            totalMinutes: 0,
            averageSessionLength: 0,
            favoriteType: nil,
            weeklyGoal: 7,
            monthlyGoal: 30,
            achievements: []
        )
        self.settings = MeditationSettings.default
        
        self.audioManager = MeditationAudioManager()
        self.progressTracker = MeditationProgressTracker()
        self.insightGenerator = MeditationInsightGenerator()
        self.achievementManager = MeditationAchievementManager()
        self.reminderManager = MeditationReminderManager()
        self.downloadManager = MeditationDownloadManager()
        
        setupHealthKit()
        loadData()
        setupAudioSession()
        loadAvailableSessions()
        setupNotifications()
    }
    
    // MARK: - Public Methods
    
    func startSession(_ session: MeditationSession) {
        guard !isPlaying else { return }
        
        currentSession = session
        currentTime = 0
        progress = 0
        
        // Create progress tracking
        currentProgress = MeditationProgress(
            sessionId: session.id,
            userId: "current_user", // Replace with actual user ID
            startTime: Date(),
            endTime: nil,
            duration: 0,
            completionPercentage: 0,
            heartRateData: [],
            stressLevel: nil,
            moodBefore: nil,
            moodAfter: nil,
            notes: "",
            interruptions: 0,
            qualityRating: 0,
            benefits: [],
            challenges: [],
            environment: getCurrentEnvironment(),
            deviceUsed: UIDevice.current.model,
            location: nil
        )
        
        // Start audio playback
        audioManager.playSession(session) { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success:
                    self?.isPlaying = true
                    self?.startTimer()
                    self?.startHeartRateMonitoring()
                case .failure(let error):
                    self?.error = error
                }
            }
        }
        
        // Start background sounds if enabled
        if settings.backgroundSoundsEnabled, let backgroundSound = settings.defaultBackgroundSound {
            audioManager.playBackgroundSound(backgroundSound)
        }
    }
    
    func pauseSession() {
        guard isPlaying else { return }
        
        audioManager.pause()
        isPlaying = false
        isPaused = true
        stopTimer()
        
        // Track interruption
        currentProgress?.interruptions += 1
    }
    
    func resumeSession() {
        guard isPaused else { return }
        
        audioManager.resume()
        isPlaying = true
        isPaused = false
        startTimer()
    }
    
    func stopSession() {
        audioManager.stop()
        isPlaying = false
        isPaused = false
        stopTimer()
        stopHeartRateMonitoring()
        
        // Complete progress tracking
        if var progress = currentProgress {
            progress.endTime = Date()
            progress.duration = currentTime
            progress.completionPercentage = self.progress
            
            // Save progress
            progressTracker.saveProgress(progress)
            
            // Update streak
            updateStreak()
            
            // Check achievements
            achievementManager.checkAchievements(progress: progress, streak: streak)
            
            // Generate insights
            Task {
                await generateInsights()
            }
        }
        
        currentSession = nil
        currentProgress = nil
        currentTime = 0
        progress = 0
    }
    
    func seekTo(_ time: TimeInterval) {
        audioManager.seekTo(time)
        currentTime = time
        updateProgress()
    }
    
    func setVolume(_ volume: Float) {
        self.volume = volume
        audioManager.setVolume(volume)
    }
    
    func setPlaybackRate(_ rate: Float) {
        self.playbackRate = rate
        audioManager.setPlaybackRate(rate)
    }
    
    func downloadSession(_ session: MeditationSession) {
        downloadManager.downloadSession(session) { [weak self] progress in
            DispatchQueue.main.async {
                self?.downloadProgress[session.id] = progress
            }
        } completion: { [weak self] result in
            DispatchQueue.main.async {
                self?.downloadProgress.removeValue(forKey: session.id)
                switch result {
                case .success:
                    // Session downloaded successfully
                    break
                case .failure(let error):
                    self?.error = error
                }
            }
        }
    }
    
    func deleteDownloadedSession(_ session: MeditationSession) {
        downloadManager.deleteDownloadedSession(session)
    }
    
    func isSessionDownloaded(_ session: MeditationSession) -> Bool {
        return downloadManager.isSessionDownloaded(session)
    }
    
    func updateSettings(_ newSettings: MeditationSettings) {
        settings = newSettings
        saveSettings()
        
        // Update reminders
        reminderManager.updateReminders(settings.reminderTimes, enabled: settings.reminderEnabled)
    }
    
    func getSessionsByType(_ type: MeditationType) -> [MeditationSession] {
        return availableSessions.filter { $0.type == type }
    }
    
    func getSessionsByCategory(_ category: MeditationCategory) -> [MeditationSession] {
        return availableSessions.filter { $0.category == category }
    }
    
    func getSessionsByDifficulty(_ difficulty: DifficultyLevel) -> [MeditationSession] {
        return availableSessions.filter { $0.difficulty == difficulty }
    }
    
    func searchSessions(_ query: String) -> [MeditationSession] {
        let lowercaseQuery = query.lowercased()
        return availableSessions.filter {
            $0.title.lowercased().contains(lowercaseQuery) ||
            $0.description.lowercased().contains(lowercaseQuery) ||
            $0.tags.contains { $0.lowercased().contains(lowercaseQuery) }
        }
    }
    
    func getRecommendedSessions() -> [MeditationSession] {
        // AI-powered recommendations based on user history, preferences, and current state
        var recommendations: [MeditationSession] = []
        
        // Add favorite types
        for type in settings.favoriteTypes {
            recommendations.append(contentsOf: getSessionsByType(type).prefix(2))
        }
        
        // Add sessions based on current time
        let hour = Calendar.current.component(.hour, from: Date())
        if hour < 10 {
            // Morning sessions
            recommendations.append(contentsOf: availableSessions.filter { $0.tags.contains("morning") }.prefix(2))
        } else if hour > 20 {
            // Evening/sleep sessions
            recommendations.append(contentsOf: getSessionsByType(.sleep).prefix(2))
        }
        
        // Add sessions based on recent mood/stress
        if let lastProgress = progressTracker.getRecentProgress().first {
            if lastProgress.stressLevel == .high || lastProgress.stressLevel == .veryHigh {
                recommendations.append(contentsOf: getSessionsByType(.stressReduction).prefix(2))
            }
        }
        
        return Array(Set(recommendations)).prefix(10).map { $0 }
    }
    
    // MARK: - Private Methods
    
    private func setupHealthKit() {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        
        healthStore = HKHealthStore()
        
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        
        healthStore?.requestAuthorization(toShare: nil, read: [heartRateType, hrvType]) { [weak self] success, error in
            if !success {
                DispatchQueue.main.async {
                    self?.error = .healthKitNotAuthorized
                }
            }
        }
    }
    
    private func setupAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [.allowBluetooth, .allowBluetoothA2DP])
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            self.error = .audioSessionSetupFailed
        }
    }
    
    private func loadData() {
        // Load streak data
        if let streakData = userDefaults.data(forKey: "meditation_streak"),
           let loadedStreak = try? JSONDecoder().decode(MeditationStreak.self, from: streakData) {
            streak = loadedStreak
        }
        
        // Load settings
        if let settingsData = userDefaults.data(forKey: "meditation_settings"),
           let loadedSettings = try? JSONDecoder().decode(MeditationSettings.self, from: settingsData) {
            settings = loadedSettings
        }
        
        // Load insights
        insights = insightGenerator.loadInsights()
        
        // Load achievements
        achievements = achievementManager.loadAchievements()
    }
    
    private func saveData() {
        // Save streak
        if let streakData = try? JSONEncoder().encode(streak) {
            userDefaults.set(streakData, forKey: "meditation_streak")
        }
        
        saveSettings()
    }
    
    private func saveSettings() {
        if let settingsData = try? JSONEncoder().encode(settings) {
            userDefaults.set(settingsData, forKey: "meditation_settings")
        }
    }
    
    private func loadAvailableSessions() {
        // Load predefined sessions
        availableSessions = createPredefinedSessions()
        
        // Load downloaded sessions
        availableSessions.append(contentsOf: downloadManager.getDownloadedSessions())
    }
    
    private func createPredefinedSessions() -> [MeditationSession] {
        return [
            MeditationSession(
                type: .mindfulness,
                title: "Basic Mindfulness",
                description: "A gentle introduction to mindfulness meditation",
                duration: 600, // 10 minutes
                category: .beginner
            ),
            MeditationSession(
                type: .breathing,
                title: "4-7-8 Breathing",
                description: "Calming breathing technique for stress relief",
                duration: 480, // 8 minutes
                category: .beginner
            ),
            MeditationSession(
                type: .bodyScanning,
                title: "Full Body Scan",
                description: "Progressive relaxation through body awareness",
                duration: 900, // 15 minutes
                category: .intermediate
            ),
            MeditationSession(
                type: .painRelief,
                title: "Pain Management",
                description: "Specialized meditation for chronic pain relief",
                duration: 1200, // 20 minutes
                category: .therapeutic
            ),
            MeditationSession(
                type: .sleep,
                title: "Deep Sleep",
                description: "Guided meditation for better sleep quality",
                duration: 1800, // 30 minutes
                category: .sleep
            )
        ]
    }
    
    private func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(audioSessionInterruption),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(audioSessionRouteChange),
            name: AVAudioSession.routeChangeNotification,
            object: nil
        )
    }
    
    @objc private func audioSessionInterruption(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }
        
        switch type {
        case .began:
            if isPlaying {
                pauseSession()
            }
        case .ended:
            if let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt {
                let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                if options.contains(.shouldResume) && isPaused {
                    resumeSession()
                }
            }
        @unknown default:
            break
        }
    }
    
    @objc private func audioSessionRouteChange(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
            return
        }
        
        switch reason {
        case .oldDeviceUnavailable:
            if isPlaying {
                pauseSession()
            }
        default:
            break
        }
    }
    
    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.updateTime()
        }
    }
    
    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
    
    private func updateTime() {
        guard let session = currentSession else { return }
        
        currentTime += 0.1
        updateProgress()
        
        // Check if session is complete
        if currentTime >= session.duration {
            stopSession()
        }
    }
    
    private func updateProgress() {
        guard let session = currentSession else { return }
        progress = Float(currentTime / session.duration)
        
        // Update current progress
        currentProgress?.duration = currentTime
        currentProgress?.completionPercentage = progress
    }
    
    private func startHeartRateMonitoring() {
        guard settings.trackHeartRate,
              let healthStore = healthStore,
              let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            return
        }
        
        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            DispatchQueue.main.async {
                for sample in samples {
                    let bpm = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                    let reading = HeartRateReading(timestamp: sample.startDate, bpm: bpm, hrv: nil)
                    self?.currentProgress?.heartRateData.append(reading)
                }
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            DispatchQueue.main.async {
                for sample in samples {
                    let bpm = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                    let reading = HeartRateReading(timestamp: sample.startDate, bpm: bpm, hrv: nil)
                    self?.currentProgress?.heartRateData.append(reading)
                }
            }
        }
        
        heartRateQuery = query
        healthStore.execute(query)
    }
    
    private func stopHeartRateMonitoring() {
        if let query = heartRateQuery {
            healthStore?.stop(query)
            heartRateQuery = nil
        }
    }
    
    private func updateStreak() {
        let today = Calendar.current.startOfDay(for: Date())
        
        if let lastDate = streak.lastMeditationDate {
            let lastMeditationDay = Calendar.current.startOfDay(for: lastDate)
            let daysBetween = Calendar.current.dateComponents([.day], from: lastMeditationDay, to: today).day ?? 0
            
            if daysBetween == 1 {
                // Consecutive day
                streak.currentStreak += 1
                if streak.currentStreak > streak.longestStreak {
                    streak.longestStreak = streak.currentStreak
                }
            } else if daysBetween > 1 {
                // Streak broken
                streak.currentStreak = 1
            }
            // If daysBetween == 0, it's the same day, don't change streak
        } else {
            // First meditation
            streak.currentStreak = 1
            streak.longestStreak = 1
        }
        
        streak.lastMeditationDate = Date()
        streak.totalSessions += 1
        
        if let progress = currentProgress {
            streak.totalMinutes += progress.duration
            streak.averageSessionLength = streak.totalMinutes / TimeInterval(streak.totalSessions)
        }
        
        saveData()
    }
    
    private func getCurrentEnvironment() -> MeditationEnvironment {
        return MeditationEnvironment(
            location: "home", // Could be determined by location services
            noiseLevel: "quiet", // Could be determined by microphone
            lighting: "natural", // Could be determined by camera
            temperature: "comfortable", // Could be determined by sensors
            distractions: [],
            companions: 0
        )
    }
    
    private func generateInsights() async {
        let newInsights = await insightGenerator.generateInsights(
            progress: progressTracker.getAllProgress(),
            streak: streak,
            settings: settings
        )
        
        DispatchQueue.main.async {
            self.insights = newInsights
        }
    }
}

// MARK: - Supporting Manager Classes

class MeditationAudioManager {
    private var audioPlayer: AVAudioPlayer?
    private var backgroundAudioPlayer: AVAudioPlayer?
    
    func playSession(_ session: MeditationSession, completion: @escaping (Result<Void, MeditationError>) -> Void) {
        // Implementation for playing meditation audio
        completion(.success(()))
    }
    
    func playBackgroundSound(_ sound: BackgroundSound) {
        // Implementation for playing background sounds
    }
    
    func pause() {
        audioPlayer?.pause()
        backgroundAudioPlayer?.pause()
    }
    
    func resume() {
        audioPlayer?.play()
        backgroundAudioPlayer?.play()
    }
    
    func stop() {
        audioPlayer?.stop()
        backgroundAudioPlayer?.stop()
    }
    
    func seekTo(_ time: TimeInterval) {
        audioPlayer?.currentTime = time
    }
    
    func setVolume(_ volume: Float) {
        audioPlayer?.volume = volume
    }
    
    func setPlaybackRate(_ rate: Float) {
        audioPlayer?.rate = rate
    }
}

class MeditationProgressTracker {
    private let userDefaults = UserDefaults.standard
    
    func saveProgress(_ progress: MeditationProgress) {
        var allProgress = getAllProgress()
        allProgress.append(progress)
        
        if let data = try? JSONEncoder().encode(allProgress) {
            userDefaults.set(data, forKey: "meditation_progress")
        }
    }
    
    func getAllProgress() -> [MeditationProgress] {
        guard let data = userDefaults.data(forKey: "meditation_progress"),
              let progress = try? JSONDecoder().decode([MeditationProgress].self, from: data) else {
            return []
        }
        return progress
    }
    
    func getRecentProgress(days: Int = 7) -> [MeditationProgress] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()
        return getAllProgress().filter { $0.startTime >= cutoffDate }
    }
    
    func getProgressForPeriod(_ period: InsightPeriod) -> [MeditationProgress] {
        let calendar = Calendar.current
        let now = Date()
        let startDate: Date
        
        switch period {
        case .daily:
            startDate = calendar.startOfDay(for: now)
        case .weekly:
            startDate = calendar.dateInterval(of: .weekOfYear, for: now)?.start ?? now
        case .monthly:
            startDate = calendar.dateInterval(of: .month, for: now)?.start ?? now
        case .quarterly:
            let quarter = (calendar.component(.month, from: now) - 1) / 3
            startDate = calendar.date(from: DateComponents(year: calendar.component(.year, from: now), month: quarter * 3 + 1)) ?? now
        case .yearly:
            startDate = calendar.dateInterval(of: .year, for: now)?.start ?? now
        }
        
        return getAllProgress().filter { $0.startTime >= startDate }
    }
}

class MeditationInsightGenerator {
    func generateInsights(progress: [MeditationProgress], streak: MeditationStreak, settings: MeditationSettings) async -> [MeditationInsight] {
        var insights: [MeditationInsight] = []
        
        // Generate progress insights
        insights.append(contentsOf: generateProgressInsights(progress: progress))
        
        // Generate pattern insights
        insights.append(contentsOf: generatePatternInsights(progress: progress))
        
        // Generate health insights
        insights.append(contentsOf: generateHealthInsights(progress: progress))
        
        // Generate consistency insights
        insights.append(contentsOf: generateConsistencyInsights(streak: streak))
        
        return insights
    }
    
    func loadInsights() -> [MeditationInsight] {
        guard let data = UserDefaults.standard.data(forKey: "meditation_insights"),
              let insights = try? JSONDecoder().decode([MeditationInsight].self, from: data) else {
            return []
        }
        return insights
    }
    
    private func generateProgressInsights(progress: [MeditationProgress]) -> [MeditationInsight] {
        // Implementation for generating progress-based insights
        return []
    }
    
    private func generatePatternInsights(progress: [MeditationProgress]) -> [MeditationInsight] {
        // Implementation for generating pattern-based insights
        return []
    }
    
    private func generateHealthInsights(progress: [MeditationProgress]) -> [MeditationInsight] {
        // Implementation for generating health-based insights
        return []
    }
    
    private func generateConsistencyInsights(streak: MeditationStreak) -> [MeditationInsight] {
        // Implementation for generating consistency-based insights
        return []
    }
}

class MeditationAchievementManager {
    func checkAchievements(progress: MeditationProgress, streak: MeditationStreak) {
        // Implementation for checking and unlocking achievements
    }
    
    func loadAchievements() -> [MeditationAchievement] {
        // Implementation for loading achievements
        return []
    }
}

class MeditationReminderManager {
    func updateReminders(_ times: [Date], enabled: Bool) {
        // Implementation for updating meditation reminders
    }
}

class MeditationDownloadManager {
    func downloadSession(_ session: MeditationSession, progress: @escaping (Float) -> Void, completion: @escaping (Result<Void, MeditationError>) -> Void) {
        // Implementation for downloading meditation sessions
        completion(.success(()))
    }
    
    func deleteDownloadedSession(_ session: MeditationSession) {
        // Implementation for deleting downloaded sessions
    }
    
    func isSessionDownloaded(_ session: MeditationSession) -> Bool {
        // Implementation for checking if session is downloaded
        return false
    }
    
    func getDownloadedSessions() -> [MeditationSession] {
        // Implementation for getting downloaded sessions
        return []
    }
}