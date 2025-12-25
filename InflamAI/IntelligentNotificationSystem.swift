//
//  IntelligentNotificationSystem.swift
//  InflamAI-Swift
//
//  Advanced notification system with AI-powered scheduling and contextual awareness
//

import Foundation
import UserNotifications
import Combine
import CoreLocation
import HealthKit
import EventKit
import CoreML
import SwiftUI
import WatchConnectivity
import BackgroundTasks
import CoreData
import CloudKit

// MARK: - Notification Models

struct SmartNotification: Codable, Identifiable {
    let id: String
    let type: NotificationType
    let priority: NotificationPriority
    let title: String
    let body: String
    let category: NotificationCategory
    let scheduledDate: Date
    let deliveryDate: Date?
    let isDelivered: Bool
    let isRead: Bool
    let isActionTaken: Bool
    let context: NotificationContext
    let personalization: PersonalizationData
    let actions: [NotificationAction]
    let attachments: [NotificationAttachment]
    let sound: NotificationSound
    let badge: Int?
    let expirationDate: Date?
    let retryCount: Int
    let maxRetries: Int
    let deliveryMethod: DeliveryMethod
    let targetDevices: [TargetDevice]
    let analytics: NotificationAnalytics
    let aiRecommendations: [AIRecommendation]
    
    var isExpired: Bool {
        guard let expirationDate = expirationDate else { return false }
        return Date() > expirationDate
    }
    
    var shouldRetry: Bool {
        return !isDelivered && retryCount < maxRetries && !isExpired
    }
}

enum NotificationType: String, Codable, CaseIterable {
    case medicationReminder = "medication_reminder"
    case symptomTracking = "symptom_tracking"
    case appointmentReminder = "appointment_reminder"
    case exerciseReminder = "exercise_reminder"
    case moodCheck = "mood_check"
    case flareUpAlert = "flare_up_alert"
    case weatherAlert = "weather_alert"
    case sleepReminder = "sleep_reminder"
    case hydrationReminder = "hydration_reminder"
    case stressManagement = "stress_management"
    case socialSupport = "social_support"
    case achievementCelebration = "achievement_celebration"
    case healthInsight = "health_insight"
    case emergencyAlert = "emergency_alert"
    case dataSync = "data_sync"
    case appUpdate = "app_update"
    case educationalContent = "educational_content"
    case motivationalMessage = "motivational_message"
    case habitReminder = "habit_reminder"
    case journalPrompt = "journal_prompt"
    
    var displayName: String {
        switch self {
        case .medicationReminder: return "Medication Reminder"
        case .symptomTracking: return "Symptom Check"
        case .appointmentReminder: return "Appointment Reminder"
        case .exerciseReminder: return "Exercise Reminder"
        case .moodCheck: return "Mood Check-in"
        case .flareUpAlert: return "Flare-up Alert"
        case .weatherAlert: return "Weather Alert"
        case .sleepReminder: return "Sleep Reminder"
        case .hydrationReminder: return "Hydration Reminder"
        case .stressManagement: return "Stress Management"
        case .socialSupport: return "Social Support"
        case .achievementCelebration: return "Achievement"
        case .healthInsight: return "Health Insight"
        case .emergencyAlert: return "Emergency Alert"
        case .dataSync: return "Data Sync"
        case .appUpdate: return "App Update"
        case .educationalContent: return "Educational Content"
        case .motivationalMessage: return "Motivation"
        case .habitReminder: return "Habit Reminder"
        case .journalPrompt: return "Journal Prompt"
        }
    }
    
    var defaultPriority: NotificationPriority {
        switch self {
        case .emergencyAlert, .flareUpAlert: return .critical
        case .medicationReminder, .appointmentReminder: return .high
        case .symptomTracking, .exerciseReminder, .moodCheck: return .medium
        case .weatherAlert, .sleepReminder, .hydrationReminder: return .medium
        case .stressManagement, .socialSupport, .healthInsight: return .low
        case .achievementCelebration, .motivationalMessage: return .low
        case .dataSync, .appUpdate, .educationalContent: return .low
        case .habitReminder, .journalPrompt: return .medium
        }
    }
    
    var category: NotificationCategory {
        switch self {
        case .medicationReminder: return .medication
        case .symptomTracking, .moodCheck: return .tracking
        case .appointmentReminder: return .healthcare
        case .exerciseReminder, .sleepReminder, .hydrationReminder: return .lifestyle
        case .flareUpAlert, .emergencyAlert: return .health
        case .weatherAlert: return .environmental
        case .stressManagement: return .wellness
        case .socialSupport: return .social
        case .achievementCelebration, .motivationalMessage: return .motivation
        case .healthInsight, .educationalContent: return .education
        case .dataSync, .appUpdate: return .system
        case .habitReminder, .journalPrompt: return .lifestyle
        }
    }
}

enum NotificationPriority: String, Codable, Comparable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
    
    static func < (lhs: NotificationPriority, rhs: NotificationPriority) -> Bool {
        let order: [NotificationPriority] = [.low, .medium, .high, .critical]
        guard let lhsIndex = order.firstIndex(of: lhs),
              let rhsIndex = order.firstIndex(of: rhs) else { return false }
        return lhsIndex < rhsIndex
    }
    
    var interruptionLevel: UNNotificationInterruptionLevel {
        switch self {
        case .low: return .passive
        case .medium: return .active
        case .high: return .timeSensitive
        case .critical: return .critical
        }
    }
    
    var deliveryDelay: TimeInterval {
        switch self {
        case .critical: return 0
        case .high: return 60 // 1 minute
        case .medium: return 300 // 5 minutes
        case .low: return 900 // 15 minutes
        }
    }
}

enum NotificationCategory: String, Codable {
    case medication = "medication"
    case tracking = "tracking"
    case healthcare = "healthcare"
    case lifestyle = "lifestyle"
    case health = "health"
    case environmental = "environmental"
    case wellness = "wellness"
    case social = "social"
    case motivation = "motivation"
    case education = "education"
    case system = "system"
    
    var color: String {
        switch self {
        case .medication: return "blue"
        case .tracking: return "green"
        case .healthcare: return "red"
        case .lifestyle: return "orange"
        case .health: return "purple"
        case .environmental: return "cyan"
        case .wellness: return "mint"
        case .social: return "pink"
        case .motivation: return "yellow"
        case .education: return "indigo"
        case .system: return "gray"
        }
    }
    
    var icon: String {
        switch self {
        case .medication: return "pills.fill"
        case .tracking: return "chart.line.uptrend.xyaxis"
        case .healthcare: return "cross.fill"
        case .lifestyle: return "figure.walk"
        case .health: return "heart.fill"
        case .environmental: return "cloud.sun.fill"
        case .wellness: return "leaf.fill"
        case .social: return "person.2.fill"
        case .motivation: return "star.fill"
        case .education: return "book.fill"
        case .system: return "gear"
        }
    }
}

struct NotificationContext: Codable {
    let userLocation: LocationContext?
    let timeContext: TimeContext
    let healthContext: HealthContext?
    let environmentalContext: EnvironmentalContext?
    let socialContext: SocialContext?
    let deviceContext: DeviceContext
    let appUsageContext: AppUsageContext?
    let calendarContext: CalendarContext?
    let activityContext: ActivityContext?
    
    var contextScore: Double {
        var score = 0.0
        var factors = 0
        
        if let location = userLocation {
            score += location.appropriatenessScore
            factors += 1
        }
        
        score += timeContext.appropriatenessScore
        factors += 1
        
        if let health = healthContext {
            score += health.appropriatenessScore
            factors += 1
        }
        
        if let environmental = environmentalContext {
            score += environmental.appropriatenessScore
            factors += 1
        }
        
        score += deviceContext.appropriatenessScore
        factors += 1
        
        return factors > 0 ? score / Double(factors) : 0.5
    }
}

struct LocationContext: Codable {
    let currentLocation: CLLocationCoordinate2D?
    let locationName: String?
    let locationType: LocationType
    let isAtHome: Bool
    let isAtWork: Bool
    let isAtHealthcareFacility: Bool
    let isMoving: Bool
    let speed: Double?
    
    var appropriatenessScore: Double {
        switch locationType {
        case .home: return 0.9
        case .work: return 0.7
        case .healthcare: return 0.8
        case .public: return 0.5
        case .moving: return 0.3
        case .unknown: return 0.5
        }
    }
}

enum LocationType: String, Codable {
    case home = "home"
    case work = "work"
    case healthcare = "healthcare"
    case public = "public"
    case moving = "moving"
    case unknown = "unknown"
}

struct TimeContext: Codable {
    let currentTime: Date
    let timeOfDay: TimeOfDay
    let dayOfWeek: DayOfWeek
    let isWeekend: Bool
    let isHoliday: Bool
    let userSleepSchedule: SleepSchedule?
    let userWorkSchedule: WorkSchedule?
    let timeZone: TimeZone
    
    var appropriatenessScore: Double {
        let hour = Calendar.current.component(.hour, from: currentTime)
        
        // Check if it's during sleep hours
        if let sleepSchedule = userSleepSchedule {
            if sleepSchedule.isSleepTime(hour) {
                return 0.1 // Very low score during sleep
            }
        }
        
        // Score based on time of day
        switch timeOfDay {
        case .morning: return 0.8
        case .afternoon: return 0.9
        case .evening: return 0.7
        case .night: return 0.3
        case .lateNight: return 0.1
        }
    }
}

enum TimeOfDay: String, Codable {
    case morning = "morning" // 6-12
    case afternoon = "afternoon" // 12-17
    case evening = "evening" // 17-22
    case night = "night" // 22-24
    case lateNight = "late_night" // 0-6
    
    static func from(hour: Int) -> TimeOfDay {
        switch hour {
        case 6..<12: return .morning
        case 12..<17: return .afternoon
        case 17..<22: return .evening
        case 22..<24: return .night
        default: return .lateNight
        }
    }
}

enum DayOfWeek: String, Codable {
    case monday = "monday"
    case tuesday = "tuesday"
    case wednesday = "wednesday"
    case thursday = "thursday"
    case friday = "friday"
    case saturday = "saturday"
    case sunday = "sunday"
}

struct SleepSchedule: Codable {
    let bedtime: Int // Hour (0-23)
    let wakeTime: Int // Hour (0-23)
    
    func isSleepTime(_ hour: Int) -> Bool {
        if bedtime < wakeTime {
            return hour >= bedtime || hour < wakeTime
        } else {
            return hour >= bedtime && hour < wakeTime
        }
    }
}

struct WorkSchedule: Codable {
    let startTime: Int // Hour (0-23)
    let endTime: Int // Hour (0-23)
    let workDays: [DayOfWeek]
    
    func isWorkTime(_ hour: Int, _ dayOfWeek: DayOfWeek) -> Bool {
        return workDays.contains(dayOfWeek) && hour >= startTime && hour < endTime
    }
}

struct HealthContext: Codable {
    let currentPainLevel: Double?
    let currentMood: Double?
    let currentEnergyLevel: Double?
    let recentSymptoms: [String]
    let medicationSchedule: [MedicationScheduleItem]
    let isInFlareUp: Bool
    let lastSymptomEntry: Date?
    let healthTrend: HealthTrend
    
    var appropriatenessScore: Double {
        var score = 0.5
        
        if isInFlareUp {
            score += 0.3 // Higher priority during flare-ups
        }
        
        if let painLevel = currentPainLevel {
            if painLevel > 7 {
                score += 0.2 // Higher priority for high pain
            } else if painLevel < 3 {
                score -= 0.1 // Lower priority for low pain
            }
        }
        
        if let mood = currentMood, mood < 4 {
            score += 0.1 // Slightly higher priority for low mood
        }
        
        return min(1.0, max(0.0, score))
    }
}

struct MedicationScheduleItem: Codable {
    let medicationName: String
    let scheduledTime: Date
    let isTaken: Bool
    let dosage: String
}

enum HealthTrend: String, Codable {
    case improving = "improving"
    case stable = "stable"
    case declining = "declining"
    case unknown = "unknown"
}

struct EnvironmentalContext: Codable {
    let weather: WeatherCondition?
    let airQuality: AirQualityLevel?
    let barometricPressure: Double?
    let humidity: Double?
    let temperature: Double?
    let uvIndex: Int?
    let pollenCount: PollenLevel?
    
    var appropriatenessScore: Double {
        var score = 0.5
        
        if let weather = weather {
            switch weather {
            case .sunny, .partlyCloudy: score += 0.1
            case .rainy, .stormy: score += 0.2 // Weather changes may affect symptoms
            case .snowy, .foggy: score += 0.1
            }
        }
        
        if let pressure = barometricPressure {
            if pressure < 29.8 || pressure > 30.2 {
                score += 0.2 // Pressure changes may affect symptoms
            }
        }
        
        return min(1.0, max(0.0, score))
    }
}

enum WeatherCondition: String, Codable {
    case sunny = "sunny"
    case partlyCloudy = "partly_cloudy"
    case cloudy = "cloudy"
    case rainy = "rainy"
    case stormy = "stormy"
    case snowy = "snowy"
    case foggy = "foggy"
}

enum AirQualityLevel: String, Codable {
    case good = "good"
    case moderate = "moderate"
    case unhealthyForSensitive = "unhealthy_for_sensitive"
    case unhealthy = "unhealthy"
    case veryUnhealthy = "very_unhealthy"
    case hazardous = "hazardous"
}

enum PollenLevel: String, Codable {
    case low = "low"
    case moderate = "moderate"
    case high = "high"
    case veryHigh = "very_high"
}

struct SocialContext: Codable {
    let isWithOthers: Bool
    let socialActivityLevel: SocialActivityLevel
    let recentSocialInteractions: Int
    let supportNetworkEngagement: Double
    let communityParticipation: Double
    
    var appropriatenessScore: Double {
        switch socialActivityLevel {
        case .high: return 0.7
        case .medium: return 0.8
        case .low: return 0.9
        case .none: return 0.6
        }
    }
}

enum SocialActivityLevel: String, Codable {
    case none = "none"
    case low = "low"
    case medium = "medium"
    case high = "high"
}

struct DeviceContext: Codable {
    let deviceType: DeviceType
    let batteryLevel: Double?
    let isCharging: Bool
    let screenBrightness: Double?
    let doNotDisturbEnabled: Bool
    let focusMode: FocusMode?
    let networkConnection: NetworkConnection
    let isLocked: Bool
    let lastInteraction: Date?
    
    var appropriatenessScore: Double {
        var score = 0.5
        
        if doNotDisturbEnabled {
            score -= 0.4
        }
        
        if let focusMode = focusMode {
            switch focusMode {
            case .sleep: score -= 0.5
            case .work: score -= 0.2
            case .personal: score += 0.1
            case .fitness: score += 0.2
            }
        }
        
        if let batteryLevel = batteryLevel, batteryLevel < 0.2 {
            score -= 0.1 // Lower priority when battery is low
        }
        
        return min(1.0, max(0.0, score))
    }
}

enum DeviceType: String, Codable {
    case iPhone = "iPhone"
    case iPad = "iPad"
    case appleWatch = "Apple Watch"
    case mac = "Mac"
}

enum FocusMode: String, Codable {
    case sleep = "sleep"
    case work = "work"
    case personal = "personal"
    case fitness = "fitness"
}

enum NetworkConnection: String, Codable {
    case wifi = "wifi"
    case cellular = "cellular"
    case none = "none"
}

struct AppUsageContext: Codable {
    let lastAppOpen: Date?
    let sessionDuration: TimeInterval?
    let dailyUsageTime: TimeInterval
    let weeklyUsageTime: TimeInterval
    let engagementLevel: EngagementLevel
    let lastDataEntry: Date?
    let streakDays: Int
    
    var appropriatenessScore: Double {
        switch engagementLevel {
        case .high: return 0.9
        case .medium: return 0.7
        case .low: return 0.5
        case .none: return 0.3
        }
    }
}

enum EngagementLevel: String, Codable {
    case none = "none"
    case low = "low"
    case medium = "medium"
    case high = "high"
}

struct CalendarContext: Codable {
    let upcomingEvents: [CalendarEvent]
    let isBusy: Bool
    let nextFreeSlot: Date?
    let hasHealthcareAppointments: Bool
    
    var appropriatenessScore: Double {
        return isBusy ? 0.3 : 0.8
    }
}

struct CalendarEvent: Codable {
    let title: String
    let startDate: Date
    let endDate: Date
    let isHealthRelated: Bool
}

struct ActivityContext: Codable {
    let currentActivity: ActivityType?
    let activityLevel: ActivityLevel
    let heartRate: Double?
    let steps: Int?
    let isExercising: Bool
    
    var appropriatenessScore: Double {
        if isExercising {
            return 0.2 // Low priority during exercise
        }
        
        switch activityLevel {
        case .sedentary: return 0.8
        case .light: return 0.7
        case .moderate: return 0.5
        case .vigorous: return 0.2
        }
    }
}

enum ActivityType: String, Codable {
    case walking = "walking"
    case running = "running"
    case cycling = "cycling"
    case swimming = "swimming"
    case yoga = "yoga"
    case meditation = "meditation"
    case sleeping = "sleeping"
    case working = "working"
    case driving = "driving"
    case unknown = "unknown"
}

enum ActivityLevel: String, Codable {
    case sedentary = "sedentary"
    case light = "light"
    case moderate = "moderate"
    case vigorous = "vigorous"
}

struct PersonalizationData: Codable {
    let userPreferences: UserNotificationPreferences
    let historicalEngagement: HistoricalEngagement
    let personalityProfile: PersonalityProfile?
    let communicationStyle: CommunicationStyle
    let motivationalFactors: [MotivationalFactor]
    let learningPreferences: LearningPreferences
    let accessibilityNeeds: AccessibilityNeeds
    
    var personalizationScore: Double {
        var score = 0.0
        score += userPreferences.preferenceScore * 0.3
        score += historicalEngagement.engagementScore * 0.3
        score += communicationStyle.effectivenessScore * 0.2
        score += learningPreferences.alignmentScore * 0.2
        return score
    }
}

struct UserNotificationPreferences: Codable {
    let preferredTimes: [TimeRange]
    let quietHours: [TimeRange]
    let preferredFrequency: NotificationFrequency
    let enabledCategories: [NotificationCategory]
    let disabledCategories: [NotificationCategory]
    let maxDailyNotifications: Int
    let preferredDeliveryMethod: DeliveryMethod
    let soundEnabled: Bool
    let vibrationEnabled: Bool
    let badgeEnabled: Bool
    
    var preferenceScore: Double {
        // Calculate based on how well current time aligns with preferences
        return 0.7 // Simplified
    }
}

struct TimeRange: Codable {
    let startHour: Int
    let endHour: Int
    
    func contains(hour: Int) -> Bool {
        if startHour <= endHour {
            return hour >= startHour && hour <= endHour
        } else {
            return hour >= startHour || hour <= endHour
        }
    }
}

enum NotificationFrequency: String, Codable {
    case minimal = "minimal"
    case low = "low"
    case medium = "medium"
    case high = "high"
    case maximum = "maximum"
    
    var maxDailyCount: Int {
        switch self {
        case .minimal: return 3
        case .low: return 6
        case .medium: return 10
        case .high: return 15
        case .maximum: return 25
        }
    }
}

struct HistoricalEngagement: Codable {
    let openRate: Double
    let actionRate: Double
    let dismissRate: Double
    let averageResponseTime: TimeInterval
    let preferredResponseTimes: [Int] // Hours
    let categoryEngagement: [NotificationCategory: Double]
    let typeEngagement: [NotificationType: Double]
    
    var engagementScore: Double {
        return (openRate * 0.4) + (actionRate * 0.6)
    }
}

struct PersonalityProfile: Codable {
    let openness: Double // 0-1
    let conscientiousness: Double // 0-1
    let extraversion: Double // 0-1
    let agreeableness: Double // 0-1
    let neuroticism: Double // 0-1
    let motivationStyle: MotivationStyle
    let communicationPreference: CommunicationPreference
}

enum MotivationStyle: String, Codable {
    case achievement = "achievement"
    case social = "social"
    case autonomy = "autonomy"
    case mastery = "mastery"
    case purpose = "purpose"
}

enum CommunicationPreference: String, Codable {
    case direct = "direct"
    case supportive = "supportive"
    case analytical = "analytical"
    case expressive = "expressive"
}

enum CommunicationStyle: String, Codable {
    case formal = "formal"
    case casual = "casual"
    case encouraging = "encouraging"
    case informative = "informative"
    case urgent = "urgent"
    
    var effectivenessScore: Double {
        // This would be calculated based on user response patterns
        return 0.7 // Simplified
    }
}

struct MotivationalFactor: Codable {
    let type: MotivationType
    let effectiveness: Double
    let frequency: Double
    let context: [String]
}

enum MotivationType: String, Codable {
    case progress = "progress"
    case social = "social"
    case achievement = "achievement"
    case health = "health"
    case routine = "routine"
    case reward = "reward"
    case fear = "fear"
    case curiosity = "curiosity"
}

struct LearningPreferences: Codable {
    let preferredContentType: [ContentType]
    let learningStyle: LearningStyle
    let attentionSpan: AttentionSpan
    let complexityPreference: ComplexityLevel
    
    var alignmentScore: Double {
        // Calculate based on content alignment with preferences
        return 0.6 // Simplified
    }
}

enum ContentType: String, Codable {
    case text = "text"
    case image = "image"
    case video = "video"
    case audio = "audio"
    case interactive = "interactive"
}

enum LearningStyle: String, Codable {
    case visual = "visual"
    case auditory = "auditory"
    case kinesthetic = "kinesthetic"
    case reading = "reading"
}

enum AttentionSpan: String, Codable {
    case short = "short" // < 30 seconds
    case medium = "medium" // 30 seconds - 2 minutes
    case long = "long" // > 2 minutes
}

enum ComplexityLevel: String, Codable {
    case simple = "simple"
    case moderate = "moderate"
    case complex = "complex"
}

struct AccessibilityNeeds: Codable {
    let visualImpairment: VisualImpairmentLevel
    let hearingImpairment: HearingImpairmentLevel
    let motorImpairment: MotorImpairmentLevel
    let cognitiveImpairment: CognitiveImpairmentLevel
    let preferredFontSize: FontSize
    let highContrastEnabled: Bool
    let voiceOverEnabled: Bool
    let reduceMotionEnabled: Bool
}

enum VisualImpairmentLevel: String, Codable {
    case none = "none"
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
    case blind = "blind"
}

enum HearingImpairmentLevel: String, Codable {
    case none = "none"
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
    case deaf = "deaf"
}

enum MotorImpairmentLevel: String, Codable {
    case none = "none"
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
}

enum CognitiveImpairmentLevel: String, Codable {
    case none = "none"
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
}

enum FontSize: String, Codable {
    case small = "small"
    case medium = "medium"
    case large = "large"
    case extraLarge = "extra_large"
}

struct NotificationAction: Codable {
    let id: String
    let title: String
    let type: ActionType
    let isDestructive: Bool
    let requiresAuthentication: Bool
    let deepLink: String?
    let parameters: [String: String]
    let icon: String?
    let accessibility: ActionAccessibility
}

enum ActionType: String, Codable {
    case markAsTaken = "mark_as_taken"
    case snooze = "snooze"
    case dismiss = "dismiss"
    case logSymptom = "log_symptom"
    case viewDetails = "view_details"
    case callDoctor = "call_doctor"
    case emergencyContact = "emergency_contact"
    case openApp = "open_app"
    case reschedule = "reschedule"
    case complete = "complete"
    case skip = "skip"
    case remind = "remind"
    case share = "share"
    case learn = "learn"
}

struct ActionAccessibility: Codable {
    let voiceOverLabel: String
    let voiceOverHint: String?
    let isAccessibilityElement: Bool
    let accessibilityTraits: [String]
}

struct NotificationAttachment: Codable {
    let id: String
    let type: AttachmentType
    let url: String
    let title: String?
    let subtitle: String?
    let thumbnailUrl: String?
    let accessibility: AttachmentAccessibility
}

enum AttachmentType: String, Codable {
    case image = "image"
    case video = "video"
    case audio = "audio"
    case document = "document"
    case chart = "chart"
    case map = "map"
}

struct AttachmentAccessibility: Codable {
    let alternativeText: String
    let description: String?
    let isDecorative: Bool
}

enum NotificationSound: String, Codable {
    case `default` = "default"
    case gentle = "gentle"
    case urgent = "urgent"
    case medication = "medication"
    case achievement = "achievement"
    case reminder = "reminder"
    case alert = "alert"
    case none = "none"
    case custom = "custom"
    
    var fileName: String? {
        switch self {
        case .default: return nil
        case .gentle: return "gentle_chime.wav"
        case .urgent: return "urgent_alert.wav"
        case .medication: return "medication_bell.wav"
        case .achievement: return "achievement_fanfare.wav"
        case .reminder: return "soft_reminder.wav"
        case .alert: return "health_alert.wav"
        case .none: return nil
        case .custom: return "custom_sound.wav"
        }
    }
}

enum DeliveryMethod: String, Codable {
    case push = "push"
    case inApp = "in_app"
    case email = "email"
    case sms = "sms"
    case watch = "watch"
    case widget = "widget"
    case lockScreen = "lock_screen"
    case banner = "banner"
    case alert = "alert"
}

enum TargetDevice: String, Codable {
    case iPhone = "iPhone"
    case iPad = "iPad"
    case appleWatch = "Apple Watch"
    case mac = "Mac"
    case all = "all"
}

struct NotificationAnalytics: Codable {
    let deliveryAttempts: Int
    let deliveryTime: Date?
    let openTime: Date?
    let actionTime: Date?
    let dismissTime: Date?
    let responseTime: TimeInterval?
    let engagementDuration: TimeInterval?
    let actionTaken: String?
    let contextAtDelivery: NotificationContext?
    let effectivenessScore: Double?
}

struct AIRecommendation: Codable {
    let type: RecommendationType
    let confidence: Double
    let reasoning: String
    let suggestedAction: String
    let impact: ImpactLevel
}

enum RecommendationType: String, Codable {
    case timing = "timing"
    case content = "content"
    case frequency = "frequency"
    case channel = "channel"
    case personalization = "personalization"
}

enum ImpactLevel: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
}

// MARK: - Intelligent Notification System

class IntelligentNotificationSystem: NSObject, ObservableObject {
    // Core Services
    private let notificationCenter = UNUserNotificationCenter.current()
    private let locationManager = CLLocationManager()
    private let healthStore = HKHealthStore()
    private let eventStore = EKEventStore()
    
    // Published Properties
    @Published var pendingNotifications: [SmartNotification] = []
    @Published var deliveredNotifications: [SmartNotification] = []
    @Published var notificationHistory: [SmartNotification] = []
    @Published var userPreferences: UserNotificationPreferences?
    @Published var isEnabled = true
    @Published var dailyNotificationCount = 0
    @Published var weeklyNotificationCount = 0
    
    // Internal State
    private var cancellables = Set<AnyCancellable>()
    private var contextCollectors: [ContextCollector] = []
    private var aiEngine: NotificationAIEngine?
    private var schedulingQueue = DispatchQueue(label: "notification.scheduling", qos: .userInitiated)
    private var deliveryQueue = DispatchQueue(label: "notification.delivery", qos: .userInitiated)
    
    // Configuration
    private let maxDailyNotifications = 20
    private let maxPendingNotifications = 50
    private let contextUpdateInterval: TimeInterval = 300 // 5 minutes
    private let analyticsRetentionDays = 90
    
    override init() {
        super.init()
        setupNotificationSystem()
        setupContextCollectors()
        setupAIEngine()
        requestPermissions()
    }
    
    // MARK: - Setup
    
    private func setupNotificationSystem() {
        notificationCenter.delegate = self
        
        // Setup notification categories
        setupNotificationCategories()
        
        // Setup periodic tasks
        setupPeriodicTasks()
        
        // Load user preferences
        loadUserPreferences()
        
        // Setup background tasks
        setupBackgroundTasks()
    }
    
    private func setupNotificationCategories() {
        var categories: Set<UNNotificationCategory> = []
        
        // Medication category
        let medicationActions = [
            UNNotificationAction(identifier: "MARK_TAKEN", title: "Mark as Taken", options: []),
            UNNotificationAction(identifier: "SNOOZE_5", title: "Snooze 5 min", options: []),
            UNNotificationAction(identifier: "SKIP", title: "Skip", options: [.destructive])
        ]
        categories.insert(UNNotificationCategory(
            identifier: "MEDICATION",
            actions: medicationActions,
            intentIdentifiers: [],
            options: [.customDismissAction]
        ))
        
        // Symptom tracking category
        let trackingActions = [
            UNNotificationAction(identifier: "LOG_SYMPTOM", title: "Log Now", options: [.foreground]),
            UNNotificationAction(identifier: "REMIND_LATER", title: "Remind Later", options: []),
            UNNotificationAction(identifier: "DISMISS", title: "Dismiss", options: [])
        ]
        categories.insert(UNNotificationCategory(
            identifier: "TRACKING",
            actions: trackingActions,
            intentIdentifiers: [],
            options: []
        ))
        
        // Emergency category
        let emergencyActions = [
            UNNotificationAction(identifier: "CALL_EMERGENCY", title: "Call Emergency", options: [.foreground]),
            UNNotificationAction(identifier: "CONTACT_DOCTOR", title: "Contact Doctor", options: [.foreground]),
            UNNotificationAction(identifier: "I_AM_OK", title: "I'm OK", options: [])
        ]
        categories.insert(UNNotificationCategory(
            identifier: "EMERGENCY",
            actions: emergencyActions,
            intentIdentifiers: [],
            options: [.customDismissAction]
        ))
        
        notificationCenter.setNotificationCategories(categories)
    }
    
    private func setupPeriodicTasks() {
        // Update context periodically
        Timer.publish(every: contextUpdateInterval, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.updateContext()
                }
            }
            .store(in: &cancellables)
        
        // Daily cleanup
        Timer.publish(every: 86400, on: .main, in: .common) // 24 hours
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.performDailyCleanup()
                }
            }
            .store(in: &cancellables)
        
        // Reset daily counters at midnight
        Timer.publish(every: 3600, on: .main, in: .common) // Every hour
            .autoconnect()
            .sink { [weak self] _ in
                self?.checkForMidnight()
            }
            .store(in: &cancellables)
    }
    
    private func setupContextCollectors() {
        contextCollectors = [
            LocationContextCollector(locationManager: locationManager),
            HealthContextCollector(healthStore: healthStore),
            DeviceContextCollector(),
            CalendarContextCollector(eventStore: eventStore),
            EnvironmentalContextCollector()
        ]
    }
    
    private func setupAIEngine() {
        aiEngine = NotificationAIEngine()
    }
    
    private func setupBackgroundTasks() {
        // Register background task for notification processing
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.inflamai.notification.processing",
            using: nil
        ) { [weak self] task in
            self?.handleBackgroundNotificationProcessing(task as! BGProcessingTask)
        }
    }
    
    private func requestPermissions() {
        let options: UNAuthorizationOptions = [.alert, .sound, .badge, .criticalAlert, .timeSensitive]
        
        notificationCenter.requestAuthorization(options: options) { [weak self] granted, error in
            DispatchQueue.main.async {
                if granted {
                    self?.isEnabled = true
                } else {
                    self?.isEnabled = false
                    print("Notification permission denied: \(error?.localizedDescription ?? "Unknown error")")
                }
            }
        }
        
        // Request location permission
        locationManager.requestWhenInUseAuthorization()
        
        // Request HealthKit permission
        if HKHealthStore.isHealthDataAvailable() {
            let typesToRead: Set<HKObjectType> = [
                HKObjectType.quantityType(forIdentifier: .heartRate)!,
                HKObjectType.quantityType(forIdentifier: .stepCount)!,
                HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!
            ]
            
            healthStore.requestAuthorization(toShare: [], read: typesToRead) { success, error in
                if !success {
                    print("HealthKit authorization failed: \(error?.localizedDescription ?? "Unknown error")")
                }
            }
        }
        
        // Request calendar permission
        eventStore.requestAccess(to: .event) { granted, error in
            if !granted {
                print("Calendar permission denied: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }
    
    // MARK: - Context Collection
    
    private func updateContext() async {
        // Collect context from all collectors
        await withTaskGroup(of: Void.self) { group in
            for collector in contextCollectors {
                group.addTask {
                    await collector.updateContext()
                }
            }
        }
    }
    
    private func getCurrentContext() async -> NotificationContext {
        let timeContext = TimeContext(
            currentTime: Date(),
            timeOfDay: TimeOfDay.from(hour: Calendar.current.component(.hour, from: Date())),
            dayOfWeek: DayOfWeek(rawValue: Calendar.current.component(.weekday, from: Date()).description) ?? .monday,
            isWeekend: Calendar.current.isDateInWeekend(Date()),
            isHoliday: false, // Would check against holiday calendar
            userSleepSchedule: userPreferences?.quietHours.first.map { SleepSchedule(bedtime: $0.startHour, wakeTime: $0.endHour) },
            userWorkSchedule: nil, // Would be configured by user
            timeZone: TimeZone.current
        )
        
        let deviceContext = DeviceContext(
            deviceType: .iPhone, // Would detect actual device
            batteryLevel: UIDevice.current.batteryLevel >= 0 ? Double(UIDevice.current.batteryLevel) : nil,
            isCharging: UIDevice.current.batteryState == .charging,
            screenBrightness: UIScreen.main.brightness,
            doNotDisturbEnabled: false, // Would check actual DND status
            focusMode: nil, // Would check actual focus mode
            networkConnection: .wifi, // Would check actual connection
            isLocked: false, // Would check actual lock status
            lastInteraction: Date()
        )
        
        return NotificationContext(
            userLocation: nil, // Would be collected by LocationContextCollector
            timeContext: timeContext,
            healthContext: nil, // Would be collected by HealthContextCollector
            environmentalContext: nil, // Would be collected by EnvironmentalContextCollector
            socialContext: nil, // Would be inferred from usage patterns
            deviceContext: deviceContext,
            appUsageContext: nil, // Would track app usage
            calendarContext: nil, // Would be collected by CalendarContextCollector
            activityContext: nil // Would be collected from HealthKit
        )
    }
    
    // MARK: - Notification Scheduling
    
    func scheduleNotification(_ notification: SmartNotification) async {
        guard isEnabled && dailyNotificationCount < maxDailyNotifications else {
            print("Notification limit reached or notifications disabled")
            return
        }
        
        // Get current context
        let context = await getCurrentContext()
        
        // Use AI to optimize timing and content
        let optimizedNotification = await aiEngine?.optimizeNotification(notification, context: context) ?? notification
        
        // Calculate optimal delivery time
        let deliveryTime = await calculateOptimalDeliveryTime(for: optimizedNotification, context: context)
        
        // Create notification request
        let request = await createNotificationRequest(for: optimizedNotification, deliveryTime: deliveryTime)
        
        // Schedule with system
        do {
            try await notificationCenter.add(request)
            
            // Add to pending notifications
            DispatchQueue.main.async {
                self.pendingNotifications.append(optimizedNotification)
                self.dailyNotificationCount += 1
            }
            
            print("Notification scheduled: \(optimizedNotification.title)")
        } catch {
            print("Failed to schedule notification: \(error.localizedDescription)")
        }
    }
    
    private func calculateOptimalDeliveryTime(for notification: SmartNotification, context: NotificationContext) async -> Date {
        var deliveryTime = notification.scheduledDate
        
        // Apply priority-based delay
        deliveryTime = deliveryTime.addingTimeInterval(notification.priority.deliveryDelay)
        
        // Consider user preferences
        if let preferences = userPreferences {
            deliveryTime = adjustForUserPreferences(deliveryTime, preferences: preferences)
        }
        
        // Consider context appropriateness
        if context.contextScore < 0.5 {
            // Delay if context is not appropriate
            deliveryTime = findNextAppropriateTime(after: deliveryTime, context: context)
        }
        
        // Use AI recommendations
        if let aiRecommendation = await aiEngine?.recommendDeliveryTime(for: notification, context: context) {
            deliveryTime = aiRecommendation
        }
        
        return deliveryTime
    }
    
    private func adjustForUserPreferences(_ time: Date, preferences: UserNotificationPreferences) -> Date {
        let hour = Calendar.current.component(.hour, from: time)
        
        // Check if time is in quiet hours
        for quietHour in preferences.quietHours {
            if quietHour.contains(hour: hour) {
                // Move to next preferred time
                if let nextPreferredTime = preferences.preferredTimes.first {
                    var components = Calendar.current.dateComponents([.year, .month, .day], from: time)
                    components.hour = nextPreferredTime.startHour
                    components.minute = 0
                    
                    if let adjustedTime = Calendar.current.date(from: components) {
                        return adjustedTime > time ? adjustedTime : Calendar.current.date(byAdding: .day, value: 1, to: adjustedTime)!
                    }
                }
            }
        }
        
        return time
    }
    
    private func findNextAppropriateTime(after time: Date, context: NotificationContext) -> Date {
        // Simple implementation - find next hour with better context score
        var nextTime = time
        
        for _ in 0..<24 { // Check next 24 hours
            nextTime = Calendar.current.date(byAdding: .hour, value: 1, to: nextTime)!
            
            // Would recalculate context for this time
            // For now, just return time + 1 hour
            break
        }
        
        return nextTime
    }
    
    private func createNotificationRequest(for notification: SmartNotification, deliveryTime: Date) async -> UNNotificationRequest {
        let content = UNMutableNotificationContent()
        content.title = notification.title
        content.body = notification.body
        content.categoryIdentifier = notification.category.rawValue.uppercased()
        content.userInfo = createUserInfo(for: notification)
        content.badge = notification.badge as NSNumber?
        content.interruptionLevel = notification.priority.interruptionLevel
        
        // Add sound
        if let soundFileName = notification.sound.fileName {
            content.sound = UNNotificationSound(named: UNNotificationSoundName(soundFileName))
        } else if notification.sound != .none {
            content.sound = .default
        }
        
        // Add attachments
        content.attachments = await createAttachments(for: notification)
        
        // Create trigger
        let trigger = UNTimeIntervalNotificationTrigger(
            timeInterval: deliveryTime.timeIntervalSinceNow,
            repeats: false
        )
        
        return UNNotificationRequest(
            identifier: notification.id,
            content: content,
            trigger: trigger
        )
    }
    
    private func createUserInfo(for notification: SmartNotification) -> [String: Any] {
        var userInfo: [String: Any] = [
            "notificationId": notification.id,
            "type": notification.type.rawValue,
            "priority": notification.priority.rawValue,
            "category": notification.category.rawValue
        ]
        
        // Add custom data based on notification type
        switch notification.type {
        case .medicationReminder:
            userInfo["medicationName"] = "" // Would be extracted from notification
            userInfo["dosage"] = "" // Would be extracted from notification
        case .appointmentReminder:
            userInfo["appointmentTime"] = "" // Would be extracted from notification
            userInfo["doctorName"] = "" // Would be extracted from notification
        default:
            break
        }
        
        return userInfo
    }
    
    private func createAttachments(for notification: SmartNotification) async -> [UNNotificationAttachment] {
        var attachments: [UNNotificationAttachment] = []
        
        for attachment in notification.attachments {
            if let url = URL(string: attachment.url),
               let data = try? Data(contentsOf: url),
               let tempURL = saveToTempFile(data: data, filename: "attachment_\(attachment.id)") {
                
                do {
                    let notificationAttachment = try UNNotificationAttachment(
                        identifier: attachment.id,
                        url: tempURL,
                        options: nil
                    )
                    attachments.append(notificationAttachment)
                } catch {
                    print("Failed to create attachment: \(error.localizedDescription)")
                }
            }
        }
        
        return attachments
    }
    
    private func saveToTempFile(data: Data, filename: String) -> URL? {
        let tempDir = FileManager.default.temporaryDirectory
        let tempURL = tempDir.appendingPathComponent(filename)
        
        do {
            try data.write(to: tempURL)
            return tempURL
        } catch {
            print("Failed to save temp file: \(error.localizedDescription)")
            return nil
        }
    }
    
    // MARK: - Notification Management
    
    func cancelNotification(id: String) {
        notificationCenter.removePendingNotificationRequests(withIdentifiers: [id])
        
        DispatchQueue.main.async {
            self.pendingNotifications.removeAll { $0.id == id }
        }
    }
    
    func cancelAllNotifications() {
        notificationCenter.removeAllPendingNotificationRequests()
        
        DispatchQueue.main.async {
            self.pendingNotifications.removeAll()
        }
    }
    
    func snoozeNotification(id: String, duration: TimeInterval) async {
        guard let notification = pendingNotifications.first(where: { $0.id == id }) else { return }
        
        // Cancel current notification
        cancelNotification(id: id)
        
        // Reschedule with new time
        var snoozedNotification = notification
        snoozedNotification = SmartNotification(
            id: UUID().uuidString,
            type: notification.type,
            priority: notification.priority,
            title: notification.title,
            body: notification.body,
            category: notification.category,
            scheduledDate: Date().addingTimeInterval(duration),
            deliveryDate: nil,
            isDelivered: false,
            isRead: false,
            isActionTaken: false,
            context: notification.context,
            personalization: notification.personalization,
            actions: notification.actions,
            attachments: notification.attachments,
            sound: notification.sound,
            badge: notification.badge,
            expirationDate: notification.expirationDate,
            retryCount: notification.retryCount + 1,
            maxRetries: notification.maxRetries,
            deliveryMethod: notification.deliveryMethod,
            targetDevices: notification.targetDevices,
            analytics: notification.analytics,
            aiRecommendations: notification.aiRecommendations
        )
        
        await scheduleNotification(snoozedNotification)
    }
    
    // MARK: - Analytics and Learning
    
    private func trackNotificationEngagement(_ notification: SmartNotification, action: String) {
        var updatedNotification = notification
        var analytics = notification.analytics
        
        switch action {
        case "opened":
            analytics = NotificationAnalytics(
                deliveryAttempts: analytics.deliveryAttempts,
                deliveryTime: analytics.deliveryTime,
                openTime: Date(),
                actionTime: analytics.actionTime,
                dismissTime: analytics.dismissTime,
                responseTime: analytics.deliveryTime != nil ? Date().timeIntervalSince(analytics.deliveryTime!) : nil,
                engagementDuration: nil,
                actionTaken: analytics.actionTaken,
                contextAtDelivery: analytics.contextAtDelivery,
                effectivenessScore: analytics.effectivenessScore
            )
        case "dismissed":
            analytics = NotificationAnalytics(
                deliveryAttempts: analytics.deliveryAttempts,
                deliveryTime: analytics.deliveryTime,
                openTime: analytics.openTime,
                actionTime: analytics.actionTime,
                dismissTime: Date(),
                responseTime: analytics.responseTime,
                engagementDuration: analytics.openTime != nil ? Date().timeIntervalSince(analytics.openTime!) : nil,
                actionTaken: analytics.actionTaken,
                contextAtDelivery: analytics.contextAtDelivery,
                effectivenessScore: calculateEffectivenessScore(analytics)
            )
        default:
            analytics = NotificationAnalytics(
                deliveryAttempts: analytics.deliveryAttempts,
                deliveryTime: analytics.deliveryTime,
                openTime: analytics.openTime,
                actionTime: Date(),
                dismissTime: analytics.dismissTime,
                responseTime: analytics.deliveryTime != nil ? Date().timeIntervalSince(analytics.deliveryTime!) : nil,
                engagementDuration: analytics.openTime != nil ? Date().timeIntervalSince(analytics.openTime!) : nil,
                actionTaken: action,
                contextAtDelivery: analytics.contextAtDelivery,
                effectivenessScore: calculateEffectivenessScore(analytics)
            )
        }
        
        // Update notification in history
        DispatchQueue.main.async {
            if let index = self.notificationHistory.firstIndex(where: { $0.id == notification.id }) {
                self.notificationHistory[index] = SmartNotification(
                    id: notification.id,
                    type: notification.type,
                    priority: notification.priority,
                    title: notification.title,
                    body: notification.body,
                    category: notification.category,
                    scheduledDate: notification.scheduledDate,
                    deliveryDate: notification.deliveryDate,
                    isDelivered: notification.isDelivered,
                    isRead: action == "opened" || notification.isRead,
                    isActionTaken: action != "opened" && action != "dismissed" || notification.isActionTaken,
                    context: notification.context,
                    personalization: notification.personalization,
                    actions: notification.actions,
                    attachments: notification.attachments,
                    sound: notification.sound,
                    badge: notification.badge,
                    expirationDate: notification.expirationDate,
                    retryCount: notification.retryCount,
                    maxRetries: notification.maxRetries,
                    deliveryMethod: notification.deliveryMethod,
                    targetDevices: notification.targetDevices,
                    analytics: analytics,
                    aiRecommendations: notification.aiRecommendations
                )
            }
        }
        
        // Send analytics to AI engine for learning
        Task {
            await aiEngine?.updateLearningModel(notification: updatedNotification, action: action)
        }
    }
    
    private func calculateEffectivenessScore(_ analytics: NotificationAnalytics) -> Double {
        var score = 0.0
        
        // Base score for delivery
        if analytics.deliveryTime != nil {
            score += 0.3
        }
        
        // Score for opening
        if analytics.openTime != nil {
            score += 0.4
        }
        
        // Score for taking action
        if analytics.actionTaken != nil && analytics.actionTaken != "dismissed" {
            score += 0.3
        }
        
        // Adjust for response time
        if let responseTime = analytics.responseTime {
            if responseTime < 300 { // 5 minutes
                score += 0.1
            } else if responseTime > 3600 { // 1 hour
                score -= 0.1
            }
        }
        
        return max(0.0, min(1.0, score))
    }
    
    // MARK: - User Preferences
    
    private func loadUserPreferences() {
        // Load from UserDefaults or Core Data
        if let data = UserDefaults.standard.data(forKey: "notificationPreferences"),
           let preferences = try? JSONDecoder().decode(UserNotificationPreferences.self, from: data) {
            self.userPreferences = preferences
        } else {
            // Set default preferences
            self.userPreferences = createDefaultPreferences()
        }
    }
    
    func createDefaultPreferences() -> UserNotificationPreferences {
        return UserNotificationPreferences(
            preferredTimes: [
                TimeRange(startHour: 8, endHour: 10),
                TimeRange(startHour: 12, endHour: 14),
                TimeRange(startHour: 18, endHour: 20)
            ],
            quietHours: [
                TimeRange(startHour: 22, endHour: 7)
            ],
            preferredFrequency: .medium,
            enabledCategories: NotificationCategory.allCases,
            disabledCategories: [],
            maxDailyNotifications: 15,
            preferredDeliveryMethod: .push,
            soundEnabled: true,
            vibrationEnabled: true,
            badgeEnabled: true
        )
    }
    
    func updateUserPreferences(_ preferences: UserNotificationPreferences) {
        self.userPreferences = preferences
        
        // Save to UserDefaults
        if let data = try? JSONEncoder().encode(preferences) {
            UserDefaults.standard.set(data, forKey: "notificationPreferences")
        }
    }
    
    // MARK: - Background Processing
    
    private func handleBackgroundNotificationProcessing(_ task: BGProcessingTask) {
        task.expirationHandler = {
            task.setTaskCompleted(success: false)
        }
        
        Task {
            await updateContext()
            await processRetryNotifications()
            await performDailyCleanup()
            
            task.setTaskCompleted(success: true)
        }
    }
    
    private func processRetryNotifications() async {
        let retryNotifications = pendingNotifications.filter { $0.shouldRetry }
        
        for notification in retryNotifications {
            await scheduleNotification(notification)
        }
    }
    
    private func performDailyCleanup() async {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -analyticsRetentionDays, to: Date())!
        
        DispatchQueue.main.async {
            self.notificationHistory.removeAll { $0.scheduledDate < cutoffDate }
        }
    }
    
    private func checkForMidnight() {
        let hour = Calendar.current.component(.hour, from: Date())
        if hour == 0 {
            dailyNotificationCount = 0
        }
    }
    
    // MARK: - Public API
    
    func createMedicationReminder(
        medicationName: String,
        dosage: String,
        scheduledTime: Date
    ) async {
        let notification = SmartNotification(
            id: UUID().uuidString,
            type: .medicationReminder,
            priority: .high,
            title: "Time for \(medicationName)",
            body: "Take \(dosage) of \(medicationName)",
            category: .medication,
            scheduledDate: scheduledTime,
            deliveryDate: nil,
            isDelivered: false,
            isRead: false,
            isActionTaken: false,
            context: await getCurrentContext(),
            personalization: createDefaultPersonalization(),
            actions: [
                NotificationAction(
                    id: "mark_taken",
                    title: "Mark as Taken",
                    type: .markAsTaken,
                    isDestructive: false,
                    requiresAuthentication: false,
                    deepLink: "inflamai://medication/taken",
                    parameters: ["medication": medicationName],
                    icon: "checkmark.circle.fill",
                    accessibility: ActionAccessibility(
                        voiceOverLabel: "Mark \(medicationName) as taken",
                        voiceOverHint: "Double tap to confirm you have taken this medication",
                        isAccessibilityElement: true,
                        accessibilityTraits: ["button"]
                    )
                ),
                NotificationAction(
                    id: "snooze",
                    title: "Snooze 15 min",
                    type: .snooze,
                    isDestructive: false,
                    requiresAuthentication: false,
                    deepLink: nil,
                    parameters: ["duration": "900"],
                    icon: "clock.fill",
                    accessibility: ActionAccessibility(
                        voiceOverLabel: "Snooze medication reminder",
                        voiceOverHint: "Double tap to be reminded again in 15 minutes",
                        isAccessibilityElement: true,
                        accessibilityTraits: ["button"]
                    )
                )
            ],
            attachments: [],
            sound: .medication,
            badge: 1,
            expirationDate: Calendar.current.date(byAdding: .hour, value: 2, to: scheduledTime),
            retryCount: 0,
            maxRetries: 3,
            deliveryMethod: .push,
            targetDevices: [.iPhone, .appleWatch],
            analytics: NotificationAnalytics(
                deliveryAttempts: 0,
                deliveryTime: nil,
                openTime: nil,
                actionTime: nil,
                dismissTime: nil,
                responseTime: nil,
                engagementDuration: nil,
                actionTaken: nil,
                contextAtDelivery: nil,
                effectivenessScore: nil
            ),
            aiRecommendations: []
        )
        
        await scheduleNotification(notification)
    }
    
    func createSymptomTrackingReminder(scheduledTime: Date) async {
        let notification = SmartNotification(
            id: UUID().uuidString,
            type: .symptomTracking,
            priority: .medium,
            title: "How are you feeling?",
            body: "Take a moment to log your current symptoms and pain level",
            category: .tracking,
            scheduledDate: scheduledTime,
            deliveryDate: nil,
            isDelivered: false,
            isRead: false,
            isActionTaken: false,
            context: await getCurrentContext(),
            personalization: createDefaultPersonalization(),
            actions: [
                NotificationAction(
                    id: "log_symptoms",
                    title: "Log Symptoms",
                    type: .logSymptom,
                    isDestructive: false,
                    requiresAuthentication: false,
                    deepLink: "inflamai://symptoms/log",
                    parameters: [:],
                    icon: "heart.text.square.fill",
                    accessibility: ActionAccessibility(
                        voiceOverLabel: "Log symptoms",
                        voiceOverHint: "Double tap to open symptom tracking",
                        isAccessibilityElement: true,
                        accessibilityTraits: ["button"]
                    )
                )
            ],
            attachments: [],
            sound: .gentle,
            badge: nil,
            expirationDate: Calendar.current.date(byAdding: .hour, value: 4, to: scheduledTime),
            retryCount: 0,
            maxRetries: 2,
            deliveryMethod: .push,
            targetDevices: [.iPhone],
            analytics: NotificationAnalytics(
                deliveryAttempts: 0,
                deliveryTime: nil,
                openTime: nil,
                actionTime: nil,
                dismissTime: nil,
                responseTime: nil,
                engagementDuration: nil,
                actionTaken: nil,
                contextAtDelivery: nil,
                effectivenessScore: nil
            ),
            aiRecommendations: []
        )
        
        await scheduleNotification(notification)
    }
    
    func createFlareUpAlert(severity: String, predictedTime: Date?) async {
        let timeText = predictedTime != nil ? "in the next few hours" : "soon"
        
        let notification = SmartNotification(
            id: UUID().uuidString,
            type: .flareUpAlert,
            priority: .critical,
            title: "Potential Flare-Up Detected",
            body: "Our AI predicts a \(severity) flare-up may occur \(timeText). Consider taking preventive measures.",
            category: .health,
            scheduledDate: Date(),
            deliveryDate: nil,
            isDelivered: false,
            isRead: false,
            isActionTaken: false,
            context: await getCurrentContext(),
            personalization: createDefaultPersonalization(),
            actions: [
                NotificationAction(
                    id: "view_recommendations",
                    title: "View Recommendations",
                    type: .viewDetails,
                    isDestructive: false,
                    requiresAuthentication: false,
                    deepLink: "inflamai://flareup/recommendations",
                    parameters: ["severity": severity],
                    icon: "lightbulb.fill",
                    accessibility: ActionAccessibility(
                        voiceOverLabel: "View flare-up recommendations",
                        voiceOverHint: "Double tap to see suggested preventive actions",
                        isAccessibilityElement: true,
                        accessibilityTraits: ["button"]
                    )
                ),
                NotificationAction(
                    id: "contact_doctor",
                    title: "Contact Doctor",
                    type: .callDoctor,
                    isDestructive: false,
                    requiresAuthentication: false,
                    deepLink: "inflamai://contact/doctor",
                    parameters: [:],
                    icon: "phone.fill",
                    accessibility: ActionAccessibility(
                        voiceOverLabel: "Contact doctor",
                        voiceOverHint: "Double tap to contact your healthcare provider",
                        isAccessibilityElement: true,
                        accessibilityTraits: ["button"]
                    )
                )
            ],
            attachments: [],
            sound: .urgent,
            badge: 1,
            expirationDate: nil,
            retryCount: 0,
            maxRetries: 1,
            deliveryMethod: .push,
            targetDevices: [.iPhone, .appleWatch],
            analytics: NotificationAnalytics(
                deliveryAttempts: 0,
                deliveryTime: nil,
                openTime: nil,
                actionTime: nil,
                dismissTime: nil,
                responseTime: nil,
                engagementDuration: nil,
                actionTaken: nil,
                contextAtDelivery: nil,
                effectivenessScore: nil
            ),
            aiRecommendations: []
        )
        
        await scheduleNotification(notification)
    }
    
    private func createDefaultPersonalization() -> PersonalizationData {
        return PersonalizationData(
            userPreferences: userPreferences ?? createDefaultPreferences(),
            historicalEngagement: HistoricalEngagement(
                openRate: 0.7,
                actionRate: 0.5,
                dismissRate: 0.3,
                averageResponseTime: 300,
                preferredResponseTimes: [8, 12, 18],
                categoryEngagement: [:],
                typeEngagement: [:]
            ),
            personalityProfile: nil,
            communicationStyle: .encouraging,
            motivationalFactors: [],
            learningPreferences: LearningPreferences(
                preferredContentType: [.text, .image],
                learningStyle: .visual,
                attentionSpan: .medium,
                complexityPreference: .moderate
            ),
            accessibilityNeeds: AccessibilityNeeds(
                visualImpairment: .none,
                hearingImpairment: .none,
                motorImpairment: .none,
                cognitiveImpairment: .none,
                preferredFontSize: .medium,
                highContrastEnabled: false,
                voiceOverEnabled: false,
                reduceMotionEnabled: false
            )
        )
    }
}

// MARK: - UNUserNotificationCenterDelegate

extension IntelligentNotificationSystem: UNUserNotificationCenterDelegate {
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        // Update notification as delivered
        if let notificationId = notification.request.content.userInfo["notificationId"] as? String,
           let index = pendingNotifications.firstIndex(where: { $0.id == notificationId }) {
            
            var updatedNotification = pendingNotifications[index]
            var analytics = updatedNotification.analytics
            analytics = NotificationAnalytics(
                deliveryAttempts: analytics.deliveryAttempts + 1,
                deliveryTime: Date(),
                openTime: analytics.openTime,
                actionTime: analytics.actionTime,
                dismissTime: analytics.dismissTime,
                responseTime: analytics.responseTime,
                engagementDuration: analytics.engagementDuration,
                actionTaken: analytics.actionTaken,
                contextAtDelivery: analytics.contextAtDelivery,
                effectivenessScore: analytics.effectivenessScore
            )
            
            updatedNotification = SmartNotification(
                id: updatedNotification.id,
                type: updatedNotification.type,
                priority: updatedNotification.priority,
                title: updatedNotification.title,
                body: updatedNotification.body,
                category: updatedNotification.category,
                scheduledDate: updatedNotification.scheduledDate,
                deliveryDate: Date(),
                isDelivered: true,
                isRead: updatedNotification.isRead,
                isActionTaken: updatedNotification.isActionTaken,
                context: updatedNotification.context,
                personalization: updatedNotification.personalization,
                actions: updatedNotification.actions,
                attachments: updatedNotification.attachments,
                sound: updatedNotification.sound,
                badge: updatedNotification.badge,
                expirationDate: updatedNotification.expirationDate,
                retryCount: updatedNotification.retryCount,
                maxRetries: updatedNotification.maxRetries,
                deliveryMethod: updatedNotification.deliveryMethod,
                targetDevices: updatedNotification.targetDevices,
                analytics: analytics,
                aiRecommendations: updatedNotification.aiRecommendations
            )
            
            DispatchQueue.main.async {
                self.pendingNotifications[index] = updatedNotification
                self.deliveredNotifications.append(updatedNotification)
            }
        }
        
        completionHandler([.banner, .sound, .badge])
    }
    
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let notificationId = response.notification.request.content.userInfo["notificationId"] as? String ?? ""
        let actionIdentifier = response.actionIdentifier
        
        // Find the notification
        if let notification = deliveredNotifications.first(where: { $0.id == notificationId }) {
            handleNotificationAction(notification: notification, actionIdentifier: actionIdentifier)
        }
        
        completionHandler()
    }
    
    private func handleNotificationAction(notification: SmartNotification, actionIdentifier: String) {
        switch actionIdentifier {
        case "MARK_TAKEN":
            trackNotificationEngagement(notification, action: "mark_taken")
            // Handle medication taken logic
            
        case "SNOOZE_5", "SNOOZE_15":
            let duration: TimeInterval = actionIdentifier == "SNOOZE_5" ? 300 : 900
            Task {
                await snoozeNotification(id: notification.id, duration: duration)
            }
            trackNotificationEngagement(notification, action: "snoozed")
            
        case "LOG_SYMPTOM":
            trackNotificationEngagement(notification, action: "log_symptom")
            // Open symptom logging screen
            
        case "CALL_EMERGENCY":
            trackNotificationEngagement(notification, action: "emergency_call")
            // Handle emergency call
            
        case "CONTACT_DOCTOR":
            trackNotificationEngagement(notification, action: "contact_doctor")
            // Handle doctor contact
            
        case UNNotificationDefaultActionIdentifier:
            trackNotificationEngagement(notification, action: "opened")
            
        case UNNotificationDismissActionIdentifier:
            trackNotificationEngagement(notification, action: "dismissed")
            
        default:
            trackNotificationEngagement(notification, action: actionIdentifier)
        }
    }
}

// MARK: - Context Collectors

protocol ContextCollector {
    func updateContext() async
}

class LocationContextCollector: ContextCollector {
    private let locationManager: CLLocationManager
    
    init(locationManager: CLLocationManager) {
        self.locationManager = locationManager
    }
    
    func updateContext() async {
        // Implementation would collect location context
    }
}

class HealthContextCollector: ContextCollector {
    private let healthStore: HKHealthStore
    
    init(healthStore: HKHealthStore) {
        self.healthStore = healthStore
    }
    
    func updateContext() async {
        // Implementation would collect health context from HealthKit
    }
}

class DeviceContextCollector: ContextCollector {
    func updateContext() async {
        // Implementation would collect device context
    }
}

class CalendarContextCollector: ContextCollector {
    private let eventStore: EKEventStore
    
    init(eventStore: EKEventStore) {
        self.eventStore = eventStore
    }
    
    func updateContext() async {
        // Implementation would collect calendar context
    }
}

class EnvironmentalContextCollector: ContextCollector {
    func updateContext() async {
        // Implementation would collect environmental context (weather, etc.)
    }
}

// MARK: - AI Engine

class NotificationAIEngine {
    func optimizeNotification(_ notification: SmartNotification, context: NotificationContext) async -> SmartNotification {
        // AI optimization logic would go here
        return notification
    }
    
    func recommendDeliveryTime(for notification: SmartNotification, context: NotificationContext) async -> Date {
        // AI delivery time recommendation logic would go here
        return notification.scheduledDate
    }
    
    func updateLearningModel(notification: SmartNotification, action: String) async {
        // Machine learning model update logic would go here
    }
}

// MARK: - SwiftUI Integration

struct NotificationSettingsView: View {
    @ObservedObject var notificationSystem: IntelligentNotificationSystem
    @State private var preferences: UserNotificationPreferences
    
    init(notificationSystem: IntelligentNotificationSystem) {
        self.notificationSystem = notificationSystem
        self._preferences = State(initialValue: notificationSystem.userPreferences ?? notificationSystem.createDefaultPreferences())
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section("Notification Frequency") {
                    Picker("Frequency", selection: $preferences.preferredFrequency) {
                        ForEach([NotificationFrequency.minimal, .low, .medium, .high, .maximum], id: \.self) { frequency in
                            Text(frequency.rawValue.capitalized).tag(frequency)
                        }
                    }
                }
                
                Section("Preferred Times") {
                    ForEach(preferences.preferredTimes.indices, id: \.self) { index in
                        HStack {
                            Text("\(preferences.preferredTimes[index].startHour):00")
                            Text("-")
                            Text("\(preferences.preferredTimes[index].endHour):00")
                        }
                    }
                }
                
                Section("Categories") {
                    ForEach(NotificationCategory.allCases, id: \.self) { category in
                        Toggle(category.rawValue.capitalized, isOn: Binding(
                            get: { preferences.enabledCategories.contains(category) },
                            set: { enabled in
                                if enabled {
                                    preferences.enabledCategories.append(category)
                                    preferences.disabledCategories.removeAll { $0 == category }
                                } else {
                                    preferences.disabledCategories.append(category)
                                    preferences.enabledCategories.removeAll { $0 == category }
                                }
                            }
                        ))
                    }
                }
                
                Section("Sound & Vibration") {
                    Toggle("Sound", isOn: $preferences.soundEnabled)
                    Toggle("Vibration", isOn: $preferences.vibrationEnabled)
                    Toggle("Badge", isOn: $preferences.badgeEnabled)
                }
            }
            .navigationTitle("Notification Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        notificationSystem.updateUserPreferences(preferences)
                    }
                }
            }
        }
    }
}