//
//  MeditationSessionModel.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import Foundation

// MARK: - Meditation Session Model (Available Sessions)

struct MeditationSessionModel: Identifiable, Codable, Hashable {
    let id: UUID
    let title: String
    let description: String
    let category: MeditationCategory
    let type: MeditationType
    let duration: TimeInterval // in seconds
    let difficulty: DifficultyLevel
    let instructor: String?
    let audioURL: String?
    let backgroundSoundURL: String?
    let imageURL: String?
    let tags: [String]
    let benefits: [String]
    let isGuided: Bool
    let hasTimer: Bool
    let hasBreathingGuide: Bool
    let breathingPattern: BreathingPattern?
    let targetSymptoms: [TargetSymptom]
    let recommendedTime: [TimeOfDay]
    let isFavorite: Bool

    init(
        id: UUID = UUID(),
        title: String,
        description: String,
        category: MeditationCategory,
        type: MeditationType,
        duration: TimeInterval,
        difficulty: DifficultyLevel = .beginner,
        instructor: String? = nil,
        audioURL: String? = nil,
        backgroundSoundURL: String? = nil,
        imageURL: String? = nil,
        tags: [String] = [],
        benefits: [String] = [],
        isGuided: Bool = true,
        hasTimer: Bool = true,
        hasBreathingGuide: Bool = false,
        breathingPattern: BreathingPattern? = nil,
        targetSymptoms: [TargetSymptom] = [],
        recommendedTime: [TimeOfDay] = [.anytime],
        isFavorite: Bool = false
    ) {
        self.id = id
        self.title = title
        self.description = description
        self.category = category
        self.type = type
        self.duration = duration
        self.difficulty = difficulty
        self.instructor = instructor
        self.audioURL = audioURL
        self.backgroundSoundURL = backgroundSoundURL
        self.imageURL = imageURL
        self.tags = tags
        self.benefits = benefits
        self.isGuided = isGuided
        self.hasTimer = hasTimer
        self.hasBreathingGuide = hasBreathingGuide
        self.breathingPattern = breathingPattern
        self.targetSymptoms = targetSymptoms
        self.recommendedTime = recommendedTime
        self.isFavorite = isFavorite
    }

    var durationMinutes: Int {
        Int(duration / 60)
    }

    var durationFormatted: String {
        let minutes = Int(duration / 60)
        if minutes < 60 {
            return "\(minutes) min"
        } else {
            let hours = minutes / 60
            let remainingMinutes = minutes % 60
            if remainingMinutes == 0 {
                return "\(hours) hr"
            } else {
                return "\(hours) hr \(remainingMinutes) min"
            }
        }
    }

    // MARK: - Hashable & Equatable

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    static func == (lhs: MeditationSessionModel, rhs: MeditationSessionModel) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - AS-Specific Default Sessions

extension MeditationSessionModel {
    /// 15+ AS-specific meditation sessions
    static let asSpecificSessions: [MeditationSessionModel] = [
        // MORNING SESSIONS
        MeditationSessionModel(
            title: "Morning Stiffness Relief",
            description: "Gentle breathing and body scan designed specifically to ease morning stiffness common in AS. Start your day with reduced pain and improved mobility.",
            category: .painManagement,
            type: .bodyScan,
            duration: 600, // 10 minutes
            difficulty: .beginner,
            benefits: [
                "Reduces morning stiffness intensity",
                "Improves spinal mobility",
                "Gentle transition into the day",
                "Reduces inflammatory response"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .deepBelly, cycles: 8),
            targetSymptoms: [.morningStiffness, .spinalPain, .jointPain],
            recommendedTime: [.earlyMorning, .morning]
        ),

        MeditationSessionModel(
            title: "Energizing Morning Breath",
            description: "Brief but powerful breathing exercise to combat AS-related fatigue. Increase energy and mental clarity for the day ahead.",
            category: .energyBoost,
            type: .breathingExercise,
            duration: 300, // 5 minutes
            difficulty: .beginner,
            benefits: [
                "Increases energy levels",
                "Reduces fatigue",
                "Improves focus and alertness",
                "Gentle stimulation without strain"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .bellowsBreath, cycles: 10),
            targetSymptoms: [.fatigue, .brainfog, .lowMood],
            recommendedTime: [.morning]
        ),

        // PAIN MANAGEMENT SESSIONS
        MeditationSessionModel(
            title: "Spinal Pain Relief",
            description: "Comprehensive guided meditation targeting chronic spinal pain through mindful awareness, breath work, and visualization techniques specifically designed for AS patients.",
            category: .painManagement,
            type: .guided,
            duration: 900, // 15 minutes
            difficulty: .beginner,
            benefits: [
                "Reduces pain perception through mindfulness",
                "Calms nervous system response to inflammation",
                "Promotes muscle relaxation in affected areas",
                "Improves pain coping strategies"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .fourSevenEight, cycles: 6),
            targetSymptoms: [.spinalPain, .inflammation, .chronicPain],
            recommendedTime: [.afternoon, .evening, .anytime]
        ),

        MeditationSessionModel(
            title: "Hip & Lower Back Relief",
            description: "Focused body scan meditation for hip and lower back pain, common problem areas for AS patients. Uses targeted relaxation and breathing.",
            category: .painManagement,
            type: .bodyScan,
            duration: 720, // 12 minutes
            difficulty: .beginner,
            benefits: [
                "Relieves lower back tension",
                "Reduces hip joint discomfort",
                "Improves sacroiliac joint awareness",
                "Gentle pain management technique"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .coherentBreathing, cycles: 12),
            targetSymptoms: [.lowerBackPain, .hipPain, .jointPain],
            recommendedTime: [.anytime]
        ),

        MeditationSessionModel(
            title: "Neck & Upper Spine Relief",
            description: "Targeted meditation for cervical spine pain and neck stiffness. Combines breath awareness with gentle neck tension release.",
            category: .painManagement,
            type: .bodyScan,
            duration: 480, // 8 minutes
            difficulty: .beginner,
            benefits: [
                "Reduces neck pain and stiffness",
                "Relieves shoulder tension",
                "Improves cervical mobility awareness",
                "Reduces headache frequency"
            ],
            hasBreathingGuide: true,
            targetSymptoms: [.neckPain, .spinalPain, .headaches],
            recommendedTime: [.anytime]
        ),

        // FLARE MANAGEMENT
        MeditationSessionModel(
            title: "Flare Emergency Relief",
            description: "Crisis meditation for managing acute flare episodes. Short, calming practice to help you cope with sudden pain increases and anxiety.",
            category: .painManagement,
            type: .guided,
            duration: 480, // 8 minutes
            difficulty: .beginner,
            benefits: [
                "Rapid anxiety reduction during flares",
                "Pain distraction techniques",
                "Activates relaxation response",
                "Provides sense of control"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .fourSevenEight, cycles: 8),
            targetSymptoms: [.flare, .spinalPain, .anxiety, .stress],
            recommendedTime: [.anytime]
        ),

        MeditationSessionModel(
            title: "Inflammation Calming Breathwork",
            description: "Scientifically-backed breathing techniques that may help reduce inflammatory markers. Based on research showing breath work's effect on immune response.",
            category: .breathwork,
            type: .breathingExercise,
            duration: 600, // 10 minutes
            difficulty: .intermediate,
            benefits: [
                "May reduce inflammatory response",
                "Activates parasympathetic nervous system",
                "Improves heart rate variability",
                "Promotes systemic relaxation"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .resonantBreathing, cycles: 10),
            targetSymptoms: [.inflammation, .spinalPain, .chronicPain],
            recommendedTime: [.anytime]
        ),

        // SLEEP & EVENING SESSIONS
        MeditationSessionModel(
            title: "Sleep Preparation for AS",
            description: "Evening meditation designed specifically for AS patients struggling with sleep. Addresses nocturnal pain, anxiety, and promotes deep relaxation.",
            category: .sleepImprovement,
            type: .guided,
            duration: 1200, // 20 minutes
            difficulty: .beginner,
            benefits: [
                "Improves sleep quality despite pain",
                "Reduces nocturnal pain perception",
                "Calms pre-sleep anxiety",
                "Progressive deep relaxation"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .fourSevenEight, cycles: 6),
            targetSymptoms: [.insomnia, .chronicPain, .anxiety, .stress],
            recommendedTime: [.evening, .night]
        ),

        MeditationSessionModel(
            title: "Body Scan for Sleep",
            description: "Gentle full-body scan meditation to release tension before bed. Particularly helpful for AS patients experiencing widespread pain.",
            category: .sleepImprovement,
            type: .bodyScan,
            duration: 900, // 15 minutes
            difficulty: .beginner,
            benefits: [
                "Progressive muscle relaxation",
                "Releases accumulated daily tension",
                "Promotes sleep onset",
                "Reduces nocturnal awakenings"
            ],
            targetSymptoms: [.insomnia, .muscleTension, .chronicPain],
            recommendedTime: [.night]
        ),

        // STRESS & ANXIETY MANAGEMENT
        MeditationSessionModel(
            title: "AS Stress Relief",
            description: "Managing a chronic condition creates unique stress. This session addresses the emotional burden of AS with compassion and practical coping strategies.",
            category: .stressReduction,
            type: .guided,
            duration: 720, // 12 minutes
            difficulty: .beginner,
            benefits: [
                "Reduces disease-related stress",
                "Improves emotional resilience",
                "Cultivates self-compassion",
                "Reduces cortisol levels"
            ],
            hasBreathingGuide: true,
            targetSymptoms: [.stress, .anxiety, .lowMood, .depression],
            recommendedTime: [.anytime]
        ),

        MeditationSessionModel(
            title: "Box Breathing for Anxiety",
            description: "Simple yet powerful 4-4-4-4 breathing technique used by Navy SEALs. Excellent for managing anxiety during challenging AS days.",
            category: .anxietyRelief,
            type: .breathingExercise,
            duration: 360, // 6 minutes
            difficulty: .beginner,
            benefits: [
                "Rapid anxiety reduction",
                "Improves emotional regulation",
                "Increases sense of control",
                "Can be done anywhere"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .boxBreathing, cycles: 8),
            targetSymptoms: [.anxiety, .stress, .irritability],
            recommendedTime: [.anytime]
        ),

        // MINDFULNESS & BODY AWARENESS
        MeditationSessionModel(
            title: "Mindful Movement for AS",
            description: "Gentle mindful movement meditation adapted for AS. Combines awareness with subtle movements to maintain flexibility without strain.",
            category: .movementMeditation,
            type: .guided,
            duration: 900, // 15 minutes
            difficulty: .intermediate,
            benefits: [
                "Improves body awareness",
                "Maintains joint mobility",
                "Gentle stretching with mindfulness",
                "Reduces fear of movement"
            ],
            targetSymptoms: [.morningStiffness, .jointPain, .spinalPain],
            recommendedTime: [.morning, .afternoon]
        ),

        MeditationSessionModel(
            title: "Pain Acceptance & Resilience",
            description: "Advanced meditation practice teaching acceptance-based coping for chronic pain. Build psychological resilience alongside physical healing.",
            category: .emotionalWellbeing,
            type: .guided,
            duration: 1020, // 17 minutes
            difficulty: .intermediate,
            benefits: [
                "Develops healthier pain relationship",
                "Reduces pain catastrophizing",
                "Improves quality of life with chronic pain",
                "Cultivates resilience"
            ],
            targetSymptoms: [.chronicPain, .depression, .anxiety, .lowMood],
            recommendedTime: [.anytime]
        ),

        // QUICK SESSIONS
        MeditationSessionModel(
            title: "3-Minute Reset",
            description: "Ultra-short meditation for busy days or sudden pain spikes. Quick grounding and pain management technique you can use anywhere.",
            category: .mindfulness,
            type: .timer,
            duration: 180, // 3 minutes
            difficulty: .beginner,
            benefits: [
                "Rapid stress relief",
                "Immediate pain distraction",
                "Quick mental reset",
                "Accessible anytime"
            ],
            hasBreathingGuide: true,
            targetSymptoms: [.stress, .anxiety, .chronicPain],
            recommendedTime: [.anytime]
        ),

        MeditationSessionModel(
            title: "5-Minute Breathing Space",
            description: "Classic 'breathing space' meditation adapted for AS. Three steps: acknowledge, breathe, expand. Perfect for midday pain management.",
            category: .breathwork,
            type: .breathingExercise,
            duration: 300, // 5 minutes
            difficulty: .beginner,
            benefits: [
                "Quick mental clarity",
                "Pain and stress relief",
                "Increased present-moment awareness",
                "Easy to remember"
            ],
            hasBreathingGuide: true,
            breathingPattern: BreathingPattern.from(technique: .coherentBreathing, cycles: 10),
            targetSymptoms: [.stress, .brainfog, .anxiety],
            recommendedTime: [.anytime]
        ),

        // GRATITUDE & POSITIVITY
        MeditationSessionModel(
            title: "Gratitude Despite Pain",
            description: "Cultivate gratitude while living with chronic illness. Research-backed practice shown to improve mood and reduce pain perception.",
            category: .gratitude,
            type: .guided,
            duration: 600, // 10 minutes
            difficulty: .beginner,
            benefits: [
                "Improves mood despite pain",
                "Reduces depression symptoms",
                "Shifts focus from pain to positivity",
                "Builds emotional resilience"
            ],
            targetSymptoms: [.lowMood, .depression, .stress],
            recommendedTime: [.morning, .evening]
        )
    ]

    /// Get sessions recommended for current symptoms
    static func recommended(for symptoms: [TargetSymptom]) -> [MeditationSessionModel] {
        asSpecificSessions.filter { session in
            !Set(session.targetSymptoms).isDisjoint(with: Set(symptoms))
        }
    }

    /// Get sessions recommended for current time of day
    static func recommendedForCurrentTime() -> [MeditationSessionModel] {
        let currentTime = TimeOfDay.current
        return asSpecificSessions.filter { session in
            session.recommendedTime.contains(currentTime) || session.recommendedTime.contains(.anytime)
        }
    }

    /// Get quick sessions (under 10 minutes)
    static var quickSessions: [MeditationSessionModel] {
        asSpecificSessions.filter { $0.duration <= 600 }
    }

    /// Get sessions by category
    static func sessions(for category: MeditationCategory) -> [MeditationSessionModel] {
        asSpecificSessions.filter { $0.category == category }
    }

    /// Get beginner-friendly sessions
    static var beginnerSessions: [MeditationSessionModel] {
        asSpecificSessions.filter { $0.difficulty == .beginner }
    }
}
