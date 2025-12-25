//
//  DemoDataSeeder.swift
//  InflamAI
//
//  Generates 200 days of clinically-realistic demo data for "Anna"
//  Based on peer-reviewed AS research - see ANNA_PERSONA_CLINICAL_RESEARCH.md
//
//  IMPORTANT: Uses deterministic random generation for consistent demo data
//  Every launch produces the SAME Anna story - perfect for keynotes!
//

import CoreData
import Foundation

// MARK: - Deterministic Random Generator (for consistent demo data)

/// Seeded random number generator that produces the same sequence every time
/// This ensures Anna's 200-day story is identical on every app launch
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64 = 42) {  // "42" - the answer to life, universe, and demo data
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // Linear Congruential Generator (LCG) - simple but deterministic
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

/// Demo data seeder that generates 200 days of realistic AS patient data
/// Based on clinical research for a 30-year-old female AS patient
final class DemoDataSeeder {

    // MARK: - Singleton

    static let shared = DemoDataSeeder()
    private init() {}

    /// Deterministic random generator - reset before each seeding for consistency
    private var rng = SeededRandomNumberGenerator(seed: 42)

    // MARK: - Anna's Profile Constants

    struct AnnaProfile {
        static let name = "Anna"
        // Safe date creation with fallback to prevent crash on non-Gregorian calendars
        static let dateOfBirth = Calendar.current.date(from: DateComponents(year: 1995, month: 3, day: 15))
            ?? Date(timeIntervalSince1970: 795225600) // March 15, 1995 UTC fallback
        static let diagnosisDate = Calendar.current.date(from: DateComponents(year: 2015, month: 6, day: 1))
            ?? Date(timeIntervalSince1970: 1433116800) // June 1, 2015 UTC fallback
        static let heightCm: Float = 178.0
        static let weightKg: Float = 51.0
        static let bmi: Float = 16.1 // Underweight - associated with worse outcomes
        static let gender = "female"
        static let hlaB27Positive = true
        static let biologicExperienced = true
        static let smokingStatus = "never"

        // Healthcare contacts
        static let rheumatologistName = "Dr. Sarah Mueller"
        static let rheumatologistPhone = "+49 89 1234567"
        static let primaryPhysicianName = "Dr. Thomas Weber"
        static let primaryPhysicianPhone = "+49 89 7654321"
    }

    // MARK: - Data Generation Configuration

    private let totalDays = 200
    private let checkInComplianceRate = 0.92 // 92% daily check-in compliance

    // Flare configuration based on research (25% time in flare)
    private let majorFlareCount = 4
    private let minorFlareCount = 10

    // Exercise configuration (65-75% adherence)
    private let targetExerciseDaysPerWeek = 5
    private let exerciseAdherenceRate = 0.70

    // Meditation configuration (2-3x per week average)
    private let meditationSessionsPerWeek = 2.5

    // MARK: - Body Region IDs - Must match BodyDiagramRegion.rawValue (used by PainTrackingView)
    // FIXED: Changed from lowercase IDs to match BodyDiagramRegion enum in BodyDiagramView.swift

    private let spineRegions = [
        "Cervical C1", "Cervical C2", "Cervical C3", "Cervical C4", "Cervical C5", "Cervical C6", "Cervical C7",  // Cervical
        "Thoracic T1", "Thoracic T2", "Thoracic T3", "Thoracic T4", "Thoracic T5", "Thoracic T6",
        "Thoracic T7", "Thoracic T8", "Thoracic T9", "Thoracic T10", "Thoracic T11", "Thoracic T12",  // Thoracic
        "Lumbar L1", "Lumbar L2", "Lumbar L3", "Lumbar L4", "Lumbar L5",  // Lumbar
        "Sacral S1", "Sacral S2", "Tailbone/Coccyx"  // Sacral
    ]

    private let peripheralRegions = [
        "Left Shoulder", "Right Shoulder",
        "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist",
        "Left Hand", "Right Hand",
        "Left Hip", "Right Hip",
        "Left Knee", "Right Knee",
        "Left Ankle", "Right Ankle",
        "Left Foot", "Right Foot"
    ]

    // Anna's primary affected regions (female pattern - more neck, hips, knees)
    // FIXED: Use BodyDiagramRegion rawValues
    private let primaryAffectedRegions = [
        "Sacral S1", "Sacral S2", "Lumbar L3", "Lumbar L4", "Lumbar L5",
        "Left Hip", "Right Hip",
        "Cervical C5", "Cervical C6", "Cervical C7",
        "Left Knee", "Right Knee"
    ]

    // FIXED: Use BodyDiagramRegion rawValues
    private let secondaryAffectedRegions = [
        "Thoracic T8", "Thoracic T9", "Thoracic T10", "Thoracic T11", "Thoracic T12",
        "Left Shoulder", "Right Shoulder",
        "Left Ankle", "Right Ankle",
        "Left Foot", "Right Foot"
    ]

    // MARK: - Trigger Categories

    private let triggerCategories: [(name: String, category: String, frequency: Double)] = [
        ("stress", "mental_health", 0.35),
        ("poor_sleep", "sleep", 0.30),
        ("overexertion", "physical_activity", 0.20),
        ("weather_change", "environmental", 0.25),
        ("missed_medication", "medication", 0.05),
        ("alcohol", "diet", 0.08),
        ("cold_weather", "environmental", 0.15),
        ("prolonged_sitting", "physical_activity", 0.25),
        ("high_humidity", "environmental", 0.12),
        ("infection", "health", 0.03),
        ("menstrual_cycle", "hormonal", 0.10)
    ]

    // MARK: - Exercise Types (Matching actual exercises in ExerciseData.swift)

    private let exerciseTypes = [
        // Stretching exercises (most common for AS)
        (type: "Cat-Cow Stretch", duration: 5...8, intensity: 2...3),
        (type: "Hip Flexor Stretch", duration: 8...12, intensity: 3...4),
        (type: "Spinal Twist (Supine)", duration: 8...10, intensity: 2...3),
        (type: "Hamstring Stretch", duration: 6...10, intensity: 3...4),
        (type: "Chest Opener Stretch", duration: 5...8, intensity: 2...3),
        (type: "Child's Pose", duration: 5...10, intensity: 1...2),
        (type: "Knee-to-Chest Stretch", duration: 5...8, intensity: 2...3),

        // Mobility exercises (important for AS spinal health)
        (type: "Cervical Spine Rotation", duration: 5...8, intensity: 2...3),
        (type: "Thoracic Rotation (Seated)", duration: 8...12, intensity: 3...4),
        (type: "Hip Circles", duration: 5...8, intensity: 2...3),
        (type: "Shoulder Rolls", duration: 3...5, intensity: 1...2),

        // Strengthening exercises
        (type: "Pelvic Tilts", duration: 8...12, intensity: 3...4),
        (type: "Bridge Exercise", duration: 10...15, intensity: 4...5),
        (type: "Bird Dog", duration: 10...15, intensity: 4...5),
        (type: "Wall Squats", duration: 10...15, intensity: 4...6),

        // Breathing exercises (crucial for AS rib cage mobility)
        (type: "Deep Breathing Exercise", duration: 5...10, intensity: 1...2),
        (type: "Box Breathing", duration: 5...10, intensity: 1...2),
        (type: "Rib Cage Expansion", duration: 8...12, intensity: 2...3),
        (type: "Diaphragmatic Breathing", duration: 5...10, intensity: 1...2),

        // Posture exercises
        (type: "Wall Angels", duration: 8...12, intensity: 3...4),
        (type: "Chin Tucks", duration: 5...8, intensity: 2...3),
        (type: "Brugger's Relief Position", duration: 3...5, intensity: 2...3),

        // Balance exercises
        (type: "Single Leg Stand", duration: 5...10, intensity: 3...4),
        (type: "Heel-to-Toe Walk", duration: 5...8, intensity: 3...4)
    ]

    // MARK: - Anna's Routines (5 total: 1 daily, 2 rare, 3 flare-specific)

    /// Local struct for encoding routine exercises (mirrors RoutineExerciseItem from RoutineStubs)
    private struct DemoRoutineExerciseItem: Codable {
        let id: UUID
        let exerciseId: String
        let duration: Int
        let order: Int
    }

    struct RoutineDefinition {
        let id: UUID
        let name: String
        let exercises: [String]
        let totalDuration: Int16
        let isActive: Bool
        let usageFrequency: Double  // How often she uses it (0-1)
        let isForFlares: Bool
        let customNotes: String
    }

    private let annaRoutines: [RoutineDefinition] = [
        // DAILY ROUTINE - Her main morning routine (active, high usage)
        RoutineDefinition(
            id: UUID(uuidString: "A1111111-1111-1111-1111-111111111111")!,
            name: "Morning Mobility Flow",
            exercises: [
                "Cat-Cow Stretch",
                "Hip Flexor Stretch",
                "Spinal Twist (Supine)",
                "Shoulder Rolls",
                "Deep Breathing Exercise",
                "Chin Tucks"
            ],
            totalDuration: 20,
            isActive: true,
            usageFrequency: 0.75,  // 75% of days
            isForFlares: false,
            customNotes: "My go-to morning routine. Helps with stiffness after waking up."
        ),

        // RARE ROUTINE 1 - Strength focus (used ~1x per week)
        RoutineDefinition(
            id: UUID(uuidString: "B2222222-2222-2222-2222-222222222222")!,
            name: "Core Strength Builder",
            exercises: [
                "Pelvic Tilts",
                "Bridge Exercise",
                "Bird Dog",
                "Plank Hold",
                "Wall Squats"
            ],
            totalDuration: 25,
            isActive: false,
            usageFrequency: 0.12,  // ~1x per week
            isForFlares: false,
            customNotes: "Strength routine for good days. Skip if hips are acting up."
        ),

        // RARE ROUTINE 2 - Posture/Balance (used occasionally)
        RoutineDefinition(
            id: UUID(uuidString: "C3333333-3333-3333-3333-333333333333")!,
            name: "Posture & Balance",
            exercises: [
                "Wall Angels",
                "Brugger's Relief Position",
                "Single Leg Stand",
                "Heel-to-Toe Walk",
                "Standing Posture Check"
            ],
            totalDuration: 18,
            isActive: false,
            usageFrequency: 0.08,  // ~2x per month
            isForFlares: false,
            customNotes: "For days when my posture feels off. Good for office work days."
        ),

        // FLARE ROUTINE 1 - Gentle stretching
        RoutineDefinition(
            id: UUID(uuidString: "D4444444-4444-4444-4444-444444444444")!,
            name: "Gentle Flare Relief",
            exercises: [
                "Child's Pose",
                "Knee-to-Chest Stretch",
                "Deep Breathing Exercise",
                "Diaphragmatic Breathing"
            ],
            totalDuration: 12,
            isActive: false,
            usageFrequency: 0.0,  // Only during flares
            isForFlares: true,
            customNotes: "Very gentle routine for acute flare days. Focus on breathing and relaxation."
        ),

        // FLARE ROUTINE 2 - SI Joint focus
        RoutineDefinition(
            id: UUID(uuidString: "E5555555-5555-5555-5555-555555555555")!,
            name: "SI Joint Care",
            exercises: [
                "Pelvic Tilts",
                "Piriformis Stretch",
                "Knee-to-Chest Stretch",
                "Hip Circles"
            ],
            totalDuration: 15,
            isActive: false,
            usageFrequency: 0.0,  // Only during flares
            isForFlares: true,
            customNotes: "When SI joints are flaring. Gentle hip and pelvis focus."
        ),

        // FLARE ROUTINE 3 - Neck/Upper back focus
        RoutineDefinition(
            id: UUID(uuidString: "F6666666-6666-6666-6666-666666666666")!,
            name: "Neck & Shoulder Ease",
            exercises: [
                "Neck Side Stretch",
                "Cervical Spine Rotation",
                "Shoulder Rolls",
                "Chest Opener Stretch",
                "Box Breathing"
            ],
            totalDuration: 14,
            isActive: false,
            usageFrequency: 0.0,  // Only during flares
            isForFlares: true,
            customNotes: "For cervical flares. Very gentle neck movements with breathing."
        )
    ]

    // MARK: - Meditation Types

    private let meditationTypes = [
        (type: "body_scan", title: "Body Scan Meditation", duration: 15...20, category: "relaxation"),
        (type: "breathing", title: "Deep Breathing Exercise", duration: 5...10, category: "breathing"),
        (type: "mindfulness", title: "Morning Mindfulness", duration: 10...15, category: "mindfulness"),
        (type: "pain_relief", title: "Pain Relief Visualization", duration: 15...20, category: "pain_management"),
        (type: "sleep", title: "Sleep Preparation", duration: 20...25, category: "sleep"),
        (type: "stress_relief", title: "Stress Release", duration: 10...15, category: "stress")
    ]

    // MARK: - Journal Entry Themes

    private let journalThemes: [(mood: String, themes: [String])] = [
        ("positive", [
            "Had a good day today. Morning stiffness wasn't as bad. Managed to complete my stretching routine.",
            "Feeling hopeful. The new medication seems to be helping. Less fatigue than usual.",
            "Great swimming session today. The water feels so good on my joints. Mood is much better.",
            "Slept through the night for once! What a difference it makes. Energy levels are up.",
            "Work was manageable today. Colleagues were understanding about my breaks."
        ]),
        ("neutral", [
            "Average day. Some stiffness in the morning but it eased after an hour. Trying to stay active.",
            "Weather is changing. Hoping it won't trigger anything. Did some light stretching.",
            "Took my medication on time. Feeling okay, not great but not terrible either.",
            "Had to take more rest breaks than I'd like. Trying not to overdo it.",
            "Appointment with Dr. Mueller next week. Need to note down my symptoms."
        ]),
        ("challenging", [
            "Rough night. Woke up several times from pain. Morning was difficult to get moving.",
            "Feeling frustrated. The fatigue is overwhelming today. Had to cancel plans.",
            "Flare seems to be starting. SI joints are really acting up. Taking extra care.",
            "Couldn't complete my exercise routine. Too much pain in my hips. Resting instead.",
            "Brain fog is terrible today. Hard to concentrate at work. Took NSAID for relief."
        ]),
        ("flare", [
            "Full flare day. Everything hurts. Stayed in bed most of the day. Heat packs helping a bit.",
            "Third day of this flare. The fatigue is crushing. Even simple tasks feel impossible.",
            "Called in sick to work. Can barely move without pain. Hoping this passes soon.",
            "Flare continuing. Took extra Celebrex. Trying gentle stretches when I can.",
            "Starting to feel slightly better. The worst might be over. Still very tired."
        ])
    ]

    // MARK: - Flare Schedule Generation

    private struct FlareEvent_Data {
        let startDay: Int
        let duration: Int
        let severity: Int
        let isMajor: Bool
        let triggers: [String]
        let primaryRegions: [String]
    }

    private func generateFlareSchedule() -> [FlareEvent_Data] {
        var flares: [FlareEvent_Data] = []
        var usedDays = Set<Int>()

        // Generate major flares (duration 10-21 days, severity 7-9)
        for _ in 0..<majorFlareCount {
            var startDay: Int
            repeat {
                startDay = Int.random(in: 10...(totalDays - 25))
            } while usedDays.contains(where: { abs($0 - startDay) < 25 })

            let duration = Int.random(in: 10...21)
            // Clamp buffer end to prevent exceeding totalDays
            let bufferEnd = min(totalDays - 1, startDay + duration + 5)
            for d in startDay...bufferEnd {
                usedDays.insert(d)
            }

            let triggers = ["stress", "overexertion", "weather_change"].shuffled().prefix(2).map { $0 }
            let regions = primaryAffectedRegions.shuffled().prefix(4).map { $0 }

            flares.append(FlareEvent_Data(
                startDay: startDay,
                duration: duration,
                severity: Int.random(in: 7...9),
                isMajor: true,
                triggers: triggers,
                primaryRegions: regions
            ))
        }

        // Generate minor flares (duration 3-7 days, severity 5-7)
        for _ in 0..<minorFlareCount {
            var startDay: Int
            var attempts = 0
            repeat {
                startDay = Int.random(in: 5...(totalDays - 10))
                attempts += 1
            } while usedDays.contains(where: { abs($0 - startDay) < 10 }) && attempts < 50

            if attempts >= 50 { continue }

            let duration = Int.random(in: 3...7)
            // Clamp buffer end to prevent exceeding totalDays
            let bufferEnd = min(totalDays - 1, startDay + duration + 3)
            for d in startDay...bufferEnd {
                usedDays.insert(d)
            }

            let triggers = triggerCategories.shuffled().prefix(1).map { $0.name }
            let regions = primaryAffectedRegions.shuffled().prefix(2).map { $0 }

            flares.append(FlareEvent_Data(
                startDay: startDay,
                duration: duration,
                severity: Int.random(in: 5...7),
                isMajor: false,
                triggers: triggers,
                primaryRegions: regions
            ))
        }

        return flares.sorted { $0.startDay < $1.startDay }
    }

    // MARK: - Day State Calculation

    private enum DayState {
        case good
        case average
        case bad
        case flare(severity: Int, dayInFlare: Int)
    }

    private func calculateDayState(day: Int, flares: [FlareEvent_Data]) -> DayState {
        // Check if in flare
        for flare in flares {
            if day >= flare.startDay && day < flare.startDay + flare.duration {
                let dayInFlare = day - flare.startDay
                return .flare(severity: flare.severity, dayInFlare: dayInFlare)
            }
            // Post-flare recovery (3 days of "bad")
            if day >= flare.startDay + flare.duration && day < flare.startDay + flare.duration + 3 {
                return .bad
            }
        }

        // Normal day distribution (based on research)
        let random = Double.random(in: 0...1)
        if random < 0.35 {
            return .good
        } else if random < 0.80 {
            return .average
        } else {
            return .bad
        }
    }

    // MARK: - BASDAI Score Generation

    private func generateBASDAIScore(state: DayState, previousScore: Double) -> (score: Double, answers: [Double]) {
        let baseRange: ClosedRange<Double>

        switch state {
        case .good:
            baseRange = 2.5...3.8
        case .average:
            baseRange = 3.8...5.2
        case .bad:
            baseRange = 5.0...6.5
        case .flare(let severity, let dayInFlare):
            // Peak at day 2-3, then gradual improvement
            let peakAdjustment = dayInFlare < 3 ? Double(dayInFlare) * 0.3 : max(0, 0.9 - Double(dayInFlare - 3) * 0.15)
            let baseValue = Double(severity) * 0.7 + peakAdjustment
            // Ensure valid range: lower bound must be <= upper bound
            let lowerBound = max(5.5, baseValue - 0.5)
            let upperBound = max(lowerBound, min(9.5, baseValue + 0.5))
            baseRange = lowerBound...upperBound
        }

        // Add some continuity with previous day (20% influence)
        let rawScore = Double.random(in: baseRange)
        let smoothedScore = rawScore * 0.8 + previousScore * 0.2
        let finalScore = min(10, max(0, smoothedScore))

        // Generate individual answers that produce this score
        // BASDAI = (Q1 + Q2 + Q3 + Q4 + ((Q5 + Q6) / 2)) / 5

        let q1_fatigue = generateAnswer(baseScore: finalScore, variation: 1.5, femaleBonus: 0.8)  // Higher fatigue in women
        let q2_spinalPain = generateAnswer(baseScore: finalScore, variation: 1.2, femaleBonus: -0.3)  // Less spinal in women
        let q3_peripheralPain = generateAnswer(baseScore: finalScore, variation: 1.3, femaleBonus: 0.5)  // More peripheral in women
        let q4_enthesitis = generateAnswer(baseScore: finalScore, variation: 1.0, femaleBonus: 0.2)
        let q5_stiffnessSeverity = generateAnswer(baseScore: finalScore, variation: 1.2, femaleBonus: 0.4)  // Longer in women
        let q6_stiffnessDuration = generateAnswer(baseScore: finalScore, variation: 1.5, femaleBonus: 0.6)

        return (finalScore, [q1_fatigue, q2_spinalPain, q3_peripheralPain, q4_enthesitis, q5_stiffnessSeverity, q6_stiffnessDuration])
    }

    private func generateAnswer(baseScore: Double, variation: Double, femaleBonus: Double) -> Double {
        let adjusted = baseScore + femaleBonus + Double.random(in: -variation...variation)
        return min(10, max(0, adjusted))
    }

    // MARK: - Morning Stiffness Duration

    private func generateMorningStiffness(state: DayState) -> Int16 {
        switch state {
        case .good:
            return Int16.random(in: 25...50)
        case .average:
            return Int16.random(in: 50...90)
        case .bad:
            return Int16.random(in: 80...130)
        case .flare(_, let dayInFlare):
            let peak = dayInFlare < 3 ? 30 : -10 * dayInFlare
            return Int16(min(180, max(90, 140 + peak + Int.random(in: -20...20))))
        }
    }

    // MARK: - Weather Generation

    private func generateWeather(day: Int, isFlareDay: Bool) -> (pressure: Double, pressureChange: Double, humidity: Int16, temp: Double, precipitation: Bool) {
        // Base seasonal pattern (assuming start in spring)
        let seasonalOffset = sin(Double(day) / 60.0 * .pi) * 5  // Temperature variation
        let baseTemp = 15.0 + seasonalOffset + Double.random(in: -5...5)

        // Pressure: 1013 hPa average, drops before flares
        var basePressure = 1013.0 + Double.random(in: -15...15)
        var pressureChange = Double.random(in: -3...3)

        // If flare day, simulate pressure drop (research: >5 mmHg drop triggers symptoms)
        if isFlareDay {
            basePressure -= Double.random(in: 5...12)
            pressureChange = Double.random(in: (-8)...(-4))
        }

        let humidity = Int16.random(in: 40...85)
        let precipitation = Double.random(in: 0...1) < 0.25

        return (basePressure, pressureChange, humidity, baseTemp, precipitation)
    }

    // MARK: - Biometric Generation

    private func generateBiometrics(state: DayState) -> (hrv: Double, hr: Int16, steps: Int32, sleepEff: Double, sleepDur: Double) {
        // HRV lower in AS patients, even lower during flares
        let baseHRV: Double
        let baseHR: Int16
        let baseSteps: Int32
        let baseSleepEff: Double
        let baseSleepDur: Double

        switch state {
        case .good:
            baseHRV = Double.random(in: 45...58)
            baseHR = Int16.random(in: 62...70)
            baseSteps = Int32.random(in: 7000...9500)
            baseSleepEff = Double.random(in: 0.82...0.90)
            baseSleepDur = Double.random(in: 6.5...8.0)
        case .average:
            baseHRV = Double.random(in: 38...50)
            baseHR = Int16.random(in: 66...74)
            baseSteps = Int32.random(in: 5000...7500)
            baseSleepEff = Double.random(in: 0.72...0.85)
            baseSleepDur = Double.random(in: 6.0...7.5)
        case .bad:
            baseHRV = Double.random(in: 32...42)
            baseHR = Int16.random(in: 70...78)
            baseSteps = Int32.random(in: 3500...5500)
            baseSleepEff = Double.random(in: 0.60...0.75)
            baseSleepDur = Double.random(in: 5.0...7.0)
        case .flare(let severity, _):
            let severityFactor = Double(severity) / 10.0
            baseHRV = Double.random(in: 25...35) - severityFactor * 5
            baseHR = Int16(Double.random(in: 72...85) + severityFactor * 5)
            baseSteps = Int32(Double.random(in: 1500...4000) - severityFactor * 500)
            baseSleepEff = Double.random(in: 0.50...0.70) - severityFactor * 0.05
            baseSleepDur = Double.random(in: 4.5...6.5)
        }

        return (max(20, baseHRV), baseHR, max(500, baseSteps), max(0.4, min(0.95, baseSleepEff)), max(3, baseSleepDur))
    }

    // MARK: - Body Region Pain Generation

    private func generateBodyRegionPain(state: DayState, flare: FlareEvent_Data?) -> [(regionID: String, pain: Int16, stiffness: Int16, swelling: Bool, warmth: Bool)] {
        var regions: [(regionID: String, pain: Int16, stiffness: Int16, swelling: Bool, warmth: Bool)] = []

        let basePain: Int16
        switch state {
        case .good: basePain = 2
        case .average: basePain = 4
        case .bad: basePain = 5
        case .flare(let severity, _): basePain = Int16(severity)
        }

        // Primary affected regions (always have some pain for Anna)
        for region in primaryAffectedRegions {
            let pain = min(10, max(0, basePain + Int16.random(in: -1...2)))
            let stiffness = Int16(Double(pain) * Double.random(in: 8...15))
            let swelling = pain >= 6 && Double.random(in: 0...1) < 0.3
            let warmth = pain >= 7 && Double.random(in: 0...1) < 0.2

            if pain > 0 {
                regions.append((region, pain, stiffness, swelling, warmth))
            }
        }

        // Secondary regions (occasional pain)
        for region in secondaryAffectedRegions {
            if Double.random(in: 0...1) < 0.4 {
                let pain = min(10, max(0, basePain - 1 + Int16.random(in: -1...1)))
                if pain > 0 {
                    let stiffness = Int16(Double(pain) * Double.random(in: 5...12))
                    regions.append((region, pain, stiffness, false, false))
                }
            }
        }

        // During flares, add extra regions
        if let flare = flare {
            for region in flare.primaryRegions where !regions.contains(where: { $0.regionID == region }) {
                let pain = Int16(flare.severity) + Int16.random(in: -1...1)
                let stiffness = Int16(Double(pain) * Double.random(in: 10...20))
                regions.append((region, min(10, pain), stiffness, pain >= 7, pain >= 8))
            }
        }

        return regions
    }

    // MARK: - CRP Value Generation

    private func generateCRP(state: DayState) -> Double {
        // Anna is underweight, associated with higher CRP
        let underweightBonus = 3.0

        switch state {
        case .good:
            return Double.random(in: 3...8) + underweightBonus
        case .average:
            return Double.random(in: 6...14) + underweightBonus
        case .bad:
            return Double.random(in: 12...22) + underweightBonus
        case .flare(let severity, _):
            return Double.random(in: 18...35) + Double(severity) * 2 + underweightBonus
        }
    }

    // MARK: - Main Seeding Function

    @MainActor
    func seedDemoData(context: NSManagedObjectContext) async throws {
        print("🌱 Starting demo data seeding for Anna...")

        // Check if already seeded (robust check)
        let profileRequest: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        let existingProfiles = try context.fetch(profileRequest)

        if let existingProfile = existingProfiles.first, existingProfile.name == AnnaProfile.name {
            print("✅ Demo data already exists for Anna. Marking as complete.")
            // Ensure the flag is set even if we skip seeding
            UserDefaults.standard.hasDemoDataBeenSeeded = true
            return
        }

        // Check if there's any symptom log data (secondary check)
        let logRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        logRequest.fetchLimit = 1
        let existingLogs = try context.fetch(logRequest)
        if !existingLogs.isEmpty {
            print("✅ Symptom log data already exists. Marking as complete.")
            UserDefaults.standard.hasDemoDataBeenSeeded = true
            return
        }

        // Clear existing data first
        try await clearAllData(context: context)

        // Generate flare schedule
        let flares = generateFlareSchedule()
        print("📅 Generated \(flares.count) flare events")

        // 1. Create User Profile
        try createUserProfile(context: context)
        print("✅ Created Anna's profile")

        // 2. Create Medications
        try createMedications(context: context)
        print("✅ Created medications")

        // 2.5. Create Anna's Exercise Routines
        let createdRoutines = try createUserRoutines(context: context)
        print("✅ Created \(createdRoutines.count) exercise routines")

        // 3. Create Flare Events
        try createFlareEvents(context: context, flares: flares)
        print("✅ Created flare events")

        // 4. Generate 200 days of symptom logs
        var previousBASDAI = 4.5
        var exerciseCount = 0
        var meditationCount = 0
        var journalCount = 0

        let calendar = Calendar.current
        let today = Date()

        // Intermediate save interval (every 25 days to prevent memory bloat)
        let saveInterval = 25

        for day in 0..<totalDays {
            // Calculate date (going backwards from today)
            guard let logDate = calendar.date(byAdding: .day, value: -(totalDays - 1 - day), to: today) else { continue }

            // Skip some days for realism (8% skip rate)
            // BUT: Never skip the last 7 days - we need recent data for the dashboard
            let isRecentDay = day >= totalDays - 7
            if !isRecentDay && Double.random(in: 0...1) > checkInComplianceRate { continue }

            // Determine day state
            let state = calculateDayState(day: day, flares: flares)
            let currentFlare = flares.first { day >= $0.startDay && day < $0.startDay + $0.duration }

            // Generate symptom log
            let (basdaiScore, basdaiAnswers) = generateBASDAIScore(state: state, previousScore: previousBASDAI)
            previousBASDAI = basdaiScore

            try createSymptomLog(
                context: context,
                date: logDate,
                state: state,
                basdaiScore: basdaiScore,
                basdaiAnswers: basdaiAnswers,
                flare: currentFlare,
                day: day
            )

            // Generate exercise (varies by day state)
            let exerciseProbability: Double
            switch state {
            case .good: exerciseProbability = 0.85
            case .average: exerciseProbability = 0.70
            case .bad: exerciseProbability = 0.40
            case .flare(_, let dayInFlare): exerciseProbability = dayInFlare > 5 ? 0.30 : 0.15
            }

            if Double.random(in: 0...1) < exerciseProbability {
                try createExerciseSession(context: context, date: logDate, state: state, routines: createdRoutines)
                exerciseCount += 1
            }

            // Generate meditation (2-3x per week)
            if Double.random(in: 0...1) < (meditationSessionsPerWeek / 7.0) {
                try createMeditationSession(context: context, date: logDate, state: state)
                meditationCount += 1
            }

            // Generate journal entry (60% of days)
            if Double.random(in: 0...1) < 0.60 {
                try createJournalEntry(context: context, date: logDate, state: state)
                journalCount += 1
            }

            // Intermediate save every saveInterval days to prevent memory bloat
            if day > 0 && day % saveInterval == 0 {
                try context.save()
                context.reset()  // Release memory to prevent crash on low-memory devices
                print("   💾 Checkpoint save at day \(day)/\(totalDays)")
            }
        }

        // 5. Create medication dose logs
        try createDoseLogs(context: context, totalDays: totalDays)
        print("✅ Created medication dose logs")

        // 6. Create meditation streak
        try createMeditationStreak(context: context, totalSessions: meditationCount)
        print("✅ Created meditation streak")

        // 7. Create questionnaire responses (BASDAI, BASFI, BAS-G, ASQoL)
        let questionnaireCount = try createQuestionnaireResponses(context: context, flares: flares)
        print("✅ Created \(questionnaireCount) questionnaire responses")

        // Save all changes
        try context.save()

        // CRITICAL: Mark demo data as seeded to prevent regeneration on next launch
        UserDefaults.standard.hasDemoDataBeenSeeded = true

        print("🎉 Demo data seeding complete!")
        print("   📊 Symptom logs: ~\(Int(Double(totalDays) * checkInComplianceRate))")
        print("   🏋️ Exercise sessions: \(exerciseCount)")
        print("   🧘 Meditation sessions: \(meditationCount)")
        print("   📝 Journal entries: \(journalCount)")
        print("   🔥 Flare events: \(flares.count)")
        print("   📋 Questionnaire responses: \(questionnaireCount)")
    }

    // MARK: - Entity Creation Helpers

    private func clearAllData(context: NSManagedObjectContext) async throws {
        let entities = [
            "SymptomLog", "BodyRegionLog", "ContextSnapshot", "TriggerLog",
            "Medication", "DoseLog", "MedicationIntake",
            "ExerciseSession", "ExercisePainAlert", "ExerciseCompletion",
            "FlareEvent", "JournalEntry", "BASSDAIAssessment",
            "MeditationSession", "MeditationStreak",
            "UserProfile", "PainEntry",
            // Additional entities to ensure clean reseed
            "JointComfortProfile", "QuestionnaireResponse", "UserRoutine",
            "KNNTrainingDay", "NeuralModelVersion", "TriggerAnalysisCache"
        ]

        for entityName in entities {
            let fetchRequest = NSFetchRequest<NSFetchRequestResult>(entityName: entityName)
            let deleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest)
            try context.execute(deleteRequest)
        }

        try context.save()
        print("🧹 Cleared existing data")
    }

    private func createUserProfile(context: NSManagedObjectContext) throws {
        let profile = UserProfile(context: context)
        profile.id = UUID()
        profile.name = AnnaProfile.name
        profile.dateOfBirth = AnnaProfile.dateOfBirth
        profile.diagnosisDate = AnnaProfile.diagnosisDate
        profile.gender = AnnaProfile.gender
        profile.heightCm = AnnaProfile.heightCm
        profile.weightKg = AnnaProfile.weightKg
        profile.bmi = AnnaProfile.bmi
        profile.hlaB27Positive = AnnaProfile.hlaB27Positive
        profile.biologicExperienced = AnnaProfile.biologicExperienced
        profile.smokingStatus = AnnaProfile.smokingStatus
        profile.rheumatologistName = AnnaProfile.rheumatologistName
        profile.rheumatologistPhone = AnnaProfile.rheumatologistPhone
        profile.primaryPhysicianName = AnnaProfile.primaryPhysicianName
        profile.primaryPhysicianPhone = AnnaProfile.primaryPhysicianPhone
        profile.healthKitEnabled = true
        profile.weatherKitEnabled = true
        profile.notificationsEnabled = true
        profile.biometricLockEnabled = false
        profile.cloudSyncEnabled = false
        profile.createdAt = Calendar.current.date(byAdding: .day, value: -totalDays, to: Date()) ?? Date()
        profile.lastModified = Date()
    }

    private func createMedications(context: NSManagedObjectContext) throws {
        // Current biologic: Cosentyx (secukinumab)
        let cosentyx = Medication(context: context)
        cosentyx.id = UUID()
        cosentyx.name = "Cosentyx (secukinumab)"
        cosentyx.category = "Biologic - IL-17 Inhibitor"
        cosentyx.dosage = 150
        cosentyx.dosageUnit = "mg"
        cosentyx.unit = "mg"
        cosentyx.frequency = "Every 4 weeks"
        cosentyx.route = "subcutaneous"
        cosentyx.isBiologic = true
        cosentyx.isActive = true
        cosentyx.startDate = Calendar.current.date(byAdding: .month, value: -24, to: Date())
        cosentyx.prescribedBy = AnnaProfile.rheumatologistName
        cosentyx.reminderEnabled = true
        cosentyx.notes = "Self-injection. Store in refrigerator."

        // PRN NSAID: Celebrex (celecoxib)
        let celebrex = Medication(context: context)
        celebrex.id = UUID()
        celebrex.name = "Celebrex (celecoxib)"
        celebrex.category = "NSAID - COX-2 Inhibitor"
        celebrex.dosage = 200
        celebrex.dosageUnit = "mg"
        celebrex.unit = "mg"
        celebrex.frequency = "As needed (PRN)"
        celebrex.route = "oral"
        celebrex.isBiologic = false
        celebrex.isActive = true
        celebrex.startDate = Calendar.current.date(byAdding: .year, value: -8, to: Date())
        celebrex.prescribedBy = AnnaProfile.rheumatologistName
        celebrex.reminderEnabled = false
        celebrex.notes = "Take with food. Max 400mg/day during flares."

        // Supplements
        let vitaminD = Medication(context: context)
        vitaminD.id = UUID()
        vitaminD.name = "Vitamin D3"
        vitaminD.category = "Supplement"
        vitaminD.dosage = 2000
        vitaminD.dosageUnit = "IU"
        vitaminD.unit = "IU"
        vitaminD.frequency = "Daily"
        vitaminD.route = "oral"
        vitaminD.isBiologic = false
        vitaminD.isActive = true
        vitaminD.startDate = Calendar.current.date(byAdding: .year, value: -5, to: Date())
        vitaminD.reminderEnabled = true
        vitaminD.notes = "For bone health"

        let calcium = Medication(context: context)
        calcium.id = UUID()
        calcium.name = "Calcium"
        calcium.category = "Supplement"
        calcium.dosage = 600
        calcium.dosageUnit = "mg"
        calcium.unit = "mg"
        calcium.frequency = "Daily"
        calcium.route = "oral"
        calcium.isBiologic = false
        calcium.isActive = true
        calcium.startDate = Calendar.current.date(byAdding: .year, value: -5, to: Date())
        calcium.reminderEnabled = true
        calcium.notes = "Take with Vitamin D"

        // Previous medication (inactive)
        let humira = Medication(context: context)
        humira.id = UUID()
        humira.name = "Humira (adalimumab)"
        humira.category = "Biologic - TNF Inhibitor"
        humira.dosage = 40
        humira.dosageUnit = "mg"
        humira.unit = "mg"
        humira.frequency = "Every 2 weeks"
        humira.route = "subcutaneous"
        humira.isBiologic = true
        humira.isActive = false
        humira.startDate = Calendar.current.date(byAdding: .year, value: -6, to: Date())
        humira.endDate = Calendar.current.date(byAdding: .month, value: -24, to: Date())
        humira.prescribedBy = AnnaProfile.rheumatologistName
        humira.notes = "Discontinued due to loss of efficacy after 4 years."
    }

    private func createFlareEvents(context: NSManagedObjectContext, flares: [FlareEvent_Data]) throws {
        let calendar = Calendar.current
        let today = Date()

        for flare in flares {
            guard let startDate = calendar.date(byAdding: .day, value: -(totalDays - 1 - flare.startDay), to: today),
                  let endDate = calendar.date(byAdding: .day, value: flare.duration, to: startDate) else { continue }

            let event = FlareEvent(context: context)
            event.id = UUID()
            event.startDate = startDate
            event.endDate = endDate
            event.severity = Int16(flare.severity)
            event.isResolved = true
            event.interventions = flare.isMajor ? "Increased Celebrex to 400mg/day, extra rest, heat therapy" : "Added PRN Celebrex, gentle stretching"
            event.notes = flare.isMajor ? "Major flare - had to take sick leave from work" : "Minor flare - manageable with medication"

            // Store triggers and regions as JSON data
            if let triggersData = try? JSONEncoder().encode(flare.triggers) {
                event.suspectedTriggers = triggersData
            }
            if let regionsData = try? JSONEncoder().encode(flare.primaryRegions) {
                event.primaryRegions = regionsData
            }
        }
    }

    /// Creates Anna's 5 exercise routines with realistic completion history
    private func createUserRoutines(context: NSManagedObjectContext) throws -> [UserRoutine] {
        var createdRoutines: [UserRoutine] = []
        let calendar = Calendar.current
        let today = Date()

        for routineDef in annaRoutines {
            let routine = UserRoutine(context: context)
            routine.id = routineDef.id
            routine.name = routineDef.name
            routine.totalDuration = routineDef.totalDuration
            routine.isActive = routineDef.isActive
            routine.customNotes = routineDef.customNotes
            routine.reminderEnabled = routineDef.isActive  // Only daily routine has reminder
            routine.createdAt = calendar.date(byAdding: .day, value: -totalDays - 30, to: today)  // Created before tracking period

            // Set reminder time for daily routine (7:30 AM)
            if routineDef.isActive {
                var reminderComponents = DateComponents()
                reminderComponents.hour = 7
                reminderComponents.minute = 30
                routine.reminderTime = calendar.date(from: reminderComponents)
            }

            // Encode exercises as JSON (using DemoRoutineExerciseItem format)
            let exerciseItems = routineDef.exercises.enumerated().map { index, exerciseName in
                DemoRoutineExerciseItem(
                    id: UUID(),
                    exerciseId: exerciseName,  // Using exercise name as ID
                    duration: 30,  // Default 30 seconds
                    order: index
                )
            }
            if let exercisesData = try? JSONEncoder().encode(exerciseItems) {
                routine.exercises = exercisesData
            }

            // Calculate realistic completion counts based on usage
            // Daily routine: ~75% of 200 days = 150 completions
            // Rare routines: based on their frequency
            // Flare routines: Only during flare periods
            let baseCompletions: Int
            if routineDef.isForFlares {
                // Flare routines used during ~14 flares averaging 5 days each = ~70 flare days
                // Split among 3 flare routines, ~15-25 times each
                baseCompletions = Int.random(in: 15...25)
            } else if routineDef.isActive {
                // Daily routine used ~75% of days
                baseCompletions = Int(Double(totalDays) * routineDef.usageFrequency)
            } else {
                // Rare routines based on their frequency
                baseCompletions = Int(Double(totalDays) * routineDef.usageFrequency)
            }

            routine.timesCompleted = Int16(baseCompletions)

            // Set last performed date
            if routineDef.isActive {
                // Daily routine - used yesterday or today
                routine.lastPerformed = calendar.date(byAdding: .day, value: -Int.random(in: 0...1), to: today)
            } else if routineDef.isForFlares {
                // Flare routine - last used during most recent flare
                routine.lastPerformed = calendar.date(byAdding: .day, value: -Int.random(in: 10...30), to: today)
            } else {
                // Rare routine - used occasionally
                routine.lastPerformed = calendar.date(byAdding: .day, value: -Int.random(in: 5...14), to: today)
            }

            createdRoutines.append(routine)
        }

        return createdRoutines
    }

    private func createSymptomLog(context: NSManagedObjectContext, date: Date, state: DayState, basdaiScore: Double, basdaiAnswers: [Double], flare: FlareEvent_Data?, day: Int) throws {
        let log = SymptomLog(context: context)
        log.id = UUID()

        // Set time to morning (8-10 AM)
        var components = Calendar.current.dateComponents([.year, .month, .day], from: date)
        components.hour = Int.random(in: 8...10)
        components.minute = Int.random(in: 0...59)
        log.timestamp = Calendar.current.date(from: components) ?? date  // Fallback to base date

        // BASDAI
        log.basdaiScore = basdaiScore
        if let answersData = try? JSONEncoder().encode(basdaiAnswers) {
            log.basdaiAnswers = answersData
        }

        // Basic metrics
        log.fatigueLevel = Int16(basdaiAnswers[0])  // Q1 is fatigue
        log.morningStiffnessMinutes = generateMorningStiffness(state: state)
        log.source = "pain_tracking"  // FIXED: Use "pain_tracking" so entries appear in PainTrackingHistoryView

        // Mental health metrics
        let moodBase: Int16
        switch state {
        case .good: moodBase = Int16.random(in: 6...8)
        case .average: moodBase = Int16.random(in: 5...7)
        case .bad: moodBase = Int16.random(in: 3...5)
        case .flare(let severity, _): moodBase = Int16(max(2, 8 - severity))
        }
        log.moodScore = moodBase
        log.stressLevel = Float(10 - moodBase) + Float.random(in: -1...1)
        log.anxietyLevel = Float(10 - moodBase) * 0.8 + Float.random(in: -1...1)
        log.mentalWellbeing = Float(moodBase)
        log.cognitiveFunction = Float(moodBase) + Float.random(in: -1...1)

        // Flare flag
        if case .flare = state {
            log.isFlareEvent = true
        } else {
            log.isFlareEvent = false
        }

        // Sleep metrics
        let biometrics = generateBiometrics(state: state)
        log.sleepQuality = Int16(biometrics.sleepEff * 10)
        log.sleepDurationHours = biometrics.sleepDur

        // CRP and clinical
        let crp = generateCRP(state: state)
        log.crpValue = crp
        log.crpLevel = Float(crp)

        // ASDAS-CRP calculation
        // ASDAS = 0.12×BackPain + 0.06×MorningStiffness + 0.11×PatientGlobal + 0.07×PeripheralPain + 0.58×ln(CRP+1)
        let naturalLog = Darwin.log(crp + 1)
        let asdas = 0.12 * basdaiAnswers[1] + 0.06 * (Double(log.morningStiffnessMinutes) / 12.0) +
                    0.11 * basdaiScore + 0.07 * basdaiAnswers[2] + 0.58 * naturalLog
        log.asdasScore = min(10, asdas)

        // Pain metrics
        log.painAverage24h = Float(basdaiScore * 0.9)
        log.painMax24h = Float(min(10, basdaiScore * 1.2 + Double.random(in: 0...1)))
        log.nocturnalPain = Float(basdaiAnswers[1] * 0.8)
        log.morningStiffnessSeverity = Float(basdaiAnswers[4])
        log.painAching = Float.random(in: 3...7)
        log.painBurning = Float.random(in: 1...4)
        log.painSharp = Float.random(in: 2...5)

        // Activity metrics
        log.exerciseMinutesToday = 0  // Will be updated if exercise logged
        log.energyLevel = Float(10 - log.fatigueLevel)
        log.physicalFunctionScore = Float(max(3, 10 - basdaiScore))

        // Create context snapshot
        let (pressure, pressureChange, humidity, temp, precipitation) = generateWeather(day: day, isFlareDay: flare != nil)

        let snapshot = ContextSnapshot(context: context)
        snapshot.id = UUID()
        snapshot.timestamp = log.timestamp
        snapshot.barometricPressure = pressure
        snapshot.pressureChange12h = pressureChange
        snapshot.humidity = humidity
        snapshot.temperature = temp
        snapshot.precipitation = precipitation
        snapshot.hrvValue = biometrics.hrv
        snapshot.restingHeartRate = biometrics.hr
        snapshot.stepCount = biometrics.steps
        snapshot.sleepEfficiency = biometrics.sleepEff
        snapshot.sleepDuration = biometrics.sleepDur
        log.contextSnapshot = snapshot

        // Create body region logs
        let regionPains = generateBodyRegionPain(state: state, flare: flare)
        for regionPain in regionPains {
            let regionLog = BodyRegionLog(context: context)
            regionLog.id = UUID()
            regionLog.regionID = regionPain.regionID
            regionLog.painLevel = regionPain.pain
            regionLog.stiffnessDuration = regionPain.stiffness
            regionLog.swelling = regionPain.swelling
            regionLog.warmth = regionPain.warmth
            regionLog.symptomLog = log
        }

        log.painLocationCount = Int16(regionPains.count)

        // Create trigger logs for this day
        for trigger in triggerCategories {
            if Double.random(in: 0...1) < trigger.frequency {
                let triggerLog = TriggerLog(context: context)
                triggerLog.id = UUID()
                triggerLog.timestamp = log.timestamp
                triggerLog.triggerName = trigger.name
                triggerLog.triggerCategory = trigger.category
                triggerLog.isBinary = true
                triggerLog.triggerValue = 1.0
                triggerLog.source = "manual"
                triggerLog.confidence = 0.9
                triggerLog.symptomLog = log
            }
        }
    }

    private func createExerciseSession(context: NSManagedObjectContext, date: Date, state: DayState, routines: [UserRoutine]) throws {
        let session = ExerciseSession(context: context)
        session.id = UUID()

        // Set time to afternoon/evening
        var components = Calendar.current.dateComponents([.year, .month, .day], from: date)
        components.hour = Int.random(in: 16...20)
        components.minute = Int.random(in: 0...59)
        session.timestamp = Calendar.current.date(from: components) ?? date  // Fallback to base date

        // Decide whether to use a routine or standalone exercise
        var usedRoutine: UserRoutine?
        var usedRoutineDef: RoutineDefinition?

        // During flares, prefer flare routines
        let isInFlare = { () -> Bool in
            if case .flare = state { return true }
            return false
        }()

        if isInFlare {
            // 70% chance to use a flare routine during flares
            if Double.random(in: 0...1) < 0.70 {
                let flareRoutines = annaRoutines.filter { $0.isForFlares }
                if let selectedDef = flareRoutines.randomElement(),
                   let matchingRoutine = routines.first(where: { $0.id == selectedDef.id }) {
                    usedRoutine = matchingRoutine
                    usedRoutineDef = selectedDef
                }
            }
        } else {
            // On good/average days, decide between routines based on frequency
            let randomValue = Double.random(in: 0...1)
            var cumulativeProbability = 0.0

            for routineDef in annaRoutines where !routineDef.isForFlares {
                cumulativeProbability += routineDef.usageFrequency
                if randomValue < cumulativeProbability {
                    if let matchingRoutine = routines.first(where: { $0.id == routineDef.id }) {
                        usedRoutine = matchingRoutine
                        usedRoutineDef = routineDef
                        break
                    }
                }
            }
        }

        // If using a routine
        if let routine = usedRoutine, let routineDef = usedRoutineDef {
            session.routineType = routine.name
            session.flowTitle = routine.name
            session.durationMinutes = routine.totalDuration
            session.intensityLevel = isInFlare ? Int16.random(in: 2...4) : Int16.random(in: 3...5)

            // Create ExerciseCompletion records for each exercise in the routine
            for (index, exerciseName) in routineDef.exercises.enumerated() {
                let completion = ExerciseCompletion(context: context)
                completion.id = UUID()
                completion.timestamp = session.timestamp
                completion.exerciseName = exerciseName
                completion.exerciseID = UUID()  // Generate new ID
                completion.durationSeconds = Int32.random(in: 30...120)
                completion.fromRoutine = true
                completion.routineName = routine.name
                completion.routineID = routine.id
                completion.stepsCompleted = Int16.random(in: 4...8)
                completion.totalSteps = completion.stepsCompleted
                completion.wasCompleted = Double.random(in: 0...1) < 0.92  // 92% completion rate

                // Feedback
                let feedbacks = ["great", "good", "okay", "challenging", nil]
                completion.feedback = feedbacks.randomElement() ?? nil

                if index == 0 || index == routineDef.exercises.count - 1 {
                    // First or last exercise might get notes
                    let feedbackNotes = [
                        "Felt nice and loose after this one",
                        "A bit stiff at first",
                        "Good stretch",
                        nil, nil, nil
                    ]
                    completion.feedbackNotes = feedbackNotes.randomElement() ?? nil
                }
            }
        } else {
            // Standalone exercise (no routine)
            guard let exerciseType = exerciseTypes.randomElement() else {
                print("⚠️ Exercise types array is empty - skipping session")
                return
            }

            session.routineType = exerciseType.type
            session.durationMinutes = Int16(Int.random(in: exerciseType.duration))
            session.intensityLevel = Int16(Int.random(in: exerciseType.intensity))

            // Create single ExerciseCompletion record
            let completion = ExerciseCompletion(context: context)
            completion.id = UUID()
            completion.timestamp = session.timestamp
            completion.exerciseName = exerciseType.type
            completion.exerciseID = UUID()
            completion.durationSeconds = Int32(session.durationMinutes) * 60
            completion.fromRoutine = false
            completion.stepsCompleted = Int16.random(in: 3...6)
            completion.totalSteps = completion.stepsCompleted
            completion.wasCompleted = true
        }

        // Pain before/after
        let basePain: Int16
        switch state {
        case .good: basePain = 3
        case .average: basePain = 5
        case .bad: basePain = 6
        case .flare(let severity, _): basePain = Int16(severity)
        }

        session.painBefore = basePain
        session.painAfter = max(1, basePain - Int16.random(in: 0...2))  // Usually improves after exercise
        session.completedSuccessfully = true
        session.stoppedEarly = false
        session.userConfidence = Int16.random(in: 3...5)
        if case .flare = state {
            session.wasInFlareMode = true
        } else {
            session.wasInFlareMode = false
        }
        session.romMultiplier = Double.random(in: 0.7...1.0)
        session.speedMultiplier = Double.random(in: 0.8...1.0)
        session.cyclesCompleted = Int16.random(in: 2...4)
        session.cyclesTarget = session.cyclesCompleted

        // Notes
        let notes = [
            "Felt good during stretches, hips a bit tight",
            "Good session, managed all exercises",
            "Had to modify some moves but completed",
            "Energy was low but pushed through",
            "Best session this week!",
            "Took extra breaks but finished",
            nil, nil  // Sometimes no notes
        ]
        session.notes = notes.randomElement() ?? nil
    }

    private func createMeditationSession(context: NSManagedObjectContext, date: Date, state: DayState) throws {
        guard let meditationType = meditationTypes.randomElement() else {
            print("⚠️ Meditation types array is empty - skipping session")
            return
        }

        let session = MeditationSession(context: context)
        session.id = UUID()

        // Set time (morning or evening)
        var components = Calendar.current.dateComponents([.year, .month, .day], from: date)
        components.hour = Bool.random() ? Int.random(in: 7...9) : Int.random(in: 20...22)
        components.minute = Int.random(in: 0...59)
        session.timestamp = Calendar.current.date(from: components) ?? date  // Fallback to base date

        session.sessionType = meditationType.type
        session.title = meditationType.title
        session.category = meditationType.category
        session.durationSeconds = Int32(Int.random(in: meditationType.duration) * 60)
        session.completedDuration = Int32(Double(session.durationSeconds) * Double.random(in: 0.85...1.0))
        session.isCompleted = session.completedDuration >= Int32(Double(session.durationSeconds) * 0.8)
        session.difficulty = ["beginner", "intermediate"].randomElement()
        session.breathingTechnique = ["4-7-8", "box breathing", "diaphragmatic"].randomElement()

        // Before/after metrics
        let stressBefore: Int16
        switch state {
        case .good: stressBefore = Int16.random(in: 3...5)
        case .average: stressBefore = Int16.random(in: 4...6)
        case .bad: stressBefore = Int16.random(in: 5...7)
        case .flare: stressBefore = Int16.random(in: 6...8)
        }

        session.stressLevelBefore = stressBefore
        session.stressLevelAfter = max(1, stressBefore - Int16.random(in: 1...3))

        session.painLevelBefore = stressBefore
        session.painLevelAfter = max(1, session.painLevelBefore - Int16.random(in: 0...2))

        session.moodBefore = 10 - stressBefore
        session.moodAfter = min(10, session.moodBefore + Int16.random(in: 1...2))

        session.energyBefore = 10 - stressBefore
        session.energyAfter = min(10, session.energyBefore + Int16.random(in: 0...2))

        session.avgHeartRate = Double.random(in: 58...72)
        session.hrvValue = Double.random(in: 35...55)
    }

    private func createJournalEntry(context: NSManagedObjectContext, date: Date, state: DayState) throws {
        let entry = JournalEntry(context: context)
        entry.id = UUID()
        entry.date = date

        let moodCategory: String
        let themes: [String]

        switch state {
        case .good:
            moodCategory = "positive"
            themes = journalThemes.first { $0.mood == "positive" }?.themes ?? []
        case .average:
            moodCategory = "neutral"
            themes = journalThemes.first { $0.mood == "neutral" }?.themes ?? []
        case .bad:
            moodCategory = "challenging"
            themes = journalThemes.first { $0.mood == "challenging" }?.themes ?? []
        case .flare:
            moodCategory = "flare"
            themes = journalThemes.first { $0.mood == "flare" }?.themes ?? []
        }

        entry.mood = moodCategory
        entry.notes = themes.randomElement()

        // Metrics
        switch state {
        case .good:
            entry.energyLevel = Double.random(in: 6...8)
            entry.sleepQuality = Double.random(in: 6...8)
            entry.painLevel = Double.random(in: 2...4)
        case .average:
            entry.energyLevel = Double.random(in: 4...6)
            entry.sleepQuality = Double.random(in: 5...7)
            entry.painLevel = Double.random(in: 4...6)
        case .bad:
            entry.energyLevel = Double.random(in: 3...5)
            entry.sleepQuality = Double.random(in: 3...5)
            entry.painLevel = Double.random(in: 5...7)
        case .flare(let severity, _):
            entry.energyLevel = Double(max(1, 8 - severity))
            entry.sleepQuality = Double(max(2, 7 - severity))
            entry.painLevel = Double(severity)
        }

        // Symptoms mentioned
        let symptoms = ["morning stiffness", "fatigue", "hip pain", "back pain", "neck pain", "knee discomfort"]
        entry.symptoms = symptoms.shuffled().prefix(Int.random(in: 1...3)).joined(separator: ", ")

        // Activities
        let activities = ["stretching", "walking", "swimming", "rest", "work", "physio exercises", "meditation"]
        entry.activities = activities.shuffled().prefix(Int.random(in: 1...3)).joined(separator: ", ")
    }

    private func createDoseLogs(context: NSManagedObjectContext, totalDays: Int) throws {
        let calendar = Calendar.current
        let today = Date()

        // Fetch medications
        let request: NSFetchRequest<Medication> = Medication.fetchRequest()
        request.predicate = NSPredicate(format: "isActive == YES")
        let medications = try context.fetch(request)

        for med in medications {
            guard let medName = med.name else { continue }

            for day in 0..<totalDays {
                guard let logDate = calendar.date(byAdding: .day, value: -(totalDays - 1 - day), to: today) else { continue }

                var shouldLog = false

                if medName.contains("Cosentyx") {
                    // Every 28 days
                    shouldLog = day % 28 == 0
                } else if medName.contains("Vitamin") || medName.contains("Calcium") {
                    // Daily with 85% adherence
                    shouldLog = Double.random(in: 0...1) < 0.85
                } else if medName.contains("Celebrex") {
                    // PRN - during flares/bad days (random 15% of days)
                    shouldLog = Double.random(in: 0...1) < 0.15
                }

                if shouldLog {
                    let doseLog = DoseLog(context: context)
                    doseLog.id = UUID()

                    var components = calendar.dateComponents([.year, .month, .day], from: logDate)
                    components.hour = medName.contains("Cosentyx") ? 10 : 8
                    components.minute = Int.random(in: 0...30)
                    doseLog.timestamp = calendar.date(from: components)
                    doseLog.scheduledTime = doseLog.timestamp
                    doseLog.dosageTaken = med.dosage
                    doseLog.wasSkipped = false
                    doseLog.medication = med
                }
            }
        }
    }

    private func createMeditationStreak(context: NSManagedObjectContext, totalSessions: Int) throws {
        let streak = MeditationStreak(context: context)
        streak.id = UUID()
        streak.currentStreak = Int16.random(in: 3...12)
        streak.longestStreak = Int16(max(Int(streak.currentStreak), Int.random(in: 10...21)))
        streak.totalSessions = Int32(totalSessions)
        streak.totalMinutes = Double(totalSessions * Int.random(in: 12...18))
        streak.lastSessionDate = Calendar.current.date(byAdding: .day, value: -Int.random(in: 0...2), to: Date())
        streak.weeklyGoal = 5
        streak.monthlyGoal = 20
        streak.weeklyProgress = Int16.random(in: 2...5)
        streak.monthlyProgress = Int16.random(in: 12...18)
        streak.createdAt = Calendar.current.date(byAdding: .day, value: -totalDays, to: Date())
        streak.lastUpdated = Date()
    }

    // MARK: - Questionnaire Response Seeding

    /// Creates realistic questionnaire responses for Anna over the 200-day period
    /// - BASDAI: Every 5-7 days (disease activity monitoring)
    /// - BASFI: Weekly (functional assessment)
    /// - BAS-G: Weekly (global wellbeing)
    /// - ASQoL: Monthly (quality of life)
    private func createQuestionnaireResponses(context: NSManagedObjectContext, flares: [FlareEvent_Data]) throws -> Int {
        let calendar = Calendar.current
        let today = Date()
        let timezone = TimeZone(identifier: "Europe/Berlin") ?? .current
        var responseCount = 0

        // Track last submission dates to control frequency
        var lastBASDAI: Int = -7
        var lastBASFI: Int = -7
        var lastBASG: Int = -7
        var lastASQoL: Int = -30

        // Track previous scores for smooth transitions
        var previousBASDAIScore = 4.5
        var previousBASFIScore = 4.0
        var previousBASGScore = 4.5
        var previousASQoLScore = 8.0

        for day in 0..<totalDays {
            guard let logDate = calendar.date(byAdding: .day, value: -(totalDays - 1 - day), to: today) else { continue }

            // Determine day state for score correlation
            let state = calculateDayState(day: day, flares: flares)

            // BASDAI - Every 5-7 days (primary disease activity measure)
            if day - lastBASDAI >= Int.random(in: 5...7) {
                let score = generateQuestionnaireScore(
                    baseScore: previousBASDAIScore,
                    state: state,
                    questionnaireType: .basdai
                )
                previousBASDAIScore = score

                try createQuestionnaireResponse(
                    context: context,
                    questionnaireID: .basdai,
                    date: logDate,
                    score: score,
                    timezone: timezone,
                    state: state
                )
                lastBASDAI = day
                responseCount += 1
            }

            // BASFI - Weekly (functional index)
            if day - lastBASFI >= 7 {
                let score = generateQuestionnaireScore(
                    baseScore: previousBASFIScore,
                    state: state,
                    questionnaireType: .basfi
                )
                previousBASFIScore = score

                try createQuestionnaireResponse(
                    context: context,
                    questionnaireID: .basfi,
                    date: logDate,
                    score: score,
                    timezone: timezone,
                    state: state
                )
                lastBASFI = day
                responseCount += 1
            }

            // BAS-G - Weekly (global wellbeing)
            if day - lastBASG >= 7 {
                let score = generateQuestionnaireScore(
                    baseScore: previousBASGScore,
                    state: state,
                    questionnaireType: .basg
                )
                previousBASGScore = score

                try createQuestionnaireResponse(
                    context: context,
                    questionnaireID: .basg,
                    date: logDate,
                    score: score,
                    timezone: timezone,
                    state: state
                )
                lastBASG = day
                responseCount += 1
            }

            // ASQoL - Monthly (quality of life, 0-18 scale)
            if day - lastASQoL >= 28 {
                let score = generateQuestionnaireScore(
                    baseScore: previousASQoLScore,
                    state: state,
                    questionnaireType: .asqol
                )
                previousASQoLScore = score

                try createQuestionnaireResponse(
                    context: context,
                    questionnaireID: .asqol,
                    date: logDate,
                    score: score,
                    timezone: timezone,
                    state: state
                )
                lastASQoL = day
                responseCount += 1
            }
        }

        return responseCount
    }

    /// Generates a realistic score based on Anna's health state
    private func generateQuestionnaireScore(
        baseScore: Double,
        state: DayState,
        questionnaireType: QuestionnaireID
    ) -> Double {
        let maxScore: Double
        let goodRange: ClosedRange<Double>
        let averageRange: ClosedRange<Double>
        let badRange: ClosedRange<Double>
        let flareBaseMultiplier: Double

        switch questionnaireType {
        case .basdai:
            // BASDAI: 0-10 scale, higher = worse
            maxScore = 10.0
            goodRange = 2.5...4.0
            averageRange = 3.8...5.5
            badRange = 5.0...7.0
            flareBaseMultiplier = 0.85

        case .basfi:
            // BASFI: 0-10 scale, higher = worse function
            maxScore = 10.0
            goodRange = 2.0...3.5
            averageRange = 3.5...5.0
            badRange = 4.5...6.5
            flareBaseMultiplier = 0.80

        case .basg:
            // BAS-G: 0-10 scale, higher = worse wellbeing
            maxScore = 10.0
            goodRange = 2.0...4.0
            averageRange = 3.5...5.5
            badRange = 5.0...7.0
            flareBaseMultiplier = 0.85

        case .asqol:
            // ASQoL: 0-18 scale (count of "yes" answers), higher = worse QoL
            maxScore = 18.0
            goodRange = 4.0...7.0
            averageRange = 6.0...10.0
            badRange = 9.0...13.0
            flareBaseMultiplier = 0.75

        default:
            maxScore = 10.0
            goodRange = 2.0...4.0
            averageRange = 4.0...6.0
            badRange = 5.0...7.0
            flareBaseMultiplier = 0.8
        }

        let targetRange: ClosedRange<Double>
        switch state {
        case .good:
            targetRange = goodRange
        case .average:
            targetRange = averageRange
        case .bad:
            targetRange = badRange
        case .flare(let severity, let dayInFlare):
            // Peak at day 2-3 of flare, then gradual improvement
            let peakAdjustment = dayInFlare < 3 ? Double(dayInFlare) * 0.2 : max(0, 0.6 - Double(dayInFlare - 3) * 0.1)
            let baseValue = Double(severity) * flareBaseMultiplier + peakAdjustment
            let lowerBound = max(badRange.lowerBound, baseValue - 1.0)
            let upperBound = min(maxScore, baseValue + 1.0)
            targetRange = lowerBound...max(lowerBound, upperBound)
        }

        // Generate score with some continuity from previous (20% influence)
        let rawScore = Double.random(in: targetRange)
        let smoothedScore = rawScore * 0.8 + baseScore * 0.2
        return min(maxScore, max(0, smoothedScore))
    }

    /// Creates a single QuestionnaireResponse entity
    private func createQuestionnaireResponse(
        context: NSManagedObjectContext,
        questionnaireID: QuestionnaireID,
        date: Date,
        score: Double,
        timezone: TimeZone,
        state: DayState
    ) throws {
        guard let definition = QuestionnaireDefinition.definition(for: questionnaireID) else { return }

        let response = QuestionnaireResponse(context: context)
        response.id = UUID()
        response.questionnaireID = questionnaireID.rawValue

        // Set timestamp to evening (when questionnaires are typically filled)
        var components = Calendar.current.dateComponents([.year, .month, .day], from: date)
        components.hour = Int.random(in: 19...21)
        components.minute = Int.random(in: 0...59)
        let timestamp = Calendar.current.date(from: components) ?? date
        response.createdAt = timestamp

        // Format local date
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timezone
        let dateComponents = calendar.dateComponents([.year, .month, .day], from: timestamp)
        if let year = dateComponents.year, let month = dateComponents.month, let day = dateComponents.day {
            response.localDate = String(format: "%04d-%02d-%02d", year, month, day)
        }

        response.timezoneIdentifier = timezone.identifier
        response.score = score

        // Generate realistic individual answers that produce the target score
        let answers = generateAnswersForScore(
            questionnaireID: questionnaireID,
            definition: definition,
            targetScore: score,
            state: state
        )
        if let answersData = try? JSONEncoder().encode(answers) {
            response.answersData = answersData
        }

        // Duration: 90-180 seconds for most questionnaires
        let durationSeconds = Double.random(in: 90...180)
        response.durationMs = durationSeconds * 1000

        // Meta payload
        let meta = QuestionnaireMetaPayload(
            appVersion: "1.0.0",
            durationMs: response.durationMs,
            isDraft: false,
            deviceLocale: "de_DE"
        )
        if let metaData = try? JSONEncoder().encode(meta) {
            response.metaData = metaData
        }

        // Occasional notes
        if Double.random(in: 0...1) < 0.15 {
            response.note = generateQuestionnaireNote(state: state, questionnaireID: questionnaireID)
        }
    }

    /// Generates individual answers that approximately produce the target score
    private func generateAnswersForScore(
        questionnaireID: QuestionnaireID,
        definition: QuestionnaireDefinition,
        targetScore: Double,
        state: DayState
    ) -> [String: Double] {
        var answers: [String: Double] = [:]

        switch questionnaireID {
        case .basdai:
            // BASDAI formula: (Q1+Q2+Q3+Q4+((Q5+Q6)/2))/5
            // Generate answers that roughly produce the target score
            let baseValue = targetScore
            let variation = 1.5

            answers["Q1"] = min(10, max(0, baseValue + Double.random(in: -variation...variation) + 0.8)) // Fatigue (higher in women)
            answers["Q2"] = min(10, max(0, baseValue + Double.random(in: -variation...variation) - 0.3)) // Spinal pain (lower in women)
            answers["Q3"] = min(10, max(0, baseValue + Double.random(in: -variation...variation) + 0.5)) // Peripheral (higher in women)
            answers["Q4"] = min(10, max(0, baseValue + Double.random(in: -variation...variation)))       // Enthesitis
            answers["Q5"] = min(10, max(0, baseValue + Double.random(in: -variation...variation) + 0.4)) // Stiffness severity
            answers["Q6"] = min(10, max(0, baseValue + Double.random(in: -variation...variation) + 0.6)) // Stiffness duration

        case .basfi:
            // BASFI: Average of 10 items
            for i in 1...10 {
                let itemVariation = Double.random(in: -1.5...1.5)
                answers["Q\(i)"] = min(10, max(0, targetScore + itemVariation))
            }

        case .basg:
            // BAS-G: Average of 2 items
            let variation = Double.random(in: -1.0...1.0)
            answers["Q1"] = min(10, max(0, targetScore + variation))
            answers["Q2"] = min(10, max(0, targetScore - variation))

        case .asqol:
            // ASQoL: 18 yes/no questions, score is count of "yes" (1)
            let targetYesCount = Int(round(targetScore))
            var yesIndices = Set<Int>()

            // Randomly select which questions get "yes"
            while yesIndices.count < targetYesCount && yesIndices.count < 18 {
                yesIndices.insert(Int.random(in: 1...18))
            }

            for i in 1...18 {
                answers["Q\(i)"] = yesIndices.contains(i) ? 1.0 : 0.0
            }

        default:
            // Generic: Generate average answers
            for item in definition.items {
                let maxVal = Double(item.maximum)
                let ratio = targetScore / 10.0
                answers[item.id] = min(maxVal, max(Double(item.minimum), ratio * maxVal + Double.random(in: -1...1)))
            }
        }

        return answers
    }

    /// Generates occasional notes for questionnaire responses
    private func generateQuestionnaireNote(state: DayState, questionnaireID: QuestionnaireID) -> String {
        let notes: [String]

        switch state {
        case .good:
            notes = [
                "Feeling better this week.",
                "Good week overall.",
                "Exercise routine is helping.",
                "Sleep has been good.",
                "Medication working well."
            ]
        case .average:
            notes = [
                "About the same as usual.",
                "Some good days, some not so good.",
                "Weather might be affecting me.",
                "Trying to stay consistent with exercises.",
                "Manageable but not great."
            ]
        case .bad:
            notes = [
                "Rough week.",
                "Stiffness worse than usual.",
                "Fatigue is challenging.",
                "Stress from work affecting symptoms.",
                "Had to take extra rest."
            ]
        case .flare:
            notes = [
                "In a flare currently.",
                "SI joints very painful.",
                "Difficulty with daily activities.",
                "Increased medication for flare.",
                "Hope this passes soon.",
                "Third day of flare - struggling.",
                "Using heat packs and rest."
            ]
        }

        return notes.randomElement() ?? ""
    }
}

// MARK: - UserDefaults Key for Demo Mode

extension UserDefaults {
    static let demoDataSeededKey = "com.inflamai.demoDataSeeded"

    var hasDemoDataBeenSeeded: Bool {
        get { bool(forKey: Self.demoDataSeededKey) }
        set { set(newValue, forKey: Self.demoDataSeededKey) }
    }
}
