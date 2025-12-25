import Foundation
import CoreML
import Accelerate

/// Generates synthetic Ankylosing Spondylitis patient data that closely mimics real-world patterns
/// Based on medical literature and clinical studies for AS/Morbus Bechterew patients
class ASPatientSyntheticDataGenerator {

    // MARK: - Demographics Distribution
    struct Demographics {
        let age: Int                    // 18-44 range
        let gender: Gender
        let location: GeographicLocation
        let hlaB27Positive: Bool        // 80-95% positive in AS patients
        let diseaseOnsetAge: Int        // Typically 15-30
        let diseaseDuration: Int        // Years since onset
        let bmi: Double                 // Body Mass Index
        let smokingStatus: SmokingStatus
        let familyHistory: Bool         // 15-20% have family history
    }

    enum Gender: CaseIterable {
        case male, female

        // Real-world AS gender ratio is approximately 2:1 (male:female)
        var prevalenceWeight: Double {
            switch self {
            case .male: return 0.67
            case .female: return 0.33
            }
        }

        // Gender affects disease presentation
        var diseaseModifiers: DiseaseModifiers {
            switch self {
            case .male:
                return DiseaseModifiers(
                    radiographicProgressionRate: 1.2,
                    peripheralJointInvolvement: 0.35,
                    extraArticularManifestation: 0.25,
                    diagnosticDelay: 5.0  // years
                )
            case .female:
                return DiseaseModifiers(
                    radiographicProgressionRate: 0.8,
                    peripheralJointInvolvement: 0.45,
                    extraArticularManifestation: 0.30,
                    diagnosticDelay: 8.5  // years - longer in females
                )
            }
        }
    }

    struct DiseaseModifiers {
        let radiographicProgressionRate: Double
        let peripheralJointInvolvement: Double
        let extraArticularManifestation: Double
        let diagnosticDelay: Double
    }

    enum GeographicLocation: CaseIterable {
        case northernEurope     // Highest prevalence (0.5-1.4%)
        case centralEurope      // High prevalence (0.3-0.8%)
        case mediterraneanEurope // Moderate (0.2-0.5%)
        case northAmerica       // Moderate (0.2-0.7%)
        case eastAsia          // Lower (0.1-0.3%)
        case southAmerica      // Variable (0.1-0.5%)
        case africa            // Lowest (0.02-0.2%)
        case middleEast        // Moderate (0.2-0.4%)
        case southAsia         // Lower (0.1-0.3%)
        case oceania           // Moderate (0.2-0.6%)

        var hlaB27Prevalence: Double {
            switch self {
            case .northernEurope: return 0.95      // 95% HLA-B27+
            case .centralEurope: return 0.90
            case .mediterraneanEurope: return 0.75
            case .northAmerica: return 0.85
            case .eastAsia: return 0.90
            case .southAmerica: return 0.80
            case .africa: return 0.60               // Lower HLA-B27 association
            case .middleEast: return 0.70
            case .southAsia: return 0.85
            case .oceania: return 0.85
            }
        }

        var typicalWeatherPattern: WeatherPattern {
            switch self {
            case .northernEurope:
                return WeatherPattern(
                    avgBarometricPressure: 1013,
                    pressureVariability: 15,       // High variability
                    avgHumidity: 75,
                    avgTemperature: 10,
                    seasonalVariation: 20
                )
            case .mediterraneanEurope:
                return WeatherPattern(
                    avgBarometricPressure: 1015,
                    pressureVariability: 8,         // More stable
                    avgHumidity: 65,
                    avgTemperature: 18,
                    seasonalVariation: 15
                )
            case .eastAsia:
                return WeatherPattern(
                    avgBarometricPressure: 1012,
                    pressureVariability: 12,
                    avgHumidity: 70,
                    avgTemperature: 16,
                    seasonalVariation: 25
                )
            default:
                return WeatherPattern(
                    avgBarometricPressure: 1013,
                    pressureVariability: 10,
                    avgHumidity: 60,
                    avgTemperature: 15,
                    seasonalVariation: 15
                )
            }
        }
    }

    struct WeatherPattern {
        let avgBarometricPressure: Double  // hPa
        let pressureVariability: Double    // Standard deviation
        let avgHumidity: Double            // Percentage
        let avgTemperature: Double         // Celsius
        let seasonalVariation: Double      // Temperature range
    }

    enum SmokingStatus: CaseIterable {
        case never, former, current

        // Smoking affects AS progression
        var progressionMultiplier: Double {
            switch self {
            case .never: return 1.0
            case .former: return 1.2
            case .current: return 1.5  // Worse progression in smokers
            }
        }
    }

    // MARK: - Disease Activity Patterns
    struct DiseaseActivityPattern {
        let baseBASDAI: Double          // Baseline BASDAI (0-10)
        let variability: Double         // Day-to-day variation
        let flareFrequency: Double      // Flares per year
        let flareDuration: Int          // Days
        let flareIntensity: Double      // BASDAI increase during flare
        let morningStiffnessMinutes: Int
        let fatigueLevel: Double        // 0-10
        let sleepQuality: Double        // 0-10
    }

    // MARK: - Body Region Involvement
    struct BodyRegionPattern {
        let siJointInvolvement: Bool           // Almost always in AS
        let lumbarInvolvement: Double          // 0-1 probability
        let thoracicInvolvement: Double
        let cervicalInvolvement: Double
        let peripheralJoints: Set<PeripheralJoint>
        let enthesitisLocations: Set<EnthesitisLocation>
    }

    enum PeripheralJoint: String, CaseIterable {
        case leftHip, rightHip              // Most common (30-40%)
        case leftShoulder, rightShoulder    // Second most common (20-30%)
        case leftKnee, rightKnee            // 15-20%
        case leftAnkle, rightAnkle          // 10-15%
        case leftElbow, rightElbow          // 5-10%
        case leftWrist, rightWrist          // 5-10%
    }

    enum EnthesitisLocation: String, CaseIterable {
        case achillesTendon     // Most common
        case plantar fascia
        case greaterTrochanter
        case ischialTuberosity
        case iliacCrest
    }

    // MARK: - Treatment Response Patterns
    struct TreatmentResponse {
        let nsaidResponse: NSAIDResponse
        let biologicResponse: BiologicResponse?  // If on biologics
        let exerciseBenefit: Double             // 0-1 scale
        let physiotherapyResponse: Double       // 0-1 scale
    }

    enum NSAIDResponse {
        case excellent  // 70% reduction in symptoms
        case good      // 50% reduction
        case moderate  // 30% reduction
        case poor      // <30% reduction

        var basdaiReduction: Double {
            switch self {
            case .excellent: return 0.7
            case .good: return 0.5
            case .moderate: return 0.3
            case .poor: return 0.15
            }
        }
    }

    enum BiologicResponse {
        case basdai50    // 50% improvement (60-70% of patients)
        case basdai20    // 20% improvement (80-85% of patients)
        case nonResponder // 15-20% don't respond

        var basdaiReduction: Double {
            switch self {
            case .basdai50: return 0.5
            case .basdai20: return 0.2
            case .nonResponder: return 0.0
            }
        }
    }

    // MARK: - Comorbidities
    struct Comorbidities {
        let uveitis: Bool           // 25-40% lifetime risk
        let ibd: Bool              // 5-10%
        let psoriasis: Bool        // 10-15%
        let cardiovascular: Bool    // Increased risk
        let osteoporosis: Bool     // Common in advanced disease
        let depression: Bool       // 30-40%
        let anxiety: Bool          // 25-35%
    }

    // MARK: - Patient Profile Generator
    class PatientProfile {
        let id: UUID
        let demographics: Demographics
        let diseaseActivity: DiseaseActivityPattern
        let bodyRegions: BodyRegionPattern
        let treatment: TreatmentResponse
        let comorbidities: Comorbidities
        let createdAt: Date

        init(seed: Int? = nil) {
            if let seed = seed {
                srand48(seed)
            }

            self.id = UUID()
            self.createdAt = Date()

            // Generate demographics with realistic distributions
            let gender = Self.selectGender()
            let age = Self.generateAge()
            let location = Self.selectLocation()

            self.demographics = Demographics(
                age: age,
                gender: gender,
                location: location,
                hlaB27Positive: drand48() < location.hlaB27Prevalence,
                diseaseOnsetAge: Self.generateOnsetAge(),
                diseaseDuration: max(0, age - Self.generateOnsetAge()),
                bmi: Self.generateBMI(),
                smokingStatus: Self.selectSmokingStatus(),
                familyHistory: drand48() < 0.175  // 17.5% have family history
            )

            // Generate disease activity based on demographics
            self.diseaseActivity = Self.generateDiseaseActivity(
                demographics: demographics,
                location: location
            )

            // Generate body region involvement based on disease duration
            self.bodyRegions = Self.generateBodyRegionPattern(
                diseaseDuration: demographics.diseaseDuration,
                gender: gender
            )

            // Generate treatment response
            self.treatment = Self.generateTreatmentResponse(
                diseaseActivity: diseaseActivity
            )

            // Generate comorbidities
            self.comorbidities = Self.generateComorbidities(
                age: age,
                diseaseDuration: demographics.diseaseDuration
            )
        }

        // MARK: - Generation Methods
        static func selectGender() -> Gender {
            drand48() < 0.67 ? .male : .female
        }

        static func generateAge() -> Int {
            // Weighted towards 25-35 (peak diagnosis age)
            let base = Int(18 + drand48() * 26)  // 18-44
            if base >= 25 && base <= 35 {
                return base
            } else {
                // 30% chance to shift towards peak age
                return drand48() < 0.3 ? Int(25 + drand48() * 10) : base
            }
        }

        static func selectLocation() -> GeographicLocation {
            // Weighted selection based on AS prevalence
            let locations = GeographicLocation.allCases
            let weights = [0.20, 0.15, 0.12, 0.15, 0.08, 0.10, 0.05, 0.08, 0.05, 0.02]

            var cumulative = 0.0
            let random = drand48()

            for (index, weight) in weights.enumerated() {
                cumulative += weight
                if random < cumulative {
                    return locations[index]
                }
            }
            return .northAmerica
        }

        static func generateOnsetAge() -> Int {
            // Peak onset 20-30, rare before 15 or after 40
            let mean = 25.0
            let stdDev = 5.0
            let gaussian = mean + stdDev * sqrt(-2.0 * log(drand48())) * cos(2.0 * .pi * drand48())
            return max(15, min(40, Int(gaussian)))
        }

        static func generateBMI() -> Double {
            // AS patients tend to have slightly lower BMI
            let mean = 24.5
            let stdDev = 4.0
            let gaussian = mean + stdDev * sqrt(-2.0 * log(drand48())) * cos(2.0 * .pi * drand48())
            return max(18.5, min(35.0, gaussian))
        }

        static func selectSmokingStatus() -> SmokingStatus {
            let random = drand48()
            if random < 0.60 {
                return .never
            } else if random < 0.85 {
                return .former
            } else {
                return .current
            }
        }

        static func generateDiseaseActivity(demographics: Demographics, location: GeographicLocation) -> DiseaseActivityPattern {
            // Base BASDAI influenced by multiple factors
            var baseBASDAI = 3.0 + drand48() * 3.0  // 3-6 range

            // Adjust for disease duration (tends to stabilize over time)
            if demographics.diseaseDuration > 10 {
                baseBASDAI *= 0.9
            }

            // Adjust for treatment era (modern treatments = lower scores)
            baseBASDAI *= 0.85

            // Smoking increases activity
            if demographics.smokingStatus == .current {
                baseBASDAI *= 1.15
            }

            baseBASDAI = min(8.0, max(1.0, baseBASDAI))

            return DiseaseActivityPattern(
                baseBASDAI: baseBASDAI,
                variability: 0.5 + drand48() * 1.5,
                flareFrequency: 2.0 + drand48() * 4.0,  // 2-6 flares/year
                flareDuration: Int(7 + drand48() * 14),  // 7-21 days
                flareIntensity: 1.5 + drand48() * 2.0,
                morningStiffnessMinutes: Int(30 + drand48() * 90),  // 30-120 min
                fatigueLevel: baseBASDAI * 0.8 + drand48() * 2.0,
                sleepQuality: 8.0 - baseBASDAI * 0.6 - drand48() * 2.0
            )
        }

        static func generateBodyRegionPattern(diseaseDuration: Int, gender: Gender) -> BodyRegionPattern {
            // Disease progression: SI joints → Lumbar → Thoracic → Cervical
            let progressionFactor = min(1.0, Double(diseaseDuration) / 20.0)

            var lumbarInvolvement = 0.7 + progressionFactor * 0.3
            var thoracicInvolvement = 0.3 + progressionFactor * 0.5
            var cervicalInvolvement = 0.1 + progressionFactor * 0.4

            // Gender differences
            if gender == .female {
                cervicalInvolvement *= 1.2  // More cervical involvement in females
            }

            // Peripheral joint involvement
            var peripheralJoints = Set<PeripheralJoint>()
            let peripheralProbability = gender.diseaseModifiers.peripheralJointInvolvement

            if drand48() < peripheralProbability {
                // Hips most common
                if drand48() < 0.4 {
                    peripheralJoints.insert(drand48() < 0.5 ? .leftHip : .rightHip)
                }
                // Shoulders second
                if drand48() < 0.3 {
                    peripheralJoints.insert(drand48() < 0.5 ? .leftShoulder : .rightShoulder)
                }
                // Other joints less common
                if drand48() < 0.2 {
                    let otherJoints: [PeripheralJoint] = [.leftKnee, .rightKnee, .leftAnkle, .rightAnkle]
                    if let random = otherJoints.randomElement() {
                        peripheralJoints.insert(random)
                    }
                }
            }

            // Enthesitis locations
            var enthesitisLocations = Set<EnthesitisLocation>()
            if drand48() < 0.3 {  // 30% have enthesitis
                enthesitisLocations.insert(.achillesTendon)
                if drand48() < 0.4 {
                    enthesitisLocations.insert(.plantarFascia)
                }
            }

            return BodyRegionPattern(
                siJointInvolvement: true,  // Always involved in AS
                lumbarInvolvement: lumbarInvolvement,
                thoracicInvolvement: thoracicInvolvement,
                cervicalInvolvement: cervicalInvolvement,
                peripheralJoints: peripheralJoints,
                enthesitisLocations: enthesitisLocations
            )
        }

        static func generateTreatmentResponse(diseaseActivity: DiseaseActivityPattern) -> TreatmentResponse {
            // NSAID response (70-80% respond well)
            let nsaidResponse: NSAIDResponse
            let random = drand48()
            if random < 0.35 {
                nsaidResponse = .excellent
            } else if random < 0.65 {
                nsaidResponse = .good
            } else if random < 0.85 {
                nsaidResponse = .moderate
            } else {
                nsaidResponse = .poor
            }

            // Biologic response (if high disease activity)
            var biologicResponse: BiologicResponse? = nil
            if diseaseActivity.baseBASDAI > 4.0 {
                let bioRandom = drand48()
                if bioRandom < 0.65 {
                    biologicResponse = .basdai50
                } else if bioRandom < 0.85 {
                    biologicResponse = .basdai20
                } else {
                    biologicResponse = .nonResponder
                }
            }

            return TreatmentResponse(
                nsaidResponse: nsaidResponse,
                biologicResponse: biologicResponse,
                exerciseBenefit: 0.6 + drand48() * 0.3,  // Exercise always helps
                physiotherapyResponse: 0.5 + drand48() * 0.4
            )
        }

        static func generateComorbidities(age: Int, diseaseDuration: Int) -> Comorbidities {
            // Risk increases with disease duration
            let durationFactor = min(1.5, 1.0 + Double(diseaseDuration) / 20.0)

            return Comorbidities(
                uveitis: drand48() < (0.3 * durationFactor),           // 30% base risk
                ibd: drand48() < 0.075,                                // 7.5%
                psoriasis: drand48() < 0.125,                          // 12.5%
                cardiovascular: drand48() < (0.1 * Double(age) / 40),  // Age-dependent
                osteoporosis: drand48() < (0.05 * durationFactor),
                depression: drand48() < 0.35,                          // 35%
                anxiety: drand48() < 0.30                              // 30%
            )
        }
    }
}

// MARK: - Time Series Data Generator
extension ASPatientSyntheticDataGenerator {

    /// Generates daily symptom logs for a patient over a specified period
    static func generateTimeSeriesData(
        for patient: PatientProfile,
        days: Int,
        startDate: Date = Date().addingTimeInterval(-90 * 24 * 60 * 60) // 90 days ago
    ) -> [DailySymptomData] {
        var data: [DailySymptomData] = []
        var currentDate = startDate
        var daysSinceLastFlare = 30
        var isInFlare = false
        var flareDaysRemaining = 0

        let weatherPattern = patient.demographics.location.typicalWeatherPattern

        for dayIndex in 0..<days {
            // Determine if starting a flare
            if !isInFlare && daysSinceLastFlare > 14 {
                let flareProb = patient.diseaseActivity.flareFrequency / 365.0
                if drand48() < flareProb {
                    isInFlare = true
                    flareDaysRemaining = patient.diseaseActivity.flareDuration
                    daysSinceLastFlare = 0
                }
            }

            // Update flare status
            if isInFlare {
                flareDaysRemaining -= 1
                if flareDaysRemaining <= 0 {
                    isInFlare = false
                }
            } else {
                daysSinceLastFlare += 1
            }

            // Calculate BASDAI for the day
            var dailyBASDAI = patient.diseaseActivity.baseBASDAI

            // Add daily variability
            dailyBASDAI += (drand48() - 0.5) * patient.diseaseActivity.variability

            // Increase during flare
            if isInFlare {
                dailyBASDAI += patient.diseaseActivity.flareIntensity * (Double(flareDaysRemaining) / Double(patient.diseaseActivity.flareDuration))
            }

            // Apply treatment effects
            dailyBASDAI *= (1.0 - patient.treatment.nsaidResponse.basdaiReduction * 0.7) // Partial effect
            if let biologicResponse = patient.treatment.biologicResponse {
                dailyBASDAI *= (1.0 - biologicResponse.basdaiReduction * 0.8)
            }

            // Weather effects
            let pressureChange = (drand48() - 0.5) * weatherPattern.pressureVariability * 2
            if pressureChange < -5.0 {  // Rapid pressure drop
                dailyBASDAI *= 1.15
            }

            // Generate weather data
            let weather = generateWeatherData(
                pattern: weatherPattern,
                dayOfYear: dayIndex,
                previousPressure: data.last?.weather.barometricPressure
            )

            // Generate body region data
            let bodyRegions = generateBodyRegionData(
                pattern: patient.bodyRegions,
                basePain: dailyBASDAI * 0.8,
                isFlare: isInFlare
            )

            // Generate biometric data
            let biometrics = generateBiometricData(
                diseaseActivity: dailyBASDAI,
                sleepQuality: patient.diseaseActivity.sleepQuality,
                isFlare: isInFlare
            )

            // Create daily data point
            let dailyData = DailySymptomData(
                date: currentDate,
                patientId: patient.id,
                basdaiScore: min(10.0, max(0.0, dailyBASDAI)),
                painLevel: min(10.0, max(0.0, dailyBASDAI * 0.85 + (drand48() - 0.5))),
                morningStiffnessMinutes: Int(Double(patient.diseaseActivity.morningStiffnessMinutes) * (0.8 + drand48() * 0.4)),
                fatigueLevel: min(10.0, max(0.0, patient.diseaseActivity.fatigueLevel + (isInFlare ? 2.0 : 0.0))),
                moodScore: min(10.0, max(1.0, 8.0 - dailyBASDAI * 0.3)),
                isFlareEvent: isInFlare,
                daysSinceLastFlare: daysSinceLastFlare,
                weather: weather,
                bodyRegions: bodyRegions,
                biometrics: biometrics
            )

            data.append(dailyData)
            currentDate = currentDate.addingTimeInterval(24 * 60 * 60) // Next day
        }

        return data
    }

    static func generateWeatherData(pattern: WeatherPattern, dayOfYear: Int, previousPressure: Double?) -> WeatherData {
        // Seasonal variation
        let seasonalOffset = sin(Double(dayOfYear) * 2 * .pi / 365) * pattern.seasonalVariation

        // Generate pressure with continuity
        let basePressure = previousPressure ?? pattern.avgBarometricPressure
        let pressureChange = (drand48() - 0.5) * pattern.pressureVariability
        let newPressure = basePressure + pressureChange * 0.3 // Smooth changes

        return WeatherData(
            barometricPressure: newPressure,
            pressureChange12h: pressureChange,
            temperature: pattern.avgTemperature + seasonalOffset + (drand48() - 0.5) * 5,
            humidity: min(100, max(20, pattern.avgHumidity + (drand48() - 0.5) * 20)),
            precipitation: drand48() < 0.2
        )
    }

    static func generateBodyRegionData(pattern: BodyRegionPattern, basePain: Double, isFlare: Bool) -> [BodyRegionData] {
        var regions: [BodyRegionData] = []
        let flareMultiplier = isFlare ? 1.5 : 1.0

        // SI Joints (always affected in AS)
        regions.append(BodyRegionData(
            regionID: "SI_LEFT",
            painLevel: min(10, Int(basePain * flareMultiplier)),
            stiffnessMinutes: Int(30 + drand48() * 60),
            swelling: false,
            warmth: isFlare && drand48() < 0.3
        ))
        regions.append(BodyRegionData(
            regionID: "SI_RIGHT",
            painLevel: min(10, Int(basePain * flareMultiplier)),
            stiffnessMinutes: Int(30 + drand48() * 60),
            swelling: false,
            warmth: isFlare && drand48() < 0.3
        ))

        // Lumbar spine
        if pattern.lumbarInvolvement > drand48() {
            for i in 1...5 {
                regions.append(BodyRegionData(
                    regionID: "L\(i)",
                    painLevel: min(10, Int(basePain * 0.9 * flareMultiplier)),
                    stiffnessMinutes: Int(20 + drand48() * 40),
                    swelling: false,
                    warmth: false
                ))
            }
        }

        // Thoracic spine
        if pattern.thoracicInvolvement > drand48() {
            for i in [1, 6, 12] {  // Sample thoracic regions
                regions.append(BodyRegionData(
                    regionID: "T\(i)",
                    painLevel: min(10, Int(basePain * 0.7 * flareMultiplier)),
                    stiffnessMinutes: Int(15 + drand48() * 30),
                    swelling: false,
                    warmth: false
                ))
            }
        }

        // Cervical spine
        if pattern.cervicalInvolvement > drand48() {
            for i in [2, 4, 6] {  // Sample cervical regions
                regions.append(BodyRegionData(
                    regionID: "C\(i)",
                    painLevel: min(10, Int(basePain * 0.6 * flareMultiplier)),
                    stiffnessMinutes: Int(10 + drand48() * 20),
                    swelling: false,
                    warmth: false
                ))
            }
        }

        // Peripheral joints
        for joint in pattern.peripheralJoints {
            regions.append(BodyRegionData(
                regionID: joint.rawValue,
                painLevel: min(10, Int(basePain * 0.8 * flareMultiplier)),
                stiffnessMinutes: Int(15 + drand48() * 45),
                swelling: drand48() < 0.2,
                warmth: drand48() < 0.1
            ))
        }

        // Enthesitis locations
        for location in pattern.enthesitisLocations {
            regions.append(BodyRegionData(
                regionID: location.rawValue,
                painLevel: min(10, Int(basePain * 0.6 * flareMultiplier)),
                stiffnessMinutes: 0,
                swelling: drand48() < 0.15,
                warmth: drand48() < 0.1
            ))
        }

        return regions
    }

    static func generateBiometricData(diseaseActivity: Double, sleepQuality: Double, isFlare: Bool) -> BiometricData {
        // HRV decreases with inflammation
        let baseHRV = 50.0 - diseaseActivity * 3.0
        let hrvValue = max(20, baseHRV - (isFlare ? 10.0 : 0.0) + (drand48() - 0.5) * 10)

        // Resting HR increases with inflammation
        let restingHR = 60.0 + diseaseActivity * 2.0 + (isFlare ? 5.0 : 0.0) + (drand48() - 0.5) * 5

        // Activity decreases with pain
        let stepCount = max(1000, Int(10000 - diseaseActivity * 1000 - (isFlare ? 2000 : 0) + (drand48() - 0.5) * 2000))

        // Sleep affected by pain
        let sleepHours = max(4, min(9, 8.0 - diseaseActivity * 0.3 + (drand48() - 0.5)))
        let sleepEfficiency = max(0.5, min(1.0, 0.9 - diseaseActivity * 0.05))

        return BiometricData(
            hrvValue: hrvValue,
            restingHeartRate: Int(restingHR),
            stepCount: stepCount,
            sleepDurationHours: sleepHours,
            sleepEfficiency: sleepEfficiency
        )
    }
}

// MARK: - Data Structures
struct DailySymptomData {
    let date: Date
    let patientId: UUID
    let basdaiScore: Double
    let painLevel: Double
    let morningStiffnessMinutes: Int
    let fatigueLevel: Double
    let moodScore: Double
    let isFlareEvent: Bool
    let daysSinceLastFlare: Int
    let weather: WeatherData
    let bodyRegions: [BodyRegionData]
    let biometrics: BiometricData
}

struct WeatherData {
    let barometricPressure: Double
    let pressureChange12h: Double
    let temperature: Double
    let humidity: Double
    let precipitation: Bool
}

struct BodyRegionData {
    let regionID: String
    let painLevel: Int
    let stiffnessMinutes: Int
    let swelling: Bool
    let warmth: Bool
}

struct BiometricData {
    let hrvValue: Double
    let restingHeartRate: Int
    let stepCount: Int
    let sleepDurationHours: Double
    let sleepEfficiency: Double
}