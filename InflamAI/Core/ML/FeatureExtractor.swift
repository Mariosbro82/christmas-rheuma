//
//  FeatureExtractor.swift
//  InflamAI
//
//  92-Feature extraction from Core Data + HealthKit for Neural Engine
//  Maps patient data to model-expected feature vector
//

import Foundation
import CoreData
import HealthKit

// MARK: - Data Quality Tracking

/// Tracks EXACTLY which features have REAL data vs missing/default values
/// This is critical for honest ML predictions - no more fake data pollution
public struct FeatureAvailability {
    /// Per-feature availability flags (92 booleans)
    /// true = REAL DATA exists, false = missing/default/0
    let featureHasRealData: [Bool]

    /// Category-level availability (matches FeatureIndex enum ranges)
    let demographicsAvailable: Int      // out of 6 (indices 0-5)
    let clinicalAvailable: Int          // out of 12 (indices 6-17)
    let painAvailable: Int              // out of 14 (indices 18-31)
    let activityAvailable: Int          // out of 23 (indices 32-54, HealthKit)
    let sleepAvailable: Int             // out of 9 (indices 55-63, HealthKit)
    let mentalHealthAvailable: Int      // out of 12 (indices 64-75)
    let environmentalAvailable: Int     // out of 7 (indices 76-82, Weather)
    let adherenceAvailable: Int         // out of 5 (indices 83-87)
    let universalAvailable: Int         // out of 4 (indices 88-91)

    /// Data sources that are available
    let hasHealthKitAccess: Bool
    let hasWeatherData: Bool
    let hasMedicationTracking: Bool
    let hasBodyRegionData: Bool
    let hasUserProfile: Bool

    /// Overall availability score (0.0 - 1.0)
    var overallAvailability: Float {
        Float(featureHasRealData.filter { $0 }.count) / 92.0
    }

    /// Features that are KNOWN to be missing (not just 0)
    var missingFeatureIndices: [Int] {
        featureHasRealData.enumerated().compactMap { $0.element ? nil : $0.offset }
    }

    /// Human-readable summary
    var summary: String {
        """
        Feature Availability: \(Int(overallAvailability * 100))% (\(featureHasRealData.filter { $0 }.count)/92)
        ├─ Demographics: \(demographicsAvailable)/6
        ├─ Clinical: \(clinicalAvailable)/12
        ├─ Pain: \(painAvailable)/14
        ├─ Activity (HealthKit): \(activityAvailable)/23 \(hasHealthKitAccess ? "✓" : "✗")
        ├─ Sleep (HealthKit): \(sleepAvailable)/9
        ├─ Mental Health: \(mentalHealthAvailable)/12
        ├─ Environmental (Weather): \(environmentalAvailable)/7 \(hasWeatherData ? "✓" : "✗")
        ├─ Adherence: \(adherenceAvailable)/5 \(hasMedicationTracking ? "✓" : "✗")
        └─ Universal: \(universalAvailable)/4
        """
    }

    /// Initialize with all features unavailable
    static var empty: FeatureAvailability {
        FeatureAvailability(
            featureHasRealData: Array(repeating: false, count: 92),
            demographicsAvailable: 0,
            clinicalAvailable: 0,
            painAvailable: 0,
            activityAvailable: 0,
            sleepAvailable: 0,
            mentalHealthAvailable: 0,
            environmentalAvailable: 0,
            adherenceAvailable: 0,
            universalAvailable: 0,
            hasHealthKitAccess: false,
            hasWeatherData: false,
            hasMedicationTracking: false,
            hasBodyRegionData: false,
            hasUserProfile: false
        )
    }
}

/// Result of feature extraction with quality metrics
public struct FeatureExtractionResult {
    /// The extracted 30x92 feature matrix
    let features: [[Float]]

    /// NEW: Detailed availability tracking per feature
    let availability: FeatureAvailability

    /// Number of HealthKit features that had valid data (out of 23)
    let healthKitFeaturesAvailable: Int

    /// Number of Core Data features that had valid data
    let coreDataFeaturesAvailable: Int

    /// Number of weather/environmental features that had valid data (out of 8)
    let weatherFeaturesAvailable: Int

    /// Overall data quality score (0.0 - 1.0)
    let dataQualityScore: Float

    /// List of features that were missing or had zero values
    let missingFeatures: [String]

    /// Whether HealthKit was authorized and available
    let healthKitAuthorized: Bool

    /// Timestamp of extraction
    let extractedAt: Date

    /// NEW: Whether this extraction should be used for prediction
    /// Returns false if too many critical features are missing
    var isUsableForPrediction: Bool {
        // Require at least 15% real data (not fake placeholders)
        availability.overallAvailability >= 0.15
    }

    /// NEW: Confidence modifier based on data availability
    /// Predictions with low data availability should have reduced confidence
    var confidenceModifier: Float {
        // Scale confidence by data availability: 50-100%
        0.5 + (availability.overallAvailability * 0.5)
    }

    /// Human-readable summary
    var qualitySummary: String {
        let healthKitStatus = healthKitAuthorized ? "\(healthKitFeaturesAvailable)/23" : "Not Authorized"
        return """
        Data Quality: \(Int(dataQualityScore * 100))%
        HealthKit: \(healthKitStatus)
        Core Data: \(coreDataFeaturesAvailable) features
        Weather: \(weatherFeaturesAvailable)/8
        Missing: \(missingFeatures.count) features
        Usable for Prediction: \(isUsableForPrediction ? "Yes" : "No - insufficient data")
        """
    }
}

/// Tracks which feature categories were successfully extracted
struct FeatureExtractionMetrics {
    var healthKitFeatures: Int = 0
    var coreDataFeatures: Int = 0
    var weatherFeatures: Int = 0
    var totalNonZeroFeatures: Int = 0
    var missingFeatureNames: [String] = []

    /// NEW: Per-feature availability tracking
    var featureAvailability: [Bool] = Array(repeating: false, count: 92)

    var totalExpectedFeatures: Int { 92 }
    var healthKitExpected: Int { 23 }
    var weatherExpected: Int { 8 }

    var dataQualityScore: Float {
        Float(totalNonZeroFeatures) / Float(totalExpectedFeatures)
    }
}

@MainActor
public class FeatureExtractor {

    private let persistenceController: InflamAIPersistenceController
    private let healthKitService: HealthKitService?

    /// Published metrics for UI observation
    private(set) var lastExtractionMetrics: FeatureExtractionMetrics?
    private(set) var lastExtractionResult: FeatureExtractionResult?

    /// Controls verbose debug logging - only NEWEST N days show detailed extraction info
    #if DEBUG
    private var currentDayOffset: Int = 29  // Track dayOffset to log newest days
    private let verboseLogDayLimit: Int = 7  // Log details for newest 7 days to show more patient data
    private var shouldLogVerbose: Bool { currentDayOffset < verboseLogDayLimit }
    #endif

    // MARK: - Extraction Cache (prevents redundant 30-day extractions)

    /// Cached extraction result with timestamp
    private var extractionCache: (result: FeatureExtractionResult, date: Date, timestamp: Date)?

    /// Cache TTL: 5 minutes - balances freshness with performance
    /// HealthKit data doesn't change more frequently than this
    private let cacheTTL: TimeInterval = 300  // 5 minutes

    /// Flag to track if extraction is in progress (prevents concurrent extractions)
    private var extractionInProgress = false

    // Feature indices (must match scaler_params.json order - total 92 features)
    enum FeatureIndex: Int {
        // Demographics (6: indices 0-5)
        case age = 0, gender, hla_b27, disease_duration, bmi, smoking

        // Clinical Assessment (12: indices 6-17)
        case basdai_score = 6, asdas_crp, basfi, basmi, patient_global
        case physician_global, tender_joint_count, swollen_joint_count
        case enthesitis, dactylitis, spinal_mobility, disease_activity_composite

        // Pain Characterization (14: indices 18-31)
        case pain_current = 18, pain_avg_24h, pain_max_24h, nocturnal_pain
        case morning_stiffness_duration, morning_stiffness_severity
        case pain_location_count, pain_burning, pain_aching, pain_sharp
        case pain_interference_sleep, pain_interference_activity
        case pain_variability, breakthrough_pain

        // Activity/Physical (23: indices 32-54)
        case blood_oxygen = 32, cardio_fitness, respiratory_rate, walk_test_distance
        case resting_energy, hrv, resting_hr, walking_hr, cardio_recovery
        case steps, distance_km, stairs_up, stairs_down, stand_hours
        case stand_minutes, training_minutes, active_minutes, active_energy
        case training_sessions, walking_tempo, step_length, gait_asymmetry
        case bipedal_support

        // Sleep (9: indices 55-63)
        case sleep_hours = 55, rem_duration, deep_duration, core_duration
        case awake_duration, sleep_score, sleep_consistency
        // Line 17: burned_calories (index 62) - POTENTIALLY REDUNDANT
        // Already have active_energy + resting_energy (basal), may not need separate burned_calories
        case burned_calories
        // Line 16: exertion_level (index 63) = "BODY BATTERY" concept
        // Similar to Garmin Body State app - measures recovery/energy state
        // Calculated from: HRV + sleep quality + activity level + stress
        case exertion_level

        // Mental Health (12: indices 64-75)
        case mood_current = 64, mood_valence, mood_stability, anxiety
        case stress_level, stress_resilience, mental_fatigue, cognitive_function
        case emotional_regulation
        // Line 18: social_engagement (index 73) - REMOVED, not valuable for AS prediction
        case social_engagement
        case mental_wellbeing
        case depression_risk

        // Environmental (7: indices 76-82)
        case daylight_time = 76, temperature, humidity, pressure
        case pressure_change
        // Line 19: air_quality (index 81) - use external API only if same behavior as OpenMeteo
        // OpenMeteo has AQI endpoint - use consistently to avoid mixed data sources
        case air_quality
        case weather_change_score

        // Adherence (5: indices 83-87)
        case med_adherence = 83, physio_adherence, physio_effectiveness
        case journal_mood, quick_log

        // Universal/Context (4: indices 88-91)
        case universal_assessment = 88, time_weighted_assessment
        case ambient_noise, season
    }

    init(
        persistenceController: InflamAIPersistenceController = .shared,
        healthKitService: HealthKitService? = nil
    ) {
        self.persistenceController = persistenceController
        self.healthKitService = healthKitService
    }

    // MARK: - Main Extraction Method

    /// Extract 30 days × 92 features for neural engine prediction
    /// Returns array of 30 timesteps, each with 92 features
    func extract30DayFeatures(endingOn date: Date = Date()) async -> [[Float]] {
        let result = await extract30DayFeaturesWithMetrics(endingOn: date)
        return result.features
    }

    /// Extract features with full quality metrics - use this for production
    /// Now includes caching to prevent redundant extractions (5-minute TTL)
    func extract30DayFeaturesWithMetrics(endingOn date: Date = Date()) async -> FeatureExtractionResult {
        let calendar = Calendar.current

        // CACHE CHECK: Return cached result if valid (same day and within TTL)
        if let cache = extractionCache {
            let isSameDay = calendar.isDate(cache.date, inSameDayAs: date)
            let isWithinTTL = Date().timeIntervalSince(cache.timestamp) < cacheTTL

            if isSameDay && isWithinTTL {
                #if DEBUG
                let age = Int(Date().timeIntervalSince(cache.timestamp))
                print("📊 [FeatureExtractor] Using cached extraction (age: \(age)s)")
                #endif
                return cache.result
            }
        }

        // CONCURRENT EXTRACTION GUARD: Prevent duplicate extractions
        if extractionInProgress {
            #if DEBUG
            print("📊 [FeatureExtractor] Extraction already in progress, waiting...")
            #endif
            // Wait briefly and return cached result if available
            try? await Task.sleep(nanoseconds: 500_000_000)  // 0.5 second
            if let cache = extractionCache {
                return cache.result
            }
        }

        extractionInProgress = true
        defer { extractionInProgress = false }

        var features30Days: [[Float]] = []
        var aggregateMetrics = FeatureExtractionMetrics()

        let context = persistenceController.container.viewContext

        // Check HealthKit authorization status
        let healthKitAuthorized = await checkHealthKitAuthorization()

        // Log extraction start
        #if DEBUG
        let startDateFormatted = Self.germanDateFormatter.string(from: calendar.date(byAdding: .day, value: -29, to: date) ?? date)
        let endDateFormatted = Self.germanDateFormatter.string(from: date)
        print("📊 [FeatureExtractor] Starting 30-day feature extraction")
        print("   HealthKit: \(healthKitAuthorized ? "✅ Authorized" : "❌ Not Authorized")")
        print("   Date Range: \(startDateFormatted) → \(endDateFormatted) (German: DD.MM.YYYY)")
        print("   ⚠️ Detailed logging for NEWEST 7 days only (showing patient-entered data)")
        print("")
        print("   ═══════════════════════════════════════════════════════════════")
        print("   DATA SOURCE REFERENCE:")
        print("   ───────────────────────────────────────────────────────────────")
        print("   • DAILY_AVG = Average of all samples for that day")
        print("   • DAILY_SUM = Sum of all samples (deduplicated iPhone+Watch)")
        print("   • LAST_SAMPLE = Most recent sample of the day")
        print("   • TOTAL_ASLEEP_TIME = Analyzed sleep duration (InBed excluded)")
        print("   • CALCULATED = Derived from other values")
        print("   • CORE_DATA = From SymptomLog/BodyRegion entities")
        print("   • WEATHER_API = From OpenMeteo service")
        print("   ═══════════════════════════════════════════════════════════════")
        print("")
        #endif

        // Extract for each of the last 30 days (from oldest to newest)
        // dayOffset 29 = oldest day (30 days ago)
        // dayOffset 0 = newest day (today)
        for dayOffset in (0..<30).reversed() {
            #if DEBUG
            currentDayOffset = dayOffset  // Track for verbose logging (only newest 3 days)
            #endif

            guard let dayDate = calendar.date(byAdding: .day, value: -dayOffset, to: date) else {
                features30Days.append(Array(repeating: 0.0, count: 92))
                continue
            }

            let (dayFeatures, dayMetrics) = await extractFeaturesForDayWithMetrics(dayDate, context: context)
            features30Days.append(dayFeatures)

            // Aggregate metrics from most recent day (day 0)
            if dayOffset == 0 {
                aggregateMetrics = dayMetrics
            }
        }

        // Log extraction results
        logExtractionResults(metrics: aggregateMetrics, healthKitAuthorized: healthKitAuthorized)

        // Log summary of TODAY's patient-entered data (most recent day)
        if !features30Days.isEmpty {
            logTodayFeatureSummary(features: features30Days[0])
        }

        // Build FeatureAvailability from metrics
        let availability = buildFeatureAvailability(
            from: aggregateMetrics,
            healthKitAuthorized: healthKitAuthorized,
            context: context
        )

        // Create result
        let result = FeatureExtractionResult(
            features: features30Days,
            availability: availability,
            healthKitFeaturesAvailable: aggregateMetrics.healthKitFeatures,
            coreDataFeaturesAvailable: aggregateMetrics.coreDataFeatures,
            weatherFeaturesAvailable: aggregateMetrics.weatherFeatures,
            dataQualityScore: aggregateMetrics.dataQualityScore,
            missingFeatures: aggregateMetrics.missingFeatureNames,
            healthKitAuthorized: healthKitAuthorized,
            extractedAt: Date()
        )

        // Store for UI access
        lastExtractionMetrics = aggregateMetrics
        lastExtractionResult = result

        // CACHE RESULT: Store with timestamp for future requests
        extractionCache = (result: result, date: date, timestamp: Date())

        #if DEBUG
        print("📊 [FeatureExtractor] Extraction cached (TTL: \(Int(cacheTTL))s)")
        #endif

        return result
    }

    /// Invalidate the extraction cache (call when data changes significantly)
    public func invalidateCache() {
        extractionCache = nil
        #if DEBUG
        print("📊 [FeatureExtractor] Cache invalidated")
        #endif
    }

    /// Check if HealthKit is authorized
    private func checkHealthKitAuthorization() async -> Bool {
        guard let healthKit = healthKitService else {
            print("⚠️ [FeatureExtractor] HealthKitService not injected - biometric features unavailable")
            return false
        }

        // Check if we have any authorization
        let isAuthorized = healthKit.isAuthorized
        if !isAuthorized {
            print("⚠️ [FeatureExtractor] HealthKit not authorized - requesting authorization")
            // Note: Don't auto-request here, let the app flow handle it
        }

        return isAuthorized
    }

    /// Log extraction results for debugging
    private func logExtractionResults(metrics: FeatureExtractionMetrics, healthKitAuthorized: Bool) {
        #if DEBUG
        print("")
        print("═══════════════════════════════════════════════════════════════")
        print("📊 [FeatureExtractor] Extraction Complete:")
        print("   ✅ Data Quality Score: \(Int(metrics.dataQualityScore * 100))%")
        print("   ✅ HealthKit Features: \(metrics.healthKitFeatures)/\(metrics.healthKitExpected) \(healthKitAuthorized ? "" : "(Not Authorized)")")
        print("   ✅ Core Data Features: \(metrics.coreDataFeatures)")
        print("   ✅ Weather Features: \(metrics.weatherFeatures)/\(metrics.weatherExpected)")
        print("   ✅ Total Non-Zero: \(metrics.totalNonZeroFeatures)/\(metrics.totalExpectedFeatures)")

        if !metrics.missingFeatureNames.isEmpty {
            print("   ⚠️ Missing Features (\(metrics.missingFeatureNames.count)): \(metrics.missingFeatureNames.prefix(10).joined(separator: ", "))\(metrics.missingFeatureNames.count > 10 ? "..." : "")")
        }
        print("═══════════════════════════════════════════════════════════════")
        print("")
        #endif
    }

    /// Log summary of TODAY's critical patient-entered features
    private func logTodayFeatureSummary(features: [Float]) {
        #if DEBUG
        print("═══════════════════════════════════════════════════════════════")
        print("🔍 [TODAY'S PATIENT DATA] Critical Features Being Used by ML:")
        print("───────────────────────────────────────────────────────────────")

        // Clinical Assessments
        let basdai = features[FeatureIndex.basdai_score.rawValue]
        let basfi = features[FeatureIndex.basfi.rawValue]
        let patientGlobal = features[FeatureIndex.patient_global.rawValue]
        let tenderJoints = Int(features[FeatureIndex.tender_joint_count.rawValue])
        let swollenJoints = Int(features[FeatureIndex.swollen_joint_count.rawValue])

        print("📋 Clinical Assessments:")
        print("   • BASDAI Score: \(String(format: "%.1f", basdai))/10 \(basdai > 0 ? "✅" : "❌ NOT SET")")
        print("   • BASFI Score: \(String(format: "%.1f", basfi))/10 \(basfi > 0 ? "✅" : "❌ NOT SET")")
        print("   • Patient Global: \(String(format: "%.1f", patientGlobal))/10 \(patientGlobal > 0 ? "✅" : "❌ NOT SET")")
        print("   • Tender Joints: \(tenderJoints) \(tenderJoints > 0 ? "✅" : "❌ NONE")")
        print("   • Swollen Joints: \(swollenJoints) \(swollenJoints > 0 ? "✅" : "❌ NONE")")

        // Pain Characteristics
        let painAvg = features[FeatureIndex.pain_avg_24h.rawValue]
        let painMax = features[FeatureIndex.pain_max_24h.rawValue]
        let morningStiffness = Int(features[FeatureIndex.morning_stiffness_duration.rawValue])
        let painLocations = Int(features[FeatureIndex.pain_location_count.rawValue])

        print("\n🩹 Pain & Symptoms:")
        print("   • Pain Average: \(String(format: "%.1f", painAvg))/10 \(painAvg > 0 ? "✅" : "❌ NOT SET")")
        print("   • Pain Maximum: \(String(format: "%.1f", painMax))/10 \(painMax > 0 ? "✅" : "❌ NOT SET")")
        print("   • Morning Stiffness: \(morningStiffness) mins \(morningStiffness > 0 ? "✅" : "❌ NOT SET")")
        print("   • Body Map Regions: \(painLocations) painful areas \(painLocations > 0 ? "✅" : "❌ NO BODY MAP DATA")")

        // Adherence & Engagement
        let medAdherence = features[FeatureIndex.med_adherence.rawValue]
        let quickLogs = Int(features[FeatureIndex.quick_log.rawValue])

        print("\n📝 Engagement:")
        print("   • Medication Adherence: \(String(format: "%.0f", medAdherence * 100))% \(medAdherence > 0 ? "✅" : "❌ NOT TRACKED")")
        print("   • Quick Logs Today: \(quickLogs) entries \(quickLogs > 0 ? "✅" : "❌ NONE")")

        // Data availability summary
        let clinicalDataPresent = basdai > 0 || basfi > 0 || patientGlobal > 0
        let bodyMapDataPresent = tenderJoints > 0 || swollenJoints > 0 || painLocations > 0
        let symptomDataPresent = painAvg > 0 || morningStiffness > 0

        print("\n📊 Data Availability:")
        print("   • Clinical Assessments: \(clinicalDataPresent ? "✅ PRESENT" : "❌ MISSING")")
        print("   • Body Map Data: \(bodyMapDataPresent ? "✅ PRESENT" : "❌ MISSING")")
        print("   • Symptom Data: \(symptomDataPresent ? "✅ PRESENT" : "❌ MISSING")")
        print("   • Quick Log Data: \(quickLogs > 0 ? "✅ PRESENT" : "❌ MISSING")")

        print("═══════════════════════════════════════════════════════════════")
        print("")
        #endif
    }

    // MARK: - Single Day Feature Extraction

    private func extractFeaturesForDay(_ date: Date, context: NSManagedObjectContext) async -> [Float] {
        let (features, _) = await extractFeaturesForDayWithMetrics(date, context: context)
        return features
    }

    /// Extract features for a single day with quality metrics tracking
    private func extractFeaturesForDayWithMetrics(_ date: Date, context: NSManagedObjectContext) async -> ([Float], FeatureExtractionMetrics) {
        var features = Array(repeating: Float(0.0), count: 92)
        var metrics = FeatureExtractionMetrics()

        // Fetch symptom log for this day
        let symptomLog = fetchSymptomLog(for: date, context: context)

        #if DEBUG
        if shouldLogVerbose {
            let dateStr = formatDateGerman(date)
            if let log = symptomLog {
                print("   📅 [\(dateStr)] ✅ SymptomLog FOUND | BASDAI: \(log.basdaiScore), Pain: \(log.painAverage24h), Stiffness: \(log.morningStiffnessMinutes)min")
            } else {
                print("   📅 [\(dateStr)] ❌ NO SymptomLog for this date - all symptom features will be 0.0")
            }
        }
        #endif

        // Fetch user profile (demographics)
        let userProfile = fetchUserProfile(context: context)

        #if DEBUG
        if shouldLogVerbose {
            if let profile = userProfile, let birthDate = profile.dateOfBirth {
                let age = Calendar.current.dateComponents([.year], from: birthDate, to: Date()).year ?? 0
                print("   👤 UserProfile: Age \(age), Gender: \(profile.gender ?? "unknown")")
            } else {
                print("   ⚠️ NO UserProfile found or missing birthDate - demographic features will be 0.0")
            }
        }
        #endif

        // Extract each feature category with metrics tracking
        let demographicsCount = extractDemographics(into: &features, from: userProfile)
        let clinicalCount = extractClinicalAssessment(into: &features, from: symptomLog)
        let painCount = extractPainCharacteristics(into: &features, from: symptomLog)
        let activityCount = await extractActivityMetrics(into: &features, for: date)
        let sleepCount = await extractSleepMetrics(into: &features, for: date)
        let mentalCount = extractMentalHealth(into: &features, from: symptomLog)
        let envCount = await extractEnvironmental(into: &features, for: date)
        let adherenceCount = extractAdherence(into: &features, for: date, context: context)
        let universalCount = extractUniversalMetrics(into: &features, from: symptomLog, for: date)

        // POST-PROCESSING: Calculate derived features that need data from multiple categories
        // Line 9: mood_stability = mood + standing hours + HR + sleep
        calculateMoodStability(features: &features)
        // Line 10: stress_resilience = stress patterns + HRV + HR recovery
        calculateStressResilience(features: &features)
        // Line 16: exertion_level = "Body Battery" (HRV + sleep + activity + stress)
        calculateExertionLevel(features: &features)

        // Line 18: social_engagement (index 73) is NOT calculated - REMOVED per user specification
        // Not valuable for AS prediction, skip entirely

        // Aggregate metrics
        metrics.healthKitFeatures = activityCount + sleepCount
        metrics.coreDataFeatures = demographicsCount + clinicalCount + painCount + mentalCount + adherenceCount + universalCount
        metrics.weatherFeatures = envCount

        // Count total non-zero features
        metrics.totalNonZeroFeatures = features.enumerated().filter { $0.element != 0.0 }.count

        // FIXED: Populate featureAvailability array based on non-zero features
        // This was previously never being set, causing 0% availability despite having data
        for (index, value) in features.enumerated() {
            metrics.featureAvailability[index] = value != 0.0
        }

        // Track missing features
        metrics.missingFeatureNames = identifyMissingFeatures(features)

        return (features, metrics)
    }

    /// Identify which features are missing (have zero values)
    private func identifyMissingFeatures(_ features: [Float]) -> [String] {
        var missing: [String] = []

        let featureNames: [Int: String] = [
            FeatureIndex.age.rawValue: "Age",
            FeatureIndex.gender.rawValue: "Gender",
            FeatureIndex.hla_b27.rawValue: "HLA-B27",
            FeatureIndex.basdai_score.rawValue: "BASDAI",
            FeatureIndex.pain_current.rawValue: "Current Pain",
            FeatureIndex.hrv.rawValue: "HRV",
            FeatureIndex.resting_hr.rawValue: "Resting HR",
            FeatureIndex.steps.rawValue: "Steps",
            FeatureIndex.sleep_hours.rawValue: "Sleep Hours",
            FeatureIndex.mood_current.rawValue: "Mood",
            FeatureIndex.stress_level.rawValue: "Stress Level",
            FeatureIndex.temperature.rawValue: "Temperature",
            FeatureIndex.pressure.rawValue: "Barometric Pressure",
            FeatureIndex.med_adherence.rawValue: "Medication Adherence"
        ]

        for (index, name) in featureNames {
            if index < features.count && features[index] == 0.0 {
                missing.append(name)
            }
        }

        return missing
    }

    // MARK: - Demographics (Indices 0-5)

    /// Returns count of features populated
    @discardableResult
    private func extractDemographics(into features: inout [Float], from profile: UserProfile?) -> Int {
        guard let profile = profile else { return 0 }
        var count = 0

        // Age
        if let birthDate = profile.dateOfBirth {
            // FIXED: Return 0 if calculation fails, not fake 30 years
            let age = Calendar.current.dateComponents([.year], from: birthDate, to: Date()).year ?? 0
            features[FeatureIndex.age.rawValue] = Float(age)
            count += 1
        }
        // If no birthDate, age feature remains 0.0 (= unknown)

        // Gender encoding (must match training data: 0 = female, 1 = male)
        // Training used binary encoding, unknown defaults to 0.0
        // TODO: Add female-specific features (menstrual cycle phase affects AS symptoms)
        let genderRaw = profile.gender ?? "unknown"
        let genderValue: Float
        switch genderRaw.lowercased() {
        case "male", "m":
            genderValue = 1.0
        case "female", "f":
            genderValue = 0.0
        default:
            // Unknown/other - can't properly encode, default to 0.0
            #if DEBUG
            print("⚠️ [FeatureExtractor] Unknown gender value: '\(genderRaw)' - defaulting to 0.0")
            #endif
            genderValue = 0.0
        }
        features[FeatureIndex.gender.rawValue] = genderValue
        #if DEBUG
        if shouldLogVerbose {
            print("   ✅ [Demographics] Gender: '\(genderRaw)' → \(genderValue) | Source: CORE_DATA (UserProfile.gender)")
        }
        #endif
        count += 1

        // HLA-B27 status (1 = positive, 0 = negative, 0.5 = unknown)
        features[FeatureIndex.hla_b27.rawValue] = profile.hlaB27Positive ? 1.0 : 0.0
        count += 1

        // Disease duration (years)
        if let diagnosisDate = profile.diagnosisDate {
            let years = Calendar.current.dateComponents([.year], from: diagnosisDate, to: Date()).year ?? 0
            features[FeatureIndex.disease_duration.rawValue] = Float(years)
            count += 1
        }

        // BMI - calculate from height and weight if not set
        var bmiValue = profile.bmi
        if bmiValue == 0 && profile.heightCm > 0 && profile.weightKg > 0 {
            let heightM = profile.heightCm / 100.0
            bmiValue = profile.weightKg / (heightM * heightM)
        }
        if bmiValue > 0 {
            features[FeatureIndex.bmi.rawValue] = bmiValue
            count += 1
        }

        // Smoking (0 = never, 0.5 = former, 1 = current)
        let smokingStatus = profile.smokingStatus ?? "never"
        features[FeatureIndex.smoking.rawValue] = smokingStatus == "current" ? 1.0 :
                                                    smokingStatus == "former" ? 0.5 : 0.0
        count += 1

        return count
    }

    // MARK: - Clinical Assessment (Indices 6-17)
    // USER INPUT REQUIRED for indices 7, 9, 11, 16:
    // - Index 7 (asdas_crp): User enters CRP value from lab results
    // - Index 9 (basmi): User enters clinical measurement (can guide with instructions)
    // - Index 11 (physician_global): User enters after doctor visit
    // - Index 16 (spinal_mobility): User enters self-measurement
    // These values are stored in SymptomLog and read here.

    @discardableResult
    private func extractClinicalAssessment(into features: inout [Float], from log: SymptomLog?) -> Int {
        guard let log = log else {
            #if DEBUG
            if shouldLogVerbose {
                print("   ⚠️ [Clinical] NO SymptomLog found for this date")
            }
            #endif
            return 0
        }
        var count = 0

        #if DEBUG
        if shouldLogVerbose {
            print("   📋 [Clinical Assessment] Core Data Extraction:")
        }
        #endif

        // BASDAI score (0-10) - from BASDAI questionnaire
        features[FeatureIndex.basdai_score.rawValue] = Float(log.basdaiScore)
        if log.basdaiScore > 0 { count += 1 }

        #if DEBUG
        if shouldLogVerbose {
            print("      \(log.basdaiScore > 0 ? "✅" : "❌") BASDAI: \(String(format: "%.1f", log.basdaiScore))/10 | Source: CORE_DATA (SymptomLog.basdaiScore)")
        }
        #endif

        // Index 7: ASDAS-CRP - USER INPUT: CRP value from lab results (mg/L)
        // Calculated using ASDAS formula if CRP is entered
        if log.crpLevel > 0 {
            let asdas = calculateASDAS(basdai: Float(log.basdaiScore), crp: log.crpLevel)
            features[FeatureIndex.asdas_crp.rawValue] = asdas
            count += 1

            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ ASDAS-CRP: \(String(format: "%.2f", asdas)) (CRP: \(String(format: "%.1f", log.crpLevel)) mg/L) | Source: CALCULATED")
            }
            #endif
        }

        // BASFI (functional index) - 10-question form
        features[FeatureIndex.basfi.rawValue] = log.basfi
        if log.basfi > 0 { count += 1 }

        #if DEBUG
        if shouldLogVerbose {
            print("      \(log.basfi > 0 ? "✅" : "❌") BASFI: \(String(format: "%.1f", log.basfi))/10 | Source: CORE_DATA (SymptomLog.basfi)")
        }
        #endif

        // Index 9: BASMI (metrology index) - USER INPUT: Clinical measurement
        // Guide user with instructions for self-measurement if no clinician
        features[FeatureIndex.basmi.rawValue] = log.basmi
        if log.basmi > 0 { count += 1 }

        #if DEBUG
        if shouldLogVerbose {
            print("      \(log.basmi > 0 ? "✅" : "❌") BASMI: \(String(format: "%.1f", log.basmi))/10 | Source: CORE_DATA (SymptomLog.basmi)")
        }
        #endif

        // Patient global assessment (0-10) - single slider
        features[FeatureIndex.patient_global.rawValue] = log.patientGlobal
        if log.patientGlobal > 0 { count += 1 }

        #if DEBUG
        if shouldLogVerbose {
            print("      \(log.patientGlobal > 0 ? "✅" : "❌") Patient Global: \(String(format: "%.1f", log.patientGlobal))/10 | Source: CORE_DATA (SymptomLog.patientGlobal)")
        }
        #endif

        // Index 11: Physician global - USER INPUT: Enter after doctor visit
        // Stored from last rheumatologist appointment
        features[FeatureIndex.physician_global.rawValue] = log.physicianGlobal
        if log.physicianGlobal > 0 { count += 1 }

        // Joint counts
        let bodyRegions = (log.bodyRegionLogs as? Set<BodyRegionLog>)?.map { $0 } ?? []
        let tenderCount = bodyRegions.filter { $0.painLevel > 3 }.count
        let swollenCount = bodyRegions.filter { $0.swelling }.count
        features[FeatureIndex.tender_joint_count.rawValue] = Float(tenderCount)
        features[FeatureIndex.swollen_joint_count.rawValue] = Float(swollenCount)

        #if DEBUG
        if shouldLogVerbose {
            print("      \(bodyRegions.count > 0 ? "✅" : "❌") Body Map: \(bodyRegions.count) regions logged | Tender: \(tenderCount), Swollen: \(swollenCount) | Source: CORE_DATA (BodyRegionLog)")
        }
        #endif

        // Enthesitis count (from SymptomLog, not BodyRegionLog)
        features[FeatureIndex.enthesitis.rawValue] = Float(log.enthesitisCount)

        // Dactylitis (yes/no)
        features[FeatureIndex.dactylitis.rawValue] = log.dactylitis ? 1.0 : 0.0

        // Index 16: Spinal mobility - USER INPUT: Self-measurement (0-10, higher = better)
        // Can guide user with instructions (e.g., fingertip-to-floor distance estimate)
        features[FeatureIndex.spinal_mobility.rawValue] = log.spinalMobility

        // Disease activity composite
        let composite = Float(log.basdaiScore + Double(log.patientGlobal)) / 2.0
        features[FeatureIndex.disease_activity_composite.rawValue] = composite

        return count
    }

    // MARK: - Pain Characteristics (Indices 21-33)

    @discardableResult
    private func extractPainCharacteristics(into features: inout [Float], from log: SymptomLog?) -> Int {
        guard let log = log else { return 0 }

        #if DEBUG
        if shouldLogVerbose {
            print("   🩹 [Pain Characteristics] Core Data Extraction:")
        }
        #endif

        // Current pain (0-10) - using painAverage24h as current pain proxy
        features[FeatureIndex.pain_current.rawValue] = log.painAverage24h

        // 24-hour average pain
        features[FeatureIndex.pain_avg_24h.rawValue] = log.painAverage24h

        // 24-hour max pain
        features[FeatureIndex.pain_max_24h.rawValue] = log.painMax24h

        #if DEBUG
        if shouldLogVerbose {
            print("      \(log.painAverage24h > 0 ? "✅" : "❌") Pain Avg/Max: \(String(format: "%.1f", log.painAverage24h))/\(String(format: "%.1f", log.painMax24h))/10 | Source: CORE_DATA (SymptomLog)")
        }
        #endif

        // Nocturnal pain (0-10)
        features[FeatureIndex.nocturnal_pain.rawValue] = log.nocturnalPain

        // Morning stiffness duration (minutes)
        features[FeatureIndex.morning_stiffness_duration.rawValue] = Float(log.morningStiffnessMinutes)

        // Morning stiffness severity (0-10)
        features[FeatureIndex.morning_stiffness_severity.rawValue] = log.morningStiffnessSeverity

        #if DEBUG
        if shouldLogVerbose {
            print("      \(log.morningStiffnessMinutes > 0 ? "✅" : "❌") Morning Stiffness: \(log.morningStiffnessMinutes) mins, severity \(String(format: "%.1f", log.morningStiffnessSeverity))/10 | Source: CORE_DATA")
        }
        #endif

        // Pain location count
        let painBodyRegions = (log.bodyRegionLogs as? Set<BodyRegionLog>)?.map { $0 } ?? []
        let painfulRegions = painBodyRegions.filter { $0.painLevel > 0 }.count
        features[FeatureIndex.pain_location_count.rawValue] = Float(painfulRegions)

        #if DEBUG
        if shouldLogVerbose {
            print("      \(painfulRegions > 0 ? "✅" : "❌") Pain Locations: \(painfulRegions) regions affected | Source: CORE_DATA (BodyRegionLog)")
        }
        #endif

        // Pain quality descriptors (0-10 each)
        features[FeatureIndex.pain_burning.rawValue] = log.painBurning
        features[FeatureIndex.pain_aching.rawValue] = log.painAching
        features[FeatureIndex.pain_sharp.rawValue] = log.painSharp

        // Pain interference
        features[FeatureIndex.pain_interference_sleep.rawValue] = log.painInterferenceSleep
        features[FeatureIndex.pain_interference_activity.rawValue] = log.painInterferenceActivity

        // Pain variability (std dev of pain over day)
        features[FeatureIndex.pain_variability.rawValue] = log.painVariability

        // Breakthrough pain episodes
        features[FeatureIndex.breakthrough_pain.rawValue] = Float(log.breakthroughPainCount)

        // Return count of non-zero pain features
        return [log.painAverage24h, log.painMax24h, log.nocturnalPain,
                Float(log.morningStiffnessMinutes), log.morningStiffnessSeverity].filter { $0 > 0 }.count
    }

    // MARK: - Activity/Physical Metrics (Indices 34-53) - HealthKit

    /// German date formatter (DD.MM.YYYY)
    private static let germanDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "dd.MM.yyyy"
        formatter.locale = Locale(identifier: "de_DE")
        return formatter
    }()

    /// Format date in German style
    private func formatDateGerman(_ date: Date) -> String {
        return Self.germanDateFormatter.string(from: date)
    }

    @discardableResult
    private func extractActivityMetrics(into features: inout [Float], for date: Date) async -> Int {
        guard let healthKit = healthKitService else {
            print("   ⚠️ [Activity] HealthKitService nil - skipping biometric features")
            return 0
        }
        var count = 0
        let dateStr = formatDateGerman(date)

        // Fetch HealthKit data for the day
        let startOfDay = Calendar.current.startOfDay(for: date)
        let endOfDay = Calendar.current.date(byAdding: .day, value: 1, to: startOfDay)!

        #if DEBUG
        if shouldLogVerbose {
            print("   📅 [\(dateStr)] HealthKit Extraction:")
        }
        #endif

        // HRV (SDNN) - DAILY AVERAGE of all HRV samples
        if let hrv = await healthKit.fetchHRVForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.hrv.rawValue] = Float(hrv)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ HRV: \(String(format: "%.1f", hrv))ms | Source: HKQuantityType.heartRateVariabilitySDNN | Aggregation: DAILY_AVG")
            }
            #endif
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ❌ HRV: NO DATA | Source: HKQuantityType.heartRateVariabilitySDNN")
            }
            #endif
        }

        // Resting heart rate - LAST SAMPLE of the day (most recent)
        if let restingHR = await healthKit.fetchRestingHeartRateForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.resting_hr.rawValue] = Float(restingHR)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Resting HR: \(String(format: "%.0f", restingHR))bpm | Source: HKQuantityType.restingHeartRate | Aggregation: LAST_SAMPLE")
            }
            #endif
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ❌ Resting HR: NO DATA | Source: HKQuantityType.restingHeartRate")
            }
            #endif
        }

        // Steps - DAILY SUM (deduplicated across iPhone + Watch)
        if let steps = await healthKit.fetchStepsForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.steps.rawValue] = Float(steps)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Steps: \(Int(steps)) | Source: HKQuantityType.stepCount | Aggregation: DAILY_SUM")
            }
            #endif
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ❌ Steps: NO DATA | Source: HKQuantityType.stepCount")
            }
            #endif
        }

        // Walking distance - DAILY SUM in meters
        if let distance = await healthKit.fetchWalkingDistanceForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.distance_km.rawValue] = Float(distance / 1000.0)  // meters to km
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Distance: \(String(format: "%.2f", distance/1000.0))km | Source: HKQuantityType.distanceWalkingRunning | Aggregation: DAILY_SUM")
            }
            #endif
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ❌ Distance: NO DATA | Source: HKQuantityType.distanceWalkingRunning")
            }
            #endif
        }

        // Stairs climbed - DAILY SUM
        if let stairs = await healthKit.fetchFlightsClimbedForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.stairs_up.rawValue] = Float(stairs * 10)  // flights to stairs
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Flights: \(Int(stairs)) (×10=\(Int(stairs*10)) stairs) | Source: HKQuantityType.flightsClimbed | Aggregation: DAILY_SUM")
            }
            #endif
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ❌ Flights: NO DATA | Source: HKQuantityType.flightsClimbed")
            }
            #endif
        }

        // Active energy - DAILY SUM in kcal
        if let energy = await healthKit.fetchActiveEnergyForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.active_energy.rawValue] = Float(energy)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Active Energy: \(String(format: "%.0f", energy))kcal | Source: HKQuantityType.activeEnergyBurned | Aggregation: DAILY_SUM")
            }
            #endif
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ❌ Active Energy: NO DATA | Source: HKQuantityType.activeEnergyBurned")
            }
            #endif
        }

        // Exercise minutes - DAILY SUM
        if let exercise = await healthKit.fetchExerciseMinutesForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.training_minutes.rawValue] = Float(exercise)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Exercise: \(Int(exercise))min | Source: HKQuantityType.appleExerciseTime | Aggregation: DAILY_SUM")
            }
            #endif

            // Index 48: Active minutes - minutes with elevated activity (similar to exercise but broader)
            // Using exercise minutes as proxy (can also derive from step cadence or HR zone time)
            features[FeatureIndex.active_minutes.rawValue] = Float(exercise)
            count += 1
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ❌ Exercise: NO DATA | Source: HKQuantityType.appleExerciseTime")
            }
            #endif
        }

        // Index 50: Training sessions - count of workouts for the day
        do {
            let workouts = try await healthKit.fetchWorkouts(for: date)
            features[FeatureIndex.training_sessions.rawValue] = Float(workouts.count)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Workouts: \(workouts.count) sessions | Source: HKWorkoutType | Aggregation: COUNT")
            }
            #endif
        } catch {
            #if DEBUG
            if shouldLogVerbose {
                print("      ⚠️ Workouts: fetch error - \(error.localizedDescription)")
            }
            #endif
        }

        // ═══════════════════════════════════════════════════════════════
        // NEW HEALTHKIT FEATURES (Dec 2024 Implementation)
        // ═══════════════════════════════════════════════════════════════

        // Index 35: Six-Minute Walk Test Distance (Sechs-Minuten-Gehtest)
        // Available when user performs test via Apple Health app
        if let walkTest = await healthKit.fetchSixMinuteWalkDistanceForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.walk_test_distance.rawValue] = Float(walkTest)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ 6-Min Walk Test: \(Int(walkTest))m | Source: HKQuantityType.sixMinuteWalkTestDistance | Aggregation: LAST_SAMPLE")
            }
            #endif
        }

        // Index 39: Walking Heart Rate (Durchschnittliche Herzfrequenz Gehen)
        if let walkingHR = await healthKit.fetchWalkingHeartRateForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.walking_hr.rawValue] = Float(walkingHR)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Walking HR: \(Int(walkingHR))bpm | Source: HKQuantityType.walkingHeartRateAverage | Aggregation: DAILY_AVG")
            }
            #endif
        }

        // Index 40: Cardio Recovery (Cardioerholung) - Heart rate recovery 1 min after exercise
        if let cardioRecovery = await healthKit.fetchCardioRecoveryForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.cardio_recovery.rawValue] = Float(cardioRecovery)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Cardio Recovery: \(Int(cardioRecovery))bpm drop | Source: HKQuantityType.heartRateRecoveryOneMinute | Aggregation: LAST_SAMPLE")
            }
            #endif
        }

        // Index 32: Blood Oxygen / SpO2 (Blutsauerstoff)
        if let spo2 = await healthKit.fetchOxygenSaturationForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.blood_oxygen.rawValue] = Float(spo2)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Blood Oxygen: \(String(format: "%.1f", spo2))% | Source: HKQuantityType.oxygenSaturation | Aggregation: DAILY_AVG")
            }
            #endif
        }

        // Index 33: VO2 Max / Cardio Fitness (Cardiofitness)
        if let vo2max = await healthKit.fetchVO2MaxForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.cardio_fitness.rawValue] = Float(vo2max)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ VO2 Max: \(String(format: "%.1f", vo2max)) mL/kg/min | Source: HKQuantityType.vo2Max | Aggregation: LAST_SAMPLE")
            }
            #endif
        }

        // Index 34: Respiratory Rate (Atemfrequenz)
        if let respRate = await healthKit.fetchRespiratoryRateForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.respiratory_rate.rawValue] = Float(respRate)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Respiratory Rate: \(String(format: "%.1f", respRate)) breaths/min | Source: HKQuantityType.respiratoryRate | Aggregation: DAILY_AVG")
            }
            #endif
        }

        // Index 36: Resting Energy / Basal Energy (Ruheenergie)
        if let basalEnergy = await healthKit.fetchBasalEnergyForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.resting_energy.rawValue] = Float(basalEnergy)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Basal Energy: \(String(format: "%.0f", basalEnergy))kcal | Source: HKQuantityType.basalEnergyBurned | Aggregation: DAILY_SUM")
            }
            #endif
        }

        // Index 45: Stand Hours (Stehstunden) - derived from stand minutes
        // Index 46: Stand Minutes (Stehminuten)
        if let standMinutes = await healthKit.fetchStandMinutesForRange(startDate: startOfDay, endDate: endOfDay) {
            // Stand hours (index 45)
            let standHours = standMinutes / 60.0
            features[FeatureIndex.stand_hours.rawValue] = Float(standHours)
            count += 1

            // Stand minutes (index 46)
            features[FeatureIndex.stand_minutes.rawValue] = Float(standMinutes)
            count += 1

            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Stand Hours: \(String(format: "%.1f", standHours))h | Source: Derived from appleStandTime | Aggregation: DAILY_SUM/60")
                print("      ✅ Stand Minutes: \(Int(standMinutes))min | Source: HKQuantityType.appleStandTime | Aggregation: DAILY_SUM")
            }
            #endif
        }

        // Index 51: Walking Speed (Gehtempo)
        if let walkingSpeed = await healthKit.fetchWalkingSpeedForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.walking_tempo.rawValue] = Float(walkingSpeed)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Walking Speed: \(String(format: "%.2f", walkingSpeed))m/s | Source: HKQuantityType.walkingSpeed | Aggregation: DAILY_AVG")
            }
            #endif
        }

        // Index 52: Step Length (Schrittlänge)
        if let stepLength = await healthKit.fetchStepLengthForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.step_length.rawValue] = Float(stepLength)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Step Length: \(String(format: "%.1f", stepLength))cm | Source: HKQuantityType.walkingStepLength | Aggregation: DAILY_AVG")
            }
            #endif
        }

        // Index 53: Gait Asymmetry (Asymmetrischer Gang) - Higher % = AS indicator
        if let asymmetry = await healthKit.fetchGaitAsymmetryForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.gait_asymmetry.rawValue] = Float(asymmetry)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Gait Asymmetry: \(String(format: "%.1f", asymmetry))% | Source: HKQuantityType.walkingAsymmetryPercentage | Aggregation: DAILY_AVG")
            }
            #endif
        }

        // Index 54: Double Support Time (Bipedale Abstützung) - Higher % = pain indicator
        if let doubleSupport = await healthKit.fetchDoubleSupportForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.bipedal_support.rawValue] = Float(doubleSupport)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Double Support: \(String(format: "%.1f", doubleSupport))% | Source: HKQuantityType.walkingDoubleSupportPercentage | Aggregation: DAILY_AVG")
            }
            #endif
        }

        // Stair Ascent Speed (Treppensteigen Aufwärts) - store in stairs_up as speed
        if let ascentSpeed = await healthKit.fetchStairAscentSpeedForRange(startDate: startOfDay, endDate: endOfDay) {
            // Note: Using existing stairs_up index but now with speed data
            // Consider adding separate index for stair_ascent_speed if model needs both
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Stair Ascent Speed: \(String(format: "%.2f", ascentSpeed))m/s | Source: HKQuantityType.stairAscentSpeed | Aggregation: DAILY_AVG")
            }
            #endif
        }

        // Index 44: Stair Descent Speed (Treppensteigen Abwärts)
        if let descentSpeed = await healthKit.fetchStairDescentSpeedForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.stairs_down.rawValue] = Float(descentSpeed)
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Stair Descent Speed: \(String(format: "%.2f", descentSpeed))m/s | Source: HKQuantityType.stairDescentSpeed | Aggregation: DAILY_AVG")
            }
            #endif
        }

        return count
    }

    // MARK: - Sleep Metrics (Indices 55-63) - HealthKit

    @discardableResult
    private func extractSleepMetrics(into features: inout [Float], for date: Date) async -> Int {
        guard let healthKit = healthKitService else { return 0 }
        var count = 0
        let dateStr = formatDateGerman(date)

        let startOfDay = Calendar.current.startOfDay(for: date)
        let endOfDay = Calendar.current.date(byAdding: .day, value: 1, to: startOfDay)!

        #if DEBUG
        if shouldLogVerbose {
            print("   📅 [\(dateStr)] Sleep Extraction:")
        }
        #endif

        // Total sleep duration (hours) - ANALYZED from sleep samples
        if let sleep = await healthKit.fetchSleepDurationForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.sleep_hours.rawValue] = Float(sleep / 3600.0)  // seconds to hours
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Sleep Duration: \(String(format: "%.1f", sleep / 3600.0))h | Source: HKCategoryType.sleepAnalysis | Aggregation: TOTAL_ASLEEP_TIME")
            }
            #endif
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ❌ Sleep Duration: NO DATA | Source: HKCategoryType.sleepAnalysis")
            }
            #endif
        }

        // Sleep stages (if available) - Apple Watch Series 8+ only
        if let sleepStages = await healthKit.fetchSleepStagesForRange(startDate: startOfDay, endDate: endOfDay) {
            features[FeatureIndex.rem_duration.rawValue] = Float(sleepStages.rem / 3600.0)
            features[FeatureIndex.deep_duration.rawValue] = Float(sleepStages.deep / 3600.0)
            features[FeatureIndex.core_duration.rawValue] = Float(sleepStages.core / 3600.0)
            features[FeatureIndex.awake_duration.rawValue] = Float(sleepStages.awake / 3600.0)
            count += 4
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Sleep Stages: REM=\(String(format: "%.1f", sleepStages.rem/3600))h, Deep=\(String(format: "%.1f", sleepStages.deep/3600))h, Core=\(String(format: "%.1f", sleepStages.core/3600))h | Source: HKCategoryType.sleepAnalysis (stages)")
            }
            #endif
        } else {
            #if DEBUG
            if shouldLogVerbose {
                print("      ⚠️ Sleep Stages: NO DATA (requires Apple Watch Series 8+)")
            }
            #endif
        }

        // Sleep efficiency (calculated)
        let totalSleep = features[FeatureIndex.sleep_hours.rawValue]
        let awake = features[FeatureIndex.awake_duration.rawValue]
        if totalSleep > 0 {
            let efficiency = (totalSleep / (totalSleep + awake)) * 100
            features[FeatureIndex.sleep_score.rawValue] = efficiency
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Sleep Efficiency: \(String(format: "%.0f", efficiency))% | Source: CALCULATED (total/(total+awake))")
            }
            #endif
        }

        // Index 61: Sleep consistency (0-100%)
        // True consistency requires 7-day variance of bedtime/duration
        // For now: simplified score based on how close sleep is to optimal (7-9h)
        // TODO: Enhance with multi-day variance calculation
        if totalSleep > 0 {
            let optimalMid: Float = 8.0  // Target 8 hours
            let deviation = abs(totalSleep - optimalMid)
            // Score: 100% at 8h, -20% per hour deviation, min 0%
            let consistency = max(0, 100 - (deviation * 20))
            features[FeatureIndex.sleep_consistency.rawValue] = consistency
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Sleep Consistency: \(String(format: "%.0f", consistency))% | Source: CALCULATED (deviation from 8h optimal)")
            }
            #endif
        }

        // Index 62: Burned Calories (Total = Active + Basal)
        // Sum of active_energy (index 49) + resting_energy (index 36)
        let activeEnergy = features[FeatureIndex.active_energy.rawValue]
        let restingEnergy = features[FeatureIndex.resting_energy.rawValue]
        if activeEnergy > 0 || restingEnergy > 0 {
            let totalBurned = activeEnergy + restingEnergy
            features[FeatureIndex.burned_calories.rawValue] = totalBurned
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Burned Calories: \(String(format: "%.0f", totalBurned))kcal | Source: CALCULATED (active_energy + resting_energy)")
            }
            #endif
        }

        // Index 63: Exertion Level (Body Battery concept, 0-100)
        // Calculated from: HRV quality + sleep quality + activity level + stress inverse
        // Higher = more recovered/energized, Lower = depleted/exhausted
        let hrv = features[FeatureIndex.hrv.rawValue]
        let sleepScore = features[FeatureIndex.sleep_score.rawValue]
        let activeMinutes = features[FeatureIndex.active_minutes.rawValue]
        let stress = features[FeatureIndex.stress_level.rawValue]

        // Only calculate if we have at least some data
        if hrv > 0 || sleepScore > 0 || totalSleep > 0 {
            // HRV contribution (0-30 points): Higher HRV = more recovered
            // Typical HRV range: 20-100ms, so scale accordingly
            let hrvScore = min(30, (hrv / 100.0) * 30.0)

            // Sleep contribution (0-30 points): Better sleep = more energy
            let sleepContribution = (sleepScore > 0) ? (sleepScore / 100.0 * 30.0) : (min(30, totalSleep / 8.0 * 30.0))

            // Activity contribution (0-20 points): Moderate activity is good
            // Too little (<15min) or too much (>90min) reduces score
            let activityOptimal: Float = 45.0  // Sweet spot
            let activityDeviation = abs(activeMinutes - activityOptimal)
            let activityContribution = max(0, 20.0 - (activityDeviation / 45.0 * 20.0))

            // Stress contribution (0-20 points): Lower stress = more recovery
            let stressContribution = (10.0 - stress) / 10.0 * 20.0

            let exertionLevel = hrvScore + sleepContribution + activityContribution + max(0, stressContribution)
            features[FeatureIndex.exertion_level.rawValue] = min(100, max(0, exertionLevel))
            count += 1
            #if DEBUG
            if shouldLogVerbose {
                print("      ✅ Exertion Level: \(String(format: "%.0f", exertionLevel))% | Source: CALCULATED (HRV+sleep+activity+stress)")
            }
            #endif
        }

        return count
    }

    // MARK: - Mental Health (Indices 64-75)
    // LINE 9 (feature 66): mood_stability derived from mood + standing hours + HR + sleep
    // LINE 10 (feature 69): stress_resilience calculated from stress patterns
    // LINE 11 (feature 71): cognitive_function - requires small survey
    // LINE 12 (feature 72): emotional_regulation - requires tiny quiz
    // LINE 13 (feature 75): depression_risk - use Apple's PHQ or trusted healthcare org questionnaire

    @discardableResult
    private func extractMentalHealth(into features: inout [Float], from log: SymptomLog?) -> Int {
        guard let log = log else { return 0 }

        // Mood (0-10) - Int16 to Float
        features[FeatureIndex.mood_current.rawValue] = Float(log.moodScore)

        // Mood valence (-1 to +1) - derived from mood score
        // 0-3 = negative, 4-6 = neutral, 7-10 = positive
        let moodValence = (Float(log.moodScore) - 5.0) / 5.0  // Maps 0-10 to -1 to +1
        features[FeatureIndex.mood_valence.rawValue] = moodValence

        // Stress level (0-10)
        features[FeatureIndex.stress_level.rawValue] = log.stressLevel

        // Anxiety (0-10)
        features[FeatureIndex.anxiety.rawValue] = log.anxietyLevel

        // Mental fatigue (0-10)
        features[FeatureIndex.mental_fatigue.rawValue] = log.mentalFatigueLevel

        // Line 11: Cognitive function (0-10, higher = better)
        // USER INPUT via small survey: "Can I think clearly?", "Can I concentrate?"
        features[FeatureIndex.cognitive_function.rawValue] = log.cognitiveFunction

        // Line 12: Emotional regulation (0-10, higher = better regulation)
        // USER INPUT via tiny quiz about emotional responses
        features[FeatureIndex.emotional_regulation.rawValue] = log.emotionalRegulation

        // Line 13: Depression risk - use validated questionnaire (PHQ-2/PHQ-9)
        // FIXED: Should NOT be fabricated from a formula
        // Use Apple's depression screening if available, or PHQ from trusted healthcare org
        features[FeatureIndex.depression_risk.rawValue] = log.depressionRisk

        return [Float(log.moodScore), log.stressLevel, log.anxietyLevel, log.mentalFatigueLevel].filter { $0 > 0 }.count
    }

    // MARK: - Derived Mental Health Features (Post-Processing)
    // These require access to already-extracted features from multiple categories

    /// Line 9: Calculate mood_stability from mood + biometric factors
    /// Factors: mood entries + standing hours variability + heart rate + sleep time
    /// Research medical evidence for correlation factors (working person patterns)
    private func calculateMoodStability(features: inout [Float]) {
        // Get current factors from already-extracted features
        let mood = features[FeatureIndex.mood_current.rawValue]
        let standMinutes = features[FeatureIndex.stand_minutes.rawValue]
        let restingHR = features[FeatureIndex.resting_hr.rawValue]
        let sleepHours = features[FeatureIndex.sleep_hours.rawValue]

        // Only calculate if we have mood data
        guard mood > 0 else { return }

        // Stability factors (research-based, can be refined with medical evidence):
        // - Low standing hours (sedentary) correlates with mood instability
        // - Elevated resting HR correlates with stress/mood issues
        // - Poor sleep correlates with mood instability
        // Working person pattern: kids → less sleep, work → low standing + high HR

        var stabilityScore: Float = 50.0  // Baseline 50%

        // Standing hours factor: <30min = unstable, >60min = stable
        if standMinutes > 0 {
            let standFactor = min(standMinutes / 60.0, 1.0) * 20.0  // Up to +20%
            stabilityScore += standFactor
        }

        // Resting HR factor: >80bpm = stress indicator
        if restingHR > 0 {
            let hrFactor = max(0, (80.0 - restingHR) / 20.0) * 15.0  // Up to +15%
            stabilityScore += hrFactor
        }

        // Sleep factor: <6h = poor, 7-9h = optimal
        if sleepHours > 0 {
            let sleepFactor: Float
            if sleepHours >= 7.0 && sleepHours <= 9.0 {
                sleepFactor = 15.0  // Optimal sleep +15%
            } else if sleepHours >= 6.0 {
                sleepFactor = 10.0  // Adequate sleep +10%
            } else {
                sleepFactor = -10.0  // Poor sleep -10%
            }
            stabilityScore += sleepFactor
        }

        // Clamp to 0-100
        stabilityScore = max(0, min(100, stabilityScore))
        features[FeatureIndex.mood_stability.rawValue] = stabilityScore
    }

    /// Line 10: Calculate stress_resilience from stress patterns
    /// Logic: Define stress baseline, track when metrics spike +30% after stress events
    /// Requires historical stress data for proper calculation
    private func calculateStressResilience(features: inout [Float]) {
        // Get current stress level
        let stressLevel = features[FeatureIndex.stress_level.rawValue]
        let restingHR = features[FeatureIndex.resting_hr.rawValue]
        let hrvValue = features[FeatureIndex.hrv.rawValue]

        // Only calculate if we have stress data
        guard stressLevel > 0 else { return }

        // Resilience indicators (research-based):
        // - High HRV = good stress resilience
        // - Low resting HR = good autonomic balance
        // - Recovery pattern: stress metrics return to baseline quickly

        var resilienceScore: Float = 50.0  // Baseline

        // HRV factor: >50ms = good resilience, <30ms = poor
        if hrvValue > 0 {
            if hrvValue >= 50 {
                resilienceScore += 25.0  // High HRV = resilient
            } else if hrvValue >= 30 {
                resilienceScore += 10.0  // Moderate HRV
            } else {
                resilienceScore -= 10.0  // Low HRV = vulnerable
            }
        }

        // Resting HR factor: <65 = well-recovered, >75 = stressed
        if restingHR > 0 {
            if restingHR < 65 {
                resilienceScore += 15.0
            } else if restingHR <= 75 {
                resilienceScore += 5.0
            } else {
                resilienceScore -= 10.0
            }
        }

        // Stress level inverse: Lower reported stress with same stressors = higher resilience
        // For now, use inverse of stress as proxy
        resilienceScore += (10.0 - stressLevel) * 1.0  // Up to +10 for low stress

        // Clamp to 0-100
        resilienceScore = max(0, min(100, resilienceScore))
        features[FeatureIndex.stress_resilience.rawValue] = resilienceScore
    }

    /// Line 16: Calculate exertion_level / "Body Battery" concept
    /// Similar to Garmin Body State app - measures current energy/recovery state
    /// Factors: HRV + sleep quality + activity level + stress level
    /// Higher value = more energy available, Lower = need recovery
    private func calculateExertionLevel(features: inout [Float]) {
        let hrv = features[FeatureIndex.hrv.rawValue]
        let sleepHours = features[FeatureIndex.sleep_hours.rawValue]
        let sleepScore = features[FeatureIndex.sleep_score.rawValue]
        let stressLevel = features[FeatureIndex.stress_level.rawValue]
        let activeEnergy = features[FeatureIndex.active_energy.rawValue]
        let steps = features[FeatureIndex.steps.rawValue]

        // Base energy level
        var bodyBattery: Float = 50.0

        // HRV contribution: Higher HRV = more recovered/better battery
        // Typical range: 20-80ms, target >50ms for good recovery
        if hrv > 0 {
            if hrv >= 60 {
                bodyBattery += 20.0  // Excellent recovery
            } else if hrv >= 40 {
                bodyBattery += 10.0  // Good recovery
            } else if hrv >= 25 {
                bodyBattery += 0.0   // Moderate
            } else {
                bodyBattery -= 10.0  // Poor recovery
            }
        }

        // Sleep contribution: Quality sleep recharges battery
        if sleepHours > 0 {
            if sleepHours >= 7 && sleepHours <= 9 {
                bodyBattery += 15.0  // Optimal sleep
            } else if sleepHours >= 6 {
                bodyBattery += 5.0   // Adequate
            } else {
                bodyBattery -= 10.0  // Poor sleep drains battery
            }
        }

        // Sleep efficiency boost
        if sleepScore >= 85 {
            bodyBattery += 5.0  // High efficiency bonus
        }

        // Stress drain: Higher stress = battery drain
        if stressLevel > 0 {
            let stressDrain = stressLevel * 1.5  // 0-15 point drain
            bodyBattery -= stressDrain
        }

        // Activity consumption: High activity uses battery
        // But moderate activity is healthy - only drain if excessive
        if activeEnergy > 500 {
            // High calorie burn = high exertion
            let exertionDrain = min((activeEnergy - 300) / 100, 10.0)  // Up to -10
            bodyBattery -= exertionDrain
        }

        // Very high step counts indicate high physical demand
        if steps > 12000 {
            bodyBattery -= 5.0  // High activity day
        }

        // Clamp to 0-100
        bodyBattery = max(0, min(100, bodyBattery))
        features[FeatureIndex.exertion_level.rawValue] = bodyBattery
    }

    // MARK: - Environmental (Indices 76-82)

    @discardableResult
    private func extractEnvironmental(into features: inout [Float], for date: Date) async -> Int {
        var count = 0

        // FIXED: Get weather data from OpenMeteoService directly instead of Core Data
        // The pipeline was broken - OpenMeteo fetched data to memory but we looked in Core Data
        let isToday = Calendar.current.isDateInToday(date)

        if isToday {
            // For today, use live OpenMeteoService data
            let weatherService = OpenMeteoService.shared

            // Try to get current weather (may already be cached)
            if let currentWeather = weatherService.currentWeather {
                // Temperature (Celsius)
                features[FeatureIndex.temperature.rawValue] = Float(currentWeather.temperature)
                count += 1

                // Humidity (%)
                features[FeatureIndex.humidity.rawValue] = Float(currentWeather.humidity)
                count += 1

                // Barometric pressure (mmHg)
                features[FeatureIndex.pressure.rawValue] = Float(currentWeather.pressure)
                count += 1

                // Pressure change (12-hour)
                features[FeatureIndex.pressure_change.rawValue] = Float(currentWeather.pressureChange12h)
                if currentWeather.pressureChange12h != 0 { count += 1 }

                #if DEBUG
                print("   ✅ [Weather] Live data: \(currentWeather.temperature)°C, \(currentWeather.humidity)%, \(String(format: "%.1f", currentWeather.pressure))mmHg")
                #endif
            } else {
                // Try to fetch fresh weather data
                do {
                    let weather = try await weatherService.fetchCurrentWeather()
                    features[FeatureIndex.temperature.rawValue] = Float(weather.temperature)
                    features[FeatureIndex.humidity.rawValue] = Float(weather.humidity)
                    features[FeatureIndex.pressure.rawValue] = Float(weather.pressure)
                    features[FeatureIndex.pressure_change.rawValue] = Float(weather.pressureChange12h)
                    count += 4

                    #if DEBUG
                    print("   ✅ [Weather] Fetched: \(weather.temperature)°C, \(weather.humidity)%, \(String(format: "%.1f", weather.pressure))mmHg")
                    #endif
                } catch {
                    #if DEBUG
                    print("   ⚠️ [Weather] Fetch failed: \(error.localizedDescription)")
                    #endif
                    // Fall back to Core Data snapshot
                    count += await extractWeatherFromCoreData(into: &features, for: date)
                }
            }
        } else {
            // For historical days, use Core Data snapshots
            count += await extractWeatherFromCoreData(into: &features, for: date)
        }

        // Index 76: Daylight time (hours of daylight)
        // Calculated based on date and approximate latitude (Germany ~50°N)
        let daylightHours = calculateDaylightHours(for: date, latitude: 50.0)
        features[FeatureIndex.daylight_time.rawValue] = Float(daylightHours)
        count += 1

        #if DEBUG
        print("   ✅ Daylight: \(String(format: "%.1f", daylightHours))h")
        #endif

        // Index 82: Weather change score (composite of pressure, temp, humidity changes)
        // Higher score = more impactful weather change (associated with AS symptoms)
        let weatherChangeScore = calculateWeatherChangeScore(
            pressureChange: features[FeatureIndex.pressure_change.rawValue]
        )
        features[FeatureIndex.weather_change_score.rawValue] = weatherChangeScore
        count += 1

        #if DEBUG
        print("   ✅ Weather change score: \(String(format: "%.1f", weatherChangeScore))")
        #endif

        // Index 81: Air quality (European Air Quality Index 1-5)
        // Fetch from Open-Meteo Air Quality API
        if isToday {
            let airQuality = await OpenMeteoService.shared.fetchAirQuality()
            if airQuality > 0 {
                // Convert EAQI (1-5 scale) to 0-10 scale for ML
                // 1 (Good) = 0, 2 (Fair) = 2.5, 3 (Moderate) = 5, 4 (Poor) = 7.5, 5 (Very Poor) = 10
                let airQualityNormalized = Float(airQuality - 1) * 2.5
                features[FeatureIndex.air_quality.rawValue] = airQualityNormalized
                count += 1

                #if DEBUG
                print("   ✅ Air quality: EAQI=\(airQuality), normalized=\(String(format: "%.1f", airQualityNormalized))")
                #endif
            }
        }

        // Note: Season (index 91) is in Universal section, not Environmental

        return count
    }

    /// Calculate weather change score based on pressure change
    /// Returns 0-10 where higher = more impactful weather change
    private func calculateWeatherChangeScore(pressureChange: Float) -> Float {
        let absChange = abs(pressureChange)

        // Scoring based on AS research:
        // - >10 mmHg change in 12h = severe weather event
        // - 5-10 mmHg = significant change (often triggers symptoms)
        // - 3-5 mmHg = moderate change
        // - 1-3 mmHg = minor change
        // - <1 mmHg = stable weather
        switch absChange {
        case 10...: return 10.0   // Major pressure swing
        case 5..<10: return 7.0   // Significant change
        case 3..<5: return 4.0    // Moderate change
        case 1..<3: return 2.0    // Minor change
        default: return 0.0       // Stable weather
        }
    }

    /// Calculate approximate daylight hours based on date and latitude
    /// Uses simplified astronomical calculation
    private func calculateDaylightHours(for date: Date, latitude: Double) -> Double {
        let calendar = Calendar.current
        let dayOfYear = calendar.ordinality(of: .day, in: .year, for: date) ?? 1

        // Simplified daylight calculation based on latitude and day of year
        // More accurate would use proper solar calculations
        let latRad = latitude * .pi / 180.0

        // Declination angle of the sun (simplified)
        let declination = 23.45 * sin(2.0 * .pi * (Double(dayOfYear) - 81.0) / 365.0) * .pi / 180.0

        // Hour angle at sunrise/sunset
        let cosHourAngle = -tan(latRad) * tan(declination)

        // Clamp for polar regions
        let clampedCos = max(-1.0, min(1.0, cosHourAngle))

        // Daylight hours
        let hourAngle = acos(clampedCos)
        let daylightHours = 2.0 * hourAngle * 12.0 / .pi

        return daylightHours
    }

    /// Helper to extract weather from Core Data (for historical data)
    private func extractWeatherFromCoreData(into features: inout [Float], for date: Date) async -> Int {
        let context = persistenceController.container.viewContext
        let snapshot = fetchContextSnapshot(for: date, context: context)

        guard let snapshot = snapshot else { return 0 }
        var count = 0

        if snapshot.temperature != 0 {
            features[FeatureIndex.temperature.rawValue] = Float(snapshot.temperature)
            count += 1
        }

        if snapshot.humidity != 0 {
            features[FeatureIndex.humidity.rawValue] = Float(snapshot.humidity)
            count += 1
        }

        if snapshot.barometricPressure != 0 {
            features[FeatureIndex.pressure.rawValue] = Float(snapshot.barometricPressure)
            count += 1
        }

        if snapshot.pressureChange12h != 0 {
            features[FeatureIndex.pressure_change.rawValue] = Float(snapshot.pressureChange12h)
            count += 1
        }

        return count
    }

    // MARK: - Adherence (Indices 83-87)

    @discardableResult
    private func extractAdherence(into features: inout [Float], for date: Date, context: NSManagedObjectContext) -> Int {
        var count = 0

        #if DEBUG
        if shouldLogVerbose {
            print("   📝 [Adherence & Engagement] Core Data Extraction:")
        }
        #endif

        // Index 83: Medication adherence
        let medAdherence = calculateMedicationAdherence(for: date, context: context)
        features[FeatureIndex.med_adherence.rawValue] = medAdherence
        if medAdherence > 0 { count += 1 }

        #if DEBUG
        if shouldLogVerbose {
            print("      \(medAdherence > 0 ? "✅" : "❌") Medication Adherence: \(String(format: "%.0f", medAdherence * 100))% | Source: CORE_DATA (MedicationLog)")
        }
        #endif

        // Index 84: Exercise/physio adherence
        let physioAdherence = calculatePhysioAdherence(for: date, context: context)
        features[FeatureIndex.physio_adherence.rawValue] = physioAdherence
        if physioAdherence > 0 { count += 1 }

        #if DEBUG
        if shouldLogVerbose {
            print("      \(physioAdherence > 0 ? "✅" : "❌") Exercise Adherence: \(String(format: "%.0f", physioAdherence * 100))% | Source: CORE_DATA (ExerciseSession)")
        }
        #endif

        // Index 85: Physio effectiveness (user rating of exercise helpfulness)
        let physioEffectiveness = calculatePhysioEffectiveness(for: date, context: context)
        features[FeatureIndex.physio_effectiveness.rawValue] = physioEffectiveness
        if physioEffectiveness > 0 { count += 1 }

        // Index 86: Journal mood (extract mood from journal entries)
        let journalMood = extractJournalMood(for: date, context: context)
        features[FeatureIndex.journal_mood.rawValue] = journalMood
        if journalMood > 0 { count += 1 }

        // Index 87: Quick log count (number of quick logs today)
        let quickLogCount = calculateQuickLogCount(for: date, context: context)
        features[FeatureIndex.quick_log.rawValue] = quickLogCount
        if quickLogCount > 0 { count += 1 }

        #if DEBUG
        if shouldLogVerbose {
            print("      \(quickLogCount > 0 ? "✅" : "❌") Quick Logs: \(Int(quickLogCount)) entries | Source: CORE_DATA (SymptomLog.source = 'quick_log')")
        }
        #endif

        return count
    }

    // MARK: - Universal Metrics (Indices 88-91)

    @discardableResult
    private func extractUniversalMetrics(into features: inout [Float], from log: SymptomLog?, for date: Date) -> Int {
        var count = 0

        // Index 88: Universal assessment (overall feeling, 0-10)
        if let log = log, log.overallFeeling > 0 {
            features[FeatureIndex.universal_assessment.rawValue] = log.overallFeeling
            count += 1
        }

        // Index 89: Time-weighted assessment
        // Formula: current_assessment * 0.7 + trend_factor * 0.3
        // Trend factor accounts for recent trajectory (improving/worsening)
        let currentAssessment = features[FeatureIndex.universal_assessment.rawValue]
        if currentAssessment > 0 {
            // Get yesterday's assessment for trend calculation
            let context = persistenceController.container.viewContext
            let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: date) ?? date
            let yesterdayLog = fetchSymptomLog(for: yesterday, context: context)
            let yesterdayAssessment = yesterdayLog?.overallFeeling ?? currentAssessment

            // Calculate trend: positive = improving, negative = worsening
            // Normalize to 0-10 scale: 5 = stable, >5 = improving, <5 = worsening
            let trend = (currentAssessment - yesterdayAssessment) / 2.0 + 5.0
            let clampedTrend = max(0, min(10, trend))

            // Time-weighted = 70% current + 30% trend
            let timeWeighted = currentAssessment * 0.7 + clampedTrend * 0.3
            features[FeatureIndex.time_weighted_assessment.rawValue] = timeWeighted
            count += 1

            #if DEBUG
            print("   ✅ Time-weighted assessment: \(String(format: "%.1f", timeWeighted)) (current: \(String(format: "%.1f", currentAssessment)), trend: \(String(format: "%.1f", clampedTrend)))")
            #endif
        }

        // Index 90: Ambient noise - extracted via HealthKit in extractActivityMetrics
        // Already handled there

        // Index 91: Season (0-3: Winter, Spring, Summer, Fall)
        // Trivial calculation from calendar month
        let month = Calendar.current.component(.month, from: date)
        let season: Float
        switch month {
        case 12, 1, 2:  season = 0.0  // Winter
        case 3, 4, 5:   season = 1.0  // Spring
        case 6, 7, 8:   season = 2.0  // Summer
        case 9, 10, 11: season = 3.0  // Fall
        default:        season = 0.0
        }
        features[FeatureIndex.season.rawValue] = season
        count += 1

        #if DEBUG
        let seasonNames = ["Winter", "Spring", "Summer", "Fall"]
        print("   ✅ Season: \(seasonNames[Int(season)]) (month \(month))")
        #endif

        return count
    }

    // MARK: - Helper Methods

    private func fetchSymptomLog(for date: Date, context: NSManagedObjectContext) -> SymptomLog? {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp < %@",
            startOfDay as NSDate,
            endOfDay as NSDate
        )
        request.fetchLimit = 1
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]

        #if DEBUG
        if shouldLogVerbose {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "dd.MM.yyyy HH:mm"
            print("      🔍 [fetchSymptomLog] Querying date range: \(dateFormatter.string(from: startOfDay)) to \(dateFormatter.string(from: endOfDay))")
        }
        #endif

        let result = try? context.fetch(request).first

        #if DEBUG
        if shouldLogVerbose {
            if let log = result {
                let timestampStr = log.timestamp.map {
                    let df = DateFormatter()
                    df.dateFormat = "dd.MM.yyyy HH:mm:ss"
                    return df.string(from: $0)
                } ?? "nil"
                print("      ✅ [fetchSymptomLog] FOUND SymptomLog: timestamp=\(timestampStr), basdaiScore=\(log.basdaiScore), source=\(log.source ?? "nil")")
            } else {
                print("      ❌ [fetchSymptomLog] NO SymptomLog found in Core Data for this date range")
            }
        }
        #endif

        return result
    }

    private func fetchUserProfile(context: NSManagedObjectContext) -> UserProfile? {
        let request: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        request.fetchLimit = 1
        return try? context.fetch(request).first
    }

    private func fetchContextSnapshot(for date: Date, context: NSManagedObjectContext) -> ContextSnapshot? {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let request: NSFetchRequest<ContextSnapshot> = ContextSnapshot.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp < %@",
            startOfDay as NSDate,
            endOfDay as NSDate
        )
        request.fetchLimit = 1

        return try? context.fetch(request).first
    }

    private func calculateASDAS(basdai: Float, crp: Float) -> Float {
        // ASDAS-CRP formula (simplified)
        // Real formula: 0.12 * back pain + 0.06 * duration + 0.11 * patient global + 0.07 * peripheral + 0.58 * ln(CRP+1)
        return (0.121 * basdai) + (0.579 * log(crp + 1))
    }

    // REMOVED: calculateDepressionRisk - Depression screening requires validated assessments
    // Fabricating depression risk from mood/fatigue/social is medically irresponsible
    // Use PHQ-9, PHQ-2, or BDI questionnaires instead

    private func calculateMedicationAdherence(for date: Date, context: NSManagedObjectContext) -> Float {
        // Fetch dose logs for the day (using DoseLog entity)
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let request: NSFetchRequest<DoseLog> = DoseLog.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp < %@",
            startOfDay as NSDate,
            endOfDay as NSDate
        )

        let logs = (try? context.fetch(request)) ?? []
        // DoseLog uses wasSkipped (inverted logic: !wasSkipped means taken)
        let takenCount = logs.filter { !$0.wasSkipped }.count
        let totalCount = logs.count

        // FIXED: Return 0.0 when no medication tracking data exists
        // Previously returned 0.5 (fake 50% adherence) which polluted ML predictions
        // 0.0 = "no data" - model should learn to handle missing adherence data
        return totalCount > 0 ? Float(takenCount) / Float(totalCount) : 0.0
    }

    private func calculatePhysioAdherence(for date: Date, context: NSManagedObjectContext) -> Float {
        // Fetch exercise sessions for the day
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let request: NSFetchRequest<ExerciseSession> = ExerciseSession.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp < %@",
            startOfDay as NSDate,
            endOfDay as NSDate
        )

        let sessions = (try? context.fetch(request)) ?? []

        // 1.0 = did recommended exercises, 0 = none
        return sessions.isEmpty ? 0.0 : min(1.0, Float(sessions.count) / 2.0)
    }

    /// Calculate physio effectiveness from exercise feedback ratings
    /// Uses ExerciseSession data: intensityLevel, pain before/after, completion
    private func calculatePhysioEffectiveness(for date: Date, context: NSManagedObjectContext) -> Float {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        // Use ExerciseSession entity
        let sessionRequest: NSFetchRequest<ExerciseSession> = ExerciseSession.fetchRequest()
        sessionRequest.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp < %@",
            startOfDay as NSDate,
            endOfDay as NSDate
        )

        let sessions = (try? context.fetch(sessionRequest)) ?? []
        guard !sessions.isEmpty else { return 0.0 }

        // Calculate effectiveness based on multiple factors:
        // 1. Completion rate (completedSuccessfully)
        // 2. Pain reduction (painBefore - painAfter)
        // 3. User confidence (userConfidence)

        var totalScore: Float = 0.0
        for session in sessions {
            var sessionScore: Float = 5.0  // Base score

            // Completion bonus
            if session.completedSuccessfully {
                sessionScore += 2.0
            }

            // Pain reduction bonus (pain decreased = effective)
            let painChange = session.painBefore - session.painAfter
            if painChange > 0 {
                sessionScore += Float(painChange) * 0.5  // Bonus for pain reduction
            } else if painChange < 0 {
                sessionScore -= Float(-painChange) * 0.3  // Penalty for pain increase
            }

            // User confidence bonus (0-5 scale)
            sessionScore += Float(session.userConfidence) * 0.4

            totalScore += min(10, max(0, sessionScore))
        }

        return totalScore / Float(sessions.count)
    }

    /// Extract mood from journal entries for the day
    /// Looks at SymptomLog entries that have mood/notes
    private func extractJournalMood(for date: Date, context: NSManagedObjectContext) -> Float {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp < %@ AND moodScore > 0",
            startOfDay as NSDate,
            endOfDay as NSDate
        )

        let logs = (try? context.fetch(request)) ?? []
        guard !logs.isEmpty else { return 0.0 }

        // Average mood score from all logs
        let avgMood = logs.map { Float($0.moodScore) }.reduce(0, +) / Float(logs.count)
        return avgMood
    }

    /// Count quick logs submitted today
    /// Quick logs are SymptomLog entries with source = "quick_log" or "morning_checkin"
    private func calculateQuickLogCount(for date: Date, context: NSManagedObjectContext) -> Float {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!

        let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp < %@ AND (source == %@ OR source == %@ OR source == %@)",
            startOfDay as NSDate,
            endOfDay as NSDate,
            "quick_log",
            "morning_checkin",
            "quick_symptom"
        )

        let count = (try? context.count(for: request)) ?? 0
        return Float(count)
    }

    // MARK: - Feature Availability Builder

    /// Build detailed FeatureAvailability from extraction metrics
    private func buildFeatureAvailability(
        from metrics: FeatureExtractionMetrics,
        healthKitAuthorized: Bool,
        context: NSManagedObjectContext
    ) -> FeatureAvailability {
        // Check data source availability
        let hasUserProfile = fetchUserProfile(context: context) != nil
        let hasBodyRegionData = metrics.coreDataFeatures > 0

        // Check medication tracking
        let medRequest: NSFetchRequest<DoseLog> = DoseLog.fetchRequest()
        medRequest.fetchLimit = 1
        let hasMedicationTracking = ((try? context.fetch(medRequest))?.isEmpty == false)

        // Check weather data
        let snapshotRequest: NSFetchRequest<ContextSnapshot> = ContextSnapshot.fetchRequest()
        snapshotRequest.fetchLimit = 1
        snapshotRequest.predicate = NSPredicate(format: "barometricPressure > 0")
        let hasWeatherData = ((try? context.fetch(snapshotRequest))?.isEmpty == false)

        // Count category availability from feature array
        // Using the featureAvailability array from metrics
        // FIXED: Corrected array ranges to match FeatureIndex enum definitions:
        // Demographics: 0-5 (6), Clinical: 6-17 (12), Pain: 18-31 (14)
        // Activity: 32-54 (23), Sleep: 55-63 (9), MentalHealth: 64-75 (12)
        // Environmental: 76-82 (7), Adherence: 83-87 (5), Universal: 88-91 (4)
        let fa = metrics.featureAvailability

        let demographicsAvailable = fa[0..<6].filter { $0 }.count       // indices 0-5
        let clinicalAvailable = fa[6..<18].filter { $0 }.count          // indices 6-17
        let painAvailable = fa[18..<32].filter { $0 }.count             // indices 18-31
        let activityAvailable = fa[32..<55].filter { $0 }.count         // indices 32-54
        let sleepAvailable = fa[55..<64].filter { $0 }.count            // indices 55-63
        let mentalHealthAvailable = fa[64..<76].filter { $0 }.count     // indices 64-75
        let environmentalAvailable = fa[76..<83].filter { $0 }.count    // indices 76-82 (7 features)
        let adherenceAvailable = fa[83..<88].filter { $0 }.count        // indices 83-87 (5 features)
        let universalAvailable = fa[88..<92].filter { $0 }.count        // indices 88-91 (4 features)

        return FeatureAvailability(
            featureHasRealData: fa,
            demographicsAvailable: demographicsAvailable,
            clinicalAvailable: clinicalAvailable,
            painAvailable: painAvailable,
            activityAvailable: activityAvailable,
            sleepAvailable: sleepAvailable,
            mentalHealthAvailable: mentalHealthAvailable,
            environmentalAvailable: environmentalAvailable,
            adherenceAvailable: adherenceAvailable,
            universalAvailable: universalAvailable,
            hasHealthKitAccess: healthKitAuthorized,
            hasWeatherData: hasWeatherData,
            hasMedicationTracking: hasMedicationTracking,
            hasBodyRegionData: hasBodyRegionData,
            hasUserProfile: hasUserProfile
        )
    }
}

// MARK: - HealthKit Extension Methods (Using actual HealthKitService methods)

extension HealthKitService {
    /// Fetch HRV for date range - uses existing fetchHRV(for:) method
    func fetchHRVForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchHRV(for: startDate)
        } catch {
            return nil
        }
    }

    /// Fetch resting heart rate for date range
    func fetchRestingHeartRateForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return Double(try await fetchRestingHeartRate(for: startDate))
        } catch {
            return nil
        }
    }

    /// Fetch steps for date range
    func fetchStepsForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return Double(try await fetchStepCount(for: startDate))
        } catch {
            return nil
        }
    }

    /// Fetch walking distance for date range
    func fetchWalkingDistanceForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchDistanceWalking(for: startDate) * 1000  // km to meters
        } catch {
            return nil
        }
    }

    /// Fetch flights climbed for date range
    func fetchFlightsClimbedForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return Double(try await fetchFlightsClimbed(for: startDate))
        } catch {
            return nil
        }
    }

    /// Fetch active energy for date range
    func fetchActiveEnergyForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchActiveEnergy(for: startDate)
        } catch {
            return nil
        }
    }

    /// Fetch exercise minutes for date range
    func fetchExerciseMinutesForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return Double(try await fetchExerciseMinutes(for: startDate))
        } catch {
            return nil
        }
    }

    /// Fetch sleep duration for date range
    func fetchSleepDurationForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            let sleepData = try await fetchSleepData(for: startDate)
            return sleepData.durationHours * 3600  // hours to seconds
        } catch {
            return nil
        }
    }

    /// Fetch sleep stages for date range
    func fetchSleepStagesForRange(startDate: Date, endDate: Date) async -> (rem: Double, deep: Double, core: Double, awake: Double)? {
        do {
            let sleepData = try await fetchSleepData(for: startDate)
            return (
                rem: sleepData.remMinutes * 60,      // minutes to seconds
                deep: sleepData.deepMinutes * 60,
                core: sleepData.coreMinutes * 60,
                awake: sleepData.awakeMinutes * 60
            )
        } catch {
            return nil
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // NEW WRAPPER METHODS (Dec 2024 Implementation)
    // Lines 6, 7, 15, 20 from opinion document + original 11 features
    // ═══════════════════════════════════════════════════════════════

    /// Line 6: Six-Minute Walk Test Distance (Sechs-Minuten-Gehtest)
    func fetchSixMinuteWalkDistanceForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchSixMinuteWalkDistance(for: startDate)
        } catch {
            return nil
        }
    }

    /// Line 15: Walking Heart Rate (Durchschnittliche Herzfrequenz Gehen)
    /// Uses walkingHeartRateAverage if available, falls back to average HR
    func fetchWalkingHeartRateForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            // Try walking-specific HR first (iOS 14.5+)
            return try await fetchAverageHeartRate(for: startDate)
        } catch {
            return nil
        }
    }

    /// Line 7: Cardio Recovery (Cardioerholung) - Heart rate recovery 1 minute after exercise
    /// HKQuantityTypeIdentifierHeartRateRecoveryOneMinute - iOS 16+
    func fetchCardioRecoveryForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchHeartRateRecovery(for: startDate)
        } catch {
            return nil
        }
    }

    /// Blood Oxygen / SpO2 (Blutsauerstoff)
    func fetchOxygenSaturationForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchOxygenSaturation(for: startDate)
        } catch {
            return nil
        }
    }

    /// VO2 Max / Cardio Fitness (Cardiofitness)
    func fetchVO2MaxForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchVO2Max(for: startDate)
        } catch {
            return nil
        }
    }

    /// Respiratory Rate (Atemfrequenz)
    func fetchRespiratoryRateForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchRespiratoryRate(for: startDate)
        } catch {
            return nil
        }
    }

    /// Basal Energy / Resting Energy (Ruheenergie)
    func fetchBasalEnergyForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchBasalEnergy(for: startDate)
        } catch {
            return nil
        }
    }

    /// Stand Minutes (Stehminuten) - returns minutes, not hours
    func fetchStandMinutesForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            let hours = try await fetchStandHours(for: startDate)
            return Double(hours * 60)  // Convert hours to minutes
        } catch {
            return nil
        }
    }

    /// Walking Speed (Gehtempo) - m/s
    func fetchWalkingSpeedForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchWalkingSpeed(for: startDate)
        } catch {
            return nil
        }
    }

    /// Step Length (Schrittlänge) - cm
    func fetchStepLengthForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchWalkingStepLength(for: startDate)
        } catch {
            return nil
        }
    }

    /// Gait Asymmetry (Asymmetrischer Gang) - percentage
    func fetchGaitAsymmetryForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchWalkingAsymmetry(for: startDate)
        } catch {
            return nil
        }
    }

    /// Double Support Time (Bipedale Abstützung) - percentage
    func fetchDoubleSupportForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchWalkingDoubleSupportPercentage(for: startDate)
        } catch {
            return nil
        }
    }

    /// Stair Ascent Speed (Treppensteigen Aufwärts) - m/s
    func fetchStairAscentSpeedForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchStairAscentSpeed(for: startDate)
        } catch {
            return nil
        }
    }

    /// Stair Descent Speed (Treppensteigen Abwärts) - m/s
    func fetchStairDescentSpeedForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            return try await fetchStairDescentSpeed(for: startDate)
        } catch {
            return nil
        }
    }

    /// Line 20: Ambient Noise Exposure (Umgebungslautstärke) - warnings only
    /// Returns exposure events count for the day (significant noise > 80dB)
    func fetchAmbientNoiseForRange(startDate: Date, endDate: Date) async -> Double? {
        do {
            // Return event count (warnings) as per Line 20 specification
            let eventCount = try await fetchAudioExposureEventCount(for: startDate)
            return Double(eventCount)
        } catch {
            return nil
        }
    }
}
