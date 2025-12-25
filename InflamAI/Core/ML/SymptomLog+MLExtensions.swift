//
//  SymptomLog+MLExtensions.swift
//  InflamAI
//
//  Extension to auto-populate ALL 92 ML properties for comprehensive feature extraction
//  Target: 92/92 features with real or intelligently estimated data
//

import Foundation
import CoreData

extension SymptomLog {
    
    /// Populate ALL 92 ML properties from available data sources
    /// Call this after creating a SymptomLog to ensure complete ML feature coverage
    func populateMLProperties(context: NSManagedObjectContext) {
        // 1. Auto-calculate from body regions
        if let regions = bodyRegionLogs?.allObjects as? [BodyRegionLog], !regions.isEmpty {
            calculatePainMetricsComprehensive(from: regions)
            calculateJointCounts(from: regions)
        }
        
        // 2. Calculate mental health composites
        calculateMentalHealthMetricsComplete()
        
        // 3. Estimate activity/fitness metrics
        estimateActivityMetrics()
        
        // 4. Estimate sleep metrics
        estimateSleepMetrics()
        
        // 5. Set environmental context
        if let snapshot = contextSnapshot {
            extractEnvironmentalFeatures(from: snapshot)
        } else {
            estimateEnvironmentalDefaults()
        }
        
        // 6. Calculate clinical assessment scores
        calculateClinicalScores()
        
        // 7. Fill all remaining gaps with intelligent defaults
        applyComprehensiveDefaults()

        // 8. Derived scores are calculated at extraction time in FeatureExtractor
        // No additional derived calculations needed here
    }
    
    // MARK: - Comprehensive Pain Calculations (21 features)

    private func calculatePainMetricsComprehensive(from regions: [BodyRegionLog]) {
        let painLevels = regions.map { Float($0.painLevel) }
        guard !painLevels.isEmpty else { return }

        // Location count - VALID: Direct count from body regions
        painLocationCount = Int16(regions.filter { $0.painLevel > 0 }.count)

        let painfulRegions = painLevels.filter { $0 > 0 }
        guard !painfulRegions.isEmpty else { return }

        // Average pain - VALID: Actual average of reported pain
        let avgPain = painfulRegions.reduce(0, +) / Float(painfulRegions.count)
        if painAverage24h == 0.0 { painAverage24h = avgPain }

        // Max pain - VALID: Actual maximum of reported pain
        if painMax24h == 0.0 { painMax24h = painLevels.max() ?? 0.0 }

        // Pain variability - VALID: Statistical measure of actual data
        let mean = painLevels.reduce(0, +) / Float(painLevels.count)
        let variance = painLevels.map { pow($0 - mean, 2) }.reduce(0, +) / Float(painLevels.count)
        painVariability = sqrt(variance)

        // FIXED: NO MORE PAIN QUALITY ESTIMATIONS
        // Pain quality (burning, aching, sharp) should be ASKED, not assumed
        // REMOVED: painBurning = warmthCount * 2.5 (warmth ≠ burning pain)
        // REMOVED: painAching = achingCount / regionCount * 10 (stiffness ≠ aching)
        // REMOVED: painSharp = (pain > 7 count) (high pain ≠ sharp pain)
        // If these are 0, it means NOT ASKED - not "no burning/aching/sharp pain"

        // FIXED: NO MORE NOCTURNAL PAIN CASCADING
        // REMOVED: nocturnalPain = avgPain * 0.6 + (10-sleepQuality) * 0.4
        // Nocturnal pain should be explicitly asked: "Did you have pain at night?"
        // Poor sleep doesn't necessarily mean night pain

        // FIXED: NO MORE PAIN INTERFERENCE CASCADING
        // REMOVED: painInterferenceSleep = nocturnalPain * 0.8 + avgPain * 0.2
        // REMOVED: painInterferenceActivity = avgPain (direct correlation assumption)
        // Pain interference should be explicitly asked, not calculated
        // This was cascading fake data (derived from already-fake nocturnal pain)

        // FIXED: NO MORE BREAKTHROUGH PAIN ESTIMATION
        // REMOVED: breakthroughPainCount = painVariability * 2
        // Breakthrough pain episodes should be COUNTED by user, not estimated
        // Pain variability ≠ breakthrough episodes
    }
    
    private func calculateJointCounts(from regions: [BodyRegionLog]) {
        // Tender (pain > 3)
        tenderJointCount = Int16(regions.filter { $0.painLevel > 3 }.count)
        
        // Swollen
        swollenJointCount = Int16(regions.filter { $0.swelling }.count)
        
        // Enthesitis (stiffness > 30 min)
        enthesitisCount = Int16(regions.filter { $0.stiffnessDuration > 30 }.count)
        
        // Dactylitis detection (finger/toe swelling)
        // FIXED: Use actual Core Data regionID values (snake_case from BodyRegion.swift)
        let digitalRegions = ["hand_left", "hand_right", "foot_left", "foot_right"]
        dactylitis = regions.contains { region in
            guard let regionID = region.regionID else { return false }
            return digitalRegions.contains(regionID) && region.swelling
        }

        // Peripheral involvement
        // FIXED: Use actual Core Data regionID values (snake_case from BodyRegion.swift)
        let peripheralRegions = ["hand_left", "hand_right", "foot_left", "foot_right",
                                "knee_left", "knee_right", "ankle_left", "ankle_right"]
        peripheralJointInvolvement = regions.contains { region in
            guard let regionID = region.regionID else { return false }
            return peripheralRegions.contains(regionID) && region.painLevel > 0
        }
    }
    
    // MARK: - Complete Mental Health (12 features)

    private func calculateMentalHealthMetricsComplete() {
        // FIXED: NO MORE FABRICATED MENTAL HEALTH SCORES
        // These metrics should come from validated psychological assessments
        // or explicit user self-reporting, not formulas

        // Mental Wellbeing
        // REMOVED: Fake composite from (10-stress + 10-anxiety + mood) / 3
        // Mental wellbeing requires validated assessment (e.g., WHO-5, WEMWBS)
        // If mentalWellbeing == 5.0 (default), leave it - means NOT ASSESSED

        // Depression Risk
        // REMOVED: Fake calculation ((10-mood) + fatigue + (10-social)) / 3
        // Depression screening requires validated tools (PHQ-9, PHQ-2, BDI)
        // FABRICATING A DEPRESSION RISK SCORE IS MEDICALLY DANGEROUS
        // If depressionRisk == 0.0, it means NOT SCREENED (honest)

        // Mood Valence
        // Keep this - it's a simple transformation of actual mood score
        if moodValence == 0.0 && moodScore > 0 {
            moodValence = (Float(moodScore) - 5.0) * 2.0 // Convert 0-10 to -10 to +10
        }

        // Mood Stability
        // REMOVED: Fake estimate (10 - stress - anxiety*0.5)
        // Mood stability requires tracking over time, not instant calculation
        // If moodStability == 5.0 (default), leave it - means NOT TRACKED

        // Emotional Regulation
        // REMOVED: Fake estimate from stress/anxiety/coping formula
        // This is a complex psychological construct, not a simple formula

        // Stress Resilience
        // REMOVED: Fake estimate (10 - stress + coping) / 2
        // Resilience requires validated assessment (Connor-Davidson, BRS)
    }
    
    // MARK: - Activity/Physical Metrics (20 features)

    private func estimateActivityMetrics() {
        // FIXED: NO MORE FAKE HEALTHKIT DATA
        // Previously estimated HRV, HR, steps, gait metrics from symptoms
        // This created fake biometric data that polluted ML predictions

        // HealthKit data MUST come from real HealthKit - do not fabricate
        // If contextSnapshot.hrvValue == 0, it means NO HRV DATA (not "estimated HRV")
        // If contextSnapshot.stepCount == 0, it means NO STEP DATA (not "estimated steps")

        // The ML model should learn to handle missing HealthKit data patterns
        // Users without Apple Watch will have 0s for these features - that's VALID

        // REMOVED: All fake HRV estimates (30-80ms from age+stress)
        // REMOVED: All fake resting HR estimates (65 + stress + fatigue)
        // REMOVED: All fake step estimates (0-5000 from activity factor)
        // REMOVED: All fake gait metrics (walkingTempo, stepLength, gaitAsymmetry)
        // REMOVED: All fake cardio fitness estimates
        // REMOVED: All fake bipedal support estimates

        // Keep only REAL data from HealthKit via contextSnapshot
        // If no real data exists, features remain 0 (which is honest)
    }

    private func estimateActivityFromSymptoms() {
        // FIXED: NO MORE FAKE ACTIVITY DATA
        // Previously fabricated step counts and exercise minutes from symptoms
        // This is circular logic - symptoms shouldn't create fake activity data

        // If user doesn't have HealthKit data, activity features should be 0
        // The model will learn that 0 = "no tracking" pattern
    }
    
    // MARK: - Sleep Metrics (8 features)

    private func estimateSleepMetrics() {
        // FIXED: NO MORE FAKE SLEEP DATA
        // Previously estimated sleep duration from quality (5-9 hours)
        // Previously fabricated REM/Deep/Core/Awake durations with arbitrary percentages
        // This created fake sleep architecture data that doesn't exist

        // Sleep data MUST come from real HealthKit sleep tracking
        // If sleepDurationHours == 0, it means NO SLEEP DATA TRACKED (not "estimated sleep")

        // REMOVED: Fake sleep duration estimation (5 + quality*4 hours)
        // REMOVED: Fake REM duration (20-25% arbitrary)
        // REMOVED: Fake Deep sleep duration (15-20% arbitrary)
        // REMOVED: Fake Core sleep duration (50% hardcoded)
        // REMOVED: Fake Awake duration calculation
        // REMOVED: Fake sleep consistency from stress

        // Keep only REAL sleep data from HealthKit
        // Users without sleep tracking will have 0s - that's honest
    }
    
    // MARK: - Environmental Features (11 features)
    
    private func extractEnvironmentalFeatures(from snapshot: ContextSnapshot) {
        // FIXED: NO MORE FAKE WEATHER DATA
        // Previously filled missing weather with placeholders (20°C, 60%, 1013 mmHg)
        // This broke flare trigger detection since pressure changes were masked

        // Temperature, humidity, pressure - keep as-is from real WeatherKit data
        // If 0, it means no weather data available - DO NOT fabricate
        // The ML model should learn to handle missing weather data

        // REMOVED: Fake pressure change estimation from pain variability
        // This was circular logic that polluted the model

        // Daylight time (calculate from date - this IS valid to compute)
        let month = Calendar.current.component(.month, from: timestamp ?? Date())
        // Note: daylightHours not stored in Core Data, computed at extraction time

        // Season (0-3: winter, spring, summer, fall) - valid to compute from date
        // Note: season not stored in Core Data, computed at extraction time
    }
    
    private func estimateEnvironmentalDefaults() {
        // FIXED: NO MORE FAKE WEATHER DATA
        // Previously created fake ContextSnapshot with placeholder values:
        // - temperature = 20.0°C (fake room temperature)
        // - humidity = 60% (arbitrary middle value)
        // - barometricPressure = 1013.0 mmHg (standard atmosphere - meaningless for predictions)
        // - pressureChange12h = 0.0 (breaks flare trigger detection!)

        // Now: If no ContextSnapshot exists, leave it nil
        // The FeatureExtractor will handle missing weather data appropriately
        // ML model should learn from "missing weather" pattern, not fake "stable weather"

        // DO NOT create fake ContextSnapshot - let it remain nil
    }
    
    // MARK: - Clinical Scores (12 features)

    private func calculateClinicalScores() {
        // FIXED: NO MORE FAKE CLINICAL SCORES
        // These are MEDICAL ASSESSMENTS that require real clinical evaluation

        // BASFI (Bath Ankylosing Spondylitis Functional Index)
        // REMOVED: Fake estimation from (activityLimitation + pain + fatigue) / 3
        // BASFI MUST come from validated 10-question patient questionnaire
        // If basfi == 0.0, it means NOT ASSESSED (not "estimated BASFI")

        // BASMI (Bath Ankylosing Spondylitis Metrology Index)
        // REMOVED: Fake estimation from stiffness
        // BASMI MUST come from actual physical measurements by clinician
        // Includes: tragus-wall, lumbar flexion, cervical rotation, etc.

        // Spinal Mobility
        // REMOVED: Fake estimation (10 - stiffness - pain*0.3)
        // Spinal mobility requires actual measurement

        // Patient Global
        // Keep this one - it's a self-reported metric that CAN be derived
        // from other self-reported symptoms if explicitly asked
        if patientGlobal == 0.0 && overallFeeling > 0 {
            // Only derive if we have real self-reported data
            patientGlobal = (10 - overallFeeling) // Convert feeling (10=good) to global (10=bad)
        }

        // Physician Global
        // REMOVED: Fake estimation from (tender + swollen + CRP + BASDAI) / 4
        // This is a PHYSICIAN'S clinical judgment - CANNOT be fabricated
        // If physicianGlobal == 0.0, it means NO PHYSICIAN VISIT (honest)

        // CRP Level
        // REMOVED: Fake estimation from (swollen * 2 + pain) / 3
        // CRP is a LABORATORY TEST - requires blood draw
        // If crpLevel == 0.0 and crpValue == 0.0, it means NO LAB DATA
        if crpLevel == 0.0 && crpValue > 0 {
            crpLevel = Float(crpValue) // Only use if actual lab value exists
        }

        // ASDAS-CRP - only calculate if we have REAL CRP from lab
        // Formula is valid, but inputs must be real
    }
    
    // MARK: - Comprehensive Defaults (All Remaining Fields)

    private func applyComprehensiveDefaults() {
        // FIXED: NO MORE "INTELLIGENT" DEFAULTS
        // Most of these were fabricating data from formulas
        // If a value is 5.0 (default) or 0.0 (unset), it means NOT TRACKED

        // REMOVED: energyLevel = 10 - fatigue (energy should be asked separately)
        // REMOVED: overallFeeling = (mood + energy + 10-pain) / 3 (should be asked)
        // REMOVED: dayQuality = overallFeeling (should be asked separately)
        // REMOVED: copingAbility = formula (coping is a complex construct)
        // REMOVED: cognitiveFunction = 10 - mentalFatigue - stress (needs assessment)
        // REMOVED: socialEngagement = 7 - pain*0.3 - (10-energy)*0.2 (major assumption!)
        // REMOVED: activityLimitationScore = (pain + fatigue + stiffness) / 3 (needs BASFI)
        // REMOVED: physicalFunctionScore = 10 - activityLimitation (circular)
        // REMOVED: balanceScore from pain/stiffness (balance needs assessment)
        // REMOVED: mentalFatigueLevel = fatigue*0.8 + stress*0.2 (should be separate)

        // Keep ONLY valid derivations:

        // Morning stiffness severity from duration - VALID transformation
        // Duration in minutes → severity on 0-10 scale (clinical standard)
        if morningStiffnessSeverity == 0.0 && morningStiffnessMinutes > 0 {
            // Standard: 0min=0, 30min≈2.5, 60min=5, 120min+=10
            morningStiffnessSeverity = min(10, Float(morningStiffnessMinutes) / 12.0)
        }

        // Time-weighted assessment - only copy if overallFeeling was actually reported
        // Don't create timeWeightedAssessment from a fabricated overallFeeling
    }
    
    // MARK: - Helpers
    
    private func calculateAge() -> Int {
        // Try to get from UserProfile
        let request: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        request.fetchLimit = 1

        guard let profile = try? managedObjectContext?.fetch(request).first,
              let birthDate = profile.dateOfBirth else {
            // FIXED: Return 0 instead of fake 40 years
            // 0 = "age unknown" - model should handle missing age
            return 0
        }

        // FIXED: Return 0 if calculation fails, not fake 40
        let age = Calendar.current.dateComponents([.year], from: birthDate, to: Date()).year ?? 0
        return age
    }
}

// MARK: - ML Validation & Debugging (moved from duplicate extension)

extension SymptomLog {

    /// Validate that essential ML fields are populated
    /// Returns tuple: (isValid, missingFieldsCount)
    func validateMLFields() -> (Bool, Int) {
        var missingCount = 0

        // Check essential fields that should always have values
        let essentialFields: [(value: Float, name: String)] = [
            (overallFeeling, "overallFeeling"),
            (energyLevel, "energyLevel"),
            (Float(moodScore), "moodScore"),
            (Float(fatigueLevel), "fatigueLevel")
        ]

        for field in essentialFields {
            if field.value == 0.0 || (field.value == 5.0 && field.name.contains("Level")) {
                missingCount += 1
            }
        }

        return (missingCount == 0, missingCount)
    }

    #if DEBUG
    func printMLFieldCoverage() {
        var nonZeroCount = 0
        var totalCount = 0

        let fields: [(String, Float)] = [
            ("overallFeeling", overallFeeling),
            ("energyLevel", energyLevel),
            ("stressLevel", stressLevel),
            ("anxietyLevel", anxietyLevel),
            ("painAverage24h", painAverage24h),
            ("patientGlobal", patientGlobal),
            ("basfi", basfi),
            ("basmi", basmi),
            ("mentalWellbeing", mentalWellbeing),
            ("depressionRisk", depressionRisk),
            ("activityLimitationScore", activityLimitationScore),
            ("cognitiveFunction", cognitiveFunction),
            ("socialEngagement", socialEngagement),
            ("copingAbility", copingAbility)
        ]

        for (name, value) in fields {
            totalCount += 1
            if value != 0.0 && !(value == 5.0 && name.contains("Engagement")) {
                nonZeroCount += 1
            }
        }

        print("📊 ML Field Coverage: \(nonZeroCount)/\(totalCount) fields populated")
        print("Coverage: \(Int(Double(nonZeroCount) / Double(totalCount) * 100))%")
    }
    #endif
}
