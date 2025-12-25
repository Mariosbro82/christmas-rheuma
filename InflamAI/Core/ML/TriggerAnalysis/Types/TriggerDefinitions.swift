//
//  TriggerDefinitions.swift
//  InflamAI
//
//  Default trigger definitions for AS symptom tracking
//  Based on clinical literature and patient-reported triggers
//

import Foundation

// MARK: - Default Trigger Definitions

/// All default triggers available for tracking
public let defaultTriggerDefinitions: [TriggerDefinition] = [

    // MARK: - Food & Drink

    TriggerDefinition(
        id: "coffee",
        name: "Coffee",
        category: .food,
        icon: "cup.and.saucer.fill",
        unit: "cups",
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Caffeine may affect sleep quality and inflammation markers. Some AS patients report increased symptoms with high intake.",
        trackingPrompt: "How many cups of coffee today?"
    ),

    TriggerDefinition(
        id: "alcohol",
        name: "Alcohol",
        category: .food,
        icon: "wineglass.fill",
        unit: "drinks",
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Alcohol can disrupt sleep, affect medication efficacy, and may modulate inflammatory pathways.",
        trackingPrompt: "How many alcoholic drinks today?"
    ),

    TriggerDefinition(
        id: "sugar_high",
        name: "High Sugar Intake",
        category: .food,
        icon: "birthday.cake.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "High sugar intake may promote systemic inflammation and worsen autoimmune symptoms.",
        trackingPrompt: "High sugar intake today?"
    ),

    TriggerDefinition(
        id: "dairy",
        name: "Dairy Products",
        category: .food,
        icon: "mug.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 48,
        dataSource: .manual,
        clinicalRelevance: "Some AS patients report dairy sensitivity. The gut-joint axis may play a role in inflammation.",
        trackingPrompt: "Did you consume dairy today?"
    ),

    TriggerDefinition(
        id: "gluten",
        name: "Gluten",
        category: .food,
        icon: "leaf.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 48,
        dataSource: .manual,
        clinicalRelevance: "Gluten sensitivity is common in autoimmune conditions. Some patients report symptom improvement with gluten-free diet.",
        trackingPrompt: "Did you consume gluten today?"
    ),

    TriggerDefinition(
        id: "processed_food",
        name: "Processed Food",
        category: .food,
        icon: "takeoutbag.and.cup.and.straw.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Ultra-processed foods contain additives that may promote inflammation.",
        trackingPrompt: "Did you eat processed/fast food today?"
    ),

    TriggerDefinition(
        id: "red_meat",
        name: "Red Meat",
        category: .food,
        icon: "flame.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 48,
        dataSource: .manual,
        clinicalRelevance: "Red meat consumption has been associated with increased inflammatory markers in some studies.",
        trackingPrompt: "Did you eat red meat today?"
    ),

    TriggerDefinition(
        id: "nightshades",
        name: "Nightshades",
        category: .food,
        icon: "carrot.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 48,
        dataSource: .manual,
        clinicalRelevance: "Nightshades (tomatoes, peppers, potatoes, eggplant) contain alkaloids that some patients report trigger symptoms.",
        trackingPrompt: "Did you eat nightshades today?"
    ),

    // MARK: - Sleep

    TriggerDefinition(
        id: "sleep_duration",
        name: "Sleep Duration",
        category: .sleep,
        icon: "moon.zzz.fill",
        unit: "hours",
        isBinary: false,
        minValue: 0,
        maxValue: 14,
        defaultValue: 7,
        expectedLagHours: 0,
        dataSource: .healthKit,
        clinicalRelevance: "Sleep deprivation increases inflammatory cytokines and pain sensitivity. 7-9 hours is optimal for most adults.",
        trackingPrompt: "Hours of sleep last night"
    ),

    TriggerDefinition(
        id: "sleep_quality",
        name: "Sleep Quality",
        category: .sleep,
        icon: "bed.double.fill",
        unit: nil,
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 5,
        expectedLagHours: 0,
        dataSource: .manual,
        clinicalRelevance: "Poor sleep quality strongly correlates with next-day pain intensity in AS patients.",
        trackingPrompt: "Rate your sleep quality (0-10)"
    ),

    TriggerDefinition(
        id: "late_bedtime",
        name: "Late Bedtime",
        category: .sleep,
        icon: "moon.stars.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 0,
        dataSource: .manual,
        clinicalRelevance: "Late bedtimes disrupt circadian rhythm which regulates immune function and inflammation.",
        trackingPrompt: "Did you go to bed after midnight?"
    ),

    TriggerDefinition(
        id: "sleep_interruptions",
        name: "Sleep Interruptions",
        category: .sleep,
        icon: "bell.slash.fill",
        unit: "times",
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 0,
        expectedLagHours: 0,
        dataSource: .manual,
        clinicalRelevance: "Frequent awakenings prevent restorative sleep and increase next-day pain and fatigue.",
        trackingPrompt: "How many times did you wake up?"
    ),

    // MARK: - Physical Activity

    TriggerDefinition(
        id: "steps",
        name: "Daily Steps",
        category: .activity,
        icon: "figure.walk",
        unit: "steps",
        isBinary: false,
        minValue: 0,
        maxValue: 50000,
        defaultValue: 0,
        expectedLagHours: 0,
        dataSource: .healthKit,
        clinicalRelevance: "Regular walking helps maintain joint mobility. Both very low and very high step counts can affect symptoms.",
        trackingPrompt: "Automatically tracked from HealthKit"
    ),

    TriggerDefinition(
        id: "exercise",
        name: "Exercise",
        category: .activity,
        icon: "figure.strengthtraining.traditional",
        unit: "minutes",
        isBinary: false,
        minValue: 0,
        maxValue: 300,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .healthKit,
        clinicalRelevance: "Regular exercise improves AS symptoms long-term but may cause temporary soreness. Moderate intensity is ideal.",
        trackingPrompt: "Minutes of exercise today"
    ),

    TriggerDefinition(
        id: "prolonged_sitting",
        name: "Prolonged Sitting",
        category: .activity,
        icon: "chair.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Prolonged sitting (>4 hours without breaks) worsens spinal stiffness and is a known AS trigger.",
        trackingPrompt: "Did you sit for long periods without breaks?"
    ),

    TriggerDefinition(
        id: "no_stretching",
        name: "Skipped Stretching",
        category: .activity,
        icon: "figure.flexibility",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Daily stretching is essential for maintaining spinal mobility in AS. Skipping increases stiffness.",
        trackingPrompt: "Did you skip your stretching routine?"
    ),

    TriggerDefinition(
        id: "heavy_lifting",
        name: "Heavy Lifting",
        category: .activity,
        icon: "dumbbell.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Heavy lifting can strain affected joints and trigger flares, especially for spine and SI joints.",
        trackingPrompt: "Did you lift heavy objects today?"
    ),

    // MARK: - Weather

    TriggerDefinition(
        id: "pressure_drop",
        name: "Barometric Pressure Drop",
        category: .weather,
        icon: "barometer",
        unit: "hPa",
        isBinary: false,
        minValue: -30,
        maxValue: 30,
        defaultValue: 0,
        expectedLagHours: 12,
        dataSource: .weather,
        clinicalRelevance: "Rapid barometric pressure drops (>5 hPa/12h) strongly correlate with AS flares in clinical studies.",
        trackingPrompt: "Automatically tracked from weather"
    ),

    TriggerDefinition(
        id: "high_humidity",
        name: "High Humidity",
        category: .weather,
        icon: "humidity.fill",
        unit: "%",
        isBinary: false,
        minValue: 0,
        maxValue: 100,
        defaultValue: 50,
        expectedLagHours: 0,
        dataSource: .weather,
        clinicalRelevance: "High humidity (>80%) may increase joint stiffness and discomfort in inflammatory arthritis.",
        trackingPrompt: "Automatically tracked from weather"
    ),

    TriggerDefinition(
        id: "cold_temperature",
        name: "Cold Temperature",
        category: .weather,
        icon: "thermometer.snowflake",
        unit: "°C",
        isBinary: false,
        minValue: -30,
        maxValue: 50,
        defaultValue: 20,
        expectedLagHours: 0,
        dataSource: .weather,
        clinicalRelevance: "Cold weather can worsen joint pain and stiffness, particularly for peripheral joints.",
        trackingPrompt: "Automatically tracked from weather"
    ),

    TriggerDefinition(
        id: "temperature_swing",
        name: "Temperature Swing",
        category: .weather,
        icon: "thermometer.variable.and.figure",
        unit: "°C",
        isBinary: false,
        minValue: 0,
        maxValue: 30,
        defaultValue: 0,
        expectedLagHours: 0,
        dataSource: .weather,
        clinicalRelevance: "Large temperature swings (>10°C in a day) may affect symptoms more than steady temperatures.",
        trackingPrompt: "Automatically tracked from weather"
    ),

    // MARK: - Stress & Mental

    TriggerDefinition(
        id: "stress",
        name: "Stress Level",
        category: .stress,
        icon: "brain.head.profile",
        unit: nil,
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Psychological stress activates inflammatory pathways and is a well-documented AS flare trigger.",
        trackingPrompt: "Rate your stress level today (0-10)"
    ),

    TriggerDefinition(
        id: "anxiety",
        name: "Anxiety Level",
        category: .stress,
        icon: "waveform.path.ecg",
        unit: nil,
        isBinary: false,
        minValue: 0,
        maxValue: 10,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Anxiety increases muscle tension and inflammation, commonly exacerbating AS symptoms.",
        trackingPrompt: "Rate your anxiety level today (0-10)"
    ),

    TriggerDefinition(
        id: "work_hours",
        name: "Long Work Day",
        category: .stress,
        icon: "briefcase.fill",
        unit: "hours",
        isBinary: false,
        minValue: 0,
        maxValue: 24,
        defaultValue: 8,
        expectedLagHours: 24,
        dataSource: .manual,
        clinicalRelevance: "Extended work hours (>10h) combine stress and often prolonged sitting, triggering symptoms.",
        trackingPrompt: "How many hours did you work today?"
    ),

    TriggerDefinition(
        id: "emotional_upset",
        name: "Emotional Upset",
        category: .stress,
        icon: "heart.slash.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 48,
        dataSource: .manual,
        clinicalRelevance: "Major emotional upsets can trigger prolonged inflammatory responses.",
        trackingPrompt: "Did you experience emotional upset today?"
    ),

    // MARK: - Medication

    TriggerDefinition(
        id: "missed_medication",
        name: "Missed Medication",
        category: .medication,
        icon: "pills.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 24,
        dataSource: .medication,
        clinicalRelevance: "Missing NSAID or DMARD doses allows inflammation to rebound, often triggering flares within 24-48h.",
        trackingPrompt: "Did you miss any medication doses?"
    ),

    TriggerDefinition(
        id: "missed_biologic",
        name: "Missed Biologic Dose",
        category: .medication,
        icon: "syringe.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 168,  // 7 days
        dataSource: .medication,
        clinicalRelevance: "Missing biologic doses (TNF inhibitors, IL-17 inhibitors) can trigger significant flares within 1-2 weeks.",
        trackingPrompt: "Did you miss your biologic dose?"
    ),

    // MARK: - Other

    TriggerDefinition(
        id: "travel",
        name: "Travel (Long Distance)",
        category: .other,
        icon: "airplane",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 48,
        dataSource: .manual,
        clinicalRelevance: "Long-distance travel combines sitting, stress, sleep disruption, and often missed medications.",
        trackingPrompt: "Did you travel long distance today?"
    ),

    TriggerDefinition(
        id: "illness",
        name: "Illness/Infection",
        category: .other,
        icon: "microbe.fill",
        isBinary: true,
        minValue: 0,
        maxValue: 1,
        defaultValue: 0,
        expectedLagHours: 48,
        dataSource: .manual,
        clinicalRelevance: "Infections activate the immune system and commonly trigger AS flares.",
        trackingPrompt: "Are you sick or fighting an infection?"
    )
]

// MARK: - Trigger Definition Lookup

/// Get trigger definition by ID
public func getTriggerDefinition(id: String) -> TriggerDefinition? {
    defaultTriggerDefinitions.first { $0.id == id }
}

/// Get all triggers for a category
public func getTriggers(for category: TriggerCategory) -> [TriggerDefinition] {
    defaultTriggerDefinitions.filter { $0.category == category }
}

/// Get all manual triggers (user-logged)
public func getManualTriggers() -> [TriggerDefinition] {
    defaultTriggerDefinitions.filter { $0.dataSource == .manual }
}

/// Get all automatic triggers (HealthKit, Weather)
public func getAutomaticTriggers() -> [TriggerDefinition] {
    defaultTriggerDefinitions.filter { $0.dataSource != .manual }
}

/// Get triggers by data source
public func getTriggers(source: TriggerDataSource) -> [TriggerDefinition] {
    defaultTriggerDefinitions.filter { $0.dataSource == source }
}
