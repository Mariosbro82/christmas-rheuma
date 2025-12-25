package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Main symptom tracking entity - equivalent to iOS SymptomLog Core Data entity
 * Contains BASDAI scores, ML features (60+ fields), and symptom data
 */
@Entity(
    tableName = "symptom_logs",
    indices = [
        Index("timestamp"),
        Index("isFlareEvent")
    ]
)
data class SymptomLogEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),

    // BASDAI Core Scores (0-10)
    val basdaiScore: Double = 0.0,
    val asdasScore: Double = 0.0,

    // Primary Symptom Metrics
    val fatigueLevel: Int = 0,        // 0-10
    val moodScore: Int = 5,           // 1-10
    val sleepQuality: Int = 5,        // 1-10
    val sleepDurationHours: Double = 0.0,
    val morningStiffnessMinutes: Int = 0,
    val isFlareEvent: Boolean = false,

    // ML Clinical Features
    val basfi: Double? = null,           // Bath AS Functional Index
    val basmi: Double? = null,           // Bath AS Metrology Index
    val patientGlobal: Int? = null,      // 0-10
    val tenderJointCount: Int? = null,
    val swollenJointCount: Int? = null,
    val crpValue: Double? = null,        // mg/L for ASDAS-CRP
    val esrValue: Double? = null,        // mm/hr

    // Pain Features (0-10)
    val painAverage24h: Double? = null,
    val painMax24h: Double? = null,
    val nocturnalPain: Int? = null,
    val painBurning: Boolean = false,
    val painSharp: Boolean = false,
    val painDull: Boolean = false,
    val painRadiating: Boolean = false,

    // Mental Health Features
    val stressLevel: Int? = null,        // 0-10
    val anxietyLevel: Int? = null,       // 0-10
    val mentalFatigue: Int? = null,      // 0-10
    val depressionRisk: Int? = null,     // 0-10
    val moodValence: Int? = null,        // -5 to +5

    // Activity Features
    val exerciseMinutesToday: Int? = null,
    val activityLimitationScore: Int? = null,
    val physicalFunctionScore: Double? = null,

    // Context Features
    val overallFeeling: Int? = null,     // 1-10
    val dayQuality: Int? = null,         // 1-10
    val energyLevel: Int? = null,        // 1-10
    val copingAbility: Int? = null,      // 1-10

    // Check-in Question Responses (Q1-Q6 for BASDAI)
    val q1Fatigue: Double = 0.0,         // 0-10
    val q2SpinalPain: Double = 0.0,      // 0-10
    val q3PeripheralPain: Double = 0.0,  // 0-10
    val q4Tenderness: Double = 0.0,      // 0-10 (enthesitis)
    val q5MorningStiffnessSeverity: Double = 0.0, // 0-10
    val q6MorningStiffnessDuration: Double = 0.0, // 0-10 (scaled from minutes)

    // Notes
    val notes: String? = null,

    // Sync
    val lastModified: Long = System.currentTimeMillis(),
    val isSynced: Boolean = false
)
