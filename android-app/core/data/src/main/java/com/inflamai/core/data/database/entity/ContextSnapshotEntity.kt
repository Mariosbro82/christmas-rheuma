package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Environmental and biometric context data
 * Equivalent to iOS ContextSnapshot Core Data entity
 *
 * Captures weather conditions and biometric readings at the time of symptom logging
 * Critical for pattern analysis (barometric pressure drops are key AS triggers)
 */
@Entity(
    tableName = "context_snapshots",
    foreignKeys = [
        ForeignKey(
            entity = SymptomLogEntity::class,
            parentColumns = ["id"],
            childColumns = ["symptomLogId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index("symptomLogId", unique = true),
        Index("timestamp")
    ]
)
data class ContextSnapshotEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val symptomLogId: String,
    val timestamp: Long = System.currentTimeMillis(),

    // Weather Data (from Weather API)
    val barometricPressure: Double? = null,      // mmHg (critical for AS)
    val pressureChange12h: Double? = null,       // Change in last 12 hours
    val pressureChange24h: Double? = null,       // Change in last 24 hours
    val humidity: Int? = null,                   // Percentage 0-100
    val temperature: Double? = null,             // Celsius
    val temperatureFeelsLike: Double? = null,
    val precipitation: Boolean = false,
    val precipitationType: String? = null,       // "rain", "snow", "sleet"
    val cloudCover: Int? = null,                 // Percentage
    val windSpeed: Double? = null,               // km/h
    val uvIndex: Int? = null,
    val weatherCondition: String? = null,        // "clear", "cloudy", "rainy"

    // Location (for weather context, NOT stored long-term)
    val latitude: Double? = null,
    val longitude: Double? = null,
    val locationName: String? = null,

    // Biometric Data (from Health Connect)
    val hrvValue: Double? = null,                // HRV SDNN in ms
    val hrvRmssd: Double? = null,                // HRV RMSSD in ms
    val restingHeartRate: Int? = null,           // bpm
    val averageHeartRate: Int? = null,           // bpm
    val stepCount: Int? = null,                  // Steps today
    val distanceMeters: Double? = null,          // Distance walked

    // Sleep Data
    val sleepDurationHours: Double? = null,
    val sleepEfficiency: Double? = null,         // 0-1 ratio
    val sleepStartTime: Long? = null,
    val sleepEndTime: Long? = null,
    val sleepStages: String? = null,             // JSON of sleep stages

    // Activity Data
    val activeEnergyBurned: Double? = null,      // kcal
    val flightsClimbed: Int? = null,
    val exerciseMinutes: Int? = null,

    // Oxygen & Respiratory
    val oxygenSaturation: Double? = null,        // Percentage
    val respiratoryRate: Double? = null,         // breaths/min

    val lastModified: Long = System.currentTimeMillis()
)
