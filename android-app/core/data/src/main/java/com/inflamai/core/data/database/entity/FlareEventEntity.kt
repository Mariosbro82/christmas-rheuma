package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Flare episode tracking
 * Equivalent to iOS FlareEvent Core Data entity
 */
@Entity(
    tableName = "flare_events",
    indices = [
        Index("startDate"),
        Index("isResolved")
    ]
)
data class FlareEventEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),

    // Timing
    val startDate: Long,
    val endDate: Long? = null,
    val isResolved: Boolean = false,

    // Severity (1-10)
    val severity: Int,
    val peakSeverity: Int? = null,
    val peakSeverityDate: Long? = null,

    // Affected Regions (JSON array of region IDs)
    val primaryRegionsJson: String = "[]",
    val secondaryRegionsJson: String = "[]",

    // Triggers (JSON array of suspected triggers)
    val suspectedTriggersJson: String = "[]",

    // Type
    val flareType: FlareType = FlareType.GENERAL,

    // Interventions
    val interventionsJson: String = "[]",  // JSON array of interventions taken
    val medicationsTakenJson: String = "[]", // Medications specifically for flare

    // Impact
    val workMissedDays: Int = 0,
    val hospitalVisit: Boolean = false,
    val emergencyVisit: Boolean = false,

    // Notes
    val notes: String? = null,
    val triggerNotes: String? = null,

    val createdAt: Long = System.currentTimeMillis(),
    val lastModified: Long = System.currentTimeMillis(),
    val isSynced: Boolean = false
)

enum class FlareType {
    GENERAL,
    AXIAL,        // Spine-focused
    PERIPHERAL,   // Peripheral joints
    ENTHESITIS,   // Tendon attachments
    DACTYLITIS,   // Sausage digits
    UVEITIS,      // Eye inflammation
    MIXED
}

/**
 * Common suspected triggers for AS flares
 */
enum class FlareTrigger(val displayName: String) {
    WEATHER_PRESSURE("Barometric pressure change"),
    WEATHER_COLD("Cold weather"),
    WEATHER_HUMIDITY("Humidity"),
    STRESS("Stress"),
    POOR_SLEEP("Poor sleep"),
    MISSED_MEDICATION("Missed medication"),
    OVEREXERTION("Physical overexertion"),
    PROLONGED_SITTING("Prolonged sitting"),
    INFECTION("Infection/illness"),
    DIET("Diet"),
    ALCOHOL("Alcohol"),
    UNKNOWN("Unknown")
}

/**
 * Common interventions for managing flares
 */
enum class FlareIntervention(val displayName: String) {
    REST("Rest"),
    ICE("Ice/cold therapy"),
    HEAT("Heat therapy"),
    NSAID("NSAID medication"),
    PAIN_RELIEVER("Pain reliever"),
    STRETCHING("Gentle stretching"),
    HOT_BATH("Hot bath/shower"),
    MASSAGE("Massage"),
    DOCTOR_VISIT("Doctor visit"),
    EMERGENCY_CARE("Emergency care"),
    BIOLOGIC_DOSE("Biologic dose"),
    STEROID_INJECTION("Steroid injection"),
    OTHER("Other")
}
