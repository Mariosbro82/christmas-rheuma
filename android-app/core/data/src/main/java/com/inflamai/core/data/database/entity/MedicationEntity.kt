package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Medication/prescription tracking
 * Equivalent to iOS Medication Core Data entity
 */
@Entity(
    tableName = "medications",
    indices = [
        Index("isActive"),
        Index("name")
    ]
)
data class MedicationEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),

    // Medication Details
    val name: String,
    val genericName: String? = null,
    val category: MedicationCategory = MedicationCategory.OTHER,
    val dosage: Double,
    val dosageUnit: String = "mg",
    val frequency: MedicationFrequency = MedicationFrequency.DAILY,
    val route: MedicationRoute = MedicationRoute.ORAL,

    // AS-Specific
    val isBiologic: Boolean = false,
    val isNSAID: Boolean = false,
    val isDMARD: Boolean = false,

    // Schedule
    val startDate: Long,
    val endDate: Long? = null,
    val isActive: Boolean = true,

    // Reminders (JSON array of times in "HH:mm" format)
    val reminderEnabled: Boolean = false,
    val reminderTimesJson: String = "[]",

    // Prescriber
    val prescribedBy: String? = null,
    val prescribedDate: Long? = null,

    // Notes
    val instructions: String? = null,
    val sideEffects: String? = null,
    val notes: String? = null,

    // Refill
    val refillReminder: Boolean = false,
    val lastRefillDate: Long? = null,
    val pillCount: Int? = null,

    val createdAt: Long = System.currentTimeMillis(),
    val lastModified: Long = System.currentTimeMillis(),
    val isSynced: Boolean = false
)

enum class MedicationCategory {
    NSAID,           // Non-steroidal anti-inflammatory
    BIOLOGIC,        // TNF inhibitors, IL-17 inhibitors
    DMARD,           // Disease-modifying antirheumatic drugs
    CORTICOSTEROID,
    PAIN_RELIEVER,
    MUSCLE_RELAXANT,
    SUPPLEMENT,
    OTHER
}

enum class MedicationFrequency {
    AS_NEEDED,
    ONCE_DAILY,
    TWICE_DAILY,
    THREE_TIMES_DAILY,
    FOUR_TIMES_DAILY,
    WEEKLY,
    BIWEEKLY,
    MONTHLY,
    DAILY;

    fun displayName(): String = when (this) {
        AS_NEEDED -> "As needed"
        ONCE_DAILY -> "Once daily"
        TWICE_DAILY -> "Twice daily"
        THREE_TIMES_DAILY -> "Three times daily"
        FOUR_TIMES_DAILY -> "Four times daily"
        WEEKLY -> "Weekly"
        BIWEEKLY -> "Every two weeks"
        MONTHLY -> "Monthly"
        DAILY -> "Daily"
    }
}

enum class MedicationRoute {
    ORAL,
    INJECTION,
    TOPICAL,
    SUBLINGUAL,
    INHALATION,
    IV;

    fun displayName(): String = when (this) {
        ORAL -> "By mouth"
        INJECTION -> "Injection"
        TOPICAL -> "Topical"
        SUBLINGUAL -> "Under tongue"
        INHALATION -> "Inhaled"
        IV -> "Intravenous"
    }
}
