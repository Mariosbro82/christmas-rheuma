package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Medication adherence tracking
 * Equivalent to iOS DoseLog/MedicationLog Core Data entity
 */
@Entity(
    tableName = "dose_logs",
    foreignKeys = [
        ForeignKey(
            entity = MedicationEntity::class,
            parentColumns = ["id"],
            childColumns = ["medicationId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index("medicationId"),
        Index("timestamp"),
        Index("scheduledTime")
    ]
)
data class DoseLogEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val medicationId: String,
    val timestamp: Long = System.currentTimeMillis(),

    // Scheduling
    val scheduledTime: Long,
    val actualTakenTime: Long? = null,

    // Dose Info
    val dosageTaken: Double,
    val wasSkipped: Boolean = false,
    val wasTakenLate: Boolean = false,

    // Skip Reason
    val skipReason: SkipReason? = null,
    val skipReasonOther: String? = null,

    // Effects
    val hadSideEffects: Boolean = false,
    val sideEffectNotes: String? = null,

    // Notes
    val notes: String? = null,

    val lastModified: Long = System.currentTimeMillis(),
    val isSynced: Boolean = false
)

enum class SkipReason {
    FORGOT,
    FELT_BETTER,
    SIDE_EFFECTS,
    RAN_OUT,
    COST,
    DOCTOR_ADVISED,
    FELT_WORSE,
    OTHER
}
