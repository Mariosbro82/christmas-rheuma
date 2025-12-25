package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Body region pain/symptom tracking - 47 anatomical regions
 * Equivalent to iOS BodyRegionLog Core Data entity
 *
 * Regions include:
 * - Spine: C1-C7 (cervical), T1-T12 (thoracic), L1-L5 (lumbar), sacrum, SI joints
 * - Peripherals: shoulders, elbows, wrists, hands (bilateral)
 * - Lower: hips, knees, ankles, feet (bilateral)
 */
@Entity(
    tableName = "body_region_logs",
    foreignKeys = [
        ForeignKey(
            entity = SymptomLogEntity::class,
            parentColumns = ["id"],
            childColumns = ["symptomLogId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index("symptomLogId"),
        Index("regionId"),
        Index("timestamp")
    ]
)
data class BodyRegionLogEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val symptomLogId: String,
    val timestamp: Long = System.currentTimeMillis(),

    // Region Identification
    val regionId: String,                // e.g., "c1", "l5", "si_left", "shoulder_right"
    val regionName: String,              // Human-readable name

    // Pain Assessment (0-10)
    val painLevel: Int = 0,

    // Stiffness
    val stiffnessDuration: Int = 0,      // Minutes
    val stiffnessSeverity: Int = 0,      // 0-10

    // Inflammation Signs
    val hasSwelling: Boolean = false,
    val hasWarmth: Boolean = false,
    val hasRedness: Boolean = false,

    // Mobility
    val rangeOfMotion: String? = null,   // "full", "limited", "severely_limited"
    val mobilityScore: Int? = null,      // 0-10

    // Notes
    val notes: String? = null,

    // Photo reference (file path or URI)
    val photoUri: String? = null,

    val lastModified: Long = System.currentTimeMillis()
)

/**
 * Enum representing all 47 trackable body regions
 * Used for type-safe region identification
 */
enum class BodyRegion(
    val id: String,
    val displayName: String,
    val category: BodyRegionCategory,
    val isSymmetric: Boolean = true
) {
    // Cervical Spine (C1-C7)
    CERVICAL_C1("c1", "C1 (Atlas)", BodyRegionCategory.SPINE, false),
    CERVICAL_C2("c2", "C2 (Axis)", BodyRegionCategory.SPINE, false),
    CERVICAL_C3("c3", "C3", BodyRegionCategory.SPINE, false),
    CERVICAL_C4("c4", "C4", BodyRegionCategory.SPINE, false),
    CERVICAL_C5("c5", "C5", BodyRegionCategory.SPINE, false),
    CERVICAL_C6("c6", "C6", BodyRegionCategory.SPINE, false),
    CERVICAL_C7("c7", "C7", BodyRegionCategory.SPINE, false),

    // Thoracic Spine (T1-T12)
    THORACIC_T1("t1", "T1", BodyRegionCategory.SPINE, false),
    THORACIC_T2("t2", "T2", BodyRegionCategory.SPINE, false),
    THORACIC_T3("t3", "T3", BodyRegionCategory.SPINE, false),
    THORACIC_T4("t4", "T4", BodyRegionCategory.SPINE, false),
    THORACIC_T5("t5", "T5", BodyRegionCategory.SPINE, false),
    THORACIC_T6("t6", "T6", BodyRegionCategory.SPINE, false),
    THORACIC_T7("t7", "T7", BodyRegionCategory.SPINE, false),
    THORACIC_T8("t8", "T8", BodyRegionCategory.SPINE, false),
    THORACIC_T9("t9", "T9", BodyRegionCategory.SPINE, false),
    THORACIC_T10("t10", "T10", BodyRegionCategory.SPINE, false),
    THORACIC_T11("t11", "T11", BodyRegionCategory.SPINE, false),
    THORACIC_T12("t12", "T12", BodyRegionCategory.SPINE, false),

    // Lumbar Spine (L1-L5)
    LUMBAR_L1("l1", "L1", BodyRegionCategory.SPINE, false),
    LUMBAR_L2("l2", "L2", BodyRegionCategory.SPINE, false),
    LUMBAR_L3("l3", "L3", BodyRegionCategory.SPINE, false),
    LUMBAR_L4("l4", "L4", BodyRegionCategory.SPINE, false),
    LUMBAR_L5("l5", "L5", BodyRegionCategory.SPINE, false),

    // Sacrum & SI Joints
    SACRUM("sacrum", "Sacrum", BodyRegionCategory.SPINE, false),
    SI_JOINT_LEFT("si_left", "Left SI Joint", BodyRegionCategory.SPINE, true),
    SI_JOINT_RIGHT("si_right", "Right SI Joint", BodyRegionCategory.SPINE, true),

    // Upper Extremities
    SHOULDER_LEFT("shoulder_left", "Left Shoulder", BodyRegionCategory.UPPER_EXTREMITY, true),
    SHOULDER_RIGHT("shoulder_right", "Right Shoulder", BodyRegionCategory.UPPER_EXTREMITY, true),
    ELBOW_LEFT("elbow_left", "Left Elbow", BodyRegionCategory.UPPER_EXTREMITY, true),
    ELBOW_RIGHT("elbow_right", "Right Elbow", BodyRegionCategory.UPPER_EXTREMITY, true),
    WRIST_LEFT("wrist_left", "Left Wrist", BodyRegionCategory.UPPER_EXTREMITY, true),
    WRIST_RIGHT("wrist_right", "Right Wrist", BodyRegionCategory.UPPER_EXTREMITY, true),
    HAND_LEFT("hand_left", "Left Hand", BodyRegionCategory.UPPER_EXTREMITY, true),
    HAND_RIGHT("hand_right", "Right Hand", BodyRegionCategory.UPPER_EXTREMITY, true),

    // Lower Extremities
    HIP_LEFT("hip_left", "Left Hip", BodyRegionCategory.LOWER_EXTREMITY, true),
    HIP_RIGHT("hip_right", "Right Hip", BodyRegionCategory.LOWER_EXTREMITY, true),
    KNEE_LEFT("knee_left", "Left Knee", BodyRegionCategory.LOWER_EXTREMITY, true),
    KNEE_RIGHT("knee_right", "Right Knee", BodyRegionCategory.LOWER_EXTREMITY, true),
    ANKLE_LEFT("ankle_left", "Left Ankle", BodyRegionCategory.LOWER_EXTREMITY, true),
    ANKLE_RIGHT("ankle_right", "Right Ankle", BodyRegionCategory.LOWER_EXTREMITY, true),
    FOOT_LEFT("foot_left", "Left Foot", BodyRegionCategory.LOWER_EXTREMITY, true),
    FOOT_RIGHT("foot_right", "Right Foot", BodyRegionCategory.LOWER_EXTREMITY, true),

    // Chest & Ribs
    CHEST("chest", "Chest", BodyRegionCategory.THORAX, false),
    RIBS_LEFT("ribs_left", "Left Ribs", BodyRegionCategory.THORAX, true),
    RIBS_RIGHT("ribs_right", "Right Ribs", BodyRegionCategory.THORAX, true);

    companion object {
        fun fromId(id: String): BodyRegion? = entries.find { it.id == id }

        fun spineRegions(): List<BodyRegion> =
            entries.filter { it.category == BodyRegionCategory.SPINE }

        fun peripheralRegions(): List<BodyRegion> =
            entries.filter { it.category != BodyRegionCategory.SPINE }
    }
}

enum class BodyRegionCategory {
    SPINE,
    UPPER_EXTREMITY,
    LOWER_EXTREMITY,
    THORAX
}
