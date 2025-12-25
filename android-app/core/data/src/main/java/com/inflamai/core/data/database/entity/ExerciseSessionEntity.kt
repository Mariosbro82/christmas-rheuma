package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Exercise/workout session tracking
 * Equivalent to iOS ExerciseSession Core Data entity
 */
@Entity(
    tableName = "exercise_sessions",
    indices = [
        Index("timestamp"),
        Index("routineType")
    ]
)
data class ExerciseSessionEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),

    // Session Details
    val routineType: ExerciseRoutineType = ExerciseRoutineType.CUSTOM,
    val routineName: String? = null,
    val durationMinutes: Int,
    val durationSeconds: Int = 0,

    // Exercises (JSON array of exercise IDs completed)
    val exercisesCompletedJson: String = "[]",
    val exerciseCount: Int = 0,

    // Intensity & Effort
    val intensityLevel: Int = 5,         // 1-10
    val perceivedExertion: Int? = null,  // RPE 1-10

    // Pain Tracking
    val painBefore: Int = 0,             // 0-10
    val painAfter: Int = 0,              // 0-10
    val painDelta: Int = 0,              // After - Before
    val hadPainIncrease: Boolean = false,

    // Joint-Specific Pain (JSON)
    val preExerciseJointPainJson: String = "{}",  // {"region_id": pain_level}
    val postExerciseJointPainJson: String = "{}",

    // Mannequin Coach Fields (for guided sessions)
    val flowId: String? = null,
    val flowTitle: String? = null,
    val cyclesCompleted: Int = 1,
    val cyclesTarget: Int = 1,
    val romMultiplier: Double = 1.0,     // Range of motion adjustment
    val speedMultiplier: Double = 1.0,   // Speed adjustment
    val userConfidence: Int? = null,     // 0-5

    // Flare Mode
    val wasInFlareMode: Boolean = false,
    val flareAdjustments: String? = null,

    // Notes
    val notes: String? = null,
    val mood: String? = null,

    // Biometrics During Exercise (from Health Connect)
    val averageHeartRate: Int? = null,
    val maxHeartRate: Int? = null,
    val caloriesBurned: Double? = null,

    val createdAt: Long = System.currentTimeMillis(),
    val lastModified: Long = System.currentTimeMillis(),
    val isSynced: Boolean = false
)

enum class ExerciseRoutineType {
    MORNING_STRETCH,
    EVENING_STRETCH,
    STRENGTH,
    CARDIO,
    FLEXIBILITY,
    BALANCE,
    AQUATIC,
    YOGA,
    PILATES,
    WALKING,
    CUSTOM,
    COACH_GUIDED
}

/**
 * Individual exercise definition (for the 50+ exercise library)
 */
@Entity(
    tableName = "exercises",
    indices = [
        Index("category"),
        Index("difficultyLevel")
    ]
)
data class ExerciseEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),

    val name: String,
    val description: String,
    val category: ExerciseCategory,
    val difficultyLevel: Int,            // 1-5
    val durationSeconds: Int,
    val repetitions: Int? = null,
    val sets: Int? = null,

    // Target Areas (JSON array of body region IDs)
    val targetAreasJson: String = "[]",

    // Instructions
    val instructionsJson: String = "[]", // JSON array of step-by-step instructions
    val tips: String? = null,
    val warnings: String? = null,

    // Media
    val videoUrl: String? = null,
    val imageUrl: String? = null,
    val animationData: String? = null,   // Lottie or Rive animation

    // AS-Specific
    val isAsRecommended: Boolean = true,
    val flareModeSafe: Boolean = true,
    val flareAlternativeId: String? = null,

    // ROM & Speed Modifiers Allowed
    val allowRomAdjustment: Boolean = true,
    val allowSpeedAdjustment: Boolean = true,

    val createdAt: Long = System.currentTimeMillis()
)

enum class ExerciseCategory {
    STRETCHING,
    STRENGTHENING,
    CARDIO,
    BALANCE,
    BREATHING,
    RELAXATION,
    POSTURE,
    AQUATIC
}
