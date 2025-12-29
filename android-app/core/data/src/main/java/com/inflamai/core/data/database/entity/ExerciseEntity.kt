package com.inflamai.core.data.database.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Exercise library entity - stores exercise definitions
 * Based on iOS exercise library with 52+ exercises
 */
@Entity(tableName = "exercises")
data class ExerciseEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),

    @ColumnInfo(name = "name")
    val name: String,

    @ColumnInfo(name = "category")
    val category: ExerciseCategory,

    @ColumnInfo(name = "difficulty")
    val level: ExerciseLevel,

    @ColumnInfo(name = "durationMinutes")
    val durationMinutes: Int,

    // Target areas (comma-separated)
    @ColumnInfo(name = "targetAreas")
    val targetAreas: String,

    // Steps as JSON array
    val stepsJson: String = "[]",

    // Media
    val videoUrl: String? = null,
    val thumbnailUrl: String? = null,

    // Benefits and safety tips as JSON arrays
    val benefitsJson: String = "[]",
    val safetyTipsJson: String = "[]",

    // Display order
    val displayOrder: Int = 0,

    // Metadata
    val isBuiltIn: Boolean = true,
    val createdAt: Long = System.currentTimeMillis()
)

enum class ExerciseCategory {
    STRETCHING,
    STRENGTHENING,
    MOBILITY,
    BREATHING,
    BALANCE,
    POSTURE,
    RELAXATION
}

enum class ExerciseLevel {
    BEGINNER,
    INTERMEDIATE,
    ADVANCED
}

/**
 * Exercise step data class (used in JSON serialization)
 */
data class ExerciseStep(
    val order: Int,
    val instruction: String,
    val type: StepType,
    val durationSeconds: Int? = null,
    val reps: Int? = null,
    val iconType: String = "circle"
)

enum class StepType {
    INFO,
    TIMER,
    REPS
}
