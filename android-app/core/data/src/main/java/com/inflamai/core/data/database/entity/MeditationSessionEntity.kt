package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Meditation session tracking with before/after metrics
 * Equivalent to iOS MeditationSession Core Data entity
 */
@Entity(
    tableName = "meditation_sessions",
    indices = [
        Index("timestamp"),
        Index("meditationType")
    ]
)
data class MeditationSessionEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),

    // Session Details
    val meditationType: MeditationType = MeditationType.GUIDED,
    val durationMinutes: Int,
    val durationSeconds: Int = 0,
    val wasCompleted: Boolean = true,

    // Audio/Content
    val contentId: String? = null,
    val contentTitle: String? = null,
    val instructorName: String? = null,

    // Before Metrics (1-10)
    val stressBefore: Int? = null,
    val painBefore: Int? = null,
    val anxietyBefore: Int? = null,
    val moodBefore: Int? = null,

    // After Metrics (1-10)
    val stressAfter: Int? = null,
    val painAfter: Int? = null,
    val anxietyAfter: Int? = null,
    val moodAfter: Int? = null,

    // Biometrics (from Health Connect)
    val heartRateBefore: Int? = null,
    val heartRateAfter: Int? = null,
    val hrvBefore: Double? = null,
    val hrvAfter: Double? = null,

    // Experience
    val focus: Int? = null,          // 1-10 how focused
    val relaxation: Int? = null,     // 1-10 how relaxed
    val difficulty: Int? = null,     // 1-5 how hard to maintain

    // Notes
    val notes: String? = null,

    val createdAt: Long = System.currentTimeMillis(),
    val lastModified: Long = System.currentTimeMillis()
)

enum class MeditationType {
    GUIDED,
    BREATHING,
    BODY_SCAN,
    MINDFULNESS,
    VISUALIZATION,
    LOVING_KINDNESS,
    SLEEP,
    PAIN_MANAGEMENT,
    UNGUIDED
}

/**
 * Meditation streak tracking (singleton)
 */
@Entity(tableName = "meditation_streak")
data class MeditationStreakEntity(
    @PrimaryKey
    val id: String = "meditation_streak",

    val currentStreak: Int = 0,
    val longestStreak: Int = 0,
    val totalSessions: Int = 0,
    val totalMinutes: Int = 0,

    val lastSessionDate: Long? = null,
    val streakStartDate: Long? = null,

    val lastModified: Long = System.currentTimeMillis()
)
