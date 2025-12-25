package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Meditation streak tracking entity
 */
@Entity(tableName = "meditation_streaks")
data class MeditationStreakEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),

    // Current streak
    val currentStreak: Int = 0,
    val longestStreak: Int = 0,

    // Total stats
    val totalSessions: Int = 0,
    val totalMinutes: Int = 0,

    // Streak tracking dates
    val lastSessionDate: Long? = null,
    val streakStartDate: Long? = null,

    // Weekly goal
    val weeklyGoalMinutes: Int = 60,
    val weeklyCompletedMinutes: Int = 0,
    val weekStartDate: Long = System.currentTimeMillis(),

    val createdAt: Long = System.currentTimeMillis(),
    val lastModified: Long = System.currentTimeMillis()
)
