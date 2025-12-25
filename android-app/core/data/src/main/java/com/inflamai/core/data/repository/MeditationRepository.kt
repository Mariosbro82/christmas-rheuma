package com.inflamai.core.data.repository

import com.inflamai.core.data.database.dao.MeditationSessionDao
import com.inflamai.core.data.database.dao.MeditationStreakDao
import com.inflamai.core.data.database.entity.MeditationSessionEntity
import com.inflamai.core.data.database.entity.MeditationStreakEntity
import kotlinx.coroutines.flow.Flow
import java.time.LocalDate
import java.time.ZoneId
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for meditation data
 * Abstracts MeditationSessionDao and MeditationStreakDao
 */
@Singleton
class MeditationRepository @Inject constructor(
    private val meditationSessionDao: MeditationSessionDao,
    private val meditationStreakDao: MeditationStreakDao
) {
    // Sessions
    fun observeAllSessions(): Flow<List<MeditationSessionEntity>> =
        meditationSessionDao.observeAll()

    fun observeRecentSessions(limit: Int): Flow<List<MeditationSessionEntity>> =
        meditationSessionDao.observeRecent(limit)

    fun observeSessionsByDateRange(startDate: Long, endDate: Long): Flow<List<MeditationSessionEntity>> =
        meditationSessionDao.observeByDateRange(startDate, endDate)

    suspend fun getTotalMinutes(startDate: Long, endDate: Long): Int =
        meditationSessionDao.getTotalMinutes(startDate, endDate) ?: 0

    suspend fun insertSession(session: MeditationSessionEntity): Long =
        meditationSessionDao.insert(session)

    suspend fun updateSession(session: MeditationSessionEntity) =
        meditationSessionDao.update(session)

    suspend fun deleteSession(session: MeditationSessionEntity) =
        meditationSessionDao.delete(session)

    // Streak
    fun observeStreak(): Flow<MeditationStreakEntity?> = meditationStreakDao.observe()

    suspend fun getStreak(): MeditationStreakEntity? = meditationStreakDao.get()

    suspend fun updateStreak(streak: MeditationStreakEntity) =
        meditationStreakDao.update(streak)

    // Log a completed meditation session and update streak
    suspend fun logMeditationSession(
        sessionType: String,
        durationMinutes: Int,
        category: String
    ): Long {
        // Insert session
        val session = MeditationSessionEntity(
            id = UUID.randomUUID().toString(),
            sessionType = sessionType,
            durationMinutes = durationMinutes,
            timestamp = System.currentTimeMillis(),
            category = category,
            wasCompleted = true
        )
        val sessionId = meditationSessionDao.insert(session)

        // Update streak
        val currentStreak = meditationStreakDao.get()
        val today = LocalDate.now()
        val todayMillis = today.atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()

        if (currentStreak == null) {
            // First meditation session ever
            meditationStreakDao.insert(
                MeditationStreakEntity(
                    currentStreak = 1,
                    longestStreak = 1,
                    totalMinutes = durationMinutes,
                    lastSessionDate = todayMillis
                )
            )
        } else {
            val lastSessionDate = LocalDate.ofEpochDay(currentStreak.lastSessionDate / 86400000)
            val daysDiff = today.toEpochDay() - lastSessionDate.toEpochDay()

            val newStreak = when {
                daysDiff == 0L -> currentStreak.currentStreak // Same day, no change
                daysDiff == 1L -> currentStreak.currentStreak + 1 // Consecutive day
                else -> 1 // Streak broken, reset
            }

            val newLongest = maxOf(newStreak, currentStreak.longestStreak)

            meditationStreakDao.update(
                currentStreak.copy(
                    currentStreak = newStreak,
                    longestStreak = newLongest,
                    totalMinutes = currentStreak.totalMinutes + durationMinutes,
                    lastSessionDate = todayMillis
                )
            )
        }

        return sessionId
    }
}
