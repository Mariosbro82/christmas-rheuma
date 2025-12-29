package com.inflamai.core.data.repository

import com.inflamai.core.data.database.dao.ExerciseDao
import com.inflamai.core.data.database.dao.ExerciseSessionDao
import com.inflamai.core.data.database.entity.ExerciseEntity
import com.inflamai.core.data.database.entity.ExerciseSessionEntity
import kotlinx.coroutines.flow.Flow
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for exercise data
 * Abstracts ExerciseDao and ExerciseSessionDao
 */
@Singleton
class ExerciseRepository @Inject constructor(
    private val exerciseDao: ExerciseDao,
    private val exerciseSessionDao: ExerciseSessionDao
) {
    // Exercise Library
    fun observeAllExercises(): Flow<List<ExerciseEntity>> = exerciseDao.observeAll()

    fun observeExercisesByCategory(category: String): Flow<List<ExerciseEntity>> =
        exerciseDao.observeByCategory(category)

    fun observeBeginnerExercises(): Flow<List<ExerciseEntity>> = exerciseDao.observeBeginner()

    fun searchExercises(query: String): Flow<List<ExerciseEntity>> = exerciseDao.search(query)

    suspend fun getExerciseById(id: String): ExerciseEntity? = exerciseDao.getById(id)

    suspend fun insertExercise(exercise: ExerciseEntity): Long = exerciseDao.insert(exercise)

    suspend fun insertExercises(exercises: List<ExerciseEntity>) = exerciseDao.insertAll(exercises)

    // Exercise Sessions
    fun observeAllSessions(): Flow<List<ExerciseSessionEntity>> = exerciseSessionDao.observeAll()

    fun observeRecentSessions(limit: Int): Flow<List<ExerciseSessionEntity>> =
        exerciseSessionDao.observeRecent(limit)

    fun observeSessionsByDateRange(startDate: Long, endDate: Long): Flow<List<ExerciseSessionEntity>> =
        exerciseSessionDao.observeByDateRange(startDate, endDate)

    suspend fun getTotalExerciseMinutes(startDate: Long, endDate: Long): Int =
        exerciseSessionDao.getTotalMinutes(startDate, endDate) ?: 0

    suspend fun getSessionCount(startDate: Long, endDate: Long): Int =
        exerciseSessionDao.getSessionCount(startDate, endDate)

    suspend fun insertSession(session: ExerciseSessionEntity): Long =
        exerciseSessionDao.insert(session)

    suspend fun updateSession(session: ExerciseSessionEntity) =
        exerciseSessionDao.update(session)

    suspend fun deleteSession(session: ExerciseSessionEntity) =
        exerciseSessionDao.delete(session)

    // Log a completed exercise session
    suspend fun logExerciseSession(
        routineName: String,
        durationMinutes: Int,
        intensityLevel: Int,
        notes: String? = null,
        painBefore: Int? = null,
        painAfter: Int? = null
    ): Long {
        val session = ExerciseSessionEntity(
            id = UUID.randomUUID().toString(),
            routineName = routineName,
            timestamp = System.currentTimeMillis(),
            durationMinutes = durationMinutes,
            intensityLevel = intensityLevel,
            notes = notes,
            painBefore = painBefore ?: 0,
            painAfter = painAfter ?: 0,
            painDelta = (painAfter ?: 0) - (painBefore ?: 0),
            hadPainIncrease = (painAfter ?: 0) > (painBefore ?: 0)
        )
        return exerciseSessionDao.insert(session)
    }
}
