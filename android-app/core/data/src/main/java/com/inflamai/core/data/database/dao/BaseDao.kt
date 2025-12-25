package com.inflamai.core.data.database.dao

import androidx.room.*
import com.inflamai.core.data.database.entity.*
import kotlinx.coroutines.flow.Flow

/**
 * BodyRegionLog DAO
 */
@Dao
interface BodyRegionLogDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(bodyRegionLog: BodyRegionLogEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(bodyRegionLogs: List<BodyRegionLogEntity>)

    @Update
    suspend fun update(bodyRegionLog: BodyRegionLogEntity)

    @Delete
    suspend fun delete(bodyRegionLog: BodyRegionLogEntity)

    @Query("SELECT * FROM body_region_logs WHERE symptomLogId = :symptomLogId")
    fun observeBySymptomLog(symptomLogId: String): Flow<List<BodyRegionLogEntity>>

    @Query("SELECT * FROM body_region_logs WHERE symptomLogId = :symptomLogId")
    suspend fun getBySymptomLog(symptomLogId: String): List<BodyRegionLogEntity>

    @Query("SELECT * FROM body_region_logs WHERE regionId = :regionId ORDER BY timestamp DESC")
    fun observeByRegion(regionId: String): Flow<List<BodyRegionLogEntity>>

    @Query("SELECT AVG(painLevel) FROM body_region_logs WHERE regionId = :regionId AND timestamp BETWEEN :startDate AND :endDate")
    suspend fun getAveragePainForRegion(regionId: String, startDate: Long, endDate: Long): Double?

    @Query("DELETE FROM body_region_logs WHERE symptomLogId = :symptomLogId")
    suspend fun deleteBySymptomLog(symptomLogId: String)
}

/**
 * ContextSnapshot DAO
 */
@Dao
interface ContextSnapshotDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(contextSnapshot: ContextSnapshotEntity): Long

    @Update
    suspend fun update(contextSnapshot: ContextSnapshotEntity)

    @Delete
    suspend fun delete(contextSnapshot: ContextSnapshotEntity)

    @Query("SELECT * FROM context_snapshots WHERE symptomLogId = :symptomLogId")
    suspend fun getBySymptomLog(symptomLogId: String): ContextSnapshotEntity?

    @Query("SELECT * FROM context_snapshots WHERE symptomLogId = :symptomLogId")
    fun observeBySymptomLog(symptomLogId: String): Flow<ContextSnapshotEntity?>

    @Query("SELECT * FROM context_snapshots ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecent(limit: Int): List<ContextSnapshotEntity>

    @Query("SELECT * FROM context_snapshots WHERE timestamp BETWEEN :startDate AND :endDate ORDER BY timestamp DESC")
    suspend fun getByDateRange(startDate: Long, endDate: Long): List<ContextSnapshotEntity>

    @Query("DELETE FROM context_snapshots WHERE symptomLogId = :symptomLogId")
    suspend fun deleteBySymptomLog(symptomLogId: String)
}

/**
 * Medication DAO
 */
@Dao
interface MedicationDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(medication: MedicationEntity): Long

    @Update
    suspend fun update(medication: MedicationEntity)

    @Delete
    suspend fun delete(medication: MedicationEntity)

    @Query("SELECT * FROM medications WHERE id = :id")
    suspend fun getById(id: String): MedicationEntity?

    @Query("SELECT * FROM medications WHERE id = :id")
    fun observeById(id: String): Flow<MedicationEntity?>

    @Query("SELECT * FROM medications WHERE isActive = 1 ORDER BY name")
    fun observeActive(): Flow<List<MedicationEntity>>

    @Query("SELECT * FROM medications ORDER BY isActive DESC, name")
    fun observeAll(): Flow<List<MedicationEntity>>

    @Query("SELECT * FROM medications WHERE isBiologic = 1 AND isActive = 1")
    fun observeActiveBiologics(): Flow<List<MedicationEntity>>

    @Query("UPDATE medications SET isActive = 0, endDate = :endDate WHERE id = :id")
    suspend fun deactivate(id: String, endDate: Long = System.currentTimeMillis())

    @Query("SELECT * FROM medications WHERE isSynced = 0")
    suspend fun getUnsynced(): List<MedicationEntity>
}

/**
 * DoseLog DAO
 */
@Dao
interface DoseLogDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(doseLog: DoseLogEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(doseLogs: List<DoseLogEntity>)

    @Update
    suspend fun update(doseLog: DoseLogEntity)

    @Delete
    suspend fun delete(doseLog: DoseLogEntity)

    @Query("SELECT * FROM dose_logs WHERE medicationId = :medicationId ORDER BY timestamp DESC")
    fun observeByMedication(medicationId: String): Flow<List<DoseLogEntity>>

    @Query("SELECT * FROM dose_logs WHERE timestamp BETWEEN :startDate AND :endDate ORDER BY timestamp DESC")
    fun observeByDateRange(startDate: Long, endDate: Long): Flow<List<DoseLogEntity>>

    @Query("SELECT * FROM dose_logs WHERE scheduledTime BETWEEN :startDate AND :endDate ORDER BY scheduledTime")
    suspend fun getScheduledForDateRange(startDate: Long, endDate: Long): List<DoseLogEntity>

    @Query("SELECT COUNT(*) FROM dose_logs WHERE medicationId = :medicationId AND wasSkipped = 0 AND timestamp BETWEEN :startDate AND :endDate")
    suspend fun getAdherenceCount(medicationId: String, startDate: Long, endDate: Long): Int

    @Query("SELECT COUNT(*) FROM dose_logs WHERE medicationId = :medicationId AND timestamp BETWEEN :startDate AND :endDate")
    suspend fun getTotalDosesScheduled(medicationId: String, startDate: Long, endDate: Long): Int
}

/**
 * FlareEvent DAO
 */
@Dao
interface FlareEventDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(flareEvent: FlareEventEntity): Long

    @Update
    suspend fun update(flareEvent: FlareEventEntity)

    @Delete
    suspend fun delete(flareEvent: FlareEventEntity)

    @Query("SELECT * FROM flare_events WHERE id = :id")
    suspend fun getById(id: String): FlareEventEntity?

    @Query("SELECT * FROM flare_events ORDER BY startDate DESC")
    fun observeAll(): Flow<List<FlareEventEntity>>

    @Query("SELECT * FROM flare_events WHERE isResolved = 0 ORDER BY startDate DESC")
    fun observeActive(): Flow<List<FlareEventEntity>>

    @Query("SELECT * FROM flare_events WHERE startDate BETWEEN :startDate AND :endDate ORDER BY startDate DESC")
    fun observeByDateRange(startDate: Long, endDate: Long): Flow<List<FlareEventEntity>>

    @Query("SELECT COUNT(*) FROM flare_events WHERE startDate BETWEEN :startDate AND :endDate")
    suspend fun getCountForDateRange(startDate: Long, endDate: Long): Int

    @Query("SELECT AVG(severity) FROM flare_events WHERE startDate BETWEEN :startDate AND :endDate")
    suspend fun getAverageSeverity(startDate: Long, endDate: Long): Double?

    @Query("UPDATE flare_events SET isResolved = 1, endDate = :endDate WHERE id = :id")
    suspend fun resolve(id: String, endDate: Long = System.currentTimeMillis())
}

/**
 * ExerciseSession DAO
 */
@Dao
interface ExerciseSessionDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(session: ExerciseSessionEntity): Long

    @Update
    suspend fun update(session: ExerciseSessionEntity)

    @Delete
    suspend fun delete(session: ExerciseSessionEntity)

    @Query("SELECT * FROM exercise_sessions ORDER BY timestamp DESC")
    fun observeAll(): Flow<List<ExerciseSessionEntity>>

    @Query("SELECT * FROM exercise_sessions ORDER BY timestamp DESC LIMIT :limit")
    fun observeRecent(limit: Int): Flow<List<ExerciseSessionEntity>>

    @Query("SELECT * FROM exercise_sessions WHERE timestamp BETWEEN :startDate AND :endDate ORDER BY timestamp DESC")
    fun observeByDateRange(startDate: Long, endDate: Long): Flow<List<ExerciseSessionEntity>>

    @Query("SELECT SUM(durationMinutes) FROM exercise_sessions WHERE timestamp BETWEEN :startDate AND :endDate")
    suspend fun getTotalMinutes(startDate: Long, endDate: Long): Int?

    @Query("SELECT COUNT(*) FROM exercise_sessions WHERE timestamp BETWEEN :startDate AND :endDate")
    suspend fun getSessionCount(startDate: Long, endDate: Long): Int
}

/**
 * Exercise DAO (for exercise library)
 */
@Dao
interface ExerciseDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(exercise: ExerciseEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(exercises: List<ExerciseEntity>)

    @Query("SELECT * FROM exercises WHERE id = :id")
    suspend fun getById(id: String): ExerciseEntity?

    @Query("SELECT * FROM exercises ORDER BY name")
    fun observeAll(): Flow<List<ExerciseEntity>>

    @Query("SELECT * FROM exercises WHERE category = :category ORDER BY level, name")
    fun observeByCategory(category: String): Flow<List<ExerciseEntity>>

    @Query("SELECT * FROM exercises WHERE level = 'BEGINNER' ORDER BY category, name")
    fun observeBeginner(): Flow<List<ExerciseEntity>>

    @Query("SELECT * FROM exercises WHERE name LIKE '%' || :query || '%' OR targetAreas LIKE '%' || :query || '%'")
    fun search(query: String): Flow<List<ExerciseEntity>>
}

/**
 * UserProfile DAO
 */
@Dao
interface UserProfileDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(userProfile: UserProfileEntity): Long

    @Update
    suspend fun update(userProfile: UserProfileEntity)

    @Query("SELECT * FROM user_profile WHERE id = 'user_profile'")
    suspend fun get(): UserProfileEntity?

    @Query("SELECT * FROM user_profile WHERE id = 'user_profile'")
    fun observe(): Flow<UserProfileEntity?>

    @Query("DELETE FROM user_profile")
    suspend fun delete()

    // Increment check-in streak
    @Query("UPDATE user_profile SET totalCheckIns = totalCheckIns + 1, streakDays = :streakDays, longestStreak = CASE WHEN :streakDays > longestStreak THEN :streakDays ELSE longestStreak END WHERE id = 'user_profile'")
    suspend fun incrementCheckIn(streakDays: Int)
}

/**
 * MeditationSession DAO
 */
@Dao
interface MeditationSessionDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(session: MeditationSessionEntity): Long

    @Update
    suspend fun update(session: MeditationSessionEntity)

    @Delete
    suspend fun delete(session: MeditationSessionEntity)

    @Query("SELECT * FROM meditation_sessions ORDER BY timestamp DESC")
    fun observeAll(): Flow<List<MeditationSessionEntity>>

    @Query("SELECT * FROM meditation_sessions ORDER BY timestamp DESC LIMIT :limit")
    fun observeRecent(limit: Int): Flow<List<MeditationSessionEntity>>

    @Query("SELECT * FROM meditation_sessions WHERE timestamp BETWEEN :startDate AND :endDate ORDER BY timestamp DESC")
    fun observeByDateRange(startDate: Long, endDate: Long): Flow<List<MeditationSessionEntity>>

    @Query("SELECT SUM(durationMinutes) FROM meditation_sessions WHERE timestamp BETWEEN :startDate AND :endDate")
    suspend fun getTotalMinutes(startDate: Long, endDate: Long): Int?
}

/**
 * MeditationStreak DAO
 */
@Dao
interface MeditationStreakDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(streak: MeditationStreakEntity): Long

    @Update
    suspend fun update(streak: MeditationStreakEntity)

    @Query("SELECT * FROM meditation_streaks LIMIT 1")
    suspend fun get(): MeditationStreakEntity?

    @Query("SELECT * FROM meditation_streaks LIMIT 1")
    fun observe(): Flow<MeditationStreakEntity?>
}

/**
 * TriggerLog DAO
 */
@Dao
interface TriggerLogDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(triggerLog: TriggerLogEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(triggerLogs: List<TriggerLogEntity>)

    @Query("SELECT * FROM trigger_logs WHERE symptomLogId = :symptomLogId")
    suspend fun getBySymptomLog(symptomLogId: String): List<TriggerLogEntity>

    @Query("SELECT * FROM trigger_logs WHERE timestamp BETWEEN :startDate AND :endDate ORDER BY timestamp DESC")
    suspend fun getByDateRange(startDate: Long, endDate: Long): List<TriggerLogEntity>

    @Query("SELECT triggerType, COUNT(*) as count FROM trigger_logs GROUP BY triggerType ORDER BY count DESC")
    suspend fun getTriggerCounts(): List<TriggerCount>
}

data class TriggerCount(
    val triggerType: String,
    val count: Int
)

/**
 * TriggerAnalysisCache DAO
 */
@Dao
interface TriggerAnalysisCacheDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(cache: TriggerAnalysisCacheEntity): Long

    @Query("SELECT * FROM trigger_analysis_cache WHERE isValid = 1 ORDER BY analysisDate DESC LIMIT 1")
    suspend fun getLatestValid(): TriggerAnalysisCacheEntity?

    @Query("UPDATE trigger_analysis_cache SET isValid = 0")
    suspend fun invalidateAll()

    @Query("DELETE FROM trigger_analysis_cache WHERE analysisDate < :olderThan")
    suspend fun deleteOlderThan(olderThan: Long)
}
