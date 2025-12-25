package com.inflamai.core.data.database.dao

import androidx.room.*
import com.inflamai.core.data.database.entity.SymptomLogEntity
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for SymptomLog operations
 */
@Dao
interface SymptomLogDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(symptomLog: SymptomLogEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(symptomLogs: List<SymptomLogEntity>)

    @Update
    suspend fun update(symptomLog: SymptomLogEntity)

    @Delete
    suspend fun delete(symptomLog: SymptomLogEntity)

    @Query("DELETE FROM symptom_logs WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("DELETE FROM symptom_logs")
    suspend fun deleteAll()

    @Query("SELECT * FROM symptom_logs WHERE id = :id")
    suspend fun getById(id: String): SymptomLogEntity?

    @Query("SELECT * FROM symptom_logs WHERE id = :id")
    fun observeById(id: String): Flow<SymptomLogEntity?>

    @Query("SELECT * FROM symptom_logs ORDER BY timestamp DESC")
    fun observeAll(): Flow<List<SymptomLogEntity>>

    @Query("SELECT * FROM symptom_logs ORDER BY timestamp DESC LIMIT :limit")
    fun observeRecent(limit: Int): Flow<List<SymptomLogEntity>>

    @Query("SELECT * FROM symptom_logs ORDER BY timestamp DESC LIMIT 1")
    suspend fun getLatest(): SymptomLogEntity?

    @Query("SELECT * FROM symptom_logs ORDER BY timestamp DESC LIMIT 1")
    fun observeLatest(): Flow<SymptomLogEntity?>

    // Date range queries
    @Query("SELECT * FROM symptom_logs WHERE timestamp BETWEEN :startDate AND :endDate ORDER BY timestamp DESC")
    fun observeByDateRange(startDate: Long, endDate: Long): Flow<List<SymptomLogEntity>>

    @Query("SELECT * FROM symptom_logs WHERE timestamp BETWEEN :startDate AND :endDate ORDER BY timestamp DESC")
    suspend fun getByDateRange(startDate: Long, endDate: Long): List<SymptomLogEntity>

    // Today's log
    @Query("SELECT * FROM symptom_logs WHERE timestamp >= :startOfDay ORDER BY timestamp DESC LIMIT 1")
    suspend fun getTodaysLog(startOfDay: Long): SymptomLogEntity?

    @Query("SELECT * FROM symptom_logs WHERE timestamp >= :startOfDay ORDER BY timestamp DESC LIMIT 1")
    fun observeTodaysLog(startOfDay: Long): Flow<SymptomLogEntity?>

    // Flare events
    @Query("SELECT * FROM symptom_logs WHERE isFlareEvent = 1 ORDER BY timestamp DESC")
    fun observeFlareEvents(): Flow<List<SymptomLogEntity>>

    // Statistics
    @Query("SELECT COUNT(*) FROM symptom_logs")
    suspend fun getCount(): Int

    @Query("SELECT COUNT(*) FROM symptom_logs WHERE timestamp >= :since")
    suspend fun getCountSince(since: Long): Int

    @Query("SELECT AVG(basdaiScore) FROM symptom_logs WHERE timestamp BETWEEN :startDate AND :endDate")
    suspend fun getAverageBASDAI(startDate: Long, endDate: Long): Double?

    @Query("SELECT AVG(asdasScore) FROM symptom_logs WHERE timestamp BETWEEN :startDate AND :endDate AND asdasScore > 0")
    suspend fun getAverageASDAS(startDate: Long, endDate: Long): Double?

    @Query("SELECT AVG(fatigueLevel) FROM symptom_logs WHERE timestamp BETWEEN :startDate AND :endDate")
    suspend fun getAverageFatigue(startDate: Long, endDate: Long): Double?

    @Query("SELECT AVG(morningStiffnessMinutes) FROM symptom_logs WHERE timestamp BETWEEN :startDate AND :endDate")
    suspend fun getAverageMorningStiffness(startDate: Long, endDate: Long): Double?

    // For sync
    @Query("SELECT * FROM symptom_logs WHERE isSynced = 0")
    suspend fun getUnsynced(): List<SymptomLogEntity>

    @Query("UPDATE symptom_logs SET isSynced = 1 WHERE id IN (:ids)")
    suspend fun markAsSynced(ids: List<String>)
}
