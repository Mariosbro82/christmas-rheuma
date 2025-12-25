package com.inflamai.core.data.repository

import com.inflamai.core.data.database.dao.BodyRegionLogDao
import com.inflamai.core.data.database.dao.ContextSnapshotDao
import com.inflamai.core.data.database.dao.SymptomLogDao
import com.inflamai.core.data.database.entity.BodyRegionLogEntity
import com.inflamai.core.data.database.entity.ContextSnapshotEntity
import com.inflamai.core.data.database.entity.SymptomLogEntity
import kotlinx.coroutines.flow.Flow
import java.time.LocalDate
import java.time.ZoneId
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for symptom tracking data
 * Abstracts SymptomLogDao, BodyRegionLogDao, and ContextSnapshotDao
 */
@Singleton
class SymptomRepository @Inject constructor(
    private val symptomLogDao: SymptomLogDao,
    private val bodyRegionLogDao: BodyRegionLogDao,
    private val contextSnapshotDao: ContextSnapshotDao
) {
    // Symptom Logs
    fun observeAllSymptomLogs(): Flow<List<SymptomLogEntity>> = symptomLogDao.observeAll()

    fun observeRecentSymptomLogs(limit: Int): Flow<List<SymptomLogEntity>> =
        symptomLogDao.observeRecent(limit)

    fun observeLatestSymptomLog(): Flow<SymptomLogEntity?> = symptomLogDao.observeLatest()

    fun observeSymptomLogsByDateRange(startDate: Long, endDate: Long): Flow<List<SymptomLogEntity>> =
        symptomLogDao.observeByDateRange(startDate, endDate)

    fun observeFlareEvents(): Flow<List<SymptomLogEntity>> = symptomLogDao.observeFlareEvents()

    suspend fun getSymptomLogById(id: String): SymptomLogEntity? = symptomLogDao.getById(id)

    suspend fun getLatestSymptomLog(): SymptomLogEntity? = symptomLogDao.getLatest()

    suspend fun getTodaysLog(): SymptomLogEntity? {
        val startOfDay = LocalDate.now()
            .atStartOfDay(ZoneId.systemDefault())
            .toInstant()
            .toEpochMilli()
        return symptomLogDao.getTodaysLog(startOfDay)
    }

    fun observeTodaysLog(): Flow<SymptomLogEntity?> {
        val startOfDay = LocalDate.now()
            .atStartOfDay(ZoneId.systemDefault())
            .toInstant()
            .toEpochMilli()
        return symptomLogDao.observeTodaysLog(startOfDay)
    }

    suspend fun getSymptomLogsByDateRange(startDate: Long, endDate: Long): List<SymptomLogEntity> =
        symptomLogDao.getByDateRange(startDate, endDate)

    suspend fun insertSymptomLog(symptomLog: SymptomLogEntity): Long =
        symptomLogDao.insert(symptomLog)

    suspend fun updateSymptomLog(symptomLog: SymptomLogEntity) =
        symptomLogDao.update(symptomLog)

    suspend fun deleteSymptomLog(symptomLog: SymptomLogEntity) =
        symptomLogDao.delete(symptomLog)

    suspend fun deleteSymptomLogById(id: String) = symptomLogDao.deleteById(id)

    // Statistics
    suspend fun getSymptomLogCount(): Int = symptomLogDao.getCount()

    suspend fun getAverageBASDAI(startDate: Long, endDate: Long): Double? =
        symptomLogDao.getAverageBASDAI(startDate, endDate)

    suspend fun getAverageASDAS(startDate: Long, endDate: Long): Double? =
        symptomLogDao.getAverageASDAS(startDate, endDate)

    suspend fun getAverageFatigue(startDate: Long, endDate: Long): Double? =
        symptomLogDao.getAverageFatigue(startDate, endDate)

    suspend fun getAverageMorningStiffness(startDate: Long, endDate: Long): Double? =
        symptomLogDao.getAverageMorningStiffness(startDate, endDate)

    // Body Region Logs
    fun observeBodyRegionsBySymptomLog(symptomLogId: String): Flow<List<BodyRegionLogEntity>> =
        bodyRegionLogDao.observeBySymptomLog(symptomLogId)

    fun observeBodyRegionHistory(regionId: String): Flow<List<BodyRegionLogEntity>> =
        bodyRegionLogDao.observeByRegion(regionId)

    suspend fun getBodyRegionsBySymptomLog(symptomLogId: String): List<BodyRegionLogEntity> =
        bodyRegionLogDao.getBySymptomLog(symptomLogId)

    suspend fun getAveragePainForRegion(regionId: String, startDate: Long, endDate: Long): Double? =
        bodyRegionLogDao.getAveragePainForRegion(regionId, startDate, endDate)

    suspend fun insertBodyRegionLog(bodyRegionLog: BodyRegionLogEntity): Long =
        bodyRegionLogDao.insert(bodyRegionLog)

    suspend fun insertBodyRegionLogs(bodyRegionLogs: List<BodyRegionLogEntity>) =
        bodyRegionLogDao.insertAll(bodyRegionLogs)

    suspend fun deleteBodyRegionLogsBySymptomLog(symptomLogId: String) =
        bodyRegionLogDao.deleteBySymptomLog(symptomLogId)

    // Context Snapshots
    fun observeContextBySymptomLog(symptomLogId: String): Flow<ContextSnapshotEntity?> =
        contextSnapshotDao.observeBySymptomLog(symptomLogId)

    suspend fun getContextBySymptomLog(symptomLogId: String): ContextSnapshotEntity? =
        contextSnapshotDao.getBySymptomLog(symptomLogId)

    suspend fun getRecentContextSnapshots(limit: Int): List<ContextSnapshotEntity> =
        contextSnapshotDao.getRecent(limit)

    suspend fun getContextByDateRange(startDate: Long, endDate: Long): List<ContextSnapshotEntity> =
        contextSnapshotDao.getByDateRange(startDate, endDate)

    suspend fun insertContext(contextSnapshot: ContextSnapshotEntity): Long =
        contextSnapshotDao.insert(contextSnapshot)

    suspend fun deleteContextBySymptomLog(symptomLogId: String) =
        contextSnapshotDao.deleteBySymptomLog(symptomLogId)

    // Combined save operation for a complete symptom log entry
    suspend fun saveCompleteSymptomLog(
        symptomLog: SymptomLogEntity,
        bodyRegionLogs: List<BodyRegionLogEntity>,
        contextSnapshot: ContextSnapshotEntity?
    ): Long {
        val logId = symptomLogDao.insert(symptomLog)

        if (bodyRegionLogs.isNotEmpty()) {
            val logsWithId = bodyRegionLogs.map { it.copy(symptomLogId = symptomLog.id) }
            bodyRegionLogDao.insertAll(logsWithId)
        }

        contextSnapshot?.let {
            contextSnapshotDao.insert(it.copy(symptomLogId = symptomLog.id))
        }

        return logId
    }
}
