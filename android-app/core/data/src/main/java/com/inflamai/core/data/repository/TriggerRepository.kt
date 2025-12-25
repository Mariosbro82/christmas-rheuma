package com.inflamai.core.data.repository

import com.inflamai.core.data.database.dao.TriggerAnalysisCacheDao
import com.inflamai.core.data.database.dao.TriggerCount
import com.inflamai.core.data.database.dao.TriggerLogDao
import com.inflamai.core.data.database.entity.TriggerAnalysisCacheEntity
import com.inflamai.core.data.database.entity.TriggerLogEntity
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for trigger pattern analysis data
 * Abstracts TriggerLogDao and TriggerAnalysisCacheDao
 */
@Singleton
class TriggerRepository @Inject constructor(
    private val triggerLogDao: TriggerLogDao,
    private val triggerAnalysisCacheDao: TriggerAnalysisCacheDao
) {
    // Trigger Logs
    suspend fun getTriggersBySymptomLog(symptomLogId: String): List<TriggerLogEntity> =
        triggerLogDao.getBySymptomLog(symptomLogId)

    suspend fun getTriggersByDateRange(startDate: Long, endDate: Long): List<TriggerLogEntity> =
        triggerLogDao.getByDateRange(startDate, endDate)

    suspend fun getTriggerCounts(): List<TriggerCount> = triggerLogDao.getTriggerCounts()

    suspend fun insertTriggerLog(triggerLog: TriggerLogEntity): Long =
        triggerLogDao.insert(triggerLog)

    suspend fun insertTriggerLogs(triggerLogs: List<TriggerLogEntity>) =
        triggerLogDao.insertAll(triggerLogs)

    // Analysis Cache
    suspend fun getLatestValidAnalysis(): TriggerAnalysisCacheEntity? =
        triggerAnalysisCacheDao.getLatestValid()

    suspend fun cacheAnalysis(cache: TriggerAnalysisCacheEntity): Long =
        triggerAnalysisCacheDao.insert(cache)

    suspend fun invalidateAllAnalyses() = triggerAnalysisCacheDao.invalidateAll()

    suspend fun cleanupOldAnalyses(olderThan: Long) =
        triggerAnalysisCacheDao.deleteOlderThan(olderThan)

    // Check if we need to recompute analysis
    suspend fun isAnalysisCacheValid(): Boolean {
        val latest = triggerAnalysisCacheDao.getLatestValid()
        if (latest == null) return false

        // Cache is valid for 24 hours
        val maxAge = 24 * 60 * 60 * 1000L
        return (System.currentTimeMillis() - latest.analysisDate) < maxAge
    }
}
