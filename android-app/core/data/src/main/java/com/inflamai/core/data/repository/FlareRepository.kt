package com.inflamai.core.data.repository

import com.inflamai.core.data.database.dao.FlareEventDao
import com.inflamai.core.data.database.entity.FlareEventEntity
import kotlinx.coroutines.flow.Flow
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for flare event data
 * Abstracts FlareEventDao
 */
@Singleton
class FlareRepository @Inject constructor(
    private val flareEventDao: FlareEventDao
) {
    fun observeAllFlares(): Flow<List<FlareEventEntity>> = flareEventDao.observeAll()

    fun observeActiveFlares(): Flow<List<FlareEventEntity>> = flareEventDao.observeActive()

    fun observeFlaresByDateRange(startDate: Long, endDate: Long): Flow<List<FlareEventEntity>> =
        flareEventDao.observeByDateRange(startDate, endDate)

    suspend fun getFlareById(id: String): FlareEventEntity? = flareEventDao.getById(id)

    suspend fun getFlareCount(startDate: Long, endDate: Long): Int =
        flareEventDao.getCountForDateRange(startDate, endDate)

    suspend fun getAverageSeverity(startDate: Long, endDate: Long): Double? =
        flareEventDao.getAverageSeverity(startDate, endDate)

    suspend fun insertFlare(flare: FlareEventEntity): Long = flareEventDao.insert(flare)

    suspend fun updateFlare(flare: FlareEventEntity) = flareEventDao.update(flare)

    suspend fun deleteFlare(flare: FlareEventEntity) = flareEventDao.delete(flare)

    suspend fun resolveFlare(id: String, endDate: Long = System.currentTimeMillis()) =
        flareEventDao.resolve(id, endDate)

    // Quick flare logging
    suspend fun logQuickFlare(
        severity: Int,
        suspectedTriggers: List<String> = emptyList(),
        affectedRegions: List<String> = emptyList(),
        notes: String? = null
    ): Long {
        val flare = FlareEventEntity(
            id = UUID.randomUUID().toString(),
            startDate = System.currentTimeMillis(),
            severity = severity,
            suspectedTriggersJson = "[${suspectedTriggers.joinToString(",") { "\"$it\"" }}]",
            primaryRegionsJson = "[${affectedRegions.joinToString(",") { "\"$it\"" }}]",
            notes = notes,
            isResolved = false
        )
        return flareEventDao.insert(flare)
    }
}
