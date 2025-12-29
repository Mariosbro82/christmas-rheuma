package com.inflamai.core.data.repository

import com.inflamai.core.data.database.dao.DoseLogDao
import com.inflamai.core.data.database.dao.MedicationDao
import com.inflamai.core.data.database.entity.DoseLogEntity
import com.inflamai.core.data.database.entity.MedicationEntity
import com.inflamai.core.data.database.entity.SkipReason
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for medication tracking data
 * Abstracts MedicationDao and DoseLogDao
 */
@Singleton
class MedicationRepository @Inject constructor(
    private val medicationDao: MedicationDao,
    private val doseLogDao: DoseLogDao
) {
    // Medications
    fun observeAllMedications(): Flow<List<MedicationEntity>> = medicationDao.observeAll()

    fun observeActiveMedications(): Flow<List<MedicationEntity>> = medicationDao.observeActive()

    fun observeActiveBiologics(): Flow<List<MedicationEntity>> = medicationDao.observeActiveBiologics()

    fun observeMedicationById(id: String): Flow<MedicationEntity?> = medicationDao.observeById(id)

    suspend fun getMedicationById(id: String): MedicationEntity? = medicationDao.getById(id)

    suspend fun insertMedication(medication: MedicationEntity): Long =
        medicationDao.insert(medication)

    suspend fun updateMedication(medication: MedicationEntity) =
        medicationDao.update(medication)

    suspend fun deleteMedication(medication: MedicationEntity) =
        medicationDao.delete(medication)

    suspend fun deactivateMedication(id: String, endDate: Long = System.currentTimeMillis()) =
        medicationDao.deactivate(id, endDate)

    // Dose Logs
    fun observeDoseLogsByMedication(medicationId: String): Flow<List<DoseLogEntity>> =
        doseLogDao.observeByMedication(medicationId)

    fun observeDoseLogsByDateRange(startDate: Long, endDate: Long): Flow<List<DoseLogEntity>> =
        doseLogDao.observeByDateRange(startDate, endDate)

    suspend fun getScheduledDoses(startDate: Long, endDate: Long): List<DoseLogEntity> =
        doseLogDao.getScheduledForDateRange(startDate, endDate)

    suspend fun insertDoseLog(doseLog: DoseLogEntity): Long = doseLogDao.insert(doseLog)

    suspend fun insertDoseLogs(doseLogs: List<DoseLogEntity>) = doseLogDao.insertAll(doseLogs)

    suspend fun updateDoseLog(doseLog: DoseLogEntity) = doseLogDao.update(doseLog)

    suspend fun deleteDoseLog(doseLog: DoseLogEntity) = doseLogDao.delete(doseLog)

    // Adherence calculation
    suspend fun calculateAdherence(medicationId: String, startDate: Long, endDate: Long): Float {
        val takenCount = doseLogDao.getAdherenceCount(medicationId, startDate, endDate)
        val totalScheduled = doseLogDao.getTotalDosesScheduled(medicationId, startDate, endDate)

        return if (totalScheduled > 0) {
            takenCount.toFloat() / totalScheduled
        } else {
            0f
        }
    }

    // Log a dose taken now
    suspend fun logDoseTaken(
        medicationId: String,
        dosage: Double,
        notes: String? = null
    ): Long {
        val doseLog = DoseLogEntity(
            medicationId = medicationId,
            timestamp = System.currentTimeMillis(),
            scheduledTime = System.currentTimeMillis(),
            actualTakenTime = System.currentTimeMillis(),
            dosageTaken = dosage,
            wasSkipped = false,
            notes = notes
        )
        return doseLogDao.insert(doseLog)
    }

    // Log a skipped dose
    suspend fun logDoseSkipped(
        medicationId: String,
        scheduledTime: Long,
        reason: SkipReason? = null
    ): Long {
        val doseLog = DoseLogEntity(
            medicationId = medicationId,
            timestamp = System.currentTimeMillis(),
            scheduledTime = scheduledTime,
            dosageTaken = 0.0,
            wasSkipped = true,
            skipReason = reason
        )
        return doseLogDao.insert(doseLog)
    }
}
