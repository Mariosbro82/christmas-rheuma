package com.inflamai.feature.medication.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.database.dao.MedicationDao
import com.inflamai.core.data.database.dao.DoseLogDao
import com.inflamai.core.data.database.entity.MedicationEntity
import com.inflamai.core.data.database.entity.DoseLogEntity
import com.inflamai.core.data.database.entity.MedicationFrequency
import com.inflamai.core.data.database.entity.MedicationCategory
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneId
import java.util.UUID
import javax.inject.Inject

/**
 * ViewModel for Medication tracking feature
 *
 * Manages medication list, adherence tracking, and reminders.
 */
@HiltViewModel
class MedicationViewModel @Inject constructor(
    private val medicationDao: MedicationDao,
    private val doseLogDao: DoseLogDao
) : ViewModel() {

    private val _uiState = MutableStateFlow(MedicationUiState())
    val uiState: StateFlow<MedicationUiState> = _uiState.asStateFlow()

    private val _showAddDialog = MutableStateFlow(false)
    val showAddDialog: StateFlow<Boolean> = _showAddDialog.asStateFlow()

    private val _editingMedication = MutableStateFlow<MedicationEntity?>(null)
    val editingMedication: StateFlow<MedicationEntity?> = _editingMedication.asStateFlow()

    init {
        loadMedications()
        loadTodaysDoses()
        calculateAdherence()
    }

    private fun loadMedications() {
        viewModelScope.launch {
            medicationDao.observeActive().collect { medications ->
                _uiState.update { it.copy(medications = medications) }
            }
        }
    }

    private fun loadTodaysDoses() {
        viewModelScope.launch {
            val today = LocalDate.now()
            val startOfDay = today.atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()
            val endOfDay = today.plusDays(1).atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()

            doseLogDao.observeByDateRange(startOfDay, endOfDay).collect { logs ->
                _uiState.update { state ->
                    state.copy(
                        todaysDoses = logs,
                        takenMedicationIds = logs.filter { !it.wasSkipped }.map { it.medicationId }.toSet()
                    )
                }
            }
        }
    }

    private fun calculateAdherence() {
        viewModelScope.launch {
            val thirtyDaysAgo = System.currentTimeMillis() - (30L * 24 * 60 * 60 * 1000)
            val now = System.currentTimeMillis()

            doseLogDao.observeByDateRange(thirtyDaysAgo, now).collect { logs ->
                if (logs.isEmpty()) {
                    _uiState.update { it.copy(adherenceRate = 0f) }
                    return@collect
                }

                val taken = logs.count { !it.wasSkipped }
                val total = logs.size
                val rate = taken.toFloat() / total.toFloat()

                _uiState.update { it.copy(adherenceRate = rate) }
            }
        }
    }

    fun showAddMedicationDialog() {
        _editingMedication.value = null
        _showAddDialog.value = true
    }

    fun showEditMedicationDialog(medication: MedicationEntity) {
        _editingMedication.value = medication
        _showAddDialog.value = true
    }

    fun dismissDialog() {
        _showAddDialog.value = false
        _editingMedication.value = null
    }

    fun saveMedication(
        name: String,
        dosage: Double,
        dosageUnit: String,
        frequency: MedicationFrequency,
        category: MedicationCategory,
        instructions: String?,
        reminderTimes: List<LocalTime>
    ) {
        viewModelScope.launch {
            val existing = _editingMedication.value

            // Convert reminder times to JSON format
            val reminderTimesJson = reminderTimes.joinToString(",", "[", "]") { "\"${it}\"" }

            val medication = MedicationEntity(
                id = existing?.id ?: UUID.randomUUID().toString(),
                name = name,
                dosage = dosage,
                dosageUnit = dosageUnit,
                frequency = frequency,
                category = category,
                instructions = instructions,
                reminderEnabled = reminderTimes.isNotEmpty(),
                reminderTimesJson = reminderTimesJson,
                startDate = existing?.startDate ?: System.currentTimeMillis(),
                endDate = null,
                isActive = true,
                prescribedBy = existing?.prescribedBy,
                createdAt = existing?.createdAt ?: System.currentTimeMillis(),
                lastModified = System.currentTimeMillis()
            )

            if (existing != null) {
                medicationDao.update(medication)
            } else {
                medicationDao.insert(medication)
            }

            dismissDialog()
        }
    }

    fun toggleMedicationTaken(medication: MedicationEntity) {
        viewModelScope.launch {
            val today = LocalDate.now()
            val startOfDay = today.atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()
            val endOfDay = today.plusDays(1).atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()

            // Check if already logged today
            val existingLogs = _uiState.value.todaysDoses
                .filter { it.medicationId == medication.id }

            if (existingLogs.isEmpty()) {
                // Create new dose log
                val now = System.currentTimeMillis()
                val doseLog = DoseLogEntity(
                    id = UUID.randomUUID().toString(),
                    medicationId = medication.id,
                    timestamp = now,
                    scheduledTime = now,
                    actualTakenTime = now,
                    dosageTaken = medication.dosage,
                    wasSkipped = false,
                    wasTakenLate = false,
                    skipReason = null,
                    notes = null,
                    lastModified = now
                )
                doseLogDao.insert(doseLog)
            } else {
                // Toggle existing log
                val log = existingLogs.first()
                val wasTaken = !log.wasSkipped
                doseLogDao.update(log.copy(
                    wasSkipped = wasTaken,
                    actualTakenTime = if (!wasTaken) System.currentTimeMillis() else null,
                    lastModified = System.currentTimeMillis()
                ))
            }
        }
    }

    fun skipDose(medication: MedicationEntity, reason: String?) {
        viewModelScope.launch {
            val now = System.currentTimeMillis()
            val doseLog = DoseLogEntity(
                id = UUID.randomUUID().toString(),
                medicationId = medication.id,
                timestamp = now,
                scheduledTime = now,
                actualTakenTime = null,
                dosageTaken = medication.dosage,
                wasSkipped = true,
                wasTakenLate = false,
                skipReason = null, // Would need to convert String to SkipReason enum
                skipReasonOther = reason,
                notes = null,
                lastModified = now
            )
            doseLogDao.insert(doseLog)
        }
    }

    fun deleteMedication(medication: MedicationEntity) {
        viewModelScope.launch {
            // Soft delete - mark as inactive
            medicationDao.update(medication.copy(
                isActive = false,
                endDate = System.currentTimeMillis(),
                lastModified = System.currentTimeMillis()
            ))
        }
    }

    fun getNextDoseTime(medication: MedicationEntity): LocalTime? {
        // Parse reminder times from JSON
        val timesJson = medication.reminderTimesJson
        if (timesJson == "[]" || timesJson.isBlank()) return null

        try {
            val times = timesJson
                .removeSurrounding("[", "]")
                .split(",")
                .mapNotNull { timeStr ->
                    val cleaned = timeStr.trim().removeSurrounding("\"")
                    if (cleaned.isNotBlank()) LocalTime.parse(cleaned) else null
                }

            val now = LocalTime.now()
            return times.filter { it.isAfter(now) }.minByOrNull { it }
                ?: times.minByOrNull { it }
        } catch (e: Exception) {
            return null
        }
    }
}

data class MedicationUiState(
    val medications: List<MedicationEntity> = emptyList(),
    val todaysDoses: List<DoseLogEntity> = emptyList(),
    val takenMedicationIds: Set<String> = emptySet(),
    val adherenceRate: Float = 0f,
    val isLoading: Boolean = false,
    val error: String? = null
)
