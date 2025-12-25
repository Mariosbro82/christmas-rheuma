package com.inflamai.feature.bodymap.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.database.dao.BodyRegionLogDao
import com.inflamai.core.data.database.dao.SymptomLogDao
import com.inflamai.core.data.database.entity.BodyRegion
import com.inflamai.core.data.database.entity.BodyRegionCategory
import com.inflamai.core.data.database.entity.BodyRegionLogEntity
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.temporal.ChronoUnit
import java.util.UUID
import javax.inject.Inject

/**
 * Body Map ViewModel
 *
 * Manages the interactive 47-region body map:
 * - Spine: C1-C7, T1-T12, L1-L5, Sacrum, SI joints
 * - Upper: Shoulders, Elbows, Wrists, Hands (bilateral)
 * - Lower: Hips, Knees, Ankles, Feet (bilateral)
 * - Thorax: Chest, Ribs (bilateral)
 *
 * Features:
 * - Real-time pain heatmap overlay
 * - 7/30/90-day average visualization
 * - Individual region pain logging
 * - Accessibility with anatomical names
 */
@HiltViewModel
class BodyMapViewModel @Inject constructor(
    private val bodyRegionLogDao: BodyRegionLogDao,
    private val symptomLogDao: SymptomLogDao
) : ViewModel() {

    private val _uiState = MutableStateFlow(BodyMapUiState())
    val uiState: StateFlow<BodyMapUiState> = _uiState.asStateFlow()

    private val _selectedRegion = MutableStateFlow<BodyRegion?>(null)
    val selectedRegion: StateFlow<BodyRegion?> = _selectedRegion.asStateFlow()

    init {
        loadRegionData()
    }

    private fun loadRegionData() {
        viewModelScope.launch {
            // Load pain averages for all regions
            val now = Instant.now()
            val sevenDaysAgo = now.minus(7, ChronoUnit.DAYS).toEpochMilli()
            val thirtyDaysAgo = now.minus(30, ChronoUnit.DAYS).toEpochMilli()
            val ninetyDaysAgo = now.minus(90, ChronoUnit.DAYS).toEpochMilli()
            val nowMillis = now.toEpochMilli()

            val regionPainMap = mutableMapOf<String, RegionPainData>()

            BodyRegion.entries.forEach { region ->
                val avg7Day = bodyRegionLogDao.getAveragePainForRegion(
                    region.id, sevenDaysAgo, nowMillis
                ) ?: 0.0

                val avg30Day = bodyRegionLogDao.getAveragePainForRegion(
                    region.id, thirtyDaysAgo, nowMillis
                ) ?: 0.0

                val avg90Day = bodyRegionLogDao.getAveragePainForRegion(
                    region.id, ninetyDaysAgo, nowMillis
                ) ?: 0.0

                regionPainMap[region.id] = RegionPainData(
                    regionId = region.id,
                    averagePain7Day = avg7Day,
                    averagePain30Day = avg30Day,
                    averagePain90Day = avg90Day,
                    currentPainLevel = null
                )
            }

            _uiState.update { state ->
                state.copy(
                    isLoading = false,
                    regionPainData = regionPainMap
                )
            }
        }
    }

    fun selectRegion(region: BodyRegion) {
        _selectedRegion.value = region
        _uiState.update { state ->
            state.copy(selectedRegionId = region.id)
        }

        // Load region history
        viewModelScope.launch {
            bodyRegionLogDao.observeByRegion(region.id).collect { logs ->
                _uiState.update { state ->
                    state.copy(selectedRegionHistory = logs.take(10))
                }
            }
        }
    }

    fun clearSelection() {
        _selectedRegion.value = null
        _uiState.update { state ->
            state.copy(
                selectedRegionId = null,
                selectedRegionHistory = emptyList()
            )
        }
    }

    fun updatePainLevel(regionId: String, painLevel: Int) {
        _uiState.update { state ->
            val updatedData = state.regionPainData.toMutableMap()
            updatedData[regionId] = updatedData[regionId]?.copy(
                currentPainLevel = painLevel
            ) ?: RegionPainData(
                regionId = regionId,
                currentPainLevel = painLevel
            )
            state.copy(
                regionPainData = updatedData,
                hasUnsavedChanges = true
            )
        }
    }

    fun updateStiffness(regionId: String, stiffnessMinutes: Int) {
        _uiState.update { state ->
            val updatedData = state.regionPainData.toMutableMap()
            updatedData[regionId] = updatedData[regionId]?.copy(
                stiffnessMinutes = stiffnessMinutes
            ) ?: RegionPainData(
                regionId = regionId,
                stiffnessMinutes = stiffnessMinutes
            )
            state.copy(
                regionPainData = updatedData,
                hasUnsavedChanges = true
            )
        }
    }

    fun toggleSwelling(regionId: String) {
        _uiState.update { state ->
            val current = state.regionPainData[regionId]
            val updatedData = state.regionPainData.toMutableMap()
            updatedData[regionId] = current?.copy(
                hasSwelling = !(current.hasSwelling ?: false)
            ) ?: RegionPainData(regionId = regionId, hasSwelling = true)
            state.copy(
                regionPainData = updatedData,
                hasUnsavedChanges = true
            )
        }
    }

    fun toggleWarmth(regionId: String) {
        _uiState.update { state ->
            val current = state.regionPainData[regionId]
            val updatedData = state.regionPainData.toMutableMap()
            updatedData[regionId] = current?.copy(
                hasWarmth = !(current.hasWarmth ?: false)
            ) ?: RegionPainData(regionId = regionId, hasWarmth = true)
            state.copy(
                regionPainData = updatedData,
                hasUnsavedChanges = true
            )
        }
    }

    fun setTimeRange(range: TimeRange) {
        _uiState.update { it.copy(selectedTimeRange = range) }
    }

    fun setViewMode(mode: BodyMapViewMode) {
        _uiState.update { it.copy(viewMode = mode) }
    }

    fun saveRegionData(symptomLogId: String? = null) {
        viewModelScope.launch {
            _uiState.update { it.copy(isSaving = true) }

            try {
                val timestamp = System.currentTimeMillis()
                val logId = symptomLogId ?: symptomLogDao.getLatest()?.id ?: UUID.randomUUID().toString()

                val logsToSave = _uiState.value.regionPainData
                    .filter { it.value.currentPainLevel != null && it.value.currentPainLevel!! > 0 }
                    .map { (regionId, data) ->
                        val region = BodyRegion.fromId(regionId)
                        BodyRegionLogEntity(
                            symptomLogId = logId,
                            timestamp = timestamp,
                            regionId = regionId,
                            regionName = region?.displayName ?: regionId,
                            painLevel = data.currentPainLevel ?: 0,
                            stiffnessDuration = data.stiffnessMinutes ?: 0,
                            hasSwelling = data.hasSwelling ?: false,
                            hasWarmth = data.hasWarmth ?: false
                        )
                    }

                if (logsToSave.isNotEmpty()) {
                    bodyRegionLogDao.insertAll(logsToSave)
                }

                _uiState.update { state ->
                    state.copy(
                        isSaving = false,
                        hasUnsavedChanges = false
                    )
                }

                // Reload data
                loadRegionData()

            } catch (e: Exception) {
                _uiState.update { state ->
                    state.copy(
                        isSaving = false,
                        error = "Failed to save: ${e.message}"
                    )
                }
            }
        }
    }

    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }
}

data class BodyMapUiState(
    val isLoading: Boolean = true,
    val isSaving: Boolean = false,
    val hasUnsavedChanges: Boolean = false,
    val regionPainData: Map<String, RegionPainData> = emptyMap(),
    val selectedRegionId: String? = null,
    val selectedRegionHistory: List<BodyRegionLogEntity> = emptyList(),
    val selectedTimeRange: TimeRange = TimeRange.DAYS_7,
    val viewMode: BodyMapViewMode = BodyMapViewMode.FRONT,
    val error: String? = null
)

data class RegionPainData(
    val regionId: String,
    val averagePain7Day: Double = 0.0,
    val averagePain30Day: Double = 0.0,
    val averagePain90Day: Double = 0.0,
    val currentPainLevel: Int? = null,
    val stiffnessMinutes: Int? = null,
    val hasSwelling: Boolean? = null,
    val hasWarmth: Boolean? = null
)

enum class TimeRange(val label: String, val days: Int) {
    DAYS_7("7 Days", 7),
    DAYS_30("30 Days", 30),
    DAYS_90("90 Days", 90)
}

enum class BodyMapViewMode {
    FRONT,
    BACK,
    SPINE
}
