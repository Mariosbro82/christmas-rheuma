package com.inflamai.feature.quickcapture

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.database.entity.SymptomLogEntity
import com.inflamai.core.data.repository.FlareRepository
import com.inflamai.core.data.repository.SymptomRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.time.LocalDateTime
import java.util.UUID
import javax.inject.Inject

/**
 * Quick Capture / SOS Flare UI State
 */
data class QuickCaptureUiState(
    val painLevel: Int = 5,
    val selectedSymptoms: Set<String> = emptySet(),
    val selectedRegions: Set<String> = emptySet(),
    val notes: String = "",
    val timestamp: LocalDateTime = LocalDateTime.now(),
    val isSaving: Boolean = false,
    val isComplete: Boolean = false,
    val error: String? = null
)

/**
 * Quick Capture ViewModel
 * Manages fast symptom logging during flares
 */
@HiltViewModel
class QuickCaptureViewModel @Inject constructor(
    private val flareRepository: FlareRepository,
    private val symptomRepository: SymptomRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(QuickCaptureUiState())
    val uiState: StateFlow<QuickCaptureUiState> = _uiState.asStateFlow()

    fun updatePainLevel(level: Int) {
        _uiState.value = _uiState.value.copy(painLevel = level.coerceIn(0, 10))
    }

    fun toggleSymptom(symptomId: String) {
        val currentSymptoms = _uiState.value.selectedSymptoms.toMutableSet()
        if (currentSymptoms.contains(symptomId)) {
            currentSymptoms.remove(symptomId)
        } else {
            currentSymptoms.add(symptomId)
        }
        _uiState.value = _uiState.value.copy(selectedSymptoms = currentSymptoms)
    }

    fun toggleRegion(regionId: String) {
        val currentRegions = _uiState.value.selectedRegions.toMutableSet()
        if (currentRegions.contains(regionId)) {
            currentRegions.remove(regionId)
        } else {
            currentRegions.add(regionId)
        }
        _uiState.value = _uiState.value.copy(selectedRegions = currentRegions)
    }

    fun updateNotes(notes: String) {
        _uiState.value = _uiState.value.copy(notes = notes)
    }

    // Legacy method for compatibility
    fun logQuickSymptom(symptom: String) {
        toggleSymptom(symptom)
    }

    fun saveQuickCapture() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isSaving = true, error = null)
            try {
                val state = _uiState.value

                // Create FlareEvent using repository
                flareRepository.logQuickFlare(
                    severity = state.painLevel,
                    symptoms = state.selectedSymptoms.toList(),
                    affectedRegions = state.selectedRegions.toList(),
                    notes = state.notes.ifBlank { null }
                )

                // Also create SymptomLog with flare flag
                val symptomLog = SymptomLogEntity(
                    id = UUID.randomUUID().toString(),
                    timestamp = System.currentTimeMillis(),
                    isFlareEvent = true,
                    notes = state.notes.ifBlank { null }
                )
                symptomRepository.insertSymptomLog(symptomLog)

                _uiState.value = _uiState.value.copy(
                    isSaving = false,
                    isComplete = true
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isSaving = false,
                    error = e.message ?: "Failed to save flare log"
                )
            }
        }
    }

    fun clearError() {
        _uiState.value = _uiState.value.copy(error = null)
    }

    fun reset() {
        _uiState.value = QuickCaptureUiState()
    }
}
