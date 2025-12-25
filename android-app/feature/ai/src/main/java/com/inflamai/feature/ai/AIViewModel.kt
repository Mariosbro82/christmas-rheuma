package com.inflamai.feature.ai

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.repository.SymptomRepository
import com.inflamai.core.data.repository.TriggerRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class AIUiState(
    val isLoading: Boolean = false,
    val hasEnoughData: Boolean = false,
    val daysOfData: Int = 0,
    val requiredDays: Int = 30,
    val correlations: List<CorrelationResult> = emptyList(),
    val topTriggers: List<TriggerInfo> = emptyList(),
    val error: String? = null
)

data class CorrelationResult(
    val factor: String,
    val category: String,
    val correlation: Double,
    val pValue: Double,
    val isSignificant: Boolean,
    val lag: String = "0h"
)

data class TriggerInfo(
    val name: String,
    val count: Int,
    val impact: String
)

@HiltViewModel
class AIViewModel @Inject constructor(
    private val symptomRepository: SymptomRepository,
    private val triggerRepository: TriggerRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(AIUiState())
    val uiState: StateFlow<AIUiState> = _uiState.asStateFlow()

    init {
        loadPatternAnalysis()
    }

    private fun loadPatternAnalysis() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)

            try {
                // Check how much data we have
                val logCount = symptomRepository.getSymptomLogCount()
                val hasEnoughData = logCount >= 30

                // Get trigger counts
                val triggerCounts = triggerRepository.getTriggerCounts()
                val topTriggers = triggerCounts.take(5).map { trigger ->
                    TriggerInfo(
                        name = trigger.triggerType,
                        count = trigger.count,
                        impact = when {
                            trigger.count > 10 -> "High"
                            trigger.count > 5 -> "Medium"
                            else -> "Low"
                        }
                    )
                }

                // Check for cached analysis
                val cachedAnalysis = triggerRepository.getLatestValidAnalysis()

                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    hasEnoughData = hasEnoughData,
                    daysOfData = logCount,
                    topTriggers = topTriggers,
                    error = null
                )

            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = e.message ?: "Failed to load pattern analysis"
                )
            }
        }
    }

    fun refreshAnalysis() {
        viewModelScope.launch {
            triggerRepository.invalidateAllAnalyses()
            loadPatternAnalysis()
        }
    }
}
