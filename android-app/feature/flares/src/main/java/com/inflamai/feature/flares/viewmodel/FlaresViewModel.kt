package com.inflamai.feature.flares.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.database.dao.FlareEventDao
import com.inflamai.core.data.database.entity.FlareEventEntity
import com.inflamai.core.data.database.entity.FlareTrigger
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.util.*
import java.util.concurrent.TimeUnit
import javax.inject.Inject

/**
 * ViewModel for Flare Timeline feature
 *
 * Manages flare history, pattern analysis, and quick flare logging.
 */
@HiltViewModel
class FlaresViewModel @Inject constructor(
    private val flareEventDao: FlareEventDao
) : ViewModel() {

    private val _uiState = MutableStateFlow(FlaresUiState())
    val uiState: StateFlow<FlaresUiState> = _uiState.asStateFlow()

    private val _showAddFlareDialog = MutableStateFlow(false)
    val showAddFlareDialog: StateFlow<Boolean> = _showAddFlareDialog.asStateFlow()

    init {
        loadFlares()
    }

    private fun loadFlares() {
        viewModelScope.launch {
            flareEventDao.observeAll().collect { flares ->
                val thirtyDaysAgo = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(30)
                val recentFlares = flares.filter { it.startDate >= thirtyDaysAgo }

                // Calculate average duration
                val completedFlares = recentFlares.filter { it.endDate != null }
                val avgDuration = if (completedFlares.isNotEmpty()) {
                    completedFlares.map { flare ->
                        TimeUnit.MILLISECONDS.toHours(flare.endDate!! - flare.startDate).toFloat()
                    }.average().toFloat()
                } else 0f

                // Find most common triggers (parse JSON)
                val triggerCounts = mutableMapOf<FlareTrigger, Int>()
                recentFlares.forEach { flare ->
                    parseTriggers(flare.suspectedTriggersJson).forEach { trigger ->
                        triggerCounts[trigger] = (triggerCounts[trigger] ?: 0) + 1
                    }
                }
                val topTriggers = triggerCounts.entries
                    .sortedByDescending { it.value }
                    .take(3)
                    .map { it.key to it.value }

                _uiState.update { state ->
                    state.copy(
                        flares = flares,
                        activeFlare = flares.firstOrNull { !it.isResolved },
                        flaresLast30Days = recentFlares.size,
                        averageDurationHours = avgDuration,
                        topTriggers = topTriggers
                    )
                }
            }
        }
    }

    private fun parseTriggers(json: String): List<FlareTrigger> {
        return try {
            if (json == "[]" || json.isBlank()) {
                emptyList()
            } else {
                // Simple parsing - triggers stored as enum names
                json.removeSurrounding("[", "]")
                    .split(",")
                    .map { it.trim().removeSurrounding("\"") }
                    .mapNotNull { name ->
                        try { FlareTrigger.valueOf(name) } catch (_: Exception) { null }
                    }
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun showAddFlareDialog() {
        _showAddFlareDialog.value = true
    }

    fun dismissAddFlareDialog() {
        _showAddFlareDialog.value = false
    }

    fun logFlare(
        severity: Int,
        triggers: List<FlareTrigger>,
        notes: String?
    ) {
        viewModelScope.launch {
            val triggersJson = if (triggers.isEmpty()) {
                "[]"
            } else {
                "[${triggers.joinToString(",") { "\"${it.name}\"" }}]"
            }

            val flare = FlareEventEntity(
                id = UUID.randomUUID().toString(),
                startDate = System.currentTimeMillis(),
                endDate = null,
                isResolved = false,
                severity = severity,
                peakSeverity = severity,
                suspectedTriggersJson = triggersJson,
                notes = notes
            )

            flareEventDao.insert(flare)
            dismissAddFlareDialog()
        }
    }

    fun endFlare(flare: FlareEventEntity) {
        viewModelScope.launch {
            flareEventDao.resolve(flare.id, System.currentTimeMillis())
        }
    }

    fun deleteFlare(flare: FlareEventEntity) {
        viewModelScope.launch {
            flareEventDao.delete(flare)
        }
    }
}

data class FlaresUiState(
    val flares: List<FlareEventEntity> = emptyList(),
    val activeFlare: FlareEventEntity? = null,
    val flaresLast30Days: Int = 0,
    val averageDurationHours: Float = 0f,
    val topTriggers: List<Pair<FlareTrigger, Int>> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null
)
