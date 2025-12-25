package com.inflamai.feature.home.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.database.dao.FlareEventDao
import com.inflamai.core.data.database.dao.MedicationDao
import com.inflamai.core.data.database.dao.SymptomLogDao
import com.inflamai.core.data.database.dao.UserProfileDao
import com.inflamai.core.data.database.entity.FlareEventEntity
import com.inflamai.core.data.database.entity.MedicationEntity
import com.inflamai.core.data.database.entity.SymptomLogEntity
import com.inflamai.core.data.service.health.DailyHealthSnapshot
import com.inflamai.core.data.service.health.HealthConnectService
import com.inflamai.core.data.service.weather.FlareRiskAssessment
import com.inflamai.core.data.service.weather.FlareRiskLevel
import com.inflamai.core.data.service.weather.WeatherData
import com.inflamai.core.data.service.weather.WeatherService
import com.inflamai.core.domain.calculator.BASDAICalculator
import com.inflamai.core.domain.calculator.BASDAIInterpretation
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.temporal.ChronoUnit
import javax.inject.Inject

/**
 * Home Dashboard ViewModel
 *
 * Displays:
 * - Current BASDAI score with interpretation
 * - Recent symptom trend (7-day sparkline)
 * - Active flares
 * - Medication reminders
 * - Weather-based flare risk
 * - Quick actions (Check-in, SOS, Body Map)
 */
@HiltViewModel
class HomeViewModel @Inject constructor(
    private val symptomLogDao: SymptomLogDao,
    private val flareEventDao: FlareEventDao,
    private val medicationDao: MedicationDao,
    private val userProfileDao: UserProfileDao,
    private val healthConnectService: HealthConnectService,
    private val weatherService: WeatherService
) : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    init {
        loadDashboardData()
    }

    private fun loadDashboardData() {
        viewModelScope.launch {
            // Combine multiple data streams
            combine(
                symptomLogDao.observeLatest(),
                symptomLogDao.observeRecent(7),
                flareEventDao.observeActive(),
                medicationDao.observeActive()
            ) { latestLog, recentLogs, activeFlares, activeMeds ->
                DashboardData(latestLog, recentLogs, activeFlares, activeMeds)
            }.collect { data ->
                updateUiState(data)
            }
        }

        // Load weather data
        viewModelScope.launch {
            try {
                val weather = weatherService.getCurrentWeather()
                weather?.let {
                    val riskAssessment = weatherService.calculateFlareRisk(it)
                    _uiState.update { state ->
                        state.copy(
                            weatherData = weather,
                            flareRisk = riskAssessment
                        )
                    }
                }
            } catch (e: Exception) {
                // Weather is optional, don't fail
            }
        }

        // Load health snapshot
        viewModelScope.launch {
            try {
                val snapshot = healthConnectService.getDailyHealthSnapshot(Instant.now())
                _uiState.update { state ->
                    state.copy(healthSnapshot = snapshot)
                }
            } catch (e: Exception) {
                // Health data is optional
            }
        }

        // Load user profile
        viewModelScope.launch {
            userProfileDao.observe().collect { profile ->
                _uiState.update { state ->
                    state.copy(
                        userName = profile?.name,
                        hasCompletedOnboarding = profile?.hasCompletedOnboarding ?: false,
                        streakDays = profile?.streakDays ?: 0
                    )
                }
            }
        }
    }

    private fun updateUiState(data: DashboardData) {
        val latestScore = data.latestLog?.basdaiScore ?: 0.0
        val interpretation = if (latestScore > 0) {
            BASDAICalculator.interpret(latestScore)
        } else null

        // Calculate 7-day trend
        val trendData = data.recentLogs
            .sortedBy { it.timestamp }
            .map { it.basdaiScore }

        val trend = calculateTrend(trendData)

        // Check if today's check-in is done
        val startOfToday = LocalDate.now()
            .atStartOfDay(ZoneId.systemDefault())
            .toInstant()
            .toEpochMilli()

        val hasCheckedInToday = data.latestLog?.timestamp?.let { it >= startOfToday } ?: false

        _uiState.update { state ->
            state.copy(
                isLoading = false,
                currentBasdaiScore = latestScore,
                basdaiInterpretation = interpretation,
                recentScores = trendData,
                scoreTrend = trend,
                hasCheckedInToday = hasCheckedInToday,
                lastCheckInDate = data.latestLog?.timestamp,
                activeFlares = data.activeFlares,
                activeMedications = data.activeMedications,
                pendingMedicationReminders = data.activeMedications.filter { it.reminderEnabled }
            )
        }
    }

    private fun calculateTrend(scores: List<Double>): ScoreTrend {
        if (scores.size < 2) return ScoreTrend.STABLE

        val recent = scores.takeLast(3).average()
        val earlier = scores.take(3).average()
        val delta = recent - earlier

        return when {
            delta <= -1.0 -> ScoreTrend.IMPROVING
            delta >= 1.0 -> ScoreTrend.WORSENING
            else -> ScoreTrend.STABLE
        }
    }

    fun refresh() {
        _uiState.update { it.copy(isLoading = true) }
        loadDashboardData()
    }
}

data class HomeUiState(
    val isLoading: Boolean = true,
    val userName: String? = null,
    val hasCompletedOnboarding: Boolean = false,
    val streakDays: Int = 0,

    // BASDAI
    val currentBasdaiScore: Double = 0.0,
    val basdaiInterpretation: BASDAIInterpretation? = null,
    val recentScores: List<Double> = emptyList(),
    val scoreTrend: ScoreTrend = ScoreTrend.STABLE,

    // Check-in status
    val hasCheckedInToday: Boolean = false,
    val lastCheckInDate: Long? = null,

    // Flares
    val activeFlares: List<FlareEventEntity> = emptyList(),

    // Medications
    val activeMedications: List<MedicationEntity> = emptyList(),
    val pendingMedicationReminders: List<MedicationEntity> = emptyList(),

    // Weather & Health
    val weatherData: WeatherData? = null,
    val flareRisk: FlareRiskAssessment? = null,
    val healthSnapshot: DailyHealthSnapshot? = null,

    // Errors
    val error: String? = null
)

enum class ScoreTrend {
    IMPROVING,
    STABLE,
    WORSENING
}

private data class DashboardData(
    val latestLog: SymptomLogEntity?,
    val recentLogs: List<SymptomLogEntity>,
    val activeFlares: List<FlareEventEntity>,
    val activeMedications: List<MedicationEntity>
)
