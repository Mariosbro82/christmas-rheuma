package com.inflamai.feature.trends.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.database.dao.ExerciseSessionDao
import com.inflamai.core.data.database.dao.FlareEventDao
import com.inflamai.core.data.database.dao.SymptomLogDao
import com.inflamai.core.data.database.entity.SymptomLogEntity
import com.inflamai.core.domain.calculator.BASDAICalculator
import com.inflamai.core.domain.calculator.ChangeSignificance
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit
import javax.inject.Inject

/**
 * Trends ViewModel
 *
 * Provides analytics and visualizations for:
 * - BASDAI score trends (7/30/90 day)
 * - Individual symptom breakdowns
 * - Flare frequency
 * - Exercise correlation
 * - Statistical insights
 */
@HiltViewModel
class TrendsViewModel @Inject constructor(
    private val symptomLogDao: SymptomLogDao,
    private val flareEventDao: FlareEventDao,
    private val exerciseSessionDao: ExerciseSessionDao
) : ViewModel() {

    private val _uiState = MutableStateFlow(TrendsUiState())
    val uiState: StateFlow<TrendsUiState> = _uiState.asStateFlow()

    init {
        loadTrendsData()
    }

    private fun loadTrendsData() {
        viewModelScope.launch {
            val now = Instant.now()
            val thirtyDaysAgo = now.minus(30, ChronoUnit.DAYS).toEpochMilli()
            val ninetyDaysAgo = now.minus(90, ChronoUnit.DAYS).toEpochMilli()
            val nowMillis = now.toEpochMilli()

            symptomLogDao.observeByDateRange(ninetyDaysAgo, nowMillis).collect { logs ->
                if (logs.isEmpty()) {
                    _uiState.update { it.copy(isLoading = false, hasData = false) }
                    return@collect
                }

                // Calculate chart data points
                val chartData = logs
                    .sortedBy { it.timestamp }
                    .map { log ->
                        ChartDataPoint(
                            timestamp = log.timestamp,
                            date = formatDate(log.timestamp),
                            basdaiScore = log.basdaiScore,
                            fatigueLevel = log.fatigueLevel,
                            painLevel = ((log.q2SpinalPain + log.q3PeripheralPain) / 2).toInt(),
                            stiffnessLevel = log.q5MorningStiffnessSeverity.toInt(),
                            isFlare = log.isFlareEvent
                        )
                    }

                // Calculate statistics
                val last30Days = logs.filter { it.timestamp >= thirtyDaysAgo }
                val stats = calculateStatistics(logs, last30Days)

                // Calculate symptom breakdown
                val symptomBreakdown = calculateSymptomBreakdown(last30Days)

                // Calculate trend direction
                val trend = calculateTrendDirection(chartData)

                // Get flare count
                val flareCount = flareEventDao.getCountForDateRange(thirtyDaysAgo, nowMillis)

                // Get exercise correlation
                val exerciseMinutes = exerciseSessionDao.getTotalMinutes(thirtyDaysAgo, nowMillis) ?: 0
                val exerciseDays = exerciseSessionDao.getSessionCount(thirtyDaysAgo, nowMillis)

                _uiState.update { state ->
                    state.copy(
                        isLoading = false,
                        hasData = true,
                        chartData = chartData,
                        statistics = stats,
                        symptomBreakdown = symptomBreakdown,
                        trendDirection = trend,
                        flareCount30Days = flareCount,
                        exerciseMinutes30Days = exerciseMinutes,
                        exerciseDays30Days = exerciseDays
                    )
                }
            }
        }
    }

    private fun calculateStatistics(
        allLogs: List<SymptomLogEntity>,
        last30Days: List<SymptomLogEntity>
    ): TrendStatistics {
        val currentAvg = last30Days.map { it.basdaiScore }.average()
        val previousPeriod = allLogs
            .filter {
                val thirtyDaysAgo = Instant.now().minus(30, ChronoUnit.DAYS).toEpochMilli()
                val sixtyDaysAgo = Instant.now().minus(60, ChronoUnit.DAYS).toEpochMilli()
                it.timestamp in sixtyDaysAgo until thirtyDaysAgo
            }
            .map { it.basdaiScore }
            .average()
            .takeIf { !it.isNaN() }

        val change = previousPeriod?.let { currentAvg - it }
        val changeSignificance = previousPeriod?.let {
            BASDAICalculator.isSignificantChange(it, currentAvg)
        }

        val lowestScore = last30Days.minOfOrNull { it.basdaiScore } ?: 0.0
        val highestScore = last30Days.maxOfOrNull { it.basdaiScore } ?: 0.0

        return TrendStatistics(
            averageScore = currentAvg,
            lowestScore = lowestScore,
            highestScore = highestScore,
            changeFromPrevious = change,
            changeSignificance = changeSignificance,
            totalCheckIns = last30Days.size
        )
    }

    private fun calculateSymptomBreakdown(logs: List<SymptomLogEntity>): List<SymptomBreakdown> {
        if (logs.isEmpty()) return emptyList()

        return listOf(
            SymptomBreakdown(
                name = "Fatigue",
                averageScore = logs.map { it.q1Fatigue }.average(),
                trend = calculateSymptomTrend(logs.map { it.q1Fatigue })
            ),
            SymptomBreakdown(
                name = "Spinal Pain",
                averageScore = logs.map { it.q2SpinalPain }.average(),
                trend = calculateSymptomTrend(logs.map { it.q2SpinalPain })
            ),
            SymptomBreakdown(
                name = "Peripheral Pain",
                averageScore = logs.map { it.q3PeripheralPain }.average(),
                trend = calculateSymptomTrend(logs.map { it.q3PeripheralPain })
            ),
            SymptomBreakdown(
                name = "Tenderness",
                averageScore = logs.map { it.q4Tenderness }.average(),
                trend = calculateSymptomTrend(logs.map { it.q4Tenderness })
            ),
            SymptomBreakdown(
                name = "Morning Stiffness",
                averageScore = logs.map { it.q5MorningStiffnessSeverity }.average(),
                trend = calculateSymptomTrend(logs.map { it.q5MorningStiffnessSeverity })
            )
        )
    }

    private fun calculateSymptomTrend(values: List<Double>): SymptomTrend {
        if (values.size < 4) return SymptomTrend.STABLE

        val recent = values.takeLast(values.size / 2).average()
        val earlier = values.take(values.size / 2).average()
        val delta = recent - earlier

        return when {
            delta <= -0.5 -> SymptomTrend.IMPROVING
            delta >= 0.5 -> SymptomTrend.WORSENING
            else -> SymptomTrend.STABLE
        }
    }

    private fun calculateTrendDirection(data: List<ChartDataPoint>): TrendDirection {
        if (data.size < 7) return TrendDirection.INSUFFICIENT_DATA

        val recentWeek = data.takeLast(7).map { it.basdaiScore }
        val previousWeek = data.dropLast(7).takeLast(7).map { it.basdaiScore }

        if (previousWeek.isEmpty()) return TrendDirection.INSUFFICIENT_DATA

        val recentAvg = recentWeek.average()
        val previousAvg = previousWeek.average()
        val delta = recentAvg - previousAvg

        return when {
            delta <= -1.0 -> TrendDirection.IMPROVING_SIGNIFICANTLY
            delta <= -0.5 -> TrendDirection.IMPROVING
            delta >= 1.0 -> TrendDirection.WORSENING_SIGNIFICANTLY
            delta >= 0.5 -> TrendDirection.WORSENING
            else -> TrendDirection.STABLE
        }
    }

    private fun formatDate(timestamp: Long): String {
        val date = Instant.ofEpochMilli(timestamp)
            .atZone(ZoneId.systemDefault())
            .toLocalDate()
        return date.format(DateTimeFormatter.ofPattern("MMM d"))
    }

    fun setTimeRange(range: TrendTimeRange) {
        _uiState.update { it.copy(selectedTimeRange = range) }
    }

    fun setChartType(type: ChartType) {
        _uiState.update { it.copy(selectedChartType = type) }
    }
}

data class TrendsUiState(
    val isLoading: Boolean = true,
    val hasData: Boolean = false,
    val selectedTimeRange: TrendTimeRange = TrendTimeRange.DAYS_30,
    val selectedChartType: ChartType = ChartType.BASDAI,
    val chartData: List<ChartDataPoint> = emptyList(),
    val statistics: TrendStatistics? = null,
    val symptomBreakdown: List<SymptomBreakdown> = emptyList(),
    val trendDirection: TrendDirection = TrendDirection.INSUFFICIENT_DATA,
    val flareCount30Days: Int = 0,
    val exerciseMinutes30Days: Int = 0,
    val exerciseDays30Days: Int = 0
)

data class ChartDataPoint(
    val timestamp: Long,
    val date: String,
    val basdaiScore: Double,
    val fatigueLevel: Int,
    val painLevel: Int,
    val stiffnessLevel: Int,
    val isFlare: Boolean
)

data class TrendStatistics(
    val averageScore: Double,
    val lowestScore: Double,
    val highestScore: Double,
    val changeFromPrevious: Double?,
    val changeSignificance: ChangeSignificance?,
    val totalCheckIns: Int
)

data class SymptomBreakdown(
    val name: String,
    val averageScore: Double,
    val trend: SymptomTrend
)

enum class SymptomTrend {
    IMPROVING, STABLE, WORSENING
}

enum class TrendDirection {
    IMPROVING_SIGNIFICANTLY,
    IMPROVING,
    STABLE,
    WORSENING,
    WORSENING_SIGNIFICANTLY,
    INSUFFICIENT_DATA
}

enum class TrendTimeRange(val label: String, val days: Int) {
    DAYS_7("7 Days", 7),
    DAYS_30("30 Days", 30),
    DAYS_90("90 Days", 90)
}

enum class ChartType(val label: String) {
    BASDAI("BASDAI Score"),
    SYMPTOMS("Symptoms"),
    PAIN("Pain Levels"),
    STIFFNESS("Stiffness")
}
