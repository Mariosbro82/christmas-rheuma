package com.inflamai.feature.trends.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.TrendingDown
import androidx.compose.material.icons.automirrored.filled.TrendingFlat
import androidx.compose.material.icons.automirrored.filled.TrendingUp
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.inflamai.core.domain.calculator.ChangeSignificance
import com.inflamai.core.ui.theme.InflamAIColors
import com.inflamai.feature.trends.viewmodel.*
import java.util.Locale

/**
 * Trends & Analytics Screen
 *
 * Displays comprehensive health analytics including:
 * - BASDAI score trend chart
 * - Statistics summary
 * - Symptom breakdown
 * - Exercise correlation
 * - Flare frequency
 *
 * WCAG AA compliant with accessibility support.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrendsScreen(
    onNavigateBack: () -> Unit,
    viewModel: TrendsViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Trends & Analytics") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Navigate back"
                        )
                    }
                }
            )
        }
    ) { padding ->
        if (uiState.isLoading) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator()
            }
        } else if (!uiState.hasData) {
            EmptyDataState(modifier = Modifier.padding(padding))
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Time Range Selector
                item {
                    TimeRangeSelector(
                        selectedRange = uiState.selectedTimeRange,
                        onRangeSelected = { viewModel.setTimeRange(it) }
                    )
                }

                // Trend Direction Card
                item {
                    TrendDirectionCard(
                        direction = uiState.trendDirection,
                        statistics = uiState.statistics
                    )
                }

                // BASDAI Chart
                item {
                    BASDAIChartCard(
                        chartData = uiState.chartData,
                        selectedChartType = uiState.selectedChartType,
                        onChartTypeChange = { viewModel.setChartType(it) }
                    )
                }

                // Statistics Summary
                uiState.statistics?.let { stats ->
                    item {
                        StatisticsSummaryCard(statistics = stats)
                    }
                }

                // Symptom Breakdown
                if (uiState.symptomBreakdown.isNotEmpty()) {
                    item {
                        SymptomBreakdownCard(symptoms = uiState.symptomBreakdown)
                    }
                }

                // Activity Summary
                item {
                    ActivitySummaryCard(
                        flareCount = uiState.flareCount30Days,
                        exerciseMinutes = uiState.exerciseMinutes30Days,
                        exerciseDays = uiState.exerciseDays30Days
                    )
                }

                // Insights Card (Correlation Analysis)
                item {
                    InsightsCard()
                }

                // Medical Disclaimer
                item {
                    MedicalDisclaimer()
                }
            }
        }
    }
}

@Composable
private fun TimeRangeSelector(
    selectedRange: TrendTimeRange,
    onRangeSelected: (TrendTimeRange) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center
    ) {
        TrendTimeRange.entries.forEach { range ->
            FilterChip(
                selected = range == selectedRange,
                onClick = { onRangeSelected(range) },
                label = { Text(range.label) },
                modifier = Modifier.padding(horizontal = 4.dp)
            )
        }
    }
}

@Composable
private fun TrendDirectionCard(
    direction: TrendDirection,
    statistics: TrendStatistics?
) {
    val (icon, text, color, bgColor) = when (direction) {
        TrendDirection.IMPROVING_SIGNIFICANTLY -> {
            listOf(Icons.AutoMirrored.Filled.TrendingDown, "Significantly Improving", InflamAIColors.TrendImproving, InflamAIColors.TrendImproving.copy(alpha = 0.1f))
        }
        TrendDirection.IMPROVING -> {
            listOf(Icons.AutoMirrored.Filled.TrendingDown, "Improving", InflamAIColors.TrendImproving, InflamAIColors.TrendImproving.copy(alpha = 0.1f))
        }
        TrendDirection.STABLE -> {
            listOf(Icons.AutoMirrored.Filled.TrendingFlat, "Stable", InflamAIColors.TrendStable, InflamAIColors.TrendStable.copy(alpha = 0.1f))
        }
        TrendDirection.WORSENING -> {
            listOf(Icons.AutoMirrored.Filled.TrendingUp, "Worsening", InflamAIColors.TrendWorsening, InflamAIColors.TrendWorsening.copy(alpha = 0.1f))
        }
        TrendDirection.WORSENING_SIGNIFICANTLY -> {
            listOf(Icons.AutoMirrored.Filled.TrendingUp, "Significantly Worsening", InflamAIColors.TrendWorsening, InflamAIColors.TrendWorsening.copy(alpha = 0.1f))
        }
        TrendDirection.INSUFFICIENT_DATA -> {
            listOf(Icons.Default.DataUsage, "Need More Data", MaterialTheme.colorScheme.outline, MaterialTheme.colorScheme.surfaceVariant)
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics { contentDescription = "Your trend is $text" },
        colors = CardDefaults.cardColors(
            containerColor = bgColor as Color
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon as ImageVector,
                contentDescription = null,
                tint = color as Color,
                modifier = Modifier.size(40.dp)
            )

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Your Trend",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = text as String,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = color
                )

                statistics?.changeFromPrevious?.let { change ->
                    val changeText = if (change > 0) "+${String.format(Locale.getDefault(), "%.1f", change)}"
                    else String.format(Locale.getDefault(), "%.1f", change)

                    Text(
                        text = "$changeText vs previous period",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun BASDAIChartCard(
    chartData: List<ChartDataPoint>,
    selectedChartType: ChartType,
    onChartTypeChange: (ChartType) -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "Score History",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Chart type selector
            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(ChartType.entries.toList()) { type ->
                    FilterChip(
                        selected = type == selectedChartType,
                        onClick = { onChartTypeChange(type) },
                        label = { Text(type.label) }
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Simple bar chart visualization
            if (chartData.isNotEmpty()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(120.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.Bottom
                ) {
                    chartData.takeLast(14).forEach { point ->
                        val value = when (selectedChartType) {
                            ChartType.BASDAI -> point.basdaiScore
                            ChartType.SYMPTOMS -> ((point.fatigueLevel + point.painLevel + point.stiffnessLevel) / 3.0)
                            ChartType.PAIN -> point.painLevel.toDouble()
                            ChartType.STIFFNESS -> point.stiffnessLevel.toDouble()
                        }

                        val height = ((value / 10.0) * 100).dp.coerceAtLeast(4.dp)
                        val barColor = getScoreColor(value)

                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.weight(1f)
                        ) {
                            Box(
                                modifier = Modifier
                                    .width(12.dp)
                                    .height(height)
                                    .clip(RoundedCornerShape(topStart = 4.dp, topEnd = 4.dp))
                                    .background(barColor)
                            )

                            if (point.isFlare) {
                                Icon(
                                    imageVector = Icons.Default.LocalFireDepartment,
                                    contentDescription = "Flare",
                                    modifier = Modifier.size(12.dp),
                                    tint = InflamAIColors.FlareActive
                                )
                            }
                        }
                    }
                }

                // X-axis labels (simplified)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    chartData.takeLast(14).firstOrNull()?.let { first ->
                        Text(
                            text = first.date,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    chartData.lastOrNull()?.let { last ->
                        Text(
                            text = last.date,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // Legend
            Spacer(modifier = Modifier.height(16.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                LegendItem(color = InflamAIColors.ScoreRemission, label = "Low")
                LegendItem(color = InflamAIColors.ScoreModerateActivity, label = "Moderate")
                LegendItem(color = InflamAIColors.ScoreHighActivity, label = "High")
            }
        }
    }
}

@Composable
private fun LegendItem(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(12.dp)
                .clip(CircleShape)
                .background(color)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun StatisticsSummaryCard(statistics: TrendStatistics) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "30-Day Statistics",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )

            Spacer(modifier = Modifier.height(16.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                StatItem(
                    value = String.format(Locale.getDefault(), "%.1f", statistics.averageScore),
                    label = "Average",
                    color = getScoreColor(statistics.averageScore)
                )
                StatItem(
                    value = String.format(Locale.getDefault(), "%.1f", statistics.lowestScore),
                    label = "Best",
                    color = InflamAIColors.ScoreRemission
                )
                StatItem(
                    value = String.format(Locale.getDefault(), "%.1f", statistics.highestScore),
                    label = "Worst",
                    color = InflamAIColors.ScoreHighActivity
                )
                StatItem(
                    value = statistics.totalCheckIns.toString(),
                    label = "Check-ins",
                    color = MaterialTheme.colorScheme.primary
                )
            }

            // Significance indicator
            statistics.changeSignificance?.let { significance ->
                Spacer(modifier = Modifier.height(16.dp))
                HorizontalDivider()
                Spacer(modifier = Modifier.height(12.dp))

                val (significanceText, significanceColor) = when (significance) {
                    ChangeSignificance.MAJOR_IMPROVEMENT ->
                        "Major improvement detected - great progress!" to InflamAIColors.TrendImproving
                    ChangeSignificance.CLINICALLY_MEANINGFUL_IMPROVEMENT ->
                        "Clinically meaningful improvement detected" to InflamAIColors.TrendImproving
                    ChangeSignificance.NO_SIGNIFICANT_CHANGE ->
                        "Changes within normal variation" to MaterialTheme.colorScheme.outline
                    ChangeSignificance.CLINICALLY_MEANINGFUL_WORSENING ->
                        "Clinically meaningful worsening - consider consulting your doctor" to InflamAIColors.TrendWorsening
                    ChangeSignificance.MAJOR_WORSENING ->
                        "Significant worsening detected - please consult your rheumatologist" to InflamAIColors.TrendWorsening
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Info,
                        contentDescription = null,
                        tint = significanceColor,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = significanceText,
                        style = MaterialTheme.typography.bodySmall,
                        color = significanceColor
                    )
                }
            }
        }
    }
}

@Composable
private fun StatItem(
    value: String,
    label: String,
    color: Color
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = value,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
            color = color
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun SymptomBreakdownCard(symptoms: List<SymptomBreakdown>) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "Symptom Breakdown",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )

            Spacer(modifier = Modifier.height(16.dp))

            symptoms.forEach { symptom ->
                SymptomRow(symptom = symptom)
                if (symptom != symptoms.last()) {
                    Spacer(modifier = Modifier.height(12.dp))
                }
            }
        }
    }
}

@Composable
private fun SymptomRow(symptom: SymptomBreakdown) {
    val trendIcon = when (symptom.trend) {
        SymptomTrend.IMPROVING -> Icons.AutoMirrored.Filled.TrendingDown
        SymptomTrend.STABLE -> Icons.AutoMirrored.Filled.TrendingFlat
        SymptomTrend.WORSENING -> Icons.AutoMirrored.Filled.TrendingUp
    }

    val trendColor = when (symptom.trend) {
        SymptomTrend.IMPROVING -> InflamAIColors.TrendImproving
        SymptomTrend.STABLE -> InflamAIColors.TrendStable
        SymptomTrend.WORSENING -> InflamAIColors.TrendWorsening
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = symptom.name,
                style = MaterialTheme.typography.bodyMedium
            )
            // Progress bar for severity
            LinearProgressIndicator(
                progress = { (symptom.averageScore / 10.0).toFloat() },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp)),
                color = getScoreColor(symptom.averageScore),
                trackColor = MaterialTheme.colorScheme.surfaceVariant
            )
        }

        Spacer(modifier = Modifier.width(12.dp))

        Text(
            text = String.format(Locale.getDefault(), "%.1f", symptom.averageScore),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = getScoreColor(symptom.averageScore)
        )

        Spacer(modifier = Modifier.width(8.dp))

        Icon(
            imageVector = trendIcon,
            contentDescription = "Trend: ${symptom.trend.name}",
            tint = trendColor,
            modifier = Modifier.size(20.dp)
        )
    }
}

@Composable
private fun ActivitySummaryCard(
    flareCount: Int,
    exerciseMinutes: Int,
    exerciseDays: Int
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "Activity Summary",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )

            Spacer(modifier = Modifier.height(16.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                ActivityItem(
                    icon = Icons.Outlined.LocalFireDepartment,
                    value = flareCount.toString(),
                    label = "Flares",
                    color = if (flareCount > 0) InflamAIColors.FlareActive else InflamAIColors.TrendImproving
                )
                ActivityItem(
                    icon = Icons.Outlined.FitnessCenter,
                    value = "${exerciseMinutes}m",
                    label = "Exercise",
                    color = MaterialTheme.colorScheme.primary
                )
                ActivityItem(
                    icon = Icons.Outlined.CalendarMonth,
                    value = exerciseDays.toString(),
                    label = "Active Days",
                    color = MaterialTheme.colorScheme.tertiary
                )
            }
        }
    }
}

@Composable
private fun ActivityItem(
    icon: ImageVector,
    value: String,
    label: String,
    color: Color
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = color,
            modifier = Modifier.size(28.dp)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = value,
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = color
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun InsightsCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.5f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Psychology,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.tertiary
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Pattern Analysis",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Mock insight - In real implementation, this would come from CorrelationEngine
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Outlined.WbCloudy,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(
                            text = "Weather Correlation",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "Low barometric pressure may affect symptoms",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Continue tracking for more personalized insights. Minimum 30 days of data recommended.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun MedicalDisclaimer() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Outlined.Info,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "This data is for informational purposes only. Always discuss trends with your rheumatologist.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun EmptyDataState(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp)
        ) {
            Icon(
                imageVector = Icons.Outlined.Analytics,
                contentDescription = null,
                modifier = Modifier.size(80.dp),
                tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
            )

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "No Data Yet",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Start logging your daily symptoms to see trends and insights",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
            )
        }
    }
}

// Helper function to get color based on score
@Composable
private fun getScoreColor(score: Double): Color {
    return when {
        score <= 2 -> InflamAIColors.ScoreRemission
        score <= 4 -> InflamAIColors.ScoreLowActivity
        score <= 6 -> InflamAIColors.ScoreModerateActivity
        score <= 8 -> InflamAIColors.ScoreHighActivity
        else -> InflamAIColors.ScoreVeryHighActivity
    }
}