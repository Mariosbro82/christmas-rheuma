package com.inflamai.feature.home.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
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
import com.inflamai.core.domain.calculator.BASDAIInterpretation
import com.inflamai.core.ui.theme.InflamAIColors
import com.inflamai.feature.home.viewmodel.HomeUiState
import com.inflamai.feature.home.viewmodel.HomeViewModel
import com.inflamai.feature.home.viewmodel.ScoreTrend
import java.text.SimpleDateFormat
import java.util.*

/**
 * Home Dashboard Screen
 *
 * Main landing page showing:
 * - Current BASDAI score with circular gauge
 * - Quick action cards
 * - Recent trends sparkline
 * - Active flare alerts
 * - Medication reminders
 * - Weather-based flare risk
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onNavigateToCheckIn: () -> Unit,
    onNavigateToBodyMap: () -> Unit,
    onNavigateToTrends: () -> Unit,
    onNavigateToMedication: () -> Unit,
    onNavigateToFlares: () -> Unit,
    onNavigateToQuickCapture: () -> Unit,
    viewModel: HomeViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "Welcome${uiState.userName?.let { ", $it" } ?: ""}",
                            style = MaterialTheme.typography.titleLarge
                        )
                        if (uiState.streakDays > 0) {
                            Text(
                                text = "${uiState.streakDays} day streak",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                },
                actions = {
                    IconButton(onClick = { viewModel.refresh() }) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh")
                    }
                }
            )
        }
    ) { paddingValues ->
        if (uiState.isLoading) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator()
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // BASDAI Score Card
                item {
                    BASDAIScoreCard(
                        score = uiState.currentBasdaiScore,
                        interpretation = uiState.basdaiInterpretation,
                        trend = uiState.scoreTrend,
                        hasCheckedInToday = uiState.hasCheckedInToday,
                        onCheckInClick = onNavigateToCheckIn
                    )
                }

                // Quick Actions
                item {
                    QuickActionsRow(
                        onCheckInClick = onNavigateToCheckIn,
                        onBodyMapClick = onNavigateToBodyMap,
                        onSOSClick = onNavigateToQuickCapture,
                        hasCheckedInToday = uiState.hasCheckedInToday
                    )
                }

                // Weather Flare Risk
                uiState.flareRisk?.let { risk ->
                    item {
                        WeatherRiskCard(
                            risk = risk,
                            weather = uiState.weatherData
                        )
                    }
                }

                // Active Flares Alert
                if (uiState.activeFlares.isNotEmpty()) {
                    item {
                        ActiveFlaresCard(
                            flareCount = uiState.activeFlares.size,
                            onClick = onNavigateToFlares
                        )
                    }
                }

                // Health Snapshot
                uiState.healthSnapshot?.let { snapshot ->
                    item {
                        HealthSnapshotCard(snapshot = snapshot)
                    }
                }

                // Medication Reminders
                if (uiState.pendingMedicationReminders.isNotEmpty()) {
                    item {
                        MedicationRemindersCard(
                            medications = uiState.pendingMedicationReminders.map { it.name },
                            onClick = onNavigateToMedication
                        )
                    }
                }

                // Trend Sparkline
                if (uiState.recentScores.isNotEmpty()) {
                    item {
                        TrendCard(
                            scores = uiState.recentScores,
                            trend = uiState.scoreTrend,
                            onClick = onNavigateToTrends
                        )
                    }
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
fun BASDAIScoreCard(
    score: Double,
    interpretation: BASDAIInterpretation?,
    trend: ScoreTrend,
    hasCheckedInToday: Boolean,
    onCheckInClick: () -> Unit
) {
    val scoreColor = when (interpretation) {
        BASDAIInterpretation.REMISSION -> InflamAIColors.ScoreRemission
        BASDAIInterpretation.LOW_ACTIVITY -> InflamAIColors.ScoreLowActivity
        BASDAIInterpretation.MODERATE_ACTIVITY -> InflamAIColors.ScoreModerateActivity
        BASDAIInterpretation.HIGH_ACTIVITY -> InflamAIColors.ScoreHighActivity
        BASDAIInterpretation.VERY_HIGH_ACTIVITY -> InflamAIColors.ScoreVeryHighActivity
        null -> MaterialTheme.colorScheme.outline
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics { contentDescription = "BASDAI score ${String.format(Locale.getDefault(), "%.1f", score)} out of 10" },
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "BASDAI Score",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Large Score Display
            Box(
                modifier = Modifier
                    .size(140.dp)
                    .clip(CircleShape)
                    .background(scoreColor.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = String.format(Locale.getDefault(), "%.1f", score),
                        style = MaterialTheme.typography.displayLarge,
                        fontWeight = FontWeight.Bold,
                        color = scoreColor
                    )
                    Text(
                        text = "/ 10",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Interpretation
            interpretation?.let {
                Text(
                    text = it.displayName,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = scoreColor
                )

                Text(
                    text = it.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 16.dp)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Trend indicator
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                val (trendIcon, trendText, trendColor) = when (trend) {
                    ScoreTrend.IMPROVING -> Triple(Icons.Default.TrendingDown, "Improving", InflamAIColors.TrendImproving)
                    ScoreTrend.WORSENING -> Triple(Icons.Default.TrendingUp, "Worsening", InflamAIColors.TrendWorsening)
                    ScoreTrend.STABLE -> Triple(Icons.Default.TrendingFlat, "Stable", InflamAIColors.TrendStable)
                }

                Icon(
                    imageVector = trendIcon,
                    contentDescription = null,
                    tint = trendColor,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = trendText,
                    style = MaterialTheme.typography.bodyMedium,
                    color = trendColor
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Check-in button
            if (!hasCheckedInToday) {
                Button(
                    onClick = onCheckInClick,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.CheckCircle, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Complete Today's Check-in")
                }
            } else {
                OutlinedButton(
                    onClick = onCheckInClick,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.Edit, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Update Check-in")
                }
            }
        }
    }
}

@Composable
fun QuickActionsRow(
    onCheckInClick: () -> Unit,
    onBodyMapClick: () -> Unit,
    onSOSClick: () -> Unit,
    hasCheckedInToday: Boolean
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        QuickActionCard(
            modifier = Modifier.weight(1f),
            icon = if (hasCheckedInToday) Icons.Filled.CheckCircle else Icons.Outlined.CheckCircle,
            title = "Check-in",
            subtitle = if (hasCheckedInToday) "Done" else "Due",
            color = if (hasCheckedInToday) InflamAIColors.AdherenceGood else MaterialTheme.colorScheme.primary,
            onClick = onCheckInClick
        )

        QuickActionCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Outlined.Person,
            title = "Body Map",
            subtitle = "Track pain",
            color = MaterialTheme.colorScheme.secondary,
            onClick = onBodyMapClick
        )

        QuickActionCard(
            modifier = Modifier.weight(1f),
            icon = Icons.Filled.Warning,
            title = "SOS",
            subtitle = "Log flare",
            color = InflamAIColors.FlareActive,
            onClick = onSOSClick
        )
    }
}

@Composable
fun QuickActionCard(
    modifier: Modifier = Modifier,
    icon: ImageVector,
    title: String,
    subtitle: String,
    color: Color,
    onClick: () -> Unit
) {
    Card(
        modifier = modifier
            .clickable(onClick = onClick)
            .semantics { contentDescription = "$title: $subtitle" },
        colors = CardDefaults.cardColors(
            containerColor = color.copy(alpha = 0.1f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(28.dp)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.labelLarge,
                color = color
            )
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun WeatherRiskCard(
    risk: com.inflamai.core.data.service.weather.FlareRiskAssessment,
    weather: com.inflamai.core.data.service.weather.WeatherData?
) {
    val riskColor = when (risk.level) {
        com.inflamai.core.data.service.weather.FlareRiskLevel.LOW -> InflamAIColors.WeatherLowRisk
        com.inflamai.core.data.service.weather.FlareRiskLevel.MODERATE -> InflamAIColors.WeatherModerateRisk
        com.inflamai.core.data.service.weather.FlareRiskLevel.HIGH -> InflamAIColors.WeatherHighRisk
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = riskColor.copy(alpha = 0.1f)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Cloud,
                contentDescription = null,
                tint = riskColor,
                modifier = Modifier.size(40.dp)
            )

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Weather Flare Risk: ${risk.level.name}",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = riskColor
                )

                weather?.let {
                    Text(
                        text = "${it.temperature?.toInt() ?: "--"}°C • ${it.weatherCondition}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                if (risk.factors.isNotEmpty()) {
                    Text(
                        text = risk.factors.first(),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
fun ActiveFlaresCard(
    flareCount: Int,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = InflamAIColors.FlareActive.copy(alpha = 0.1f)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.LocalFireDepartment,
                contentDescription = null,
                tint = InflamAIColors.FlareActive,
                modifier = Modifier.size(32.dp)
            )

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "$flareCount Active Flare${if (flareCount > 1) "s" else ""}",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = InflamAIColors.FlareActive
                )
                Text(
                    text = "Tap to view details and track progress",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Icon(
                imageVector = Icons.Default.ChevronRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun HealthSnapshotCard(
    snapshot: com.inflamai.core.data.service.health.DailyHealthSnapshot
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "Today's Health",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                HealthMetric(
                    icon = Icons.Default.Favorite,
                    value = snapshot.restingHeartRate?.toString() ?: "--",
                    unit = "bpm",
                    label = "Resting HR"
                )

                HealthMetric(
                    icon = Icons.Default.DirectionsWalk,
                    value = (snapshot.stepCount).toString(),
                    unit = "",
                    label = "Steps"
                )

                snapshot.sleepDurationMinutes?.let { sleepMinutes ->
                    HealthMetric(
                        icon = Icons.Default.Bedtime,
                        value = "${sleepMinutes / 60}h ${sleepMinutes % 60}m",
                        unit = "",
                        label = "Sleep"
                    )
                }

                snapshot.latestHrv?.let { hrv ->
                    HealthMetric(
                        icon = Icons.Default.ShowChart,
                        value = String.format(Locale.getDefault(), "%.0f", hrv),
                        unit = "ms",
                        label = "HRV"
                    )
                }
            }
        }
    }
}

@Composable
fun HealthMetric(
    icon: ImageVector,
    value: String,
    unit: String,
    label: String
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(24.dp)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "$value$unit",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
fun MedicationRemindersCard(
    medications: List<String>,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Medication,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(32.dp)
            )

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Medication Reminders",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = medications.take(2).joinToString(", ") +
                           if (medications.size > 2) " +${medications.size - 2} more" else "",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Icon(
                imageVector = Icons.Default.ChevronRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun TrendCard(
    scores: List<Double>,
    trend: ScoreTrend,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "7-Day Trend",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )

                Text(
                    text = "View Details →",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Simple sparkline visualization
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(60.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.Bottom
            ) {
                scores.forEach { score ->
                    val height = (score / 10.0 * 60).dp
                    val color = when {
                        score <= 2 -> InflamAIColors.ScoreRemission
                        score <= 4 -> InflamAIColors.ScoreLowActivity
                        score <= 6 -> InflamAIColors.ScoreModerateActivity
                        else -> InflamAIColors.ScoreHighActivity
                    }

                    Box(
                        modifier = Modifier
                            .width(20.dp)
                            .height(height.coerceAtLeast(4.dp))
                            .clip(RoundedCornerShape(topStart = 4.dp, topEnd = 4.dp))
                            .background(color)
                    )
                }
            }
        }
    }
}

@Composable
fun MedicalDisclaimer() {
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
                text = "This app is for informational purposes only. Always consult your healthcare provider.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
