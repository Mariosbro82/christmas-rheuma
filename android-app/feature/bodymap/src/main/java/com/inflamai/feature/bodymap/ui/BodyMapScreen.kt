package com.inflamai.feature.bodymap.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.inflamai.core.data.database.entity.BodyRegion
import com.inflamai.core.ui.theme.InflamAIColors
import com.inflamai.feature.bodymap.component.BodyMapCanvas
import com.inflamai.feature.bodymap.viewmodel.BodyMapUiState
import com.inflamai.feature.bodymap.viewmodel.BodyMapViewMode
import com.inflamai.feature.bodymap.viewmodel.BodyMapViewModel
import com.inflamai.feature.bodymap.viewmodel.TimeRange

/**
 * Body Map Screen
 *
 * Interactive 47-region body map for pain tracking.
 *
 * Features:
 * - Tap regions to log pain
 * - View modes: Front, Back, Spine
 * - Time range toggle for heatmap
 * - Region detail panel
 * - Save functionality
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BodyMapScreen(
    onNavigateBack: () -> Unit,
    viewModel: BodyMapViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val selectedRegion by viewModel.selectedRegion.collectAsStateWithLifecycle()
    val haptic = LocalHapticFeedback.current

    var showRegionSheet by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Body Map") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    if (uiState.hasUnsavedChanges) {
                        TextButton(
                            onClick = {
                                viewModel.saveRegionData()
                            },
                            enabled = !uiState.isSaving
                        ) {
                            if (uiState.isSaving) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    strokeWidth = 2.dp
                                )
                            } else {
                                Text("Save")
                            }
                        }
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // View mode toggle
            ViewModeSelector(
                selectedMode = uiState.viewMode,
                onModeSelected = { viewModel.setViewMode(it) }
            )

            // Time range toggle
            TimeRangeSelector(
                selectedRange = uiState.selectedTimeRange,
                onRangeSelected = { viewModel.setTimeRange(it) }
            )

            // Body map canvas
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                if (uiState.isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.align(Alignment.Center)
                    )
                } else {
                    BodyMapCanvas(
                        viewMode = uiState.viewMode,
                        regionPainData = uiState.regionPainData,
                        selectedRegionId = uiState.selectedRegionId,
                        timeRange = uiState.selectedTimeRange,
                        onRegionTap = { region ->
                            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                            viewModel.selectRegion(region)
                            showRegionSheet = true
                        }
                    )
                }
            }

            // Legend
            PainLegend()
        }
    }

    // Region detail bottom sheet
    if (showRegionSheet && selectedRegion != null) {
        ModalBottomSheet(
            onDismissRequest = {
                showRegionSheet = false
                viewModel.clearSelection()
            }
        ) {
            RegionDetailSheet(
                region = selectedRegion!!,
                painData = uiState.regionPainData[selectedRegion!!.id],
                history = uiState.selectedRegionHistory,
                onPainLevelChange = { viewModel.updatePainLevel(selectedRegion!!.id, it) },
                onStiffnessChange = { viewModel.updateStiffness(selectedRegion!!.id, it) },
                onSwellingToggle = { viewModel.toggleSwelling(selectedRegion!!.id) },
                onWarmthToggle = { viewModel.toggleWarmth(selectedRegion!!.id) },
                onDismiss = {
                    showRegionSheet = false
                    viewModel.clearSelection()
                }
            )
        }
    }
}

@Composable
fun ViewModeSelector(
    selectedMode: BodyMapViewMode,
    onModeSelected: (BodyMapViewMode) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.Center
    ) {
        BodyMapViewMode.entries.forEach { mode ->
            FilterChip(
                selected = mode == selectedMode,
                onClick = { onModeSelected(mode) },
                label = {
                    Text(
                        when (mode) {
                            BodyMapViewMode.FRONT -> "Front"
                            BodyMapViewMode.BACK -> "Back"
                            BodyMapViewMode.SPINE -> "Spine"
                        }
                    )
                },
                modifier = Modifier.padding(horizontal = 4.dp)
            )
        }
    }
}

@Composable
fun TimeRangeSelector(
    selectedRange: TimeRange,
    onRangeSelected: (TimeRange) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        horizontalArrangement = Arrangement.Center
    ) {
        TimeRange.entries.forEach { range ->
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
fun PainLegend() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        LegendItem(color = InflamAIColors.PainNone, label = "None")
        LegendItem(color = InflamAIColors.PainMild, label = "Mild")
        LegendItem(color = InflamAIColors.PainModerate, label = "Moderate")
        LegendItem(color = InflamAIColors.PainSevere, label = "Severe")
        LegendItem(color = InflamAIColors.PainExtreme, label = "Extreme")
    }
}

@Composable
fun LegendItem(color: androidx.compose.ui.graphics.Color, label: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(12.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(color)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
fun RegionDetailSheet(
    region: BodyRegion,
    painData: com.inflamai.feature.bodymap.viewmodel.RegionPainData?,
    history: List<com.inflamai.core.data.database.entity.BodyRegionLogEntity>,
    onPainLevelChange: (Int) -> Unit,
    onStiffnessChange: (Int) -> Unit,
    onSwellingToggle: () -> Unit,
    onWarmthToggle: () -> Unit,
    onDismiss: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
    ) {
        // Header
        Text(
            text = region.displayName,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = region.category.name.replace("_", " "),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Pain level slider
        Text(
            text = "Pain Level: ${painData?.currentPainLevel ?: 0}/10",
            style = MaterialTheme.typography.titleMedium
        )

        Slider(
            value = (painData?.currentPainLevel ?: 0).toFloat(),
            onValueChange = { onPainLevelChange(it.toInt()) },
            valueRange = 0f..10f,
            steps = 9,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Stiffness duration
        Text(
            text = "Stiffness: ${painData?.stiffnessMinutes ?: 0} min",
            style = MaterialTheme.typography.titleMedium
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            listOf(0, 15, 30, 60, 120).forEach { mins ->
                FilterChip(
                    selected = painData?.stiffnessMinutes == mins,
                    onClick = { onStiffnessChange(mins) },
                    label = {
                        Text(
                            when (mins) {
                                0 -> "None"
                                60 -> "1h"
                                120 -> "2h+"
                                else -> "${mins}m"
                            }
                        )
                    }
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Inflammation signs
        Text(
            text = "Signs of Inflammation",
            style = MaterialTheme.typography.titleMedium
        )

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            FilterChip(
                selected = painData?.hasSwelling == true,
                onClick = onSwellingToggle,
                label = { Text("Swelling") },
                leadingIcon = {
                    if (painData?.hasSwelling == true) {
                        Icon(Icons.Default.Check, contentDescription = null, Modifier.size(16.dp))
                    }
                }
            )

            FilterChip(
                selected = painData?.hasWarmth == true,
                onClick = onWarmthToggle,
                label = { Text("Warmth") },
                leadingIcon = {
                    if (painData?.hasWarmth == true) {
                        Icon(Icons.Default.Check, contentDescription = null, Modifier.size(16.dp))
                    }
                }
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // History summary
        if (history.isNotEmpty()) {
            Text(
                text = "Recent History",
                style = MaterialTheme.typography.titleMedium
            )

            Spacer(modifier = Modifier.height(8.dp))

            val avgPain = history.map { it.painLevel }.average()
            Text(
                text = "Average pain (last ${history.size} entries): ${String.format("%.1f", avgPain)}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Done button
        Button(
            onClick = onDismiss,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Done")
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}
