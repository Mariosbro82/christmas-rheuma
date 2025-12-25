package com.inflamai.feature.flares.ui

import androidx.compose.foundation.background
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.inflamai.core.data.database.entity.FlareEventEntity
import com.inflamai.core.data.database.entity.FlareTrigger
import com.inflamai.feature.flares.viewmodel.FlaresViewModel
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit

/**
 * Flares Timeline Screen
 *
 * Displays flare history, active flares, and analytics.
 * Follows Material Design 3 and WCAG AA accessibility.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FlaresScreen(
    onNavigateBack: () -> Unit,
    viewModel: FlaresViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val showAddDialog by viewModel.showAddFlareDialog.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Flare Timeline") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Navigate back"
                        )
                    }
                }
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { viewModel.showAddFlareDialog() },
                icon = {
                    Icon(
                        Icons.Default.LocalFireDepartment,
                        contentDescription = null
                    )
                },
                text = { Text("Log Flare") },
                containerColor = Color(0xFFF44336),
                contentColor = Color.White
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Active Flare Warning
            uiState.activeFlare?.let { activeFlare ->
                item {
                    ActiveFlareCard(
                        flare = activeFlare,
                        onEndFlare = { viewModel.endFlare(activeFlare) }
                    )
                }
            }

            // Stats Summary
            item {
                FlareStatsCard(
                    flaresLast30Days = uiState.flaresLast30Days,
                    averageDurationHours = uiState.averageDurationHours,
                    topTriggers = uiState.topTriggers
                )
            }

            // Flare History
            items(
                items = uiState.flares,
                key = { it.id }
            ) { flare ->
                FlareCard(
                    flare = flare,
                    onEndFlare = { viewModel.endFlare(flare) },
                    onDelete = { viewModel.deleteFlare(flare) }
                )
            }

            if (uiState.flares.isEmpty()) {
                item {
                    EmptyFlaresState()
                }
            }

            // Bottom spacing for FAB
            item {
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
    }

    // Add Flare Dialog
    if (showAddDialog) {
        AddFlareDialog(
            onDismiss = { viewModel.dismissAddFlareDialog() },
            onSave = { severity, triggers, notes ->
                viewModel.logFlare(severity, triggers, notes)
            }
        )
    }
}

@Composable
private fun ActiveFlareCard(
    flare: FlareEventEntity,
    onEndFlare: () -> Unit
) {
    val durationMs = System.currentTimeMillis() - flare.startDate
    val hours = TimeUnit.MILLISECONDS.toHours(durationMs)
    val minutes = TimeUnit.MILLISECONDS.toMinutes(durationMs) % 60

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFF44336).copy(alpha = 0.15f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = Icons.Default.LocalFireDepartment,
                    contentDescription = null,
                    tint = Color(0xFFF44336),
                    modifier = Modifier.size(32.dp)
                )

                Spacer(modifier = Modifier.width(12.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Active Flare",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFF44336)
                    )
                    Text(
                        text = "Duration: ${hours}h ${minutes}m",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                    )
                }

                SeverityChip(severity = flare.severity)
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = onEndFlare,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.CheckCircle, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("End Flare")
                }
            }
        }
    }
}

@Composable
private fun FlareStatsCard(
    flaresLast30Days: Int,
    averageDurationHours: Float,
    topTriggers: List<Pair<FlareTrigger, Int>>
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "30-Day Summary",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                StatItem(
                    value = flaresLast30Days.toString(),
                    label = "Flares"
                )

                StatItem(
                    value = if (averageDurationHours > 0) "${averageDurationHours.toInt()}h" else "-",
                    label = "Avg Duration"
                )
            }

            if (topTriggers.isNotEmpty()) {
                Spacer(modifier = Modifier.height(16.dp))
                HorizontalDivider()
                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = "Common Triggers",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(8.dp))

                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(topTriggers) { (trigger, count) ->
                        AssistChip(
                            onClick = { },
                            label = { Text("${trigger.displayName} ($count)") },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Outlined.TrendingUp,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp)
                                )
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatItem(
    value: String,
    label: String
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FlareCard(
    flare: FlareEventEntity,
    onEndFlare: () -> Unit,
    onDelete: () -> Unit
) {
    var showMenu by remember { mutableStateOf(false) }
    val dateFormat = SimpleDateFormat("MMM d, yyyy", Locale.getDefault())
    val isActive = !flare.isResolved

    val duration = flare.endDate?.let { endDate ->
        val hours = TimeUnit.MILLISECONDS.toHours(endDate - flare.startDate)
        if (hours < 1) "${TimeUnit.MILLISECONDS.toMinutes(endDate - flare.startDate)}m"
        else "${hours}h"
    } ?: "Ongoing"

    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.Top
        ) {
            // Timeline indicator
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .size(12.dp)
                        .clip(CircleShape)
                        .background(
                            if (isActive) Color(0xFFF44336)
                            else MaterialTheme.colorScheme.primary
                        )
                )
                if (!isActive) {
                    Box(
                        modifier = Modifier
                            .width(2.dp)
                            .height(48.dp)
                            .background(MaterialTheme.colorScheme.outlineVariant)
                    )
                }
            }

            Spacer(modifier = Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = dateFormat.format(Date(flare.startDate)),
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold
                    )

                    Spacer(modifier = Modifier.width(8.dp))

                    SeverityChip(severity = flare.severity)
                }

                Spacer(modifier = Modifier.height(4.dp))

                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Schedule,
                        contentDescription = null,
                        modifier = Modifier.size(14.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = duration,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    flare.peakSeverity?.let { peak ->
                        Spacer(modifier = Modifier.width(12.dp))
                        Icon(
                            imageVector = Icons.Outlined.TrendingUp,
                            contentDescription = null,
                            modifier = Modifier.size(14.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "Peak: $peak/10",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                flare.notes?.let { notes ->
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = notes,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Box {
                IconButton(onClick = { showMenu = true }) {
                    Icon(
                        imageVector = Icons.Default.MoreVert,
                        contentDescription = "More options"
                    )
                }

                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = { showMenu = false }
                ) {
                    if (isActive) {
                        DropdownMenuItem(
                            text = { Text("End Flare") },
                            onClick = {
                                showMenu = false
                                onEndFlare()
                            },
                            leadingIcon = {
                                Icon(Icons.Outlined.CheckCircle, contentDescription = null)
                            }
                        )
                    }
                    DropdownMenuItem(
                        text = { Text("Delete") },
                        onClick = {
                            showMenu = false
                            onDelete()
                        },
                        leadingIcon = {
                            Icon(
                                Icons.Outlined.Delete,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error
                            )
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun SeverityChip(severity: Int) {
    val (color, text) = when {
        severity <= 3 -> Color(0xFFFF9800) to "Mild"
        severity <= 6 -> Color(0xFFFF5722) to "Moderate"
        severity <= 8 -> Color(0xFFF44336) to "Severe"
        else -> Color(0xFFD32F2F) to "Extreme"
    }

    Surface(
        shape = RoundedCornerShape(4.dp),
        color = color.copy(alpha = 0.2f)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall,
            color = color,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
        )
    }
}

@Composable
private fun EmptyFlaresState() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = Icons.Outlined.LocalFireDepartment,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "No flares recorded",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )

        Text(
            text = "Track your flares to identify patterns and triggers",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AddFlareDialog(
    onDismiss: () -> Unit,
    onSave: (Int, List<FlareTrigger>, String?) -> Unit
) {
    var severity by remember { mutableStateOf(5f) }
    var selectedTriggers by remember { mutableStateOf(setOf<FlareTrigger>()) }
    var notes by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.LocalFireDepartment,
                    contentDescription = null,
                    tint = Color(0xFFF44336)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Log Flare")
            }
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Severity Slider
                Text(
                    text = "Severity: ${severity.toInt()}/10",
                    style = MaterialTheme.typography.labelMedium
                )

                Slider(
                    value = severity,
                    onValueChange = { severity = it },
                    valueRange = 1f..10f,
                    steps = 8,
                    colors = SliderDefaults.colors(
                        thumbColor = Color(0xFFF44336),
                        activeTrackColor = Color(0xFFF44336)
                    )
                )

                // Triggers
                Text(
                    text = "Triggers (optional)",
                    style = MaterialTheme.typography.labelMedium
                )

                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(FlareTrigger.entries.toList()) { trigger ->
                        val isSelected = trigger in selectedTriggers

                        FilterChip(
                            selected = isSelected,
                            onClick = {
                                selectedTriggers = if (isSelected) {
                                    selectedTriggers - trigger
                                } else {
                                    selectedTriggers + trigger
                                }
                            },
                            label = { Text(trigger.displayName) }
                        )
                    }
                }

                // Notes
                OutlinedTextField(
                    value = notes,
                    onValueChange = { notes = it },
                    label = { Text("Notes (optional)") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                    maxLines = 3
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    onSave(
                        severity.toInt(),
                        selectedTriggers.toList(),
                        notes.ifBlank { null }
                    )
                },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFF44336)
                )
            ) {
                Text("Log Flare")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}
