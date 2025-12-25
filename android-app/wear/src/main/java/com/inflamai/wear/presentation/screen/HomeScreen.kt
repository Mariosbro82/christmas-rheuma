package com.inflamai.wear.presentation.screen

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Warning
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material.*

/**
 * Wear OS Home Screen
 *
 * Shows:
 * - Current BASDAI score
 * - Quick action chips
 * - Last sync status
 */
@Composable
fun HomeScreen(
    onNavigateToQuickCheckIn: () -> Unit,
    onNavigateToPainLog: () -> Unit,
    onNavigateToSOS: () -> Unit
) {
    val listState = rememberScalingLazyListState()

    // Mock data - in production, sync from phone
    var basdaiScore by remember { mutableStateOf(4.2) }
    var lastSyncTime by remember { mutableStateOf("2 min ago") }

    ScalingLazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colors.background),
        state = listState,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // BASDAI Score Display
        item {
            BASDAIScoreCard(score = basdaiScore)
        }

        // Quick Check-in Chip
        item {
            Chip(
                onClick = onNavigateToQuickCheckIn,
                label = { Text("Quick Check-in") },
                icon = {
                    Icon(
                        imageVector = Icons.Filled.Check,
                        contentDescription = null
                    )
                },
                colors = ChipDefaults.primaryChipColors(),
                modifier = Modifier.fillMaxWidth(0.9f)
            )
        }

        // Pain Log Chip
        item {
            Chip(
                onClick = onNavigateToPainLog,
                label = { Text("Log Pain") },
                icon = {
                    Icon(
                        imageVector = Icons.Filled.Edit,
                        contentDescription = null
                    )
                },
                colors = ChipDefaults.secondaryChipColors(),
                modifier = Modifier.fillMaxWidth(0.9f)
            )
        }

        // SOS Flare Chip
        item {
            Chip(
                onClick = onNavigateToSOS,
                label = { Text("SOS Flare") },
                icon = {
                    Icon(
                        imageVector = Icons.Filled.Warning,
                        contentDescription = null
                    )
                },
                colors = ChipDefaults.chipColors(
                    backgroundColor = Color(0xFFF44336)
                ),
                modifier = Modifier.fillMaxWidth(0.9f)
            )
        }

        // Sync status
        item {
            Text(
                text = "Synced $lastSyncTime",
                style = MaterialTheme.typography.caption3,
                color = MaterialTheme.colors.onSurfaceVariant,
                textAlign = TextAlign.Center
            )
        }
    }
}

@Composable
fun BASDAIScoreCard(score: Double) {
    val scoreColor = when {
        score <= 2 -> Color(0xFF4CAF50)
        score <= 4 -> Color(0xFF8BC34A)
        score <= 6 -> Color(0xFFFF9800)
        else -> Color(0xFFF44336)
    }

    Card(
        onClick = { },
        modifier = Modifier
            .fillMaxWidth(0.9f)
            .padding(vertical = 8.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "BASDAI",
                style = MaterialTheme.typography.caption1,
                color = MaterialTheme.colors.onSurfaceVariant
            )

            Text(
                text = String.format("%.1f", score),
                style = MaterialTheme.typography.display1,
                fontWeight = FontWeight.Bold,
                color = scoreColor
            )

            Text(
                text = when {
                    score <= 2 -> "Remission"
                    score <= 4 -> "Low"
                    score <= 6 -> "Moderate"
                    else -> "High"
                },
                style = MaterialTheme.typography.body2,
                color = scoreColor
            )
        }
    }
}
