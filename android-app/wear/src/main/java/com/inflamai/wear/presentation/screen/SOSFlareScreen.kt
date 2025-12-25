package com.inflamai.wear.presentation.screen

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Warning
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.material.*

/**
 * SOS Flare Screen for Wear OS
 *
 * Quick flare logging with severity selection.
 * Designed for emergencies when detailed logging isn't possible.
 */
@Composable
fun SOSFlareScreen(
    onComplete: () -> Unit
) {
    val haptic = LocalHapticFeedback.current

    var severity by remember { mutableStateOf(7) }
    var isRecorded by remember { mutableStateOf(false) }

    if (isRecorded) {
        // Confirmation screen
        ScalingLazyColumn(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            item {
                Icon(
                    imageVector = Icons.Filled.Check,
                    contentDescription = null,
                    tint = Color(0xFFFF9800),
                    modifier = Modifier.size(48.dp)
                )
            }

            item {
                Text(
                    text = "Flare Recorded",
                    style = MaterialTheme.typography.title2,
                    modifier = Modifier.padding(8.dp)
                )
            }

            item {
                Text(
                    text = "Severity: $severity/10",
                    style = MaterialTheme.typography.body1
                )
            }

            item {
                Text(
                    text = "Syncing to phone...",
                    style = MaterialTheme.typography.caption2,
                    color = MaterialTheme.colors.onSurfaceVariant,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }

            item {
                Chip(
                    onClick = onComplete,
                    label = { Text("Done") },
                    modifier = Modifier.padding(top = 16.dp)
                )
            }
        }
        return
    }

    ScalingLazyColumn(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        item {
            Icon(
                imageVector = Icons.Filled.Warning,
                contentDescription = null,
                tint = Color(0xFFF44336),
                modifier = Modifier
                    .size(32.dp)
                    .padding(bottom = 8.dp)
            )
        }

        item {
            Text(
                text = "Log Flare",
                style = MaterialTheme.typography.title2,
                color = Color(0xFFF44336)
            )
        }

        item {
            Text(
                text = "Severity",
                style = MaterialTheme.typography.body2,
                color = MaterialTheme.colors.onSurfaceVariant,
                modifier = Modifier.padding(top = 16.dp, bottom = 8.dp)
            )
        }

        item {
            InlineSlider(
                value = severity,
                onValueChange = {
                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                    severity = it
                },
                valueProgression = IntProgression.fromClosedRange(1, 10, 1),
                decreaseIcon = { Icon(InlineSliderDefaults.Decrease, "Decrease") },
                increaseIcon = { Icon(InlineSliderDefaults.Increase, "Increase") },
                modifier = Modifier.fillMaxWidth(0.8f)
            )
        }

        item {
            Text(
                text = "$severity",
                style = MaterialTheme.typography.display2,
                color = getSeverityColor(severity)
            )
        }

        item {
            Text(
                text = getSeverityDescription(severity),
                style = MaterialTheme.typography.body2,
                color = MaterialTheme.colors.onSurfaceVariant
            )
        }

        item {
            Chip(
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    // Record flare
                    isRecorded = true
                },
                label = { Text("Record Flare") },
                colors = ChipDefaults.chipColors(
                    backgroundColor = Color(0xFFF44336)
                ),
                modifier = Modifier.padding(top = 16.dp)
            )
        }

        item {
            CompactChip(
                onClick = onComplete,
                label = { Text("Cancel") },
                modifier = Modifier.padding(top = 8.dp)
            )
        }
    }
}

@Composable
private fun getSeverityColor(severity: Int): Color {
    return when {
        severity <= 3 -> Color(0xFFFF9800)
        severity <= 6 -> Color(0xFFFF5722)
        else -> Color(0xFFF44336)
    }
}

private fun getSeverityDescription(severity: Int): String {
    return when {
        severity <= 3 -> "Mild flare"
        severity <= 5 -> "Moderate flare"
        severity <= 7 -> "Significant flare"
        else -> "Severe flare"
    }
}
