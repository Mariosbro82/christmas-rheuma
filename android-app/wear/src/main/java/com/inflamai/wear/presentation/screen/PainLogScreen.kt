package com.inflamai.wear.presentation.screen

import androidx.compose.foundation.layout.*
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
 * Pain Log Screen for Wear OS
 *
 * Quick pain level logging with body region selection.
 */
@Composable
fun PainLogScreen(
    onComplete: () -> Unit
) {
    val haptic = LocalHapticFeedback.current

    var selectedRegion by remember { mutableStateOf<String?>(null) }
    var painLevel by remember { mutableStateOf(5) }

    val regions = listOf(
        "Lower Back" to "lower_back",
        "Neck" to "neck",
        "Hips" to "hips",
        "Shoulders" to "shoulders",
        "Knees" to "knees",
        "General" to "general"
    )

    ScalingLazyColumn(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (selectedRegion == null) {
            // Region selection
            item {
                Text(
                    text = "Where is the pain?",
                    style = MaterialTheme.typography.title3,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }

            regions.forEach { (name, id) ->
                item {
                    Chip(
                        onClick = {
                            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                            selectedRegion = id
                        },
                        label = { Text(name) },
                        colors = ChipDefaults.secondaryChipColors(),
                        modifier = Modifier.fillMaxWidth(0.9f)
                    )
                }
            }
        } else {
            // Pain level selection
            item {
                Text(
                    text = "Pain Level",
                    style = MaterialTheme.typography.title3,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }

            item {
                InlineSlider(
                    value = painLevel,
                    onValueChange = {
                        haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                        painLevel = it
                    },
                    valueProgression = IntProgression.fromClosedRange(0, 10, 1),
                    decreaseIcon = { Icon(InlineSliderDefaults.Decrease, "Decrease") },
                    increaseIcon = { Icon(InlineSliderDefaults.Increase, "Increase") },
                    modifier = Modifier.fillMaxWidth(0.8f)
                )
            }

            item {
                Text(
                    text = "$painLevel",
                    style = MaterialTheme.typography.display2,
                    color = getPainColor(painLevel)
                )
            }

            item {
                Text(
                    text = getPainDescription(painLevel),
                    style = MaterialTheme.typography.body2,
                    color = MaterialTheme.colors.onSurfaceVariant
                )
            }

            item {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(top = 16.dp)
                ) {
                    CompactChip(
                        onClick = { selectedRegion = null },
                        label = { Text("Back") }
                    )

                    Chip(
                        onClick = {
                            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                            // Save and complete
                            onComplete()
                        },
                        label = { Text("Save") },
                        colors = ChipDefaults.primaryChipColors()
                    )
                }
            }
        }
    }
}

@Composable
private fun getPainColor(level: Int): Color {
    return when {
        level == 0 -> Color(0xFF4CAF50)
        level <= 3 -> Color(0xFF8BC34A)
        level <= 5 -> Color(0xFFFFEB3B)
        level <= 7 -> Color(0xFFFF9800)
        else -> Color(0xFFF44336)
    }
}

private fun getPainDescription(level: Int): String {
    return when {
        level == 0 -> "No pain"
        level <= 2 -> "Mild"
        level <= 4 -> "Moderate"
        level <= 6 -> "Significant"
        level <= 8 -> "Severe"
        else -> "Extreme"
    }
}
