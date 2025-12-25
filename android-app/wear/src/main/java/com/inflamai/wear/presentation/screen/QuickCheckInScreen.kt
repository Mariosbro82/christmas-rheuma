package com.inflamai.wear.presentation.screen

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
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
 * Quick Check-in Screen for Wear OS
 *
 * Simplified 3-step check-in:
 * 1. Overall feeling (1-10)
 * 2. Pain level (1-10)
 * 3. Stiffness level (1-10)
 *
 * Designed for quick logging in 3 taps.
 */
@Composable
fun QuickCheckInScreen(
    onComplete: () -> Unit
) {
    val haptic = LocalHapticFeedback.current

    var currentStep by remember { mutableStateOf(0) }
    var overallFeeling by remember { mutableStateOf(5) }
    var painLevel by remember { mutableStateOf(5) }
    var stiffnessLevel by remember { mutableStateOf(5) }

    val steps = listOf(
        QuickCheckInStep("How do you feel?", "Overall", overallFeeling) { overallFeeling = it },
        QuickCheckInStep("Pain level?", "Pain", painLevel) { painLevel = it },
        QuickCheckInStep("Stiffness?", "Stiffness", stiffnessLevel) { stiffnessLevel = it }
    )

    if (currentStep >= steps.size) {
        // Show completion
        CompletionScreen(
            overallFeeling = overallFeeling,
            painLevel = painLevel,
            stiffnessLevel = stiffnessLevel,
            onDone = onComplete
        )
        return
    }

    val current = steps[currentStep]

    ScalingLazyColumn(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Progress indicator
        item {
            Text(
                text = "${currentStep + 1}/${steps.size}",
                style = MaterialTheme.typography.caption2,
                color = MaterialTheme.colors.onSurfaceVariant
            )
        }

        // Question
        item {
            Text(
                text = current.question,
                style = MaterialTheme.typography.title2,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
            )
        }

        // Value selector using InlineSlider
        item {
            InlineSlider(
                value = current.value,
                onValueChange = {
                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                    current.onValueChange(it)
                },
                valueProgression = IntProgression.fromClosedRange(1, 10, 1),
                decreaseIcon = { Icon(InlineSliderDefaults.Decrease, "Decrease") },
                increaseIcon = { Icon(InlineSliderDefaults.Increase, "Increase") },
                modifier = Modifier.fillMaxWidth(0.8f)
            )
        }

        // Current value display
        item {
            Text(
                text = "${current.value}",
                style = MaterialTheme.typography.display2,
                color = getScoreColor(current.value)
            )
        }

        // Next button
        item {
            Chip(
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    currentStep++
                },
                label = {
                    Text(if (currentStep == steps.size - 1) "Done" else "Next")
                },
                colors = ChipDefaults.primaryChipColors(),
                modifier = Modifier.padding(top = 16.dp)
            )
        }
    }
}

@Composable
private fun CompletionScreen(
    overallFeeling: Int,
    painLevel: Int,
    stiffnessLevel: Int,
    onDone: () -> Unit
) {
    val haptic = LocalHapticFeedback.current

    // Trigger success haptic
    LaunchedEffect(Unit) {
        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
    }

    ScalingLazyColumn(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        item {
            Icon(
                imageVector = Icons.Filled.Check,
                contentDescription = null,
                tint = Color(0xFF4CAF50),
                modifier = Modifier.size(48.dp)
            )
        }

        item {
            Text(
                text = "Logged!",
                style = MaterialTheme.typography.title1,
                modifier = Modifier.padding(8.dp)
            )
        }

        item {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.padding(vertical = 8.dp)
            ) {
                Text("Overall: $overallFeeling", style = MaterialTheme.typography.body2)
                Text("Pain: $painLevel", style = MaterialTheme.typography.body2)
                Text("Stiffness: $stiffnessLevel", style = MaterialTheme.typography.body2)
            }
        }

        item {
            Chip(
                onClick = onDone,
                label = { Text("Done") },
                modifier = Modifier.padding(top = 8.dp)
            )
        }
    }
}

private data class QuickCheckInStep(
    val question: String,
    val label: String,
    val value: Int,
    val onValueChange: (Int) -> Unit
)

@Composable
private fun getScoreColor(score: Int): Color {
    return when {
        score <= 3 -> Color(0xFF4CAF50)
        score <= 5 -> Color(0xFF8BC34A)
        score <= 7 -> Color(0xFFFF9800)
        else -> Color(0xFFF44336)
    }
}
