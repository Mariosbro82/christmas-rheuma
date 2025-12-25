package com.inflamai.wear.presentation.theme

import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.wear.compose.material.Colors
import androidx.wear.compose.material.MaterialTheme

/**
 * InflamAI Wear OS Theme
 *
 * Matches the phone app's design language adapted for Wear OS.
 */

private val InflamAIWearColors = Colors(
    primary = Color(0xFF4FD8EB),
    primaryVariant = Color(0xFF006874),
    secondary = Color(0xFFB1CBD0),
    secondaryVariant = Color(0xFF334B4F),
    error = Color(0xFFFFB4AB),
    onPrimary = Color(0xFF00363D),
    onSecondary = Color(0xFF1C3438),
    onError = Color(0xFF690005),
    background = Color(0xFF191C1D),
    onBackground = Color(0xFFE1E3E3),
    surface = Color(0xFF191C1D),
    onSurface = Color(0xFFE1E3E3),
    onSurfaceVariant = Color(0xFFBFC8CA)
)

@Composable
fun InflamAIWearTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colors = InflamAIWearColors,
        content = content
    )
}

// AS-specific colors for Wear OS
object WearColors {
    val ScoreRemission = Color(0xFF4CAF50)
    val ScoreLowActivity = Color(0xFF8BC34A)
    val ScoreModerateActivity = Color(0xFFFF9800)
    val ScoreHighActivity = Color(0xFFFF5722)
    val ScoreVeryHighActivity = Color(0xFFF44336)

    val FlareActive = Color(0xFFF44336)
    val FlareResolving = Color(0xFFFF9800)
}
