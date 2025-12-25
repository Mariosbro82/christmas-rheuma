package com.inflamai.core.ui.theme

import androidx.compose.runtime.Composable
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * InflamAI Spacing System
 * Based on 4dp base unit for consistent spacing
 */
data class Spacing(
    val xxs: Dp = 2.dp,
    val xs: Dp = 4.dp,
    val sm: Dp = 8.dp,
    val md: Dp = 12.dp,
    val lg: Dp = 16.dp,
    val xl: Dp = 24.dp,
    val xxl: Dp = 32.dp,
    val xxxl: Dp = 48.dp
)

/**
 * Screen Margins
 */
data class ScreenMargins(
    val horizontal: Dp = 16.dp,
    val horizontalLarge: Dp = 24.dp,
    val top: Dp = 16.dp,
    val bottom: Dp = 16.dp
)

/**
 * Corner Radii
 */
data class CornerRadii(
    val xs: Dp = 4.dp,
    val sm: Dp = 8.dp,
    val md: Dp = 12.dp,
    val lg: Dp = 16.dp,
    val xl: Dp = 24.dp,
    val full: Dp = 999.dp
)

/**
 * Component Sizes
 */
object ComponentSizes {
    // Buttons
    val buttonHeight = 56.dp
    val buttonHeightSmall = 36.dp
    val buttonMinWidth = 200.dp
    val buttonCornerRadius = 28.dp

    // Icon Buttons
    val iconButtonSize = 48.dp
    val iconSizeSmall = 16.dp
    val iconSizeMedium = 24.dp
    val iconSizeLarge = 32.dp

    // Cards
    val cardPadding = 16.dp
    val cardMarginVertical = 8.dp
    val cardElevation = 2.dp
    val cardCornerRadius = 16.dp

    // Inputs
    val inputHeight = 56.dp
    val inputCornerRadius = 12.dp

    // Chips
    val chipHeight = 36.dp
    val chipHeightSmall = 32.dp
    val chipCornerRadius = 18.dp

    // Bottom Navigation
    val bottomNavHeight = 64.dp
    val bottomNavElevation = 8.dp

    // Page Indicator
    val pageIndicatorDotSize = 8.dp
    val pageIndicatorActiveWidth = 20.dp
    val pageIndicatorSpacing = 8.dp

    // Sliders
    val sliderTrackHeight = 8.dp
    val sliderThumbSize = 24.dp

    // Feature Cards
    val featureCardIconSize = 48.dp

    // Mascot
    val mascotSizeLarge = 200.dp
    val mascotSizeMedium = 120.dp
    val mascotSizeSmall = 64.dp

    // Touch Targets (accessibility minimum)
    val minTouchTarget = 48.dp

    // Body Map Region
    val bodyMapRegionSize = 24.dp

    // Avatar
    val avatarSize = 48.dp
    val avatarSizeLarge = 64.dp

    // Progress Bar
    val progressBarHeight = 4.dp
    val progressBarHeightLarge = 8.dp

    // Stat Card
    val statCardValueSize = 32.dp
    val statCardIconSize = 48.dp

    // Timer Display
    val timerDisplaySize = 80.dp

    // Reps Counter Button
    val repsCounterButtonSize = 56.dp

    // Video Player
    val videoPlayerAspectRatio = 16f / 9f
    val videoPlayerCornerRadius = 16.dp

    // Badge
    val badgeHeight = 20.dp
    val badgeCornerRadius = 10.dp

    // Number Badge
    val numberBadgeSize = 24.dp

    // Emoji Button (Feedback)
    val emojiFeedbackButtonSize = 120.dp
    val emojiSize = 48.dp

    // Chart
    val chartHeight = 200.dp
    val chartLegendDotSize = 8.dp

    // Flare Marker
    val flareMarkerSize = 16.dp
}

val LocalSpacing = compositionLocalOf { Spacing() }
val LocalScreenMargins = compositionLocalOf { ScreenMargins() }
val LocalCornerRadii = compositionLocalOf { CornerRadii() }

/**
 * Extension properties for easy access in Composables
 */
object InflamAIDimens {
    val spacing = Spacing()
    val margins = ScreenMargins()
    val radii = CornerRadii()
}
