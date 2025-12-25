package com.inflamai.core.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp

/**
 * InflamAI Shape System
 */
val InflamAIShapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(12.dp),
    large = RoundedCornerShape(16.dp),
    extraLarge = RoundedCornerShape(24.dp)
)

/**
 * Custom shapes for specific components
 */
object CustomShapes {
    // Buttons
    val buttonShape = RoundedCornerShape(28.dp)
    val buttonShapeSmall = RoundedCornerShape(18.dp)

    // Cards
    val cardShape = RoundedCornerShape(16.dp)
    val cardShapeSmall = RoundedCornerShape(12.dp)

    // Chips
    val chipShape = RoundedCornerShape(18.dp)
    val chipShapeSmall = RoundedCornerShape(16.dp)

    // Inputs
    val inputShape = RoundedCornerShape(12.dp)
    val searchBarShape = RoundedCornerShape(24.dp)

    // Badges
    val badgeShape = RoundedCornerShape(10.dp)
    val pillBadgeShape = RoundedCornerShape(50)

    // Bottom Sheet
    val bottomSheetShape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)

    // Video Player
    val videoPlayerShape = RoundedCornerShape(16.dp)

    // Page Indicator Dot (active = pill, inactive = circle)
    val pageIndicatorActive = RoundedCornerShape(4.dp)
    val pageIndicatorInactive = RoundedCornerShape(50)

    // Toggle Switch Thumb
    val toggleThumbShape = RoundedCornerShape(50)

    // Number Badge (circle)
    val numberBadgeShape = RoundedCornerShape(50)

    // Body Map Region
    val bodyMapRegionShape = RoundedCornerShape(50)

    // Emoji Feedback Button
    val emojiFeedbackShape = RoundedCornerShape(16.dp)

    // Stat Card
    val statCardShape = RoundedCornerShape(16.dp)

    // Chart Container
    val chartShape = RoundedCornerShape(16.dp)

    // Settings Item
    val settingsItemShape = RoundedCornerShape(12.dp)

    // Progress Bar
    val progressBarShape = RoundedCornerShape(4.dp)

    // Flare Marker (triangle) - handled in custom drawing

    // Avatar
    val avatarShape = RoundedCornerShape(50)
}
