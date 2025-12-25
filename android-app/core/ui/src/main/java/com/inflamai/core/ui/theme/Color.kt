package com.inflamai.core.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * InflamAI Color System
 * Extracted from iOS app video frame-by-frame analysis
 */

// Primary Colors - Blue (matches iOS primary)
val PrimaryBlue = Color(0xFF4A7AB0)
val PrimaryBlueLight = Color(0xFFE3F2FD)
val PrimaryBlueDark = Color(0xFF3D6691)

// Accent Colors
val AccentPink = Color(0xFFE91E63)
val AccentPinkLight = Color(0xFFFCE4EC)
val AccentPurple = Color(0xFF9C27B0)
val AccentPurpleLight = Color(0xFFF3E5F5)
val AccentOrange = Color(0xFFFF9800)
val AccentOrangeLight = Color(0xFFFFF3E0)
val AccentTeal = Color(0xFF26A69A)
val AccentTealLight = Color(0xFFE0F2F1)
val AccentCoral = Color(0xFFFF7043)

// Semantic Colors
val Success = Color(0xFF4CAF50)
val SuccessLight = Color(0xFFE8F5E9)
val Error = Color(0xFFF44336)
val ErrorLight = Color(0xFFFFEBEE)
val Warning = Color(0xFFFF9800)
val WarningLight = Color(0xFFFFF3E0)
val Info = Color(0xFF2196F3)
val InfoLight = Color(0xFFE3F2FD)

// Neutral Colors
val TextPrimary = Color(0xFF1A1A1A)
val TextSecondary = Color(0xFF666666)
val TextTertiary = Color(0xFF999999)
val TextDisabled = Color(0xFFCCCCCC)

val BackgroundPrimary = Color(0xFFF8F9FA)
val BackgroundSecondary = Color(0xFFF5F5F5)
val Surface = Color(0xFFFFFFFF)
val SurfaceElevated = Color(0xFFFFFFFF)

val Divider = Color(0xFFE0E0E0)
val Border = Color(0xFFE0E0E0)

// Pain Level Gradient Colors
val PainSliderStart = Color(0xFFFF6B6B)
val PainSliderEnd = Color(0xFFFF0000)
val StiffnessColor = Color(0xFFFF9800)
val FatigueColor = Color(0xFF9C27B0)

// Severity Colors (0-10 scale)
val Severity0 = Color(0xFF4CAF50)  // Green - None/Minimal
val Severity3 = Color(0xFF8BC34A)  // Light Green - Mild
val Severity5 = Color(0xFFFFC107)  // Amber - Moderate
val Severity7 = Color(0xFFFF9800)  // Orange - High
val Severity9 = Color(0xFFF44336)  // Red - Severe

// BASDAI Score Colors
val BasdaiRemission = Color(0xFF4CAF50)      // 0-2: Green
val BasdaiLowActivity = Color(0xFF8BC34A)    // 2-4: Light Green
val BasdaiModerateActivity = Color(0xFFFF9800) // 4-6: Orange
val BasdaiHighActivity = Color(0xFFF44336)   // 6+: Red

// Mood Badge Colors
val MoodPositive = Color(0xFF4CAF50)
val MoodChallenging = Color(0xFF757575)
val MoodNeutral = Color(0xFF9E9E9E)

// Trend Badge Colors
val TrendImproving = Color(0xFF4CAF50)
val TrendStable = Color(0xFF26A69A)
val TrendWorsening = Color(0xFFF44336)

// Card Icon Circle Backgrounds
val IconCircleBlue = Color(0xFFE3F2FD)
val IconCirclePurple = Color(0xFFF3E5F5)
val IconCirclePink = Color(0xFFFCE4EC)
val IconCircleOrange = Color(0xFFFFF3E0)
val IconCircleGreen = Color(0xFFE8F5E9)
val IconCircleRed = Color(0xFFFFEBEE)
val IconCircleTeal = Color(0xFFE0F2F1)
val IconCircleCream = Color(0xFFFFF8E1)

// Medication Adherence
val AdherenceExcellent = Color(0xFF4CAF50)
val AdherenceGood = Color(0xFF8BC34A)
val AdherencePartial = Color(0xFFFF9800)
val AdherencePoor = Color(0xFFF44336)

// Flare Status
val FlareActive = Color(0xFFF44336)
val FlareWarning = Color(0xFFFF9800)
val FlareResolved = Color(0xFF4CAF50)

// Chart Colors
val ChartBasdai = Color(0xFF2196F3)
val ChartPain = Color(0xFFF44336)
val ChartStiffness = Color(0xFFFF9800)
val ChartFatigue = Color(0xFF9C27B0)
val ChartFill = Color(0xFFE3F2FD)

// Assessment Legend Colors
val AssessmentAsqol = Color(0xFF4CAF50)
val AssessmentBasdai = Color(0xFFE91E63)
val AssessmentBasfi = Color(0xFFF44336)
val AssessmentBasg = Color(0xFF2196F3)

// Exercise Difficulty Colors
val DifficultyBeginner = Color(0xFF26A69A)
val DifficultyIntermediate = Color(0xFFFF9800)
val DifficultyAdvanced = Color(0xFFF44336)

// Onboarding Feature Card Icon Colors
val FeatureTrackIcon = Color(0xFF2196F3)   // Blue - Chart
val FeaturePatternsIcon = Color(0xFF9C27B0) // Purple - Brain
val FeatureLiveIcon = Color(0xFFE91E63)     // Pink - Clipboard

// Take Button Color
val TakeButtonColor = Color(0xFFFF7043)

// Toggle Switch Colors
val ToggleOnBlue = Color(0xFF2196F3)
val ToggleOnPurple = Color(0xFF9C27B0)
val ToggleOnCyan = Color(0xFF00BCD4)
val ToggleOff = Color(0xFFE0E0E0)

// Page Indicator Colors
val PageIndicatorActive = Color(0xFF4A7AB0)
val PageIndicatorInactive = Color(0xFFE0E0E0)

// Slider Track Colors
object SliderColors {
    val pain = listOf(Color(0xFFFFCDD2), Color(0xFFF44336))
    val stiffness = listOf(Color(0xFFFFE0B2), Color(0xFFFF9800))
    val fatigue = listOf(Color(0xFFE1BEE7), Color(0xFF9C27B0))
}

// Body Map Region Colors
object BodyMapColors {
    val unselected = Color(0xFFBBDEFB)
    val selected = Color(0xFFE91E63)
    val heatLow = Color(0xFF4CAF50)
    val heatMedium = Color(0xFFFFEB3B)
    val heatHigh = Color(0xFFFF9800)
    val heatExtreme = Color(0xFFF44336)
}

// Meditation Player Colors
object MeditationColors {
    val gradientStart = Color(0xFF9C27B0)
    val gradientEnd = Color(0xFFE91E63)
}

// Exercise Coach Progress Colors
object ExerciseColors {
    val progressGreen = Color(0xFF4CAF50)
    val progressBlue = Color(0xFF2196F3)
    val timerOrange = Color(0xFFFF9800)
    val repsBlue = Color(0xFF2196F3)
    val repsMinus = Color(0xFFF44336)
    val repsPlus = Color(0xFF2196F3)
}
