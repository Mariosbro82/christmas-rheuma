package com.inflamai.core.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

/**
 * InflamAI Color System
 *
 * AS-specific colors:
 * - Inflammation indicators (reds/oranges)
 * - Healing/improvement (greens)
 * - Activity levels (gradient)
 * - Accessibility compliant (WCAG AA 4.5:1 contrast)
 */

// Primary Colors - Soft Purple (medical-grade, trust)
private val PrimaryLight = Color(0xFF6B5DD3)
private val OnPrimaryLight = Color(0xFFFFFFFF)
private val PrimaryContainerLight = Color(0xFFE8E0FF)
private val OnPrimaryContainerLight = Color(0xFF21005E)

private val PrimaryDark = Color(0xFFCABEFF)
private val OnPrimaryDark = Color(0xFF352896)
private val PrimaryContainerDark = Color(0xFF4D3EB0)
private val OnPrimaryContainerDark = Color(0xFFE8E0FF)

// Secondary Colors - Coral/Orange (pain/flares, activity)
private val SecondaryLight = Color(0xFFFF8A65)
private val OnSecondaryLight = Color(0xFFFFFFFF)
private val SecondaryContainerLight = Color(0xFFFFDBD0)
private val OnSecondaryContainerLight = Color(0xFF380D00)

private val SecondaryDark = Color(0xFFFFB59C)
private val OnSecondaryDark = Color(0xFF5A1A00)
private val SecondaryContainerDark = Color(0xFF7D2E00)
private val OnSecondaryContainerDark = Color(0xFFFFDBD0)

// Tertiary Colors - Teal (health metrics, calm)
private val TertiaryLight = Color(0xFF006874)
private val OnTertiaryLight = Color(0xFFFFFFFF)
private val TertiaryContainerLight = Color(0xFF9EEFFD)
private val OnTertiaryContainerLight = Color(0xFF001F24)

private val TertiaryDark = Color(0xFF4FD8EB)
private val OnTertiaryDark = Color(0xFF00363D)
private val TertiaryContainerDark = Color(0xFF004F58)
private val OnTertiaryContainerDark = Color(0xFF9EEFFD)

// Error Colors (M3 prefix to avoid conflict with Color.kt semantic colors)
private val M3ErrorLight = Color(0xFFBA1A1A)
private val OnErrorLight = Color(0xFFFFFFFF)
private val ErrorContainerLight = Color(0xFFFFDAD6)
private val OnErrorContainerLight = Color(0xFF410002)

private val ErrorDark = Color(0xFFFFB4AB)
private val OnErrorDark = Color(0xFF690005)
private val ErrorContainerDark = Color(0xFF93000A)
private val OnErrorContainerDark = Color(0xFFFFDAD6)

// Neutral Colors - Off-white background for medical aesthetic
private val BackgroundLight = Color(0xFFF5F7FA)
private val OnBackgroundLight = Color(0xFF1A1C1E)
private val SurfaceLight = Color(0xFFFFFFFF)
private val OnSurfaceLight = Color(0xFF1A1C1E)
private val SurfaceVariantLight = Color(0xFFE7E0EC)
private val OnSurfaceVariantLight = Color(0xFF49454F)

private val BackgroundDark = Color(0xFF1A1C1E)
private val OnBackgroundDark = Color(0xFFE3E2E6)
private val SurfaceDark = Color(0xFF1A1C1E)
private val OnSurfaceDark = Color(0xFFE3E2E6)
private val SurfaceVariantDark = Color(0xFF49454F)
private val OnSurfaceVariantDark = Color(0xFFCAC4D0)

private val OutlineLight = Color(0xFF79747E)
private val OutlineDark = Color(0xFF938F99)

// AS-Specific Semantic Colors
object InflamAIColors {
    // BASDAI/ASDAS Score Colors
    val ScoreRemission = Color(0xFF4CAF50)         // Green
    val ScoreLowActivity = Color(0xFF8BC34A)       // Light Green
    val ScoreModerateActivity = Color(0xFFFF9800)  // Orange
    val ScoreHighActivity = Color(0xFFFF5722)      // Deep Orange
    val ScoreVeryHighActivity = Color(0xFFF44336) // Red

    // Pain Heatmap Colors
    val PainNone = Color(0xFF4CAF50)
    val PainMild = Color(0xFF8BC34A)
    val PainModerate = Color(0xFFFFEB3B)
    val PainSevere = Color(0xFFFF9800)
    val PainExtreme = Color(0xFFF44336)

    // Trend Colors
    val TrendImproving = Color(0xFF4CAF50)
    val TrendStable = Color(0xFF2196F3)
    val TrendWorsening = Color(0xFFF44336)

    // Flare Status
    val FlareActive = Color(0xFFF44336)
    val FlareResolving = Color(0xFFFF9800)
    val FlareResolved = Color(0xFF4CAF50)

    // Medication Adherence
    val AdherenceGood = Color(0xFF4CAF50)
    val AdherencePartial = Color(0xFFFF9800)
    val AdherencePoor = Color(0xFFF44336)

    // Weather Risk
    val WeatherLowRisk = Color(0xFF4CAF50)
    val WeatherModerateRisk = Color(0xFFFF9800)
    val WeatherHighRisk = Color(0xFFF44336)
}

private val LightColorScheme = lightColorScheme(
    primary = PrimaryLight,
    onPrimary = OnPrimaryLight,
    primaryContainer = PrimaryContainerLight,
    onPrimaryContainer = OnPrimaryContainerLight,
    secondary = SecondaryLight,
    onSecondary = OnSecondaryLight,
    secondaryContainer = SecondaryContainerLight,
    onSecondaryContainer = OnSecondaryContainerLight,
    tertiary = TertiaryLight,
    onTertiary = OnTertiaryLight,
    tertiaryContainer = TertiaryContainerLight,
    onTertiaryContainer = OnTertiaryContainerLight,
    error = M3ErrorLight,
    onError = OnErrorLight,
    errorContainer = ErrorContainerLight,
    onErrorContainer = OnErrorContainerLight,
    background = BackgroundLight,
    onBackground = OnBackgroundLight,
    surface = SurfaceLight,
    onSurface = OnSurfaceLight,
    surfaceVariant = SurfaceVariantLight,
    onSurfaceVariant = OnSurfaceVariantLight,
    outline = OutlineLight
)

private val DarkColorScheme = darkColorScheme(
    primary = PrimaryDark,
    onPrimary = OnPrimaryDark,
    primaryContainer = PrimaryContainerDark,
    onPrimaryContainer = OnPrimaryContainerDark,
    secondary = SecondaryDark,
    onSecondary = OnSecondaryDark,
    secondaryContainer = SecondaryContainerDark,
    onSecondaryContainer = OnSecondaryContainerDark,
    tertiary = TertiaryDark,
    onTertiary = OnTertiaryDark,
    tertiaryContainer = TertiaryContainerDark,
    onTertiaryContainer = OnTertiaryContainerDark,
    error = ErrorDark,
    onError = OnErrorDark,
    errorContainer = ErrorContainerDark,
    onErrorContainer = OnErrorContainerDark,
    background = BackgroundDark,
    onBackground = OnBackgroundDark,
    surface = SurfaceDark,
    onSurface = OnSurfaceDark,
    surfaceVariant = SurfaceVariantDark,
    onSurfaceVariant = OnSurfaceVariantDark,
    outline = OutlineDark
)

@Composable
fun InflamAITheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // Dynamic color is available on Android 12+
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = InflamAITypography,
        content = content
    )
}
