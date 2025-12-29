package com.inflamai.core.ui.component

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.inflamai.core.ui.theme.*

/**
 * Page indicator dots for onboarding/pagers
 * Based on Frame Analysis: 8dp dots, 6dp spacing, active = PrimaryBlue
 */
@Composable
fun PageIndicator(
    pageCount: Int,
    currentPage: Int,
    modifier: Modifier = Modifier,
    activeColor: Color = PrimaryBlue,
    inactiveColor: Color = BackgroundTertiary,
    dotSize: Dp = 8.dp,
    spacing: Dp = 6.dp
) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(spacing),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(pageCount) { index ->
            val color by animateColorAsState(
                targetValue = if (index == currentPage) activeColor else inactiveColor,
                animationSpec = tween(300),
                label = "dot_color"
            )

            Box(
                modifier = Modifier
                    .size(dotSize)
                    .clip(CircleShape)
                    .background(color)
            )
        }
    }
}

/**
 * Step progress indicator for wizards
 * Based on Frame Analysis: Numbered circles with connecting lines
 */
@Composable
fun StepProgressIndicator(
    totalSteps: Int,
    currentStep: Int,
    modifier: Modifier = Modifier,
    completedColor: Color = PrimaryBlue,
    currentColor: Color = PrimaryBlue,
    pendingColor: Color = BackgroundTertiary,
    stepLabels: List<String>? = null
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        for (step in 0 until totalSteps) {
            val isCompleted = step < currentStep
            val isCurrent = step == currentStep

            val circleColor = when {
                isCompleted -> completedColor
                isCurrent -> currentColor
                else -> pendingColor
            }

            val textColor = when {
                isCompleted || isCurrent -> Color.White
                else -> TextTertiary
            }

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.weight(1f)
            ) {
                // Step circle
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(circleColor),
                    contentAlignment = Alignment.Center
                ) {
                    if (isCompleted) {
                        Text(
                            text = "✓",
                            color = textColor,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                    } else {
                        Text(
                            text = "${step + 1}",
                            color = textColor,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }

                // Step label
                if (stepLabels != null && step < stepLabels.size) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = stepLabels[step],
                        fontSize = 11.sp,
                        color = if (isCurrent) TextPrimary else TextTertiary,
                        textAlign = TextAlign.Center,
                        maxLines = 1
                    )
                }
            }

            // Connecting line (except after last step)
            if (step < totalSteps - 1) {
                Box(
                    modifier = Modifier
                        .weight(0.5f)
                        .height(2.dp)
                        .background(if (step < currentStep) completedColor else pendingColor)
                )
            }
        }
    }
}

/**
 * Circular progress indicator (for meditation timer, exercise timer)
 * Based on Frame Analysis: Large circular timer with progress arc
 */
@Composable
fun CircularProgressTimer(
    progress: Float, // 0f to 1f
    timeText: String,
    modifier: Modifier = Modifier,
    size: Dp = 200.dp,
    strokeWidth: Dp = 12.dp,
    progressColor: Color = PrimaryBlue,
    trackColor: Color = BackgroundSecondary,
    subtitle: String? = null
) {
    Box(
        modifier = modifier.size(size),
        contentAlignment = Alignment.Center
    ) {
        // Background track
        Canvas(modifier = Modifier.fillMaxSize()) {
            val stroke = Stroke(
                width = strokeWidth.toPx(),
                cap = StrokeCap.Round
            )

            // Track
            drawArc(
                color = trackColor,
                startAngle = -90f,
                sweepAngle = 360f,
                useCenter = false,
                style = stroke,
                size = Size(size.toPx() - strokeWidth.toPx(), size.toPx() - strokeWidth.toPx()),
                topLeft = Offset(strokeWidth.toPx() / 2, strokeWidth.toPx() / 2)
            )

            // Progress
            drawArc(
                color = progressColor,
                startAngle = -90f,
                sweepAngle = 360f * progress,
                useCenter = false,
                style = stroke,
                size = Size(size.toPx() - strokeWidth.toPx(), size.toPx() - strokeWidth.toPx()),
                topLeft = Offset(strokeWidth.toPx() / 2, strokeWidth.toPx() / 2)
            )
        }

        // Center content
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = timeText,
                fontSize = 48.sp,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )
            if (subtitle != null) {
                Text(
                    text = subtitle,
                    fontSize = 14.sp,
                    color = TextSecondary
                )
            }
        }
    }
}

/**
 * Linear progress with percentage
 * Based on Frame Analysis: Exercise completion, adherence bars
 */
@Composable
fun LinearProgressWithLabel(
    progress: Float,
    label: String,
    modifier: Modifier = Modifier,
    progressColor: Color = PrimaryBlue,
    trackColor: Color = BackgroundSecondary,
    showPercentage: Boolean = true
) {
    Column(modifier = modifier) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                fontSize = 14.sp,
                color = TextSecondary
            )
            if (showPercentage) {
                Text(
                    text = "${(progress * 100).toInt()}%",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        LinearProgressIndicator(
            progress = { progress.coerceIn(0f, 1f) },
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp)),
            color = progressColor,
            trackColor = trackColor
        )
    }
}

/**
 * BASDAI score badge
 * Based on Frame Analysis: Colored badge showing score category
 */
@Composable
fun BASDAIScoreBadge(
    score: Float,
    modifier: Modifier = Modifier,
    showLabel: Boolean = true
) {
    val badgeInfo: Triple<Color, Color, String> = when {
        score < 2 -> Triple(BASDAIRemissionLight, BASDAIRemission, "Remission")
        score < 4 -> Triple(BASDAILowLight, BASDAILow, "Low Activity")
        score < 6 -> Triple(BASDAIModerateLight, BASDAIModerate, "Moderate")
        else -> Triple(BASDAIHighLight, BASDAIHigh, "High Activity")
    }
    val backgroundColor = badgeInfo.first
    val textColor = badgeInfo.second
    val label = badgeInfo.third

    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        color = backgroundColor
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = String.format("%.1f", score),
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = textColor
            )
            if (showLabel) {
                Text(
                    text = label,
                    fontSize = 13.sp,
                    color = textColor
                )
            }
        }
    }
}

/**
 * Streak badge (flame icon with count)
 * Based on Frame Analysis: Streak indicator on dashboard
 */
@Composable
fun StreakBadge(
    streakDays: Int,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(16.dp),
        color = StreakFlameLight
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = "🔥",
                fontSize = 16.sp
            )
            Text(
                text = "$streakDays day${if (streakDays != 1) "s" else ""}",
                fontSize = 14.sp,
                fontWeight = FontWeight.SemiBold,
                color = StreakFlame
            )
        }
    }
}

/**
 * Notification badge (for tab icons, etc.)
 */
@Composable
fun NotificationBadge(
    count: Int,
    modifier: Modifier = Modifier
) {
    if (count > 0) {
        Surface(
            modifier = modifier.size(18.dp),
            shape = CircleShape,
            color = Error
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(
                    text = if (count > 9) "9+" else count.toString(),
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
        }
    }
}

/**
 * Severity indicator dot
 * Based on Frame Analysis: Pain level indicators
 */
@Composable
fun SeverityDot(
    level: Int, // 0-10
    modifier: Modifier = Modifier,
    size: Dp = 12.dp
) {
    val dotColor: Color = when {
        level <= 2 -> SeverityNone
        level <= 4 -> SeverityMild
        level <= 6 -> SeverityModerate
        level <= 8 -> SeveritySevere
        else -> SeverityExtreme
    }

    Box(
        modifier = modifier
            .size(size)
            .clip(CircleShape)
            .background(color = dotColor)
    )
}

/**
 * Trend indicator arrow
 * Based on Frame Analysis: Trend arrows in dashboard stats
 */
@Composable
fun TrendIndicator(
    trend: String, // "improving", "stable", "worsening"
    modifier: Modifier = Modifier
) {
    val (icon, color) = when (trend.lowercase()) {
        "improving" -> "↓" to TrendImproving
        "worsening" -> "↑" to TrendWorsening
        else -> "—" to TrendStable
    }

    Text(
        text = icon,
        fontSize = 16.sp,
        fontWeight = FontWeight.Bold,
        color = color,
        modifier = modifier
    )
}

/**
 * Loading spinner with optional message
 */
@Composable
fun LoadingIndicator(
    modifier: Modifier = Modifier,
    message: String? = null
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    )  {
        CircularProgressIndicator(
            color = PrimaryBlue,
            strokeWidth = 4.dp
        )
        if (message != null) {
            Text(
                text = message,
                fontSize = 14.sp,
                color = TextSecondary
            )
        }
    }
}

/**
 * Breathing animation circle for meditation
 * Based on Frame Analysis: Pulsing circle during breathing exercises
 */
@Composable
fun BreathingCircle(
    phase: String, // "inhale", "hold", "exhale"
    modifier: Modifier = Modifier,
    size: Dp = 200.dp
) {
    val infiniteTransition = rememberInfiniteTransition(label = "breathing")

    val scale by infiniteTransition.animateFloat(
        initialValue = when (phase) {
            "inhale" -> 0.6f
            "hold" -> 1f
            "exhale" -> 1f
            else -> 0.8f
        },
        targetValue = when (phase) {
            "inhale" -> 1f
            "hold" -> 1f
            "exhale" -> 0.6f
            else -> 0.8f
        },
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = when (phase) {
                    "inhale" -> 4000
                    "hold" -> 2000
                    "exhale" -> 4000
                    else -> 1000
                },
                easing = LinearEasing
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    val color = when (phase) {
        "inhale" -> MeditationBreathIn
        "hold" -> MeditationHold
        "exhale" -> MeditationBreathOut
        else -> PrimaryBlue
    }

    Box(
        modifier = modifier
            .size(size * scale)
            .clip(CircleShape)
            .background(color.copy(alpha = 0.3f)),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .size(size * scale * 0.7f)
                .clip(CircleShape)
                .background(color.copy(alpha = 0.5f))
        )

        Text(
            text = phase.replaceFirstChar { it.uppercase() },
            fontSize = 24.sp,
            fontWeight = FontWeight.Medium,
            color = TextPrimary
        )
    }
}
