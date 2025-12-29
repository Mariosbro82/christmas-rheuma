package com.inflamai.core.ui.component

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.inflamai.core.ui.theme.*

/**
 * Feature card for onboarding and dashboard
 * Based on Frame Analysis: Icon in colored circle + title + subtitle
 */
@Composable
fun FeatureCard(
    title: String,
    subtitle: String,
    icon: ImageVector,
    iconBackgroundColor: Color,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null
) {
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .then(
                if (onClick != null) Modifier.clickable(onClick = onClick)
                else Modifier
            ),
        shape = RoundedCornerShape(16.dp),
        color = Surface,
        shadowElevation = 2.dp
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Icon in colored circle
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(50))
                    .background(iconBackgroundColor),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = PrimaryBlue,
                    modifier = Modifier.size(24.dp)
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = subtitle,
                    fontSize = 14.sp,
                    color = TextSecondary,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }

            if (onClick != null) {
                Icon(
                    imageVector = Icons.Default.ChevronRight,
                    contentDescription = null,
                    tint = TextTertiary,
                    modifier = Modifier.size(24.dp)
                )
            }
        }
    }
}

/**
 * Stat card for dashboard summary
 * Based on Frame Analysis: 2x2 grid with icon, value, label
 */
@Composable
fun StatCard(
    value: String,
    label: String,
    icon: ImageVector,
    iconBackgroundColor: Color,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null
) {
    Surface(
        modifier = modifier
            .then(
                if (onClick != null) Modifier.clickable(onClick = onClick)
                else Modifier
            ),
        shape = RoundedCornerShape(16.dp),
        color = Surface,
        shadowElevation = 1.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(50))
                    .background(iconBackgroundColor),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = when (iconBackgroundColor) {
                        IconCircleBlue -> PrimaryBlue
                        IconCircleGreen -> Success
                        IconCirclePink -> AccentPink
                        IconCirclePurple -> AccentPurple
                        IconCircleOrange -> AccentOrange
                        else -> TextPrimary
                    },
                    modifier = Modifier.size(24.dp)
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = value,
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )

            Text(
                text = label,
                fontSize = 14.sp,
                color = TextSecondary
            )
        }
    }
}

/**
 * Section header with title and action link
 */
@Composable
fun SectionHeader(
    title: String,
    icon: ImageVector? = null,
    iconBackgroundColor: Color = IconCircleBlue,
    actionText: String? = null,
    onAction: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (icon != null) {
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(RoundedCornerShape(50))
                        .background(iconBackgroundColor),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = icon,
                        contentDescription = null,
                        tint = when (iconBackgroundColor) {
                            IconCircleBlue -> PrimaryBlue
                            IconCircleGreen -> Success
                            IconCirclePink -> AccentPink
                            IconCirclePurple -> AccentPurple
                            IconCircleOrange -> AccentOrange
                            IconCircleTeal -> AccentTeal
                            else -> TextPrimary
                        },
                        modifier = Modifier.size(18.dp)
                    )
                }
                Spacer(modifier = Modifier.width(12.dp))
            }

            Text(
                text = title,
                fontSize = 18.sp,
                fontWeight = FontWeight.SemiBold,
                color = TextPrimary
            )
        }

        if (actionText != null && onAction != null) {
            TextButton(onClick = onAction) {
                Text(
                    text = actionText,
                    fontSize = 14.sp,
                    color = PrimaryBlue
                )
                Icon(
                    imageVector = Icons.Default.ChevronRight,
                    contentDescription = null,
                    tint = PrimaryBlue,
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    }
}

/**
 * Info card with colored background
 * Based on Frame Analysis: Educational cards with different background colors
 */
@Composable
fun InfoCard(
    title: String,
    content: String,
    icon: ImageVector,
    backgroundColor: Color,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        color = backgroundColor
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.Top
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = when (backgroundColor) {
                    WarningLight, IconCircleCream -> Warning
                    InfoLight, IconCircleBlue -> Info
                    SuccessLight, IconCircleGreen -> Success
                    ErrorLight, IconCircleRed -> Error
                    AccentPinkLight -> AccentPink
                    else -> TextPrimary
                },
                modifier = Modifier.size(24.dp)
            )

            Spacer(modifier = Modifier.width(12.dp))

            Column {
                Text(
                    text = title,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = content,
                    fontSize = 14.sp,
                    color = TextSecondary,
                    lineHeight = 20.sp
                )
            }
        }
    }
}

/**
 * Medication card
 * Based on Frame Analysis: Pill icon, name, dosage, frequency, type badge
 */
@Composable
fun MedicationCard(
    name: String,
    genericName: String? = null,
    dosage: String,
    frequency: String,
    type: String,
    hasReminder: Boolean = false,
    isTaken: Boolean = false,
    onTakeClick: () -> Unit,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        color = BackgroundSecondary
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Pill icon
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(50))
                    .background(IconCirclePink),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "💊",
                    fontSize = 20.sp
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = name,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                    if (genericName != null) {
                        Text(
                            text = " ($genericName)",
                            fontSize = 14.sp,
                            color = TextSecondary
                        )
                    }
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "$dosage • $frequency",
                        fontSize = 14.sp,
                        color = TextSecondary
                    )
                    if (hasReminder) {
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(text = "🔔", fontSize = 14.sp)
                    }
                }

                // Type badge
                Surface(
                    modifier = Modifier.padding(top = 4.dp),
                    shape = RoundedCornerShape(8.dp),
                    color = AccentTealLight
                ) {
                    Text(
                        text = type,
                        fontSize = 11.sp,
                        color = AccentTeal,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                    )
                }
            }

            TakeButton(
                onClick = onTakeClick,
                isTaken = isTaken
            )

            Spacer(modifier = Modifier.width(8.dp))

            Icon(
                imageVector = Icons.Default.ChevronRight,
                contentDescription = null,
                tint = TextTertiary,
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

/**
 * Journal entry card
 * Based on Frame Analysis: Date, mood badge, metrics row, tags, content preview
 */
@Composable
fun JournalEntryCard(
    date: String,
    moodStatus: String, // "positive", "challenging", "neutral"
    energyLevel: Int,
    sleepQuality: Int,
    painLevel: Int,
    tags: List<String>,
    contentPreview: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val moodColor = when (moodStatus) {
        "positive" -> Success
        "challenging" -> Error
        else -> TextTertiary
    }
    val moodBgColor = when (moodStatus) {
        "positive" -> SuccessLight
        "challenging" -> ErrorLight
        else -> BackgroundSecondary
    }

    Surface(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        color = Surface
    ) {
        Column {
            Column(modifier = Modifier.padding(16.dp)) {
                // Header: Date + Mood badge
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = date,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )

                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = moodBgColor
                    ) {
                        Text(
                            text = moodStatus,
                            fontSize = 12.sp,
                            color = moodColor,
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Metrics row
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text(text = "⚡ $energyLevel/10", fontSize = 13.sp, color = TextSecondary)
                    Text(text = "🛏 $sleepQuality/10", fontSize = 13.sp, color = TextSecondary)
                    Text(text = "🔺 $painLevel/10", fontSize = 13.sp, color = TextSecondary)
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Tags
                if (tags.isNotEmpty()) {
                    Text(
                        text = tags.joinToString(", "),
                        fontSize = 13.sp,
                        color = TextTertiary,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Content preview
                Text(
                    text = contentPreview,
                    fontSize = 14.sp,
                    color = TextSecondary,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    lineHeight = 20.sp
                )
            }

            HorizontalDivider(color = Divider)
        }
    }
}

/**
 * Exercise card for library
 * Based on Frame Analysis: Thumbnail, title, level badge, duration, target areas
 */
@Composable
fun ExerciseCard(
    name: String,
    level: String,
    durationMinutes: Int,
    targetAreas: String,
    thumbnailColor: Color = IconCircleBlue,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val levelColor = when (level.uppercase()) {
        "BEGINNER" -> DifficultyBeginner
        "INTERMEDIATE" -> DifficultyIntermediate
        "ADVANCED" -> DifficultyAdvanced
        else -> TextSecondary
    }

    Surface(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        color = Surface,
        shadowElevation = 1.dp
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Thumbnail
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(thumbnailColor),
                contentAlignment = Alignment.Center
            ) {
                Text(text = "🧘", fontSize = 32.sp)
            }

            Spacer(modifier = Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = name,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )

                Spacer(modifier = Modifier.height(4.dp))

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = level,
                        fontSize = 13.sp,
                        color = levelColor,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = " • $durationMinutes min",
                        fontSize = 13.sp,
                        color = TextSecondary
                    )
                }

                Spacer(modifier = Modifier.height(2.dp))

                Text(
                    text = targetAreas,
                    fontSize = 13.sp,
                    color = AccentTeal
                )
            }

            Icon(
                imageVector = Icons.Default.ChevronRight,
                contentDescription = null,
                tint = TextTertiary,
                modifier = Modifier.size(24.dp)
            )
        }
    }
}

/**
 * Trend row for dashboard
 * Based on Frame Analysis: Label, value, trend status with icon
 */
@Composable
fun TrendRow(
    label: String,
    value: String,
    trendStatus: String, // "stable", "improving", "worsening"
    modifier: Modifier = Modifier
) {
    val (trendText, trendColor) = when (trendStatus.lowercase()) {
        "improving" -> "↓ Improving" to TrendImproving
        "worsening" -> "↑ Worsening" to TrendWorsening
        else -> "— Stable" to TrendStable
    }

    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            fontSize = 15.sp,
            color = TextSecondary
        )

        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = value,
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold,
                color = TextPrimary
            )

            Text(
                text = trendText,
                fontSize = 13.sp,
                color = trendColor,
                fontWeight = FontWeight.Medium
            )
        }
    }
}
