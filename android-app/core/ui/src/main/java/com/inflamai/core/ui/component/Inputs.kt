package com.inflamai.core.ui.component

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CalendarToday
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.inflamai.core.ui.theme.*
import com.inflamai.core.ui.theme.SliderColors

/**
 * Pain level slider with gradient track
 * Based on Frame Analysis: 0-10 scale, red gradient
 */
@Composable
fun PainLevelSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
    valueRange: ClosedFloatingPointRange<Float> = 0f..10f
) {
    Column(modifier = modifier) {
        // Value display
        Text(
            text = "${value.toInt()}/10",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary,
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Gradient slider track
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(
                    brush = Brush.horizontalGradient(SliderColors.pain)
                )
        )

        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            steps = 9,
            modifier = Modifier.fillMaxWidth(),
            colors = SliderDefaults.colors(
                thumbColor = Error,
                activeTrackColor = Color.Transparent,
                inactiveTrackColor = Color.Transparent
            )
        )

        // Labels
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = "None", fontSize = 12.sp, color = TextTertiary)
            Text(text = "Severe", fontSize = 12.sp, color = TextTertiary)
        }
    }
}

/**
 * Stiffness slider (minutes)
 * Based on Frame Analysis: Orange gradient, 0-120+ minutes
 */
@Composable
fun StiffnessSlider(
    minutes: Int,
    onMinutesChange: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = if (minutes >= 120) "120+ min" else "$minutes min",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary,
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(16.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(
                    brush = Brush.horizontalGradient(SliderColors.stiffness)
                )
        )

        Slider(
            value = minutes.toFloat(),
            onValueChange = { onMinutesChange(it.toInt()) },
            valueRange = 0f..120f,
            steps = 11,
            modifier = Modifier.fillMaxWidth(),
            colors = SliderDefaults.colors(
                thumbColor = StiffnessColor,
                activeTrackColor = Color.Transparent,
                inactiveTrackColor = Color.Transparent
            )
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = "0 min", fontSize = 12.sp, color = TextTertiary)
            Text(text = "120+ min", fontSize = 12.sp, color = TextTertiary)
        }
    }
}

/**
 * Fatigue slider
 * Based on Frame Analysis: Purple gradient, 0-10
 */
@Composable
fun FatigueSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = "${value.toInt()}/10",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary,
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(16.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(
                    brush = Brush.horizontalGradient(SliderColors.fatigue)
                )
        )

        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0f..10f,
            steps = 9,
            modifier = Modifier.fillMaxWidth(),
            colors = SliderDefaults.colors(
                thumbColor = FatigueColor,
                activeTrackColor = Color.Transparent,
                inactiveTrackColor = Color.Transparent
            )
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = "None", fontSize = 12.sp, color = TextTertiary)
            Text(text = "Exhausted", fontSize = 12.sp, color = TextTertiary)
        }
    }
}

/**
 * BASDAI question slider (0-10)
 * Based on Frame Analysis: Yellow-orange gradient for BASDAI
 */
@Composable
fun BASDAISlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    minLabel: String = "None",
    maxLabel: String = "Very severe",
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = if (value == 0f) "— /10" else "${String.format("%.1f", value)} /10",
            fontSize = 48.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary,
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(24.dp))

        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0f..10f,
            steps = 19, // 0.5 increments
            modifier = Modifier.fillMaxWidth(),
            colors = SliderDefaults.colors(
                thumbColor = AccentOrange,
                activeTrackColor = AccentOrange,
                inactiveTrackColor = BackgroundSecondary
            )
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = minLabel, fontSize = 13.sp, color = TextSecondary)
            Text(text = maxLabel, fontSize = 13.sp, color = TextSecondary)
        }
    }
}

/**
 * Mood selector (5 emoji options)
 * Based on Frame Analysis: Great, Good, Okay, Poor, Bad
 */
@Composable
fun MoodSelector(
    selectedMood: String?,
    onMoodSelected: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val moods = listOf(
        "great" to "😊",
        "good" to "🙂",
        "okay" to "😐",
        "poor" to "😣",
        "bad" to "😰"
    )

    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        moods.forEach { (mood, emoji) ->
            val isSelected = selectedMood == mood

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .clip(RoundedCornerShape(12.dp))
                    .clickable { onMoodSelected(mood) }
                    .background(if (isSelected) PrimaryBlueLight else Color.Transparent)
                    .padding(12.dp)
            ) {
                Text(
                    text = emoji,
                    fontSize = 32.sp
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = mood.replaceFirstChar { it.uppercase() },
                    fontSize = 12.sp,
                    color = if (isSelected) PrimaryBlue else TextSecondary,
                    fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal
                )
            }
        }
    }
}

/**
 * Date picker field
 */
@Composable
fun DatePickerField(
    label: String,
    value: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = label,
            fontSize = 14.sp,
            color = TextSecondary,
            fontWeight = FontWeight.Medium
        )

        Spacer(modifier = Modifier.height(8.dp))

        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(onClick = onClick),
            shape = RoundedCornerShape(12.dp),
            color = BackgroundSecondary
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.CalendarToday,
                    contentDescription = null,
                    tint = TextSecondary,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = value.ifEmpty { "Select date" },
                    fontSize = 16.sp,
                    color = if (value.isEmpty()) TextTertiary else TextPrimary
                )
            }
        }
    }
}

/**
 * Dropdown field
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DropdownField(
    label: String,
    value: String,
    options: List<String>,
    onOptionSelected: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    var expanded by remember { mutableStateOf(false) }

    Column(modifier = modifier) {
        Text(
            text = label,
            fontSize = 14.sp,
            color = TextSecondary,
            fontWeight = FontWeight.Medium
        )

        Spacer(modifier = Modifier.height(8.dp))

        ExposedDropdownMenuBox(
            expanded = expanded,
            onExpandedChange = { expanded = !expanded }
        ) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .menuAnchor(),
                shape = RoundedCornerShape(12.dp),
                color = BackgroundSecondary
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = value.ifEmpty { "Select..." },
                        fontSize = 16.sp,
                        color = if (value.isEmpty()) TextTertiary else TextPrimary,
                        modifier = Modifier.weight(1f)
                    )
                    Icon(
                        imageVector = Icons.Default.KeyboardArrowDown,
                        contentDescription = null,
                        tint = TextSecondary
                    )
                }
            }

            ExposedDropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                options.forEach { option ->
                    DropdownMenuItem(
                        text = { Text(option) },
                        onClick = {
                            onOptionSelected(option)
                            expanded = false
                        }
                    )
                }
            }
        }
    }
}

/**
 * Text input field
 */
@Composable
fun TextInputField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    placeholder: String = "",
    maxLines: Int = 1,
    minLines: Int = 1
) {
    Column(modifier = modifier) {
        Text(
            text = label,
            fontSize = 14.sp,
            color = TextSecondary,
            fontWeight = FontWeight.Medium
        )

        Spacer(modifier = Modifier.height(8.dp))

        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.fillMaxWidth(),
            placeholder = {
                Text(
                    text = placeholder,
                    color = TextTertiary
                )
            },
            maxLines = maxLines,
            minLines = minLines,
            shape = RoundedCornerShape(12.dp),
            colors = OutlinedTextFieldDefaults.colors(
                unfocusedBorderColor = Border,
                focusedBorderColor = PrimaryBlue,
                unfocusedContainerColor = BackgroundSecondary,
                focusedContainerColor = Surface
            )
        )
    }
}

/**
 * Segmented button toggle (e.g., Imperial/Metric)
 */
@Composable
fun SegmentedToggle(
    options: List<String>,
    selectedIndex: Int,
    onOptionSelected: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(24.dp))
            .background(BackgroundSecondary)
            .padding(4.dp)
    ) {
        options.forEachIndexed { index, option ->
            val isSelected = index == selectedIndex

            Surface(
                modifier = Modifier
                    .weight(1f)
                    .clickable { onOptionSelected(index) },
                shape = RoundedCornerShape(20.dp),
                color = if (isSelected) Surface else Color.Transparent,
                shadowElevation = if (isSelected) 2.dp else 0.dp
            ) {
                Text(
                    text = option,
                    fontSize = 14.sp,
                    fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
                    color = if (isSelected) PrimaryBlue else TextSecondary,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

/**
 * Reps counter with plus/minus buttons
 * Based on Frame Analysis: Exercise execution reps step
 */
@Composable
fun RepsCounter(
    current: Int,
    target: Int,
    onIncrement: () -> Unit,
    onDecrement: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Row(verticalAlignment = Alignment.Bottom) {
            Text(
                text = "$current",
                fontSize = 64.sp,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )
            Text(
                text = " / $target",
                fontSize = 32.sp,
                color = TextSecondary,
                modifier = Modifier.padding(bottom = 8.dp)
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Progress bar
        val progress = (current.toFloat() / target).coerceIn(0f, 1f)
        LinearProgressIndicator(
            progress = { progress },
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp)),
            color = if (current >= target) Success else PrimaryBlue,
            trackColor = BackgroundSecondary
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Control buttons
        Row(
            horizontalArrangement = Arrangement.spacedBy(32.dp)
        ) {
            // Minus button
            IconButton(
                onClick = onDecrement,
                modifier = Modifier
                    .size(56.dp)
                    .clip(RoundedCornerShape(50))
                    .background(ErrorLight)
            ) {
                Text(
                    text = "−",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    color = Error
                )
            }

            // Plus button
            IconButton(
                onClick = onIncrement,
                modifier = Modifier
                    .size(56.dp)
                    .clip(RoundedCornerShape(50))
                    .background(PrimaryBlueLight)
            ) {
                Text(
                    text = "+",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    color = PrimaryBlue
                )
            }
        }
    }
}
