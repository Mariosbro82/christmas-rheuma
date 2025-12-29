package com.inflamai.feature.quickcapture

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.inflamai.core.ui.component.*
import com.inflamai.core.ui.theme.*
import com.inflamai.core.ui.theme.SliderColors

/**
 * Quick symptom options for fast logging
 */
data class QuickSymptom(
    val id: String,
    val name: String,
    val emoji: String,
    val color: Color
)

val quickSymptoms = listOf(
    QuickSymptom("pain", "Pain", "🔥", Error),
    QuickSymptom("stiffness", "Stiffness", "🦴", StiffnessColor),
    QuickSymptom("fatigue", "Fatigue", "😴", FatigueColor),
    QuickSymptom("swelling", "Swelling", "💧", AccentTeal),
    QuickSymptom("inflammation", "Inflammation", "🌡️", AccentOrange),
    QuickSymptom("sleep_issues", "Sleep Issues", "🌙", AccentPurple)
)

/**
 * Quick body regions for fast tapping
 */
val quickBodyRegions = listOf(
    "Lower Back" to "si_left",
    "SI Joints" to "si_right",
    "Neck" to "c7",
    "Upper Back" to "t6",
    "Hip Left" to "hip_left",
    "Hip Right" to "hip_right",
    "Knee Left" to "knee_left",
    "Knee Right" to "knee_right"
)

/**
 * Quick Capture / SOS Flare Screen
 * Based on Frame Analysis: Fast symptom logging during flares
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun QuickCaptureScreen(
    onNavigateBack: () -> Unit = {},
    onComplete: () -> Unit = {},
    viewModel: QuickCaptureViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val haptic = LocalHapticFeedback.current

    // Completion screen
    if (uiState.isComplete) {
        QuickCaptureCompleteScreen(
            painLevel = uiState.painLevel,
            onDone = onComplete
        )
        return
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = null,
                            tint = Error,
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("SOS Flare Log")
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Close")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = ErrorLight,
                    titleContentColor = TextPrimary
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
        ) {
            // Emergency header
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        brush = Brush.verticalGradient(
                            colors = listOf(ErrorLight, Background)
                        )
                    )
                    .padding(24.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Quick Symptom Capture",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                    Text(
                        text = "Log your flare symptoms in seconds",
                        fontSize = 14.sp,
                        color = TextSecondary
                    )
                }
            }

            // Pain Level Section
            Column(
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                SectionHeader(
                    title = "Pain Level",
                    icon = Icons.Default.Warning,
                    iconBackgroundColor = IconCircleRed
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Large pain display
                Text(
                    text = "${uiState.painLevel}/10",
                    fontSize = 48.sp,
                    fontWeight = FontWeight.Bold,
                    color = getPainColor(uiState.painLevel),
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center
                )

                Text(
                    text = getPainLabel(uiState.painLevel),
                    fontSize = 16.sp,
                    color = TextSecondary,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Pain slider with gradient
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
                    value = uiState.painLevel.toFloat(),
                    onValueChange = {
                        haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                        viewModel.updatePainLevel(it.toInt())
                    },
                    valueRange = 0f..10f,
                    steps = 9,
                    modifier = Modifier.fillMaxWidth(),
                    colors = SliderDefaults.colors(
                        thumbColor = getPainColor(uiState.painLevel),
                        activeTrackColor = Color.Transparent,
                        inactiveTrackColor = Color.Transparent
                    )
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(text = "None", fontSize = 12.sp, color = TextTertiary)
                    Text(text = "Severe", fontSize = 12.sp, color = TextTertiary)
                }
            }

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // Quick Symptoms
            Column(
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                SectionHeader(
                    title = "What are you experiencing?",
                    icon = Icons.Default.Checklist,
                    iconBackgroundColor = IconCircleOrange
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Symptom chips
                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(quickSymptoms) { symptom ->
                        QuickSymptomChip(
                            symptom = symptom,
                            isSelected = uiState.selectedSymptoms.contains(symptom.id),
                            onClick = {
                                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                viewModel.toggleSymptom(symptom.id)
                            }
                        )
                    }
                }
            }

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // Quick Body Region
            Column(
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                SectionHeader(
                    title = "Where does it hurt?",
                    icon = Icons.Default.Accessibility,
                    iconBackgroundColor = IconCircleTeal
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Body region quick buttons
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    quickBodyRegions.chunked(2).forEach { rowItems ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            rowItems.forEach { (name, regionId) ->
                                QuickRegionButton(
                                    name = name,
                                    isSelected = uiState.selectedRegions.contains(regionId),
                                    onClick = {
                                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                        viewModel.toggleRegion(regionId)
                                    },
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }
                    }
                }
            }

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // Optional notes
            Column(
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                SectionHeader(
                    title = "Quick Note (Optional)",
                    icon = Icons.Default.Edit,
                    iconBackgroundColor = IconCircleBlue
                )

                Spacer(modifier = Modifier.height(12.dp))

                OutlinedTextField(
                    value = uiState.notes,
                    onValueChange = { viewModel.updateNotes(it) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(80.dp),
                    placeholder = { Text("Any additional notes...", color = TextTertiary) },
                    maxLines = 3,
                    shape = RoundedCornerShape(12.dp)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Action buttons
            Column(
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                // Save button
                Button(
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                        viewModel.saveQuickCapture()
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Error),
                    shape = RoundedCornerShape(28.dp),
                    enabled = !uiState.isSaving
                ) {
                    if (uiState.isSaving) {
                        CircularProgressIndicator(
                            color = Color.White,
                            modifier = Modifier.size(24.dp),
                            strokeWidth = 2.dp
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Default.Save,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Save Flare Log",
                            fontSize = 17.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Cancel button
                OutlinedButton(
                    onClick = onNavigateBack,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(28.dp)
                ) {
                    Text("Cancel", fontSize = 16.sp)
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Disclaimer
                Text(
                    text = "⚠️ If symptoms are severe or new, please consult your healthcare provider.",
                    fontSize = 12.sp,
                    color = TextTertiary,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }
}

/**
 * Quick symptom chip
 */
@Composable
fun QuickSymptomChip(
    symptom: QuickSymptom,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Surface(
        modifier = Modifier
            .clip(RoundedCornerShape(16.dp))
            .clickable(onClick = onClick),
        color = if (isSelected) symptom.color else BackgroundSecondary,
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text = symptom.emoji, fontSize = 18.sp)
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = symptom.name,
                fontSize = 14.sp,
                fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
                color = if (isSelected) Color.White else TextPrimary
            )
        }
    }
}

/**
 * Quick region button
 */
@Composable
fun QuickRegionButton(
    name: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClick = onClick),
        color = if (isSelected) PrimaryBlue else BackgroundSecondary,
        shape = RoundedCornerShape(12.dp)
    ) {
        Box(
            modifier = Modifier.padding(vertical = 14.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = name,
                fontSize = 14.sp,
                fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
                color = if (isSelected) Color.White else TextPrimary
            )
        }
    }
}

/**
 * Completion screen after saving
 */
@Composable
fun QuickCaptureCompleteScreen(
    painLevel: Int,
    onDone: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Background),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(SuccessLight),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Check,
                    contentDescription = null,
                    tint = Success,
                    modifier = Modifier.size(48.dp)
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "Flare Logged",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Pain level: $painLevel/10",
                fontSize = 16.sp,
                color = TextSecondary
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "Your symptoms have been recorded. This data will help identify patterns and triggers.",
                fontSize = 14.sp,
                color = TextSecondary,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(32.dp))

            PrimaryButton(
                text = "Done",
                onClick = onDone
            )
        }
    }
}

/**
 * Get color for pain level (0-10)
 */
private fun getPainColor(painLevel: Int): Color {
    return when {
        painLevel == 0 -> TextTertiary
        painLevel <= 2 -> SeverityNone
        painLevel <= 4 -> SeverityMild
        painLevel <= 6 -> SeverityModerate
        painLevel <= 8 -> SeveritySevere
        else -> SeverityExtreme
    }
}

/**
 * Get label for pain level
 */
private fun getPainLabel(level: Int): String {
    return when {
        level == 0 -> "No Pain"
        level <= 2 -> "Mild"
        level <= 4 -> "Moderate"
        level <= 6 -> "Significant"
        level <= 8 -> "Severe"
        else -> "Extreme"
    }
}
