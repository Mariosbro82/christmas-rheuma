package com.inflamai.feature.checkin.ui

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.inflamai.core.domain.calculator.BASDAIInterpretation
import com.inflamai.core.ui.theme.InflamAIColors
import com.inflamai.feature.checkin.viewmodel.CheckInQuestion
import com.inflamai.feature.checkin.viewmodel.CheckInUiState
import com.inflamai.feature.checkin.viewmodel.CheckInViewModel
import com.inflamai.feature.checkin.viewmodel.QuestionType

/**
 * Check-In Screen
 *
 * Step-by-step BASDAI questionnaire with:
 * - Progress indicator
 * - Large touch-friendly sliders
 * - Haptic feedback
 * - Score preview
 * - Accessibility support
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CheckInScreen(
    onNavigateBack: () -> Unit,
    onComplete: () -> Unit,
    viewModel: CheckInViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val haptic = LocalHapticFeedback.current

    val currentQuestion = viewModel.questions.getOrNull(uiState.currentQuestionIndex)
    val isLastQuestion = uiState.currentQuestionIndex == viewModel.questions.size - 1
    val isFirstQuestion = uiState.currentQuestionIndex == 0

    // Show completion screen
    if (uiState.isComplete) {
        CompletionScreen(
            score = uiState.finalScore ?: 0.0,
            interpretation = uiState.interpretation,
            onDone = onComplete
        )
        return
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Daily Check-in") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.Close, contentDescription = "Close")
                    }
                },
                actions = {
                    // Score preview after BASDAI questions
                    if (uiState.currentQuestionIndex >= 6) {
                        val previewScore = viewModel.calculatePreviewScore()
                        Text(
                            text = "Score: ${String.format("%.1f", previewScore)}",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(end = 16.dp)
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Progress indicator
            LinearProgressIndicator(
                progress = { (uiState.currentQuestionIndex + 1).toFloat() / viewModel.questions.size },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp),
            )

            // Question counter
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Question ${uiState.currentQuestionIndex + 1} of ${viewModel.questions.size}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                currentQuestion?.let {
                    Text(
                        text = it.title,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            // Question content
            currentQuestion?.let { question ->
                AnimatedContent(
                    targetState = question,
                    transitionSpec = {
                        slideInHorizontally { width -> width } + fadeIn() togetherWith
                            slideOutHorizontally { width -> -width } + fadeOut()
                    },
                    label = "question_transition"
                ) { targetQuestion ->
                    QuestionContent(
                        question = targetQuestion,
                        currentValue = uiState.answers[targetQuestion.id],
                        onValueChange = { value ->
                            haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                            viewModel.updateAnswer(targetQuestion.id, value)
                        },
                        modifier = Modifier
                            .weight(1f)
                            .padding(16.dp)
                    )
                }
            }

            // Error message
            uiState.error?.let { error ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Error,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.error
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = error,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                    }
                }
            }

            // Navigation buttons
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Back button
                if (!isFirstQuestion) {
                    OutlinedButton(
                        onClick = { viewModel.previousQuestion() },
                        modifier = Modifier.weight(1f)
                    ) {
                        Icon(Icons.Default.ArrowBack, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Back")
                    }
                } else {
                    Spacer(modifier = Modifier.weight(1f))
                }

                // Next/Submit button
                Button(
                    onClick = {
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                        if (isLastQuestion) {
                            viewModel.submitCheckIn(onComplete)
                        } else {
                            viewModel.nextQuestion()
                        }
                    },
                    modifier = Modifier.weight(1f),
                    enabled = !uiState.isSubmitting
                ) {
                    if (uiState.isSubmitting) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text(if (isLastQuestion) "Submit" else "Next")
                        Spacer(modifier = Modifier.width(8.dp))
                        Icon(
                            if (isLastQuestion) Icons.Default.Check else Icons.Default.ArrowForward,
                            contentDescription = null
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun QuestionContent(
    question: CheckInQuestion,
    currentValue: Any?,
    onValueChange: (Any) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = question.question,
            style = MaterialTheme.typography.headlineSmall,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 32.dp)
        )

        when (question.type) {
            QuestionType.SLIDER -> {
                SliderQuestion(
                    value = (currentValue as? Number)?.toFloat() ?: question.minValue.toFloat(),
                    minValue = question.minValue,
                    maxValue = question.maxValue,
                    minLabel = question.minLabel,
                    maxLabel = question.maxLabel,
                    onValueChange = { onValueChange(it) }
                )
            }

            QuestionType.DURATION -> {
                DurationQuestion(
                    minutes = (currentValue as? Number)?.toInt() ?: 0,
                    maxMinutes = question.maxValue,
                    onValueChange = { onValueChange(it) }
                )
            }

            QuestionType.TEXT -> {
                TextQuestion(
                    text = currentValue as? String ?: "",
                    onValueChange = { onValueChange(it) },
                    isOptional = question.isOptional
                )
            }
        }
    }
}

@Composable
fun SliderQuestion(
    value: Float,
    minValue: Int,
    maxValue: Int,
    minLabel: String,
    maxLabel: String,
    onValueChange: (Float) -> Unit
) {
    val scoreColor = when {
        value <= 2 -> InflamAIColors.ScoreRemission
        value <= 4 -> InflamAIColors.ScoreLowActivity
        value <= 6 -> InflamAIColors.ScoreModerateActivity
        value <= 8 -> InflamAIColors.ScoreHighActivity
        else -> InflamAIColors.ScoreVeryHighActivity
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth()
    ) {
        // Large value display
        Box(
            modifier = Modifier
                .size(120.dp)
                .clip(CircleShape)
                .background(scoreColor.copy(alpha = 0.15f)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = value.toInt().toString(),
                style = MaterialTheme.typography.displayLarge,
                fontWeight = FontWeight.Bold,
                color = scoreColor
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Slider
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = minValue.toFloat()..maxValue.toFloat(),
            steps = maxValue - minValue - 1,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .semantics {
                    contentDescription = "Score slider, current value ${value.toInt()}"
                },
            colors = SliderDefaults.colors(
                thumbColor = scoreColor,
                activeTrackColor = scoreColor
            )
        )

        // Labels
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = minLabel,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = maxLabel,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun DurationQuestion(
    minutes: Int,
    maxMinutes: Int,
    onValueChange: (Int) -> Unit
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth()
    ) {
        // Duration display
        Text(
            text = when {
                minutes == 0 -> "None"
                minutes < 60 -> "$minutes minutes"
                minutes == 60 -> "1 hour"
                minutes >= 120 -> "2+ hours"
                else -> "${minutes / 60}h ${minutes % 60}m"
            },
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(32.dp))

        // Quick select buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            listOf(0, 15, 30, 60, 120).forEach { mins ->
                FilterChip(
                    selected = minutes == mins,
                    onClick = { onValueChange(mins) },
                    label = {
                        Text(
                            when (mins) {
                                0 -> "None"
                                60 -> "1h"
                                120 -> "2h+"
                                else -> "${mins}m"
                            }
                        )
                    }
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Fine-tune slider
        Slider(
            value = minutes.toFloat(),
            onValueChange = { onValueChange(it.toInt()) },
            valueRange = 0f..maxMinutes.toFloat(),
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        )
    }
}

@Composable
fun TextQuestion(
    text: String,
    onValueChange: (String) -> Unit,
    isOptional: Boolean
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (isOptional) {
            Text(
                text = "(Optional)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 8.dp)
            )
        }

        OutlinedTextField(
            value = text,
            onValueChange = onValueChange,
            modifier = Modifier
                .fillMaxWidth()
                .height(150.dp),
            placeholder = { Text("Add any notes here...") },
            maxLines = 5
        )
    }
}

@Composable
fun CompletionScreen(
    score: Double,
    interpretation: BASDAIInterpretation?,
    onDone: () -> Unit
) {
    val scoreColor = when (interpretation) {
        BASDAIInterpretation.REMISSION -> InflamAIColors.ScoreRemission
        BASDAIInterpretation.LOW_ACTIVITY -> InflamAIColors.ScoreLowActivity
        BASDAIInterpretation.MODERATE_ACTIVITY -> InflamAIColors.ScoreModerateActivity
        BASDAIInterpretation.HIGH_ACTIVITY -> InflamAIColors.ScoreHighActivity
        BASDAIInterpretation.VERY_HIGH_ACTIVITY -> InflamAIColors.ScoreVeryHighActivity
        null -> MaterialTheme.colorScheme.primary
    }

    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp)
        ) {
            Icon(
                imageVector = Icons.Default.CheckCircle,
                contentDescription = null,
                tint = InflamAIColors.ScoreRemission,
                modifier = Modifier.size(80.dp)
            )

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "Check-in Complete!",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(32.dp))

            // Score display
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = scoreColor.copy(alpha = 0.1f)
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Your BASDAI Score",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = String.format("%.1f", score),
                        style = MaterialTheme.typography.displayLarge,
                        fontWeight = FontWeight.Bold,
                        color = scoreColor
                    )

                    interpretation?.let {
                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = it.displayName,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = scoreColor
                        )

                        Spacer(modifier = Modifier.height(4.dp))

                        Text(
                            text = it.description,
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = TextAlign.Center,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            Button(
                onClick = onDone,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Done")
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Medical disclaimer
            Text(
                text = "Remember: This score is for self-monitoring only. Always discuss concerns with your rheumatologist.",
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
