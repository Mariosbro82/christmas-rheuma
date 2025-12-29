package com.inflamai.feature.meditation

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.inflamai.core.ui.component.*
import com.inflamai.core.ui.theme.*

/**
 * Meditation Hub Screen
 * Based on Frame Analysis: Categories, session cards, breathing player, streak
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MeditationScreen(
    viewModel: MeditationViewModel = hiltViewModel(),
    onNavigateBack: () -> Unit = {}
) {
    val uiState by viewModel.uiState.collectAsState()

    // If playing session, show player
    if (uiState.isPlaying) {
        MeditationPlayerScreen(
            session = uiState.currentSession!!,
            elapsedSeconds = uiState.elapsedSeconds,
            breathingPhase = uiState.breathingPhase,
            onPause = { viewModel.pauseSession() },
            onResume = { viewModel.resumeSession() },
            onStop = { viewModel.stopSession() },
            isPaused = uiState.isPaused
        )
        return
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Meditation & Relaxation") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Streak Card
            item {
                MeditationStreakCard(
                    currentStreak = uiState.currentStreak,
                    longestStreak = uiState.longestStreak,
                    totalMinutes = uiState.totalMinutes
                )
            }

            // Quick Start - Breathing Exercise
            item {
                QuickBreathingCard(
                    onStart = { viewModel.startBreathingExercise() }
                )
            }

            // Categories
            item {
                Text(
                    text = "Categories",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }

            item {
                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(MeditationCategory.entries) { category ->
                        CategoryChip(
                            category = category,
                            isSelected = uiState.selectedCategory == category,
                            onClick = { viewModel.selectCategory(category) }
                        )
                    }
                }
            }

            // Featured Sessions
            item {
                Text(
                    text = "Sessions",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary,
                    modifier = Modifier.padding(top = 16.dp, bottom = 8.dp)
                )
            }

            items(uiState.filteredSessions) { session ->
                MeditationSessionCard(
                    session = session,
                    onStart = { viewModel.startSession(session) }
                )
            }

            // Info Card
            item {
                InfoCard(
                    title = "Benefits for AS",
                    content = "Regular meditation can help reduce pain perception, lower stress hormones, and improve sleep quality - all important factors in managing ankylosing spondylitis.",
                    icon = Icons.Default.Lightbulb,
                    backgroundColor = InfoLight
                )
            }
        }
    }
}

/**
 * Streak card showing meditation progress
 */
@Composable
fun MeditationStreakCard(
    currentStreak: Int,
    longestStreak: Int,
    totalMinutes: Int
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MeditationCalmLight)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            StreakStat(
                value = currentStreak.toString(),
                label = "Day Streak",
                icon = "🔥"
            )
            StreakStat(
                value = longestStreak.toString(),
                label = "Best Streak",
                icon = "⭐"
            )
            StreakStat(
                value = totalMinutes.toString(),
                label = "Total Min",
                icon = "⏱"
            )
        }
    }
}

@Composable
private fun StreakStat(
    value: String,
    label: String,
    icon: String
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(text = icon, fontSize = 24.sp)
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = value,
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary
        )
        Text(
            text = label,
            fontSize = 12.sp,
            color = TextSecondary
        )
    }
}

/**
 * Quick breathing exercise card
 */
@Composable
fun QuickBreathingCard(
    onStart: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onStart),
        colors = CardDefaults.cardColors(containerColor = MeditationBreathInLight)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(56.dp)
                    .clip(CircleShape)
                    .background(MeditationBreathIn),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Air,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(28.dp)
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Quick Breathing Exercise",
                    fontSize = 17.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
                Text(
                    text = "4-7-8 breathing technique • 3 minutes",
                    fontSize = 14.sp,
                    color = TextSecondary
                )
            }

            Icon(
                imageVector = Icons.Default.PlayArrow,
                contentDescription = "Start",
                tint = MeditationBreathIn,
                modifier = Modifier.size(32.dp)
            )
        }
    }
}

/**
 * Category filter chip
 */
@Composable
fun CategoryChip(
    category: MeditationCategory,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    FilterChip(
        selected = isSelected,
        onClick = onClick,
        label = { Text(category.displayName) },
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = category.color,
            selectedLabelColor = Color.White
        )
    )
}

/**
 * Session card
 */
@Composable
fun MeditationSessionCard(
    session: MeditationSession,
    onStart: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onStart),
        colors = CardDefaults.cardColors(containerColor = Surface)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(session.category.color.copy(alpha = 0.15f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = session.icon,
                    contentDescription = null,
                    tint = session.category.color,
                    modifier = Modifier.size(24.dp)
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = session.title,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                    if (session.isPremium) {
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = "⭐", fontSize = 14.sp)
                    }
                }
                Text(
                    text = "${session.durationMinutes} minutes • ${session.category.displayName}",
                    fontSize = 13.sp,
                    color = TextSecondary
                )
                Text(
                    text = session.description,
                    fontSize = 13.sp,
                    color = TextTertiary,
                    maxLines = 1
                )
            }

            IconButton(onClick = onStart) {
                Icon(
                    imageVector = Icons.Default.PlayCircle,
                    contentDescription = "Play",
                    tint = session.category.color,
                    modifier = Modifier.size(36.dp)
                )
            }
        }
    }
}

/**
 * Meditation Player Screen
 * Based on Frame Analysis: Circular timer, breathing animation, controls
 */
@Composable
fun MeditationPlayerScreen(
    session: MeditationSession,
    elapsedSeconds: Int,
    breathingPhase: String,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit,
    isPaused: Boolean
) {
    val totalSeconds = session.durationMinutes * 60
    val progress = elapsedSeconds.toFloat() / totalSeconds
    val remainingSeconds = totalSeconds - elapsedSeconds
    val timeText = String.format("%d:%02d", remainingSeconds / 60, remainingSeconds % 60)

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
            // Session title
            Text(
                text = session.title,
                fontSize = 20.sp,
                fontWeight = FontWeight.SemiBold,
                color = TextPrimary
            )

            Spacer(modifier = Modifier.height(48.dp))

            // Breathing circle animation
            if (session.category == MeditationCategory.BREATHING) {
                BreathingCircle(
                    phase = breathingPhase,
                    size = 220.dp
                )
            } else {
                // Timer circle
                CircularProgressTimer(
                    progress = progress,
                    timeText = timeText,
                    size = 220.dp,
                    progressColor = session.category.color,
                    subtitle = if (isPaused) "Paused" else "Remaining"
                )
            }

            Spacer(modifier = Modifier.height(48.dp))

            // Breathing phase instruction
            if (session.category == MeditationCategory.BREATHING) {
                Text(
                    text = when (breathingPhase) {
                        "inhale" -> "Breathe In..."
                        "hold" -> "Hold..."
                        "exhale" -> "Breathe Out..."
                        else -> ""
                    },
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Medium,
                    color = TextPrimary
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = when (breathingPhase) {
                        "inhale" -> "4 seconds"
                        "hold" -> "7 seconds"
                        "exhale" -> "8 seconds"
                        else -> ""
                    },
                    fontSize = 16.sp,
                    color = TextSecondary
                )
            }

            Spacer(modifier = Modifier.height(48.dp))

            // Controls
            Row(
                horizontalArrangement = Arrangement.spacedBy(24.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Stop button
                IconButton(
                    onClick = onStop,
                    modifier = Modifier
                        .size(56.dp)
                        .clip(CircleShape)
                        .background(ErrorLight)
                ) {
                    Icon(
                        imageVector = Icons.Default.Stop,
                        contentDescription = "Stop",
                        tint = Error,
                        modifier = Modifier.size(28.dp)
                    )
                }

                // Play/Pause button
                IconButton(
                    onClick = if (isPaused) onResume else onPause,
                    modifier = Modifier
                        .size(72.dp)
                        .clip(CircleShape)
                        .background(session.category.color)
                ) {
                    Icon(
                        imageVector = if (isPaused) Icons.Default.PlayArrow else Icons.Default.Pause,
                        contentDescription = if (isPaused) "Resume" else "Pause",
                        tint = Color.White,
                        modifier = Modifier.size(36.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Time elapsed
            Text(
                text = "Elapsed: ${elapsedSeconds / 60}:${String.format("%02d", elapsedSeconds % 60)}",
                fontSize = 14.sp,
                color = TextTertiary
            )
        }
    }
}
