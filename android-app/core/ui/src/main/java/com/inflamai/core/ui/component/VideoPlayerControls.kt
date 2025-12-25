package com.inflamai.core.ui.component

import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay

/**
 * Custom video player controls with auto-hide functionality.
 * Provides play/pause, seek, fullscreen, and PiP controls.
 *
 * @param isPlaying Whether video is currently playing
 * @param currentPosition Current playback position in milliseconds
 * @param duration Total video duration in milliseconds
 * @param onPlayPause Play/pause toggle callback
 * @param onSeek Seek to position callback
 * @param onFullscreen Fullscreen toggle callback
 * @param onPictureInPicture Picture-in-Picture callback
 * @param modifier Modifier for the controls container
 */
@Composable
fun VideoPlayerControls(
    isPlaying: Boolean,
    currentPosition: Long,
    duration: Long,
    onPlayPause: () -> Unit,
    onSeek: (Long) -> Unit,
    onFullscreen: () -> Unit,
    onPictureInPicture: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    var showControls by remember { mutableStateOf(true) }
    val interactionSource = remember { MutableInteractionSource() }

    // Auto-hide controls after 3 seconds of inactivity when playing
    LaunchedEffect(isPlaying, showControls) {
        if (isPlaying && showControls) {
            delay(3000)
            showControls = false
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .clickable(
                interactionSource = interactionSource,
                indication = null
            ) {
                showControls = !showControls
            }
    ) {
        AnimatedVisibility(
            visible = showControls,
            enter = fadeIn(animationSpec = tween(300)),
            exit = fadeOut(animationSpec = tween(300))
        ) {
            Box(modifier = Modifier.fillMaxSize()) {
                // Top gradient overlay
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(120.dp)
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(
                                    Color.Black.copy(alpha = 0.7f),
                                    Color.Transparent
                                )
                            )
                        )
                        .align(Alignment.TopCenter)
                )

                // Center play/pause button
                Surface(
                    modifier = Modifier
                        .align(Alignment.Center)
                        .size(72.dp)
                        .semantics { contentDescription = if (isPlaying) "Pause" else "Play" },
                    shape = MaterialTheme.shapes.large,
                    color = Color.White.copy(alpha = 0.3f),
                    onClick = onPlayPause
                ) {
                    Icon(
                        imageVector = if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp)
                    )
                }

                // Bottom controls bar
                BottomControlsBar(
                    isPlaying = isPlaying,
                    currentPosition = currentPosition,
                    duration = duration,
                    onPlayPause = onPlayPause,
                    onSeek = onSeek,
                    onFullscreen = onFullscreen,
                    onPictureInPicture = onPictureInPicture,
                    modifier = Modifier.align(Alignment.BottomCenter)
                )
            }
        }
    }
}

/**
 * Bottom controls bar with seek slider and action buttons.
 */
@Composable
private fun BottomControlsBar(
    isPlaying: Boolean,
    currentPosition: Long,
    duration: Long,
    onPlayPause: () -> Unit,
    onSeek: (Long) -> Unit,
    onFullscreen: () -> Unit,
    onPictureInPicture: (() -> Unit)?,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color.Transparent,
                        Color.Black.copy(alpha = 0.7f)
                    )
                )
            )
            .padding(16.dp)
    ) {
        // Seek slider
        VideoSeekBar(
            currentPosition = currentPosition,
            duration = duration,
            onSeek = onSeek,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Control buttons row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Play/Pause button
            IconButton(
                onClick = onPlayPause,
                modifier = Modifier.semantics {
                    contentDescription = if (isPlaying) "Pause video" else "Play video"
                }
            ) {
                Icon(
                    imageVector = if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                    contentDescription = null,
                    tint = Color.White
                )
            }

            // Time display
            Text(
                text = "${formatTime(currentPosition)} / ${formatTime(duration)}",
                color = Color.White,
                style = MaterialTheme.typography.bodySmall
            )

            Row {
                // Picture-in-Picture button (Android 8+)
                if (onPictureInPicture != null) {
                    IconButton(
                        onClick = onPictureInPicture,
                        modifier = Modifier.semantics {
                            contentDescription = "Enter picture-in-picture mode"
                        }
                    ) {
                        Icon(
                            imageVector = Icons.Default.PictureInPicture,
                            contentDescription = null,
                            tint = Color.White
                        )
                    }
                }

                // Fullscreen button
                IconButton(
                    onClick = onFullscreen,
                    modifier = Modifier.semantics {
                        contentDescription = "Toggle fullscreen"
                    }
                ) {
                    Icon(
                        imageVector = Icons.Default.Fullscreen,
                        contentDescription = null,
                        tint = Color.White
                    )
                }
            }
        }
    }
}

/**
 * Seek bar with progress indicator.
 */
@Composable
private fun VideoSeekBar(
    currentPosition: Long,
    duration: Long,
    onSeek: (Long) -> Unit,
    modifier: Modifier = Modifier
) {
    var sliderPosition by remember(currentPosition) {
        mutableFloatStateOf(currentPosition.toFloat())
    }

    Slider(
        value = sliderPosition,
        onValueChange = { sliderPosition = it },
        onValueChangeFinished = { onSeek(sliderPosition.toLong()) },
        valueRange = 0f..duration.coerceAtLeast(1L).toFloat(),
        modifier = modifier.semantics {
            contentDescription = "Video seek bar"
        },
        colors = SliderDefaults.colors(
            thumbColor = MaterialTheme.colorScheme.primary,
            activeTrackColor = MaterialTheme.colorScheme.primary,
            inactiveTrackColor = Color.White.copy(alpha = 0.3f)
        )
    )
}

/**
 * Format milliseconds to MM:SS or HH:MM:SS
 */
private fun formatTime(timeMs: Long): String {
    val totalSeconds = timeMs / 1000
    val hours = totalSeconds / 3600
    val minutes = (totalSeconds % 3600) / 60
    val seconds = totalSeconds % 60

    return if (hours > 0) {
        String.format("%d:%02d:%02d", hours, minutes, seconds)
    } else {
        String.format("%d:%02d", minutes, seconds)
    }
}
