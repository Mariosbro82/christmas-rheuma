package com.inflamai.core.ui.component

import android.net.Uri
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.annotation.OptIn
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import coil.compose.AsyncImagePainter
import coil.compose.rememberAsyncImagePainter
import kotlinx.coroutines.delay

/**
 * Main video player component for exercise tutorials.
 * Supports online/offline playback, thumbnail preview, and full playback controls.
 *
 * @param videoUrl URL or URI of the video to play
 * @param thumbnailUrl Optional thumbnail URL to display before playback
 * @param modifier Modifier for the player container
 * @param autoPlay Whether to start playback automatically
 * @param showControls Whether to show playback controls
 * @param onVideoComplete Callback when video playback completes
 */
@OptIn(UnstableApi::class)
@Composable
fun ExerciseVideoPlayer(
    videoUrl: String,
    thumbnailUrl: String? = null,
    modifier: Modifier = Modifier,
    autoPlay: Boolean = false,
    showControls: Boolean = true,
    onVideoComplete: (() -> Unit)? = null
) {
    val context = LocalContext.current
    var playerState by remember { mutableStateOf<VideoPlayerState>(VideoPlayerState.Idle) }
    var showThumbnail by remember { mutableStateOf(true) }

    // Remember ExoPlayer instance
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            val mediaItem = MediaItem.fromUri(videoUrl)
            setMediaItem(mediaItem)
            prepare()
            playWhenReady = autoPlay

            // Add listener for state changes
            addListener(object : Player.Listener {
                override fun onPlaybackStateChanged(playbackState: Int) {
                    when (playbackState) {
                        Player.STATE_IDLE -> {
                            playerState = VideoPlayerState.Idle
                        }
                        Player.STATE_BUFFERING -> {
                            playerState = VideoPlayerState.Loading
                        }
                        Player.STATE_READY -> {
                            if (isPlaying) {
                                playerState = VideoPlayerState.Playing(
                                    position = currentPosition,
                                    duration = duration
                                )
                            } else {
                                playerState = VideoPlayerState.Ready(duration)
                            }
                        }
                        Player.STATE_ENDED -> {
                            playerState = VideoPlayerState.Completed
                            onVideoComplete?.invoke()
                        }
                    }
                }

                override fun onIsPlayingChanged(isPlaying: Boolean) {
                    if (isPlaying) {
                        showThumbnail = false
                        playerState = VideoPlayerState.Playing(
                            position = currentPosition,
                            duration = duration
                        )
                    } else if (playbackState == Player.STATE_READY) {
                        playerState = VideoPlayerState.Paused(
                            position = currentPosition,
                            duration = duration
                        )
                    }
                }

                override fun onPlayerError(error: PlaybackException) {
                    playerState = VideoPlayerState.Error(
                        error.message ?: "Video playback failed"
                    )
                }
            })
        }
    }

    // Update playback position periodically when playing
    LaunchedEffect(playerState) {
        if (playerState.isPlaying) {
            while (true) {
                delay(100)
                playerState = VideoPlayerState.Playing(
                    position = exoPlayer.currentPosition,
                    duration = exoPlayer.duration
                )
            }
        }
    }

    // Lifecycle management
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_PAUSE -> exoPlayer.pause()
                Lifecycle.Event.ON_STOP -> exoPlayer.pause()
                Lifecycle.Event.ON_DESTROY -> exoPlayer.release()
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            exoPlayer.release()
        }
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .aspectRatio(16f / 9f)
            .background(Color.Black)
            .semantics { contentDescription = "Exercise tutorial video player" }
    ) {
        // Video player view
        AndroidView(
            factory = { ctx ->
                PlayerView(ctx).apply {
                    player = exoPlayer
                    useController = showControls
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // Thumbnail overlay (shown before playback starts)
        if (showThumbnail && thumbnailUrl != null) {
            VideoThumbnail(
                thumbnailUrl = thumbnailUrl,
                duration = formatDuration(playerState.duration),
                onClick = {
                    showThumbnail = false
                    exoPlayer.play()
                },
                modifier = Modifier.fillMaxSize()
            )
        }

        // Loading indicator
        if (playerState.isLoading) {
            CircularProgressIndicator(
                modifier = Modifier
                    .align(Alignment.Center)
                    .size(48.dp),
                color = MaterialTheme.colorScheme.primary
            )
        }

        // Error state
        if (playerState.isError) {
            ErrorOverlay(
                message = (playerState as VideoPlayerState.Error).message,
                onRetry = {
                    exoPlayer.prepare()
                    exoPlayer.play()
                },
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

/**
 * Thumbnail preview with play button overlay.
 * Matches iOS design with white play triangle in translucent circle.
 *
 * @param thumbnailUrl URL of the thumbnail image
 * @param duration Optional video duration string (e.g., "5:30")
 * @param onClick Callback when thumbnail/play button is clicked
 * @param modifier Modifier for the thumbnail container
 */
@Composable
fun VideoThumbnail(
    thumbnailUrl: String?,
    duration: String?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clickable(onClick = onClick)
            .semantics { contentDescription = "Play video" }
    ) {
        // Thumbnail image
        if (thumbnailUrl != null) {
            val painter = rememberAsyncImagePainter(model = thumbnailUrl)

            Image(
                painter = painter,
                contentDescription = "Video thumbnail",
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize()
            )

            // Loading indicator while image loads
            if (painter.state is AsyncImagePainter.State.Loading) {
                CircularProgressIndicator(
                    modifier = Modifier
                        .align(Alignment.Center)
                        .size(40.dp),
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        } else {
            // Fallback color when no thumbnail
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color(0xFF1A1A1A))
            )
        }

        // Play button overlay (white triangle in translucent circle)
        Surface(
            modifier = Modifier
                .align(Alignment.Center)
                .size(72.dp),
            shape = CircleShape,
            color = Color.White.copy(alpha = 0.3f)
        ) {
            Icon(
                imageVector = Icons.Default.PlayArrow,
                contentDescription = "Play",
                tint = Color.White,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            )
        }

        // Duration label (bottom right corner)
        if (duration != null) {
            Surface(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(8.dp),
                shape = RoundedCornerShape(4.dp),
                color = Color.Black.copy(alpha = 0.7f)
            ) {
                Text(
                    text = duration,
                    color = Color.White,
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                )
            }
        }
    }
}

/**
 * YouTube video player component.
 * Uses embedded YouTube player via WebView or YouTube Android Player API.
 *
 * @param videoId YouTube video ID (not full URL)
 * @param modifier Modifier for the player container
 */
@Composable
fun YouTubeVideoPlayer(
    videoId: String,
    modifier: Modifier = Modifier
) {
    // Note: This is a placeholder. Full implementation would use
    // YouTube Android Player API or WebView with embedded player
    Box(
        modifier = modifier
            .fillMaxWidth()
            .aspectRatio(16f / 9f)
            .background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = "YouTube Player\n(Implementation required)",
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

/**
 * Offline video player for locally cached videos.
 *
 * @param localUri URI pointing to local video file
 * @param modifier Modifier for the player container
 */
@OptIn(UnstableApi::class)
@Composable
fun OfflineVideoPlayer(
    localUri: Uri,
    modifier: Modifier = Modifier
) {
    // Reuse ExerciseVideoPlayer with local URI
    ExerciseVideoPlayer(
        videoUrl = localUri.toString(),
        thumbnailUrl = null,
        modifier = modifier,
        autoPlay = false,
        showControls = true
    )
}

/**
 * Error overlay displayed when video fails to load.
 */
@Composable
private fun ErrorOverlay(
    message: String,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.background(Color.Black.copy(alpha = 0.8f)),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Error,
                contentDescription = "Error",
                tint = MaterialTheme.colorScheme.error,
                modifier = Modifier.size(48.dp)
            )

            Text(
                text = "Video Unavailable",
                style = MaterialTheme.typography.titleMedium,
                color = Color.White,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.7f)
            )

            Button(onClick = onRetry) {
                Text("Retry")
            }
        }
    }
}

/**
 * Formats duration from milliseconds to MM:SS or HH:MM:SS format.
 */
private fun formatDuration(durationMs: Long): String? {
    if (durationMs <= 0) return null

    val totalSeconds = durationMs / 1000
    val hours = totalSeconds / 3600
    val minutes = (totalSeconds % 3600) / 60
    val seconds = totalSeconds % 60

    return if (hours > 0) {
        String.format("%d:%02d:%02d", hours, minutes, seconds)
    } else {
        String.format("%d:%02d", minutes, seconds)
    }
}
