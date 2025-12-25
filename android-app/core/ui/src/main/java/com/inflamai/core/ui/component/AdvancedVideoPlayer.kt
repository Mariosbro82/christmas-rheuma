package com.inflamai.core.ui.component

import android.net.Uri
import android.os.Build
import android.util.Rational
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.annotation.OptIn
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.database.StandaloneDatabaseProvider
import androidx.media3.datasource.DefaultDataSource
import androidx.media3.datasource.cache.CacheDataSource
import androidx.media3.datasource.cache.LeastRecentlyUsedCacheEvictor
import androidx.media3.datasource.cache.SimpleCache
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.ui.PlayerView
import kotlinx.coroutines.delay
import java.io.File
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Error
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Advanced video player with caching, PiP support, and lifecycle management.
 * Combines all features needed for exercise tutorial playback.
 *
 * Features:
 * - Automatic video caching for offline playback
 * - Picture-in-Picture support (Android 8+)
 * - Custom controls with auto-hide
 * - Thumbnail preview before playback
 * - Loading and error states
 * - Lifecycle-aware (pauses on background, releases on destroy)
 * - Configuration change handling
 *
 * @param videoUrl URL or URI of the video to play
 * @param thumbnailUrl Optional thumbnail URL
 * @param modifier Modifier for the player
 * @param autoPlay Auto-start playback
 * @param enablePiP Enable Picture-in-Picture
 * @param enableCaching Enable video caching
 * @param onVideoComplete Callback when playback completes
 */
@OptIn(UnstableApi::class)
@Composable
fun AdvancedVideoPlayer(
    videoUrl: String,
    thumbnailUrl: String? = null,
    modifier: Modifier = Modifier,
    autoPlay: Boolean = false,
    enablePiP: Boolean = true,
    enableCaching: Boolean = true,
    onVideoComplete: (() -> Unit)? = null
) {
    val context = LocalContext.current
    val activity = context.findActivity()

    var playerState by remember { mutableStateOf<VideoPlayerState>(VideoPlayerState.Idle) }
    var showThumbnail by remember { mutableStateOf(true) }
    var isInPiPMode by remember { mutableStateOf(false) }
    var isFullscreen by remember { mutableStateOf(false) }

    // Video cache setup
    val cache = remember {
        if (enableCaching) {
            val cacheDir = File(context.cacheDir, "video_cache")
            val cacheEvictor = LeastRecentlyUsedCacheEvictor(300 * 1024 * 1024) // 300 MB
            val databaseProvider = StandaloneDatabaseProvider(context)
            SimpleCache(cacheDir, cacheEvictor, databaseProvider)
        } else {
            null
        }
    }

    // ExoPlayer with caching support
    val exoPlayer = remember {
        ExoPlayer.Builder(context).apply {
            if (cache != null) {
                val dataSourceFactory = DefaultDataSource.Factory(context)
                val cacheDataSourceFactory = CacheDataSource.Factory()
                    .setCache(cache)
                    .setUpstreamDataSourceFactory(dataSourceFactory)
                    .setFlags(CacheDataSource.FLAG_IGNORE_CACHE_ON_ERROR)

                setMediaSourceFactory(
                    DefaultMediaSourceFactory(context)
                        .setDataSourceFactory(cacheDataSourceFactory)
                )
            }
        }.build().apply {
            val mediaItem = MediaItem.fromUri(videoUrl)
            setMediaItem(mediaItem)
            prepare()
            playWhenReady = autoPlay

            // Player event listener
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

    // Update playback position
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
                Lifecycle.Event.ON_PAUSE -> {
                    if (!isInPiPMode) {
                        exoPlayer.pause()
                    }
                }
                Lifecycle.Event.ON_STOP -> exoPlayer.pause()
                Lifecycle.Event.ON_DESTROY -> {
                    exoPlayer.release()
                    cache?.release()
                }
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            exoPlayer.release()
            cache?.release()
        }
    }

    // Picture-in-Picture mode listener
    if (enablePiP && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        PictureInPictureModeEffect { inPiPMode ->
            isInPiPMode = inPiPMode
        }
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .aspectRatio(16f / 9f)
            .background(Color.Black)
            .semantics { contentDescription = "Exercise tutorial video player" }
    ) {
        // ExoPlayer view
        AndroidView(
            factory = { ctx ->
                PlayerView(ctx).apply {
                    player = exoPlayer
                    useController = false // Use custom controls
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // Thumbnail overlay
        if (showThumbnail && thumbnailUrl != null && !isInPiPMode) {
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

        // Custom controls (hidden in PiP mode)
        if (!isInPiPMode && !showThumbnail) {
            VideoPlayerControls(
                isPlaying = playerState.isPlaying,
                currentPosition = playerState.currentPosition,
                duration = playerState.duration,
                onPlayPause = {
                    if (exoPlayer.isPlaying) {
                        exoPlayer.pause()
                    } else {
                        exoPlayer.play()
                    }
                },
                onSeek = { position ->
                    exoPlayer.seekTo(position)
                },
                onFullscreen = {
                    isFullscreen = !isFullscreen
                    // TODO: Implement fullscreen logic
                },
                onPictureInPicture = if (enablePiP &&
                    Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
                    activity != null &&
                    PictureInPictureHelper.isPictureInPictureSupported(context)
                ) {
                    {
                        PictureInPictureHelper.enterPictureInPictureMode(
                            activity,
                            Rational(16, 9)
                        )
                    }
                } else null,
                modifier = Modifier.fillMaxSize()
            )
        }

        // Error overlay
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
 * Format duration from milliseconds to readable string
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
