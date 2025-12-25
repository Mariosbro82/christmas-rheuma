# Video Player Component - InflamAI Android

Complete video player implementation using Media3 (ExoPlayer) for exercise tutorial playback in the InflamAI Android app.

## Overview

This implementation provides a production-ready video player with the following features:
- **Thumbnail Preview**: iOS-style thumbnail with play button overlay
- **Loading States**: Spinner during buffering
- **Error Handling**: Retry mechanism with user-friendly error messages
- **Playback Controls**: Play/pause, seek bar, fullscreen, Picture-in-Picture
- **Lifecycle Management**: Automatic pause on background, release on destroy
- **Video Caching**: Offline playback support (300MB cache)
- **Picture-in-Picture**: Android 8+ support for background playback
- **Accessibility**: Full TalkBack support with content descriptions

## Files Created

### Core Components

1. **VideoPlayerState.kt**
   - Sealed class hierarchy for player states
   - Extension properties for state checks
   - Position and duration helpers

2. **VideoPlayer.kt**
   - `ExerciseVideoPlayer`: Main video player component
   - `VideoThumbnail`: Thumbnail with play button overlay
   - `YouTubeVideoPlayer`: Placeholder for YouTube integration
   - `OfflineVideoPlayer`: Local video playback

3. **VideoPlayerControls.kt**
   - Custom playback controls with auto-hide
   - Seek bar with progress tracking
   - Play/pause, fullscreen, PiP buttons
   - Time display (current/duration)

4. **VideoPlayerViewModel.kt**
   - State management for configuration changes
   - Playback position restoration
   - ViewModel-based state handling

5. **PictureInPictureHelper.kt**
   - PiP mode detection and management
   - Activity finding utilities
   - Lifecycle-aware PiP listeners

6. **AdvancedVideoPlayer.kt**
   - Complete player with all features combined
   - Video caching with 300MB LRU cache
   - PiP integration
   - Custom controls overlay

7. **VideoPlayerExamples.kt**
   - Usage examples and patterns
   - Exercise list integration
   - Mini player example

## Dependencies Added

### libs.versions.toml

```toml
[versions]
media3 = "1.2.1"

[libraries]
media3-exoplayer = { group = "androidx.media3", name = "media3-exoplayer", version.ref = "media3" }
media3-ui = { group = "androidx.media3", name = "media3-ui", version.ref = "media3" }
media3-common = { group = "androidx.media3", name = "media3-common", version.ref = "media3" }
```

### core/ui/build.gradle.kts

```kotlin
dependencies {
    // Media3 (ExoPlayer replacement)
    api(libs.media3.exoplayer)
    api(libs.media3.ui)
    api(libs.media3.common)

    // Image Loading (for thumbnails)
    api(libs.coil.compose)
}
```

## Usage Examples

### Basic Video Player

```kotlin
ExerciseVideoPlayer(
    videoUrl = "https://example.com/videos/exercise1.mp4",
    thumbnailUrl = "https://example.com/thumbnails/exercise1.jpg",
    autoPlay = false,
    showControls = true,
    onVideoComplete = {
        // Mark exercise as completed
    }
)
```

### Advanced Player with Caching & PiP

```kotlin
AdvancedVideoPlayer(
    videoUrl = "https://example.com/videos/exercise2.mp4",
    thumbnailUrl = "https://example.com/thumbnails/exercise2.jpg",
    autoPlay = false,
    enablePiP = true,          // Picture-in-Picture support
    enableCaching = true,      // Cache for offline viewing
    onVideoComplete = {
        // Log completion
    }
)
```

### Offline Video Playback

```kotlin
val localUri = Uri.parse("file:///path/to/video.mp4")

OfflineVideoPlayer(
    localUri = localUri,
    modifier = Modifier.fillMaxWidth()
)
```

### Video Thumbnail (Before Playback)

```kotlin
VideoThumbnail(
    thumbnailUrl = "https://example.com/thumbnail.jpg",
    duration = "5:30",
    onClick = { /* Start playback */ },
    modifier = Modifier
        .fillMaxWidth()
        .height(200.dp)
)
```

## Component API Reference

### ExerciseVideoPlayer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `videoUrl` | String | Required | URL or URI of video |
| `thumbnailUrl` | String? | null | Thumbnail image URL |
| `modifier` | Modifier | Modifier | Layout modifier |
| `autoPlay` | Boolean | false | Start playback automatically |
| `showControls` | Boolean | true | Show playback controls |
| `onVideoComplete` | (() -> Unit)? | null | Completion callback |

### AdvancedVideoPlayer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `videoUrl` | String | Required | URL or URI of video |
| `thumbnailUrl` | String? | null | Thumbnail image URL |
| `modifier` | Modifier | Modifier | Layout modifier |
| `autoPlay` | Boolean | false | Start playback automatically |
| `enablePiP` | Boolean | true | Enable Picture-in-Picture |
| `enableCaching` | Boolean | true | Cache videos for offline |
| `onVideoComplete` | (() -> Unit)? | null | Completion callback |

### VideoPlayerState

```kotlin
sealed class VideoPlayerState {
    object Idle                                        // Before loading
    object Loading                                     // Buffering
    data class Ready(val duration: Long)              // Ready to play
    data class Playing(val position: Long, ...)       // Currently playing
    data class Paused(val position: Long, ...)        // Paused
    data class Error(val message: String)             // Error occurred
    object Completed                                   // Playback finished
}
```

## Features Deep Dive

### 1. Thumbnail Preview

Matches iOS design with:
- Thumbnail image loaded via Coil
- White play button in translucent circle (72dp)
- Duration label in bottom-right corner
- Tap anywhere to start playback

### 2. Playback Controls

Auto-hiding controls (3 seconds of inactivity) with:
- Center play/pause button (large, translucent)
- Bottom control bar with gradient overlay
- Seek slider with accurate position tracking
- Time display (current/total)
- Fullscreen button
- Picture-in-Picture button (Android 8+)

### 3. Video Caching

Automatic caching for offline playback:
- 300MB LRU cache in app cache directory
- Uses Media3's `CacheDataSource`
- Transparent to the user
- Survives app restarts

### 4. Picture-in-Picture (PiP)

Android 8+ support:
- 16:9 aspect ratio maintained
- Controls hidden in PiP mode
- Playback continues in background
- Automatic pause when PiP closes

### 5. Lifecycle Management

Proper Android lifecycle handling:
- **ON_PAUSE**: Pause playback (unless in PiP)
- **ON_STOP**: Pause playback
- **ON_DESTROY**: Release player and cache
- Configuration changes handled by ViewModel

### 6. Error Handling

User-friendly error states:
- Error icon with descriptive message
- "Retry" button to restart playback
- No crashes on network failures
- Fallback for missing thumbnails

### 7. Accessibility

WCAG AA compliance:
- Content descriptions for all controls
- TalkBack-friendly labels
- 48dp minimum touch targets
- Semantic properties for screen readers

## Integration with Exercise Feature

### Exercise Entity

```kotlin
data class Exercise(
    val id: String,
    val title: String,
    val videoUrl: String,
    val thumbnailUrl: String,
    val duration: String,
    val difficulty: String,
    val instructions: String
)
```

### Exercise Detail Screen

```kotlin
@Composable
fun ExerciseDetailScreen(exercise: Exercise) {
    Column(modifier = Modifier.fillMaxSize()) {
        // Video player
        AdvancedVideoPlayer(
            videoUrl = exercise.videoUrl,
            thumbnailUrl = exercise.thumbnailUrl,
            enablePiP = true,
            enableCaching = true,
            onVideoComplete = {
                // Log completion to Room database
            }
        )

        // Exercise details
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = exercise.title,
                style = MaterialTheme.typography.headlineSmall
            )

            Text(
                text = exercise.instructions,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}
```

## Performance Considerations

### Memory Management

- ExoPlayer instances are properly released
- Cache evicts LRU content when limit reached
- Lifecycle-aware disposal prevents leaks
- Coil handles image caching automatically

### Network Optimization

- Video caching reduces repeated downloads
- Thumbnail preloading via Coil
- Buffering strategy optimized for mobile
- Error retry with exponential backoff

### Battery Optimization

- Playback pauses when app backgrounded
- PiP uses optimized rendering
- Wake lock released when player released

## Testing

### Unit Tests

```kotlin
class VideoPlayerViewModelTest {
    @Test
    fun `updatePlayerState updates state flow`() = runTest {
        val viewModel = VideoPlayerViewModel()
        val state = VideoPlayerState.Playing(1000L, 5000L)

        viewModel.updatePlayerState(state)

        assertEquals(state, viewModel.playerState.value)
        assertTrue(viewModel.isPlaying.value)
    }
}
```

### UI Tests

```kotlin
@Test
fun videoPlayerDisplaysThumbnail() {
    composeTestRule.setContent {
        ExerciseVideoPlayer(
            videoUrl = "test.mp4",
            thumbnailUrl = "thumbnail.jpg"
        )
    }

    composeTestRule
        .onNodeWithContentDescription("Play video")
        .assertIsDisplayed()
}
```

## Known Limitations

1. **YouTube Integration**: `YouTubeVideoPlayer` is a placeholder. Full implementation requires YouTube Android Player API or WebView.

2. **Fullscreen**: Fullscreen toggle implemented, but full immersive mode requires Activity-level coordination.

3. **Adaptive Streaming**: Currently uses single-bitrate videos. DASH/HLS support available via Media3.

4. **DRM**: No DRM support in current implementation. Add via Media3 DRM extensions if needed.

## Future Enhancements

1. **Adaptive Streaming**: DASH/HLS for variable quality
2. **Playback Speed**: 0.5x, 1x, 1.5x, 2x controls
3. **Subtitles/Captions**: Medical terminology support
4. **Download Management**: Explicit offline download UI
5. **Analytics**: Track completion rate, seek behavior
6. **Quality Selector**: Manual quality switching

## Privacy & Medical Compliance

- **No Analytics**: Zero third-party video analytics
- **On-Device Caching**: Videos cached locally only
- **No Sharing**: No social media integration
- **Medical Disclaimers**: Display warnings for exercises

## Resources

- [Media3 Documentation](https://developer.android.com/media/media3)
- [ExoPlayer Migration Guide](https://developer.android.com/media/media3/exoplayer/migration-guide)
- [Picture-in-Picture Guide](https://developer.android.com/develop/ui/views/picture-in-picture)

---

**Last Updated**: 2025-12-24
**Version**: 1.0
**Dependencies**: Media3 1.2.1, Coil 2.5.0
