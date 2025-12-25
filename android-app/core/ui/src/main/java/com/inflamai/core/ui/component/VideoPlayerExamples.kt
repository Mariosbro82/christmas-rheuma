package com.inflamai.core.ui.component

import android.net.Uri
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Example usage of video player components.
 * These examples demonstrate various use cases for the InflamAI exercise library.
 */

/**
 * Example 1: Basic video player with thumbnail
 */
@Composable
fun BasicVideoPlayerExample() {
    ExerciseVideoPlayer(
        videoUrl = "https://example.com/videos/exercise1.mp4",
        thumbnailUrl = "https://example.com/thumbnails/exercise1.jpg",
        autoPlay = false,
        showControls = true,
        onVideoComplete = {
            // Handle completion (e.g., mark exercise as viewed)
            println("Video completed!")
        }
    )
}

/**
 * Example 2: Advanced video player with caching and PiP
 */
@Composable
fun AdvancedVideoPlayerExample() {
    AdvancedVideoPlayer(
        videoUrl = "https://example.com/videos/exercise2.mp4",
        thumbnailUrl = "https://example.com/thumbnails/exercise2.jpg",
        autoPlay = false,
        enablePiP = true,  // Enable Picture-in-Picture
        enableCaching = true,  // Cache for offline viewing
        onVideoComplete = {
            // Log exercise completion
            println("Exercise tutorial completed!")
        }
    )
}

/**
 * Example 3: Offline video player with local file
 */
@Composable
fun OfflineVideoPlayerExample() {
    val localVideoUri = Uri.parse("file:///storage/emulated/0/InflamAI/videos/exercise3.mp4")

    OfflineVideoPlayer(
        localUri = localVideoUri,
        modifier = Modifier.fillMaxWidth()
    )
}

/**
 * Example 4: Exercise list with video thumbnails
 */
data class Exercise(
    val id: String,
    val title: String,
    val videoUrl: String,
    val thumbnailUrl: String,
    val duration: String,
    val difficulty: String
)

@Composable
fun ExerciseListWithVideos() {
    val exercises = remember {
        listOf(
            Exercise(
                id = "1",
                title = "Spinal Extension Exercise",
                videoUrl = "https://example.com/videos/spinal_extension.mp4",
                thumbnailUrl = "https://example.com/thumbnails/spinal_extension.jpg",
                duration = "5:30",
                difficulty = "Beginner"
            ),
            Exercise(
                id = "2",
                title = "Hip Flexor Stretch",
                videoUrl = "https://example.com/videos/hip_flexor.mp4",
                thumbnailUrl = "https://example.com/thumbnails/hip_flexor.jpg",
                duration = "3:45",
                difficulty = "Intermediate"
            ),
            Exercise(
                id = "3",
                title = "Thoracic Mobility",
                videoUrl = "https://example.com/videos/thoracic_mobility.mp4",
                thumbnailUrl = "https://example.com/thumbnails/thoracic_mobility.jpg",
                duration = "7:15",
                difficulty = "Advanced"
            )
        )
    }

    var selectedExercise by remember { mutableStateOf<Exercise?>(null) }

    if (selectedExercise != null) {
        // Full screen video player
        Column(modifier = Modifier.fillMaxSize()) {
            AdvancedVideoPlayer(
                videoUrl = selectedExercise!!.videoUrl,
                thumbnailUrl = selectedExercise!!.thumbnailUrl,
                enablePiP = true,
                enableCaching = true,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(16.dp))

            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = selectedExercise!!.title,
                    style = MaterialTheme.typography.headlineSmall
                )
                Text(
                    text = "Difficulty: ${selectedExercise!!.difficulty}",
                    style = MaterialTheme.typography.bodyMedium
                )

                Spacer(modifier = Modifier.height(8.dp))

                Button(onClick = { selectedExercise = null }) {
                    Text("Back to List")
                }
            }
        }
    } else {
        // Exercise list
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            items(exercises) { exercise ->
                ExerciseCard(
                    exercise = exercise,
                    onClick = { selectedExercise = exercise }
                )
            }
        }
    }
}

@Composable
private fun ExerciseCard(
    exercise: Exercise,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        onClick = onClick,
        modifier = modifier.fillMaxWidth()
    ) {
        Column {
            // Video thumbnail with play button
            VideoThumbnail(
                thumbnailUrl = exercise.thumbnailUrl,
                duration = exercise.duration,
                onClick = onClick,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp)
            )

            // Exercise details
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = exercise.title,
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = exercise.difficulty,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = exercise.duration,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
    }
}

/**
 * Example 5: Video player with custom state management
 */
@Composable
fun VideoPlayerWithStateManagement() {
    var playerState by remember { mutableStateOf<VideoPlayerState>(VideoPlayerState.Idle) }

    Column(modifier = Modifier.fillMaxSize()) {
        ExerciseVideoPlayer(
            videoUrl = "https://example.com/videos/exercise.mp4",
            thumbnailUrl = "https://example.com/thumbnails/exercise.jpg",
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Show player state
        Card(modifier = Modifier.padding(16.dp)) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Player State:",
                    style = MaterialTheme.typography.titleSmall
                )
                Text(
                    text = when (playerState) {
                        is VideoPlayerState.Idle -> "Idle"
                        is VideoPlayerState.Loading -> "Loading..."
                        is VideoPlayerState.Ready -> "Ready to play"
                        is VideoPlayerState.Playing -> {
                            val state = playerState as VideoPlayerState.Playing
                            "Playing: ${state.position}ms / ${state.duration}ms"
                        }
                        is VideoPlayerState.Paused -> {
                            val state = playerState as VideoPlayerState.Paused
                            "Paused at ${state.position}ms"
                        }
                        is VideoPlayerState.Error -> {
                            val state = playerState as VideoPlayerState.Error
                            "Error: ${state.message}"
                        }
                        is VideoPlayerState.Completed -> "Completed"
                    },
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}

/**
 * Example 6: Mini player for quick preview
 */
@Composable
fun MiniVideoPlayer(
    videoUrl: String,
    thumbnailUrl: String,
    modifier: Modifier = Modifier
) {
    ExerciseVideoPlayer(
        videoUrl = videoUrl,
        thumbnailUrl = thumbnailUrl,
        autoPlay = false,
        showControls = true,
        modifier = modifier
            .fillMaxWidth()
            .height(200.dp)
    )
}
