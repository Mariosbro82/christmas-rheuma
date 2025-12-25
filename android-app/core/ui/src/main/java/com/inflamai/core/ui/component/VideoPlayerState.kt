package com.inflamai.core.ui.component

/**
 * Represents the various states of video playback.
 * Used to manage UI updates and handle lifecycle events properly.
 */
sealed class VideoPlayerState {
    /**
     * Initial state before video is loaded
     */
    object Idle : VideoPlayerState()

    /**
     * Video is currently loading/buffering
     */
    object Loading : VideoPlayerState()

    /**
     * Video is ready to play
     * @param duration Total duration of the video in milliseconds
     */
    data class Ready(val duration: Long) : VideoPlayerState()

    /**
     * Video is currently playing
     * @param position Current playback position in milliseconds
     * @param duration Total duration of the video in milliseconds
     */
    data class Playing(val position: Long, val duration: Long) : VideoPlayerState()

    /**
     * Video is paused
     * @param position Current playback position in milliseconds
     * @param duration Total duration of the video in milliseconds
     */
    data class Paused(val position: Long, val duration: Long) : VideoPlayerState()

    /**
     * Video encountered an error
     * @param message Error message to display
     */
    data class Error(val message: String) : VideoPlayerState()

    /**
     * Video playback completed
     */
    object Completed : VideoPlayerState()
}

/**
 * Helper extensions for VideoPlayerState
 */
val VideoPlayerState.isPlaying: Boolean
    get() = this is VideoPlayerState.Playing

val VideoPlayerState.isPaused: Boolean
    get() = this is VideoPlayerState.Paused

val VideoPlayerState.isLoading: Boolean
    get() = this is VideoPlayerState.Loading

val VideoPlayerState.isError: Boolean
    get() = this is VideoPlayerState.Error

val VideoPlayerState.isCompleted: Boolean
    get() = this is VideoPlayerState.Completed

val VideoPlayerState.currentPosition: Long
    get() = when (this) {
        is VideoPlayerState.Playing -> position
        is VideoPlayerState.Paused -> position
        else -> 0L
    }

val VideoPlayerState.duration: Long
    get() = when (this) {
        is VideoPlayerState.Ready -> duration
        is VideoPlayerState.Playing -> duration
        is VideoPlayerState.Paused -> duration
        else -> 0L
    }
