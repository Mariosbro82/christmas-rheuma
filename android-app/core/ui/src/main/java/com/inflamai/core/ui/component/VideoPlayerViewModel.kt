package com.inflamai.core.ui.component

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for managing video player state across configuration changes.
 * Preserves playback position, state, and manages lifecycle properly.
 */
class VideoPlayerViewModel : ViewModel() {
    private val _playerState = MutableStateFlow<VideoPlayerState>(VideoPlayerState.Idle)
    val playerState: StateFlow<VideoPlayerState> = _playerState.asStateFlow()

    private val _currentPosition = MutableStateFlow(0L)
    val currentPosition: StateFlow<Long> = _currentPosition.asStateFlow()

    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()

    /**
     * Update the current player state
     */
    fun updatePlayerState(state: VideoPlayerState) {
        viewModelScope.launch {
            _playerState.value = state
            _isPlaying.value = state.isPlaying
            _currentPosition.value = state.currentPosition
        }
    }

    /**
     * Update playback position
     */
    fun updatePosition(position: Long) {
        viewModelScope.launch {
            _currentPosition.value = position
        }
    }

    /**
     * Save current playback position for restoration
     */
    fun savePlaybackPosition(position: Long) {
        _currentPosition.value = position
    }

    /**
     * Restore playback position after configuration change
     */
    fun getRestoredPosition(): Long = _currentPosition.value

    /**
     * Reset player state
     */
    fun reset() {
        _playerState.value = VideoPlayerState.Idle
        _currentPosition.value = 0L
        _isPlaying.value = false
    }
}
