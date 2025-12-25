package com.inflamai.feature.meditation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.repository.MeditationRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Meditation Hub ViewModel
 * Manages meditation sessions, breathing exercises, and streaks
 */
data class MeditationUiState(
    val sessions: List<MeditationSession> = defaultSessions,
    val filteredSessions: List<MeditationSession> = defaultSessions,
    val selectedCategory: MeditationCategory? = null,
    val currentSession: MeditationSession? = null,
    val isPlaying: Boolean = false,
    val isPaused: Boolean = false,
    val elapsedSeconds: Int = 0,
    val breathingPhase: String = "inhale", // "inhale", "hold", "exhale"
    val currentStreak: Int = 0,
    val longestStreak: Int = 0,
    val totalMinutes: Int = 0
)

private val defaultSessions = listOf(
    MeditationSession(
        id = "1",
        title = "Morning Stiffness Relief",
        description = "Gentle guided meditation for morning stiffness",
        durationMinutes = 10,
        category = MeditationCategory.PAIN_RELIEF,
        icon = Icons.Default.WbSunny
    ),
    MeditationSession(
        id = "2",
        title = "Deep Breathing for Pain",
        description = "4-7-8 breathing technique for pain management",
        durationMinutes = 5,
        category = MeditationCategory.BREATHING,
        icon = Icons.Default.Air
    ),
    MeditationSession(
        id = "3",
        title = "Body Scan Relaxation",
        description = "Progressive body scan for tension release",
        durationMinutes = 15,
        category = MeditationCategory.STRESS,
        icon = Icons.Default.Accessibility
    ),
    MeditationSession(
        id = "4",
        title = "Sleep Preparation",
        description = "Wind down routine for better sleep",
        durationMinutes = 20,
        category = MeditationCategory.SLEEP,
        icon = Icons.Default.Bedtime
    ),
    MeditationSession(
        id = "5",
        title = "Quick Stress Relief",
        description = "Fast relaxation technique for flare days",
        durationMinutes = 3,
        category = MeditationCategory.STRESS,
        icon = Icons.Default.Psychology
    ),
    MeditationSession(
        id = "6",
        title = "Focus & Clarity",
        description = "Mindfulness practice for mental clarity",
        durationMinutes = 10,
        category = MeditationCategory.FOCUS,
        icon = Icons.Default.Lightbulb
    ),
    MeditationSession(
        id = "7",
        title = "Pain Acceptance",
        description = "ACT-based meditation for chronic pain",
        durationMinutes = 15,
        category = MeditationCategory.PAIN_RELIEF,
        icon = Icons.Default.Favorite
    ),
    MeditationSession(
        id = "8",
        title = "Evening Wind Down",
        description = "Gentle relaxation for end of day",
        durationMinutes = 12,
        category = MeditationCategory.SLEEP,
        icon = Icons.Default.NightsStay
    )
)

@HiltViewModel
class MeditationViewModel @Inject constructor(
    private val meditationRepository: MeditationRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(MeditationUiState())
    val uiState: StateFlow<MeditationUiState> = _uiState.asStateFlow()

    private var timerJob: Job? = null
    private var breathingJob: Job? = null

    init {
        loadStreakData()
    }

    private fun loadStreakData() {
        viewModelScope.launch {
            meditationRepository.observeStreak().collect { streak ->
                _uiState.value = _uiState.value.copy(
                    currentStreak = streak?.currentStreak ?: 0,
                    longestStreak = streak?.longestStreak ?: 0,
                    totalMinutes = streak?.totalMinutes ?: 0
                )
            }
        }
    }

    fun selectCategory(category: MeditationCategory?) {
        val filtered = if (category == null) {
            defaultSessions
        } else {
            defaultSessions.filter { it.category == category }
        }

        _uiState.value = _uiState.value.copy(
            selectedCategory = if (_uiState.value.selectedCategory == category) null else category,
            filteredSessions = if (_uiState.value.selectedCategory == category) defaultSessions else filtered
        )
    }

    fun startSession(session: MeditationSession) {
        _uiState.value = _uiState.value.copy(
            currentSession = session,
            isPlaying = true,
            isPaused = false,
            elapsedSeconds = 0
        )
        startTimer()

        if (session.category == MeditationCategory.BREATHING) {
            startBreathingCycle()
        }
    }

    fun startBreathingExercise() {
        val breathingSession = MeditationSession(
            id = "breathing_quick",
            title = "4-7-8 Breathing",
            description = "Quick breathing exercise",
            durationMinutes = 3,
            category = MeditationCategory.BREATHING,
            icon = Icons.Default.Air
        )
        startSession(breathingSession)
    }

    private fun startTimer() {
        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            while (_uiState.value.isPlaying && !_uiState.value.isPaused) {
                delay(1000)
                val newElapsed = _uiState.value.elapsedSeconds + 1
                val totalSeconds = (_uiState.value.currentSession?.durationMinutes ?: 0) * 60

                if (newElapsed >= totalSeconds) {
                    completeSession()
                } else {
                    _uiState.value = _uiState.value.copy(elapsedSeconds = newElapsed)
                }
            }
        }
    }

    private fun startBreathingCycle() {
        breathingJob?.cancel()
        breathingJob = viewModelScope.launch {
            // 4-7-8 breathing cycle
            while (_uiState.value.isPlaying && !_uiState.value.isPaused) {
                // Inhale - 4 seconds
                _uiState.value = _uiState.value.copy(breathingPhase = "inhale")
                delay(4000)

                // Hold - 7 seconds
                _uiState.value = _uiState.value.copy(breathingPhase = "hold")
                delay(7000)

                // Exhale - 8 seconds
                _uiState.value = _uiState.value.copy(breathingPhase = "exhale")
                delay(8000)
            }
        }
    }

    fun pauseSession() {
        timerJob?.cancel()
        breathingJob?.cancel()
        _uiState.value = _uiState.value.copy(isPaused = true)
    }

    fun resumeSession() {
        _uiState.value = _uiState.value.copy(isPaused = false)
        startTimer()

        if (_uiState.value.currentSession?.category == MeditationCategory.BREATHING) {
            startBreathingCycle()
        }
    }

    fun stopSession() {
        timerJob?.cancel()
        breathingJob?.cancel()
        _uiState.value = _uiState.value.copy(
            currentSession = null,
            isPlaying = false,
            isPaused = false,
            elapsedSeconds = 0
        )
    }

    private fun completeSession() {
        val session = _uiState.value.currentSession ?: return

        timerJob?.cancel()
        breathingJob?.cancel()

        // Save session to database and update streak
        viewModelScope.launch {
            meditationRepository.logMeditationSession(
                sessionType = session.title,
                durationMinutes = session.durationMinutes,
                category = session.category.name
            )
        }

        _uiState.value = _uiState.value.copy(
            currentSession = null,
            isPlaying = false,
            isPaused = false,
            elapsedSeconds = 0
        )
    }

    override fun onCleared() {
        super.onCleared()
        timerJob?.cancel()
        breathingJob?.cancel()
    }
}
