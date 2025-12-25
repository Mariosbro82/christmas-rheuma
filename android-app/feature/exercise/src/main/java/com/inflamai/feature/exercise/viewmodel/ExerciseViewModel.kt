package com.inflamai.feature.exercise.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.database.dao.ExerciseSessionDao
import com.inflamai.core.data.database.entity.ExerciseSessionEntity
import com.inflamai.feature.exercise.model.Difficulty
import com.inflamai.feature.exercise.model.Exercise
import com.inflamai.feature.exercise.model.ExerciseCategory
import com.inflamai.feature.exercise.model.ExerciseLibrary
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.temporal.ChronoUnit
import java.util.UUID
import javax.inject.Inject

/**
 * ViewModel for Exercise Library feature
 *
 * Manages exercise library, workout logging, and routine generation.
 */
@HiltViewModel
class ExerciseViewModel @Inject constructor(
    private val exerciseSessionDao: ExerciseSessionDao
) : ViewModel() {

    private val _uiState = MutableStateFlow(ExerciseUiState())
    val uiState: StateFlow<ExerciseUiState> = _uiState.asStateFlow()

    private val _selectedExercise = MutableStateFlow<Exercise?>(null)
    val selectedExercise: StateFlow<Exercise?> = _selectedExercise.asStateFlow()

    private val _activeWorkout = MutableStateFlow<ActiveWorkout?>(null)
    val activeWorkout: StateFlow<ActiveWorkout?> = _activeWorkout.asStateFlow()

    init {
        loadExercises()
        loadStats()
    }

    private fun loadExercises() {
        val allExercises = ExerciseLibrary.exercises
        _uiState.update { state ->
            state.copy(
                allExercises = allExercises,
                filteredExercises = allExercises,
                categories = ExerciseCategory.entries.toList()
            )
        }
    }

    private fun loadStats() {
        viewModelScope.launch {
            val now = System.currentTimeMillis()
            val sevenDaysAgo = now - (7 * 24 * 60 * 60 * 1000L)

            exerciseSessionDao.observeByDateRange(sevenDaysAgo, now).collect { sessions ->
                val weeklyMinutes = sessions.sumOf { it.durationMinutes }
                val weeklyCount = sessions.size

                _uiState.update { state ->
                    state.copy(
                        weeklyMinutes = weeklyMinutes,
                        weeklySessionCount = weeklyCount,
                        recentSessions = sessions.take(5)
                    )
                }
            }
        }
    }

    fun filterByCategory(category: ExerciseCategory?) {
        _uiState.update { state ->
            val filtered = if (category == null) {
                state.allExercises
            } else {
                ExerciseLibrary.getExercisesByCategory(category)
            }
            state.copy(
                selectedCategory = category,
                filteredExercises = filtered
            )
        }
    }

    fun filterByDifficulty(difficulty: Difficulty?) {
        _uiState.update { state ->
            val filtered = if (difficulty == null) {
                state.allExercises
            } else {
                ExerciseLibrary.getExercisesByDifficulty(difficulty)
            }
            state.copy(
                selectedDifficulty = difficulty,
                filteredExercises = filtered
            )
        }
    }

    fun searchExercises(query: String) {
        _uiState.update { state ->
            val filtered = if (query.isBlank()) {
                state.allExercises
            } else {
                ExerciseLibrary.searchExercises(query)
            }
            state.copy(
                searchQuery = query,
                filteredExercises = filtered
            )
        }
    }

    fun selectExercise(exercise: Exercise) {
        _selectedExercise.value = exercise
    }

    fun clearSelectedExercise() {
        _selectedExercise.value = null
    }

    fun generateQuickRoutine(durationMinutes: Int) {
        val difficulty = _uiState.value.selectedDifficulty ?: Difficulty.BEGINNER
        val routine = ExerciseLibrary.getQuickRoutine(durationMinutes, difficulty)

        _uiState.update { state ->
            state.copy(generatedRoutine = routine)
        }
    }

    fun startWorkout(exercises: List<Exercise>) {
        _activeWorkout.value = ActiveWorkout(
            exercises = exercises,
            currentIndex = 0,
            startTime = Instant.now(),
            completedExercises = emptySet()
        )
    }

    fun completeExercise(exerciseId: String) {
        _activeWorkout.update { workout ->
            workout?.copy(
                completedExercises = workout.completedExercises + exerciseId,
                currentIndex = minOf(workout.currentIndex + 1, workout.exercises.size - 1)
            )
        }
    }

    fun finishWorkout(painBefore: Int?, painAfter: Int?, notes: String?) {
        viewModelScope.launch {
            val workout = _activeWorkout.value ?: return@launch
            val duration = ChronoUnit.MINUTES.between(workout.startTime, Instant.now()).toInt()
            val completedIds = workout.completedExercises.toList()

            val session = ExerciseSessionEntity(
                id = UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                routineName = workout.exercises.firstOrNull()?.category?.name ?: "MIXED",
                exercisesCompletedJson = completedIds.joinToString(",", "[\"", "\"]"),
                exerciseCount = completedIds.size,
                durationMinutes = maxOf(duration, 1),
                painBefore = painBefore ?: 0,
                painAfter = painAfter ?: 0,
                painDelta = (painAfter ?: 0) - (painBefore ?: 0),
                hadPainIncrease = (painAfter ?: 0) > (painBefore ?: 0),
                notes = notes
            )

            exerciseSessionDao.insert(session)
            _activeWorkout.value = null
            loadStats()
        }
    }

    fun cancelWorkout() {
        _activeWorkout.value = null
    }
}

data class ExerciseUiState(
    val allExercises: List<Exercise> = emptyList(),
    val filteredExercises: List<Exercise> = emptyList(),
    val categories: List<ExerciseCategory> = emptyList(),
    val selectedCategory: ExerciseCategory? = null,
    val selectedDifficulty: Difficulty? = null,
    val searchQuery: String = "",
    val generatedRoutine: List<Exercise>? = null,
    val weeklyMinutes: Int = 0,
    val weeklySessionCount: Int = 0,
    val recentSessions: List<ExerciseSessionEntity> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null
)

data class ActiveWorkout(
    val exercises: List<Exercise>,
    val currentIndex: Int,
    val startTime: Instant,
    val completedExercises: Set<String>
)
