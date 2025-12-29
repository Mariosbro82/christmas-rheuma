package com.inflamai.feature.checkin.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.database.dao.ContextSnapshotDao
import com.inflamai.core.data.database.dao.SymptomLogDao
import com.inflamai.core.data.database.dao.UserProfileDao
import com.inflamai.core.data.database.entity.ContextSnapshotEntity
import com.inflamai.core.data.database.entity.SymptomLogEntity
import com.inflamai.core.data.service.health.HealthConnectService
import com.inflamai.core.data.service.weather.WeatherService
import com.inflamai.core.domain.calculator.BASDAICalculator
import com.inflamai.core.domain.calculator.BASDAIInterpretation
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.Instant
import java.util.UUID
import javax.inject.Inject

/**
 * Check-In ViewModel
 *
 * Manages the 6-question BASDAI assessment flow plus additional symptom questions.
 *
 * BASDAI Questions:
 * Q1: Fatigue (0-10)
 * Q2: Spinal pain (0-10)
 * Q3: Peripheral joint pain/swelling (0-10)
 * Q4: Enthesitis/tenderness (0-10)
 * Q5: Morning stiffness severity (0-10)
 * Q6: Morning stiffness duration (minutes → scaled to 0-10)
 *
 * Additional questions:
 * - Mood (1-10)
 * - Sleep quality (1-10)
 * - Sleep duration (hours)
 * - Overall feeling (1-10)
 */
@HiltViewModel
class CheckInViewModel @Inject constructor(
    private val symptomLogDao: SymptomLogDao,
    private val contextSnapshotDao: ContextSnapshotDao,
    private val userProfileDao: UserProfileDao,
    private val healthConnectService: HealthConnectService,
    private val weatherService: WeatherService
) : ViewModel() {

    private val _uiState = MutableStateFlow(CheckInUiState())
    val uiState: StateFlow<CheckInUiState> = _uiState.asStateFlow()

    // BASDAI question answers
    private var q1Fatigue: Double = 0.0
    private var q2SpinalPain: Double = 0.0
    private var q3PeripheralPain: Double = 0.0
    private var q4Enthesitis: Double = 0.0
    private var q5MorningSeverity: Double = 0.0
    private var q6MorningDurationMinutes: Int = 0

    // Additional questions
    private var mood: Int = 5
    private var sleepQuality: Int = 5
    private var sleepDurationHours: Double = 7.0
    private var overallFeeling: Int = 5
    private var notes: String = ""

    val questions = listOf(
        CheckInQuestion(
            id = "q1_fatigue",
            title = "Fatigue",
            question = "How would you describe the overall level of fatigue/tiredness you have experienced?",
            type = QuestionType.SLIDER,
            minValue = 0,
            maxValue = 10,
            minLabel = "None",
            maxLabel = "Very severe"
        ),
        CheckInQuestion(
            id = "q2_spinal_pain",
            title = "Spinal Pain",
            question = "How would you describe the overall level of AS neck, back, or hip pain you have had?",
            type = QuestionType.SLIDER,
            minValue = 0,
            maxValue = 10,
            minLabel = "None",
            maxLabel = "Very severe"
        ),
        CheckInQuestion(
            id = "q3_peripheral_pain",
            title = "Joint Pain",
            question = "How would you describe the overall level of pain/swelling in joints other than neck, back, or hips?",
            type = QuestionType.SLIDER,
            minValue = 0,
            maxValue = 10,
            minLabel = "None",
            maxLabel = "Very severe"
        ),
        CheckInQuestion(
            id = "q4_enthesitis",
            title = "Tenderness",
            question = "How would you describe the overall level of discomfort from any areas tender to touch or pressure?",
            type = QuestionType.SLIDER,
            minValue = 0,
            maxValue = 10,
            minLabel = "None",
            maxLabel = "Very severe"
        ),
        CheckInQuestion(
            id = "q5_morning_severity",
            title = "Morning Stiffness",
            question = "How would you describe the overall level of morning stiffness you have had from the time you wake up?",
            type = QuestionType.SLIDER,
            minValue = 0,
            maxValue = 10,
            minLabel = "None",
            maxLabel = "Very severe"
        ),
        CheckInQuestion(
            id = "q6_morning_duration",
            title = "Stiffness Duration",
            question = "How long does your morning stiffness last from the time you wake up?",
            type = QuestionType.DURATION,
            minValue = 0,
            maxValue = 120,
            minLabel = "0 min",
            maxLabel = "2+ hours"
        ),
        CheckInQuestion(
            id = "mood",
            title = "Mood",
            question = "How would you rate your overall mood today?",
            type = QuestionType.SLIDER,
            minValue = 1,
            maxValue = 10,
            minLabel = "Very low",
            maxLabel = "Excellent"
        ),
        CheckInQuestion(
            id = "sleep_quality",
            title = "Sleep Quality",
            question = "How would you rate your sleep quality last night?",
            type = QuestionType.SLIDER,
            minValue = 1,
            maxValue = 10,
            minLabel = "Very poor",
            maxLabel = "Excellent"
        ),
        CheckInQuestion(
            id = "overall",
            title = "Overall",
            question = "Overall, how are you feeling today?",
            type = QuestionType.SLIDER,
            minValue = 1,
            maxValue = 10,
            minLabel = "Very unwell",
            maxLabel = "Great"
        ),
        CheckInQuestion(
            id = "notes",
            title = "Notes",
            question = "Any additional notes about how you're feeling?",
            type = QuestionType.TEXT,
            minValue = 0,
            maxValue = 0,
            isOptional = true
        )
    )

    fun updateAnswer(questionId: String, value: Any) {
        when (questionId) {
            "q1_fatigue" -> q1Fatigue = (value as Number).toDouble()
            "q2_spinal_pain" -> q2SpinalPain = (value as Number).toDouble()
            "q3_peripheral_pain" -> q3PeripheralPain = (value as Number).toDouble()
            "q4_enthesitis" -> q4Enthesitis = (value as Number).toDouble()
            "q5_morning_severity" -> q5MorningSeverity = (value as Number).toDouble()
            "q6_morning_duration" -> q6MorningDurationMinutes = (value as Number).toInt()
            "mood" -> mood = (value as Number).toInt()
            "sleep_quality" -> sleepQuality = (value as Number).toInt()
            "overall" -> overallFeeling = (value as Number).toInt()
            "notes" -> notes = value as String
        }

        // Update current answer in UI state
        _uiState.update { state ->
            state.copy(
                answers = state.answers + (questionId to value)
            )
        }
    }

    fun nextQuestion() {
        val currentIndex = _uiState.value.currentQuestionIndex
        if (currentIndex < questions.size - 1) {
            _uiState.update { it.copy(currentQuestionIndex = currentIndex + 1) }
        }
    }

    fun previousQuestion() {
        val currentIndex = _uiState.value.currentQuestionIndex
        if (currentIndex > 0) {
            _uiState.update { it.copy(currentQuestionIndex = currentIndex - 1) }
        }
    }

    fun goToQuestion(index: Int) {
        if (index in questions.indices) {
            _uiState.update { it.copy(currentQuestionIndex = index) }
        }
    }

    fun calculatePreviewScore(): Double {
        val scaledDuration = BASDAICalculator.scaleMorningStiffnessDuration(q6MorningDurationMinutes)
        return BASDAICalculator.calculate(
            fatigue = q1Fatigue,
            spinalPain = q2SpinalPain,
            peripheralPain = q3PeripheralPain,
            enthesitisPain = q4Enthesitis,
            morningSeverity = q5MorningSeverity,
            morningDuration = scaledDuration
        )
    }

    fun submitCheckIn(onComplete: () -> Unit) {
        viewModelScope.launch {
            _uiState.update { it.copy(isSubmitting = true) }

            try {
                // Calculate BASDAI score
                val scaledDuration = BASDAICalculator.scaleMorningStiffnessDuration(q6MorningDurationMinutes)
                val basdaiScore = BASDAICalculator.calculate(
                    fatigue = q1Fatigue,
                    spinalPain = q2SpinalPain,
                    peripheralPain = q3PeripheralPain,
                    enthesitisPain = q4Enthesitis,
                    morningSeverity = q5MorningSeverity,
                    morningDuration = scaledDuration
                )

                val interpretation = BASDAICalculator.interpret(basdaiScore)

                // Create symptom log
                val symptomLogId = UUID.randomUUID().toString()
                val symptomLog = SymptomLogEntity(
                    id = symptomLogId,
                    timestamp = System.currentTimeMillis(),
                    basdaiScore = basdaiScore,
                    fatigueLevel = q1Fatigue.toInt(),
                    moodScore = mood,
                    sleepQuality = sleepQuality,
                    sleepDurationHours = sleepDurationHours,
                    morningStiffnessMinutes = q6MorningDurationMinutes,
                    isFlareEvent = basdaiScore >= 6.0,
                    q1Fatigue = q1Fatigue,
                    q2SpinalPain = q2SpinalPain,
                    q3PeripheralPain = q3PeripheralPain,
                    q4Tenderness = q4Enthesitis,
                    q5MorningStiffnessSeverity = q5MorningSeverity,
                    q6MorningStiffnessDuration = scaledDuration,
                    overallFeeling = overallFeeling,
                    notes = notes.ifBlank { null }
                )

                symptomLogDao.insert(symptomLog)

                // Capture context snapshot
                captureContextSnapshot(symptomLogId)

                // Update user profile streak
                updateStreak()

                _uiState.update { state ->
                    state.copy(
                        isSubmitting = false,
                        isComplete = true,
                        finalScore = basdaiScore,
                        interpretation = interpretation
                    )
                }

                onComplete()

            } catch (e: Exception) {
                _uiState.update { state ->
                    state.copy(
                        isSubmitting = false,
                        error = "Failed to save check-in: ${e.message}"
                    )
                }
            }
        }
    }

    private suspend fun captureContextSnapshot(symptomLogId: String) {
        try {
            // Get weather data
            val weather = weatherService.getCurrentWeather()

            // Get health data
            val healthSnapshot = healthConnectService.getDailyHealthSnapshot(Instant.now())

            val snapshot = ContextSnapshotEntity(
                symptomLogId = symptomLogId,
                timestamp = System.currentTimeMillis(),
                barometricPressure = weather?.barometricPressure,
                pressureChange12h = weather?.pressureChange12h,
                humidity = weather?.humidity,
                temperature = weather?.temperature,
                weatherCondition = weather?.weatherCondition,
                hrvValue = healthSnapshot?.latestHrv,
                restingHeartRate = healthSnapshot?.restingHeartRate,
                stepCount = healthSnapshot?.stepCount,
                sleepDurationHours = healthSnapshot?.sleepDurationMinutes?.toDouble()?.div(60),
                sleepEfficiency = healthSnapshot?.sleepEfficiency
            )

            contextSnapshotDao.insert(snapshot)
        } catch (e: Exception) {
            // Context snapshot is optional, don't fail the check-in
        }
    }

    private suspend fun updateStreak() {
        try {
            val profile = userProfileDao.get()
            if (profile != null) {
                // Simple streak increment - in production, check for consecutive days
                userProfileDao.incrementCheckIn(profile.streakDays + 1)
            }
        } catch (e: Exception) {
            // Streak update is optional
        }
    }

    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }
}

data class CheckInUiState(
    val currentQuestionIndex: Int = 0,
    val answers: Map<String, Any> = emptyMap(),
    val isSubmitting: Boolean = false,
    val isComplete: Boolean = false,
    val finalScore: Double? = null,
    val interpretation: BASDAIInterpretation? = null,
    val error: String? = null
)

data class CheckInQuestion(
    val id: String,
    val title: String,
    val question: String,
    val type: QuestionType,
    val minValue: Int = 0,
    val maxValue: Int = 10,
    val minLabel: String = "",
    val maxLabel: String = "",
    val isOptional: Boolean = false
)

enum class QuestionType {
    SLIDER,
    DURATION,
    TEXT
}
