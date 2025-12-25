package com.inflamai.feature.onboarding

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.inflamai.core.data.repository.UserRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Onboarding UI State
 * Based on Frame Analysis: 11 onboarding pages
 */
data class OnboardingUiState(
    val currentStep: Int = 0,
    val totalSteps: Int = 11,
    val isComplete: Boolean = false,
    val healthConnectRequested: Boolean = false,
    val notificationsRequested: Boolean = false
)

/**
 * Onboarding ViewModel
 * Manages onboarding flow state and completion
 */
@HiltViewModel
class OnboardingViewModel @Inject constructor(
    private val userRepository: UserRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(OnboardingUiState())
    val uiState: StateFlow<OnboardingUiState> = _uiState.asStateFlow()

    init {
        // Ensure user profile exists
        viewModelScope.launch {
            userRepository.ensureProfileExists()
        }
    }

    fun setStep(step: Int) {
        if (step in 0 until _uiState.value.totalSteps) {
            _uiState.value = _uiState.value.copy(currentStep = step)
        }
    }

    fun nextStep() {
        val current = _uiState.value.currentStep
        if (current < _uiState.value.totalSteps - 1) {
            _uiState.value = _uiState.value.copy(currentStep = current + 1)
        }
    }

    fun previousStep() {
        val current = _uiState.value.currentStep
        if (current > 0) {
            _uiState.value = _uiState.value.copy(currentStep = current - 1)
        }
    }

    fun requestHealthConnect() {
        _uiState.value = _uiState.value.copy(healthConnectRequested = true)
        // Health Connect permission request would be triggered from Activity/Fragment
    }

    fun requestNotifications() {
        viewModelScope.launch {
            userRepository.updateNotificationsEnabled(true)
        }
        _uiState.value = _uiState.value.copy(notificationsRequested = true)
    }

    fun completeOnboarding() {
        viewModelScope.launch {
            userRepository.setOnboardingComplete()
            _uiState.value = _uiState.value.copy(isComplete = true)
        }
    }
}
