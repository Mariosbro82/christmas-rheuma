package com.inflamai.app.security

import android.content.Context
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import androidx.fragment.app.FragmentActivity
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import java.util.concurrent.Executor
import javax.inject.Inject
import javax.inject.Singleton

private val Context.securityDataStore: DataStore<Preferences> by preferencesDataStore(name = "security_prefs")

/**
 * Biometric Authentication Manager
 *
 * Handles Face ID / Fingerprint authentication for app security.
 * Equivalent to iOS LocalAuthentication framework usage.
 *
 * Features:
 * - Check biometric availability
 * - Authenticate with biometrics
 * - Fallback to device credentials
 * - Auto-lock on app background
 * - Non-blocking preference reads
 */
@Singleton
class BiometricAuthManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val biometricManager = BiometricManager.from(context)
    private var isLocked = true

    // Cached biometric lock state to avoid blocking main thread
    private val _biometricLockEnabled = MutableStateFlow(false)
    val biometricLockEnabledFlow: StateFlow<Boolean> = _biometricLockEnabled.asStateFlow()

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    init {
        // Load preference in background on initialization
        scope.launch {
            context.securityDataStore.data
                .map { preferences -> preferences[BIOMETRIC_LOCK_ENABLED] ?: false }
                .collect { enabled ->
                    _biometricLockEnabled.value = enabled
                }
        }
    }

    companion object {
        private val BIOMETRIC_LOCK_ENABLED = booleanPreferencesKey("biometric_lock_enabled")
    }

    /**
     * Check if biometric authentication is available on this device
     */
    fun isBiometricAvailable(): BiometricAvailability {
        return when (biometricManager.canAuthenticate(
            BiometricManager.Authenticators.BIOMETRIC_STRONG or
            BiometricManager.Authenticators.DEVICE_CREDENTIAL
        )) {
            BiometricManager.BIOMETRIC_SUCCESS -> BiometricAvailability.AVAILABLE
            BiometricManager.BIOMETRIC_ERROR_NO_HARDWARE -> BiometricAvailability.NO_HARDWARE
            BiometricManager.BIOMETRIC_ERROR_HW_UNAVAILABLE -> BiometricAvailability.HARDWARE_UNAVAILABLE
            BiometricManager.BIOMETRIC_ERROR_NONE_ENROLLED -> BiometricAvailability.NOT_ENROLLED
            BiometricManager.BIOMETRIC_ERROR_SECURITY_UPDATE_REQUIRED -> BiometricAvailability.SECURITY_UPDATE_REQUIRED
            else -> BiometricAvailability.UNKNOWN
        }
    }

    /**
     * Check if biometric lock is enabled in user preferences (non-blocking, uses cached value)
     */
    fun isBiometricLockEnabled(): Boolean {
        return _biometricLockEnabled.value
    }

    /**
     * Suspend function to check biometric lock status (for coroutine contexts)
     */
    suspend fun isBiometricLockEnabledSuspend(): Boolean {
        return context.securityDataStore.data
            .map { preferences -> preferences[BIOMETRIC_LOCK_ENABLED] ?: false }
            .first()
    }

    /**
     * Enable or disable biometric lock
     */
    suspend fun setBiometricLockEnabled(enabled: Boolean) {
        context.securityDataStore.edit { preferences ->
            preferences[BIOMETRIC_LOCK_ENABLED] = enabled
        }
    }

    /**
     * Check if the app is currently locked
     */
    fun isLocked(): Boolean = isLocked

    /**
     * Lock the app (called when going to background)
     */
    fun lockApp() {
        if (isBiometricLockEnabled()) {
            isLocked = true
        }
    }

    /**
     * Authenticate with biometrics
     */
    fun authenticate(
        activity: FragmentActivity,
        onSuccess: () -> Unit,
        onError: (String) -> Unit,
        onFallback: () -> Unit
    ) {
        val executor: Executor = ContextCompat.getMainExecutor(activity)

        val callback = object : BiometricPrompt.AuthenticationCallback() {
            override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                super.onAuthenticationSucceeded(result)
                isLocked = false
                onSuccess()
            }

            override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                super.onAuthenticationError(errorCode, errString)
                when (errorCode) {
                    BiometricPrompt.ERROR_NEGATIVE_BUTTON,
                    BiometricPrompt.ERROR_USER_CANCELED -> {
                        onError("Authentication cancelled")
                    }
                    BiometricPrompt.ERROR_LOCKOUT,
                    BiometricPrompt.ERROR_LOCKOUT_PERMANENT -> {
                        onError("Too many attempts. Please try again later.")
                    }
                    else -> {
                        onError(errString.toString())
                    }
                }
            }

            override fun onAuthenticationFailed() {
                super.onAuthenticationFailed()
                // Don't call onError here - system will show message
            }
        }

        val biometricPrompt = BiometricPrompt(activity, executor, callback)

        val promptInfo = BiometricPrompt.PromptInfo.Builder()
            .setTitle("InflamAI")
            .setSubtitle("Authenticate to access your health data")
            .setDescription("Use your fingerprint or face to unlock")
            .setAllowedAuthenticators(
                BiometricManager.Authenticators.BIOMETRIC_STRONG or
                BiometricManager.Authenticators.DEVICE_CREDENTIAL
            )
            .build()

        biometricPrompt.authenticate(promptInfo)
    }
}

enum class BiometricAvailability {
    AVAILABLE,
    NO_HARDWARE,
    HARDWARE_UNAVAILABLE,
    NOT_ENROLLED,
    SECURITY_UPDATE_REQUIRED,
    UNKNOWN
}
