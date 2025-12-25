package com.inflamai.app

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.inflamai.app.navigation.InflamAINavHost
import com.inflamai.app.security.BiometricAuthManager
import com.inflamai.core.ui.theme.InflamAITheme
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Main Activity for InflamAI
 *
 * Features:
 * - Biometric authentication on launch (deferred until foreground)
 * - Edge-to-edge display
 * - Material Design 3 theming
 * - Single Activity architecture with Compose Navigation
 */
@AndroidEntryPoint
class MainActivity : FragmentActivity() {

    @Inject
    lateinit var biometricAuthManager: BiometricAuthManager

    // Authentication state management
    private val _authState = MutableStateFlow<AuthState>(AuthState.Initializing)
    private val authState: StateFlow<AuthState> = _authState.asStateFlow()

    // Track if we need to authenticate on resume
    private var pendingAuthentication = false
    private var isFirstResume = true

    override fun onCreate(savedInstanceState: Bundle?) {
        val splashScreen = installSplashScreen()
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        // Keep splash screen until we determine auth state
        splashScreen.setKeepOnScreenCondition {
            _authState.value == AuthState.Initializing
        }

        // Setup content immediately with state-driven UI
        setContent {
            val currentAuthState by authState.collectAsState()

            InflamAITheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    when (currentAuthState) {
                        AuthState.Initializing -> {
                            // Splash screen is showing, nothing to render
                            Box(modifier = Modifier.fillMaxSize())
                        }
                        AuthState.Authenticated -> {
                            InflamAINavHost()
                        }
                        AuthState.RequiresAuthentication -> {
                            AuthenticationRequiredScreen(
                                onRetry = { triggerAuthentication() }
                            )
                        }
                    }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()

        // Defer authentication check until activity is fully in foreground
        // Use post to ensure we're past the activity transition
        window.decorView.post {
            if (isFirstResume) {
                isFirstResume = false
                checkInitialAuthState()
            } else if (pendingAuthentication || biometricAuthManager.isLocked()) {
                pendingAuthentication = false
                triggerAuthentication()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        // Lock the app when going to background if biometric lock is enabled
        if (biometricAuthManager.isBiometricLockEnabled()) {
            biometricAuthManager.lockApp()
            pendingAuthentication = true
        }
    }

    /**
     * Check initial authentication state after first resume
     */
    private fun checkInitialAuthState() {
        lifecycleScope.launch {
            val lockEnabled = biometricAuthManager.isBiometricLockEnabledSuspend()
            if (lockEnabled) {
                triggerAuthentication()
            } else {
                _authState.value = AuthState.Authenticated
            }
        }
    }

    /**
     * Trigger biometric authentication when activity is in foreground
     */
    private fun triggerAuthentication() {
        // Ensure we're in a valid state for biometric prompt
        if (!lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) {
            pendingAuthentication = true
            return
        }

        biometricAuthManager.authenticate(
            activity = this@MainActivity,
            onSuccess = {
                _authState.value = AuthState.Authenticated
            },
            onError = { errorMessage ->
                _authState.value = AuthState.RequiresAuthentication
            },
            onFallback = {
                _authState.value = AuthState.Authenticated
            }
        )
    }
}

/**
 * Authentication state for the activity
 */
sealed class AuthState {
    object Initializing : AuthState()
    object Authenticated : AuthState()
    object RequiresAuthentication : AuthState()
}

@Composable
fun AuthenticationRequiredScreen(onRetry: () -> Unit) {
    // Placeholder for authentication required UI
    androidx.compose.foundation.layout.Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            androidx.compose.material3.Text(
                text = "Authentication Required",
                style = MaterialTheme.typography.headlineSmall
            )
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = onRetry) {
                androidx.compose.material3.Text("Authenticate")
            }
        }
    }
}
