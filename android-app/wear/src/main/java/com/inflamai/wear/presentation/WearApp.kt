package com.inflamai.wear.presentation

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.wear.compose.material.*
import androidx.wear.compose.navigation.SwipeDismissableNavHost
import androidx.wear.compose.navigation.composable
import androidx.wear.compose.navigation.rememberSwipeDismissableNavController
import com.inflamai.wear.presentation.screen.HomeScreen
import com.inflamai.wear.presentation.screen.QuickCheckInScreen
import com.inflamai.wear.presentation.screen.PainLogScreen
import com.inflamai.wear.presentation.screen.SOSFlareScreen
import com.inflamai.wear.presentation.theme.InflamAIWearTheme

/**
 * InflamAI Wear OS App
 *
 * Companion app for quick symptom logging and monitoring.
 *
 * Features:
 * - Quick symptom check-in (3 taps)
 * - Pain level logging
 * - SOS flare recording
 * - Today's BASDAI score display
 * - Complications for watch face
 */
@Composable
fun WearApp() {
    InflamAIWearTheme {
        val navController = rememberSwipeDismissableNavController()

        Scaffold(
            timeText = { TimeText() },
            vignette = { Vignette(vignettePosition = VignettePosition.TopAndBottom) }
        ) {
            SwipeDismissableNavHost(
                navController = navController,
                startDestination = WearScreen.Home.route
            ) {
                composable(WearScreen.Home.route) {
                    HomeScreen(
                        onNavigateToQuickCheckIn = {
                            navController.navigate(WearScreen.QuickCheckIn.route)
                        },
                        onNavigateToPainLog = {
                            navController.navigate(WearScreen.PainLog.route)
                        },
                        onNavigateToSOS = {
                            navController.navigate(WearScreen.SOSFlare.route)
                        }
                    )
                }

                composable(WearScreen.QuickCheckIn.route) {
                    QuickCheckInScreen(
                        onComplete = { navController.popBackStack() }
                    )
                }

                composable(WearScreen.PainLog.route) {
                    PainLogScreen(
                        onComplete = { navController.popBackStack() }
                    )
                }

                composable(WearScreen.SOSFlare.route) {
                    SOSFlareScreen(
                        onComplete = { navController.popBackStack() }
                    )
                }
            }
        }
    }
}

sealed class WearScreen(val route: String) {
    object Home : WearScreen("home")
    object QuickCheckIn : WearScreen("quick_checkin")
    object PainLog : WearScreen("pain_log")
    object SOSFlare : WearScreen("sos_flare")
}
