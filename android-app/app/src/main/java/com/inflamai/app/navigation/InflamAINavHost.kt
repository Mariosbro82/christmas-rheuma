package com.inflamai.app.navigation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.FitnessCenter
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.MedicalServices
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.Analytics
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.FitnessCenter
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.MedicalServices
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.inflamai.feature.home.ui.HomeScreen
import com.inflamai.feature.checkin.ui.CheckInScreen
import com.inflamai.feature.trends.ui.TrendsScreen
import com.inflamai.feature.exercise.ui.ExerciseScreen
import com.inflamai.feature.settings.ui.SettingsScreen
import com.inflamai.feature.bodymap.ui.BodyMapScreen
import com.inflamai.feature.medication.ui.MedicationScreen
import com.inflamai.feature.flares.ui.FlaresScreen
import com.inflamai.feature.ai.AIScreen
import com.inflamai.feature.quickcapture.QuickCaptureScreen
import com.inflamai.feature.meditation.MeditationScreen
import com.inflamai.feature.onboarding.OnboardingScreen

/**
 * Navigation routes for InflamAI
 */
sealed class Screen(val route: String) {
    // Main tabs
    object Home : Screen("home")
    object CheckIn : Screen("checkin")
    object Trends : Screen("trends")
    object Exercise : Screen("exercise")
    object Settings : Screen("settings")

    // Secondary screens (navigated from main tabs)
    object BodyMap : Screen("bodymap")
    object Medication : Screen("medication")
    object Flares : Screen("flares")
    object AI : Screen("ai")
    object QuickCapture : Screen("quickcapture")
    object Meditation : Screen("meditation")
    object Onboarding : Screen("onboarding")

    // Detail screens
    object SymptomDetail : Screen("symptom/{id}") {
        fun createRoute(id: String) = "symptom/$id"
    }
    object MedicationDetail : Screen("medication/{id}") {
        fun createRoute(id: String) = "medication/$id"
    }
    object FlareDetail : Screen("flare/{id}") {
        fun createRoute(id: String) = "flare/$id"
    }
    object ExerciseDetail : Screen("exercise/{id}") {
        fun createRoute(id: String) = "exercise/$id"
    }
}

/**
 * Bottom navigation items
 */
data class BottomNavItem(
    val screen: Screen,
    val label: String,
    val selectedIcon: ImageVector,
    val unselectedIcon: ImageVector,
    val contentDescription: String
)

val bottomNavItems = listOf(
    BottomNavItem(
        screen = Screen.Home,
        label = "Home",
        selectedIcon = Icons.Filled.Home,
        unselectedIcon = Icons.Outlined.Home,
        contentDescription = "Home dashboard"
    ),
    BottomNavItem(
        screen = Screen.CheckIn,
        label = "Check-in",
        selectedIcon = Icons.Filled.CheckCircle,
        unselectedIcon = Icons.Outlined.CheckCircle,
        contentDescription = "Daily symptom check-in"
    ),
    BottomNavItem(
        screen = Screen.Trends,
        label = "Trends",
        selectedIcon = Icons.Filled.Analytics,
        unselectedIcon = Icons.Outlined.Analytics,
        contentDescription = "View symptom trends"
    ),
    BottomNavItem(
        screen = Screen.Exercise,
        label = "Exercise",
        selectedIcon = Icons.Filled.FitnessCenter,
        unselectedIcon = Icons.Outlined.FitnessCenter,
        contentDescription = "Exercise library"
    ),
    BottomNavItem(
        screen = Screen.Settings,
        label = "Settings",
        selectedIcon = Icons.Filled.Settings,
        unselectedIcon = Icons.Outlined.Settings,
        contentDescription = "App settings"
    )
)

@Composable
fun InflamAINavHost() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    // Check if we should show bottom nav (hide on detail screens)
    val showBottomNav = bottomNavItems.any { it.screen.route == currentDestination?.route }

    Scaffold(
        bottomBar = {
            if (showBottomNav) {
                NavigationBar {
                    bottomNavItems.forEach { item ->
                        val selected = currentDestination?.hierarchy?.any {
                            it.route == item.screen.route
                        } == true

                        NavigationBarItem(
                            icon = {
                                Icon(
                                    imageVector = if (selected) item.selectedIcon else item.unselectedIcon,
                                    contentDescription = item.contentDescription
                                )
                            },
                            label = { Text(item.label) },
                            selected = selected,
                            onClick = {
                                navController.navigate(item.screen.route) {
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Home.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            // Main tabs
            composable(Screen.Home.route) {
                HomeScreen(
                    onNavigateToCheckIn = {
                        navController.navigate(Screen.CheckIn.route)
                    },
                    onNavigateToBodyMap = {
                        navController.navigate(Screen.BodyMap.route)
                    },
                    onNavigateToTrends = {
                        navController.navigate(Screen.Trends.route)
                    },
                    onNavigateToMedication = {
                        navController.navigate(Screen.Medication.route)
                    },
                    onNavigateToFlares = {
                        navController.navigate(Screen.Flares.route)
                    },
                    onNavigateToQuickCapture = {
                        navController.navigate(Screen.QuickCapture.route)
                    }
                )
            }
            composable(Screen.CheckIn.route) {
                CheckInScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    },
                    onComplete = {
                        navController.popBackStack(Screen.Home.route, inclusive = false)
                    }
                )
            }
            composable(Screen.Trends.route) {
                TrendsScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            composable(Screen.Exercise.route) {
                ExerciseScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            composable(Screen.Settings.route) {
                SettingsScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }

            // Secondary screens
            composable(Screen.BodyMap.route) {
                BodyMapScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            composable(Screen.Medication.route) {
                MedicationScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            composable(Screen.Flares.route) {
                FlaresScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            composable(Screen.AI.route) {
                AIScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            composable(Screen.QuickCapture.route) {
                QuickCaptureScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    },
                    onComplete = {
                        navController.popBackStack(Screen.Home.route, inclusive = false)
                    }
                )
            }
            composable(Screen.Meditation.route) {
                MeditationScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            composable(Screen.Onboarding.route) {
                OnboardingScreen(
                    onComplete = {
                        navController.popBackStack()
                        navController.navigate(Screen.Home.route)
                    }
                )
            }
        }
    }
}

@Composable
fun PlaceholderScreen(title: String) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.headlineMedium
        )
    }
}
