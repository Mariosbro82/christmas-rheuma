package com.inflamai.feature.settings.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.inflamai.feature.settings.viewmodel.SettingsViewModel
import com.inflamai.feature.settings.viewmodel.ThemeMode
import java.time.format.DateTimeFormatter

/**
 * Settings Screen
 *
 * Comprehensive settings with privacy-first architecture.
 * WCAG AA compliant with proper accessibility labels.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit,
    onNavigateToAbout: () -> Unit = {},
    onNavigateToPrivacyPolicy: () -> Unit = {},
    viewModel: SettingsViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    var showThemeDialog by remember { mutableStateOf(false) }
    var showTimePickerDialog by remember { mutableStateOf(false) }
    var showDeleteConfirmDialog by remember { mutableStateOf(false) }
    var showProfileDialog by remember { mutableStateOf(false) }

    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(uiState.exportSuccess) {
        if (uiState.exportSuccess) {
            snackbarHostState.showSnackbar("Data exported successfully")
            viewModel.clearExportSuccess()
        }
    }

    LaunchedEffect(uiState.deleteSuccess) {
        if (uiState.deleteSuccess) {
            snackbarHostState.showSnackbar("All data deleted")
            viewModel.clearDeleteSuccess()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Navigate back"
                        )
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(vertical = 8.dp)
        ) {
            // Profile Section
            item {
                SettingsSection(title = "Profile")
            }

            item {
                SettingsItem(
                    icon = Icons.Outlined.Person,
                    title = uiState.userProfile?.name ?: "Set up profile",
                    subtitle = "Name, diagnosis date, and more",
                    onClick = { showProfileDialog = true }
                )
            }

            // Security Section
            item {
                SettingsSection(title = "Security")
            }

            item {
                SettingsSwitch(
                    icon = Icons.Outlined.Fingerprint,
                    title = "Biometric Lock",
                    subtitle = "Require Face ID or fingerprint",
                    checked = uiState.biometricEnabled,
                    onCheckedChange = { viewModel.setBiometricEnabled(it) }
                )
            }

            // Appearance Section
            item {
                SettingsSection(title = "Appearance")
            }

            item {
                SettingsItem(
                    icon = Icons.Outlined.DarkMode,
                    title = "Theme",
                    subtitle = when (uiState.themeMode) {
                        ThemeMode.SYSTEM -> "System default"
                        ThemeMode.LIGHT -> "Light"
                        ThemeMode.DARK -> "Dark"
                    },
                    onClick = { showThemeDialog = true }
                )
            }

            // Notifications Section
            item {
                SettingsSection(title = "Notifications")
            }

            item {
                SettingsSwitch(
                    icon = Icons.Outlined.Notifications,
                    title = "Daily Check-in Reminder",
                    subtitle = "Remind me to log symptoms",
                    checked = uiState.dailyReminderEnabled,
                    onCheckedChange = { viewModel.setDailyReminderEnabled(it) }
                )
            }

            if (uiState.dailyReminderEnabled) {
                item {
                    SettingsItem(
                        icon = Icons.Outlined.Schedule,
                        title = "Reminder Time",
                        subtitle = uiState.dailyReminderTime.format(
                            DateTimeFormatter.ofPattern("h:mm a")
                        ),
                        onClick = { showTimePickerDialog = true }
                    )
                }
            }

            item {
                SettingsSwitch(
                    icon = Icons.Outlined.Medication,
                    title = "Medication Reminders",
                    subtitle = "Get notified for medication doses",
                    checked = uiState.medicationRemindersEnabled,
                    onCheckedChange = { viewModel.setMedicationRemindersEnabled(it) }
                )
            }

            // Health Data Section
            item {
                SettingsSection(title = "Health Data")
            }

            item {
                SettingsSwitch(
                    icon = Icons.Outlined.MonitorHeart,
                    title = "Health Connect",
                    subtitle = "Sync heart rate, sleep, and activity data",
                    checked = uiState.healthConnectEnabled,
                    onCheckedChange = { viewModel.setHealthConnectEnabled(it) }
                )
            }

            // Accessibility Section
            item {
                SettingsSection(title = "Accessibility")
            }

            item {
                SettingsSwitch(
                    icon = Icons.Outlined.Vibration,
                    title = "Haptic Feedback",
                    subtitle = "Vibration for interactions",
                    checked = uiState.hapticFeedbackEnabled,
                    onCheckedChange = { viewModel.setHapticFeedbackEnabled(it) }
                )
            }

            item {
                SettingsSwitch(
                    icon = Icons.Outlined.Animation,
                    title = "Reduce Motion",
                    subtitle = "Minimize animations",
                    checked = uiState.reduceMotion,
                    onCheckedChange = { viewModel.setReduceMotion(it) }
                )
            }

            item {
                SettingsSwitch(
                    icon = Icons.Outlined.TextFields,
                    title = "Larger Text",
                    subtitle = "Increase text size throughout app",
                    checked = uiState.largeText,
                    onCheckedChange = { viewModel.setLargeText(it) }
                )
            }

            // Units Section
            item {
                SettingsSection(title = "Units")
            }

            item {
                SettingsSwitch(
                    icon = Icons.Outlined.Straighten,
                    title = "Metric Units",
                    subtitle = if (uiState.useMetricUnits) "kg, cm, °C" else "lb, in, °F",
                    checked = uiState.useMetricUnits,
                    onCheckedChange = { viewModel.setUseMetricUnits(it) }
                )
            }

            // Data Section
            item {
                SettingsSection(title = "Your Data")
            }

            item {
                SettingsItem(
                    icon = Icons.Outlined.Download,
                    title = "Export Data",
                    subtitle = "Download your health records as PDF",
                    onClick = { viewModel.exportData() },
                    isLoading = uiState.isExporting
                )
            }

            item {
                SettingsItem(
                    icon = Icons.Outlined.DeleteForever,
                    title = "Delete All Data",
                    subtitle = "Permanently remove all your data",
                    onClick = { showDeleteConfirmDialog = true },
                    titleColor = MaterialTheme.colorScheme.error
                )
            }

            // About Section
            item {
                SettingsSection(title = "About")
            }

            item {
                SettingsItem(
                    icon = Icons.Outlined.Info,
                    title = "About InflamAI",
                    subtitle = "Version 1.0.0",
                    onClick = onNavigateToAbout
                )
            }

            item {
                SettingsItem(
                    icon = Icons.Outlined.PrivacyTip,
                    title = "Privacy Policy",
                    subtitle = "How we protect your data",
                    onClick = onNavigateToPrivacyPolicy
                )
            }

            // Privacy Notice
            item {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f)
                    )
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.Shield,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Column {
                            Text(
                                text = "Your Privacy is Protected",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "All data is stored on-device only. No analytics, no cloud sync, no third-party SDKs. You own your health data.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                            )
                        }
                    }
                }
            }
        }
    }

    // Theme Selection Dialog
    if (showThemeDialog) {
        AlertDialog(
            onDismissRequest = { showThemeDialog = false },
            title = { Text("Choose Theme") },
            text = {
                Column {
                    ThemeMode.entries.forEach { mode ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = uiState.themeMode == mode,
                                onClick = {
                                    viewModel.setThemeMode(mode)
                                    showThemeDialog = false
                                }
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = when (mode) {
                                    ThemeMode.SYSTEM -> "System default"
                                    ThemeMode.LIGHT -> "Light"
                                    ThemeMode.DARK -> "Dark"
                                }
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showThemeDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    // Delete Confirmation Dialog
    if (showDeleteConfirmDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirmDialog = false },
            icon = {
                Icon(
                    imageVector = Icons.Outlined.Warning,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.error
                )
            },
            title = { Text("Delete All Data?") },
            text = {
                Text(
                    "This will permanently delete all your health records, symptoms, medications, and settings. This action cannot be undone."
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        viewModel.deleteAllData()
                        showDeleteConfirmDialog = false
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Text("Delete Everything")
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirmDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    // Profile Edit Dialog
    if (showProfileDialog) {
        ProfileEditDialog(
            currentProfile = uiState.userProfile,
            onDismiss = { showProfileDialog = false },
            onSave = { name, dob, diagnosisDate ->
                viewModel.updateUserProfile(name, dob, diagnosisDate)
                showProfileDialog = false
            }
        )
    }
}

@Composable
private fun SettingsSection(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.primary,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier.padding(start = 16.dp, top = 24.dp, bottom = 8.dp)
    )
}

@Composable
private fun SettingsItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    onClick: () -> Unit,
    titleColor: Color = MaterialTheme.colorScheme.onSurface,
    isLoading: Boolean = false
) {
    Surface(
        onClick = onClick,
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "$title. $subtitle"
            }
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyLarge,
                    color = titleColor
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    strokeWidth = 2.dp
                )
            } else {
                Icon(
                    imageVector = Icons.Default.ChevronRight,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun SettingsSwitch(
    icon: ImageVector,
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Surface(
        onClick = { onCheckedChange(!checked) },
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "$title. $subtitle. ${if (checked) "On" else "Off"}"
            }
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyLarge
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Switch(
                checked = checked,
                onCheckedChange = onCheckedChange
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ProfileEditDialog(
    currentProfile: com.inflamai.core.data.database.entity.UserProfileEntity?,
    onDismiss: () -> Unit,
    onSave: (String?, Long?, Long?) -> Unit
) {
    var name by remember { mutableStateOf(currentProfile?.name ?: "") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit Profile") },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                // Note: Full implementation would include date pickers for:
                // - Date of birth
                // - Diagnosis date

                Text(
                    text = "Additional profile fields like diagnosis date can be added with date pickers.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    onSave(
                        name.ifBlank { null },
                        currentProfile?.dateOfBirth,
                        currentProfile?.diagnosisDate
                    )
                }
            ) {
                Text("Save")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}
