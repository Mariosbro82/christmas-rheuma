package com.inflamai.feature.medication.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.inflamai.core.data.database.entity.MedicationCategory
import com.inflamai.core.data.database.entity.MedicationEntity
import com.inflamai.core.data.database.entity.MedicationFrequency
import com.inflamai.feature.medication.viewmodel.MedicationViewModel
import java.time.LocalTime
import java.time.format.DateTimeFormatter

/**
 * Medication Tracking Screen
 *
 * Displays medication list with adherence tracking and quick logging.
 * Follows Material Design 3 and WCAG AA accessibility.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MedicationScreen(
    onNavigateBack: () -> Unit,
    viewModel: MedicationViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val showAddDialog by viewModel.showAddDialog.collectAsStateWithLifecycle()
    val editingMedication by viewModel.editingMedication.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Medications") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Navigate back"
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = { viewModel.showAddMedicationDialog() },
                        modifier = Modifier.semantics {
                            contentDescription = "Add new medication"
                        }
                    ) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = null
                        )
                    }
                }
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { viewModel.showAddMedicationDialog() },
                icon = { Icon(Icons.Default.Add, contentDescription = null) },
                text = { Text("Add Medication") },
                modifier = Modifier.semantics {
                    contentDescription = "Add new medication"
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Adherence Summary Card
            item {
                AdherenceSummaryCard(
                    adherenceRate = uiState.adherenceRate,
                    takenToday = uiState.takenMedicationIds.size,
                    totalToday = uiState.medications.size
                )
            }

            // Today's Medications Header
            item {
                Text(
                    text = "Today's Medications",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }

            // Medication List
            if (uiState.medications.isEmpty()) {
                item {
                    EmptyMedicationState(
                        onAddClick = { viewModel.showAddMedicationDialog() }
                    )
                }
            } else {
                items(
                    items = uiState.medications,
                    key = { it.id }
                ) { medication ->
                    MedicationCard(
                        medication = medication,
                        isTaken = medication.id in uiState.takenMedicationIds,
                        nextDoseTime = viewModel.getNextDoseTime(medication),
                        onToggleTaken = { viewModel.toggleMedicationTaken(medication) },
                        onEdit = { viewModel.showEditMedicationDialog(medication) },
                        onSkip = { viewModel.skipDose(medication, null) },
                        onDelete = { viewModel.deleteMedication(medication) }
                    )
                }
            }

            // Bottom spacing for FAB
            item {
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
    }

    // Add/Edit Dialog
    if (showAddDialog) {
        AddMedicationDialog(
            existingMedication = editingMedication,
            onDismiss = { viewModel.dismissDialog() },
            onSave = { name, dosage, dosageUnit, frequency, category, instructions, times ->
                viewModel.saveMedication(name, dosage, dosageUnit, frequency, category, instructions, times)
            }
        )
    }
}

@Composable
private fun AdherenceSummaryCard(
    adherenceRate: Float,
    takenToday: Int,
    totalToday: Int
) {
    val adherencePercent = (adherenceRate * 100).toInt()
    val adherenceColor = when {
        adherencePercent >= 90 -> Color(0xFF4CAF50)
        adherencePercent >= 70 -> Color(0xFFFF9800)
        else -> Color(0xFFF44336)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "30-Day Adherence",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                )
                Text(
                    text = "$adherencePercent%",
                    style = MaterialTheme.typography.headlineLarge,
                    fontWeight = FontWeight.Bold,
                    color = adherenceColor
                )
            }

            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = "Today",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                )
                Text(
                    text = "$takenToday / $totalToday",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = "doses taken",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MedicationCard(
    medication: MedicationEntity,
    isTaken: Boolean,
    nextDoseTime: LocalTime?,
    onToggleTaken: () -> Unit,
    onEdit: () -> Unit,
    onSkip: () -> Unit,
    onDelete: () -> Unit
) {
    var showMenu by remember { mutableStateOf(false) }

    val cardColor by animateColorAsState(
        targetValue = if (isTaken) {
            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f)
        } else {
            MaterialTheme.colorScheme.surface
        },
        label = "card_color"
    )

    val categoryIcon = when (medication.category) {
        MedicationCategory.NSAID -> Icons.Outlined.LocalFireDepartment
        MedicationCategory.BIOLOGIC -> Icons.Outlined.Science
        MedicationCategory.DMARD -> Icons.Outlined.Medication
        MedicationCategory.CORTICOSTEROID -> Icons.Outlined.MedicalServices
        MedicationCategory.PAIN_RELIEVER -> Icons.Outlined.Healing
        MedicationCategory.MUSCLE_RELAXANT -> Icons.Outlined.Accessibility
        MedicationCategory.SUPPLEMENT -> Icons.Outlined.Spa
        MedicationCategory.OTHER -> Icons.Outlined.MoreHoriz
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "${medication.name}, ${medication.dosage} ${medication.dosageUnit}, " +
                    if (isTaken) "taken" else "not taken yet"
            },
        colors = CardDefaults.cardColors(containerColor = cardColor),
        onClick = onToggleTaken
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Category Icon
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(
                        if (isTaken) Color(0xFF4CAF50).copy(alpha = 0.2f)
                        else MaterialTheme.colorScheme.primaryContainer
                    ),
                contentAlignment = Alignment.Center
            ) {
                if (isTaken) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = null,
                        tint = Color(0xFF4CAF50)
                    )
                } else {
                    Icon(
                        imageVector = categoryIcon,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }
            }

            Spacer(modifier = Modifier.width(12.dp))

            // Medication Info
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = medication.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Text(
                    text = "${medication.dosage} ${medication.dosageUnit}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(top = 4.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Schedule,
                        contentDescription = null,
                        modifier = Modifier.size(14.dp),
                        tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = if (nextDoseTime != null) {
                            "Next: ${nextDoseTime.format(DateTimeFormatter.ofPattern("h:mm a"))}"
                        } else {
                            medication.frequency.displayName
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                    )
                }
            }

            // Menu
            Box {
                IconButton(onClick = { showMenu = true }) {
                    Icon(
                        imageVector = Icons.Default.MoreVert,
                        contentDescription = "More options"
                    )
                }

                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = { showMenu = false }
                ) {
                    DropdownMenuItem(
                        text = { Text("Edit") },
                        onClick = {
                            showMenu = false
                            onEdit()
                        },
                        leadingIcon = {
                            Icon(Icons.Outlined.Edit, contentDescription = null)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Skip dose") },
                        onClick = {
                            showMenu = false
                            onSkip()
                        },
                        leadingIcon = {
                            Icon(Icons.Outlined.SkipNext, contentDescription = null)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Delete") },
                        onClick = {
                            showMenu = false
                            onDelete()
                        },
                        leadingIcon = {
                            Icon(
                                Icons.Outlined.Delete,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error
                            )
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun EmptyMedicationState(
    onAddClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = Icons.Outlined.Medication,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "No medications yet",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )

        Text(
            text = "Track your AS medications and set reminders",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(onClick = onAddClick) {
            Icon(Icons.Default.Add, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Add Medication")
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AddMedicationDialog(
    existingMedication: MedicationEntity?,
    onDismiss: () -> Unit,
    onSave: (
        name: String,
        dosage: Double,
        dosageUnit: String,
        frequency: MedicationFrequency,
        category: MedicationCategory,
        instructions: String?,
        reminderTimes: List<LocalTime>
    ) -> Unit
) {
    var name by remember { mutableStateOf(existingMedication?.name ?: "") }
    var dosageText by remember { mutableStateOf(existingMedication?.dosage?.toString() ?: "") }
    var dosageUnit by remember { mutableStateOf(existingMedication?.dosageUnit ?: "mg") }
    var frequency by remember { mutableStateOf(existingMedication?.frequency ?: MedicationFrequency.ONCE_DAILY) }
    var category by remember { mutableStateOf(existingMedication?.category ?: MedicationCategory.NSAID) }
    var instructions by remember { mutableStateOf(existingMedication?.instructions ?: "") }

    var expandedFrequency by remember { mutableStateOf(false) }
    var expandedCategory by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(if (existingMedication != null) "Edit Medication" else "Add Medication")
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Medication Name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedTextField(
                        value = dosageText,
                        onValueChange = { dosageText = it },
                        label = { Text("Dosage") },
                        singleLine = true,
                        modifier = Modifier.weight(1f)
                    )

                    OutlinedTextField(
                        value = dosageUnit,
                        onValueChange = { dosageUnit = it },
                        label = { Text("Unit") },
                        singleLine = true,
                        modifier = Modifier.weight(0.5f)
                    )
                }

                // Frequency Dropdown
                ExposedDropdownMenuBox(
                    expanded = expandedFrequency,
                    onExpandedChange = { expandedFrequency = it }
                ) {
                    OutlinedTextField(
                        value = frequency.displayName,
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Frequency") },
                        trailingIcon = {
                            ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedFrequency)
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor()
                    )

                    ExposedDropdownMenu(
                        expanded = expandedFrequency,
                        onDismissRequest = { expandedFrequency = false }
                    ) {
                        MedicationFrequency.entries.forEach { freq ->
                            DropdownMenuItem(
                                text = { Text(freq.displayName) },
                                onClick = {
                                    frequency = freq
                                    expandedFrequency = false
                                }
                            )
                        }
                    }
                }

                // Category Dropdown
                ExposedDropdownMenuBox(
                    expanded = expandedCategory,
                    onExpandedChange = { expandedCategory = it }
                ) {
                    OutlinedTextField(
                        value = category.displayName,
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Category") },
                        trailingIcon = {
                            ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedCategory)
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor()
                    )

                    ExposedDropdownMenu(
                        expanded = expandedCategory,
                        onDismissRequest = { expandedCategory = false }
                    ) {
                        MedicationCategory.entries.forEach { cat ->
                            DropdownMenuItem(
                                text = { Text(cat.displayName) },
                                onClick = {
                                    category = cat
                                    expandedCategory = false
                                }
                            )
                        }
                    }
                }

                OutlinedTextField(
                    value = instructions,
                    onValueChange = { instructions = it },
                    label = { Text("Instructions (optional)") },
                    placeholder = { Text("e.g., Take with food") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                    maxLines = 3
                )
            }
        },
        confirmButton = {
            val dosageValue = dosageText.toDoubleOrNull()
            TextButton(
                onClick = {
                    val parsedDosage = dosageText.toDoubleOrNull()
                    if (name.isNotBlank() && parsedDosage != null) {
                        val reminderTimes = when (frequency) {
                            MedicationFrequency.ONCE_DAILY, MedicationFrequency.DAILY -> listOf(LocalTime.of(8, 0))
                            MedicationFrequency.TWICE_DAILY -> listOf(
                                LocalTime.of(8, 0),
                                LocalTime.of(20, 0)
                            )
                            MedicationFrequency.THREE_TIMES_DAILY -> listOf(
                                LocalTime.of(8, 0),
                                LocalTime.of(14, 0),
                                LocalTime.of(20, 0)
                            )
                            MedicationFrequency.FOUR_TIMES_DAILY -> listOf(
                                LocalTime.of(8, 0),
                                LocalTime.of(12, 0),
                                LocalTime.of(16, 0),
                                LocalTime.of(20, 0)
                            )
                            MedicationFrequency.WEEKLY -> listOf(LocalTime.of(8, 0))
                            MedicationFrequency.BIWEEKLY -> listOf(LocalTime.of(8, 0))
                            MedicationFrequency.MONTHLY -> listOf(LocalTime.of(8, 0))
                            MedicationFrequency.AS_NEEDED -> emptyList()
                        }

                        onSave(
                            name,
                            parsedDosage,
                            dosageUnit,
                            frequency,
                            category,
                            instructions.ifBlank { null },
                            reminderTimes
                        )
                    }
                },
                enabled = name.isNotBlank() && dosageValue != null
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

// Extension properties for display names
private val MedicationFrequency.displayName: String
    get() = when (this) {
        MedicationFrequency.ONCE_DAILY -> "Once daily"
        MedicationFrequency.DAILY -> "Daily"
        MedicationFrequency.TWICE_DAILY -> "Twice daily"
        MedicationFrequency.THREE_TIMES_DAILY -> "Three times daily"
        MedicationFrequency.FOUR_TIMES_DAILY -> "Four times daily"
        MedicationFrequency.WEEKLY -> "Weekly"
        MedicationFrequency.BIWEEKLY -> "Every two weeks"
        MedicationFrequency.MONTHLY -> "Monthly"
        MedicationFrequency.AS_NEEDED -> "As needed"
    }

private val MedicationCategory.displayName: String
    get() = when (this) {
        MedicationCategory.NSAID -> "NSAID"
        MedicationCategory.BIOLOGIC -> "Biologic"
        MedicationCategory.DMARD -> "DMARD"
        MedicationCategory.CORTICOSTEROID -> "Corticosteroid"
        MedicationCategory.PAIN_RELIEVER -> "Pain Reliever"
        MedicationCategory.MUSCLE_RELAXANT -> "Muscle Relaxant"
        MedicationCategory.SUPPLEMENT -> "Supplement"
        MedicationCategory.OTHER -> "Other"
    }
