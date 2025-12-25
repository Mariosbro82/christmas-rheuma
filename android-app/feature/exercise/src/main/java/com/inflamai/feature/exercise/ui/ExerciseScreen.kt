package com.inflamai.feature.exercise.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
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
import com.inflamai.feature.exercise.model.Difficulty
import com.inflamai.feature.exercise.model.Exercise
import com.inflamai.feature.exercise.model.ExerciseCategory
import com.inflamai.feature.exercise.viewmodel.ExerciseViewModel

/**
 * Exercise Library Screen
 *
 * Displays 52 AS-specific exercises with filtering and workout tracking.
 * Follows Material Design 3 and WCAG AA accessibility.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExerciseScreen(
    onNavigateBack: () -> Unit,
    viewModel: ExerciseViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val selectedExercise by viewModel.selectedExercise.collectAsStateWithLifecycle()
    val activeWorkout by viewModel.activeWorkout.collectAsStateWithLifecycle()

    var showQuickRoutineDialog by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Exercise Library") },
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
                        onClick = { showQuickRoutineDialog = true },
                        modifier = Modifier.semantics {
                            contentDescription = "Generate quick routine"
                        }
                    ) {
                        Icon(
                            imageVector = Icons.Default.AutoAwesome,
                            contentDescription = null
                        )
                    }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Weekly Stats Card
            item {
                WeeklyStatsCard(
                    weeklyMinutes = uiState.weeklyMinutes,
                    weeklySessionCount = uiState.weeklySessionCount
                )
            }

            // Search Bar
            item {
                OutlinedTextField(
                    value = uiState.searchQuery,
                    onValueChange = { viewModel.searchExercises(it) },
                    placeholder = { Text("Search exercises...") },
                    leadingIcon = {
                        Icon(Icons.Default.Search, contentDescription = null)
                    },
                    trailingIcon = {
                        if (uiState.searchQuery.isNotEmpty()) {
                            IconButton(onClick = { viewModel.searchExercises("") }) {
                                Icon(Icons.Default.Clear, contentDescription = "Clear search")
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    shape = RoundedCornerShape(12.dp)
                )
            }

            // Category Filter
            item {
                Text(
                    text = "Categories",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(8.dp))

                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    item {
                        FilterChip(
                            selected = uiState.selectedCategory == null,
                            onClick = { viewModel.filterByCategory(null) },
                            label = { Text("All") }
                        )
                    }

                    items(ExerciseCategory.entries.toList()) { category ->
                        FilterChip(
                            selected = uiState.selectedCategory == category,
                            onClick = { viewModel.filterByCategory(category) },
                            label = { Text(category.displayName) },
                            leadingIcon = {
                                if (uiState.selectedCategory == category) {
                                    Icon(
                                        imageVector = Icons.Default.Check,
                                        contentDescription = null,
                                        modifier = Modifier.size(16.dp)
                                    )
                                }
                            }
                        )
                    }
                }
            }

            // Difficulty Filter
            item {
                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(Difficulty.entries.toList()) { difficulty ->
                        FilterChip(
                            selected = uiState.selectedDifficulty == difficulty,
                            onClick = {
                                viewModel.filterByDifficulty(
                                    if (uiState.selectedDifficulty == difficulty) null else difficulty
                                )
                            },
                            label = { Text(difficulty.displayName) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = difficulty.color.copy(alpha = 0.2f),
                                selectedLabelColor = difficulty.color
                            )
                        )
                    }
                }
            }

            // Generated Routine
            uiState.generatedRoutine?.let { routine ->
                item {
                    RoutineCard(
                        exercises = routine,
                        onStartWorkout = { viewModel.startWorkout(routine) },
                        onDismiss = { viewModel.filterByCategory(uiState.selectedCategory) }
                    )
                }
            }

            // Exercise List
            item {
                Text(
                    text = "${uiState.filteredExercises.size} Exercises",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
            }

            items(
                items = uiState.filteredExercises,
                key = { it.id }
            ) { exercise ->
                ExerciseCard(
                    exercise = exercise,
                    onClick = { viewModel.selectExercise(exercise) }
                )
            }
        }
    }

    // Exercise Detail Sheet
    selectedExercise?.let { exercise ->
        ExerciseDetailSheet(
            exercise = exercise,
            onDismiss = { viewModel.clearSelectedExercise() },
            onStartSingle = {
                viewModel.startWorkout(listOf(exercise))
                viewModel.clearSelectedExercise()
            }
        )
    }

    // Quick Routine Dialog
    if (showQuickRoutineDialog) {
        QuickRoutineDialog(
            onDismiss = { showQuickRoutineDialog = false },
            onGenerate = { duration ->
                viewModel.generateQuickRoutine(duration)
                showQuickRoutineDialog = false
            }
        )
    }

    // Active Workout Overlay
    activeWorkout?.let { workout ->
        ActiveWorkoutScreen(
            workout = workout,
            onCompleteExercise = { viewModel.completeExercise(it) },
            onFinish = { painBefore, painAfter, notes ->
                viewModel.finishWorkout(painBefore, painAfter, notes)
            },
            onCancel = { viewModel.cancelWorkout() }
        )
    }
}

@Composable
private fun WeeklyStatsCard(
    weeklyMinutes: Int,
    weeklySessionCount: Int
) {
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
            horizontalArrangement = Arrangement.SpaceAround
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Outlined.Timer,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${weeklyMinutes}",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "minutes",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                )
            }

            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Outlined.FitnessCenter,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "$weeklySessionCount",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "sessions",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                )
            }

            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Outlined.LocalFireDepartment,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "0", // Streak - would need additional tracking
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "day streak",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ExerciseCard(
    exercise: Exercise,
    onClick: () -> Unit
) {
    val categoryIcon = when (exercise.category) {
        ExerciseCategory.STRETCHING -> Icons.Outlined.SelfImprovement
        ExerciseCategory.MOBILITY -> Icons.Outlined.DirectionsRun
        ExerciseCategory.STRENGTHENING -> Icons.Outlined.FitnessCenter
        ExerciseCategory.BREATHING -> Icons.Outlined.Air
        ExerciseCategory.POSTURE -> Icons.Outlined.Accessibility
        ExerciseCategory.AQUATIC -> Icons.Outlined.Pool
    }

    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primaryContainer),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = categoryIcon,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = exercise.name,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f, fill = false)
                    )

                    Spacer(modifier = Modifier.width(8.dp))

                    DifficultyBadge(difficulty = exercise.difficulty)
                }

                Text(
                    text = exercise.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )

                Row(
                    modifier = Modifier.padding(top = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Timer,
                        contentDescription = null,
                        modifier = Modifier.size(14.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "${exercise.durationMinutes} min",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Spacer(modifier = Modifier.width(12.dp))

                    Text(
                        text = exercise.targetAreas.take(2).joinToString(", "),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Icon(
                imageVector = Icons.Default.ChevronRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun DifficultyBadge(difficulty: Difficulty) {
    Surface(
        shape = RoundedCornerShape(4.dp),
        color = difficulty.color.copy(alpha = 0.2f)
    ) {
        Text(
            text = difficulty.displayName,
            style = MaterialTheme.typography.labelSmall,
            color = difficulty.color,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ExerciseDetailSheet(
    exercise: Exercise,
    onDismiss: () -> Unit,
    onStartSingle: () -> Unit
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            contentPadding = PaddingValues(bottom = 32.dp)
        ) {
            item {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = exercise.name,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.weight(1f)
                    )
                    DifficultyBadge(difficulty = exercise.difficulty)
                }

                Spacer(modifier = Modifier.height(4.dp))

                Row {
                    AssistChip(
                        onClick = { },
                        label = { Text(exercise.category.displayName) }
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    AssistChip(
                        onClick = { },
                        label = { Text("${exercise.durationMinutes} min") },
                        leadingIcon = {
                            Icon(
                                Icons.Outlined.Timer,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = exercise.description,
                    style = MaterialTheme.typography.bodyMedium
                )

                Spacer(modifier = Modifier.height(16.dp))
            }

            // Target Areas
            item {
                Text(
                    text = "Target Areas",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(8.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(exercise.targetAreas) { area ->
                        SuggestionChip(
                            onClick = { },
                            label = { Text(area) }
                        )
                    }
                }
                Spacer(modifier = Modifier.height(16.dp))
            }

            // Instructions
            item {
                Text(
                    text = "Instructions",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(8.dp))
            }

            items(exercise.instructions.withIndex().toList()) { (index, instruction) ->
                Row(modifier = Modifier.padding(vertical = 4.dp)) {
                    Text(
                        text = "${index + 1}.",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.width(24.dp)
                    )
                    Text(
                        text = instruction,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // Benefits
            item {
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Benefits for AS",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(8.dp))
            }

            items(exercise.benefits) { benefit ->
                Row(modifier = Modifier.padding(vertical = 2.dp)) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = null,
                        tint = Color(0xFF4CAF50),
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = benefit,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // Precautions
            if (exercise.precautions.isNotEmpty()) {
                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Precautions",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }

                items(exercise.precautions) { precaution ->
                    Row(modifier = Modifier.padding(vertical = 2.dp)) {
                        Icon(
                            imageVector = Icons.Outlined.Warning,
                            contentDescription = null,
                            tint = Color(0xFFFF9800),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = precaution,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }

            // Start Button
            item {
                Spacer(modifier = Modifier.height(24.dp))
                Button(
                    onClick = onStartSingle,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Start Exercise")
                }
            }
        }
    }
}

@Composable
private fun RoutineCard(
    exercises: List<Exercise>,
    onStartWorkout: () -> Unit,
    onDismiss: () -> Unit
) {
    val totalDuration = exercises.sumOf { it.durationMinutes }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Your Quick Routine",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Default.Close, contentDescription = "Dismiss")
                }
            }

            Text(
                text = "${exercises.size} exercises • $totalDuration min",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
            )

            Spacer(modifier = Modifier.height(12.dp))

            exercises.forEach { exercise ->
                Row(
                    modifier = Modifier.padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.FiberManualRecord,
                        contentDescription = null,
                        modifier = Modifier.size(8.dp),
                        tint = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = exercise.name,
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Spacer(modifier = Modifier.weight(1f))
                    Text(
                        text = "${exercise.durationMinutes}m",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = onStartWorkout,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Start Routine")
            }
        }
    }
}

@Composable
private fun QuickRoutineDialog(
    onDismiss: () -> Unit,
    onGenerate: (Int) -> Unit
) {
    var selectedDuration by remember { mutableStateOf(15) }
    val durations = listOf(10, 15, 20, 30)

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.AutoAwesome, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Quick Routine")
            }
        },
        text = {
            Column {
                Text("Select workout duration:")
                Spacer(modifier = Modifier.height(16.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    durations.forEach { duration ->
                        FilterChip(
                            selected = selectedDuration == duration,
                            onClick = { selectedDuration = duration },
                            label = { Text("$duration min") }
                        )
                    }
                }
            }
        },
        confirmButton = {
            Button(onClick = { onGenerate(selectedDuration) }) {
                Text("Generate")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ActiveWorkoutScreen(
    workout: com.inflamai.feature.exercise.viewmodel.ActiveWorkout,
    onCompleteExercise: (String) -> Unit,
    onFinish: (Int?, Int?, String?) -> Unit,
    onCancel: () -> Unit
) {
    var showFinishDialog by remember { mutableStateOf(false) }

    val currentExercise = workout.exercises.getOrNull(workout.currentIndex)
    val progress = workout.completedExercises.size.toFloat() / workout.exercises.size.toFloat()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Workout in Progress") },
                navigationIcon = {
                    IconButton(onClick = onCancel) {
                        Icon(Icons.Default.Close, contentDescription = "Cancel workout")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            // Progress
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp))
            )

            Text(
                text = "${workout.completedExercises.size} of ${workout.exercises.size} completed",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 8.dp)
            )

            Spacer(modifier = Modifier.height(24.dp))

            currentExercise?.let { exercise ->
                Text(
                    text = exercise.name,
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(8.dp))

                Row {
                    AssistChip(
                        onClick = { },
                        label = { Text("${exercise.durationMinutes} min") }
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "Instructions",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )

                Spacer(modifier = Modifier.height(8.dp))

                exercise.instructions.forEachIndexed { index, instruction ->
                    Row(modifier = Modifier.padding(vertical = 4.dp)) {
                        Text(
                            text = "${index + 1}.",
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = instruction)
                    }
                }

                Spacer(modifier = Modifier.weight(1f))

                val isCompleted = exercise.id in workout.completedExercises
                val isLast = workout.currentIndex == workout.exercises.size - 1

                Button(
                    onClick = {
                        if (!isCompleted) {
                            onCompleteExercise(exercise.id)
                        }
                        if (isLast && isCompleted) {
                            showFinishDialog = true
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isCompleted || isLast
                ) {
                    Text(
                        when {
                            isCompleted && isLast -> "Finish Workout"
                            isCompleted -> "Completed"
                            isLast -> "Complete & Finish"
                            else -> "Complete Exercise"
                        }
                    )
                }
            }
        }
    }

    if (showFinishDialog) {
        FinishWorkoutDialog(
            onDismiss = { showFinishDialog = false },
            onFinish = { painBefore, painAfter, notes ->
                onFinish(painBefore, painAfter, notes)
            }
        )
    }
}

@Composable
private fun FinishWorkoutDialog(
    onDismiss: () -> Unit,
    onFinish: (Int?, Int?, String?) -> Unit
) {
    var painBefore by remember { mutableStateOf(5f) }
    var painAfter by remember { mutableStateOf(3f) }
    var notes by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Workout Complete!") },
        text = {
            Column {
                Text("Pain level before: ${painBefore.toInt()}/10")
                Slider(
                    value = painBefore,
                    onValueChange = { painBefore = it },
                    valueRange = 0f..10f,
                    steps = 9
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text("Pain level after: ${painAfter.toInt()}/10")
                Slider(
                    value = painAfter,
                    onValueChange = { painAfter = it },
                    valueRange = 0f..10f,
                    steps = 9
                )

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = notes,
                    onValueChange = { notes = it },
                    label = { Text("Notes (optional)") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    onFinish(
                        painBefore.toInt(),
                        painAfter.toInt(),
                        notes.ifBlank { null }
                    )
                }
            ) {
                Text("Save")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Skip")
            }
        }
    )
}

// Extension properties
private val ExerciseCategory.displayName: String
    get() = when (this) {
        ExerciseCategory.STRETCHING -> "Stretching"
        ExerciseCategory.MOBILITY -> "Mobility"
        ExerciseCategory.STRENGTHENING -> "Strength"
        ExerciseCategory.BREATHING -> "Breathing"
        ExerciseCategory.POSTURE -> "Posture"
        ExerciseCategory.AQUATIC -> "Aquatic"
    }

private val Difficulty.displayName: String
    get() = when (this) {
        Difficulty.BEGINNER -> "Beginner"
        Difficulty.INTERMEDIATE -> "Intermediate"
        Difficulty.ADVANCED -> "Advanced"
    }

private val Difficulty.color: Color
    get() = when (this) {
        Difficulty.BEGINNER -> Color(0xFF4CAF50)
        Difficulty.INTERMEDIATE -> Color(0xFFFF9800)
        Difficulty.ADVANCED -> Color(0xFFF44336)
    }
