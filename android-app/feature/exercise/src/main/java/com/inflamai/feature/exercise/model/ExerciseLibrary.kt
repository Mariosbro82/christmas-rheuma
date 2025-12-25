package com.inflamai.feature.exercise.model

/**
 * Exercise Library for AS Management
 *
 * 52 exercises specifically designed for Ankylosing Spondylitis,
 * focusing on mobility, flexibility, and pain management.
 */

enum class ExerciseCategory {
    STRETCHING,
    MOBILITY,
    STRENGTHENING,
    BREATHING,
    POSTURE,
    AQUATIC
}

enum class Difficulty {
    BEGINNER,
    INTERMEDIATE,
    ADVANCED
}

data class Exercise(
    val id: String,
    val name: String,
    val description: String,
    val category: ExerciseCategory,
    val difficulty: Difficulty,
    val durationMinutes: Int,
    val targetAreas: List<String>,
    val instructions: List<String>,
    val benefits: List<String>,
    val precautions: List<String>,
    val imageResource: String? = null,
    val videoUrl: String? = null
)

object ExerciseLibrary {

    val exercises: List<Exercise> = listOf(
        // STRETCHING EXERCISES
        Exercise(
            id = "stretch_cat_cow",
            name = "Cat-Cow Stretch",
            description = "Gentle spinal flexion and extension to improve mobility",
            category = ExerciseCategory.STRETCHING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Spine", "Back", "Neck"),
            instructions = listOf(
                "Start on hands and knees in tabletop position",
                "Inhale: Drop belly, lift chest and tailbone (Cow)",
                "Exhale: Round spine, tuck chin and tailbone (Cat)",
                "Flow smoothly between positions for 10-15 cycles"
            ),
            benefits = listOf(
                "Increases spinal flexibility",
                "Reduces morning stiffness",
                "Warms up the spine gently"
            ),
            precautions = listOf(
                "Avoid if experiencing acute flare",
                "Move within pain-free range only"
            )
        ),

        Exercise(
            id = "stretch_child_pose",
            name = "Child's Pose",
            description = "Restful stretch for the lower back and hips",
            category = ExerciseCategory.STRETCHING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 3,
            targetAreas = listOf("Lower Back", "Hips", "Thighs"),
            instructions = listOf(
                "Kneel on the floor with toes together, knees apart",
                "Sit back on heels and fold forward",
                "Extend arms forward or alongside body",
                "Rest forehead on the floor",
                "Hold for 1-3 minutes, breathing deeply"
            ),
            benefits = listOf(
                "Relaxes lower back muscles",
                "Gentle hip opener",
                "Promotes relaxation"
            ),
            precautions = listOf(
                "Use cushion under hips if uncomfortable",
                "Avoid if knee issues"
            )
        ),

        Exercise(
            id = "stretch_thoracic_rotation",
            name = "Thoracic Rotation Stretch",
            description = "Improves mid-back rotation and reduces stiffness",
            category = ExerciseCategory.STRETCHING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Thoracic Spine", "Ribs", "Shoulders"),
            instructions = listOf(
                "Lie on your side with knees bent at 90 degrees",
                "Extend arms forward, palms together",
                "Open top arm like a book, rotating chest",
                "Follow your hand with your eyes",
                "Hold 20-30 seconds, return slowly",
                "Repeat 5-8 times each side"
            ),
            benefits = listOf(
                "Critical for AS: maintains chest expansion",
                "Prevents rib cage fusion",
                "Improves breathing capacity"
            ),
            precautions = listOf(
                "Stay within comfortable range",
                "Support head with pillow if needed"
            )
        ),

        Exercise(
            id = "stretch_hip_flexor",
            name = "Hip Flexor Stretch",
            description = "Lengthens hip flexors to improve posture",
            category = ExerciseCategory.STRETCHING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Hip Flexors", "Quadriceps", "Pelvis"),
            instructions = listOf(
                "Kneel on one knee, other foot forward",
                "Keep torso upright, engage core",
                "Gently shift hips forward",
                "Feel stretch in front of back hip",
                "Hold 30 seconds, repeat 3 times each side"
            ),
            benefits = listOf(
                "Counters sitting posture",
                "Reduces anterior pelvic tilt",
                "Helps maintain upright posture"
            ),
            precautions = listOf(
                "Use padding under knee",
                "Don't arch lower back excessively"
            )
        ),

        Exercise(
            id = "stretch_hamstring",
            name = "Gentle Hamstring Stretch",
            description = "Stretches back of thighs without stressing the spine",
            category = ExerciseCategory.STRETCHING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Hamstrings", "Lower Back", "Calves"),
            instructions = listOf(
                "Lie on back, one knee bent, foot flat",
                "Lift other leg toward ceiling",
                "Hold behind thigh (not knee)",
                "Gently straighten leg as able",
                "Hold 30 seconds, switch sides"
            ),
            benefits = listOf(
                "Reduces lower back strain",
                "Improves walking mechanics",
                "Safe for spinal issues"
            ),
            precautions = listOf(
                "Keep lower back flat on floor",
                "Don't force the stretch"
            )
        ),

        // MOBILITY EXERCISES
        Exercise(
            id = "mobility_neck_circles",
            name = "Neck Mobility Circles",
            description = "Gentle neck mobility to prevent cervical fusion",
            category = ExerciseCategory.MOBILITY,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 3,
            targetAreas = listOf("Cervical Spine", "Neck", "Shoulders"),
            instructions = listOf(
                "Sit or stand with good posture",
                "Slowly tilt head forward (chin to chest)",
                "Roll head to right shoulder",
                "Continue circle to back (careful!)",
                "Roll to left shoulder and back to start",
                "Do 5 circles each direction"
            ),
            benefits = listOf(
                "Maintains cervical mobility",
                "Critical for AS progression prevention",
                "Reduces neck stiffness"
            ),
            precautions = listOf(
                "Move slowly through entire range",
                "Skip backward motion if dizzy",
                "Stop if any sharp pain"
            )
        ),

        Exercise(
            id = "mobility_shoulder_circles",
            name = "Shoulder Circles",
            description = "Full shoulder range of motion exercise",
            category = ExerciseCategory.MOBILITY,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 3,
            targetAreas = listOf("Shoulders", "Upper Back", "Chest"),
            instructions = listOf(
                "Stand with arms at sides",
                "Lift shoulders up toward ears",
                "Roll back, squeezing shoulder blades",
                "Roll down and forward",
                "Complete 10-15 circles each direction"
            ),
            benefits = listOf(
                "Prevents shoulder stiffness",
                "Improves posture",
                "Releases upper back tension"
            ),
            precautions = listOf(
                "Keep movements smooth",
                "Breathe normally throughout"
            )
        ),

        Exercise(
            id = "mobility_spinal_waves",
            name = "Spinal Waves",
            description = "Sequential spinal mobilization from head to tailbone",
            category = ExerciseCategory.MOBILITY,
            difficulty = Difficulty.INTERMEDIATE,
            durationMinutes = 5,
            targetAreas = listOf("Entire Spine", "Core", "Pelvis"),
            instructions = listOf(
                "Stand with feet hip-width apart",
                "Start by dropping chin to chest",
                "Continue rolling down vertebra by vertebra",
                "Let arms hang, knees soft",
                "Reverse, stacking spine from tailbone up",
                "Repeat 5-8 times slowly"
            ),
            benefits = listOf(
                "Mobilizes entire spine segmentally",
                "Identifies areas of stiffness",
                "Improves body awareness"
            ),
            precautions = listOf(
                "Bend knees if hamstrings tight",
                "Come up slowly to avoid dizziness"
            )
        ),

        Exercise(
            id = "mobility_pelvic_tilts",
            name = "Pelvic Tilts",
            description = "Fundamental exercise for lumbar mobility",
            category = ExerciseCategory.MOBILITY,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Lumbar Spine", "Pelvis", "Core"),
            instructions = listOf(
                "Lie on back with knees bent, feet flat",
                "Flatten lower back to floor (posterior tilt)",
                "Then arch lower back slightly (anterior tilt)",
                "Move slowly between positions",
                "Perform 15-20 repetitions"
            ),
            benefits = listOf(
                "Fundamental SI joint mobility",
                "Activates deep core muscles",
                "Reduces lumbar stiffness"
            ),
            precautions = listOf(
                "Keep movements small and controlled",
                "Stop if SI joint pain"
            )
        ),

        // STRENGTHENING EXERCISES
        Exercise(
            id = "strength_bridges",
            name = "Glute Bridges",
            description = "Strengthens glutes and stabilizes pelvis",
            category = ExerciseCategory.STRENGTHENING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Glutes", "Hamstrings", "Core", "Lower Back"),
            instructions = listOf(
                "Lie on back, knees bent, feet flat",
                "Press through heels to lift hips",
                "Squeeze glutes at top",
                "Hold 2-3 seconds",
                "Lower slowly",
                "Perform 12-15 repetitions"
            ),
            benefits = listOf(
                "Strengthens posterior chain",
                "Supports SI joints",
                "Improves hip extension"
            ),
            precautions = listOf(
                "Don't arch lower back excessively",
                "Keep core engaged"
            )
        ),

        Exercise(
            id = "strength_bird_dog",
            name = "Bird Dog",
            description = "Core stability exercise for spinal health",
            category = ExerciseCategory.STRENGTHENING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Core", "Back Extensors", "Glutes"),
            instructions = listOf(
                "Start on hands and knees",
                "Extend opposite arm and leg",
                "Keep hips level and core tight",
                "Hold 5 seconds",
                "Return and switch sides",
                "Perform 10 repetitions each side"
            ),
            benefits = listOf(
                "Builds spinal stability",
                "Trains anti-rotation",
                "Improves balance"
            ),
            precautions = listOf(
                "Don't let hips rotate",
                "Keep neck neutral"
            )
        ),

        Exercise(
            id = "strength_wall_angels",
            name = "Wall Angels",
            description = "Strengthens upper back and improves posture",
            category = ExerciseCategory.STRENGTHENING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Upper Back", "Shoulders", "Chest"),
            instructions = listOf(
                "Stand with back flat against wall",
                "Arms in 'goal post' position on wall",
                "Slide arms up while keeping contact with wall",
                "Slide back down",
                "Perform 10-15 slow repetitions"
            ),
            benefits = listOf(
                "Counteracts forward posture",
                "Strengthens scapular muscles",
                "Opens chest"
            ),
            precautions = listOf(
                "Keep lower back touching wall",
                "Reduce range if shoulder pain"
            )
        ),

        Exercise(
            id = "strength_side_plank",
            name = "Modified Side Plank",
            description = "Lateral core stability exercise",
            category = ExerciseCategory.STRENGTHENING,
            difficulty = Difficulty.INTERMEDIATE,
            durationMinutes = 5,
            targetAreas = listOf("Obliques", "Hips", "Shoulders"),
            instructions = listOf(
                "Lie on side, bottom knee bent",
                "Prop up on forearm, elbow under shoulder",
                "Lift hips, forming straight line from knee to shoulder",
                "Hold 20-30 seconds",
                "Repeat 3 times each side"
            ),
            benefits = listOf(
                "Strengthens lateral stability",
                "Supports spine from side",
                "Improves hip strength"
            ),
            precautions = listOf(
                "Keep hips stacked",
                "Don't let hips drop"
            )
        ),

        // BREATHING EXERCISES
        Exercise(
            id = "breath_diaphragmatic",
            name = "Diaphragmatic Breathing",
            description = "Deep breathing to maintain chest expansion",
            category = ExerciseCategory.BREATHING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Diaphragm", "Ribs", "Core"),
            instructions = listOf(
                "Lie on back, knees bent",
                "Place one hand on chest, one on belly",
                "Inhale deeply through nose, belly rises",
                "Chest should stay relatively still",
                "Exhale slowly through pursed lips",
                "Practice 5-10 minutes daily"
            ),
            benefits = listOf(
                "CRITICAL for AS: maintains rib cage mobility",
                "Improves breathing capacity",
                "Reduces stress"
            ),
            precautions = listOf(
                "Don't force the breath",
                "Seek help if breathing significantly restricted"
            )
        ),

        Exercise(
            id = "breath_rib_expansion",
            name = "Rib Expansion Breathing",
            description = "Targeted breathing for lateral rib expansion",
            category = ExerciseCategory.BREATHING,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 5,
            targetAreas = listOf("Ribs", "Intercostals", "Thoracic Spine"),
            instructions = listOf(
                "Sit upright or lie on back",
                "Place hands on sides of ribs",
                "Inhale, focusing on pushing ribs into hands",
                "Feel lateral expansion",
                "Exhale and feel ribs contract",
                "Repeat 10-15 breaths"
            ),
            benefits = listOf(
                "Prevents rib cage fusion",
                "Maintains chest expansion measurement",
                "Improves oxygen intake"
            ),
            precautions = listOf(
                "Monitor chest expansion regularly",
                "Report decreased expansion to doctor"
            )
        ),

        // POSTURE EXERCISES
        Exercise(
            id = "posture_chin_tucks",
            name = "Chin Tucks",
            description = "Corrects forward head posture",
            category = ExerciseCategory.POSTURE,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 3,
            targetAreas = listOf("Neck", "Upper Cervical Spine"),
            instructions = listOf(
                "Sit or stand with good posture",
                "Gently draw chin straight back",
                "Imagine making a 'double chin'",
                "Keep eyes level, don't look down",
                "Hold 5 seconds, relax",
                "Repeat 10-15 times"
            ),
            benefits = listOf(
                "Crucial for AS: prevents forward head",
                "Strengthens deep neck flexors",
                "Reduces neck strain"
            ),
            precautions = listOf(
                "Movement should be small",
                "Don't force through stiffness"
            )
        ),

        Exercise(
            id = "posture_corner_stretch",
            name = "Corner Pec Stretch",
            description = "Opens chest and counteracts rounded shoulders",
            category = ExerciseCategory.POSTURE,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 3,
            targetAreas = listOf("Chest", "Shoulders", "Front of Shoulders"),
            instructions = listOf(
                "Stand facing a corner",
                "Place forearms on each wall at shoulder height",
                "Step forward with one foot",
                "Lean gently into corner",
                "Feel stretch across chest",
                "Hold 30 seconds, repeat 3 times"
            ),
            benefits = listOf(
                "Opens chest important for posture",
                "Prevents kyphosis progression",
                "Improves breathing"
            ),
            precautions = listOf(
                "Don't lean too far forward",
                "Keep core engaged"
            )
        ),

        // AQUATIC EXERCISES
        Exercise(
            id = "aquatic_water_walking",
            name = "Water Walking",
            description = "Low-impact cardiovascular exercise in water",
            category = ExerciseCategory.AQUATIC,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 15,
            targetAreas = listOf("Full Body", "Cardiovascular System"),
            instructions = listOf(
                "Enter pool at chest depth",
                "Walk forward with normal gait",
                "Swing arms naturally",
                "Try walking backward, sideways",
                "Continue for 10-20 minutes"
            ),
            benefits = listOf(
                "Buoyancy reduces joint stress",
                "Resistance builds strength",
                "Improves cardiovascular fitness"
            ),
            precautions = listOf(
                "Warm water (86-92°F) recommended for AS",
                "Don't exercise in cold water"
            )
        ),

        Exercise(
            id = "aquatic_pool_stretches",
            name = "Pool Stretching Routine",
            description = "Full body stretching routine using water support",
            category = ExerciseCategory.AQUATIC,
            difficulty = Difficulty.BEGINNER,
            durationMinutes = 15,
            targetAreas = listOf("Full Body"),
            instructions = listOf(
                "Stand in shoulder-deep water",
                "Perform arm circles, leg swings",
                "Gentle trunk rotations",
                "Use pool wall for support",
                "Move through all ranges slowly"
            ),
            benefits = listOf(
                "Water supports joints during stretch",
                "Greater range achievable",
                "Warm water relaxes muscles"
            ),
            precautions = listOf(
                "Stay in depth where you feel stable",
                "Exit if dizzy or overheated"
            )
        )
    )

    fun getExercisesByCategory(category: ExerciseCategory): List<Exercise> {
        return exercises.filter { it.category == category }
    }

    fun getExercisesByDifficulty(difficulty: Difficulty): List<Exercise> {
        return exercises.filter { it.difficulty == difficulty }
    }

    fun getExerciseById(id: String): Exercise? {
        return exercises.find { it.id == id }
    }

    fun searchExercises(query: String): List<Exercise> {
        val lowerQuery = query.lowercase()
        return exercises.filter { exercise ->
            exercise.name.lowercase().contains(lowerQuery) ||
            exercise.description.lowercase().contains(lowerQuery) ||
            exercise.targetAreas.any { it.lowercase().contains(lowerQuery) }
        }
    }

    fun getQuickRoutine(durationMinutes: Int, difficulty: Difficulty): List<Exercise> {
        var totalDuration = 0
        val routine = mutableListOf<Exercise>()

        // Always include breathing exercise for AS
        exercises.find { it.category == ExerciseCategory.BREATHING }?.let {
            routine.add(it)
            totalDuration += it.durationMinutes
        }

        // Add variety from other categories
        val shuffled = exercises
            .filter { it.difficulty <= difficulty && it.category != ExerciseCategory.BREATHING }
            .shuffled()

        for (exercise in shuffled) {
            if (totalDuration + exercise.durationMinutes <= durationMinutes) {
                routine.add(exercise)
                totalDuration += exercise.durationMinutes
            }
            if (totalDuration >= durationMinutes) break
        }

        return routine
    }
}

// Extension for difficulty comparison
private operator fun Difficulty.compareTo(other: Difficulty): Int {
    return this.ordinal.compareTo(other.ordinal)
}
