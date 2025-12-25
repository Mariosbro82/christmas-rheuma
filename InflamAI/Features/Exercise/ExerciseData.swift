//
//  ExerciseData.swift
//  InflamAI
//
//  Comprehensive exercise database for AS patients (50+ exercises)
//

import Foundation

extension Exercise {
    static let allExercises: [Exercise] = stretchingExercises + strengthExercises + mobilityExercises + breathingExercises + postureExercises + balanceExercises

    // MARK: - Stretching Exercises (12)

    private static let stretchingExercises: [Exercise] = [
        Exercise(
            id: UUID(),
            name: "Cat-Cow Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Entire Spine", "Lower Back"],
            instructions: [
                "Start on hands and knees",
                "Arch back (cow position) - look up",
                "Hold for 3 seconds",
                "Round back (cat position) - tuck chin",
                "Hold for 3 seconds",
                "Repeat 10 cycles"
            ],
            benefits: [
                "Mobilizes entire spine",
                "Reduces stiffness",
                "Warms up spinal muscles"
            ],
            safetyTips: [
                "Move slowly between positions",
                "Keep movements gentle",
                "Breathe deeply throughout"
            ],
            videoURL: "https://www.youtube.com/watch?v=y39PrKY_4JM",
            steps: [
                .info("Start on hands and knees with wrists under shoulders and knees under hips", imageHint: "figure.mind.and.body"),
                .info("Engage your core muscles gently", imageHint: "sparkles"),
                .timer("Arch your back (Cow position) - drop belly, lift chest and tailbone, look up", duration: 5, imageHint: "arrow.down"),
                .timer("Round your back (Cat position) - tuck chin to chest, pull belly to spine", duration: 5, imageHint: "arrow.up"),
                .reps("Repeat the flow smoothly", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Hip Flexor Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 8,
            targetAreas: ["Hips", "Lower Back"],
            instructions: [
                "Kneel on right knee, left foot forward",
                "Keep back straight",
                "Push hips forward gently",
                "Hold for 30 seconds",
                "Switch sides",
                "Repeat 3 times each side"
            ],
            benefits: [
                "Reduces hip stiffness",
                "Improves posture",
                "Decreases lower back strain"
            ],
            safetyTips: [
                "Don't bounce",
                "Keep pelvis neutral",
                "Use cushion under knee"
            ],
            videoURL: "https://www.youtube.com/watch?v=y39PrKY_4JM",
            steps: [
                .info("Place a cushion under your knee for comfort", imageHint: "circle.fill"),
                .info("Kneel on right knee with left foot forward, forming 90-degree angles at both knees", imageHint: "figure.walk"),
                .info("Keep your back straight and engage your core", imageHint: "arrow.up"),
                .timer("Push your hips forward gently until you feel a stretch in the front of your right hip", duration: 30, imageHint: "timer"),
                .info("Switch sides: kneel on left knee, right foot forward", imageHint: "arrow.left.and.right"),
                .reps("Complete 3 times on each side", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Spinal Twist (Supine)",
            category: .stretching,
            difficulty: .beginner,
            duration: 8,
            targetAreas: ["Thoracic Spine", "Lumbar Spine"],
            instructions: [
                "Lie on back",
                "Bring right knee to chest",
                "Gently guide knee across body to left",
                "Keep shoulders on ground",
                "Hold for 30 seconds",
                "Repeat on other side"
            ],
            benefits: [
                "Improves spinal rotation",
                "Stretches back muscles",
                "Relieves tension"
            ],
            safetyTips: [
                "Don't force the twist",
                "Keep movement gentle",
                "Stop if pain increases"
            ],
            videoURL: "https://www.youtube.com/watch?v=zTPfzlZbtz8",
            steps: [
                .info("Lie flat on your back with arms extended out to sides", imageHint: "figure.stand"),
                .info("Bring your right knee up to your chest", imageHint: "arrow.up"),
                .info("Gently guide your right knee across your body to the left side", imageHint: "arrow.left"),
                .info("Keep both shoulders flat on the ground", imageHint: "exclamationmark.circle"),
                .timer("Hold the twist, breathing deeply", duration: 30, imageHint: "timer"),
                .info("Return to center and switch sides", imageHint: "arrow.right"),
                .reps("Complete 2 times on each side", repetitions: 2, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Hamstring Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 6,
            targetAreas: ["Hamstrings", "Lower Back"],
            instructions: [
                "Sit with one leg extended",
                "Bend other leg with foot against inner thigh",
                "Reach toward extended foot",
                "Hold for 30 seconds",
                "Switch legs",
                "Repeat 3 times each side"
            ],
            benefits: [
                "Reduces hamstring tightness",
                "Improves flexibility",
                "Decreases lower back strain"
            ],
            safetyTips: [
                "Don't bounce",
                "Keep back straight",
                "Breathe throughout stretch"
            ],
            videoURL: "https://www.youtube.com/watch?v=mNdJti7ZwKI"
        ),
        Exercise(
            id: UUID(),
            name: "Chest Opener Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Chest", "Shoulders"],
            instructions: [
                "Stand in doorway",
                "Place forearms on door frame",
                "Step forward with one foot",
                "Feel stretch across chest",
                "Hold for 30 seconds",
                "Repeat 3 times"
            ],
            benefits: [
                "Opens chest",
                "Improves posture",
                "Reduces forward rounding"
            ],
            safetyTips: [
                "Don't overstretch",
                "Keep core engaged",
                "Stop if shoulder pain"
            ],
            videoURL: "https://www.youtube.com/watch?v=u55F2jOzBVI",
            steps: [
                .info("Stand in doorway", imageHint: "figure.stand"),
                .info("Place forearms on door frame", imageHint: "circle.fill"),
                .info("Step forward with one foot", imageHint: "arrow.forward"),
                .info("Feel stretch across chest", imageHint: "circle.fill"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer"),
                .reps("Repeat 3 times", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Piriformis Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 6,
            targetAreas: ["Hips", "Glutes"],
            instructions: [
                "Lie on back",
                "Cross right ankle over left knee",
                "Pull left thigh toward chest",
                "Hold for 30 seconds",
                "Switch sides",
                "Repeat 3 times each"
            ],
            benefits: [
                "Relieves hip tightness",
                "Reduces sciatic symptoms",
                "Improves hip mobility"
            ],
            safetyTips: [
                "Keep head on ground",
                "Don't force stretch",
                "Breathe deeply"
            ],
            videoURL: "https://www.youtube.com/shorts/crnw1IKWNZY",
            steps: [
                .info("Lie on back", imageHint: "bed.double"),
                .info("Cross right ankle over left knee", imageHint: "circle.fill"),
                .info("Pull left thigh toward chest", imageHint: "circle.fill"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer"),
                .info("Switch sides", imageHint: "arrow.left.and.right"),
                .reps("Repeat 3 times each", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Quadriceps Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Quadriceps", "Hip Flexors"],
            instructions: [
                "Stand on one leg",
                "Bend other knee behind you",
                "Grab ankle with hand",
                "Pull heel toward glutes",
                "Hold for 30 seconds",
                "Switch sides"
            ],
            benefits: [
                "Stretches quad muscles",
                "Improves knee flexibility",
                "Reduces hip flexor tightness"
            ],
            safetyTips: [
                "Use wall for balance",
                "Don't arch back",
                "Keep knees together"
            ],
            videoURL: "https://www.youtube.com/watch?v=tFtUgS69rPk",
            steps: [
                .info("Stand on one leg", imageHint: "figure.stand"),
                .info("Bend other knee behind you", imageHint: "circle.fill"),
                .info("Grab ankle with hand", imageHint: "circle.fill"),
                .info("Pull heel toward glutes", imageHint: "circle.fill"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer"),
                .info("Switch sides", imageHint: "arrow.left.and.right")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Lat Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Lats", "Side Body"],
            instructions: [
                "Stand with feet shoulder-width",
                "Raise right arm overhead",
                "Lean to left side",
                "Hold for 20 seconds",
                "Switch sides",
                "Repeat 3 times each"
            ],
            benefits: [
                "Stretches side body",
                "Improves lateral flexion",
                "Reduces side stiffness"
            ],
            safetyTips: [
                "Don't twist",
                "Keep hips stable",
                "Breathe throughout"
            ],
            videoURL: "https://www.youtube.com/watch?v=Td-9CSgSFhs",
            steps: [
                .info("Stand with feet shoulder-width", imageHint: "figure.stand"),
                .info("Raise right arm overhead", imageHint: "circle.fill"),
                .info("Lean to left side", imageHint: "circle.fill"),
                .timer("Hold for 20 seconds", duration: 20, imageHint: "timer"),
                .info("Switch sides", imageHint: "arrow.left.and.right"),
                .reps("Repeat 3 times each", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Child's Pose",
            category: .stretching,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Lower Back", "Hips"],
            instructions: [
                "Start on hands and knees",
                "Sit hips back toward heels",
                "Extend arms forward",
                "Rest forehead on ground",
                "Hold for 60 seconds",
                "Breathe deeply"
            ],
            benefits: [
                "Gently stretches spine",
                "Relieves back tension",
                "Promotes relaxation"
            ],
            safetyTips: [
                "Use cushion under knees",
                "Don't force position",
                "Breathe naturally"
            ],
            videoURL: "https://www.youtube.com/watch?v=izMQh1NeyRU",
            steps: [
                .info("Start on hands and knees", imageHint: "1.circle.fill"),
                .info("Sit hips back toward heels", imageHint: "figure.seated.side"),
                .info("Extend arms forward", imageHint: "arrow.forward"),
                .info("Rest forehead on ground", imageHint: "circle.fill"),
                .timer("Hold for 60 seconds", duration: 60, imageHint: "timer"),
                .info("Breathe deeply", imageHint: "wind")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Neck Side Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 4,
            targetAreas: ["Neck", "Trapezius"],
            instructions: [
                "Sit with good posture",
                "Tilt head toward right shoulder",
                "Use right hand to gently assist",
                "Hold for 20 seconds",
                "Switch sides",
                "Repeat 3 times each"
            ],
            benefits: [
                "Reduces neck tension",
                "Improves lateral neck mobility",
                "Relieves headaches"
            ],
            safetyTips: [
                "Don't pull hard",
                "Keep shoulders down",
                "Move gently"
            ],
            videoURL: "https://www.youtube.com/watch?v=eqVMAPM00DM",
            steps: [
                .info("Sit with good posture", imageHint: "figure.seated.side"),
                .info("Tilt head toward right shoulder", imageHint: "circle.fill"),
                .info("Use right hand to gently assist", imageHint: "circle.fill"),
                .timer("Hold for 20 seconds", duration: 20, imageHint: "timer"),
                .info("Switch sides", imageHint: "arrow.left.and.right"),
                .reps("Repeat 3 times each", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Knee-to-Chest Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Lower Back", "Hips"],
            instructions: [
                "Lie on back",
                "Bring one knee to chest",
                "Hold behind thigh",
                "Pull gently",
                "Hold for 30 seconds",
                "Switch legs"
            ],
            benefits: [
                "Stretches lower back",
                "Relieves hip tightness",
                "Reduces lumbar tension"
            ],
            safetyTips: [
                "Keep other leg extended",
                "Don't strain neck",
                "Breathe throughout"
            ],
            videoURL: "https://www.youtube.com/watch?v=H5h54Q0wpps",
            steps: [
                .info("Lie on back", imageHint: "bed.double"),
                .info("Bring one knee to chest", imageHint: "circle.fill"),
                .info("Hold behind thigh", imageHint: "circle.fill"),
                .info("Pull gently", imageHint: "circle.fill"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer"),
                .info("Switch legs", imageHint: "arrow.left.and.right")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Calf Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Calves", "Achilles"],
            instructions: [
                "Stand facing wall",
                "Place hands on wall",
                "Step back with right leg",
                "Keep heel down, leg straight",
                "Lean forward",
                "Hold for 30 seconds"
            ],
            benefits: [
                "Stretches calf muscles",
                "Improves ankle mobility",
                "Reduces achilles tightness"
            ],
            safetyTips: [
                "Keep back leg straight",
                "Don't bounce",
                "Keep heel on ground"
            ],
            videoURL: "https://www.youtube.com/watch?v=LugNxxfIdvo",
            steps: [
                .info("Stand facing wall", imageHint: "figure.stand"),
                .info("Place hands on wall", imageHint: "circle.fill"),
                .info("Step back with right leg", imageHint: "arrow.backward"),
                .info("Keep heel down, leg straight", imageHint: "arrow.down"),
                .info("Lean forward", imageHint: "arrow.forward"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer")
            ]
        )
    ]

    // MARK: - Strengthening Exercises (12)

    private static let strengthExercises: [Exercise] = [
        Exercise(
            id: UUID(),
            name: "Pelvic Tilts",
            category: .strengthening,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Lower Back", "Pelvis", "Core"],
            instructions: [
                "Lie on back, knees bent",
                "Flatten lower back against floor",
                "Tilt pelvis upward",
                "Hold for 5 seconds",
                "Relax",
                "Repeat 20 times"
            ],
            benefits: [
                "Strengthens core muscles",
                "Improves pelvic mobility",
                "Reduces lower back pain"
            ],
            safetyTips: [
                "Move slowly",
                "Don't hold breath",
                "Keep movements controlled"
            ],
            videoURL: "https://www.youtube.com/watch?v=y01ri_43G50",
            steps: [
                .info("Lie on back, knees bent", imageHint: "bed.double"),
                .info("Flatten lower back against floor", imageHint: "arrow.backward"),
                .info("Tilt pelvis upward", imageHint: "arrow.up"),
                .timer("Hold for 5 seconds", duration: 5, imageHint: "timer"),
                .info("Relax", imageHint: "circle.fill"),
                .reps("Repeat 20 times", repetitions: 20, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Bridge Exercise",
            category: .strengthening,
            difficulty: .intermediate,
            duration: 10,
            targetAreas: ["Glutes", "Lower Back", "Core"],
            instructions: [
                "Lie on back, knees bent, feet flat",
                "Lift hips off ground",
                "Form straight line from knees to shoulders",
                "Hold for 10 seconds",
                "Lower slowly",
                "Repeat 15 times"
            ],
            benefits: [
                "Strengthens glutes and lower back",
                "Improves hip extension",
                "May support spinal mobility"
            ],
            safetyTips: [
                "Don't overarch back",
                "Keep core engaged",
                "Breathe throughout"
            ],
            videoURL: "https://www.youtube.com/watch?v=KWi1YgyxDaQ",
            steps: [
                .info("Lie on back, knees bent, feet flat", imageHint: "bed.double"),
                .info("Lift hips off ground", imageHint: "circle.fill"),
                .info("Form straight line from knees to shoulders", imageHint: "circle.fill"),
                .timer("Hold for 10 seconds", duration: 10, imageHint: "timer"),
                .info("Lower slowly", imageHint: "circle.fill"),
                .reps("Repeat 15 times", repetitions: 15, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Bird Dog",
            category: .strengthening,
            difficulty: .intermediate,
            duration: 8,
            targetAreas: ["Core", "Back", "Glutes"],
            instructions: [
                "Start on hands and knees",
                "Extend right arm forward",
                "Extend left leg back",
                "Hold for 10 seconds",
                "Switch sides",
                "Repeat 10 times each side"
            ],
            benefits: [
                "Improves core stability",
                "Strengthens back muscles",
                "Enhances balance"
            ],
            safetyTips: [
                "Keep back neutral",
                "Don't twist",
                "Move slowly"
            ],
            videoURL: "https://www.youtube.com/watch?v=dQqApCGd5Ss",
            steps: [
                .info("Start on hands and knees", imageHint: "1.circle.fill"),
                .info("Extend right arm forward", imageHint: "arrow.forward"),
                .info("Extend left leg back", imageHint: "arrow.backward"),
                .timer("Hold for 10 seconds", duration: 10, imageHint: "timer"),
                .info("Switch sides", imageHint: "arrow.left.and.right"),
                .reps("Repeat 10 times each side", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Plank Hold",
            category: .strengthening,
            difficulty: .intermediate,
            duration: 10,
            targetAreas: ["Core", "Shoulders"],
            instructions: [
                "Start in push-up position",
                "Keep body straight",
                "Hold on forearms",
                "Engage core",
                "Hold for 30 seconds",
                "Rest and repeat 3 times"
            ],
            benefits: [
                "Builds core strength",
                "Improves posture",
                "Stabilizes spine"
            ],
            safetyTips: [
                "Don't let hips sag",
                "Keep neck neutral",
                "Breathe normally"
            ],
            videoURL: "https://www.youtube.com/watch?v=wiFNA3sqjCA",
            steps: [
                .info("Start in push-up position", imageHint: "figure.stand"),
                .info("Keep body straight", imageHint: "circle.fill"),
                .info("Hold on forearms", imageHint: "circle.fill"),
                .info("Engage core", imageHint: "circle.fill"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer"),
                .reps("Rest and repeat 3 times", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Side Plank",
            category: .strengthening,
            difficulty: .intermediate,
            duration: 8,
            targetAreas: ["Obliques", "Core", "Shoulders"],
            instructions: [
                "Lie on side",
                "Prop up on elbow",
                "Lift hips off ground",
                "Form straight line",
                "Hold for 20 seconds",
                "Switch sides"
            ],
            benefits: [
                "Strengthens lateral core",
                "Improves lateral stability",
                "Enhances balance"
            ],
            safetyTips: [
                "Keep body straight",
                "Don't let hips drop",
                "Modified version: keep knees down"
            ],
            videoURL: "https://www.youtube.com/watch?v=pSHjTRCQxIw",
            steps: [
                .info("Lie on side", imageHint: "bed.double"),
                .info("Prop up on elbow", imageHint: "arrow.up"),
                .info("Lift hips off ground", imageHint: "circle.fill"),
                .info("Form straight line", imageHint: "circle.fill"),
                .timer("Hold for 20 seconds", duration: 20, imageHint: "timer"),
                .info("Switch sides", imageHint: "arrow.left.and.right")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Wall Squats",
            category: .strengthening,
            difficulty: .beginner,
            duration: 10,
            targetAreas: ["Quads", "Glutes", "Core"],
            instructions: [
                "Stand with back against wall",
                "Slide down to 90-degree squat",
                "Hold for 30 seconds",
                "Slide back up",
                "Rest 30 seconds",
                "Repeat 5 times"
            ],
            benefits: [
                "Strengthens legs",
                "Improves endurance",
                "Supports posture"
            ],
            safetyTips: [
                "Don't let knees go past toes",
                "Keep back flat against wall",
                "Stop if knee pain"
            ],
            videoURL: "https://www.youtube.com/watch?v=9dNL_mtObGQ",
            steps: [
                .info("Stand with back against wall", imageHint: "figure.stand"),
                .info("Slide down to 90-degree squat", imageHint: "arrow.down"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer"),
                .info("Slide back up", imageHint: "arrow.backward"),
                .info("Rest 30 seconds", imageHint: "circle.fill"),
                .reps("Repeat 5 times", repetitions: 5, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Prone Back Extension (Superman)",
            category: .strengthening,
            difficulty: .intermediate,
            duration: 8,
            targetAreas: ["Back Extensors", "Glutes"],
            instructions: [
                "Lie face down",
                "Arms extended overhead",
                "Lift arms, chest, and legs off ground",
                "Hold for 5 seconds",
                "Lower slowly",
                "Repeat 12 times"
            ],
            benefits: [
                "Strengthens back extensors",
                "Improves spinal stability",
                "Counters forward flexion"
            ],
            safetyTips: [
                "Don't hyperextend",
                "Keep movements controlled",
                "Stop if back pain"
            ],
            videoURL: "https://www.youtube.com/watch?v=aKBxiKs9n8A",
            steps: [
                .info("Lie face down", imageHint: "bed.double"),
                .info("Arms extended overhead", imageHint: "circle.fill"),
                .info("Lift arms, chest, and legs off ground", imageHint: "circle.fill"),
                .timer("Hold for 5 seconds", duration: 5, imageHint: "timer"),
                .info("Lower slowly", imageHint: "circle.fill"),
                .reps("Repeat 12 times", repetitions: 12, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Scapular Retraction",
            category: .strengthening,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Upper Back", "Rhomboids"],
            instructions: [
                "Sit or stand with good posture",
                "Pull shoulder blades together",
                "Hold for 5 seconds",
                "Relax",
                "Repeat 15 times"
            ],
            benefits: [
                "Strengthens upper back",
                "Improves posture",
                "Reduces rounded shoulders"
            ],
            safetyTips: [
                "Don't shrug shoulders",
                "Keep neck relaxed",
                "Focus on squeezing shoulder blades"
            ],
            videoURL: "https://www.youtube.com/watch?v=z6PJMT2y8GQ",
            steps: [
                .info("Sit or stand with good posture", imageHint: "figure.stand"),
                .info("Pull shoulder blades together", imageHint: "circle.fill"),
                .timer("Hold for 5 seconds", duration: 5, imageHint: "timer"),
                .info("Relax", imageHint: "circle.fill"),
                .reps("Repeat 15 times", repetitions: 15, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Resistance Band Rows",
            category: .strengthening,
            difficulty: .intermediate,
            duration: 10,
            targetAreas: ["Upper Back", "Lats"],
            instructions: [
                "Sit with legs extended",
                "Loop band around feet",
                "Pull handles toward ribs",
                "Squeeze shoulder blades",
                "Return slowly",
                "Repeat 15 times"
            ],
            benefits: [
                "Strengthens back muscles",
                "Improves posture",
                "Enhances pulling strength"
            ],
            safetyTips: [
                "Keep back straight",
                "Don't jerk the band",
                "Control the movement"
            ],
            videoURL: "https://www.youtube.com/watch?v=ouRhQE2iOI8",
            steps: [
                .info("Sit with legs extended", imageHint: "figure.seated.side"),
                .info("Loop band around feet", imageHint: "circle.fill"),
                .info("Pull handles toward ribs", imageHint: "circle.fill"),
                .info("Squeeze shoulder blades", imageHint: "circle.fill"),
                .info("Return slowly", imageHint: "circle.fill"),
                .reps("Repeat 15 times", repetitions: 15, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Standing Leg Lifts",
            category: .strengthening,
            difficulty: .beginner,
            duration: 8,
            targetAreas: ["Hip Abductors", "Glutes"],
            instructions: [
                "Stand with support",
                "Lift right leg to side",
                "Keep leg straight",
                "Hold for 2 seconds",
                "Lower slowly",
                "Repeat 15 times, switch legs"
            ],
            benefits: [
                "Strengthens hip muscles",
                "Improves balance",
                "Stabilizes pelvis"
            ],
            safetyTips: [
                "Don't lean to side",
                "Keep core engaged",
                "Move controlled"
            ],
            videoURL: "https://www.youtube.com/watch?v=LSkyinhmA8k",
            steps: [
                .info("Stand with support", imageHint: "figure.stand"),
                .info("Lift right leg to side", imageHint: "circle.fill"),
                .info("Keep leg straight", imageHint: "circle.fill"),
                .timer("Hold for 2 seconds", duration: 2, imageHint: "timer"),
                .info("Lower slowly", imageHint: "circle.fill"),
                .reps("Repeat 15 times, switch legs", repetitions: 15, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Quadruped Arm Raises",
            category: .strengthening,
            difficulty: .beginner,
            duration: 6,
            targetAreas: ["Shoulders", "Core"],
            instructions: [
                "Start on hands and knees",
                "Extend right arm forward",
                "Hold for 10 seconds",
                "Lower and switch",
                "Repeat 10 times each arm"
            ],
            benefits: [
                "Builds shoulder stability",
                "Strengthens core",
                "Improves balance"
            ],
            safetyTips: [
                "Keep back neutral",
                "Don't twist torso",
                "Move slowly"
            ],
            videoURL: "https://www.youtube.com/watch?v=Fd0LMejC1QA",
            steps: [
                .info("Start on hands and knees", imageHint: "1.circle.fill"),
                .info("Extend right arm forward", imageHint: "arrow.forward"),
                .timer("Hold for 10 seconds", duration: 10, imageHint: "timer"),
                .info("Lower and switch", imageHint: "arrow.left.and.right"),
                .reps("Repeat 10 times each arm", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Glute Kickbacks",
            category: .strengthening,
            difficulty: .beginner,
            duration: 8,
            targetAreas: ["Glutes", "Lower Back"],
            instructions: [
                "Start on hands and knees",
                "Extend right leg back and up",
                "Keep knee bent 90 degrees",
                "Hold for 2 seconds",
                "Lower and repeat",
                "Do 15 reps each leg"
            ],
            benefits: [
                "Strengthens glutes",
                "Improves hip extension",
                "Supports lower back"
            ],
            safetyTips: [
                "Don't arch lower back",
                "Keep core tight",
                "Control the movement"
            ],
            videoURL: "https://www.youtube.com/watch?v=DA69N9n9k34",
            steps: [
                .info("Start on hands and knees", imageHint: "1.circle.fill"),
                .info("Extend right leg back and up", imageHint: "arrow.backward"),
                .info("Keep knee bent 90 degrees", imageHint: "circle.fill"),
                .timer("Hold for 2 seconds", duration: 2, imageHint: "timer"),
                .info("Lower and repeat", imageHint: "circle.fill"),
                .reps("Do 15 reps each leg", repetitions: 15, imageHint: "repeat")
            ]
        )
    ]

    // MARK: - Mobility Exercises (10)

    private static let mobilityExercises: [Exercise] = [
        Exercise(
            id: UUID(),
            name: "Cervical Spine Rotation",
            category: .mobility,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Cervical Spine", "Neck"],
            instructions: [
                "Sit or stand with good posture",
                "Slowly turn head to the right as far as comfortable",
                "Hold for 5 seconds",
                "Return to center",
                "Repeat on left side",
                "Perform 10 repetitions each side"
            ],
            benefits: [
                "Improves cervical spine mobility",
                "Reduces neck stiffness",
                "Maintains range of motion"
            ],
            safetyTips: [
                "Move slowly and gently",
                "Stop if you feel sharp pain",
                "Do not force the movement"
            ],
            videoURL: "https://www.youtube.com/watch?v=tgZwcHbb3oU",
            steps: [
                .info("Sit or stand with good posture", imageHint: "figure.stand"),
                .info("Slowly turn head to the right as far as comfortable", imageHint: "circle.fill"),
                .timer("Hold for 5 seconds", duration: 5, imageHint: "timer"),
                .info("Return to center", imageHint: "circle.fill"),
                .info("Repeat on left side", imageHint: "circle.fill"),
                .reps("Perform 10 repetitions each side", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Thoracic Extension Over Foam Roller",
            category: .mobility,
            difficulty: .intermediate,
            duration: 10,
            targetAreas: ["Thoracic Spine", "Upper Back"],
            instructions: [
                "Lie on foam roller positioned at mid-back",
                "Support head with hands",
                "Slowly extend backwards over roller",
                "Hold for 3-5 seconds",
                "Return to starting position",
                "Repeat 10 times"
            ],
            benefits: [
                "Increases thoracic extension",
                "Reduces kyphosis",
                "Improves posture"
            ],
            safetyTips: [
                "Use firm foam roller",
                "Avoid if you have osteoporosis",
                "Stop if experiencing pain"
            ],
            videoURL: "https://www.youtube.com/watch?v=tHyY6ZQ8kXk",
            steps: [
                .info("Lie on foam roller positioned at mid-back", imageHint: "figure.stand"),
                .info("Support head with hands", imageHint: "arrow.up"),
                .info("Slowly extend backwards over roller", imageHint: "arrow.backward"),
                .info("Hold for 3-5 seconds", imageHint: "circle.fill"),
                .info("Return to starting position", imageHint: "figure.stand"),
                .reps("Repeat 10 times", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Hip Circles",
            category: .mobility,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Hips", "Pelvis"],
            instructions: [
                "Stand on one leg",
                "Lift other knee to 90 degrees",
                "Circle knee outward 10 times",
                "Circle knee inward 10 times",
                "Switch legs",
                "Repeat"
            ],
            benefits: [
                "Improves hip mobility",
                "Reduces hip stiffness",
                "Enhances balance"
            ],
            safetyTips: [
                "Use wall for balance",
                "Start with small circles",
                "Keep core engaged"
            ],
            videoURL: "https://www.youtube.com/watch?v=QzE8T5Ew-xA",
            steps: [
                .info("Stand on one leg", imageHint: "figure.stand"),
                .info("Lift other knee to 90 degrees", imageHint: "circle.fill"),
                .info("Circle knee outward 10 times", imageHint: "circle.fill"),
                .info("Circle knee inward 10 times", imageHint: "circle.fill"),
                .info("Switch legs", imageHint: "arrow.left.and.right"),
                .info("Repeat", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Thoracic Rotation (Seated)",
            category: .mobility,
            difficulty: .beginner,
            duration: 6,
            targetAreas: ["Thoracic Spine"],
            instructions: [
                "Sit with arms crossed on chest",
                "Rotate upper body to right",
                "Hold for 5 seconds",
                "Return to center",
                "Rotate to left",
                "Repeat 10 times each side"
            ],
            benefits: [
                "Improves thoracic rotation",
                "Maintains spinal mobility",
                "Reduces stiffness"
            ],
            safetyTips: [
                "Keep hips stationary",
                "Move slowly",
                "Don't force rotation"
            ],
            videoURL: "https://www.youtube.com/watch?v=RIJ38Z2KWsQ",
            steps: [
                .info("Sit with arms crossed on chest", imageHint: "figure.seated.side"),
                .info("Rotate upper body to right", imageHint: "arrow.up"),
                .timer("Hold for 5 seconds", duration: 5, imageHint: "timer"),
                .info("Return to center", imageHint: "circle.fill"),
                .info("Rotate to left", imageHint: "circle.fill"),
                .reps("Repeat 10 times each side", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Ankle Circles",
            category: .mobility,
            difficulty: .beginner,
            duration: 4,
            targetAreas: ["Ankles"],
            instructions: [
                "Sit or lie down",
                "Lift one foot off ground",
                "Circle ankle clockwise 10 times",
                "Circle counterclockwise 10 times",
                "Switch feet",
                "Repeat"
            ],
            benefits: [
                "Improves ankle mobility",
                "Reduces stiffness",
                "Enhances circulation"
            ],
            safetyTips: [
                "Make slow, controlled circles",
                "Point and flex through movement",
                "Don't force movement"
            ],
            videoURL: "https://www.youtube.com/watch?v=uDN_HAD1ny4",
            steps: [
                .info("Sit or lie down", imageHint: "figure.seated.side"),
                .info("Lift one foot off ground", imageHint: "circle.fill"),
                .info("Circle ankle clockwise 10 times", imageHint: "circle.fill"),
                .info("Circle counterclockwise 10 times", imageHint: "circle.fill"),
                .info("Switch feet", imageHint: "arrow.left.and.right"),
                .info("Repeat", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Shoulder Rolls",
            category: .mobility,
            difficulty: .beginner,
            duration: 3,
            targetAreas: ["Shoulders", "Upper Back"],
            instructions: [
                "Sit or stand with good posture",
                "Roll shoulders backward 10 times",
                "Roll shoulders forward 10 times",
                "Keep movements smooth",
                "Repeat as needed"
            ],
            benefits: [
                "Reduces shoulder tension",
                "Improves shoulder mobility",
                "Relieves upper back stiffness"
            ],
            safetyTips: [
                "Don't shrug excessively",
                "Keep neck relaxed",
                "Breathe normally"
            ],
            videoURL: "https://www.youtube.com/watch?v=mzTQGYGI0Ng",
            steps: [
                .info("Sit or stand with good posture", imageHint: "figure.stand"),
                .info("Roll shoulders backward 10 times", imageHint: "arrow.backward"),
                .info("Roll shoulders forward 10 times", imageHint: "arrow.forward"),
                .info("Keep movements smooth", imageHint: "circle.fill"),
                .info("Repeat as needed", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Spinal Flexion/Extension",
            category: .mobility,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Entire Spine"],
            instructions: [
                "Stand with feet shoulder-width",
                "Slowly round forward (flexion)",
                "Return to standing",
                "Gently extend backward (extension)",
                "Return to standing",
                "Repeat 10 cycles"
            ],
            benefits: [
                "Maintains spinal mobility",
                "Reduces stiffness",
                "Improves flexibility"
            ],
            safetyTips: [
                "Move slowly",
                "Support back with hands if needed",
                "Don't overextend"
            ],
            videoURL: "https://www.youtube.com/watch?v=XbzY45Z5DE8",
            steps: [
                .info("Stand with feet shoulder-width", imageHint: "figure.stand"),
                .info("Slowly round forward (flexion)", imageHint: "arrow.forward"),
                .info("Return to standing", imageHint: "figure.stand"),
                .info("Gently extend backward (extension)", imageHint: "arrow.backward"),
                .info("Return to standing", imageHint: "figure.stand"),
                .reps("Repeat 10 cycles", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Wrist Circles",
            category: .mobility,
            difficulty: .beginner,
            duration: 3,
            targetAreas: ["Wrists"],
            instructions: [
                "Extend arms forward",
                "Make circles with wrists",
                "10 clockwise",
                "10 counterclockwise",
                "Repeat as needed"
            ],
            benefits: [
                "Improves wrist mobility",
                "Reduces wrist stiffness",
                "Helps with hand function"
            ],
            safetyTips: [
                "Keep movements gentle",
                "Don't force range",
                "Stop if painful"
            ],
            videoURL: "https://www.youtube.com/watch?v=GZKW6K4L89s",
            steps: [
                .info("Extend arms forward", imageHint: "arrow.forward"),
                .info("Make circles with wrists", imageHint: "circle.fill"),
                .info("10 clockwise", imageHint: "circle.fill"),
                .info("10 counterclockwise", imageHint: "circle.fill"),
                .info("Repeat as needed", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Lumbar Side Bending",
            category: .mobility,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Lumbar Spine", "Side Body"],
            instructions: [
                "Stand with feet shoulder-width",
                "Slide right hand down thigh",
                "Bend to right side",
                "Hold for 5 seconds",
                "Return to center",
                "Repeat on left"
            ],
            benefits: [
                "Improves lateral mobility",
                "Stretches side muscles",
                "Maintains spinal flexibility"
            ],
            safetyTips: [
                "Don't twist",
                "Keep movements in frontal plane",
                "Go slowly"
            ],
            videoURL: "https://www.youtube.com/watch?v=mzTQGYGI0Ng",
            steps: [
                .info("Stand with feet shoulder-width", imageHint: "figure.stand"),
                .info("Slide right hand down thigh", imageHint: "arrow.down"),
                .info("Bend to right side", imageHint: "circle.fill"),
                .timer("Hold for 5 seconds", duration: 5, imageHint: "timer"),
                .info("Return to center", imageHint: "circle.fill"),
                .info("Repeat on left", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Cervical Lateral Flexion",
            category: .mobility,
            difficulty: .beginner,
            duration: 4,
            targetAreas: ["Cervical Spine", "Neck"],
            instructions: [
                "Sit with good posture",
                "Tilt ear toward right shoulder",
                "Hold for 10 seconds",
                "Return to center",
                "Tilt to left",
                "Repeat 8 times each side"
            ],
            benefits: [
                "Improves neck lateral mobility",
                "Reduces neck stiffness",
                "Maintains range of motion"
            ],
            safetyTips: [
                "Don't rotate head",
                "Keep shoulders down",
                "Move gently"
            ],
            videoURL: "https://www.youtube.com/watch?v=AkQt-QXQXIQ",
            steps: [
                .info("Sit with good posture", imageHint: "figure.seated.side"),
                .info("Tilt ear toward right shoulder", imageHint: "circle.fill"),
                .timer("Hold for 10 seconds", duration: 10, imageHint: "timer"),
                .info("Return to center", imageHint: "circle.fill"),
                .info("Tilt to left", imageHint: "circle.fill"),
                .reps("Repeat 8 times each side", repetitions: 8, imageHint: "repeat")
            ]
        )
    ]

    // MARK: - Breathing Exercises (6)

    private static let breathingExercises: [Exercise] = [
        Exercise(
            id: UUID(),
            name: "Deep Breathing Exercise",
            category: .breathing,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Chest", "Rib Cage"],
            instructions: [
                "Sit or lie comfortably",
                "Place hands on ribs",
                "Breathe in deeply through nose for 4 counts",
                "Feel ribs expand",
                "Exhale slowly through mouth for 6 counts",
                "Repeat for 5 minutes"
            ],
            benefits: [
                "Maintains chest expansion",
                "Prevents fusion complications",
                "Reduces anxiety"
            ],
            safetyTips: [
                "Don't hyperventilate",
                "Stop if dizzy",
                "Practice regularly"
            ],
            videoURL: "https://www.youtube.com/watch?v=Jl4hZp2ZVt0",
            steps: [
                .info("Sit or lie comfortably", imageHint: "figure.seated.side"),
                .info("Place hands on ribs", imageHint: "circle.fill"),
                .info("Breathe in deeply through nose for 4 counts", imageHint: "wind"),
                .info("Feel ribs expand", imageHint: "circle.fill"),
                .info("Exhale slowly through mouth for 6 counts", imageHint: "wind"),
                .info("Repeat for 5 minutes", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Box Breathing",
            category: .breathing,
            difficulty: .beginner,
            duration: 8,
            targetAreas: ["Chest", "Diaphragm"],
            instructions: [
                "Sit comfortably",
                "Inhale for 4 counts",
                "Hold for 4 counts",
                "Exhale for 4 counts",
                "Hold for 4 counts",
                "Repeat 10 cycles"
            ],
            benefits: [
                "Improves breath control",
                "Reduces stress",
                "Enhances lung capacity"
            ],
            safetyTips: [
                "Don't strain",
                "Adjust count if needed",
                "Breathe naturally between cycles"
            ],
            videoURL: "https://www.youtube.com/watch?v=Xo9pL-Nk0GU",
            steps: [
                .info("Sit comfortably", imageHint: "figure.seated.side"),
                .info("Inhale for 4 counts", imageHint: "wind"),
                .info("Hold for 4 counts", imageHint: "circle.fill"),
                .info("Exhale for 4 counts", imageHint: "wind"),
                .info("Hold for 4 counts", imageHint: "circle.fill"),
                .reps("Repeat 10 cycles", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Rib Cage Expansion",
            category: .breathing,
            difficulty: .beginner,
            duration: 6,
            targetAreas: ["Rib Cage", "Intercostals"],
            instructions: [
                "Sit tall",
                "Place hands on sides of ribs",
                "Breathe into hands",
                "Feel ribs expand laterally",
                "Exhale slowly",
                "Repeat 15 times"
            ],
            benefits: [
                "Maintains rib mobility",
                "Prevents stiffness",
                "Improves chest expansion"
            ],
            safetyTips: [
                "Focus on lateral expansion",
                "Don't force breathing",
                "Practice daily"
            ],
            videoURL: "https://www.youtube.com/watch?v=tEmt1Znux58",
            steps: [
                .info("Sit tall", imageHint: "figure.seated.side"),
                .info("Place hands on sides of ribs", imageHint: "circle.fill"),
                .info("Breathe into hands", imageHint: "wind"),
                .info("Feel ribs expand laterally", imageHint: "circle.fill"),
                .info("Exhale slowly", imageHint: "wind"),
                .reps("Repeat 15 times", repetitions: 15, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Pursed Lip Breathing",
            category: .breathing,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Lungs", "Airways"],
            instructions: [
                "Relax neck and shoulders",
                "Inhale through nose for 2 counts",
                "Purse lips as if whistling",
                "Exhale slowly through pursed lips for 4 counts",
                "Repeat 10 times"
            ],
            benefits: [
                "Improves breathing efficiency",
                "Reduces breathlessness",
                "Calms nervous system"
            ],
            safetyTips: [
                "Don't force exhalation",
                "Keep breathing relaxed",
                "Use during activities"
            ],
            videoURL: "https://www.youtube.com/watch?v=ycx3Pm4yR-8",
            steps: [
                .info("Relax neck and shoulders", imageHint: "1.circle.fill"),
                .info("Inhale through nose for 2 counts", imageHint: "wind"),
                .info("Purse lips as if whistling", imageHint: "circle.fill"),
                .info("Exhale slowly through pursed lips for 4 counts", imageHint: "wind"),
                .reps("Repeat 10 times", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Diaphragmatic Breathing",
            category: .breathing,
            difficulty: .beginner,
            duration: 7,
            targetAreas: ["Diaphragm", "Abdomen"],
            instructions: [
                "Lie on back, knees bent",
                "Place one hand on chest, one on belly",
                "Breathe so belly rises, chest stays still",
                "Inhale for 4 counts",
                "Exhale for 6 counts",
                "Repeat for 5 minutes"
            ],
            benefits: [
                "Strengthens diaphragm",
                "Improves oxygen exchange",
                "Promotes relaxation"
            ],
            safetyTips: [
                "Chest should barely move",
                "Focus on belly movement",
                "Practice twice daily"
            ],
            videoURL: "https://www.youtube.com/watch?v=Qw2DS1Yqv8E",
            steps: [
                .info("Lie on back, knees bent", imageHint: "bed.double"),
                .info("Place one hand on chest, one on belly", imageHint: "circle.fill"),
                .info("Breathe so belly rises, chest stays still", imageHint: "wind"),
                .info("Inhale for 4 counts", imageHint: "wind"),
                .info("Exhale for 6 counts", imageHint: "wind"),
                .info("Repeat for 5 minutes", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "4-7-8 Breathing",
            category: .breathing,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Lungs", "Nervous System"],
            instructions: [
                "Sit comfortably",
                "Exhale completely",
                "Inhale through nose for 4 counts",
                "Hold breath for 7 counts",
                "Exhale through mouth for 8 counts",
                "Repeat 4 times"
            ],
            benefits: [
                "Reduces anxiety",
                "Improves sleep",
                "Calms mind"
            ],
            safetyTips: [
                "Don't do more than 4 cycles initially",
                "Stop if lightheaded",
                "Practice before bed"
            ],
            videoURL: "https://www.youtube.com/watch?v=9jpchJcKivk",
            steps: [
                .info("Sit comfortably", imageHint: "figure.seated.side"),
                .info("Exhale completely", imageHint: "wind"),
                .info("Inhale through nose for 4 counts", imageHint: "wind"),
                .info("Hold breath for 7 counts", imageHint: "wind"),
                .info("Exhale through mouth for 8 counts", imageHint: "wind"),
                .reps("Repeat 4 times", repetitions: 4, imageHint: "repeat")
            ]
        )
    ]

    // MARK: - Posture Exercises (6)

    private static let postureExercises: [Exercise] = [
        Exercise(
            id: UUID(),
            name: "Wall Angels",
            category: .posture,
            difficulty: .intermediate,
            duration: 10,
            targetAreas: ["Shoulders", "Upper Back", "Thoracic Spine"],
            instructions: [
                "Stand with back against wall",
                "Press lower back to wall",
                "Raise arms to 90 degrees (goal post position)",
                "Keep elbows and hands touching wall",
                "Slowly raise arms overhead",
                "Return to start position",
                "Repeat 15 times"
            ],
            benefits: [
                "Improves shoulder mobility",
                "Strengthens upper back",
                "May help improve head posture"
            ],
            safetyTips: [
                "Don't arch lower back",
                "Move within pain-free range",
                "Stop if shoulder pain occurs"
            ],
            videoURL: "https://www.youtube.com/watch?v=gz4G31LGyog",
            steps: [
                .info("Stand with back against wall", imageHint: "figure.stand"),
                .info("Press lower back to wall", imageHint: "arrow.backward"),
                .info("Raise arms to 90 degrees (goal post position)", imageHint: "figure.stand"),
                .info("Keep elbows and hands touching wall", imageHint: "circle.fill"),
                .info("Slowly raise arms overhead", imageHint: "circle.fill"),
                .info("Return to start position", imageHint: "figure.stand"),
                .reps("Repeat 15 times", repetitions: 15, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Chin Tucks",
            category: .posture,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Cervical Spine", "Neck"],
            instructions: [
                "Sit or stand with good posture",
                "Look straight ahead",
                "Gently tuck chin back (double chin)",
                "Hold for 5 seconds",
                "Relax",
                "Repeat 10 times"
            ],
            benefits: [
                "May help improve head posture",
                "Strengthens deep neck flexors",
                "Reduces neck pain"
            ],
            safetyTips: [
                "Don't tilt head down",
                "Keep eyes level",
                "Movement should be small"
            ],
            videoURL: "https://www.youtube.com/watch?v=KnUfVYdeorU",
            steps: [
                .info("Sit or stand with good posture", imageHint: "figure.stand"),
                .info("Look straight ahead", imageHint: "circle.fill"),
                .info("Gently tuck chin back (double chin)", imageHint: "arrow.backward"),
                .timer("Hold for 5 seconds", duration: 5, imageHint: "timer"),
                .info("Relax", imageHint: "circle.fill"),
                .reps("Repeat 10 times", repetitions: 10, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Brugger's Relief Position",
            category: .posture,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Entire Spine", "Shoulders"],
            instructions: [
                "Sit at edge of chair",
                "Feet slightly apart and turned out",
                "Rock pelvis forward (arch lower back)",
                "Rotate arms outward",
                "Pull shoulder blades together",
                "Hold for 30 seconds"
            ],
            benefits: [
                "Counteracts slouching",
                "Opens chest",
                "Improves posture awareness"
            ],
            safetyTips: [
                "Don't overarch back",
                "Keep breathing normal",
                "Use hourly when sitting"
            ],
            videoURL: "https://www.youtube.com/watch?v=K2V-L2sTGSM",
            steps: [
                .info("Sit at edge of chair", imageHint: "figure.seated.side"),
                .info("Feet slightly apart and turned out", imageHint: "circle.fill"),
                .info("Rock pelvis forward (arch lower back)", imageHint: "arrow.forward"),
                .info("Rotate arms outward", imageHint: "circle.fill"),
                .info("Pull shoulder blades together", imageHint: "circle.fill"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Standing Posture Check",
            category: .posture,
            difficulty: .beginner,
            duration: 3,
            targetAreas: ["Entire Body"],
            instructions: [
                "Stand against wall",
                "Touch wall with back of head, shoulders, and buttocks",
                "Check space behind lower back",
                "Should fit hand width",
                "Hold position for 30 seconds",
                "Step away and maintain"
            ],
            benefits: [
                "Improves posture awareness",
                "Teaches proper alignment",
                "Reduces slouching"
            ],
            safetyTips: [
                "Don't force position",
                "Check regularly",
                "Practice throughout day"
            ],
            videoURL: "https://www.youtube.com/watch?v=0e5u-VeA0lA",
            steps: [
                .info("Stand against wall", imageHint: "figure.stand"),
                .info("Touch wall with back of head, shoulders, and buttocks", imageHint: "arrow.backward"),
                .info("Check space behind lower back", imageHint: "arrow.backward"),
                .info("Should fit hand width", imageHint: "circle.fill"),
                .info("Hold position for 30 seconds", imageHint: "figure.stand"),
                .info("Step away and maintain", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Doorway Pec Stretch",
            category: .posture,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Chest", "Shoulders"],
            instructions: [
                "Stand in doorway",
                "Place forearms on frame",
                "Step forward with one leg",
                "Lean forward until stretch felt",
                "Hold for 30 seconds",
                "Repeat 3 times"
            ],
            benefits: [
                "Opens tight chest",
                "Reduces forward rounding",
                "Improves shoulder position"
            ],
            safetyTips: [
                "Don't overstretch",
                "Keep core engaged",
                "Stop if shoulder pain"
            ],
            videoURL: "https://www.youtube.com/watch?v=K-sCGl6JVxE",
            steps: [
                .info("Stand in doorway", imageHint: "figure.stand"),
                .info("Place forearms on frame", imageHint: "circle.fill"),
                .info("Step forward with one leg", imageHint: "arrow.forward"),
                .info("Lean forward until stretch felt", imageHint: "arrow.forward"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer"),
                .reps("Repeat 3 times", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Prone Lying",
            category: .posture,
            difficulty: .beginner,
            duration: 10,
            targetAreas: ["Entire Spine", "Hip Flexors"],
            instructions: [
                "Lie face down on firm surface",
                "Arms by sides",
                "Head turned to one side",
                "Relax completely",
                "Hold for 10 minutes",
                "Practice daily"
            ],
            benefits: [
                "Counteracts flexed posture",
                "Stretches hip flexors",
                "Promotes extension"
            ],
            safetyTips: [
                "Use thin pillow if needed",
                "Don't do if painful",
                "Build up duration gradually"
            ],
            videoURL: "https://www.youtube.com/watch?v=opxgkMhfCHk",
            steps: [
                .info("Lie face down on firm surface", imageHint: "bed.double"),
                .info("Arms by sides", imageHint: "circle.fill"),
                .info("Head turned to one side", imageHint: "circle.fill"),
                .info("Relax completely", imageHint: "circle.fill"),
                .info("Hold for 10 minutes", imageHint: "circle.fill"),
                .info("Practice daily", imageHint: "circle.fill")
            ]
        )
    ]

    // MARK: - Balance Exercises (6)

    private static let balanceExercises: [Exercise] = [
        Exercise(
            id: UUID(),
            name: "Single Leg Stand",
            category: .balance,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Ankles", "Core", "Hips"],
            instructions: [
                "Stand near wall for support",
                "Lift one foot off ground",
                "Hold for 30 seconds",
                "Switch legs",
                "Repeat 3 times each leg"
            ],
            benefits: [
                "Improves balance",
                "Strengthens ankles",
                "Enhances proprioception"
            ],
            safetyTips: [
                "Use wall if needed",
                "Start with shorter holds",
                "Progress gradually"
            ],
            videoURL: "https://www.youtube.com/watch?v=RfLJKBC-odk",
            steps: [
                .info("Stand near wall for support", imageHint: "figure.stand"),
                .info("Lift one foot off ground", imageHint: "circle.fill"),
                .timer("Hold for 30 seconds", duration: 30, imageHint: "timer"),
                .info("Switch legs", imageHint: "arrow.left.and.right"),
                .reps("Repeat 3 times each leg", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Heel-to-Toe Walk",
            category: .balance,
            difficulty: .intermediate,
            duration: 5,
            targetAreas: ["Balance", "Coordination"],
            instructions: [
                "Stand with feet together",
                "Place right heel directly in front of left toes",
                "Walk forward 10 steps",
                "Turn and walk back",
                "Repeat 3 times"
            ],
            benefits: [
                "Improves dynamic balance",
                "Enhances coordination",
                "Builds confidence"
            ],
            safetyTips: [
                "Walk along wall initially",
                "Go slowly",
                "Focus on a point ahead"
            ],
            videoURL: "https://www.youtube.com/watch?v=4s-FOz95u5E",
            steps: [
                .info("Stand with feet together", imageHint: "figure.stand"),
                .info("Place right heel directly in front of left toes", imageHint: "circle.fill"),
                .info("Walk forward 10 steps", imageHint: "arrow.forward"),
                .info("Turn and walk back", imageHint: "arrow.backward"),
                .reps("Repeat 3 times", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Clock Reaches",
            category: .balance,
            difficulty: .intermediate,
            duration: 8,
            targetAreas: ["Balance", "Core", "Hips"],
            instructions: [
                "Stand on right leg",
                "Reach left leg to 12 o'clock position",
                "Return to center",
                "Reach to 3 o'clock",
                "Continue around clock",
                "Switch legs"
            ],
            benefits: [
                "Challenges balance in all directions",
                "Improves hip control",
                "Enhances stability"
            ],
            safetyTips: [
                "Use chair for support",
                "Start with small reaches",
                "Build range gradually"
            ],
            videoURL: "https://www.youtube.com/watch?v=l9gXgl1lrDA",
            steps: [
                .info("Stand on right leg", imageHint: "figure.stand"),
                .info("Reach left leg to 12 o'clock position", imageHint: "figure.stand"),
                .info("Return to center", imageHint: "circle.fill"),
                .info("Reach to 3 o'clock", imageHint: "circle.fill"),
                .info("Continue around clock", imageHint: "circle.fill"),
                .info("Switch legs", imageHint: "arrow.left.and.right")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Standing Marches",
            category: .balance,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Balance", "Hip Flexors"],
            instructions: [
                "Stand tall",
                "March in place",
                "Lift knees to 90 degrees",
                "Swing arms naturally",
                "Continue for 1 minute",
                "Rest and repeat"
            ],
            benefits: [
                "Improves dynamic balance",
                "Strengthens hip flexors",
                "Enhances coordination"
            ],
            safetyTips: [
                "Use wall if needed",
                "Don't lean backward",
                "Control the movement"
            ],
            videoURL: "https://www.youtube.com/watch?v=zwoVcrdmLOE",
            steps: [
                .info("Stand tall", imageHint: "figure.stand"),
                .info("March in place", imageHint: "circle.fill"),
                .info("Lift knees to 90 degrees", imageHint: "circle.fill"),
                .info("Swing arms naturally", imageHint: "circle.fill"),
                .info("Continue for 1 minute", imageHint: "circle.fill"),
                .info("Rest and repeat", imageHint: "circle.fill")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Tandem Stance",
            category: .balance,
            difficulty: .intermediate,
            duration: 5,
            targetAreas: ["Balance", "Ankles"],
            instructions: [
                "Place one foot directly in front of the other",
                "Touch heel to toe",
                "Stand for 30 seconds",
                "Switch lead foot",
                "Repeat 3 times each"
            ],
            benefits: [
                "Challenges balance",
                "Improves ankle stability",
                "Enhances proprioception"
            ],
            safetyTips: [
                "Use counter for support",
                "Start near wall",
                "Progress slowly"
            ],
            videoURL: "https://www.youtube.com/watch?v=SGn89BYOicg",
            steps: [
                .info("Place one foot directly in front of the other", imageHint: "1.circle.fill"),
                .info("Touch heel to toe", imageHint: "circle.fill"),
                .info("Stand for 30 seconds", imageHint: "figure.stand"),
                .info("Switch lead foot", imageHint: "arrow.left.and.right"),
                .reps("Repeat 3 times each", repetitions: 3, imageHint: "repeat")
            ]
        ),
        Exercise(
            id: UUID(),
            name: "Side Leg Raises (Balance)",
            category: .balance,
            difficulty: .intermediate,
            duration: 6,
            targetAreas: ["Balance", "Hip Abductors"],
            instructions: [
                "Stand on left leg",
                "Lift right leg to side",
                "Hold for 5 seconds",
                "Lower slowly",
                "Repeat 10 times",
                "Switch legs"
            ],
            benefits: [
                "Improves single-leg balance",
                "Strengthens hip abductors",
                "Enhances stability"
            ],
            safetyTips: [
                "Use chair for support",
                "Keep torso upright",
                "Move slowly"
            ],
            videoURL: "https://www.youtube.com/watch?v=l_U2uoePtS4",
            steps: [
                .info("Stand on left leg", imageHint: "figure.stand"),
                .info("Lift right leg to side", imageHint: "circle.fill"),
                .timer("Hold for 5 seconds", duration: 5, imageHint: "timer"),
                .info("Lower slowly", imageHint: "circle.fill"),
                .reps("Repeat 10 times", repetitions: 10, imageHint: "repeat"),
                .info("Switch legs", imageHint: "arrow.left.and.right")
            ]
        )
    ]
}
