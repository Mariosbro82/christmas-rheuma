package com.inflamai.core.domain.calculator

import kotlin.math.roundToInt

/**
 * Bath Ankylosing Spondylitis Disease Activity Index (BASDAI) Calculator
 *
 * CLINICALLY VALIDATED - DO NOT MODIFY FORMULA WITHOUT CLINICAL REVIEW
 *
 * The BASDAI is a validated self-assessment tool used to determine disease activity
 * in patients with Ankylosing Spondylitis. It consists of 6 questions, each scored
 * from 0-10, measuring:
 * - Q1: Fatigue
 * - Q2: Spinal pain
 * - Q3: Peripheral joint pain/swelling
 * - Q4: Localized tenderness (enthesitis)
 * - Q5: Severity of morning stiffness
 * - Q6: Duration of morning stiffness (scaled)
 *
 * Formula: BASDAI = (Q1 + Q2 + Q3 + Q4 + ((Q5 + Q6) / 2)) / 5
 *
 * Reference: Garrett S et al. J Rheumatol 1994;21:2286-91
 */
object BASDAICalculator {

    /**
     * Calculate BASDAI score from 6 questions
     *
     * @param fatigue Q1: Overall fatigue (0-10)
     * @param spinalPain Q2: Spinal pain (0-10)
     * @param peripheralPain Q3: Peripheral joint pain/swelling (0-10)
     * @param enthesitisPain Q4: Areas of localized tenderness/enthesitis (0-10)
     * @param morningSeverity Q5: Severity of morning stiffness (0-10)
     * @param morningDuration Q6: Duration of morning stiffness (0-10 scaled from minutes)
     * @return BASDAI score (0-10)
     */
    fun calculate(
        fatigue: Double,
        spinalPain: Double,
        peripheralPain: Double,
        enthesitisPain: Double,
        morningSeverity: Double,
        morningDuration: Double
    ): Double {
        require(fatigue in 0.0..10.0) { "Fatigue must be between 0 and 10" }
        require(spinalPain in 0.0..10.0) { "Spinal pain must be between 0 and 10" }
        require(peripheralPain in 0.0..10.0) { "Peripheral pain must be between 0 and 10" }
        require(enthesitisPain in 0.0..10.0) { "Enthesitis pain must be between 0 and 10" }
        require(morningSeverity in 0.0..10.0) { "Morning severity must be between 0 and 10" }
        require(morningDuration in 0.0..10.0) { "Morning duration must be between 0 and 10" }

        // BASDAI Formula: (Q1 + Q2 + Q3 + Q4 + ((Q5 + Q6) / 2)) / 5
        val stiffnessAverage = (morningSeverity + morningDuration) / 2.0
        val basdai = (fatigue + spinalPain + peripheralPain + enthesitisPain + stiffnessAverage) / 5.0

        return (basdai * 100).roundToInt() / 100.0 // Round to 2 decimal places
    }

    /**
     * Calculate BASDAI with morning stiffness in minutes
     *
     * @param fatigue Q1: Overall fatigue (0-10)
     * @param spinalPain Q2: Spinal pain (0-10)
     * @param peripheralPain Q3: Peripheral joint pain/swelling (0-10)
     * @param enthesitisPain Q4: Areas of localized tenderness/enthesitis (0-10)
     * @param morningSeverity Q5: Severity of morning stiffness (0-10)
     * @param morningDurationMinutes Q6: Duration of morning stiffness in minutes (0-120+)
     * @return BASDAI score (0-10)
     */
    fun calculateWithMinutes(
        fatigue: Double,
        spinalPain: Double,
        peripheralPain: Double,
        enthesitisPain: Double,
        morningSeverity: Double,
        morningDurationMinutes: Int
    ): Double {
        val scaledDuration = scaleMorningStiffnessDuration(morningDurationMinutes)
        return calculate(fatigue, spinalPain, peripheralPain, enthesitisPain, morningSeverity, scaledDuration)
    }

    /**
     * Scale morning stiffness duration from minutes to 0-10 scale
     *
     * Scale: 0 min = 0, 120+ min = 10
     * Linear interpolation between 0 and 120 minutes
     */
    fun scaleMorningStiffnessDuration(minutes: Int): Double {
        return when {
            minutes <= 0 -> 0.0
            minutes >= 120 -> 10.0
            else -> (minutes / 12.0).coerceIn(0.0, 10.0)
        }
    }

    /**
     * Interpret BASDAI score into clinical activity level
     */
    fun interpret(score: Double): BASDAIInterpretation {
        return when {
            score < 0 -> throw IllegalArgumentException("Score cannot be negative")
            score <= 2.0 -> BASDAIInterpretation.REMISSION
            score <= 4.0 -> BASDAIInterpretation.LOW_ACTIVITY
            score <= 6.0 -> BASDAIInterpretation.MODERATE_ACTIVITY
            score <= 8.0 -> BASDAIInterpretation.HIGH_ACTIVITY
            score <= 10.0 -> BASDAIInterpretation.VERY_HIGH_ACTIVITY
            else -> throw IllegalArgumentException("Score cannot exceed 10")
        }
    }

    /**
     * Check if a change in BASDAI is clinically significant
     *
     * A change of ≥1.0 is generally considered clinically meaningful
     * A change of ≥2.0 is considered a major improvement/worsening
     */
    fun isSignificantChange(previousScore: Double, currentScore: Double): ChangeSignificance {
        val delta = currentScore - previousScore
        return when {
            delta <= -2.0 -> ChangeSignificance.MAJOR_IMPROVEMENT
            delta <= -1.0 -> ChangeSignificance.CLINICALLY_MEANINGFUL_IMPROVEMENT
            delta < 1.0 -> ChangeSignificance.NO_SIGNIFICANT_CHANGE
            delta < 2.0 -> ChangeSignificance.CLINICALLY_MEANINGFUL_WORSENING
            else -> ChangeSignificance.MAJOR_WORSENING
        }
    }

    /**
     * Run validation tests to verify calculator accuracy
     * Called in DEBUG builds to ensure clinical accuracy
     */
    fun runValidationTests(): Boolean {
        val tests = listOf(
            // Test case 1: All zeros = 0
            ValidationTest(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, expected = 0.0),
            // Test case 2: All 10s = 10
            ValidationTest(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, expected = 10.0),
            // Test case 3: All 5s = 5
            ValidationTest(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, expected = 5.0),
            // Test case 4: Mixed values
            ValidationTest(4.0, 6.0, 2.0, 8.0, 5.0, 7.0, expected = 5.2),
            // Test case 5: Edge case - stiffness questions
            ValidationTest(5.0, 5.0, 5.0, 5.0, 0.0, 10.0, expected = 5.0)
        )

        return tests.all { test ->
            val result = calculate(
                test.fatigue, test.spinalPain, test.peripheralPain,
                test.enthesitisPain, test.morningSeverity, test.morningDuration
            )
            kotlin.math.abs(result - test.expected) < 0.01
        }
    }

    private data class ValidationTest(
        val fatigue: Double,
        val spinalPain: Double,
        val peripheralPain: Double,
        val enthesitisPain: Double,
        val morningSeverity: Double,
        val morningDuration: Double,
        val expected: Double
    )
}

/**
 * BASDAI interpretation levels
 */
enum class BASDAIInterpretation(
    val displayName: String,
    val description: String,
    val colorCode: String
) {
    REMISSION(
        "Remission",
        "Disease activity is well controlled. Continue current management.",
        "#4CAF50"  // Green
    ),
    LOW_ACTIVITY(
        "Low Activity",
        "Mild disease activity. Monitor and maintain treatment.",
        "#8BC34A"  // Light Green
    ),
    MODERATE_ACTIVITY(
        "Moderate Activity",
        "Moderate disease activity. Consider treatment optimization.",
        "#FF9800"  // Orange
    ),
    HIGH_ACTIVITY(
        "High Activity",
        "High disease activity. Treatment adjustment may be needed.",
        "#FF5722"  // Deep Orange
    ),
    VERY_HIGH_ACTIVITY(
        "Very High Activity",
        "Severe disease activity. Urgent treatment review recommended.",
        "#F44336"  // Red
    )
}

/**
 * Classification of BASDAI score changes
 */
enum class ChangeSignificance {
    MAJOR_IMPROVEMENT,
    CLINICALLY_MEANINGFUL_IMPROVEMENT,
    NO_SIGNIFICANT_CHANGE,
    CLINICALLY_MEANINGFUL_WORSENING,
    MAJOR_WORSENING
}
