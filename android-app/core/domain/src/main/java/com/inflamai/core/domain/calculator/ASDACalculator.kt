package com.inflamai.core.domain.calculator

import kotlin.math.ln
import kotlin.math.roundToInt

/**
 * Ankylosing Spondylitis Disease Activity Score (ASDAS) Calculator
 *
 * CLINICALLY VALIDATED - DO NOT MODIFY FORMULA WITHOUT CLINICAL REVIEW
 *
 * ASDAS is an objective composite measure of disease activity that combines
 * patient-reported outcomes with an inflammatory marker (CRP or ESR).
 *
 * ASDAS-CRP Formula:
 * ASDAS = 0.12 × BackPain + 0.06 × Duration + 0.11 × PatientGlobal
 *         + 0.07 × PeripheralPain + 0.58 × Ln(CRP + 1)
 *
 * ASDAS-ESR Formula:
 * ASDAS = 0.08 × BackPain + 0.07 × Duration + 0.11 × PatientGlobal
 *         + 0.09 × PeripheralPain + 0.29 × √ESR
 *
 * References:
 * - Lukas C et al. Ann Rheum Dis 2009;68:18-24
 * - van der Heijde D et al. Ann Rheum Dis 2009;68:1811-8
 */
object ASDACalculator {

    // Formula coefficients for ASDAS-CRP
    private const val COEF_BACK_PAIN_CRP = 0.12
    private const val COEF_DURATION_CRP = 0.06
    private const val COEF_PATIENT_GLOBAL_CRP = 0.11
    private const val COEF_PERIPHERAL_PAIN_CRP = 0.07
    private const val COEF_CRP_LN = 0.58

    // Formula coefficients for ASDAS-ESR
    private const val COEF_BACK_PAIN_ESR = 0.08
    private const val COEF_DURATION_ESR = 0.07
    private const val COEF_PATIENT_GLOBAL_ESR = 0.11
    private const val COEF_PERIPHERAL_PAIN_ESR = 0.09
    private const val COEF_ESR_SQRT = 0.29

    /**
     * Calculate ASDAS-CRP score
     *
     * @param backPain Back pain score (0-10 VAS/NRS)
     * @param morningStiffnessDuration Duration of morning stiffness (0-10 scale)
     * @param patientGlobalAssessment Patient global assessment of disease activity (0-10)
     * @param peripheralPain Peripheral joint pain/swelling (0-10)
     * @param crpMgL C-Reactive Protein in mg/L
     * @return ASDAS-CRP score
     */
    fun calculateWithCRP(
        backPain: Double,
        morningStiffnessDuration: Double,
        patientGlobalAssessment: Double,
        peripheralPain: Double,
        crpMgL: Double
    ): Double {
        require(backPain in 0.0..10.0) { "Back pain must be between 0 and 10" }
        require(morningStiffnessDuration in 0.0..10.0) { "Morning stiffness duration must be between 0 and 10" }
        require(patientGlobalAssessment in 0.0..10.0) { "Patient global must be between 0 and 10" }
        require(peripheralPain in 0.0..10.0) { "Peripheral pain must be between 0 and 10" }
        require(crpMgL >= 0) { "CRP cannot be negative" }

        val asdas = (COEF_BACK_PAIN_CRP * backPain) +
                    (COEF_DURATION_CRP * morningStiffnessDuration) +
                    (COEF_PATIENT_GLOBAL_CRP * patientGlobalAssessment) +
                    (COEF_PERIPHERAL_PAIN_CRP * peripheralPain) +
                    (COEF_CRP_LN * ln(crpMgL + 1))

        return (asdas * 100).roundToInt() / 100.0 // Round to 2 decimal places
    }

    /**
     * Calculate ASDAS-ESR score
     *
     * @param backPain Back pain score (0-10 VAS/NRS)
     * @param morningStiffnessDuration Duration of morning stiffness (0-10 scale)
     * @param patientGlobalAssessment Patient global assessment of disease activity (0-10)
     * @param peripheralPain Peripheral joint pain/swelling (0-10)
     * @param esrMmHr Erythrocyte Sedimentation Rate in mm/hr
     * @return ASDAS-ESR score
     */
    fun calculateWithESR(
        backPain: Double,
        morningStiffnessDuration: Double,
        patientGlobalAssessment: Double,
        peripheralPain: Double,
        esrMmHr: Double
    ): Double {
        require(backPain in 0.0..10.0) { "Back pain must be between 0 and 10" }
        require(morningStiffnessDuration in 0.0..10.0) { "Morning stiffness duration must be between 0 and 10" }
        require(patientGlobalAssessment in 0.0..10.0) { "Patient global must be between 0 and 10" }
        require(peripheralPain in 0.0..10.0) { "Peripheral pain must be between 0 and 10" }
        require(esrMmHr >= 0) { "ESR cannot be negative" }

        val asdas = (COEF_BACK_PAIN_ESR * backPain) +
                    (COEF_DURATION_ESR * morningStiffnessDuration) +
                    (COEF_PATIENT_GLOBAL_ESR * patientGlobalAssessment) +
                    (COEF_PERIPHERAL_PAIN_ESR * peripheralPain) +
                    (COEF_ESR_SQRT * kotlin.math.sqrt(esrMmHr))

        return (asdas * 100).roundToInt() / 100.0 // Round to 2 decimal places
    }

    /**
     * Interpret ASDAS score into clinical activity level
     */
    fun interpret(score: Double): ASDAInterpretation {
        return when {
            score < 0 -> throw IllegalArgumentException("Score cannot be negative")
            score < 1.3 -> ASDAInterpretation.INACTIVE
            score < 2.1 -> ASDAInterpretation.MODERATE_ACTIVITY
            score < 3.5 -> ASDAInterpretation.HIGH_ACTIVITY
            else -> ASDAInterpretation.VERY_HIGH_ACTIVITY
        }
    }

    /**
     * Check if a change in ASDAS represents clinically important improvement
     *
     * ASDAS cutoffs for response:
     * - Clinically important improvement: ≥1.1 unit decrease
     * - Major improvement: ≥2.0 unit decrease
     */
    fun assessChange(previousScore: Double, currentScore: Double): ASDAChange {
        val delta = previousScore - currentScore  // Positive = improvement
        return when {
            delta >= 2.0 -> ASDAChange.MAJOR_IMPROVEMENT
            delta >= 1.1 -> ASDAChange.CLINICALLY_IMPORTANT_IMPROVEMENT
            delta > 0 -> ASDAChange.MINOR_IMPROVEMENT
            delta >= -1.1 -> ASDAChange.STABLE_OR_MINOR_WORSENING
            delta >= -2.0 -> ASDAChange.CLINICALLY_IMPORTANT_WORSENING
            else -> ASDAChange.MAJOR_WORSENING
        }
    }

    /**
     * Calculate the change needed to achieve inactive disease
     *
     * @param currentScore Current ASDAS score
     * @return Points needed to decrease to reach inactive disease (<1.3)
     */
    fun pointsToInactiveDisease(currentScore: Double): Double {
        return if (currentScore >= 1.3) {
            (currentScore - 1.29).coerceAtLeast(0.0)
        } else {
            0.0
        }
    }

    /**
     * Run validation tests to verify calculator accuracy
     */
    fun runValidationTests(): Boolean {
        // Test case: Standard values with CRP
        val crpResult = calculateWithCRP(
            backPain = 5.0,
            morningStiffnessDuration = 5.0,
            patientGlobalAssessment = 5.0,
            peripheralPain = 5.0,
            crpMgL = 10.0
        )
        // Expected: 0.12*5 + 0.06*5 + 0.11*5 + 0.07*5 + 0.58*ln(11) ≈ 3.19
        val crpExpected = 3.19
        val crpValid = kotlin.math.abs(crpResult - crpExpected) < 0.05

        // Test case: Low disease activity
        val lowResult = calculateWithCRP(
            backPain = 1.0,
            morningStiffnessDuration = 1.0,
            patientGlobalAssessment = 1.0,
            peripheralPain = 1.0,
            crpMgL = 2.0
        )
        val lowValid = interpret(lowResult) == ASDAInterpretation.INACTIVE ||
                       interpret(lowResult) == ASDAInterpretation.MODERATE_ACTIVITY

        return crpValid && lowValid
    }
}

/**
 * ASDAS interpretation levels based on cutoff values
 */
enum class ASDAInterpretation(
    val displayName: String,
    val description: String,
    val cutoffDescription: String,
    val colorCode: String
) {
    INACTIVE(
        "Inactive Disease",
        "Disease activity is well controlled. Excellent response to treatment.",
        "ASDAS < 1.3",
        "#4CAF50"  // Green
    ),
    MODERATE_ACTIVITY(
        "Moderate Activity",
        "Moderate disease activity. Treatment is partially effective.",
        "1.3 ≤ ASDAS < 2.1",
        "#FF9800"  // Orange
    ),
    HIGH_ACTIVITY(
        "High Activity",
        "High disease activity. Consider treatment optimization.",
        "2.1 ≤ ASDAS < 3.5",
        "#FF5722"  // Deep Orange
    ),
    VERY_HIGH_ACTIVITY(
        "Very High Activity",
        "Very high disease activity. Urgent treatment review recommended.",
        "ASDAS ≥ 3.5",
        "#F44336"  // Red
    )
}

/**
 * Classification of ASDAS score changes
 */
enum class ASDAChange(
    val displayName: String,
    val description: String
) {
    MAJOR_IMPROVEMENT(
        "Major Improvement",
        "Decrease of ≥2.0 units - excellent response to treatment"
    ),
    CLINICALLY_IMPORTANT_IMPROVEMENT(
        "Clinically Important Improvement",
        "Decrease of ≥1.1 units - meaningful response to treatment"
    ),
    MINOR_IMPROVEMENT(
        "Minor Improvement",
        "Small decrease - may not be clinically significant"
    ),
    STABLE_OR_MINOR_WORSENING(
        "Stable/Minor Worsening",
        "No significant change in disease activity"
    ),
    CLINICALLY_IMPORTANT_WORSENING(
        "Clinically Important Worsening",
        "Increase of ≥1.1 units - consider treatment adjustment"
    ),
    MAJOR_WORSENING(
        "Major Worsening",
        "Increase of ≥2.0 units - urgent treatment review needed"
    )
}
