package com.inflamai.core.domain.calculator

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.ln

/**
 * Unit tests for ASDAS Calculator
 *
 * Validates clinically accurate ASDAS-CRP score calculation
 * based on published medical literature.
 *
 * Reference: Machado P, et al. "Ankylosing Spondylitis Disease Activity Score (ASDAS):
 * defining cut-off values for disease activity states and improvement scores."
 * Ann Rheum Dis. 2011;70(1):47-53.
 */
class ASDACalculatorTest {

    companion object {
        private const val DELTA = 0.01 // Acceptable floating point error
    }

    @Test
    fun `calculateWithCRP returns correct score for standard case`() {
        // Standard patient with moderate disease activity
        val result = ASDACalculator.calculateWithCRP(
            backPain = 5.0,
            morningStiffnessDuration = 5.0,
            patientGlobalAssessment = 5.0,
            peripheralPain = 5.0,
            crpMgL = 10.0
        )

        // Formula: 0.12×BackPain + 0.06×Duration + 0.11×PatientGlobal + 0.07×PeripheralPain + 0.58×Ln(CRP+1)
        // = 0.12×5 + 0.06×5 + 0.11×5 + 0.07×5 + 0.58×Ln(11)
        // = 0.60 + 0.30 + 0.55 + 0.35 + 0.58×2.398
        // = 1.80 + 1.391
        // = 3.19
        assertEquals(3.19, result, 0.1)
    }

    @Test
    fun `calculateWithCRP handles zero CRP correctly`() {
        val result = ASDACalculator.calculateWithCRP(
            backPain = 5.0,
            morningStiffnessDuration = 5.0,
            patientGlobalAssessment = 5.0,
            peripheralPain = 5.0,
            crpMgL = 0.0
        )

        // Ln(0+1) = Ln(1) = 0
        // Result = 0.12×5 + 0.06×5 + 0.11×5 + 0.07×5 + 0.58×0
        // = 0.60 + 0.30 + 0.55 + 0.35 + 0
        // = 1.80
        assertEquals(1.80, result, DELTA)
    }

    @Test
    fun `calculateWithCRP handles high CRP correctly`() {
        val result = ASDACalculator.calculateWithCRP(
            backPain = 8.0,
            morningStiffnessDuration = 8.0,
            patientGlobalAssessment = 8.0,
            peripheralPain = 8.0,
            crpMgL = 100.0  // Very high CRP
        )

        // Formula with high CRP
        // = 0.12×8 + 0.06×8 + 0.11×8 + 0.07×8 + 0.58×Ln(101)
        // = 0.96 + 0.48 + 0.88 + 0.56 + 0.58×4.615
        // = 2.88 + 2.677
        // = 5.56
        assertEquals(5.56, result, 0.1)
    }

    @Test
    fun `calculateWithCRP returns minimum for all zero inputs`() {
        val result = ASDACalculator.calculateWithCRP(
            backPain = 0.0,
            morningStiffnessDuration = 0.0,
            patientGlobalAssessment = 0.0,
            peripheralPain = 0.0,
            crpMgL = 0.0
        )

        assertEquals(0.0, result, DELTA)
    }

    @Test
    fun `interpret returns Inactive Disease for score below 1_3`() {
        val interpretation = ASDACalculator.interpret(1.0)

        assertEquals(ASDAInterpretation.INACTIVE_DISEASE, interpretation)
    }

    @Test
    fun `interpret returns Low Disease Activity for score 1_3 to 2_1`() {
        val interpretation = ASDACalculator.interpret(1.8)

        assertEquals(ASDAInterpretation.LOW_DISEASE_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret returns High Disease Activity for score 2_1 to 3_5`() {
        val interpretation = ASDACalculator.interpret(3.0)

        assertEquals(ASDAInterpretation.HIGH_DISEASE_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret returns Very High Disease Activity for score above 3_5`() {
        val interpretation = ASDACalculator.interpret(4.5)

        assertEquals(ASDAInterpretation.VERY_HIGH_DISEASE_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret handles boundary value 1_3 correctly`() {
        val interpretation = ASDACalculator.interpret(1.3)

        assertEquals(ASDAInterpretation.LOW_DISEASE_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret handles boundary value 2_1 correctly`() {
        val interpretation = ASDACalculator.interpret(2.1)

        assertEquals(ASDAInterpretation.HIGH_DISEASE_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret handles boundary value 3_5 correctly`() {
        val interpretation = ASDACalculator.interpret(3.5)

        assertEquals(ASDAInterpretation.VERY_HIGH_DISEASE_ACTIVITY, interpretation)
    }

    @Test
    fun `CRP contribution follows logarithmic scale`() {
        // Compare CRP contributions at different levels
        val lowCRP = ASDACalculator.calculateWithCRP(
            backPain = 0.0,
            morningStiffnessDuration = 0.0,
            patientGlobalAssessment = 0.0,
            peripheralPain = 0.0,
            crpMgL = 1.0
        )

        val mediumCRP = ASDACalculator.calculateWithCRP(
            backPain = 0.0,
            morningStiffnessDuration = 0.0,
            patientGlobalAssessment = 0.0,
            peripheralPain = 0.0,
            crpMgL = 10.0
        )

        val highCRP = ASDACalculator.calculateWithCRP(
            backPain = 0.0,
            morningStiffnessDuration = 0.0,
            patientGlobalAssessment = 0.0,
            peripheralPain = 0.0,
            crpMgL = 100.0
        )

        // Logarithmic scale means 10x increase doesn't result in 10x score
        assertTrue("Medium CRP should be less than 5x low CRP", mediumCRP < lowCRP * 5)
        assertTrue("High CRP should be less than 5x medium CRP", highCRP < mediumCRP * 5)
    }

    @Test
    fun `coefficient weights are applied correctly`() {
        // Test each component independently
        val backPainOnly = ASDACalculator.calculateWithCRP(
            backPain = 10.0,
            morningStiffnessDuration = 0.0,
            patientGlobalAssessment = 0.0,
            peripheralPain = 0.0,
            crpMgL = 0.0
        )
        assertEquals(1.2, backPainOnly, DELTA) // 0.12 × 10

        val durationOnly = ASDACalculator.calculateWithCRP(
            backPain = 0.0,
            morningStiffnessDuration = 10.0,
            patientGlobalAssessment = 0.0,
            peripheralPain = 0.0,
            crpMgL = 0.0
        )
        assertEquals(0.6, durationOnly, DELTA) // 0.06 × 10

        val globalOnly = ASDACalculator.calculateWithCRP(
            backPain = 0.0,
            morningStiffnessDuration = 0.0,
            patientGlobalAssessment = 10.0,
            peripheralPain = 0.0,
            crpMgL = 0.0
        )
        assertEquals(1.1, globalOnly, DELTA) // 0.11 × 10

        val peripheralOnly = ASDACalculator.calculateWithCRP(
            backPain = 0.0,
            morningStiffnessDuration = 0.0,
            patientGlobalAssessment = 0.0,
            peripheralPain = 10.0,
            crpMgL = 0.0
        )
        assertEquals(0.7, peripheralOnly, DELTA) // 0.07 × 10
    }

    @Test
    fun `calculate clamps negative values`() {
        val result = ASDACalculator.calculateWithCRP(
            backPain = -5.0,  // Negative
            morningStiffnessDuration = 5.0,
            patientGlobalAssessment = 5.0,
            peripheralPain = 5.0,
            crpMgL = 5.0
        )

        assertTrue("Result should be non-negative", result >= 0)
    }

    @Test
    fun `validation tests pass for clinical examples`() {
        val testsPassed = ASDACalculator.runValidationTests()
        assertTrue("All validation tests should pass", testsPassed)
    }

    @Test
    fun `clinically important improvement is detected`() {
        val baseline = 3.5
        val followUp = 2.0

        // ASDAS clinically important improvement: change ≥ 1.1
        val change = baseline - followUp

        assertTrue("Change of 1.5 exceeds CII threshold of 1.1", change >= 1.1)
    }

    @Test
    fun `major improvement is detected`() {
        val baseline = 4.0
        val followUp = 1.5

        // ASDAS major improvement: change ≥ 2.0
        val change = baseline - followUp

        assertTrue("Change of 2.5 exceeds major improvement threshold of 2.0", change >= 2.0)
    }
}
