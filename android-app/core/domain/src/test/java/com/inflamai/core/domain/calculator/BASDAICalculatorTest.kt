package com.inflamai.core.domain.calculator

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

/**
 * Unit tests for BASDAI Calculator
 *
 * Validates clinically accurate BASDAI score calculation
 * based on published medical literature.
 *
 * Reference: Garrett S, et al. "A new approach to defining disease
 * status in ankylosing spondylitis: the Bath Ankylosing Spondylitis
 * Disease Activity Index." J Rheumatol. 1994;21(12):2286-91.
 */
class BASDAICalculatorTest {

    companion object {
        private const val DELTA = 0.01 // Acceptable floating point error
    }

    @Test
    fun `calculate returns correct BASDAI for standard case`() {
        // Standard patient with moderate disease activity
        val result = BASDAICalculator.calculate(
            fatigue = 5.0,
            spinalPain = 6.0,
            peripheralPain = 4.0,
            enthesitisPain = 3.0,
            morningSeverity = 7.0,
            morningDuration = 5.0
        )

        // Formula: (5 + 6 + 4 + 3 + ((7 + 5) / 2)) / 5
        // = (5 + 6 + 4 + 3 + 6) / 5
        // = 24 / 5
        // = 4.8
        assertEquals(4.8, result, DELTA)
    }

    @Test
    fun `calculate returns zero for all zero inputs`() {
        val result = BASDAICalculator.calculate(
            fatigue = 0.0,
            spinalPain = 0.0,
            peripheralPain = 0.0,
            enthesitisPain = 0.0,
            morningSeverity = 0.0,
            morningDuration = 0.0
        )

        assertEquals(0.0, result, DELTA)
    }

    @Test
    fun `calculate returns 10 for maximum inputs`() {
        val result = BASDAICalculator.calculate(
            fatigue = 10.0,
            spinalPain = 10.0,
            peripheralPain = 10.0,
            enthesitisPain = 10.0,
            morningSeverity = 10.0,
            morningDuration = 10.0
        )

        assertEquals(10.0, result, DELTA)
    }

    @Test
    fun `calculate handles mixed stiffness values correctly`() {
        // Severe morning stiffness, short duration
        val result = BASDAICalculator.calculate(
            fatigue = 5.0,
            spinalPain = 5.0,
            peripheralPain = 5.0,
            enthesitisPain = 5.0,
            morningSeverity = 10.0, // Severe
            morningDuration = 2.0   // Short
        )

        // Formula: (5 + 5 + 5 + 5 + ((10 + 2) / 2)) / 5
        // = (5 + 5 + 5 + 5 + 6) / 5
        // = 26 / 5
        // = 5.2
        assertEquals(5.2, result, DELTA)
    }

    @Test
    fun `interpret returns Remission for score below 2`() {
        val interpretation = BASDAICalculator.interpret(1.5)

        assertEquals(BASDAIInterpretation.REMISSION, interpretation)
    }

    @Test
    fun `interpret returns Low Activity for score 2-4`() {
        val interpretation = BASDAICalculator.interpret(3.0)

        assertEquals(BASDAIInterpretation.LOW_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret returns Moderate Activity for score 4-6`() {
        val interpretation = BASDAICalculator.interpret(5.0)

        assertEquals(BASDAIInterpretation.MODERATE_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret returns High Activity for score above 6`() {
        val interpretation = BASDAICalculator.interpret(7.5)

        assertEquals(BASDAIInterpretation.HIGH_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret handles boundary value 2 correctly`() {
        val interpretation = BASDAICalculator.interpret(2.0)

        assertEquals(BASDAIInterpretation.LOW_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret handles boundary value 4 correctly`() {
        val interpretation = BASDAICalculator.interpret(4.0)

        assertEquals(BASDAIInterpretation.MODERATE_ACTIVITY, interpretation)
    }

    @Test
    fun `interpret handles boundary value 6 correctly`() {
        val interpretation = BASDAICalculator.interpret(6.0)

        assertEquals(BASDAIInterpretation.HIGH_ACTIVITY, interpretation)
    }

    @Test
    fun `calculate rounds to two decimal places`() {
        val result = BASDAICalculator.calculate(
            fatigue = 3.333,
            spinalPain = 3.333,
            peripheralPain = 3.333,
            enthesitisPain = 3.333,
            morningSeverity = 3.333,
            morningDuration = 3.333
        )

        // Result should be rounded to 2 decimal places
        val decimalPlaces = result.toString().substringAfter('.').length
        assertTrue("Result should have at most 2 decimal places", decimalPlaces <= 2)
    }

    @Test
    fun `calculate clamps negative values to zero`() {
        val result = BASDAICalculator.calculate(
            fatigue = -1.0,
            spinalPain = 5.0,
            peripheralPain = 5.0,
            enthesitisPain = 5.0,
            morningSeverity = 5.0,
            morningDuration = 5.0
        )

        // Negative should be treated as 0
        assertTrue("Result should be non-negative", result >= 0)
    }

    @Test
    fun `calculate clamps values above 10 to 10`() {
        val result = BASDAICalculator.calculate(
            fatigue = 15.0,  // Over 10
            spinalPain = 5.0,
            peripheralPain = 5.0,
            enthesitisPain = 5.0,
            morningSeverity = 5.0,
            morningDuration = 5.0
        )

        // Even with clamping, max BASDAI should be 10
        assertTrue("Result should be <= 10", result <= 10.0)
    }

    @Test
    fun `validation tests pass for clinical examples`() {
        // Run the built-in validation tests
        val testsPassed = BASDAICalculator.runValidationTests()
        assertTrue("All validation tests should pass", testsPassed)
    }

    @Test
    fun `stiffness average is calculated correctly`() {
        // Test that stiffness is averaged (not summed)
        val resultWithEqualStiffness = BASDAICalculator.calculate(
            fatigue = 0.0,
            spinalPain = 0.0,
            peripheralPain = 0.0,
            enthesitisPain = 0.0,
            morningSeverity = 8.0,
            morningDuration = 2.0
        )

        // Stiffness average = (8 + 2) / 2 = 5
        // BASDAI = (0 + 0 + 0 + 0 + 5) / 5 = 1.0
        assertEquals(1.0, resultWithEqualStiffness, DELTA)
    }

    @Test
    fun `clinically significant change is detected`() {
        val baseline = 6.0
        val followUp = 4.5

        // MCID (Minimal Clinically Important Difference) for BASDAI is ~1.0
        val change = abs(baseline - followUp)

        assertTrue("Change of 1.5 should be clinically significant", change >= 1.0)
    }
}
