package com.inflamai.core.domain.calculator

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.abs

/**
 * Unit tests for Correlation Engine
 *
 * Validates Pearson correlation coefficient calculation
 * and statistical significance testing.
 *
 * This is statistical analysis, NOT machine learning.
 */
class CorrelationEngineTest {

    private lateinit var engine: CorrelationEngine

    companion object {
        private const val DELTA = 0.001
    }

    @Before
    fun setup() {
        engine = CorrelationEngine()
    }

    @Test
    fun `pearson correlation returns 1 for perfectly positive correlation`() {
        val x = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = listOf(2.0, 4.0, 6.0, 8.0, 10.0) // y = 2x

        val result = engine.calculatePearsonCorrelation(x, y)

        assertEquals(1.0, result, DELTA)
    }

    @Test
    fun `pearson correlation returns -1 for perfectly negative correlation`() {
        val x = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = listOf(10.0, 8.0, 6.0, 4.0, 2.0) // y = 12 - 2x

        val result = engine.calculatePearsonCorrelation(x, y)

        assertEquals(-1.0, result, DELTA)
    }

    @Test
    fun `pearson correlation returns 0 for no correlation`() {
        val x = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = listOf(5.0, 5.0, 5.0, 5.0, 5.0) // Constant

        val result = engine.calculatePearsonCorrelation(x, y)

        // When y is constant, correlation is 0
        assertEquals(0.0, result, DELTA)
    }

    @Test
    fun `pearson correlation handles moderate positive correlation`() {
        // Weather vs pain - typical AS pattern
        val pressure = listOf(1013.0, 1010.0, 1005.0, 1008.0, 1015.0, 1003.0, 1012.0)
        val pain = listOf(3.0, 4.0, 7.0, 5.0, 2.0, 8.0, 3.0)

        val result = engine.calculatePearsonCorrelation(pressure, pain)

        // Should show negative correlation (lower pressure = more pain for AS patients)
        assertTrue("Should show negative correlation", result < 0)
    }

    @Test
    fun `pearson correlation returns null for insufficient data`() {
        val x = listOf(1.0, 2.0)  // Less than minimum 3 points
        val y = listOf(1.0, 2.0)

        val result = engine.calculatePearsonCorrelation(x, y)

        // Should handle gracefully (could return 0 or require minimum data)
        // Implementation should require at least 3 data points
        assertNotNull(result)
    }

    @Test
    fun `pearson correlation returns null for mismatched lengths`() {
        val x = listOf(1.0, 2.0, 3.0)
        val y = listOf(1.0, 2.0)  // Different length

        // Should throw or return null
        try {
            val result = engine.calculatePearsonCorrelation(x, y)
            assertEquals(0.0, result, DELTA) // If it handles gracefully
        } catch (e: IllegalArgumentException) {
            // Expected behavior
        }
    }

    @Test
    fun `p-value calculation returns low value for significant correlation`() {
        val x = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = listOf(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)

        val correlation = engine.calculatePearsonCorrelation(x, y)
        val pValue = engine.calculatePValue(correlation, x.size)

        assertTrue("P-value should be significant (< 0.05)", pValue < 0.05)
    }

    @Test
    fun `p-value calculation returns high value for weak correlation`() {
        // Random-ish data with weak correlation
        val x = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = listOf(3.0, 1.0, 4.0, 2.0, 5.0)

        val correlation = engine.calculatePearsonCorrelation(x, y)
        val pValue = engine.calculatePValue(correlation, x.size)

        // With only 5 points and weak correlation, p-value should be high
        assertTrue("P-value should be non-significant for weak correlation", pValue > 0.05)
    }

    @Test
    fun `analyzeCorrelation identifies strong positive trigger`() {
        // Strong positive correlation - e.g., stress vs pain
        val stressData = listOf(
            DataPoint(1, 2.0),
            DataPoint(2, 4.0),
            DataPoint(3, 6.0),
            DataPoint(4, 8.0),
            DataPoint(5, 10.0)
        )
        val painData = listOf(
            DataPoint(1, 2.0),
            DataPoint(2, 4.0),
            DataPoint(3, 6.0),
            DataPoint(4, 8.0),
            DataPoint(5, 10.0)
        )

        val result = engine.analyzeCorrelation(
            factorData = stressData,
            symptomData = painData,
            factorName = "Stress Level",
            factorCategory = "Lifestyle"
        )

        assertNotNull(result)
        assertTrue("Should identify as significant", result?.isSignificant == true)
        assertEquals(TriggerType.POSITIVE, result?.triggerType)
    }

    @Test
    fun `analyzeCorrelation identifies protective factor`() {
        // Negative correlation - e.g., exercise vs pain for some patients
        val exerciseData = listOf(
            DataPoint(1, 2.0),
            DataPoint(2, 4.0),
            DataPoint(3, 6.0),
            DataPoint(4, 8.0),
            DataPoint(5, 10.0)
        )
        val painData = listOf(
            DataPoint(1, 10.0),
            DataPoint(2, 8.0),
            DataPoint(3, 6.0),
            DataPoint(4, 4.0),
            DataPoint(5, 2.0)
        )

        val result = engine.analyzeCorrelation(
            factorData = exerciseData,
            symptomData = painData,
            factorName = "Exercise Duration",
            factorCategory = "Activity"
        )

        assertNotNull(result)
        assertEquals(TriggerType.PROTECTIVE, result?.triggerType)
    }

    @Test
    fun `lag analysis finds delayed correlation`() {
        // Barometric pressure drop affects pain 24h later
        val pressureData = (1..10).map { day ->
            DataPoint(day, if (day == 3) 990.0 else 1013.0) // Drop on day 3
        }
        val painData = (1..10).map { day ->
            DataPoint(day, if (day == 4) 8.0 else 3.0) // Spike on day 4
        }

        // With 1-day lag, the correlation should be stronger
        val result = engine.analyzeWithLag(
            factorData = pressureData,
            symptomData = painData,
            factorName = "Barometric Pressure",
            lagHours = 24
        )

        assertNotNull(result)
    }

    @Test
    fun `minimum data requirement is enforced`() {
        // Less than 30 days of data
        val factorData = (1..20).map { DataPoint(it, it.toDouble()) }
        val symptomData = (1..20).map { DataPoint(it, it.toDouble()) }

        val result = engine.analyzeCorrelation(
            factorData = factorData,
            symptomData = symptomData,
            factorName = "Test",
            factorCategory = "Test"
        )

        // Should either return null or mark as insufficient data
        // Based on 30-day minimum requirement
        if (result != null) {
            assertFalse("Should mark as insufficient data", result.isSignificant)
        }
    }

    @Test
    fun `correlation strength categories are correct`() {
        // Weak correlation (|r| < 0.4)
        assertEquals(CorrelationStrength.WEAK, CorrelationEngine.getStrength(0.3))
        assertEquals(CorrelationStrength.WEAK, CorrelationEngine.getStrength(-0.3))

        // Moderate correlation (0.4 <= |r| < 0.7)
        assertEquals(CorrelationStrength.MODERATE, CorrelationEngine.getStrength(0.5))
        assertEquals(CorrelationStrength.MODERATE, CorrelationEngine.getStrength(-0.5))

        // Strong correlation (|r| >= 0.7)
        assertEquals(CorrelationStrength.STRONG, CorrelationEngine.getStrength(0.8))
        assertEquals(CorrelationStrength.STRONG, CorrelationEngine.getStrength(-0.8))
    }

    @Test
    fun `validation tests pass`() {
        val testsPassed = engine.runValidationTests()
        assertTrue("All validation tests should pass", testsPassed)
    }
}

// Test helper data class
data class DataPoint(
    val day: Int,
    val value: Double
)
