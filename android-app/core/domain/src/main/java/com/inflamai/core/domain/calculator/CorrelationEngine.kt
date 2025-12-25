package com.inflamai.core.domain.calculator

import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Statistical Correlation Engine for Pattern Analysis
 *
 * This is a STATISTICAL engine, NOT machine learning.
 * Uses Pearson correlation coefficient with lag analysis
 * to identify potential triggers for AS symptoms.
 *
 * Key Features:
 * - Pearson correlation coefficient calculation
 * - P-value computation for significance testing
 * - Lag analysis (0h, 12h, 24h offsets)
 * - Bonferroni correction for multiple comparisons
 * - Minimum 30 days of data required
 *
 * Medical Disclaimer: This analysis is for informational purposes only.
 * Patterns identified should be discussed with a healthcare provider.
 */
class CorrelationEngine {

    companion object {
        const val MINIMUM_DATA_POINTS = 30
        const val SIGNIFICANCE_THRESHOLD = 0.05
        const val CORRELATION_THRESHOLD = 0.4
        val LAG_HOURS = listOf(0, 12, 24)
    }

    /**
     * Analyze correlations between a factor and symptom data
     *
     * @param factorData List of (timestamp, value) pairs for the factor
     * @param symptomData List of (timestamp, value) pairs for symptoms
     * @param factorName Name of the factor being analyzed
     * @param factorCategory Category of the factor
     * @return CorrelationAnalysis result or null if insufficient data
     */
    fun analyzeCorrelation(
        factorData: List<Pair<Long, Double>>,
        symptomData: List<Pair<Long, Double>>,
        factorName: String,
        factorCategory: String
    ): CorrelationAnalysis? {
        if (factorData.size < MINIMUM_DATA_POINTS || symptomData.size < MINIMUM_DATA_POINTS) {
            return null
        }

        val results = mutableListOf<LaggedCorrelation>()

        for (lagHours in LAG_HOURS) {
            val lagMs = lagHours * 60 * 60 * 1000L

            // Align data with lag
            val alignedPairs = alignDataWithLag(factorData, symptomData, lagMs)

            if (alignedPairs.size < MINIMUM_DATA_POINTS) continue

            val factorValues = alignedPairs.map { it.first }
            val symptomValues = alignedPairs.map { it.second }

            val correlation = calculatePearsonCorrelation(factorValues, symptomValues)
            val pValue = calculatePValue(correlation, alignedPairs.size)

            results.add(
                LaggedCorrelation(
                    lagHours = lagHours,
                    coefficient = correlation,
                    pValue = pValue,
                    sampleSize = alignedPairs.size,
                    isSignificant = pValue < SIGNIFICANCE_THRESHOLD && abs(correlation) >= CORRELATION_THRESHOLD
                )
            )
        }

        if (results.isEmpty()) return null

        // Find the strongest significant correlation
        val bestResult = results
            .filter { it.isSignificant }
            .maxByOrNull { abs(it.coefficient) }
            ?: results.maxByOrNull { abs(it.coefficient) }!!

        return CorrelationAnalysis(
            factorName = factorName,
            factorCategory = factorCategory,
            bestCorrelation = bestResult,
            allLagResults = results,
            direction = when {
                bestResult.coefficient > 0 -> CorrelationDirection.POSITIVE
                bestResult.coefficient < 0 -> CorrelationDirection.NEGATIVE
                else -> CorrelationDirection.NONE
            },
            isSignificant = bestResult.isSignificant
        )
    }

    /**
     * Calculate Pearson correlation coefficient
     *
     * r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
     */
    fun calculatePearsonCorrelation(x: List<Double>, y: List<Double>): Double {
        require(x.size == y.size) { "Lists must have same size" }
        require(x.size >= 2) { "Need at least 2 data points" }

        val n = x.size
        val meanX = x.average()
        val meanY = y.average()

        var sumXY = 0.0
        var sumX2 = 0.0
        var sumY2 = 0.0

        for (i in 0 until n) {
            val dx = x[i] - meanX
            val dy = y[i] - meanY
            sumXY += dx * dy
            sumX2 += dx * dx
            sumY2 += dy * dy
        }

        val denominator = sqrt(sumX2 * sumY2)
        return if (denominator == 0.0) 0.0 else sumXY / denominator
    }

    /**
     * Calculate approximate p-value for Pearson correlation
     *
     * Uses Student's t-distribution approximation:
     * t = r × √[(n-2) / (1-r²)]
     *
     * Then approximates p-value from t-statistic
     */
    fun calculatePValue(r: Double, n: Int): Double {
        if (n < 3) return 1.0
        if (abs(r) >= 1.0) return 0.0

        val degreesOfFreedom = n - 2
        val t = r * sqrt(degreesOfFreedom.toDouble() / (1 - r * r))

        // Approximate p-value using t-distribution
        // This is a simplified approximation; production code might use a proper library
        return approximateTDistributionPValue(abs(t), degreesOfFreedom)
    }

    /**
     * Apply Bonferroni correction for multiple comparisons
     *
     * @param pValues List of p-values from multiple tests
     * @return Adjusted significance threshold
     */
    fun bonferroniCorrection(numTests: Int): Double {
        return SIGNIFICANCE_THRESHOLD / numTests
    }

    /**
     * Align two time series with a lag offset
     *
     * @param factorData Factor data (earlier time)
     * @param symptomData Symptom data (later time)
     * @param lagMs Lag in milliseconds
     * @return List of (factor, symptom) pairs aligned by time
     */
    private fun alignDataWithLag(
        factorData: List<Pair<Long, Double>>,
        symptomData: List<Pair<Long, Double>>,
        lagMs: Long
    ): List<Pair<Double, Double>> {
        val aligned = mutableListOf<Pair<Double, Double>>()
        val tolerance = 6 * 60 * 60 * 1000L // 6-hour tolerance window

        for ((factorTime, factorValue) in factorData) {
            val targetTime = factorTime + lagMs

            // Find closest symptom data point within tolerance
            val closest = symptomData.minByOrNull { abs(it.first - targetTime) }

            if (closest != null && abs(closest.first - targetTime) <= tolerance) {
                aligned.add(factorValue to closest.second)
            }
        }

        return aligned
    }

    /**
     * Approximate p-value from t-distribution
     *
     * Uses a simplified approximation based on the normal distribution
     * for large degrees of freedom, and a lookup-style approximation for smaller values
     */
    private fun approximateTDistributionPValue(t: Double, df: Int): Double {
        // Simplified approximation
        // For more accuracy, use a proper statistical library

        // For df > 30, approximate with normal distribution
        if (df > 30) {
            val z = t
            // Approximation of 2-tailed p-value from z-score
            val p = 2.0 * (1.0 - normalCDF(abs(z)))
            return p.coerceIn(0.0, 1.0)
        }

        // For smaller df, use a rough approximation
        // Critical values for common significance levels
        val criticalValues = mapOf(
            0.10 to 1.812,
            0.05 to 2.228,
            0.01 to 3.169,
            0.001 to 4.587
        )

        return when {
            t >= 4.587 -> 0.0001
            t >= 3.169 -> 0.005
            t >= 2.228 -> 0.025
            t >= 1.812 -> 0.075
            else -> 0.5
        }
    }

    /**
     * Approximate standard normal CDF using Zelen & Severo approximation
     */
    private fun normalCDF(x: Double): Double {
        val a1 = 0.254829592
        val a2 = -0.284496736
        val a3 = 1.421413741
        val a4 = -1.453152027
        val a5 = 1.061405429
        val p = 0.3275911

        val sign = if (x < 0) -1 else 1
        val absX = abs(x)

        val t = 1.0 / (1.0 + p * absX)
        val y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * kotlin.math.exp(-absX * absX / 2)

        return 0.5 * (1.0 + sign * y)
    }

    /**
     * Run validation tests to verify correlation calculation accuracy
     */
    fun runValidationTests(): Boolean {
        // Test 1: Perfect positive correlation
        val x1 = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y1 = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val r1 = calculatePearsonCorrelation(x1, y1)
        if (abs(r1 - 1.0) > 0.0001) return false

        // Test 2: Perfect negative correlation
        val x2 = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y2 = listOf(5.0, 4.0, 3.0, 2.0, 1.0)
        val r2 = calculatePearsonCorrelation(x2, y2)
        if (abs(r2 - (-1.0)) > 0.0001) return false

        // Test 3: No correlation
        val x3 = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y3 = listOf(5.0, 2.0, 4.0, 1.0, 3.0)
        val r3 = calculatePearsonCorrelation(x3, y3)
        // Should be close to 0 but not exact
        if (abs(r3) > 0.5) return false

        return true
    }
}

/**
 * Result of correlation analysis for a single factor
 */
data class CorrelationAnalysis(
    val factorName: String,
    val factorCategory: String,
    val bestCorrelation: LaggedCorrelation,
    val allLagResults: List<LaggedCorrelation>,
    val direction: CorrelationDirection,
    val isSignificant: Boolean
)

/**
 * Correlation result at a specific lag
 */
data class LaggedCorrelation(
    val lagHours: Int,
    val coefficient: Double,
    val pValue: Double,
    val sampleSize: Int,
    val isSignificant: Boolean
)

/**
 * Direction of correlation
 */
enum class CorrelationDirection {
    POSITIVE,   // Higher factor values = worse symptoms
    NEGATIVE,   // Higher factor values = better symptoms
    NONE        // No correlation
}
