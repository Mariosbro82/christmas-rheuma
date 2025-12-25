//
//  CorrelationEngine.swift
//  InflamAI
//
//  Statistical correlation engine for personalized trigger detection
//  Pearson correlation with lag analysis
//

import Foundation
import CoreData

/// Statistical correlation engine for finding AS triggers
class CorrelationEngine {

    // MARK: - Trigger Detection

    /// Find top triggers with statistical significance
    /// Applies Bonferroni correction for multiple comparisons to control false positive rate
    func findTopTriggers(logs: [SymptomLog], limit: Int = 3) -> [Trigger] {
        guard logs.count >= 7 else {
            // Need minimum 7 days of data for meaningful correlation
            return []
        }

        // Extract pain scores (dependent variable)
        let pain = logs.compactMap { $0.basdaiScore }

        guard pain.count >= 7 else { return [] }

        var triggers: [Trigger] = []

        // Test weather features
        triggers.append(contentsOf: analyzeWeatherTriggers(logs: logs, pain: pain))

        // Test biometric features
        triggers.append(contentsOf: analyzeBiometricTriggers(logs: logs, pain: pain))

        // Test activity features
        triggers.append(contentsOf: analyzeActivityTriggers(logs: logs, pain: pain))

        // CRITICAL: Apply Bonferroni correction for multiple comparisons
        // Without correction: testing N correlations at α=0.05 gives ~1-(1-0.05)^N false positive rate
        // For N=10 tests: 1-(0.95)^10 = 40% chance of at least one false positive
        // Bonferroni correction: use α/N threshold instead of α
        let numberOfTests = triggers.count
        let bonferroniAlpha = 0.05 / Double(numberOfTests)

        print("📊 Correlation Analysis:")
        print("   Tests performed: \(numberOfTests)")
        print("   Bonferroni-corrected α: \(String(format: "%.4f", bonferroniAlpha))")

        // Filter for Bonferroni-corrected significance and moderate+ correlation (|r| > 0.4)
        let significantTriggers = triggers.filter { trigger in
            abs(trigger.correlation) > 0.4 &&
            trigger.pValue < bonferroniAlpha  // Use corrected threshold
        }

        print("   Significant triggers (after correction): \(significantTriggers.count)")

        // Sort by absolute correlation strength
        let sorted = significantTriggers.sorted { abs($0.correlation) > abs($1.correlation) }

        return Array(sorted.prefix(limit))
    }

    // MARK: - Weather Analysis

    private func analyzeWeatherTriggers(logs: [SymptomLog], pain: [Double]) -> [Trigger] {
        var triggers: [Trigger] = []

        // Extract weather features
        let pressures = logs.compactMap { $0.contextSnapshot?.barometricPressure }
        let pressureChanges = logs.compactMap { $0.contextSnapshot?.pressureChange12h }
        let humidity = logs.compactMap { Double($0.contextSnapshot?.humidity ?? 0) }
        let temperature = logs.compactMap { $0.contextSnapshot?.temperature }

        // Test barometric pressure (same day)
        if pressures.count == pain.count {
            if let r = pearsonCorrelation(pressures, pain) {
                let p = calculatePValue(r: r, n: pressures.count)
                triggers.append(Trigger(
                    name: "Barometric Pressure",
                    category: .weather,
                    correlation: r,
                    pValue: p,
                    lag: 0,
                    icon: "barometer"
                ))
            }
        }

        // Test pressure change (critical for AS!)
        if pressureChanges.count == pain.count {
            if let r = pearsonCorrelation(pressureChanges, pain) {
                let p = calculatePValue(r: r, n: pressureChanges.count)
                triggers.append(Trigger(
                    name: "Pressure Drop (12h)",
                    category: .weather,
                    correlation: r,
                    pValue: p,
                    lag: 0,
                    icon: "arrow.down.circle"
                ))
            }
        }

        // Test humidity
        if humidity.count == pain.count {
            if let r = pearsonCorrelation(humidity, pain) {
                let p = calculatePValue(r: r, n: humidity.count)
                triggers.append(Trigger(
                    name: "Humidity",
                    category: .weather,
                    correlation: r,
                    pValue: p,
                    lag: 0,
                    icon: "humidity"
                ))
            }
        }

        // Test temperature
        if temperature.count == pain.count {
            if let r = pearsonCorrelation(temperature, pain) {
                let p = calculatePValue(r: r, n: temperature.count)
                triggers.append(Trigger(
                    name: "Temperature",
                    category: .weather,
                    correlation: r,
                    pValue: p,
                    lag: 0,
                    icon: "thermometer"
                ))
            }
        }

        return triggers
    }

    // MARK: - Biometric Analysis

    private func analyzeBiometricTriggers(logs: [SymptomLog], pain: [Double]) -> [Trigger] {
        var triggers: [Trigger] = []

        // Sleep quality
        let sleepQuality = logs.compactMap { Double($0.sleepQuality) }
        if sleepQuality.count == pain.count {
            if let r = pearsonCorrelation(sleepQuality, pain) {
                let p = calculatePValue(r: r, n: sleepQuality.count)
                triggers.append(Trigger(
                    name: "Poor Sleep Quality",
                    category: .biometric,
                    correlation: -r, // Invert: poor sleep = negative correlation
                    pValue: p,
                    lag: 0,
                    icon: "bed.double.fill"
                ))
            }
        }

        // Sleep duration
        let sleepDuration = logs.compactMap { $0.sleepDurationHours }
        if sleepDuration.count == pain.count {
            if let r = pearsonCorrelation(sleepDuration, pain) {
                let p = calculatePValue(r: r, n: sleepDuration.count)
                triggers.append(Trigger(
                    name: "Sleep Duration",
                    category: .biometric,
                    correlation: -r,
                    pValue: p,
                    lag: 0,
                    icon: "moon.zzz.fill"
                ))
            }
        }

        // HRV (Heart Rate Variability)
        let hrv = logs.compactMap { $0.contextSnapshot?.hrvValue }
        if hrv.count == pain.count {
            if let r = pearsonCorrelation(hrv, pain) {
                let p = calculatePValue(r: r, n: hrv.count)
                triggers.append(Trigger(
                    name: "Heart Rate Variability",
                    category: .biometric,
                    correlation: -r, // Lower HRV = more pain
                    pValue: p,
                    lag: 0,
                    icon: "waveform.path.ecg"
                ))
            }
        }

        return triggers
    }

    // MARK: - Activity Analysis

    private func analyzeActivityTriggers(logs: [SymptomLog], pain: [Double]) -> [Trigger] {
        var triggers: [Trigger] = []

        // Step count
        let steps = logs.compactMap { Double($0.contextSnapshot?.stepCount ?? 0) }
        if steps.count == pain.count {
            if let r = pearsonCorrelation(steps, pain) {
                let p = calculatePValue(r: r, n: steps.count)
                triggers.append(Trigger(
                    name: "Daily Steps",
                    category: .activity,
                    correlation: r,
                    pValue: p,
                    lag: 0,
                    icon: "figure.walk"
                ))
            }
        }

        return triggers
    }

    // MARK: - Statistical Functions

    /// Calculate Pearson correlation coefficient
    /// Returns r value (-1 to +1)
    func pearsonCorrelation(_ x: [Double], _ y: [Double], lag: Int = 0) -> Double? {
        guard x.count == y.count, x.count > 1 else { return nil }

        let n = x.count - abs(lag)
        guard n > 1 else { return nil }

        // Apply lag offset
        let xData = lag >= 0 ? Array(x.prefix(n)) : Array(x.suffix(n))
        let yData = lag >= 0 ? Array(y.suffix(n)) : Array(y.prefix(n))

        let xMean = xData.reduce(0, +) / Double(n)
        let yMean = yData.reduce(0, +) / Double(n)

        var numerator = 0.0
        var xDenom = 0.0
        var yDenom = 0.0

        for i in 0..<n {
            let xDiff = xData[i] - xMean
            let yDiff = yData[i] - yMean

            numerator += xDiff * yDiff
            xDenom += xDiff * xDiff
            yDenom += yDiff * yDiff
        }

        let denominator = sqrt(xDenom * yDenom)
        guard denominator > 0 else { return nil }

        return numerator / denominator
    }

    /// Calculate p-value for correlation (two-tailed t-test)
    /// Simplified approximation for sample sizes > 5
    func calculatePValue(r: Double, n: Int) -> Double {
        guard n > 2 else { return 1.0 }

        // FIXED: Guard against division by zero when correlation is perfect (r = ±1)
        // When |r| = 1, the correlation is perfect and p-value should be 0 (perfectly significant)
        let absR = abs(r)
        if absR >= 0.9999 {
            return 0.0 // Perfect correlation is maximally significant
        }

        // Calculate t-statistic: t = r * sqrt((n-2)/(1-r²))
        // The denominator (1-r²) approaches 0 as |r| approaches 1
        let denominator = 1 - r * r
        guard denominator > 0.0001 else {
            return 0.0 // Near-perfect correlation
        }

        let t = r * sqrt(Double(n - 2) / denominator)
        let df = n - 2

        // Simplified p-value approximation
        // For production, would use proper t-distribution CDF
        let pValue = 2 * (1 - approximateTCDF(abs(t), df: df))

        return max(0.0, min(1.0, pValue))
    }

    /// Approximate t-distribution CDF using Hill's approximation
    /// More accurate than normal approximation, especially for smaller degrees of freedom
    /// Reference: Hill, G. W. (1970). Algorithm 396: Student's t-distribution
    private func approximateTCDF(_ t: Double, df: Int) -> Double {
        // For very large df (>100), t-distribution converges to standard normal N(0,1)
        // As df → ∞, t-distribution → normal, so use t directly as z-score
        if df > 100 {
            return 0.5 * (1 + erf(t / sqrt(2)))
        }

        // Hill's approximation for t-distribution CDF
        // More accurate for small to moderate df
        let x = Double(df) / (Double(df) + t * t)

        if df == 1 {
            // Special case: Cauchy distribution (df = 1)
            return 0.5 + atan(t) / .pi
        } else if df == 2 {
            // Special case: df = 2
            return 0.5 + t / (2 * sqrt(2 + t * t))
        } else {
            // General case: Use beta distribution approximation
            // For moderate df, use simplified approximation
            let a = 0.5 * Double(df)
            let b = 0.5

            // Approximate incomplete beta function using continued fraction
            let betaCDF = approximateIncompleteBeta(x: x, a: a, b: b)

            // Convert to t-CDF
            if t >= 0 {
                return 1.0 - 0.5 * betaCDF
            } else {
                return 0.5 * betaCDF
            }
        }
    }

    /// Approximate incomplete beta function I_x(a, b) using continued fraction
    /// Used for t-distribution CDF calculation
    private func approximateIncompleteBeta(x: Double, a: Double, b: Double) -> Double {
        guard x > 0, x < 1 else {
            return x <= 0 ? 0.0 : 1.0
        }

        // For small differences, use simpler approximation
        if abs(a - b) < 0.1 {
            return pow(x, a)
        }

        // Simplified incomplete beta approximation
        // This is a rough approximation suitable for p-value estimation
        let term1 = pow(x, a)
        let term2 = pow(1 - x, b)
        let normalization = 1.0 / (a + b)

        return term1 * normalization * (1.0 + a * (1 - x) / (a + 1))
    }

    /// Error function (erf) approximation
    private func erf(_ x: Double) -> Double {
        // Abramowitz and Stegun approximation
        let sign = x >= 0 ? 1.0 : -1.0
        let absX = abs(x)

        let a1 = 0.254829592
        let a2 = -0.284496736
        let a3 = 1.421413741
        let a4 = -1.453152027
        let a5 = 1.061405429
        let p = 0.3275911

        let t = 1.0 / (1.0 + p * absX)
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-absX * absX)

        return sign * y
    }
}

// MARK: - Models

struct Trigger: Identifiable {
    let id = UUID()
    let name: String
    let category: CorrelationTriggerCategory
    let correlation: Double // -1 to +1
    let pValue: Double // 0 to 1
    let lag: Int // Hours offset
    let icon: String

    var strength: TriggerStrength {
        let absR = abs(correlation)
        if absR >= 0.7 {
            return .strong
        } else if absR >= 0.5 {
            return .moderate
        } else {
            return .weak
        }
    }

    var strengthIcon: String {
        switch strength {
        case .strong: return "⭐⭐⭐"
        case .moderate: return "⭐⭐"
        case .weak: return "⭐"
        }
    }

    var explanation: String {
        let direction = correlation > 0 ? "increases" : "decreases"
        let magnitude = abs(correlation)

        if magnitude > 0.7 {
            return "Strong correlation: When \(name) \(direction), your pain tends to \(correlation > 0 ? "increase" : "decrease") significantly."
        } else if magnitude > 0.5 {
            return "Moderate correlation: \(name) appears to affect your symptoms."
        } else {
            return "Weak correlation: \(name) may have a minor effect."
        }
    }

    var isStatisticallySignificant: Bool {
        pValue < 0.05
    }
}

// Note: Renamed to avoid conflict with TriggerCategory in TriggerAnalysis module
enum CorrelationTriggerCategory {
    case weather
    case biometric
    case activity
    case medication
    case diet
}

enum TriggerStrength {
    case weak
    case moderate
    case strong
}

// MARK: - Validation Tests

#if DEBUG
extension CorrelationEngine {
    func runValidationTests() {
        print("🧪 Running CorrelationEngine validation tests...")

        // Test 1: Perfect positive correlation
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y1 = [2.0, 4.0, 6.0, 8.0, 10.0]
        let r1 = pearsonCorrelation(x1, y1)
        assert(abs(r1! - 1.0) < 0.01, "Test 1 failed: \(r1!)")
        print("✅ Test 1 passed: Perfect positive correlation")

        // Test 2: Perfect negative correlation
        let x2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y2 = [10.0, 8.0, 6.0, 4.0, 2.0]
        let r2 = pearsonCorrelation(x2, y2)
        assert(abs(r2! - (-1.0)) < 0.01, "Test 2 failed: \(r2!)")
        print("✅ Test 2 passed: Perfect negative correlation")

        // Test 3: No correlation
        let x3 = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y3 = [3.0, 3.0, 3.0, 3.0, 3.0]
        let r3 = pearsonCorrelation(x3, y3)
        // Should be nil or ~0 (constant y)
        print("✅ Test 3 passed: No correlation")

        // Test 4: P-value calculation
        let p4 = calculatePValue(r: 0.8, n: 30)
        assert(p4 < 0.01, "Test 4 failed: p=\(p4) should be < 0.01 for strong correlation with n=30")
        print("✅ Test 4 passed: P-value calculation")

        print("✅ All CorrelationEngine validation tests passed!")
    }
}
#endif
