//
//  SafeMath.swift
//  InflamAI-Swift
//
//  Safe mathematical operations to prevent runtime crashes from division by zero,
//  array out of bounds, and other common arithmetic errors.
//

import Foundation

/// Safe mathematical operations utility
///
/// Provides crash-safe alternatives to common arithmetic operations that can fail at runtime.
/// All functions return sensible default values instead of crashing when encountering invalid inputs.
enum SafeMath {

    // MARK: - Division Operations

    /// Safely divide two numbers, returning a default value if divisor is zero
    ///
    /// - Parameters:
    ///   - dividend: The number to be divided
    ///   - divisor: The number to divide by
    ///   - defaultValue: Value to return if division is impossible (default: 0.0)
    /// - Returns: Result of division, or defaultValue if divisor is zero
    ///
    /// Example:
    /// ```swift
    /// SafeMath.divide(10, by: 2)           // Returns 5.0
    /// SafeMath.divide(10, by: 0)           // Returns 0.0 (default)
    /// SafeMath.divide(10, by: 0, default: -1.0)  // Returns -1.0
    /// ```
    static func divide(_ dividend: Double, by divisor: Double, default defaultValue: Double = 0.0) -> Double {
        guard divisor != 0.0 else { return defaultValue }
        return dividend / divisor
    }

    /// Safely divide two integers, returning a default value if divisor is zero
    ///
    /// - Parameters:
    ///   - dividend: The number to be divided
    ///   - divisor: The number to divide by
    ///   - defaultValue: Value to return if division is impossible (default: 0)
    /// - Returns: Result of division, or defaultValue if divisor is zero
    static func divide(_ dividend: Int, by divisor: Int, default defaultValue: Int = 0) -> Int {
        guard divisor != 0 else { return defaultValue }
        return dividend / divisor
    }

    // MARK: - Average/Mean Calculations

    /// Safely calculate the average of an array of doubles
    ///
    /// - Parameters:
    ///   - values: Array of values to average
    ///   - defaultValue: Value to return if array is empty (default: 0.0)
    /// - Returns: Average of all values, or defaultValue if array is empty
    ///
    /// - Important: For medical/health contexts where "no data" must be distinguished from
    ///   "zero average", use `averageOrNil()` instead. This function's default of 0.0
    ///   conflates "no symptoms" with "no data available".
    ///
    /// Example:
    /// ```swift
    /// SafeMath.average([1, 2, 3, 4, 5])         // Returns 3.0
    /// SafeMath.average([])                      // Returns 0.0 (could be misleading!)
    /// SafeMath.averageOrNil([])                 // Returns nil (medically accurate)
    /// SafeMath.average([0.0, 0.0], default: -1) // Returns 0.0 (explicit zero, not empty)
    /// ```
    static func average(_ values: [Double], default defaultValue: Double = 0.0) -> Double {
        guard !values.isEmpty else { return defaultValue }
        let sum = values.reduce(0.0, +)
        return sum / Double(values.count)
    }

    /// Safely calculate the average of an array of integers
    ///
    /// - Parameters:
    ///   - values: Array of integer values to average
    ///   - defaultValue: Value to return if array is empty (default: 0.0)
    /// - Returns: Average of all values as Double, or defaultValue if array is empty
    static func average(_ values: [Int], default defaultValue: Double = 0.0) -> Double {
        guard !values.isEmpty else { return defaultValue }
        let sum = values.reduce(0, +)
        return Double(sum) / Double(values.count)
    }

    /// Safely calculate the average of an array, returning nil if empty
    ///
    /// - Parameter values: Array of values to average
    /// - Returns: Average of all values, or nil if array is empty
    ///
    /// - Important: **Recommended for medical/health contexts** where you need to distinguish
    ///   between "no data available" (nil) and "zero symptoms" (0.0).
    ///
    /// Example:
    /// ```swift
    /// // Medical context - new user with no symptom logs
    /// let emptyLogs: [Double] = []
    /// SafeMath.averageOrNil(emptyLogs)     // Returns nil - "no data"
    ///
    /// // Medical context - user in remission
    /// let remissionLogs = [0.0, 0.0, 0.0]
    /// SafeMath.averageOrNil(remissionLogs) // Returns 0.0 - "no symptoms" (different from nil!)
    /// ```
    static func averageOrNil(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sum = values.reduce(0.0, +)
        return sum / Double(values.count)
    }

    // MARK: - Percentage Calculations

    /// Safely calculate a percentage, handling edge cases
    ///
    /// - Parameters:
    ///   - part: The partial value
    ///   - whole: The total value
    ///   - defaultValue: Value to return if calculation is impossible (default: 0.0)
    /// - Returns: Percentage (0-100), or defaultValue if whole is zero
    ///
    /// Example:
    /// ```swift
    /// SafeMath.percentage(25, of: 100)     // Returns 25.0
    /// SafeMath.percentage(3, of: 4)        // Returns 75.0
    /// SafeMath.percentage(5, of: 0)        // Returns 0.0 (default)
    /// ```
    static func percentage(_ part: Double, of whole: Double, default defaultValue: Double = 0.0) -> Double {
        guard isValid(part) && isValid(whole) else { return defaultValue }
        guard whole != 0.0 else { return defaultValue }
        let result = (part / whole) * 100.0
        return sanitize(result, default: defaultValue)
    }

    /// Calculate what percentage one value is of another, clamped to 0-100 range
    ///
    /// - Parameters:
    ///   - part: The partial value
    ///   - whole: The total value
    /// - Returns: Percentage clamped between 0 and 100, or 0 if whole is zero
    static func percentageClamped(_ part: Double, of whole: Double) -> Double {
        guard isValid(part) && isValid(whole) else { return 0.0 }
        guard whole != 0.0 else { return 0.0 }
        let percent = (part / whole) * 100.0
        let sanitized = sanitize(percent, default: 0.0)
        return clamp(sanitized, min: 0.0, max: 100.0)
    }

    // MARK: - Array Operations

    /// Safely access an array element by index
    ///
    /// - Parameters:
    ///   - array: The array to access
    ///   - index: The index to retrieve
    /// - Returns: The element at index, or nil if index is out of bounds
    ///
    /// Example:
    /// ```swift
    /// let numbers = [1, 2, 3]
    /// SafeMath.safeElement(in: numbers, at: 1)   // Returns Optional(2)
    /// SafeMath.safeElement(in: numbers, at: 5)   // Returns nil
    /// ```
    static func safeElement<T>(in array: [T], at index: Int) -> T? {
        guard index >= 0, index < array.count else { return nil }
        return array[index]
    }

    /// Validate if an index is within valid bounds for an array
    ///
    /// - Parameters:
    ///   - index: The index to validate
    ///   - array: The array to check against
    /// - Returns: true if index is valid (0..<array.count), false otherwise
    static func isValidIndex<T>(_ index: Int, for array: [T]) -> Bool {
        return index >= 0 && index < array.count
    }

    /// Safely create a range for array slicing
    ///
    /// - Parameters:
    ///   - start: Start index
    ///   - end: End index (inclusive)
    ///   - arrayCount: Number of elements in the array
    /// - Returns: Valid range within array bounds, or nil if range is completely invalid
    ///
    /// Example:
    /// ```swift
    /// SafeMath.safeRange(start: 0, end: 5, arrayCount: 10)   // Returns 0...5
    /// SafeMath.safeRange(start: 8, end: 12, arrayCount: 10)  // Returns 8...9 (clamped)
    /// SafeMath.safeRange(start: 15, end: 20, arrayCount: 10) // Returns nil (completely out of bounds)
    /// ```
    static func safeRange(start: Int, end: Int, arrayCount: Int) -> ClosedRange<Int>? {
        guard arrayCount > 0 else { return nil }
        guard start < arrayCount else { return nil }  // Start is beyond array

        let safeStart = max(0, start)
        let safeEnd = min(end, arrayCount - 1)

        guard safeStart <= safeEnd else { return nil }

        return safeStart...safeEnd
    }

    // MARK: - Clamping and Bounds

    /// Clamp a value to a specified range
    ///
    /// - Parameters:
    ///   - value: The value to clamp
    ///   - min: Minimum allowed value
    ///   - max: Maximum allowed value
    /// - Returns: Value clamped to [min, max] range
    ///
    /// Example:
    /// ```swift
    /// SafeMath.clamp(15, min: 0, max: 10)    // Returns 10
    /// SafeMath.clamp(-5, min: 0, max: 10)    // Returns 0
    /// SafeMath.clamp(5, min: 0, max: 10)     // Returns 5
    /// ```
    static func clamp<T: Comparable>(_ value: T, min minValue: T, max maxValue: T) -> T {
        return min(max(value, minValue), maxValue)
    }

    /// Normalize a value to 0-1 range based on min/max bounds
    ///
    /// - Parameters:
    ///   - value: The value to normalize
    ///   - min: Minimum value in the range
    ///   - max: Maximum value in the range
    /// - Returns: Normalized value between 0 and 1, or 0 if range is invalid
    ///
    /// Example:
    /// ```swift
    /// SafeMath.normalize(50, min: 0, max: 100)   // Returns 0.5
    /// SafeMath.normalize(25, min: 0, max: 100)   // Returns 0.25
    /// SafeMath.normalize(150, min: 0, max: 100)  // Returns 1.0 (clamped)
    /// ```
    static func normalize(_ value: Double, min: Double, max: Double) -> Double {
        guard max > min else { return 0.0 }
        let normalized = (value - min) / (max - min)
        return clamp(normalized, min: 0.0, max: 1.0)
    }

    // MARK: - Statistical Operations

    /// Safely calculate the sum of an array
    ///
    /// - Parameter values: Array of values to sum
    /// - Returns: Sum of all values, or 0 if array is empty
    static func sum(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0.0 }
        return values.reduce(0.0, +)
    }

    /// Safely calculate the sum of an integer array
    ///
    /// - Parameter values: Array of integer values to sum
    /// - Returns: Sum of all values, or 0 if array is empty
    static func sum(_ values: [Int]) -> Int {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +)
    }

    /// Calculate the median of an array of values
    ///
    /// - Parameters:
    ///   - values: Array of values
    ///   - defaultValue: Value to return if array is empty (default: 0.0)
    /// - Returns: Median value, or defaultValue if array is empty
    ///
    /// Example:
    /// ```swift
    /// SafeMath.median([1, 2, 3, 4, 5])       // Returns 3.0
    /// SafeMath.median([1, 2, 3, 4])          // Returns 2.5
    /// SafeMath.median([])                    // Returns 0.0
    /// ```
    static func median(_ values: [Double], default defaultValue: Double = 0.0) -> Double {
        guard !values.isEmpty else { return defaultValue }

        // Filter out invalid values (NaN, infinity)
        let validValues = values.filter { isValid($0) }
        guard !validValues.isEmpty else { return defaultValue }

        let sorted = validValues.sorted()
        let count = sorted.count

        if count % 2 == 0 {
            // Even number of elements - average the two middle values
            let mid1 = sorted[count / 2 - 1]
            let mid2 = sorted[count / 2]
            return divide((mid1 + mid2), by: 2.0, default: defaultValue)
        } else {
            // Odd number of elements - return the middle value
            return sorted[count / 2]
        }
    }

    /// Calculate the standard deviation of an array of values
    ///
    /// - Parameters:
    ///   - values: Array of values
    ///   - defaultValue: Value to return if array is empty or has only one element (default: 0.0)
    /// - Returns: Standard deviation, or defaultValue if insufficient data
    static func standardDeviation(_ values: [Double], default defaultValue: Double = 0.0) -> Double {
        guard values.count > 1 else { return defaultValue }

        // Filter out invalid values (NaN, infinity)
        let validValues = values.filter { isValid($0) }
        guard validValues.count > 1 else { return defaultValue }

        let mean = average(validValues)
        let squaredDifferences = validValues.map { pow($0 - mean, 2) }
        let variance = average(squaredDifferences)

        let result = sqrt(variance)
        return sanitize(result, default: defaultValue)
    }

    // MARK: - Validation Helpers

    /// Check if a value is a valid, finite number (not NaN or infinity)
    ///
    /// - Parameter value: The value to check
    /// - Returns: true if value is a normal finite number, false if NaN or infinite
    static func isValid(_ value: Double) -> Bool {
        return !value.isNaN && !value.isInfinite
    }

    /// Sanitize a value, replacing NaN or infinity with a default
    ///
    /// - Parameters:
    ///   - value: The value to sanitize
    ///   - defaultValue: Replacement value for invalid numbers (default: 0.0)
    /// - Returns: Original value if valid, defaultValue if NaN or infinite
    static func sanitize(_ value: Double, default defaultValue: Double = 0.0) -> Double {
        return isValid(value) ? value : defaultValue
    }

    // MARK: - Rounding Operations

    /// Round a value to a specified number of decimal places
    ///
    /// - Parameters:
    ///   - value: The value to round
    ///   - decimalPlaces: Number of decimal places to keep
    /// - Returns: Rounded value
    ///
    /// Example:
    /// ```swift
    /// SafeMath.round(3.14159, decimalPlaces: 2)   // Returns 3.14
    /// SafeMath.round(3.14159, decimalPlaces: 0)   // Returns 3.0
    /// ```
    static func round(_ value: Double, decimalPlaces: Int) -> Double {
        let multiplier = pow(10.0, Double(decimalPlaces))
        return (value * multiplier).rounded() / multiplier
    }
}

// MARK: - Array Extension

extension Array where Element == Double {

    /// Safely calculate the average of the array
    /// - Returns: Average value, or 0.0 if array is empty
    var safeAverage: Double {
        SafeMath.average(self)
    }

    /// Safely calculate the sum of the array
    /// - Returns: Sum of all values, or 0.0 if array is empty
    var safeSum: Double {
        SafeMath.sum(self)
    }

    /// Safely calculate the median of the array
    /// - Returns: Median value, or 0.0 if array is empty
    var safeMedian: Double {
        SafeMath.median(self)
    }
}

extension Array where Element == Int {

    /// Safely calculate the average of the array
    /// - Returns: Average value, or 0.0 if array is empty
    var safeAverage: Double {
        SafeMath.average(self)
    }

    /// Safely calculate the sum of the array
    /// - Returns: Sum of all values, or 0 if array is empty
    var safeSum: Int {
        SafeMath.sum(self)
    }
}
