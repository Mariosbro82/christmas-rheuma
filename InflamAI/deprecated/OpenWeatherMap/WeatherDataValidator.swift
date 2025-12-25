//
//  WeatherDataValidator.swift
//  InflamAI-Swift
//
//  Validates and sanitizes weather data to catch invalid values
//  Ensures data integrity before storage and analysis
//

import Foundation

struct WeatherDataValidator {

    // MARK: - Validation

    static func validate(_ weather: WeatherData) -> ValidationResult {
        var issues: [ValidationIssue] = []

        // Validate pressure (typical range: 950-1050 hPa)
        if weather.barometricPressure < 900 || weather.barometricPressure > 1100 {
            issues.append(.init(
                field: "barometricPressure",
                severity: .error,
                message: "Pressure \(weather.barometricPressure) hPa is outside valid range (900-1100)"
            ))
        } else if weather.barometricPressure < 950 || weather.barometricPressure > 1050 {
            issues.append(.init(
                field: "barometricPressure",
                severity: .warning,
                message: "Pressure \(weather.barometricPressure) hPa is unusual but possible"
            ))
        }

        // Validate temperature (typical range: -50 to 60°C)
        if weather.temperature < -60 || weather.temperature > 70 {
            issues.append(.init(
                field: "temperature",
                severity: .error,
                message: "Temperature \(weather.temperature)°C is outside valid range"
            ))
        }

        // Validate humidity (0-100%)
        if weather.humidity < 0 || weather.humidity > 100 {
            issues.append(.init(
                field: "humidity",
                severity: .error,
                message: "Humidity \(weather.humidity)% is invalid"
            ))
        }

        // Validate wind speed (0-300 km/h)
        if weather.windSpeed < 0 || weather.windSpeed > 300 {
            issues.append(.init(
                field: "windSpeed",
                severity: .error,
                message: "Wind speed \(weather.windSpeed) km/h is invalid"
            ))
        }

        // Validate precipitation (0-500 mm/h)
        if weather.precipitation < 0 || weather.precipitation > 500 {
            issues.append(.init(
                field: "precipitation",
                severity: .error,
                message: "Precipitation \(weather.precipitation) mm is invalid"
            ))
        }

        // Validate UV index (0-20)
        if weather.uvIndex < 0 || weather.uvIndex > 20 {
            issues.append(.init(
                field: "uvIndex",
                severity: .error,
                message: "UV index \(weather.uvIndex) is invalid"
            ))
        }

        // Validate timestamp (shouldn't be in future or too old)
        let now = Date()
        let timeDiff = abs(now.timeIntervalSince(weather.date))
        if weather.date > now.addingTimeInterval(3600) { // More than 1 hour in future
            issues.append(.init(
                field: "date",
                severity: .error,
                message: "Weather data timestamp is in the future"
            ))
        } else if timeDiff > 86400 * 7 { // More than 7 days old
            issues.append(.init(
                field: "date",
                severity: .warning,
                message: "Weather data is more than 7 days old"
            ))
        }

        return ValidationResult(
            isValid: issues.filter { $0.severity == .error }.isEmpty,
            issues: issues
        )
    }

    static func validate(_ forecast: WeatherForecast) -> ValidationResult {
        var allIssues: [ValidationIssue] = []

        // Validate current weather
        let currentValidation = validate(forecast.current)
        allIssues.append(contentsOf: currentValidation.issues)

        // Validate hourly data
        if forecast.hourly.isEmpty {
            allIssues.append(.init(
                field: "hourly",
                severity: .warning,
                message: "No hourly forecast data available"
            ))
        } else {
            // Check for gaps in hourly data
            for i in 0..<(forecast.hourly.count - 1) {
                let current = forecast.hourly[i]
                let next = forecast.hourly[i + 1]
                let timeDiff = next.date.timeIntervalSince(current.date)

                if timeDiff > 7200 { // More than 2 hours gap
                    allIssues.append(.init(
                        field: "hourly",
                        severity: .warning,
                        message: "Gap in hourly data at \(current.date)"
                    ))
                }
            }
        }

        // Validate daily data
        if forecast.daily.isEmpty {
            allIssues.append(.init(
                field: "daily",
                severity: .warning,
                message: "No daily forecast data available"
            ))
        }

        // Check last updated timestamp
        let updateAge = Date().timeIntervalSince(forecast.lastUpdated)
        if updateAge > 3600 { // More than 1 hour old
            allIssues.append(.init(
                field: "lastUpdated",
                severity: .warning,
                message: "Forecast data is \(Int(updateAge/60)) minutes old"
            ))
        }

        return ValidationResult(
            isValid: allIssues.filter { $0.severity == .error }.isEmpty,
            issues: allIssues
        )
    }

    // MARK: - Sanitization

    static func sanitize(_ weather: WeatherData) -> WeatherData {
        // Clamp values to valid ranges
        return WeatherData(
            date: weather.date,
            temperature: max(-60, min(70, weather.temperature)),
            humidity: max(0, min(100, weather.humidity)),
            barometricPressure: max(900, min(1100, weather.barometricPressure)),
            windSpeed: max(0, min(300, weather.windSpeed)),
            precipitation: max(0, min(500, weather.precipitation)),
            uvIndex: max(0, min(20, weather.uvIndex)),
            condition: weather.condition,
            location: weather.location
        )
    }

    // MARK: - Edge Case Detection

    /// Detect rapid altitude changes that might be mistaken for weather changes
    static func detectAltitudeChange(
        currentPressure: Double,
        previousPressure: Double,
        timeInterval: TimeInterval
    ) -> AltitudeChangeDetection {
        let pressureDrop = previousPressure - currentPressure
        let dropRate = pressureDrop / (timeInterval / 3600) // hPa per hour

        // Rapid pressure changes (>10 hPa/hour) are likely altitude-related
        if abs(dropRate) > 10 {
            return .init(
                isLikelyAltitudeChange: true,
                estimatedAltitudeChange: pressureDrop * 8.5, // ~8.5 meters per hPa
                shouldIgnoreForFlareRisk: true,
                message: "Rapid pressure change detected (likely altitude change). Not used for flare prediction."
            )
        }

        return .init(
            isLikelyAltitudeChange: false,
            estimatedAltitudeChange: 0,
            shouldIgnoreForFlareRisk: false,
            message: nil
        )
    }

    /// Validate correlation statistics for reliability
    static func validateCorrelation(
        correlation: Double,
        sampleSize: Int,
        pValue: Double
    ) -> CorrelationValidation {
        // Check for sufficient sample size
        if sampleSize < 14 {
            return .init(
                isReliable: false,
                warning: "Need at least 14 days of data. Currently have \(sampleSize) days.",
                recommendedAction: .collectMoreData
            )
        }

        // Check statistical significance
        if pValue >= 0.05 {
            return .init(
                isReliable: false,
                warning: "Weather correlation is not statistically significant (p=\(String(format: "%.3f", pValue)))",
                recommendedAction: .continueTracking
            )
        }

        // Check for spurious correlation (very high r with small sample)
        if abs(correlation) > 0.9 && sampleSize < 30 {
            return .init(
                isReliable: false,
                warning: "Correlation may be spurious due to small sample size",
                recommendedAction: .interpretCautiously
            )
        }

        return .init(
            isReliable: true,
            warning: nil,
            recommendedAction: .none
        )
    }
}

// MARK: - Supporting Types

struct ValidationResult {
    let isValid: Bool
    let issues: [ValidationIssue]

    var hasWarnings: Bool {
        issues.contains { $0.severity == .warning }
    }

    var hasErrors: Bool {
        issues.contains { $0.severity == .error }
    }

    var errorMessages: [String] {
        issues.filter { $0.severity == .error }.map { $0.message }
    }

    var warningMessages: [String] {
        issues.filter { $0.severity == .warning }.map { $0.message }
    }
}

struct ValidationIssue {
    let field: String
    let severity: Severity
    let message: String

    enum Severity {
        case warning
        case error
    }
}

struct AltitudeChangeDetection {
    let isLikelyAltitudeChange: Bool
    let estimatedAltitudeChange: Double // meters
    let shouldIgnoreForFlareRisk: Bool
    let message: String?
}

struct CorrelationValidation {
    let isReliable: Bool
    let warning: String?
    let recommendedAction: Action

    enum Action {
        case none
        case collectMoreData
        case continueTracking
        case interpretCautiously
    }
}
