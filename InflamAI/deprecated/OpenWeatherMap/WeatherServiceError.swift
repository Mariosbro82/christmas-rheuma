//
//  WeatherServiceError.swift
//  InflamAI-Swift
//
//  Enhanced error handling for weather services
//  Includes recovery strategies and user-facing messages
//

import Foundation
import SwiftUI

enum WeatherServiceError: LocalizedError {
    // Network errors
    case networkUnavailable
    case requestTimeout
    case serverError(statusCode: Int)
    case invalidResponse(details: String)

    // API errors
    case invalidAPIKey
    case rateLimitExceeded(retryAfter: TimeInterval?)
    case quotaExceeded
    case serviceUnavailable

    // Location errors
    case locationNotAvailable
    case locationPermissionDenied
    case locationPermissionRestricted
    case locationServicesDisabled

    // Data errors
    case insufficientData(required: Int, available: Int)
    case corruptedData
    case parsingError(underlying: Error)

    // Cache errors
    case cacheReadFailure
    case cacheWriteFailure

    var errorDescription: String? {
        switch self {
        case .networkUnavailable:
            return "No internet connection. Using cached weather data."
        case .requestTimeout:
            return "Weather service request timed out. Please try again."
        case .serverError(let code):
            return "Weather service error (code \(code)). Please try again later."
        case .invalidResponse(let details):
            return "Invalid weather data received: \(details)"
        case .invalidAPIKey:
            return "Weather API configuration error. Please contact support."
        case .rateLimitExceeded(let retryAfter):
            if let retry = retryAfter {
                return "Too many requests. Please wait \(Int(retry)) seconds."
            }
            return "Too many weather requests. Please wait a moment."
        case .quotaExceeded:
            return "Weather service quota exceeded. Service will resume tomorrow."
        case .serviceUnavailable:
            return "Weather service is temporarily unavailable."
        case .locationNotAvailable:
            return "Unable to determine your location."
        case .locationPermissionDenied:
            return "Location access denied. Enable in Settings to get weather data."
        case .locationPermissionRestricted:
            return "Location access is restricted by device settings."
        case .locationServicesDisabled:
            return "Location services are disabled. Enable in device Settings."
        case .insufficientData(let required, let available):
            return "Need \(required) days of data, have \(available). Keep tracking!"
        case .corruptedData:
            return "Weather data is corrupted. Refreshing..."
        case .parsingError(let error):
            return "Error processing weather data: \(error.localizedDescription)"
        case .cacheReadFailure:
            return "Unable to load cached weather data."
        case .cacheWriteFailure:
            return "Unable to save weather data to cache."
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .networkUnavailable:
            return "Check your internet connection and try again."
        case .requestTimeout:
            return "Ensure you have a stable connection and retry."
        case .locationPermissionDenied, .locationPermissionRestricted:
            return "Go to Settings > Privacy > Location Services to enable."
        case .locationServicesDisabled:
            return "Go to Settings > Privacy > Location Services and turn it on."
        case .rateLimitExceeded:
            return "Wait a few minutes before requesting weather data again."
        case .invalidAPIKey:
            return "Please update the app to restore weather functionality."
        case .insufficientData:
            return "Continue logging daily to enable weather predictions."
        default:
            return "Try again later or contact support if the issue persists."
        }
    }

    var isRecoverable: Bool {
        switch self {
        case .networkUnavailable, .requestTimeout, .rateLimitExceeded, .serviceUnavailable:
            return true
        case .invalidAPIKey, .quotaExceeded:
            return false
        case .locationPermissionDenied, .locationPermissionRestricted, .locationServicesDisabled:
            return true // User can fix in settings
        default:
            return false
        }
    }

    var shouldUseCachedData: Bool {
        switch self {
        case .networkUnavailable, .requestTimeout, .serverError, .serviceUnavailable, .rateLimitExceeded:
            return true
        default:
            return false
        }
    }

    var shouldRetry: Bool {
        switch self {
        case .requestTimeout:
            return true
        case .serverError(let statusCode) where statusCode >= 500:
            return true
        default:
            return false
        }
    }

    var icon: String {
        switch self {
        case .locationPermissionDenied, .locationServicesDisabled, .locationPermissionRestricted:
            return "location.slash"
        case .networkUnavailable:
            return "wifi.slash"
        case .rateLimitExceeded, .quotaExceeded:
            return "exclamationmark.triangle"
        case .invalidAPIKey:
            return "key.slash"
        default:
            return "cloud.slash"
        }
    }

    var color: Color {
        switch self {
        case .locationPermissionDenied, .locationServicesDisabled, .locationPermissionRestricted:
            return .orange
        case .networkUnavailable:
            return .blue
        case .invalidAPIKey, .quotaExceeded:
            return .red
        case .rateLimitExceeded:
            return .yellow
        default:
            return .gray
        }
    }
}

// MARK: - Service Status

enum ServiceStatus: Equatable {
    case operational       // All systems working
    case degraded         // Using cached data
    case rateLimited      // Too many requests
    case locationError    // Location permission issue
    case unavailable      // Service down or quota exceeded

    var displayName: String {
        switch self {
        case .operational: return "Operational"
        case .degraded: return "Limited"
        case .rateLimited: return "Rate Limited"
        case .locationError: return "Location Error"
        case .unavailable: return "Unavailable"
        }
    }

    var color: Color {
        switch self {
        case .operational: return .green
        case .degraded: return .yellow
        case .rateLimited: return .orange
        case .locationError: return .orange
        case .unavailable: return .red
        }
    }

    var icon: String {
        switch self {
        case .operational: return "checkmark.circle.fill"
        case .degraded: return "exclamationmark.triangle.fill"
        case .rateLimited: return "clock.fill"
        case .locationError: return "location.slash.fill"
        case .unavailable: return "xmark.circle.fill"
        }
    }
}

// MARK: - Retry Policy

enum RetryPolicy {
    case exponentialBackoff
    case fixedInterval(TimeInterval)
    case custom(maxAttempts: Int, delays: [TimeInterval])

    var maxAttempts: Int {
        switch self {
        case .exponentialBackoff:
            return 3
        case .fixedInterval:
            return 3
        case .custom(let max, _):
            return max
        }
    }

    func delay(for attempt: Int) -> TimeInterval {
        switch self {
        case .exponentialBackoff:
            // 1s, 2s, 4s
            return pow(2.0, Double(attempt - 1))
        case .fixedInterval(let interval):
            return interval
        case .custom(_, let delays):
            return delays[min(attempt - 1, delays.count - 1)]
        }
    }
}
