//
//  ResilientWeatherService.swift
//  InflamAI-Swift
//
//  Weather service with automatic retry, fallback, and graceful degradation
//  Wraps WeatherService with error handling and caching strategies
//

import Foundation
import CoreLocation
import Combine

/// Weather service with automatic retry and fallback to cached data
@MainActor
class ResilientWeatherService: ObservableObject {

    // MARK: - Published Properties

    @Published var serviceStatus: ServiceStatus = .operational
    @Published var lastSuccessfulUpdate: Date?
    @Published var errorMessage: String?

    // MARK: - Private Properties

    let baseService: WeatherService  // Internal access for getCurrentLocation
    private let cache: WeatherCache
    private let retryPolicy: RetryPolicy
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init(
        weatherService: WeatherService = WeatherService(),
        cache: WeatherCache = WeatherCache.shared,
        retryPolicy: RetryPolicy = .exponentialBackoff
    ) {
        self.baseService = weatherService
        self.cache = cache
        self.retryPolicy = retryPolicy
    }

    // MARK: - Public Methods

    /// Get weather with automatic retry and fallback
    func getWeatherWithRetry(for location: CLLocation) async -> Result<WeatherData, WeatherServiceError> {
        var attempt = 0
        let maxAttempts = retryPolicy.maxAttempts

        while attempt < maxAttempts {
            do {
                // Attempt to fetch weather
                let weather = try await baseService.getCurrentWeather(for: location)

                // Success - cache and return
                cache.saveWeather(weather, for: location)
                await updateServiceStatus(.operational)
                await setLastSuccessfulUpdate(Date())

                return .success(weather)

            } catch let error as WeatherServiceError {
                attempt += 1

                // Handle specific error types
                switch error {
                case .rateLimitExceeded(let retryAfter):
                    await updateServiceStatus(.rateLimited)

                    if let retryAfter = retryAfter {
                        try? await Task.sleep(nanoseconds: UInt64(retryAfter * 1_000_000_000))
                        continue
                    } else {
                        // Use cached data
                        return await fallbackToCachedData(for: location, error: error)
                    }

                case .networkUnavailable, .requestTimeout, .serviceUnavailable:
                    await updateServiceStatus(.degraded)

                    // Retry with exponential backoff
                    if attempt < maxAttempts {
                        let delay = retryPolicy.delay(for: attempt)
                        try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                        continue
                    }

                    // Final attempt failed - use cache
                    return await fallbackToCachedData(for: location, error: error)

                case .locationPermissionDenied, .locationServicesDisabled:
                    await updateServiceStatus(.locationError)
                    return .failure(error)

                case .invalidAPIKey, .quotaExceeded:
                    await updateServiceStatus(.unavailable)
                    return .failure(error)

                default:
                    // Unknown error - try cache
                    return await fallbackToCachedData(for: location, error: error)
                }

            } catch {
                // Unexpected error
                attempt += 1
                if attempt >= maxAttempts {
                    let wrappedError = WeatherServiceError.invalidResponse(details: error.localizedDescription)
                    return await fallbackToCachedData(for: location, error: wrappedError)
                }
            }
        }

        // All attempts exhausted
        return await fallbackToCachedData(
            for: location,
            error: .serviceUnavailable
        )
    }

    /// Get forecast with retry and fallback
    func getForecastWithRetry(for location: CLLocation, days: Int = 2) async -> Result<WeatherForecast, WeatherServiceError> {
        var attempt = 0
        let maxAttempts = retryPolicy.maxAttempts

        while attempt < maxAttempts {
            do {
                let forecast = try await baseService.getWeatherForecast(for: location, days: days)

                // Success - cache and return
                cache.saveForecast(forecast, for: location)
                await updateServiceStatus(.operational)
                await setLastSuccessfulUpdate(Date())

                return .success(forecast)

            } catch let error as WeatherServiceError {
                attempt += 1

                if error.shouldRetry && attempt < maxAttempts {
                    let delay = retryPolicy.delay(for: attempt)
                    try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                }

                // Use cached forecast if available
                if error.shouldUseCachedData,
                   let cachedForecast = cache.loadForecast(for: location) {
                    await updateServiceStatus(.degraded)
                    await setErrorMessage("Using cached forecast data")
                    return .success(cachedForecast)
                }

                return .failure(error)
            } catch {
                attempt += 1
                if attempt >= maxAttempts {
                    if let cachedForecast = cache.loadForecast(for: location) {
                        await updateServiceStatus(.degraded)
                        await setErrorMessage("Using cached forecast data")
                        return .success(cachedForecast)
                    }
                    return .failure(.invalidResponse(details: error.localizedDescription))
                }
            }
        }

        return .failure(.serviceUnavailable)
    }

    /// Force refresh weather data (ignores cache)
    func forceRefresh(for location: CLLocation) async -> Result<WeatherData, WeatherServiceError> {
        return await getWeatherWithRetry(for: location)
    }

    // MARK: - Private Methods

    private func fallbackToCachedData(
        for location: CLLocation,
        error: WeatherServiceError
    ) async -> Result<WeatherData, WeatherServiceError> {
        if let cachedWeather = cache.loadWeather(for: location) {
            // Check cache age
            let cacheAge = Date().timeIntervalSince(cachedWeather.date)

            if cacheAge < 3600 { // Less than 1 hour old
                await updateServiceStatus(.degraded)
                await setErrorMessage("Using recent cached data (updated \(Int(cacheAge/60)) min ago)")
                return .success(cachedWeather)
            } else if cacheAge < 86400 { // Less than 24 hours old
                await updateServiceStatus(.degraded)
                await setErrorMessage("Using cached data from today (may be outdated)")
                return .success(cachedWeather)
            } else {
                // Cache too old
                await updateServiceStatus(.unavailable)
                await setErrorMessage("Weather data unavailable (last update: \(cachedWeather.date.formatted()))")
                return .failure(error)
            }
        }

        // No cache available
        await updateServiceStatus(.unavailable)
        await setErrorMessage(error.errorDescription)
        return .failure(error)
    }

    private func updateServiceStatus(_ status: ServiceStatus) {
        self.serviceStatus = status
    }

    private func setLastSuccessfulUpdate(_ date: Date) {
        self.lastSuccessfulUpdate = date
    }

    private func setErrorMessage(_ message: String?) {
        self.errorMessage = message
    }
}
