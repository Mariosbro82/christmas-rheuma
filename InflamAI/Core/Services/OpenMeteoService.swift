//
//  OpenMeteoService.swift
//  InflamAI
//
//  Open-Meteo API integration - FREE, no API key required
//  Drop-in replacement for WeatherKit that works with Personal Team accounts
//
//  API Documentation: https://open-meteo.com/en/docs
//

import Foundation
import CoreLocation
import SwiftUI

/// Open-Meteo weather service - FREE alternative to WeatherKit
/// No API key required, no entitlements needed, works with Personal Team
@MainActor
final class OpenMeteoService: NSObject, ObservableObject, CLLocationManagerDelegate {

    // MARK: - Singleton

    static let shared = OpenMeteoService()

    // MARK: - Published Properties (same interface as WeatherKitService)

    @Published var isAuthorized = false
    @Published var currentWeather: CurrentWeatherData?
    @Published var hourlyForecast: [HourlyWeatherData] = []
    @Published var dailyForecast: [DailyWeatherData] = []
    @Published var currentLocation: CLLocation?
    @Published var isLoading = false
    @Published var errorMessage: String?

    // MARK: - Private Properties

    private let locationManager = CLLocationManager()
    private var locationContinuation: CheckedContinuation<CLLocation, Error>?
    private let urlSession: URLSession
    private let pressureHistoryManager = PressureHistoryManager()

    // Cache
    private var cachedCurrentWeather: (data: CurrentWeatherData, timestamp: Date)?
    private var cachedHourlyForecast: (data: [HourlyWeatherData], timestamp: Date)?
    private var cachedDailyForecast: (data: [DailyWeatherData], timestamp: Date)?

    // Cache TTLs
    private let currentWeatherTTL: TimeInterval = 900      // 15 minutes
    private let hourlyForecastTTL: TimeInterval = 1800     // 30 minutes
    private let dailyForecastTTL: TimeInterval = 3600      // 1 hour

    // UserDefaults keys for persistent fallback cache
    private let userDefaultsWeatherKey = "lastSuccessfulWeatherData"
    private let userDefaultsWeatherTimestampKey = "lastSuccessfulWeatherTimestamp"
    private let userDefaultsFallbackMaxAge: TimeInterval = 86400  // 24 hours

    // MARK: - Initialization

    override init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.urlSession = URLSession(configuration: config)

        super.init()

        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyKilometer
        locationManager.distanceFilter = 1000

        print("🌤️ OpenMeteoService initialized - FREE weather API, no key required")
    }

    // MARK: - Authorization

    func requestAuthorization() async {
        let status = locationManager.authorizationStatus

        if status == .authorizedWhenInUse || status == .authorizedAlways {
            isAuthorized = true
            locationManager.startUpdatingLocation()
            print("✅ Location already authorized for OpenMeteo")
            return
        }

        if status == .notDetermined {
            locationManager.requestWhenInUseAuthorization()

            // Wait for authorization response
            for _ in 0..<100 {
                try? await Task.sleep(nanoseconds: 100_000_000)
                let currentStatus = locationManager.authorizationStatus
                if currentStatus == .authorizedWhenInUse || currentStatus == .authorizedAlways {
                    isAuthorized = true
                    locationManager.startUpdatingLocation()
                    print("✅ Location authorization granted for OpenMeteo")
                    return
                } else if currentStatus == .denied || currentStatus == .restricted {
                    print("❌ Location authorization denied")
                    return
                }
            }
        }

        print("⚠️ Location authorization timeout or denied")
    }

    // MARK: - CLLocationManagerDelegate

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        Task { @MainActor in
            self.currentLocation = location
            if let continuation = self.locationContinuation {
                self.locationContinuation = nil
                continuation.resume(returning: location)
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("⚠️ Location manager error: \(error.localizedDescription)")
        Task { @MainActor in
            if let continuation = self.locationContinuation {
                self.locationContinuation = nil
                continuation.resume(throwing: OpenMeteoError.locationUnavailable)
            }
        }
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        Task { @MainActor in
            self.isAuthorized = (status == .authorizedWhenInUse || status == .authorizedAlways)
            if self.isAuthorized {
                manager.startUpdatingLocation()
            }
        }
    }

    // MARK: - Location Helper

    func getCurrentLocation() async throws -> CLLocation {
        // Return cached location if recent
        if let location = currentLocation,
           Date().timeIntervalSince(location.timestamp) < 300 {
            return location
        }

        // FIXED: Race condition where requestLocation() was called BEFORE setting continuation
        // This caused the delegate to fire before continuation was set, leaking it
        return try await withCheckedThrowingContinuation { [weak self] continuation in
            guard let self = self else {
                continuation.resume(throwing: OpenMeteoError.locationUnavailable)
                return
            }

            // Cancel any existing continuation FIRST to prevent leaks
            if let existingContinuation = self.locationContinuation {
                self.locationContinuation = nil
                existingContinuation.resume(throwing: OpenMeteoError.locationUnavailable)
            }

            // Set continuation BEFORE requesting location (fixes race condition)
            self.locationContinuation = continuation

            // NOW request location - delegate can safely resume the continuation
            self.locationManager.requestLocation()

            // Timeout task with guaranteed resume
            Task { @MainActor [weak self] in
                // Use shorter timeout (5 seconds) and handle cancellation
                do {
                    try await Task.sleep(nanoseconds: 5_000_000_000)  // 5 second timeout
                } catch {
                    // Task was cancelled - continuation should be handled elsewhere
                    return
                }

                guard let self = self else { return }

                // Only resume if this continuation is still active (not already resumed by delegate)
                if let cont = self.locationContinuation {
                    self.locationContinuation = nil
                    cont.resume(throwing: OpenMeteoError.locationUnavailable)
                }
            }
        }
    }

    // MARK: - Fetch All Weather Data

    /// Fetches current weather, hourly (48h), and daily (7-day) forecasts
    func fetchAllWeatherData() async throws {
        if !isAuthorized {
            await requestAuthorization()
        }

        guard isAuthorized else {
            throw OpenMeteoError.notAuthorized
        }

        isLoading = true
        errorMessage = nil

        defer { isLoading = false }

        let location = try await getCurrentLocation()

        // Build API URL
        let url = buildAPIURL(latitude: location.coordinate.latitude, longitude: location.coordinate.longitude)

        // Fetch data
        let (data, response) = try await urlSession.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenMeteoError.networkError
        }

        guard httpResponse.statusCode == 200 else {
            throw OpenMeteoError.serverError(statusCode: httpResponse.statusCode)
        }

        // Parse response
        let decoder = JSONDecoder()
        let apiResponse = try decoder.decode(OpenMeteoResponse.self, from: data)

        // Process current weather
        let pressureHPa = apiResponse.current.pressure_msl
        let pressureMmHg = pressureHPa * 0.750062

        // Record pressure for history tracking
        await pressureHistoryManager.record(pressureMmHg)
        let pressureChange12h = await pressureHistoryManager.change12h() ?? 0

        self.currentWeather = CurrentWeatherData(
            timestamp: Date(),
            temperature: apiResponse.current.temperature_2m,
            feelsLike: apiResponse.current.apparent_temperature,
            humidity: Int(apiResponse.current.relative_humidity_2m),
            pressure: pressureMmHg,
            pressureChange12h: pressureChange12h,
            uvIndex: Int(apiResponse.current.uv_index ?? 0),
            windSpeed: apiResponse.current.wind_speed_10m,
            windDirection: compassDirection(from: apiResponse.current.wind_direction_10m),
            condition: WeatherConditionType.fromWMOCode(apiResponse.current.weather_code),
            conditionDescription: WMOWeatherCode(rawValue: apiResponse.current.weather_code)?.description ?? "Unknown"
        )

        // Process hourly forecast
        self.hourlyForecast = zip(apiResponse.hourly.time, apiResponse.hourly.temperature_2m.indices).prefix(48).compactMap { (timeString, index) in
            guard let date = parseISO8601Date(timeString) else { return nil }

            return HourlyWeatherData(
                date: date,
                temperature: apiResponse.hourly.temperature_2m[index],
                humidity: Int(apiResponse.hourly.relative_humidity_2m[index]),
                pressure: apiResponse.hourly.pressure_msl[index] * 0.750062,
                precipitationChance: Int(apiResponse.hourly.precipitation_probability?[index] ?? 0),
                condition: WeatherConditionType.fromWMOCode(apiResponse.hourly.weather_code[index])
            )
        }

        // Process daily forecast
        self.dailyForecast = zip(apiResponse.daily.time, apiResponse.daily.temperature_2m_max.indices).prefix(7).compactMap { (timeString, index) in
            guard let date = parseDateOnly(timeString) else { return nil }

            return DailyWeatherData(
                date: date,
                temperatureHigh: apiResponse.daily.temperature_2m_max[index],
                temperatureLow: apiResponse.daily.temperature_2m_min[index],
                humidity: 0, // Open-Meteo daily doesn't provide average humidity
                precipitationChance: Int(apiResponse.daily.precipitation_probability_max?[index] ?? 0),
                condition: WeatherConditionType.fromWMOCode(apiResponse.daily.weather_code[index]),
                sunrise: parseISO8601Date(apiResponse.daily.sunrise?[index] ?? ""),
                sunset: parseISO8601Date(apiResponse.daily.sunset?[index] ?? "")
            )
        }

        // Update cache
        cachedCurrentWeather = (self.currentWeather!, Date())
        cachedHourlyForecast = (self.hourlyForecast, Date())
        cachedDailyForecast = (self.dailyForecast, Date())

        print("✅ OpenMeteo data fetched: Current \(apiResponse.current.temperature_2m)°C, \(hourlyForecast.count) hourly, \(dailyForecast.count) daily")
    }

    // MARK: - Fetch Current Weather Only

    func fetchCurrentWeather() async throws -> CurrentWeatherData {
        // Check cache
        if let cached = cachedCurrentWeather,
           Date().timeIntervalSince(cached.timestamp) < currentWeatherTTL {
            return cached.data
        }

        if !isAuthorized {
            await requestAuthorization()
        }

        guard isAuthorized else {
            throw OpenMeteoError.notAuthorized
        }

        let location = try await getCurrentLocation()

        // Build URL for current weather only
        var components = URLComponents(string: "https://api.open-meteo.com/v1/forecast")!
        components.queryItems = [
            URLQueryItem(name: "latitude", value: String(location.coordinate.latitude)),
            URLQueryItem(name: "longitude", value: String(location.coordinate.longitude)),
            URLQueryItem(name: "current", value: "temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,surface_pressure,weather_code,wind_speed_10m,wind_direction_10m,precipitation,uv_index"),
            URLQueryItem(name: "timezone", value: "auto")
        ]

        let (data, response) = try await urlSession.data(from: components.url!)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw OpenMeteoError.networkError
        }

        let decoder = JSONDecoder()
        let apiResponse = try decoder.decode(OpenMeteoResponse.self, from: data)

        let pressureMmHg = apiResponse.current.pressure_msl * 0.750062
        await pressureHistoryManager.record(pressureMmHg)
        let pressureChange12h = await pressureHistoryManager.change12h() ?? 0

        let weatherData = CurrentWeatherData(
            timestamp: Date(),
            temperature: apiResponse.current.temperature_2m,
            feelsLike: apiResponse.current.apparent_temperature,
            humidity: Int(apiResponse.current.relative_humidity_2m),
            pressure: pressureMmHg,
            pressureChange12h: pressureChange12h,
            uvIndex: Int(apiResponse.current.uv_index ?? 0),
            windSpeed: apiResponse.current.wind_speed_10m,
            windDirection: compassDirection(from: apiResponse.current.wind_direction_10m),
            condition: WeatherConditionType.fromWMOCode(apiResponse.current.weather_code),
            conditionDescription: WMOWeatherCode(rawValue: apiResponse.current.weather_code)?.description ?? "Unknown"
        )

        self.currentWeather = weatherData
        cachedCurrentWeather = (weatherData, Date())

        return weatherData
    }

    // MARK: - Fetch Hourly Forecast

    func fetchHourlyForecast(hours: Int = 48) async throws -> [HourlyWeatherData] {
        // Check cache
        if let cached = cachedHourlyForecast,
           Date().timeIntervalSince(cached.timestamp) < hourlyForecastTTL {
            return Array(cached.data.prefix(hours))
        }

        try await fetchAllWeatherData()
        return Array(hourlyForecast.prefix(hours))
    }

    // MARK: - Fetch Daily Forecast

    func fetchDailyForecast(days: Int = 7) async throws -> [DailyWeatherData] {
        // Check cache
        if let cached = cachedDailyForecast,
           Date().timeIntervalSince(cached.timestamp) < dailyForecastTTL {
            return Array(cached.data.prefix(days))
        }

        try await fetchAllWeatherData()
        return Array(dailyForecast.prefix(days))
    }

    // MARK: - Fetch Air Quality

    /// Fetch current air quality from Open-Meteo Air Quality API
    /// Returns European Air Quality Index (1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor)
    /// Returns 0 if unavailable
    func fetchAirQuality() async -> Int {
        // Get location - use cached or fetch new
        var location: CLLocation?
        if let cached = currentLocation {
            location = cached
        } else {
            location = try? await getCurrentLocation()
        }

        guard let loc = location else {
            print("⚠️ [AirQuality] No location available")
            return 0
        }

        let locationToUse = loc

        // Build Air Quality API URL
        // API docs: https://open-meteo.com/en/docs/air-quality-api
        var components = URLComponents(string: "https://air-quality-api.open-meteo.com/v1/air-quality")!
        components.queryItems = [
            URLQueryItem(name: "latitude", value: String(locationToUse.coordinate.latitude)),
            URLQueryItem(name: "longitude", value: String(locationToUse.coordinate.longitude)),
            URLQueryItem(name: "current", value: "european_aqi,us_aqi,pm2_5,pm10"),
            URLQueryItem(name: "timezone", value: "auto")
        ]

        do {
            let (data, response) = try await urlSession.data(from: components.url!)

            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                print("⚠️ [AirQuality] API request failed")
                return 0
            }

            let decoder = JSONDecoder()
            let apiResponse = try decoder.decode(AirQualityResponse.self, from: data)

            let aqi = apiResponse.current.european_aqi ?? 0
            print("✅ [AirQuality] Fetched: EAQI=\(aqi), PM2.5=\(apiResponse.current.pm2_5 ?? 0)")
            return aqi

        } catch {
            print("⚠️ [AirQuality] Fetch error: \(error.localizedDescription)")
            return 0
        }
    }

    // MARK: - Flare Risk Assessment

    func assessFlareRisk() -> FlareRiskAssessment {
        guard let weather = currentWeather else {
            return FlareRiskAssessment(level: .unknown, score: 0, factors: [], recommendation: "Loading weather data...")
        }

        var riskFactors: [String] = []
        var score = 0.0

        // Rapid pressure drop (>5 mmHg in 12h) - Most important for AS
        if weather.pressureChange12h < -5 {
            riskFactors.append("Rapid pressure drop (\(String(format: "%.1f", weather.pressureChange12h)) mmHg)")
            score += 0.35
        } else if weather.pressureChange12h < -3 {
            riskFactors.append("Moderate pressure drop (\(String(format: "%.1f", weather.pressureChange12h)) mmHg)")
            score += 0.15
        }

        // High humidity (>70%)
        if weather.humidity > 80 {
            riskFactors.append("Very high humidity (\(weather.humidity)%)")
            score += 0.25
        } else if weather.humidity > 70 {
            riskFactors.append("High humidity (\(weather.humidity)%)")
            score += 0.15
        }

        // Cold temperature (<10°C)
        if weather.temperature < 5 {
            riskFactors.append("Very cold temperature (\(String(format: "%.0f", weather.temperature))°C)")
            score += 0.20
        } else if weather.temperature < 10 {
            riskFactors.append("Cold temperature (\(String(format: "%.0f", weather.temperature))°C)")
            score += 0.10
        }

        // Precipitation
        if weather.condition.hasPrecipitation {
            riskFactors.append("\(weather.condition.displayName)")
            score += 0.10
        }

        // Low absolute pressure (storm system)
        if weather.pressure < 745 {
            riskFactors.append("Low pressure system (\(String(format: "%.0f", weather.pressure)) mmHg)")
            score += 0.10
        }

        // Determine risk level
        let level: FlareRiskLevel
        let recommendation: String

        if score >= 0.6 {
            level = .high
            recommendation = "High flare risk - consider preventive measures, rest, and gentle stretching"
        } else if score >= 0.3 {
            level = .moderate
            recommendation = "Moderate flare risk - monitor symptoms and stay prepared"
        } else {
            level = .low
            recommendation = "Low flare risk - good conditions for activities"
        }

        return FlareRiskAssessment(
            level: level,
            score: min(1.0, score),
            factors: riskFactors,
            recommendation: recommendation
        )
    }

    // MARK: - Pressure Forecast for Chart

    func getPressureForecast() -> [PressureDataPoint] {
        var points: [PressureDataPoint] = []

        // Current pressure
        if let current = currentWeather {
            points.append(PressureDataPoint(
                timestamp: current.timestamp,
                pressure: current.pressure,
                change: 0
            ))
        }

        // Hourly forecast
        for (index, hour) in hourlyForecast.enumerated() {
            let previousPressure = index == 0 ? (currentWeather?.pressure ?? hour.pressure) : hourlyForecast[index - 1].pressure
            let change = hour.pressure - previousPressure

            points.append(PressureDataPoint(
                timestamp: hour.date,
                pressure: hour.pressure,
                change: change
            ))
        }

        return points
    }

    // MARK: - Helper Methods

    private func buildAPIURL(latitude: Double, longitude: Double) -> URL {
        var components = URLComponents(string: "https://api.open-meteo.com/v1/forecast")!
        components.queryItems = [
            URLQueryItem(name: "latitude", value: String(latitude)),
            URLQueryItem(name: "longitude", value: String(longitude)),
            URLQueryItem(name: "current", value: "temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,surface_pressure,weather_code,wind_speed_10m,wind_direction_10m,precipitation,uv_index"),
            URLQueryItem(name: "hourly", value: "temperature_2m,relative_humidity_2m,pressure_msl,precipitation_probability,weather_code"),
            URLQueryItem(name: "daily", value: "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code,sunrise,sunset"),
            URLQueryItem(name: "timezone", value: "auto"),
            URLQueryItem(name: "forecast_days", value: "7")
        ]
        return components.url!
    }

    private func compassDirection(from degrees: Double) -> String {
        let directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        let index = Int((degrees + 11.25) / 22.5) % 16
        return directions[index]
    }

    private func parseISO8601Date(_ string: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: string) {
            return date
        }
        formatter.formatOptions = [.withInternetDateTime]
        if let date = formatter.date(from: string) {
            return date
        }
        // Try without timezone (Open-Meteo local time format)
        let localFormatter = DateFormatter()
        localFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm"
        return localFormatter.date(from: string)
    }

    private func parseDateOnly(_ string: String) -> Date? {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.date(from: string)
    }

    // MARK: - Persistent Fallback Cache

    /// Saves weather data to UserDefaults as a fallback for when API fails
    private func saveToFallbackCache(_ weather: CurrentWeatherData) {
        let fallback = FallbackWeatherData(
            temperature: weather.temperature,
            humidity: weather.humidity,
            pressure: weather.pressure,
            pressureChange12h: weather.pressureChange12h
        )

        if let encoded = try? JSONEncoder().encode(fallback) {
            UserDefaults.standard.set(encoded, forKey: userDefaultsWeatherKey)
            UserDefaults.standard.set(Date(), forKey: userDefaultsWeatherTimestampKey)
            print("💾 Weather fallback cache saved: \(weather.pressure) mmHg, \(weather.temperature)°C")
        }
    }

    /// Loads weather data from UserDefaults fallback cache (max 24h old)
    func loadFromFallbackCache() -> CurrentWeatherData? {
        guard let timestamp = UserDefaults.standard.object(forKey: userDefaultsWeatherTimestampKey) as? Date,
              Date().timeIntervalSince(timestamp) < userDefaultsFallbackMaxAge else {
            print("⚠️ Weather fallback cache expired or not found")
            return nil
        }

        guard let data = UserDefaults.standard.data(forKey: userDefaultsWeatherKey),
              let fallback = try? JSONDecoder().decode(FallbackWeatherData.self, from: data) else {
            return nil
        }

        print("✅ Loaded weather from fallback cache (age: \(Int(Date().timeIntervalSince(timestamp) / 60)) minutes)")

        return CurrentWeatherData(
            timestamp: timestamp,
            temperature: fallback.temperature,
            feelsLike: fallback.temperature,
            humidity: fallback.humidity,
            pressure: fallback.pressure,
            pressureChange12h: fallback.pressureChange12h,
            uvIndex: 0,
            windSpeed: 0,
            windDirection: "N",
            condition: .unknown,
            conditionDescription: "Cached data"
        )
    }

    /// Fetches current weather with automatic fallback to cached data
    func fetchCurrentWeatherWithFallback() async -> CurrentWeatherData? {
        do {
            let weather = try await fetchCurrentWeather()
            saveToFallbackCache(weather)
            return weather
        } catch {
            print("⚠️ Weather API failed: \(error.localizedDescription) - trying fallback cache")
            return loadFromFallbackCache()
        }
    }
}

// MARK: - Fallback Weather Data Model

struct FallbackWeatherData: Codable {
    let temperature: Double
    let humidity: Int
    let pressure: Double
    let pressureChange12h: Double
}

// MARK: - Open-Meteo API Response Models

struct OpenMeteoResponse: Codable {
    let latitude: Double
    let longitude: Double
    let timezone: String
    let current: OpenMeteoCurrentData
    let hourly: OpenMeteoHourlyData
    let daily: OpenMeteoDailyData
}

struct OpenMeteoCurrentData: Codable {
    let time: String
    let temperature_2m: Double
    let relative_humidity_2m: Double
    let apparent_temperature: Double
    let pressure_msl: Double
    let surface_pressure: Double?
    let weather_code: Int
    let wind_speed_10m: Double
    let wind_direction_10m: Double
    let precipitation: Double?
    let uv_index: Double?
}

struct OpenMeteoHourlyData: Codable {
    let time: [String]
    let temperature_2m: [Double]
    let relative_humidity_2m: [Double]
    let pressure_msl: [Double]
    let precipitation_probability: [Double]?
    let weather_code: [Int]
}

struct OpenMeteoDailyData: Codable {
    let time: [String]
    let temperature_2m_max: [Double]
    let temperature_2m_min: [Double]
    let precipitation_probability_max: [Double]?
    let weather_code: [Int]
    let sunrise: [String]?
    let sunset: [String]?
}

// MARK: - WMO Weather Codes

enum WMOWeatherCode: Int {
    case clearSky = 0
    case mainlyClear = 1
    case partlyCloudy = 2
    case overcast = 3
    case fog = 45
    case depositingRimeFog = 48
    case drizzleLight = 51
    case drizzleModerate = 53
    case drizzleDense = 55
    case freezingDrizzleLight = 56
    case freezingDrizzleDense = 57
    case rainSlight = 61
    case rainModerate = 63
    case rainHeavy = 65
    case freezingRainLight = 66
    case freezingRainHeavy = 67
    case snowFallSlight = 71
    case snowFallModerate = 73
    case snowFallHeavy = 75
    case snowGrains = 77
    case rainShowersSlight = 80
    case rainShowersModerate = 81
    case rainShowersViolent = 82
    case snowShowersSlight = 85
    case snowShowersHeavy = 86
    case thunderstorm = 95
    case thunderstormSlightHail = 96
    case thunderstormHeavyHail = 99

    var description: String {
        switch self {
        case .clearSky: return "Clear sky"
        case .mainlyClear: return "Mainly clear"
        case .partlyCloudy: return "Partly cloudy"
        case .overcast: return "Overcast"
        case .fog, .depositingRimeFog: return "Fog"
        case .drizzleLight: return "Light drizzle"
        case .drizzleModerate: return "Moderate drizzle"
        case .drizzleDense: return "Dense drizzle"
        case .freezingDrizzleLight, .freezingDrizzleDense: return "Freezing drizzle"
        case .rainSlight: return "Light rain"
        case .rainModerate: return "Moderate rain"
        case .rainHeavy: return "Heavy rain"
        case .freezingRainLight, .freezingRainHeavy: return "Freezing rain"
        case .snowFallSlight: return "Light snow"
        case .snowFallModerate: return "Moderate snow"
        case .snowFallHeavy: return "Heavy snow"
        case .snowGrains: return "Snow grains"
        case .rainShowersSlight: return "Light rain showers"
        case .rainShowersModerate: return "Moderate rain showers"
        case .rainShowersViolent: return "Violent rain showers"
        case .snowShowersSlight, .snowShowersHeavy: return "Snow showers"
        case .thunderstorm: return "Thunderstorm"
        case .thunderstormSlightHail, .thunderstormHeavyHail: return "Thunderstorm with hail"
        }
    }

    var condition: WeatherConditionType {
        switch self {
        case .clearSky, .mainlyClear:
            return .clear
        case .partlyCloudy:
            return .partlyCloudy
        case .overcast:
            return .overcast
        case .fog, .depositingRimeFog:
            return .fog
        case .drizzleLight, .drizzleModerate, .drizzleDense:
            return .drizzle
        case .freezingDrizzleLight, .freezingDrizzleDense, .freezingRainLight, .freezingRainHeavy:
            return .sleet
        case .rainSlight, .rainShowersModerate:
            return .rain
        case .rainModerate, .rainHeavy, .rainShowersViolent:
            return .heavyRain
        case .rainShowersSlight:
            return .rain
        case .snowFallSlight, .snowFallModerate, .snowFallHeavy, .snowGrains, .snowShowersSlight, .snowShowersHeavy:
            return .snow
        case .thunderstorm, .thunderstormSlightHail, .thunderstormHeavyHail:
            return .thunderstorm
        }
    }
}

// MARK: - WeatherConditionType Extension for WMO Codes

extension WeatherConditionType {
    static func fromWMOCode(_ code: Int) -> WeatherConditionType {
        guard let wmoCode = WMOWeatherCode(rawValue: code) else {
            // Handle codes not in enum
            switch code {
            case 0...1: return .clear
            case 2: return .partlyCloudy
            case 3: return .overcast
            case 45, 48: return .fog
            case 51...57: return .drizzle
            case 61...67: return .rain
            case 71...77: return .snow
            case 80...82: return .rain
            case 85, 86: return .snow
            case 95...99: return .thunderstorm
            default: return .unknown
            }
        }
        return wmoCode.condition
    }
}

// MARK: - Air Quality API Response

struct AirQualityResponse: Codable {
    let current: AirQualityCurrentData
}

struct AirQualityCurrentData: Codable {
    let european_aqi: Int?          // European Air Quality Index (1-5)
    let us_aqi: Int?                // US Air Quality Index
    let pm2_5: Double?              // PM2.5 particulate matter
    let pm10: Double?               // PM10 particulate matter
}

// MARK: - Errors

enum OpenMeteoError: LocalizedError {
    case notAuthorized
    case locationUnavailable
    case networkError
    case serverError(statusCode: Int)
    case decodingError(Error)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .notAuthorized:
            return "Location access not authorized. Please enable in Settings."
        case .locationUnavailable:
            return "Current location unavailable"
        case .networkError:
            return "Network error. Please check your connection."
        case .serverError(let code):
            return "Server error (code: \(code))"
        case .decodingError(let error):
            return "Failed to parse weather data: \(error.localizedDescription)"
        case .invalidResponse:
            return "Invalid response from weather service"
        }
    }
}
