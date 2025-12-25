//
//  WeatherKitService.swift
//  InflamAI
//
//  Apple WeatherKit integration for weather data
//  Provides current conditions, forecasts, and flare risk analysis
//

import Foundation
import WeatherKit
import CoreLocation
import SwiftUI

/// WeatherKit service for environmental context data
@MainActor
final class WeatherKitService: NSObject, ObservableObject, CLLocationManagerDelegate {

    // MARK: - Singleton

    static let shared = WeatherKitService()

    // MARK: - Published Properties

    @Published var isAuthorized = false
    @Published var currentWeather: CurrentWeatherData?
    @Published var hourlyForecast: [HourlyWeatherData] = []
    @Published var dailyForecast: [DailyWeatherData] = []
    @Published var currentLocation: CLLocation?
    @Published var isLoading = false
    @Published var errorMessage: String?

    // MARK: - Private Properties

    private let weatherService = WeatherKit.WeatherService()
    private let locationManager = CLLocationManager()
    private var locationContinuation: CheckedContinuation<CLLocation, Error>?
    private var pressureHistory: [(date: Date, pressure: Double)] = []

    // MARK: - Initialization

    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyKilometer
        locationManager.distanceFilter = 1000
    }

    // MARK: - Authorization

    func requestAuthorization() async {
        let status = locationManager.authorizationStatus

        if status == .authorizedWhenInUse || status == .authorizedAlways {
            isAuthorized = true
            locationManager.startUpdatingLocation()
            print("✅ Location already authorized")
            return
        }

        if status == .notDetermined {
            locationManager.requestWhenInUseAuthorization()

            for _ in 0..<100 {
                try? await Task.sleep(nanoseconds: 100_000_000)
                let currentStatus = locationManager.authorizationStatus
                if currentStatus == .authorizedWhenInUse || currentStatus == .authorizedAlways {
                    isAuthorized = true
                    locationManager.startUpdatingLocation()
                    print("✅ Location authorization granted")
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
                continuation.resume(throwing: WeatherKitError.locationUnavailable)
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
        if let location = currentLocation,
           Date().timeIntervalSince(location.timestamp) < 300 {
            return location
        }

        locationManager.requestLocation()

        return try await withCheckedThrowingContinuation { continuation in
            self.locationContinuation = continuation

            Task {
                try? await Task.sleep(nanoseconds: 10_000_000_000)
                if self.locationContinuation != nil {
                    self.locationContinuation = nil
                    continuation.resume(throwing: WeatherKitError.locationUnavailable)
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
            throw WeatherKitError.notAuthorized
        }

        isLoading = true
        errorMessage = nil

        defer { isLoading = false }

        let location = try await getCurrentLocation()

        // Fetch all weather data in one call
        let weather = try await weatherService.weather(for: location)

        // Parse current weather
        let current = weather.currentWeather
        let pressureHPa = current.pressure.value
        let pressureMmHg = pressureHPa * 0.750062

        // Update pressure history
        pressureHistory.append((date: Date(), pressure: pressureMmHg))
        let cutoff = Date().addingTimeInterval(-24 * 3600)
        pressureHistory = pressureHistory.filter { $0.date > cutoff }

        let pressureChange = calculate12HourPressureChange(current: pressureMmHg)

        self.currentWeather = CurrentWeatherData(
            timestamp: Date(),
            temperature: current.temperature.value,
            feelsLike: current.apparentTemperature.value,
            humidity: Int(current.humidity * 100),
            pressure: pressureMmHg,
            pressureChange12h: pressureChange,
            uvIndex: Int(current.uvIndex.value),
            windSpeed: current.wind.speed.value,
            windDirection: current.wind.compassDirection.abbreviation,
            condition: WeatherConditionType.from(current.condition),
            conditionDescription: current.condition.description
        )

        // Parse hourly forecast (next 48 hours)
        self.hourlyForecast = weather.hourlyForecast.prefix(48).map { hour in
            HourlyWeatherData(
                date: hour.date,
                temperature: hour.temperature.value,
                humidity: Int(hour.humidity * 100),
                pressure: hour.pressure.value * 0.750062,
                precipitationChance: Int(hour.precipitationChance * 100),
                condition: WeatherConditionType.from(hour.condition)
            )
        }

        // Parse daily forecast (next 7 days)
        self.dailyForecast = weather.dailyForecast.prefix(7).map { day in
            DailyWeatherData(
                date: day.date,
                temperatureHigh: day.highTemperature.value,
                temperatureLow: day.lowTemperature.value,
                humidity: 0, // Daily doesn't provide average humidity
                precipitationChance: Int(day.precipitationChance * 100),
                condition: WeatherConditionType.from(day.condition),
                sunrise: day.sun.sunrise,
                sunset: day.sun.sunset
            )
        }

        print("✅ Weather data fetched: Current \(current.temperature.value)°C, \(hourlyForecast.count) hourly, \(dailyForecast.count) daily")
    }

    // MARK: - Fetch Current Weather Only

    func fetchCurrentWeather() async throws -> CurrentWeatherData {
        if !isAuthorized {
            await requestAuthorization()
        }

        guard isAuthorized else {
            throw WeatherKitError.notAuthorized
        }

        let location = try await getCurrentLocation()
        let weather = try await weatherService.weather(for: location)
        let current = weather.currentWeather

        let pressureHPa = current.pressure.value
        let pressureMmHg = pressureHPa * 0.750062

        pressureHistory.append((date: Date(), pressure: pressureMmHg))
        let cutoff = Date().addingTimeInterval(-24 * 3600)
        pressureHistory = pressureHistory.filter { $0.date > cutoff }

        let pressureChange = calculate12HourPressureChange(current: pressureMmHg)

        let data = CurrentWeatherData(
            timestamp: Date(),
            temperature: current.temperature.value,
            feelsLike: current.apparentTemperature.value,
            humidity: Int(current.humidity * 100),
            pressure: pressureMmHg,
            pressureChange12h: pressureChange,
            uvIndex: Int(current.uvIndex.value),
            windSpeed: current.wind.speed.value,
            windDirection: current.wind.compassDirection.abbreviation,
            condition: WeatherConditionType.from(current.condition),
            conditionDescription: current.condition.description
        )

        self.currentWeather = data
        return data
    }

    // MARK: - Fetch Hourly Forecast

    func fetchHourlyForecast(hours: Int = 48) async throws -> [HourlyWeatherData] {
        if !isAuthorized {
            await requestAuthorization()
        }

        guard isAuthorized else {
            throw WeatherKitError.notAuthorized
        }

        let location = try await getCurrentLocation()
        let weather = try await weatherService.weather(for: location)

        let forecast = weather.hourlyForecast.prefix(hours).map { hour in
            HourlyWeatherData(
                date: hour.date,
                temperature: hour.temperature.value,
                humidity: Int(hour.humidity * 100),
                pressure: hour.pressure.value * 0.750062,
                precipitationChance: Int(hour.precipitationChance * 100),
                condition: WeatherConditionType.from(hour.condition)
            )
        }

        self.hourlyForecast = Array(forecast)
        return self.hourlyForecast
    }

    // MARK: - Fetch Daily Forecast

    func fetchDailyForecast(days: Int = 7) async throws -> [DailyWeatherData] {
        if !isAuthorized {
            await requestAuthorization()
        }

        guard isAuthorized else {
            throw WeatherKitError.notAuthorized
        }

        let location = try await getCurrentLocation()
        let weather = try await weatherService.weather(for: location)

        let forecast = weather.dailyForecast.prefix(days).map { day in
            DailyWeatherData(
                date: day.date,
                temperatureHigh: day.highTemperature.value,
                temperatureLow: day.lowTemperature.value,
                humidity: 0,
                precipitationChance: Int(day.precipitationChance * 100),
                condition: WeatherConditionType.from(day.condition),
                sunrise: day.sun.sunrise,
                sunset: day.sun.sunset
            )
        }

        self.dailyForecast = Array(forecast)
        return self.dailyForecast
    }

    // MARK: - Pressure Analysis

    private func calculate12HourPressureChange(current: Double) -> Double {
        let twelveHoursAgo = Date().addingTimeInterval(-12 * 3600)
        let historicalPressures = pressureHistory.filter { $0.date < twelveHoursAgo }

        guard let oldestReading = historicalPressures.last else {
            return 0.0
        }

        return current - oldestReading.pressure
    }

    // MARK: - Flare Risk Assessment

    func assessFlareRisk() -> FlareRiskAssessment {
        guard let weather = currentWeather else {
            return FlareRiskAssessment(level: .unknown, score: 0, factors: [], recommendation: "Loading weather data...")
        }

        var riskFactors: [String] = []
        var score = 0.0

        // Rapid pressure drop (>5 mmHg in 12h)
        if weather.pressureChange12h < -5 {
            riskFactors.append("Rapid pressure drop (\(String(format: "%.1f", weather.pressureChange12h)) mmHg)")
            score += 0.3
        }

        // High humidity (>70%)
        if weather.humidity > 70 {
            riskFactors.append("High humidity (\(weather.humidity)%)")
            score += 0.2
        }

        // Cold temperature (<10°C)
        if weather.temperature < 10 {
            riskFactors.append("Cold temperature (\(String(format: "%.0f", weather.temperature))°C)")
            score += 0.2
        }

        // Precipitation
        if weather.condition.hasPrecipitation {
            riskFactors.append("Precipitation (\(weather.condition.displayName))")
            score += 0.1
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
}

// MARK: - Data Models

struct CurrentWeatherData {
    let timestamp: Date
    let temperature: Double          // °C
    let feelsLike: Double            // °C
    let humidity: Int                // 0-100%
    let pressure: Double             // mmHg
    let pressureChange12h: Double    // Δ mmHg
    let uvIndex: Int                 // 0-11+
    let windSpeed: Double            // km/h
    let windDirection: String        // N, NE, E, etc.
    let condition: WeatherConditionType
    let conditionDescription: String
    let airQuality: Int              // European Air Quality Index (1-5), 0 = unavailable

    /// Initialize with all parameters including air quality
    init(timestamp: Date, temperature: Double, feelsLike: Double, humidity: Int,
         pressure: Double, pressureChange12h: Double, uvIndex: Int,
         windSpeed: Double, windDirection: String, condition: WeatherConditionType,
         conditionDescription: String, airQuality: Int = 0) {
        self.timestamp = timestamp
        self.temperature = temperature
        self.feelsLike = feelsLike
        self.humidity = humidity
        self.pressure = pressure
        self.pressureChange12h = pressureChange12h
        self.uvIndex = uvIndex
        self.windSpeed = windSpeed
        self.windDirection = windDirection
        self.condition = condition
        self.conditionDescription = conditionDescription
        self.airQuality = airQuality
    }
}

struct HourlyWeatherData: Identifiable {
    let id = UUID()
    let date: Date
    let temperature: Double
    let humidity: Int
    let pressure: Double             // mmHg
    let precipitationChance: Int     // 0-100%
    let condition: WeatherConditionType
}

struct DailyWeatherData: Identifiable {
    let id = UUID()
    let date: Date
    let temperatureHigh: Double
    let temperatureLow: Double
    let humidity: Int
    let precipitationChance: Int
    let condition: WeatherConditionType
    let sunrise: Date?
    let sunset: Date?
}

struct PressureDataPoint: Identifiable {
    let id = UUID()
    let timestamp: Date
    let pressure: Double             // mmHg
    let change: Double               // Change from previous point
}

// MARK: - Weather Condition Type

enum WeatherConditionType: String, CaseIterable {
    case clear = "clear"
    case partlyCloudy = "partly_cloudy"
    case cloudy = "cloudy"
    case overcast = "overcast"
    case rain = "rain"
    case heavyRain = "heavy_rain"
    case drizzle = "drizzle"
    case snow = "snow"
    case sleet = "sleet"
    case hail = "hail"
    case thunderstorm = "thunderstorm"
    case fog = "fog"
    case haze = "haze"
    case windy = "windy"
    case unknown = "unknown"

    var displayName: String {
        switch self {
        case .clear: return "Clear"
        case .partlyCloudy: return "Partly Cloudy"
        case .cloudy: return "Cloudy"
        case .overcast: return "Overcast"
        case .rain: return "Rain"
        case .heavyRain: return "Heavy Rain"
        case .drizzle: return "Drizzle"
        case .snow: return "Snow"
        case .sleet: return "Sleet"
        case .hail: return "Hail"
        case .thunderstorm: return "Thunderstorm"
        case .fog: return "Fog"
        case .haze: return "Haze"
        case .windy: return "Windy"
        case .unknown: return "Unknown"
        }
    }

    var iconName: String {
        switch self {
        case .clear: return "sun.max.fill"
        case .partlyCloudy: return "cloud.sun.fill"
        case .cloudy: return "cloud.fill"
        case .overcast: return "smoke.fill"
        case .rain: return "cloud.rain.fill"
        case .heavyRain: return "cloud.heavyrain.fill"
        case .drizzle: return "cloud.drizzle.fill"
        case .snow: return "cloud.snow.fill"
        case .sleet: return "cloud.sleet.fill"
        case .hail: return "cloud.hail.fill"
        case .thunderstorm: return "cloud.bolt.rain.fill"
        case .fog: return "cloud.fog.fill"
        case .haze: return "sun.haze.fill"
        case .windy: return "wind"
        case .unknown: return "questionmark.circle"
        }
    }

    var color: Color {
        switch self {
        case .clear: return .yellow
        case .partlyCloudy: return .orange
        case .cloudy, .overcast: return .gray
        case .rain, .heavyRain, .drizzle: return .blue
        case .snow, .sleet, .hail: return .cyan
        case .thunderstorm: return .purple
        case .fog, .haze: return .gray
        case .windy: return .teal
        case .unknown: return .secondary
        }
    }

    var hasPrecipitation: Bool {
        switch self {
        case .rain, .heavyRain, .drizzle, .snow, .sleet, .hail, .thunderstorm:
            return true
        default:
            return false
        }
    }

    static func from(_ condition: WeatherKit.WeatherCondition) -> WeatherConditionType {
        switch condition {
        case .clear, .mostlyClear, .hot:
            return .clear
        case .partlyCloudy:
            return .partlyCloudy
        case .cloudy, .mostlyCloudy:
            return .cloudy
        case .rain:
            return .rain
        case .heavyRain:
            return .heavyRain
        case .drizzle:
            return .drizzle
        case .snow, .flurries, .heavySnow, .blizzard:
            return .snow
        case .sleet, .freezingRain, .freezingDrizzle, .wintryMix:
            return .sleet
        case .hail:
            return .hail
        case .thunderstorms, .tropicalStorm, .hurricane, .isolatedThunderstorms, .scatteredThunderstorms, .strongStorms:
            return .thunderstorm
        case .foggy:
            return .fog
        case .haze, .smoky:
            return .haze
        case .windy, .breezy:
            return .windy
        case .blowingDust, .blowingSnow:
            return .windy
        @unknown default:
            return .unknown
        }
    }
}

// MARK: - Flare Risk Assessment

enum FlareRiskLevel: String {
    case unknown = "Unknown"
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"

    var color: Color {
        switch self {
        case .unknown: return .gray
        case .low: return .green
        case .moderate: return .orange
        case .high: return .red
        }
    }
}

struct FlareRiskAssessment {
    let level: FlareRiskLevel
    let score: Double                // 0-1
    let factors: [String]
    let recommendation: String
}

// MARK: - Errors

enum WeatherKitError: LocalizedError {
    case notAuthorized
    case locationUnavailable
    case dataUnavailable
    case networkError

    var errorDescription: String? {
        switch self {
        case .notAuthorized:
            return "Location access not authorized. Please enable in Settings."
        case .locationUnavailable:
            return "Current location unavailable"
        case .dataUnavailable:
            return "Weather data unavailable"
        case .networkError:
            return "Network error. Please check your connection."
        }
    }
}
