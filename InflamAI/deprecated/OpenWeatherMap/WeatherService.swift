//
//  WeatherService.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CoreLocation
import Combine

// MARK: - Weather Data Models

struct WeatherData: Codable, Identifiable {
    let id = UUID()
    let date: Date
    let temperature: Double // Celsius
    let humidity: Double // Percentage
    let barometricPressure: Double // hPa
    let windSpeed: Double // km/h
    let precipitation: Double // mm
    let uvIndex: Double
    let condition: WeatherCondition
    let location: WeatherLocation
    
    enum CodingKeys: String, CodingKey {
        case date, temperature, humidity, barometricPressure, windSpeed, precipitation, uvIndex, condition, location
    }
}

struct WeatherLocation: Codable {
    let latitude: Double
    let longitude: Double
    let city: String
    let country: String
}

enum WeatherCondition: String, Codable, CaseIterable {
    case clear = "clear"
    case partlyCloudy = "partly_cloudy"
    case cloudy = "cloudy"
    case overcast = "overcast"
    case rain = "rain"
    case heavyRain = "heavy_rain"
    case snow = "snow"
    case thunderstorm = "thunderstorm"
    case fog = "fog"
    case windy = "windy"
    
    var displayName: String {
        switch self {
        case .clear: return "Clear"
        case .partlyCloudy: return "Partly Cloudy"
        case .cloudy: return "Cloudy"
        case .overcast: return "Overcast"
        case .rain: return "Rain"
        case .heavyRain: return "Heavy Rain"
        case .snow: return "Snow"
        case .thunderstorm: return "Thunderstorm"
        case .fog: return "Fog"
        case .windy: return "Windy"
        }
    }
    
    var iconName: String {
        switch self {
        case .clear: return "sun.max"
        case .partlyCloudy: return "cloud.sun"
        case .cloudy: return "cloud"
        case .overcast: return "cloud.fill"
        case .rain: return "cloud.rain"
        case .heavyRain: return "cloud.heavyrain"
        case .snow: return "cloud.snow"
        case .thunderstorm: return "cloud.bolt"
        case .fog: return "cloud.fog"
        case .windy: return "wind"
        }
    }
}

struct WeatherForecast: Codable {
    let current: WeatherData
    let hourly: [WeatherData]
    let daily: [WeatherData]
    let lastUpdated: Date
}

struct WeatherAlert: Codable, Identifiable {
    let id = UUID()
    let title: String
    let description: String
    let severity: AlertSeverity
    let startDate: Date
    let endDate: Date
    let affectedAreas: [String]
    
    enum AlertSeverity: String, Codable {
        case minor = "minor"
        case moderate = "moderate"
        case severe = "severe"
        case extreme = "extreme"
    }
    
    enum CodingKeys: String, CodingKey {
        case title, description, severity, startDate, endDate, affectedAreas
    }
}

// MARK: - Weather Service Protocol
// Note: WeatherServiceError is defined in WeatherServiceError.swift

protocol WeatherServiceProtocol {
    func getCurrentWeather(for location: CLLocation) async throws -> WeatherData
    func getWeatherForecast(for location: CLLocation, days: Int) async throws -> WeatherForecast
    func getHistoricalWeather(for location: CLLocation, from startDate: Date, to endDate: Date) async throws -> [WeatherData]
    func getWeatherAlerts(for location: CLLocation) async throws -> [WeatherAlert]
    func searchLocations(query: String) async throws -> [WeatherLocation]
}

// MARK: - Weather Service Implementation

class WeatherService: NSObject, ObservableObject, WeatherServiceProtocol {
    // MARK: - Properties
    
    @Published var currentWeather: WeatherData?
    @Published var forecast: WeatherForecast?
    @Published var alerts: [WeatherAlert] = []
    @Published var isLoading = false
    @Published var error: WeatherServiceError?
    
    private let locationManager = CLLocationManager()
    private let session = URLSession.shared
    private let apiKey: String
    private let baseURL = "https://api.openweathermap.org/data/2.5"
    private let cache = NSCache<NSString, NSData>()
    private let cacheTimeout: TimeInterval = 600 // 10 minutes
    
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization

    init(apiKey: String = "") {
        self.apiKey = apiKey.isEmpty ? Bundle.main.object(forInfoDictionaryKey: "WEATHER_API_KEY") as? String ?? "" : apiKey
        super.init()
        setupLocationManager()
        setupCache()
    }
    
    // MARK: - Public Methods
    
    func getCurrentWeather(for location: CLLocation) async throws -> WeatherData {
        let cacheKey = "current_\(location.coordinate.latitude)_\(location.coordinate.longitude)"
        
        if let cachedData = getCachedData(for: cacheKey) {
            return cachedData
        }
        
        let url = buildURL(endpoint: "weather", location: location)
        let data = try await fetchData(from: url)
        let weatherData = try parseCurrentWeatherResponse(data, location: location)
        
        cacheData(weatherData, for: cacheKey)
        
        await MainActor.run {
            self.currentWeather = weatherData
        }
        
        return weatherData
    }
    
    func getWeatherForecast(for location: CLLocation, days: Int = 5) async throws -> WeatherForecast {
        let cacheKey = "forecast_\(location.coordinate.latitude)_\(location.coordinate.longitude)_\(days)"
        
        if let cachedForecast = getCachedForecast(for: cacheKey) {
            return cachedForecast
        }
        
        async let currentTask = getCurrentWeather(for: location)
        async let forecastTask = getDetailedForecast(for: location, days: days)
        
        let (current, (hourly, daily)) = try await (currentTask, forecastTask)
        
        let forecast = WeatherForecast(
            current: current,
            hourly: hourly,
            daily: daily,
            lastUpdated: Date()
        )
        
        cacheForecast(forecast, for: cacheKey)
        
        await MainActor.run {
            self.forecast = forecast
        }
        
        return forecast
    }
    
    func getHistoricalWeather(for location: CLLocation, from startDate: Date, to endDate: Date) async throws -> [WeatherData] {
        let calendar = Calendar.current
        let daysBetween = endDate.timeIntervalSince(startDate)
        let numberOfDays = Int(daysBetween / 86400) // Convert seconds to days

        guard numberOfDays <= 365 else {
            throw WeatherServiceError.invalidResponse(details: "Historical data request exceeds 365 days")
        }
        
        var historicalData: [WeatherData] = []
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        
        // For demo purposes, generate mock historical data
        // In a real implementation, this would call a historical weather API
        for dayOffset in 0..<numberOfDays {
            guard let date = calendar.date(byAdding: .day, value: dayOffset, to: startDate) else { continue }
            
            let mockWeather = generateMockWeatherData(for: location, date: date)
            historicalData.append(mockWeather)
        }
        
        return historicalData
    }
    
    func getWeatherAlerts(for location: CLLocation) async throws -> [WeatherAlert] {
        // For demo purposes, return mock alerts
        // In a real implementation, this would fetch from weather service
        let mockAlerts = generateMockWeatherAlerts(for: location)
        
        await MainActor.run {
            self.alerts = mockAlerts
        }
        
        return mockAlerts
    }
    
    func searchLocations(query: String) async throws -> [WeatherLocation] {
        guard !query.isEmpty else { return [] }
        
        let url = buildGeocodingURL(query: query)
        let data = try await fetchData(from: url)
        return try parseLocationSearchResponse(data)
    }
    
    func requestLocationPermission() {
        locationManager.requestWhenInUseAuthorization()
    }
    
    func getCurrentLocation() async throws -> CLLocation {
        return try await withCheckedThrowingContinuation { continuation in
            locationManager.requestLocation()
            
            // Set up one-time location update
            var cancellable: AnyCancellable?
            cancellable = locationManager.publisher(for: \.location)
                .compactMap { $0 }
                .first()
                .sink(
                    receiveCompletion: { completion in
                        if case .failure(let error) = completion {
                            continuation.resume(throwing: WeatherServiceError.parsingError(underlying: error))
                        }
                        cancellable?.cancel()
                    },
                    receiveValue: { location in
                        continuation.resume(returning: location)
                        cancellable?.cancel()
                    }
                )
        }
    }
    
    // MARK: - Private Methods
    
    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyKilometer
        locationManager.distanceFilter = 1000 // 1km
    }
    
    private func setupCache() {
        cache.countLimit = 100
        cache.totalCostLimit = 10 * 1024 * 1024 // 10MB
    }
    
    private func buildURL(endpoint: String, location: CLLocation, additionalParams: [String: String] = [:]) -> URL {
        var components = URLComponents(string: "\(baseURL)/\(endpoint)")!
        
        var queryItems = [
            URLQueryItem(name: "lat", value: String(location.coordinate.latitude)),
            URLQueryItem(name: "lon", value: String(location.coordinate.longitude)),
            URLQueryItem(name: "appid", value: apiKey),
            URLQueryItem(name: "units", value: "metric")
        ]
        
        for (key, value) in additionalParams {
            queryItems.append(URLQueryItem(name: key, value: value))
        }
        
        components.queryItems = queryItems
        return components.url!
    }
    
    private func buildGeocodingURL(query: String) -> URL {
        var components = URLComponents(string: "\(baseURL)/geo/1.0/direct")!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "limit", value: "5"),
            URLQueryItem(name: "appid", value: apiKey)
        ]
        return components.url!
    }
    
    private func fetchData(from url: URL) async throws -> Data {
        await MainActor.run {
            self.isLoading = true
            self.error = nil
        }
        
        defer {
            Task { @MainActor in
                self.isLoading = false
            }
        }
        
        do {
            let (data, response) = try await session.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw WeatherServiceError.invalidResponse(details: "Not an HTTP response")
            }

            switch httpResponse.statusCode {
            case 200...299:
                return data
            case 401:
                throw WeatherServiceError.invalidAPIKey
            case 429:
                throw WeatherServiceError.rateLimitExceeded(retryAfter: nil)
            case 500...599:
                throw WeatherServiceError.serviceUnavailable
            default:
                throw WeatherServiceError.invalidResponse(details: "HTTP status \(httpResponse.statusCode)")
            }
        } catch {
            if error is WeatherServiceError {
                throw error
            } else {
                throw WeatherServiceError.parsingError(underlying: error)
            }
        }
    }
    
    private func parseCurrentWeatherResponse(_ data: Data, location: CLLocation) throws -> WeatherData {
        // For demo purposes, return mock data
        // In a real implementation, this would parse the actual API response
        return generateMockWeatherData(for: location, date: Date())
    }
    
    private func getDetailedForecast(for location: CLLocation, days: Int) async throws -> ([WeatherData], [WeatherData]) {
        // For demo purposes, generate mock forecast data
        // In a real implementation, this would call the forecast API
        
        var hourlyForecast: [WeatherData] = []
        var dailyForecast: [WeatherData] = []
        
        let calendar = Calendar.current
        let now = Date()
        
        // Generate hourly forecast for next 24 hours
        for hour in 1...24 {
            guard let date = calendar.date(byAdding: .hour, value: hour, to: now) else { continue }
            hourlyForecast.append(generateMockWeatherData(for: location, date: date))
        }
        
        // Generate daily forecast
        for day in 1...days {
            guard let date = calendar.date(byAdding: .day, value: day, to: now) else { continue }
            dailyForecast.append(generateMockWeatherData(for: location, date: date))
        }
        
        return (hourlyForecast, dailyForecast)
    }
    
    private func parseLocationSearchResponse(_ data: Data) throws -> [WeatherLocation] {
        // For demo purposes, return mock locations
        // In a real implementation, this would parse the geocoding API response
        return [
            WeatherLocation(latitude: 40.7128, longitude: -74.0060, city: "New York", country: "US"),
            WeatherLocation(latitude: 51.5074, longitude: -0.1278, city: "London", country: "GB"),
            WeatherLocation(latitude: 48.8566, longitude: 2.3522, city: "Paris", country: "FR")
        ]
    }
    
    private func generateMockWeatherData(for location: CLLocation, date: Date) -> WeatherData {
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: date)
        let dayOfYear = calendar.dayOfYear(for: date) ?? 1
        
        // Generate realistic seasonal temperature variation
        let baseTemp = 15.0 + 10.0 * sin(Double(dayOfYear) * 2.0 * .pi / 365.0)
        let dailyVariation = 5.0 * sin(Double(hour) * .pi / 12.0)
        let randomVariation = Double.random(in: -3.0...3.0)
        let temperature = baseTemp + dailyVariation + randomVariation
        
        // Generate correlated weather parameters
        let humidity = max(30, min(90, 60 + Double.random(in: -20...20)))
        let pressure = 1013.25 + Double.random(in: -20...20)
        let windSpeed = max(0, Double.random(in: 0...25))
        let precipitation = Double.random(in: 0...10)
        let uvIndex = max(0, min(11, Double(hour - 6) / 2.0 + Double.random(in: -2...2)))
        
        let conditions: [WeatherCondition] = [.clear, .partlyCloudy, .cloudy, .rain]
        let condition = conditions.randomElement() ?? .clear
        
        return WeatherData(
            date: date,
            temperature: temperature,
            humidity: humidity,
            barometricPressure: pressure,
            windSpeed: windSpeed,
            precipitation: precipitation,
            uvIndex: uvIndex,
            condition: condition,
            location: WeatherLocation(
                latitude: location.coordinate.latitude,
                longitude: location.coordinate.longitude,
                city: "Current Location",
                country: "Unknown"
            )
        )
    }
    
    private func generateMockWeatherAlerts(for location: CLLocation) -> [WeatherAlert] {
        // Generate mock weather alerts for demonstration
        let alerts = [
            WeatherAlert(
                title: "Barometric Pressure Drop",
                description: "Significant pressure drop expected. May affect joint pain.",
                severity: .moderate,
                startDate: Date(),
                endDate: Calendar.current.date(byAdding: .hour, value: 6, to: Date()) ?? Date(),
                affectedAreas: ["Current Location"]
            )
        ]
        
        return Bool.random() ? alerts : []
    }
    
    private func getCachedData(for key: String) -> WeatherData? {
        guard let data = cache.object(forKey: NSString(string: key)) as Data?,
              let cachedItem = try? JSONDecoder().decode(CachedWeatherItem.self, from: data),
              Date().timeIntervalSince(cachedItem.timestamp) < cacheTimeout else {
            return nil
        }
        
        return cachedItem.weatherData
    }
    
    private func getCachedForecast(for key: String) -> WeatherForecast? {
        guard let data = cache.object(forKey: NSString(string: key)) as Data?,
              let cachedItem = try? JSONDecoder().decode(CachedForecastItem.self, from: data),
              Date().timeIntervalSince(cachedItem.timestamp) < cacheTimeout else {
            return nil
        }
        
        return cachedItem.forecast
    }
    
    private func cacheData(_ weatherData: WeatherData, for key: String) {
        let cachedItem = CachedWeatherItem(weatherData: weatherData, timestamp: Date())
        if let data = try? JSONEncoder().encode(cachedItem) {
            cache.setObject(data as NSData, forKey: NSString(string: key))
        }
    }
    
    private func cacheForecast(_ forecast: WeatherForecast, for key: String) {
        let cachedItem = CachedForecastItem(forecast: forecast, timestamp: Date())
        if let data = try? JSONEncoder().encode(cachedItem) {
            cache.setObject(data as NSData, forKey: NSString(string: key))
        }
    }
}

// MARK: - Cache Models

private struct CachedWeatherItem: Codable {
    let weatherData: WeatherData
    let timestamp: Date
}

private struct CachedForecastItem: Codable {
    let forecast: WeatherForecast
    let timestamp: Date
}

// MARK: - Extensions

extension Calendar {
    func dayOfYear(for date: Date) -> Int? {
        return ordinality(of: .day, in: .year, for: date)
    }
}

extension CLLocationManager {
    func publisher(for keyPath: KeyPath<CLLocationManager, CLLocation?>) -> AnyPublisher<CLLocation?, Never> {
        return publisher(for: keyPath, options: [.initial, .new])
            .eraseToAnyPublisher()
    }
}

// MARK: - CLLocationManagerDelegate

extension WeatherService: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        // Handle location updates if needed
        guard let location = locations.last else { return }
        print("Location updated: \(location.coordinate.latitude), \(location.coordinate.longitude)")
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location manager failed with error: \(error.localizedDescription)")
        Task { @MainActor in
            self.error = .locationPermissionDenied
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        print("Location authorization changed: \(status.rawValue)")

        switch status {
        case .authorizedWhenInUse, .authorizedAlways:
            print("Location authorized")
        case .denied, .restricted:
            Task { @MainActor in
                self.error = .locationPermissionDenied
            }
        case .notDetermined:
            print("Location authorization not determined")
        @unknown default:
            break
        }
    }
}