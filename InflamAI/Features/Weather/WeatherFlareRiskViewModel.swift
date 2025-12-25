//
//  WeatherFlareRiskViewModel.swift
//  InflamAI
//
//  Core weather-flare prediction logic using Open-Meteo API
//  Analyzes pressure trends and calculates personalized risk
//  Now uses FREE Open-Meteo instead of WeatherKit (no API key required)
//

import Foundation
import CoreLocation
import CoreData
import Combine
import SwiftUI

@MainActor
class WeatherFlareRiskViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var currentWeather: CurrentWeatherData?
    @Published var hourlyForecast: [HourlyWeatherData] = []
    @Published var dailyForecast: [DailyWeatherData] = []
    @Published var pressureForecast: [PressureDataPoint] = []
    @Published var flareRisk: FlareRiskAssessment?
    @Published var weatherAlerts: [WeatherFlareAlert] = []
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    // MARK: - Dependencies

    private let weatherService: OpenMeteoService
    private let persistenceController: InflamAIPersistenceController
    private var cancellables = Set<AnyCancellable>()

    // MARK: - User's Pressure Sensitivity

    private var userPressureSensitivity: Double = 10.0 // hPa threshold

    // MARK: - Initialization

    init(
        weatherService: OpenMeteoService? = nil,
        persistenceController: InflamAIPersistenceController? = nil
    ) {
        let resolvedWeatherService = weatherService ?? OpenMeteoService.shared
        self.weatherService = resolvedWeatherService
        self.persistenceController = persistenceController ?? InflamAIPersistenceController.shared

        // Subscribe to weather service updates
        resolvedWeatherService.$currentWeather
            .sink { [weak self] weather in
                self?.currentWeather = weather
            }
            .store(in: &cancellables)

        resolvedWeatherService.$hourlyForecast
            .sink { [weak self] forecast in
                self?.hourlyForecast = forecast
                self?.updatePressureForecast()
            }
            .store(in: &cancellables)

        resolvedWeatherService.$dailyForecast
            .sink { [weak self] forecast in
                self?.dailyForecast = forecast
            }
            .store(in: &cancellables)

        resolvedWeatherService.$isLoading
            .sink { [weak self] loading in
                self?.isLoading = loading
            }
            .store(in: &cancellables)

        resolvedWeatherService.$errorMessage
            .sink { [weak self] error in
                self?.errorMessage = error
            }
            .store(in: &cancellables)
    }

    // MARK: - Public Methods

    func loadWeatherData() async {
        isLoading = true
        errorMessage = nil

        do {
            // Fetch all weather data
            try await weatherService.fetchAllWeatherData()

            // Calculate user's pressure sensitivity from historical data
            await calculateUserPressureSensitivity()

            // Update pressure forecast for chart
            updatePressureForecast()

            // Calculate flare risk
            flareRisk = weatherService.assessFlareRisk()

            // Generate weather alerts
            generateWeatherAlerts()

            isLoading = false

        } catch {
            isLoading = false

            if let weatherError = error as? OpenMeteoError {
                errorMessage = weatherError.errorDescription
            } else {
                errorMessage = "Weather service temporarily unavailable. Please try again."
            }

            print("❌ Weather error: \(error.localizedDescription)")
        }
    }

    func refresh() async {
        await loadWeatherData()
    }

    // MARK: - Private Methods

    private func updatePressureForecast() {
        pressureForecast = weatherService.getPressureForecast()
    }

    private func generateWeatherAlerts() {
        var alerts: [WeatherFlareAlert] = []

        // Look for significant pressure drops in the next 24 hours
        for i in 0..<min(24, hourlyForecast.count - 1) {
            let current = hourlyForecast[i]

            // Look 12 hours ahead if possible
            let lookAheadIndex = min(i + 12, hourlyForecast.count - 1)
            let future = hourlyForecast[lookAheadIndex]

            let drop = future.pressure - current.pressure

            if drop < -userPressureSensitivity {
                let severity: WeatherFlareAlert.AlertSeverity
                if drop < -(userPressureSensitivity * 1.5) {
                    severity = .high
                } else {
                    severity = .moderate
                }

                let hoursFromNow = i
                let timeframe: String
                if hoursFromNow == 0 {
                    timeframe = "Starting now"
                } else if hoursFromNow < 6 {
                    timeframe = "In \(hoursFromNow) hours"
                } else {
                    timeframe = "In ~\(hoursFromNow / 6 * 6) hours"
                }

                // Avoid duplicate alerts for overlapping windows
                let isDuplicate = alerts.contains { existingAlert in
                    existingAlert.timeframe == timeframe
                }

                if !isDuplicate {
                    alerts.append(WeatherFlareAlert(
                        message: String(format: "Pressure drop of %.1f mmHg expected", abs(drop)),
                        timeframe: timeframe,
                        severity: severity
                    ))
                }
            }
        }

        weatherAlerts = alerts
    }

    private func calculateUserPressureSensitivity() async {
        let context = persistenceController.container.viewContext

        await context.perform {
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
            request.fetchLimit = 90
            request.predicate = NSPredicate(format: "contextSnapshot != nil")

            guard let logs = try? context.fetch(request), logs.count >= 7 else {
                return
            }

            let pressureChanges = logs.compactMap { $0.contextSnapshot?.pressureChange12h }
            let pain = logs.map { $0.basdaiScore }

            guard pressureChanges.count == pain.count, pressureChanges.count >= 7 else {
                return
            }

            // Find high pain days and their associated pressure drops
            let highPainLogs = logs.filter { $0.basdaiScore > 6.0 }
            let pressureDropsOnHighPain = highPainLogs.compactMap {
                $0.contextSnapshot?.pressureChange12h
            }.filter { $0 < 0 }

            if !pressureDropsOnHighPain.isEmpty {
                let avgDrop = abs(pressureDropsOnHighPain.reduce(0, +) / Double(pressureDropsOnHighPain.count))

                Task { @MainActor in
                    self.userPressureSensitivity = max(5.0, min(20.0, avgDrop))
                }
            }
        }
    }

    // MARK: - Computed Properties

    var riskLevel: FlareRiskLevel {
        flareRisk?.level ?? .unknown
    }

    var riskRecommendation: String {
        flareRisk?.recommendation ?? "Loading weather data..."
    }

    var riskFactors: [String] {
        flareRisk?.factors ?? []
    }

    var currentTemperature: String {
        guard let temp = currentWeather?.temperature else { return "--" }
        return String(format: "%.0f°C", temp)
    }

    var currentHumidity: String {
        guard let humidity = currentWeather?.humidity else { return "--" }
        return "\(humidity)%"
    }

    var currentPressure: String {
        guard let pressure = currentWeather?.pressure else { return "--" }
        return String(format: "%.0f mmHg", pressure)
    }

    var currentCondition: WeatherConditionType {
        currentWeather?.condition ?? .unknown
    }

    var pressureChange12h: Double {
        currentWeather?.pressureChange12h ?? 0.0
    }
}

// MARK: - Supporting Models

struct WeatherFlareAlert: Identifiable {
    let id = UUID()
    let message: String
    let timeframe: String
    let severity: AlertSeverity

    enum AlertSeverity {
        case moderate, high

        var color: Color {
            switch self {
            case .moderate: return .orange
            case .high: return .red
            }
        }
    }
}

// MARK: - Legacy Compatibility

// These types are kept for compatibility with existing UI components
enum WeatherRiskLevel {
    case unknown, low, moderate, high, critical

    static func from(percentage: Double) -> WeatherRiskLevel {
        switch percentage {
        case 0..<20: return .low
        case 20..<40: return .moderate
        case 40..<70: return .high
        case 70...100: return .critical
        default: return .unknown
        }
    }

    var color: Color {
        switch self {
        case .unknown: return .gray
        case .low: return .green
        case .moderate: return .yellow
        case .high: return .orange
        case .critical: return .red
        }
    }

    var displayName: String {
        switch self {
        case .unknown: return "Unknown"
        case .low: return "Low Risk"
        case .moderate: return "Moderate"
        case .high: return "High Risk"
        case .critical: return "Critical"
        }
    }
}

enum ServiceStatus {
    case operational
    case degraded
    case offline
}
