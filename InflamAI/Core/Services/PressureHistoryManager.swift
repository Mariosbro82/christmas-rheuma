//
//  PressureHistoryManager.swift
//  InflamAI
//
//  Manages barometric pressure history for accurate delta calculations
//  Critical for AS flare prediction - pressure drops >5 mmHg in 12h correlate with inflammation
//

import Foundation

/// Actor for thread-safe pressure history management
/// Persists pressure readings to enable accurate 3h, 6h, 12h, and 24h delta calculations
actor PressureHistoryManager {

    // MARK: - Types

    struct PressureReading: Codable, Sendable {
        let timestamp: Date
        let pressureMmHg: Double
        let source: PressureSource

        enum PressureSource: String, Codable, Sendable {
            case openMeteo = "open_meteo"
            case manual = "manual"
            case watch = "watch"
        }
    }

    struct PressureTrend: Sendable {
        let direction: TrendDirection
        let rate3h: Double           // mmHg per hour (last 3h)
        let rate6h: Double           // mmHg per hour (last 6h)
        let rate12h: Double          // mmHg per hour (last 12h)
        let isAccelerating: Bool     // Is drop rate increasing?
        let flareRiskContribution: Double  // 0.0 - 0.5

        enum TrendDirection: String, Sendable {
            case rising = "Rising"
            case stable = "Stable"
            case falling = "Falling"
            case rapidDrop = "Rapid Drop"    // >5 mmHg in 12h
            case extremeDrop = "Extreme Drop" // >10 mmHg in 12h
        }
    }

    // MARK: - Properties

    private var readings: [PressureReading] = []
    private let maxHistoryDays: Int = 7
    private let fileURL: URL
    private var pendingSaveCount = 0
    private let batchSaveThreshold = 5

    // MARK: - Initialization

    init() {
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("❌ CRITICAL: Cannot access documents directory")
            // Fallback to temporary directory
            self.fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("pressure_history.json")
            return
        }
        self.fileURL = documentsDirectory.appendingPathComponent("pressure_history.json")

        // Load existing history
        Task {
            await loadHistory()
        }
    }

    // MARK: - Public Methods

    /// Record a new pressure reading
    func record(_ pressure: Double, source: PressureReading.PressureSource = .openMeteo) {
        let reading = PressureReading(
            timestamp: Date(),
            pressureMmHg: pressure,
            source: source
        )

        readings.append(reading)
        pendingSaveCount += 1

        // Clean old readings
        cleanOldReadings()

        // Batch save to reduce I/O
        if pendingSaveCount >= batchSaveThreshold {
            saveHistory()
            pendingSaveCount = 0
        }

        print("📊 Recorded pressure: \(String(format: "%.1f", pressure)) mmHg (total: \(readings.count) readings)")
    }

    /// Calculate pressure change over specified hours
    func change(over hours: Int) -> Double? {
        guard !readings.isEmpty else { return nil }

        let now = Date()
        let targetTime = now.addingTimeInterval(-Double(hours) * 3600)

        // Find reading closest to target time
        let historicalReading = readings
            .filter { $0.timestamp <= targetTime }
            .max(by: { $0.timestamp < $1.timestamp })

        guard let historical = historicalReading,
              let current = readings.last else {
            return nil
        }

        return current.pressureMmHg - historical.pressureMmHg
    }

    /// 3-hour pressure change
    func change3h() -> Double? {
        return change(over: 3)
    }

    /// 6-hour pressure change
    func change6h() -> Double? {
        return change(over: 6)
    }

    /// 12-hour pressure change (most important for AS)
    func change12h() -> Double? {
        return change(over: 12)
    }

    /// 24-hour pressure change
    func change24h() -> Double? {
        return change(over: 24)
    }

    /// Get current pressure trend analysis
    func trend() -> PressureTrend {
        let change3h = self.change3h() ?? 0
        let change6h = self.change6h() ?? 0
        let change12h = self.change12h() ?? 0

        // Calculate rates (mmHg per hour)
        let rate3h = change3h / 3.0
        let rate6h = change6h / 6.0
        let rate12h = change12h / 12.0

        // Detect acceleration (rate increasing over time)
        let isAccelerating = abs(rate3h) > abs(rate6h) && rate3h < 0

        // Determine direction
        let direction: PressureTrend.TrendDirection
        if change12h < -10 {
            direction = .extremeDrop
        } else if change12h < -5 {
            direction = .rapidDrop
        } else if change12h < -1 {
            direction = .falling
        } else if change12h > 1 {
            direction = .rising
        } else {
            direction = .stable
        }

        // Calculate flare risk contribution (0-0.5)
        var riskContribution = 0.0
        if change12h < -5 {
            riskContribution = 0.4
            if isAccelerating {
                riskContribution = 0.5
            }
        } else if change12h < -3 {
            riskContribution = 0.2
        } else if change12h < -1 {
            riskContribution = 0.1
        }

        return PressureTrend(
            direction: direction,
            rate3h: rate3h,
            rate6h: rate6h,
            rate12h: rate12h,
            isAccelerating: isAccelerating,
            flareRiskContribution: riskContribution
        )
    }

    /// Check if pressure is dropping at an accelerating rate (critical flare indicator)
    func isAcceleratingDrop() -> Bool {
        let trend = self.trend()
        return trend.isAccelerating && trend.direction == .rapidDrop
    }

    /// Get readings for specified time period
    func readings(last hours: Int) -> [PressureReading] {
        let cutoff = Date().addingTimeInterval(-Double(hours) * 3600)
        return readings.filter { $0.timestamp > cutoff }
    }

    /// Get readings between dates
    func readings(from startDate: Date, to endDate: Date) -> [PressureReading] {
        return readings.filter { $0.timestamp >= startDate && $0.timestamp <= endDate }
    }

    /// Get most recent reading
    func latestReading() -> PressureReading? {
        return readings.last
    }

    /// Get reading count
    func readingCount() -> Int {
        return readings.count
    }

    /// Check if we have sufficient data for accurate calculations
    func hasSufficientData(forHours hours: Int) -> Bool {
        let cutoff = Date().addingTimeInterval(-Double(hours) * 3600)
        return readings.contains { $0.timestamp <= cutoff }
    }

    // MARK: - Private Methods

    private func cleanOldReadings() {
        let cutoff = Date().addingTimeInterval(-Double(maxHistoryDays) * 24 * 3600)
        readings = readings.filter { $0.timestamp > cutoff }
    }

    private func loadHistory() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            print("📊 No pressure history file found, starting fresh")
            return
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            readings = try decoder.decode([PressureReading].self, from: data)
            cleanOldReadings()
            print("📊 Loaded \(readings.count) pressure readings from history")
        } catch {
            print("⚠️ Failed to load pressure history: \(error)")
            readings = []
        }
    }

    private func saveHistory() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(readings)
            try data.write(to: fileURL, options: .atomic)
            print("📊 Saved \(readings.count) pressure readings to history")
        } catch {
            print("⚠️ Failed to save pressure history: \(error)")
        }
    }

    /// Force save (call when app goes to background)
    func forceSave() {
        saveHistory()
        pendingSaveCount = 0
    }
}

// MARK: - Pressure Analysis Utilities

extension PressureHistoryManager {

    /// Get pressure statistics for a time period
    func statistics(forLastHours hours: Int) -> PressureStatistics? {
        let relevantReadings = readings(last: hours)
        guard !relevantReadings.isEmpty else { return nil }

        let pressures = relevantReadings.map { $0.pressureMmHg }
        let sum = pressures.reduce(0, +)
        let mean = sum / Double(pressures.count)

        let sortedPressures = pressures.sorted()
        guard let min = sortedPressures.first,
              let max = sortedPressures.last else {
            return nil
        }

        // Calculate standard deviation
        let squaredDifferences = pressures.map { pow($0 - mean, 2) }
        let variance = squaredDifferences.reduce(0, +) / Double(pressures.count)
        let stdDev = sqrt(variance)

        return PressureStatistics(
            mean: mean,
            min: min,
            max: max,
            range: max - min,
            standardDeviation: stdDev,
            readingCount: pressures.count
        )
    }

    struct PressureStatistics: Sendable {
        let mean: Double
        let min: Double
        let max: Double
        let range: Double
        let standardDeviation: Double
        let readingCount: Int
    }

    /// Detect significant pressure events (for notifications)
    func detectSignificantEvents() -> [PressureEvent] {
        var events: [PressureEvent] = []

        // Check for rapid drop
        if let change12h = change12h(), change12h < -5 {
            events.append(PressureEvent(
                type: .rapidDrop,
                magnitude: abs(change12h),
                timestamp: Date(),
                description: "Barometric pressure dropped \(String(format: "%.1f", abs(change12h))) mmHg in 12 hours"
            ))
        }

        // Check for accelerating drop
        if isAcceleratingDrop() {
            events.append(PressureEvent(
                type: .acceleratingDrop,
                magnitude: abs(change3h() ?? 0),
                timestamp: Date(),
                description: "Pressure drop is accelerating - flare risk elevated"
            ))
        }

        // Check for storm system (very low pressure)
        if let latest = latestReading(), latest.pressureMmHg < 745 {
            events.append(PressureEvent(
                type: .stormSystem,
                magnitude: 760 - latest.pressureMmHg,
                timestamp: Date(),
                description: "Low pressure system detected (\(String(format: "%.0f", latest.pressureMmHg)) mmHg)"
            ))
        }

        return events
    }

    struct PressureEvent: Sendable {
        let type: EventType
        let magnitude: Double
        let timestamp: Date
        let description: String

        enum EventType: String, Sendable {
            case rapidDrop = "rapid_drop"
            case acceleratingDrop = "accelerating_drop"
            case stormSystem = "storm_system"
            case recovery = "recovery"
        }
    }
}
