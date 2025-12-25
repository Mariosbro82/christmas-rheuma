//
//  PerformanceMonitor.swift
//  InflamAI
//
//  Real-time performance monitoring for ML inference
//  Tracks latency, battery, memory, and compute unit usage
//

import Foundation
import CoreML
import UIKit

@MainActor
class PerformanceMonitor: ObservableObject {

    // MARK: - Published Properties

    @Published var currentMetrics: PerformanceSnapshot?
    @Published var historicalMetrics: [PerformanceSnapshot] = []
    @Published var averageLatency: TimeInterval = 0.0
    @Published var peakMemoryMB: Double = 0.0
    @Published var batteryImpact: BatteryImpact = .negligible

    enum BatteryImpact: String {
        case negligible = "Negligible"    // < 1% per day
        case low = "Low"                  // 1-3% per day
        case moderate = "Moderate"        // 3-5% per day
        case high = "High"                // > 5% per day

        var color: String {
            switch self {
            case .negligible: return "green"
            case .low: return "blue"
            case .moderate: return "yellow"
            case .high: return "red"
            }
        }

        var icon: String {
            switch self {
            case .negligible: return "battery.100"
            case .low: return "battery.75"
            case .moderate: return "battery.50"
            case .high: return "battery.25"
            }
        }
    }

    // MARK: - Configuration

    private let maxHistorySize: Int = 100
    private let targetLatency: TimeInterval = 0.050  // 50ms target
    private let targetBatteryPercent: Double = 5.0   // 5% per day max

    // MARK: - Battery Tracking

    private var batteryStartLevel: Float?
    private var batteryTrackingStartTime: Date?
    private var predictionCountSinceReset: Int = 0

    // MARK: - Initialization

    init() {
        startBatteryTracking()
        loadHistoricalMetrics()
        computeAverageMetrics()
    }

    // MARK: - Public API

    /// Measure performance of a prediction
    func measurePrediction<T>(_ block: () async throws -> T) async rethrows -> (result: T, metrics: PerformanceSnapshot) {
        let startTime = CFAbsoluteTimeGetCurrent()
        let startMemory = getCurrentMemoryUsageMB()
        let startBattery = UIDevice.current.batteryLevel

        // Execute prediction
        let result = try await block()

        let endTime = CFAbsoluteTimeGetCurrent()
        let endMemory = getCurrentMemoryUsageMB()
        let endBattery = UIDevice.current.batteryLevel

        // Compute metrics
        let latency = endTime - startTime
        let memoryDelta = endMemory - startMemory
        let batteryDelta = startBattery > 0 ? startBattery - endBattery : 0

        let snapshot = PerformanceSnapshot(
            timestamp: Date(),
            latencySeconds: latency,
            memoryUsageMB: endMemory,
            memoryDeltaMB: memoryDelta,
            batteryDelta: batteryDelta,
            computeUnit: .all  // Would detect actual unit if available
        )

        // Update state
        currentMetrics = snapshot
        historicalMetrics.append(snapshot)

        // Trim history
        if historicalMetrics.count > maxHistorySize {
            historicalMetrics.removeFirst(historicalMetrics.count - maxHistorySize)
        }

        // Update averages
        computeAverageMetrics()

        // Track for battery impact
        predictionCountSinceReset += 1

        // Persist
        saveHistoricalMetrics()

        return (result, snapshot)
    }

    /// Get performance report
    func getPerformanceReport() -> PerformanceReport {
        let meetsLatencyTarget = averageLatency <= targetLatency
        let meetsBatteryTarget = estimatedDailyBatteryUsage() <= targetBatteryPercent
        let meetsMemoryTarget = peakMemoryMB <= 50.0  // 50MB target

        return PerformanceReport(
            averageLatency: averageLatency,
            targetLatency: targetLatency,
            meetsLatencyTarget: meetsLatencyTarget,
            peakMemoryMB: peakMemoryMB,
            meetsMemoryTarget: meetsMemoryTarget,
            batteryImpact: batteryImpact,
            estimatedDailyBatteryUsage: estimatedDailyBatteryUsage(),
            meetsBatteryTarget: meetsBatteryTarget,
            totalPredictions: historicalMetrics.count,
            oldestMetric: historicalMetrics.first?.timestamp,
            newestMetric: historicalMetrics.last?.timestamp
        )
    }

    /// Reset battery tracking (call after significant battery change)
    func resetBatteryTracking() {
        batteryStartLevel = UIDevice.current.batteryLevel
        batteryTrackingStartTime = Date()
        predictionCountSinceReset = 0
    }

    // MARK: - Metrics Computation

    private func computeAverageMetrics() {
        guard !historicalMetrics.isEmpty else {
            averageLatency = 0.0
            peakMemoryMB = 0.0
            return
        }

        // Average latency
        let totalLatency = historicalMetrics.reduce(0.0) { $0 + $1.latencySeconds }
        averageLatency = totalLatency / Double(historicalMetrics.count)

        // Peak memory
        peakMemoryMB = historicalMetrics.map { $0.memoryUsageMB }.max() ?? 0.0

        // Battery impact
        batteryImpact = computeBatteryImpact()
    }

    private func computeBatteryImpact() -> BatteryImpact {
        let dailyUsage = estimatedDailyBatteryUsage()

        if dailyUsage < 1.0 {
            return .negligible
        } else if dailyUsage < 3.0 {
            return .low
        } else if dailyUsage <= 5.0 {
            return .moderate
        } else {
            return .high
        }
    }

    private func estimatedDailyBatteryUsage() -> Double {
        guard let startLevel = batteryStartLevel,
              let startTime = batteryTrackingStartTime,
              predictionCountSinceReset > 0 else {
            return 0.0
        }

        let currentLevel = UIDevice.current.batteryLevel
        let elapsedTime = Date().timeIntervalSince(startTime)
        let elapsedDays = elapsedTime / 86400.0

        guard elapsedDays > 0, currentLevel > 0, startLevel > currentLevel else {
            return 0.0
        }

        // Battery drain in this period
        let batteryDrain = Double(startLevel - currentLevel) * 100.0  // Convert to percentage

        // Extrapolate to full day
        let dailyDrain = batteryDrain / elapsedDays

        return dailyDrain
    }

    // MARK: - Memory Monitoring

    private func getCurrentMemoryUsageMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }

        if kerr == KERN_SUCCESS {
            let usedMemoryBytes = Double(info.resident_size)
            return usedMemoryBytes / (1024 * 1024)  // Convert to MB
        } else {
            return 0.0
        }
    }

    // MARK: - Battery Monitoring

    private func startBatteryTracking() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        batteryStartLevel = UIDevice.current.batteryLevel
        batteryTrackingStartTime = Date()

        // Register for battery notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(batteryLevelDidChange),
            name: UIDevice.batteryLevelDidChangeNotification,
            object: nil
        )
    }

    @objc private func batteryLevelDidChange() {
        // Recompute battery impact
        batteryImpact = computeBatteryImpact()
    }

    // MARK: - Compute Unit Detection

    /// Detect which compute unit was used (approximation)
    private func detectComputeUnit(latency: TimeInterval) -> MLComputeUnits {
        // This is an approximation based on latency
        // Neural Engine: < 20ms
        // GPU: 20-50ms
        // CPU: > 50ms
        if latency < 0.020 {
            return .cpuAndNeuralEngine
        } else if latency < 0.050 {
            return .cpuAndGPU
        } else {
            return .cpuOnly
        }
    }

    // MARK: - Persistence

    private func saveHistoricalMetrics() {
        let cacheURL = getMetricsCacheURL()

        do {
            let data = try JSONEncoder().encode(historicalMetrics)
            try data.write(to: cacheURL)
        } catch {
            #if DEBUG
            print("⚠️ Failed to save performance metrics: \(error)")
            #endif
        }
    }

    private func loadHistoricalMetrics() {
        let cacheURL = getMetricsCacheURL()

        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            return
        }

        do {
            let data = try Data(contentsOf: cacheURL)
            historicalMetrics = try JSONDecoder().decode([PerformanceSnapshot].self, from: data)
            #if DEBUG
            print("✅ Loaded \(historicalMetrics.count) performance metrics")
            #endif
        } catch {
            #if DEBUG
            print("⚠️ Failed to load performance metrics: \(error)")
            #endif
        }
    }

    private func getMetricsCacheURL() -> URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsURL.appendingPathComponent("performance_metrics.json")
    }

    // MARK: - Diagnostics

    /// Generate detailed performance diagnostics
    func generateDiagnostics() -> String {
        let report = getPerformanceReport()

        return """
        === PERFORMANCE DIAGNOSTICS ===

        LATENCY:
        Average: \(String(format: "%.1f", report.averageLatency * 1000))ms
        Target: \(String(format: "%.1f", report.targetLatency * 1000))ms
        Status: \(report.meetsLatencyTarget ? "✅ PASS" : "❌ FAIL")

        MEMORY:
        Peak Usage: \(String(format: "%.1f", report.peakMemoryMB))MB
        Status: \(report.meetsMemoryTarget ? "✅ PASS" : "⚠️ HIGH")

        BATTERY:
        Impact: \(report.batteryImpact.rawValue)
        Estimated Daily Usage: \(String(format: "%.1f%%", report.estimatedDailyBatteryUsage))
        Target: <\(String(format: "%.1f%%", targetBatteryPercent))
        Status: \(report.meetsBatteryTarget ? "✅ PASS" : "❌ FAIL")

        STATISTICS:
        Total Predictions: \(report.totalPredictions)
        Tracking Since: \(report.oldestMetric?.formatted() ?? "N/A")

        OVERALL: \(report.meetsAllTargets ? "✅ ALL TARGETS MET" : "⚠️ TARGETS NOT MET")
        """
    }

    /// Get latency percentiles
    func getLatencyPercentiles() -> LatencyPercentiles {
        let sorted = historicalMetrics.map { $0.latencySeconds }.sorted()

        guard !sorted.isEmpty else {
            return LatencyPercentiles(p50: 0, p90: 0, p95: 0, p99: 0)
        }

        func percentile(_ p: Double) -> TimeInterval {
            let index = Int(Double(sorted.count) * p)
            return sorted[min(index, sorted.count - 1)]
        }

        return LatencyPercentiles(
            p50: percentile(0.50),
            p90: percentile(0.90),
            p95: percentile(0.95),
            p99: percentile(0.99)
        )
    }
}

// MARK: - Data Types

struct PerformanceSnapshot: Codable, Identifiable {
    let id = UUID()
    let timestamp: Date
    let latencySeconds: TimeInterval
    let memoryUsageMB: Double
    let memoryDeltaMB: Double
    let batteryDelta: Float
    let computeUnit: MLComputeUnits

    var latencyMS: Double {
        return latencySeconds * 1000.0
    }

    var meetsLatencyTarget: Bool {
        return latencySeconds <= 0.050  // 50ms
    }

    enum CodingKeys: String, CodingKey {
        case timestamp, latencySeconds, memoryUsageMB, memoryDeltaMB, batteryDelta, computeUnit
    }

    init(timestamp: Date, latencySeconds: TimeInterval, memoryUsageMB: Double,
         memoryDeltaMB: Double, batteryDelta: Float, computeUnit: MLComputeUnits) {
        self.timestamp = timestamp
        self.latencySeconds = latencySeconds
        self.memoryUsageMB = memoryUsageMB
        self.memoryDeltaMB = memoryDeltaMB
        self.batteryDelta = batteryDelta
        self.computeUnit = computeUnit
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        latencySeconds = try container.decode(TimeInterval.self, forKey: .latencySeconds)
        memoryUsageMB = try container.decode(Double.self, forKey: .memoryUsageMB)
        memoryDeltaMB = try container.decode(Double.self, forKey: .memoryDeltaMB)
        batteryDelta = try container.decode(Float.self, forKey: .batteryDelta)
        computeUnit = (try? container.decode(MLComputeUnits.self, forKey: .computeUnit)) ?? .all
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(latencySeconds, forKey: .latencySeconds)
        try container.encode(memoryUsageMB, forKey: .memoryUsageMB)
        try container.encode(memoryDeltaMB, forKey: .memoryDeltaMB)
        try container.encode(batteryDelta, forKey: .batteryDelta)
        // Note: MLComputeUnits doesn't conform to Codable, so we skip it in encoding
    }
}

struct PerformanceReport {
    let averageLatency: TimeInterval
    let targetLatency: TimeInterval
    let meetsLatencyTarget: Bool

    let peakMemoryMB: Double
    let meetsMemoryTarget: Bool

    let batteryImpact: PerformanceMonitor.BatteryImpact
    let estimatedDailyBatteryUsage: Double
    let meetsBatteryTarget: Bool

    let totalPredictions: Int
    let oldestMetric: Date?
    let newestMetric: Date?

    var meetsAllTargets: Bool {
        return meetsLatencyTarget && meetsMemoryTarget && meetsBatteryTarget
    }

    var performanceScore: Float {
        // 10/10 rating requires all targets met
        let latencyScore: Float = meetsLatencyTarget ? 1.0 : 0.5
        let memoryScore: Float = meetsMemoryTarget ? 1.0 : 0.5
        let batteryScore: Float = meetsBatteryTarget ? 1.0 : 0.5

        return (latencyScore + memoryScore + batteryScore) / 3.0 * 10.0
    }

    var summary: String {
        """
        Performance Score: \(String(format: "%.1f", performanceScore))/10
        Latency: \(meetsLatencyTarget ? "✅" : "❌") \(String(format: "%.1fms", averageLatency * 1000))
        Memory: \(meetsMemoryTarget ? "✅" : "⚠️") \(String(format: "%.1fMB", peakMemoryMB))
        Battery: \(meetsBatteryTarget ? "✅" : "❌") \(String(format: "%.1f%%/day", estimatedDailyBatteryUsage))
        """
    }
}

struct LatencyPercentiles {
    let p50: TimeInterval  // Median
    let p90: TimeInterval
    let p95: TimeInterval
    let p99: TimeInterval

    var p50MS: Double { p50 * 1000 }
    var p90MS: Double { p90 * 1000 }
    var p95MS: Double { p95 * 1000 }
    var p99MS: Double { p99 * 1000 }

    var summary: String {
        """
        P50 (median): \(String(format: "%.1fms", p50MS))
        P90: \(String(format: "%.1fms", p90MS))
        P95: \(String(format: "%.1fms", p95MS))
        P99: \(String(format: "%.1fms", p99MS))
        """
    }
}

// MARK: - MLComputeUnits Codable Extension

extension MLComputeUnits: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(Int.self)
        self = MLComputeUnits(rawValue: rawValue) ?? .all
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(self.rawValue)
    }
}
