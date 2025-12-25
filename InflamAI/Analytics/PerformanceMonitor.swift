//
//  PerformanceMonitor.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import UIKit
import os.log
import QuartzCore
import Combine

// MARK: - Performance Monitor
class PerformanceMonitor: NSObject, ObservableObject {
    
    static let shared = PerformanceMonitor()
    
    private let logger = Logger(subsystem: "InflamAI", category: "Performance")
    private let dispatchQueue = DispatchQueue(label: "performance-monitor", qos: .utility)
    
    // Published properties
    @Published var currentMetrics: PerformanceSnapshot?
    @Published var memoryWarnings: [MemoryWarning] = []
    @Published var frameDrops: [FrameDrop] = []
    @Published var slowOperations: [SlowOperation] = []
    @Published var networkMetrics: NetworkMetrics?
    @Published var batteryMetrics: BatteryMetrics?
    @Published var thermalState: ProcessInfo.ThermalState = .nominal
    @Published var isMonitoring = false
    
    // Configuration
    private let monitoringInterval: TimeInterval = 1.0
    private let frameDropThreshold: TimeInterval = 0.016 // 60 FPS threshold
    private let slowOperationThreshold: TimeInterval = 0.5
    private let maxStoredMetrics = 1000
    
    // Internal state
    private var monitoringTimer: Timer?
    private var displayLink: CADisplayLink?
    private var lastFrameTimestamp: CFTimeInterval = 0
    private var frameCount = 0
    private var frameDropCount = 0
    private var operationStartTimes: [String: CFTimeInterval] = [:]
    private var networkObserver: NSObjectProtocol?
    private var thermalObserver: NSObjectProtocol?
    private var memoryObserver: NSObjectProtocol?
    private var cancellables = Set<AnyCancellable>()
    
    // Storage
    private var metricsHistory: [PerformanceSnapshot] = []
    private let fileManager = FileManager.default
    
    private lazy var metricsDirectory: URL = {
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let metricsDir = documentsPath.appendingPathComponent("PerformanceMetrics")
        try? fileManager.createDirectory(at: metricsDir, withIntermediateDirectories: true)
        return metricsDir
    }()
    
    override init() {
        super.init()
        setupObservers()
        loadStoredMetrics()
    }
    
    deinit {
        stopMonitoring()
        removeObservers()
    }
    
    // MARK: - Public Methods
    
    func startMonitoring() {
        guard !isMonitoring else { return }
        
        isMonitoring = true
        
        // Start periodic monitoring
        monitoringTimer = Timer.scheduledTimer(withTimeInterval: monitoringInterval, repeats: true) { [weak self] _ in
            self?.collectMetrics()
        }
        
        // Start frame rate monitoring
        setupDisplayLink()
        
        logger.info("Performance monitoring started")
    }
    
    func stopMonitoring() {
        guard isMonitoring else { return }
        
        isMonitoring = false
        
        monitoringTimer?.invalidate()
        monitoringTimer = nil
        
        displayLink?.invalidate()
        displayLink = nil
        
        saveMetrics()
        
        logger.info("Performance monitoring stopped")
    }
    
    func trackOperation(_ name: String, operation: () throws -> Void) rethrows {
        let startTime = CACurrentMediaTime()
        
        defer {
            let duration = CACurrentMediaTime() - startTime
            recordOperationDuration(name, duration: duration)
        }
        
        try operation()
    }
    
    func trackAsyncOperation<T>(_ name: String, operation: @escaping () async throws -> T) async rethrows -> T {
        let startTime = CACurrentMediaTime()
        
        defer {
            let duration = CACurrentMediaTime() - startTime
            recordOperationDuration(name, duration: duration)
        }
        
        return try await operation()
    }
    
    func startOperation(_ name: String) {
        operationStartTimes[name] = CACurrentMediaTime()
    }
    
    func endOperation(_ name: String) {
        guard let startTime = operationStartTimes.removeValue(forKey: name) else { return }
        
        let duration = CACurrentMediaTime() - startTime
        recordOperationDuration(name, duration: duration)
    }
    
    func recordCustomMetric(_ name: String, value: Double, unit: String = "") {
        let metric = CustomMetric(
            name: name,
            value: value,
            unit: unit,
            timestamp: Date()
        )
        
        dispatchQueue.async { [weak self] in
            self?.storeCustomMetric(metric)
        }
    }
    
    func generatePerformanceReport() -> PerformanceReport {
        let recentMetrics = Array(metricsHistory.suffix(100))
        
        return PerformanceReport(
            timestamp: Date(),
            averageMetrics: calculateAverageMetrics(from: recentMetrics),
            peakMetrics: calculatePeakMetrics(from: recentMetrics),
            memoryWarnings: Array(memoryWarnings.suffix(10)),
            frameDrops: Array(frameDrops.suffix(20)),
            slowOperations: Array(slowOperations.suffix(20)),
            recommendations: generateRecommendations(from: recentMetrics)
        )
    }
    
    func exportMetrics() -> Data? {
        do {
            let export = PerformanceExport(
                timestamp: Date(),
                metrics: metricsHistory,
                memoryWarnings: memoryWarnings,
                frameDrops: frameDrops,
                slowOperations: slowOperations
            )
            
            return try JSONEncoder().encode(export)
        } catch {
            logger.error("Failed to export metrics: \(error.localizedDescription)")
            return nil
        }
    }
    
    func clearMetrics() {
        dispatchQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.metricsHistory.removeAll()
            
            DispatchQueue.main.async {
                self.memoryWarnings.removeAll()
                self.frameDrops.removeAll()
                self.slowOperations.removeAll()
                self.currentMetrics = nil
            }
            
            // Clear stored files
            try? self.fileManager.removeItem(at: self.metricsDirectory)
            try? self.fileManager.createDirectory(at: self.metricsDirectory, withIntermediateDirectories: true)
            
            self.logger.info("Performance metrics cleared")
        }
    }
    
    // MARK: - Private Methods
    
    private func setupObservers() {
        // Memory warning observer
        memoryObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryWarning()
        }
        
        // Thermal state observer
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleThermalStateChange()
        }
        
        // Battery state monitoring
        UIDevice.current.isBatteryMonitoringEnabled = true
        
        NotificationCenter.default.publisher(for: UIDevice.batteryStateDidChangeNotification)
            .sink { [weak self] _ in
                self?.updateBatteryMetrics()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIDevice.batteryLevelDidChangeNotification)
            .sink { [weak self] _ in
                self?.updateBatteryMetrics()
            }
            .store(in: &cancellables)
    }
    
    private func removeObservers() {
        if let observer = memoryObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        if let observer = thermalObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        if let observer = networkObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        
        cancellables.removeAll()
    }
    
    private func setupDisplayLink() {
        displayLink = CADisplayLink(target: self, selector: #selector(displayLinkCallback))
        displayLink?.add(to: .main, forMode: .common)
    }
    
    @objc private func displayLinkCallback() {
        let currentTime = displayLink?.timestamp ?? 0
        
        if lastFrameTimestamp > 0 {
            let frameDuration = currentTime - lastFrameTimestamp
            
            if frameDuration > frameDropThreshold {
                recordFrameDrop(duration: frameDuration)
            }
            
            frameCount += 1
        }
        
        lastFrameTimestamp = currentTime
    }
    
    private func collectMetrics() {
        let snapshot = PerformanceSnapshot(
            timestamp: Date(),
            memoryMetrics: collectMemoryMetrics(),
            cpuMetrics: collectCPUMetrics(),
            diskMetrics: collectDiskMetrics(),
            networkMetrics: collectNetworkMetrics(),
            batteryMetrics: collectBatteryMetrics(),
            thermalState: ProcessInfo.processInfo.thermalState,
            frameRate: calculateCurrentFrameRate()
        )
        
        DispatchQueue.main.async {
            self.currentMetrics = snapshot
        }
        
        dispatchQueue.async { [weak self] in
            self?.storeMetrics(snapshot)
        }
    }
    
    private func collectMemoryMetrics() -> MemoryMetrics {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        let usedMemory = kerr == KERN_SUCCESS ? info.resident_size : 0
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        
        return MemoryMetrics(
            usedMemory: usedMemory,
            totalMemory: totalMemory,
            availableMemory: totalMemory - usedMemory,
            memoryPressure: getMemoryPressure(usedMemory: usedMemory, totalMemory: totalMemory)
        )
    }
    
    private func collectCPUMetrics() -> CPUMetrics {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        let userTime = kerr == KERN_SUCCESS ? Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1_000_000 : 0
        let systemTime = kerr == KERN_SUCCESS ? Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1_000_000 : 0
        
        return CPUMetrics(
            userTime: userTime,
            systemTime: systemTime,
            totalTime: userTime + systemTime,
            usage: calculateCPUUsage()
        )
    }
    
    private func collectDiskMetrics() -> DiskMetrics {
        do {
            let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
            let resourceValues = try documentsPath.resourceValues(forKeys: [
                .volumeAvailableCapacityKey,
                .volumeTotalCapacityKey
            ])
            
            let availableCapacity = resourceValues.volumeAvailableCapacity ?? 0
            let totalCapacity = resourceValues.volumeTotalCapacity ?? 0
            let usedCapacity = totalCapacity - availableCapacity
            
            return DiskMetrics(
                totalSpace: UInt64(totalCapacity),
                availableSpace: UInt64(availableCapacity),
                usedSpace: UInt64(usedCapacity),
                readOperations: 0, // TODO: Implement if needed
                writeOperations: 0 // TODO: Implement if needed
            )
        } catch {
            logger.error("Failed to collect disk metrics: \(error.localizedDescription)")
            return DiskMetrics(totalSpace: 0, availableSpace: 0, usedSpace: 0, readOperations: 0, writeOperations: 0)
        }
    }
    
    private func collectNetworkMetrics() -> NetworkMetrics {
        // This is a simplified implementation
        // In a real app, you might want to use Network.framework for more detailed metrics
        return NetworkMetrics(
            bytesReceived: 0,
            bytesSent: 0,
            packetsReceived: 0,
            packetsSent: 0,
            connectionCount: 0,
            latency: 0
        )
    }
    
    private func collectBatteryMetrics() -> BatteryMetrics {
        let device = UIDevice.current
        
        return BatteryMetrics(
            level: device.batteryLevel,
            state: BatteryState(from: device.batteryState),
            isLowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
    }
    
    private func calculateCurrentFrameRate() -> Double {
        // Calculate frame rate based on recent frame count
        let currentFrameCount = frameCount
        frameCount = 0
        
        return Double(currentFrameCount) / monitoringInterval
    }
    
    private func calculateCPUUsage() -> Double {
        // This is a simplified CPU usage calculation
        // For more accurate measurements, you might need to track deltas over time
        return 0.0 // Placeholder
    }
    
    private func getMemoryPressure(usedMemory: UInt64, totalMemory: UInt64) -> MemoryPressureLevel {
        let usageRatio = Double(usedMemory) / Double(totalMemory)
        
        if usageRatio > 0.9 {
            return .critical
        } else if usageRatio > 0.7 {
            return .high
        } else if usageRatio > 0.5 {
            return .medium
        } else {
            return .low
        }
    }
    
    private func recordOperationDuration(_ name: String, duration: TimeInterval) {
        if duration > slowOperationThreshold {
            let slowOp = SlowOperation(
                name: name,
                duration: duration,
                timestamp: Date(),
                stackTrace: Thread.callStackSymbols
            )
            
            DispatchQueue.main.async {
                self.slowOperations.append(slowOp)
                
                // Keep only recent slow operations
                if self.slowOperations.count > 100 {
                    self.slowOperations = Array(self.slowOperations.suffix(100))
                }
            }
            
            logger.warning("Slow operation detected: \(name) took \(duration)s")
        }
    }
    
    private func recordFrameDrop(duration: TimeInterval) {
        let frameDrop = FrameDrop(
            duration: duration,
            timestamp: Date(),
            expectedDuration: frameDropThreshold
        )
        
        DispatchQueue.main.async {
            self.frameDrops.append(frameDrop)
            
            // Keep only recent frame drops
            if self.frameDrops.count > 200 {
                self.frameDrops = Array(self.frameDrops.suffix(200))
            }
        }
        
        frameDropCount += 1
        
        if frameDropCount % 10 == 0 {
            logger.warning("Frame drops detected: \(frameDropCount) total")
        }
    }
    
    private func handleMemoryWarning() {
        let warning = MemoryWarning(
            timestamp: Date(),
            memoryMetrics: collectMemoryMetrics(),
            activeViewControllers: getActiveViewControllers()
        )
        
        memoryWarnings.append(warning)
        
        // Keep only recent warnings
        if memoryWarnings.count > 50 {
            memoryWarnings = Array(memoryWarnings.suffix(50))
        }
        
        logger.warning("Memory warning received")
    }
    
    private func handleThermalStateChange() {
        thermalState = ProcessInfo.processInfo.thermalState
        
        switch thermalState {
        case .serious, .critical:
            logger.warning("Thermal state changed to: \(thermalState)")
        default:
            break
        }
    }
    
    private func updateBatteryMetrics() {
        batteryMetrics = collectBatteryMetrics()
    }
    
    private func getActiveViewControllers() -> [String] {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let window = windowScene.windows.first else {
            return []
        }
        
        var controllers: [String] = []
        
        func traverse(_ viewController: UIViewController) {
            controllers.append(String(describing: type(of: viewController)))
            
            for child in viewController.children {
                traverse(child)
            }
        }
        
        if let rootViewController = window.rootViewController {
            traverse(rootViewController)
        }
        
        return controllers
    }
    
    private func storeMetrics(_ snapshot: PerformanceSnapshot) {
        metricsHistory.append(snapshot)
        
        // Keep only recent metrics in memory
        if metricsHistory.count > maxStoredMetrics {
            metricsHistory = Array(metricsHistory.suffix(maxStoredMetrics))
        }
        
        // Periodically save to disk
        if metricsHistory.count % 100 == 0 {
            saveMetrics()
        }
    }
    
    private func storeCustomMetric(_ metric: CustomMetric) {
        let filename = "custom_metrics.json"
        let fileURL = metricsDirectory.appendingPathComponent(filename)
        
        var metrics: [CustomMetric] = []
        
        if let data = try? Data(contentsOf: fileURL),
           let existingMetrics = try? JSONDecoder().decode([CustomMetric].self, from: data) {
            metrics = existingMetrics
        }
        
        metrics.append(metric)
        
        // Keep only recent metrics
        if metrics.count > 1000 {
            metrics = Array(metrics.suffix(1000))
        }
        
        do {
            let data = try JSONEncoder().encode(metrics)
            try data.write(to: fileURL)
        } catch {
            logger.error("Failed to store custom metric: \(error.localizedDescription)")
        }
    }
    
    private func saveMetrics() {
        do {
            let filename = "performance_metrics.json"
            let fileURL = metricsDirectory.appendingPathComponent(filename)
            let data = try JSONEncoder().encode(metricsHistory)
            try data.write(to: fileURL)
        } catch {
            logger.error("Failed to save metrics: \(error.localizedDescription)")
        }
    }
    
    private func loadStoredMetrics() {
        dispatchQueue.async { [weak self] in
            guard let self = self else { return }
            
            let filename = "performance_metrics.json"
            let fileURL = self.metricsDirectory.appendingPathComponent(filename)
            
            if let data = try? Data(contentsOf: fileURL),
               let metrics = try? JSONDecoder().decode([PerformanceSnapshot].self, from: data) {
                self.metricsHistory = metrics
                self.logger.info("Loaded \(metrics.count) performance metrics")
            }
        }
    }
    
    private func calculateAverageMetrics(from snapshots: [PerformanceSnapshot]) -> PerformanceSnapshot? {
        guard !snapshots.isEmpty else { return nil }
        
        let count = Double(snapshots.count)
        
        let avgMemory = MemoryMetrics(
            usedMemory: UInt64(snapshots.map { Double($0.memoryMetrics.usedMemory) }.reduce(0, +) / count),
            totalMemory: snapshots.first?.memoryMetrics.totalMemory ?? 0,
            availableMemory: UInt64(snapshots.map { Double($0.memoryMetrics.availableMemory) }.reduce(0, +) / count),
            memoryPressure: .low // Simplified
        )
        
        let avgCPU = CPUMetrics(
            userTime: snapshots.map { $0.cpuMetrics.userTime }.reduce(0, +) / count,
            systemTime: snapshots.map { $0.cpuMetrics.systemTime }.reduce(0, +) / count,
            totalTime: snapshots.map { $0.cpuMetrics.totalTime }.reduce(0, +) / count,
            usage: snapshots.map { $0.cpuMetrics.usage }.reduce(0, +) / count
        )
        
        let avgDisk = DiskMetrics(
            totalSpace: snapshots.first?.diskMetrics.totalSpace ?? 0,
            availableSpace: UInt64(snapshots.map { Double($0.diskMetrics.availableSpace) }.reduce(0, +) / count),
            usedSpace: UInt64(snapshots.map { Double($0.diskMetrics.usedSpace) }.reduce(0, +) / count),
            readOperations: UInt64(snapshots.map { Double($0.diskMetrics.readOperations) }.reduce(0, +) / count),
            writeOperations: UInt64(snapshots.map { Double($0.diskMetrics.writeOperations) }.reduce(0, +) / count)
        )
        
        let avgNetwork = NetworkMetrics(
            bytesReceived: UInt64(snapshots.map { Double($0.networkMetrics.bytesReceived) }.reduce(0, +) / count),
            bytesSent: UInt64(snapshots.map { Double($0.networkMetrics.bytesSent) }.reduce(0, +) / count),
            packetsReceived: UInt64(snapshots.map { Double($0.networkMetrics.packetsReceived) }.reduce(0, +) / count),
            packetsSent: UInt64(snapshots.map { Double($0.networkMetrics.packetsSent) }.reduce(0, +) / count),
            connectionCount: Int(snapshots.map { Double($0.networkMetrics.connectionCount) }.reduce(0, +) / count),
            latency: snapshots.map { $0.networkMetrics.latency }.reduce(0, +) / count
        )
        
        let avgBattery = BatteryMetrics(
            level: Float(snapshots.map { Double($0.batteryMetrics.level) }.reduce(0, +) / count),
            state: snapshots.first?.batteryMetrics.state ?? .unknown,
            isLowPowerModeEnabled: snapshots.contains { $0.batteryMetrics.isLowPowerModeEnabled }
        )
        
        return PerformanceSnapshot(
            timestamp: Date(),
            memoryMetrics: avgMemory,
            cpuMetrics: avgCPU,
            diskMetrics: avgDisk,
            networkMetrics: avgNetwork,
            batteryMetrics: avgBattery,
            thermalState: .nominal,
            frameRate: snapshots.map { $0.frameRate }.reduce(0, +) / count
        )
    }
    
    private func calculatePeakMetrics(from snapshots: [PerformanceSnapshot]) -> PerformanceSnapshot? {
        guard !snapshots.isEmpty else { return nil }
        
        let maxMemoryUsed = snapshots.max { $0.memoryMetrics.usedMemory < $1.memoryMetrics.usedMemory }
        let maxCPUUsage = snapshots.max { $0.cpuMetrics.usage < $1.cpuMetrics.usage }
        let minFrameRate = snapshots.min { $0.frameRate < $1.frameRate }
        
        return maxMemoryUsed ?? snapshots.first!
    }
    
    private func generateRecommendations(from snapshots: [PerformanceSnapshot]) -> [PerformanceRecommendation] {
        var recommendations: [PerformanceRecommendation] = []
        
        // Memory recommendations
        let avgMemoryUsage = snapshots.map { Double($0.memoryMetrics.usedMemory) / Double($0.memoryMetrics.totalMemory) }.reduce(0, +) / Double(snapshots.count)
        
        if avgMemoryUsage > 0.8 {
            recommendations.append(PerformanceRecommendation(
                type: .memory,
                severity: .high,
                title: "High Memory Usage",
                description: "Average memory usage is \(Int(avgMemoryUsage * 100))%. Consider optimizing memory usage.",
                actionItems: [
                    "Review image caching strategies",
                    "Check for memory leaks",
                    "Optimize data structures"
                ]
            ))
        }
        
        // Frame rate recommendations
        let avgFrameRate = snapshots.map { $0.frameRate }.reduce(0, +) / Double(snapshots.count)
        
        if avgFrameRate < 50 {
            recommendations.append(PerformanceRecommendation(
                type: .rendering,
                severity: .medium,
                title: "Low Frame Rate",
                description: "Average frame rate is \(Int(avgFrameRate)) FPS. Consider optimizing UI rendering.",
                actionItems: [
                    "Reduce view hierarchy complexity",
                    "Optimize animations",
                    "Use efficient drawing techniques"
                ]
            ))
        }
        
        // Battery recommendations
        if snapshots.contains(where: { $0.batteryMetrics.isLowPowerModeEnabled }) {
            recommendations.append(PerformanceRecommendation(
                type: .battery,
                severity: .medium,
                title: "Low Power Mode Detected",
                description: "Device is in low power mode. Consider reducing background activity.",
                actionItems: [
                    "Reduce background processing",
                    "Optimize network requests",
                    "Disable non-essential features"
                ]
            ))
        }
        
        return recommendations
    }
}

// MARK: - Supporting Types

struct PerformanceSnapshot: Codable {
    let timestamp: Date
    let memoryMetrics: MemoryMetrics
    let cpuMetrics: CPUMetrics
    let diskMetrics: DiskMetrics
    let networkMetrics: NetworkMetrics
    let batteryMetrics: BatteryMetrics
    let thermalState: ProcessInfo.ThermalState
    let frameRate: Double
}

struct MemoryMetrics: Codable {
    let usedMemory: UInt64
    let totalMemory: UInt64
    let availableMemory: UInt64
    let memoryPressure: MemoryPressureLevel
}

struct CPUMetrics: Codable {
    let userTime: Double
    let systemTime: Double
    let totalTime: Double
    let usage: Double
}

struct DiskMetrics: Codable {
    let totalSpace: UInt64
    let availableSpace: UInt64
    let usedSpace: UInt64
    let readOperations: UInt64
    let writeOperations: UInt64
}

struct NetworkMetrics: Codable {
    let bytesReceived: UInt64
    let bytesSent: UInt64
    let packetsReceived: UInt64
    let packetsSent: UInt64
    let connectionCount: Int
    let latency: Double
}

struct BatteryMetrics: Codable {
    let level: Float
    let state: BatteryState
    let isLowPowerModeEnabled: Bool
}

enum BatteryState: String, Codable {
    case unknown = "unknown"
    case unplugged = "unplugged"
    case charging = "charging"
    case full = "full"
    
    init(from batteryState: UIDevice.BatteryState) {
        switch batteryState {
        case .unknown:
            self = .unknown
        case .unplugged:
            self = .unplugged
        case .charging:
            self = .charging
        case .full:
            self = .full
        @unknown default:
            self = .unknown
        }
    }
}

enum MemoryPressureLevel: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

struct MemoryWarning: Codable {
    let timestamp: Date
    let memoryMetrics: MemoryMetrics
    let activeViewControllers: [String]
}

struct FrameDrop: Codable {
    let duration: TimeInterval
    let timestamp: Date
    let expectedDuration: TimeInterval
}

struct SlowOperation: Codable {
    let name: String
    let duration: TimeInterval
    let timestamp: Date
    let stackTrace: [String]
}

struct CustomMetric: Codable {
    let name: String
    let value: Double
    let unit: String
    let timestamp: Date
}

struct PerformanceReport {
    let timestamp: Date
    let averageMetrics: PerformanceSnapshot?
    let peakMetrics: PerformanceSnapshot?
    let memoryWarnings: [MemoryWarning]
    let frameDrops: [FrameDrop]
    let slowOperations: [SlowOperation]
    let recommendations: [PerformanceRecommendation]
}

struct PerformanceRecommendation {
    let type: RecommendationType
    let severity: RecommendationSeverity
    let title: String
    let description: String
    let actionItems: [String]
}

enum RecommendationType {
    case memory
    case cpu
    case disk
    case network
    case battery
    case rendering
    case thermal
}

enum RecommendationSeverity {
    case low
    case medium
    case high
    case critical
}

struct PerformanceExport: Codable {
    let timestamp: Date
    let metrics: [PerformanceSnapshot]
    let memoryWarnings: [MemoryWarning]
    let frameDrops: [FrameDrop]
    let slowOperations: [SlowOperation]
}

// MARK: - ProcessInfo.ThermalState Extension

extension ProcessInfo.ThermalState: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(self.rawValue)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(Int.self)
        self = ProcessInfo.ThermalState(rawValue: rawValue) ?? .nominal
    }
}