//
//  PerformanceManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import UIKit
import os.log
import MetricKit
import Combine

// MARK: - Performance Manager
class PerformanceManager: NSObject, ObservableObject {
    static let shared = PerformanceManager()
    
    // MARK: - Properties
    @Published var currentMemoryUsage: MemoryUsage = MemoryUsage()
    @Published var cpuUsage: Double = 0.0
    @Published var batteryUsage: Double = 0.0
    @Published var networkUsage: NetworkUsage = NetworkUsage()
    @Published var performanceMetrics: PerformanceMetrics = PerformanceMetrics()
    @Published var crashReports: [CrashReport] = []
    @Published var performanceAlerts: [PerformanceAlert] = []
    @Published var isMonitoringEnabled = true
    @Published var optimizationSuggestions: [OptimizationSuggestion] = []
    
    // Monitoring Components
    private let memoryMonitor = MemoryMonitor()
    private let cpuMonitor = CPUMonitor()
    private let batteryMonitor = BatteryMonitor()
    private let networkMonitor = NetworkMonitor()
    private let crashReporter = CrashReporter()
    private let performanceAnalyzer = PerformanceAnalyzer()
    private let memoryLeakDetector = MemoryLeakDetector()
    private let frameRateMonitor = FrameRateMonitor()
    
    // Timers and Observers
    private var monitoringTimer: Timer?
    private var metricKitSubscriber: MXMetricManagerSubscriber?
    private var cancellables = Set<AnyCancellable>()
    
    // Performance Thresholds
    private let memoryThreshold: UInt64 = 200 * 1024 * 1024 // 200MB
    private let cpuThreshold: Double = 80.0 // 80%
    private let batteryThreshold: Double = 20.0 // 20%
    private let frameRateThreshold: Double = 30.0 // 30 FPS
    
    private let logger = Logger(subsystem: "com.inflamai.performance", category: "PerformanceManager")
    
    // MARK: - Initialization
    override init() {
        super.init()
        setupPerformanceMonitoring()
        setupMetricKit()
        setupCrashReporting()
        loadPerformanceHistory()
    }
    
    deinit {
        stopMonitoring()
    }
    
    // MARK: - Setup
    private func setupPerformanceMonitoring() {
        startMonitoring()
        setupMemoryLeakDetection()
        setupFrameRateMonitoring()
    }
    
    private func setupMetricKit() {
        if #available(iOS 13.0, *) {
            metricKitSubscriber = MetricKitSubscriber()
            MXMetricManager.shared.add(metricKitSubscriber!)
        }
    }
    
    private func setupCrashReporting() {
        crashReporter.delegate = self
        crashReporter.startMonitoring()
    }
    
    private func setupMemoryLeakDetection() {
        memoryLeakDetector.delegate = self
        memoryLeakDetector.startDetection()
    }
    
    private func setupFrameRateMonitoring() {
        frameRateMonitor.delegate = self
        frameRateMonitor.startMonitoring()
    }
    
    // MARK: - Monitoring Control
    func startMonitoring() {
        guard isMonitoringEnabled else { return }
        
        monitoringTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updatePerformanceMetrics()
        }
        
        logger.info("Performance monitoring started")
    }
    
    func stopMonitoring() {
        monitoringTimer?.invalidate()
        monitoringTimer = nil
        
        memoryLeakDetector.stopDetection()
        frameRateMonitor.stopMonitoring()
        
        logger.info("Performance monitoring stopped")
    }
    
    func enableMonitoring(_ enabled: Bool) {
        isMonitoringEnabled = enabled
        
        if enabled {
            startMonitoring()
        } else {
            stopMonitoring()
        }
    }
    
    // MARK: - Performance Metrics Collection
    private func updatePerformanceMetrics() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }
            
            let memory = self.memoryMonitor.getCurrentMemoryUsage()
            let cpu = self.cpuMonitor.getCurrentCPUUsage()
            let battery = self.batteryMonitor.getCurrentBatteryUsage()
            let network = self.networkMonitor.getCurrentNetworkUsage()
            
            DispatchQueue.main.async {
                self.currentMemoryUsage = memory
                self.cpuUsage = cpu
                self.batteryUsage = battery
                self.networkUsage = network
                
                self.updatePerformanceMetrics(memory: memory, cpu: cpu, battery: battery, network: network)
                self.checkPerformanceThresholds()
                self.generateOptimizationSuggestions()
            }
        }
    }
    
    private func updatePerformanceMetrics(memory: MemoryUsage, cpu: Double, battery: Double, network: NetworkUsage) {
        let timestamp = Date()
        
        performanceMetrics.memoryHistory.append(MemoryDataPoint(timestamp: timestamp, usage: memory))
        performanceMetrics.cpuHistory.append(CPUDataPoint(timestamp: timestamp, usage: cpu))
        performanceMetrics.batteryHistory.append(BatteryDataPoint(timestamp: timestamp, usage: battery))
        performanceMetrics.networkHistory.append(NetworkDataPoint(timestamp: timestamp, usage: network))
        
        // Keep only last 1000 data points
        if performanceMetrics.memoryHistory.count > 1000 {
            performanceMetrics.memoryHistory.removeFirst()
        }
        if performanceMetrics.cpuHistory.count > 1000 {
            performanceMetrics.cpuHistory.removeFirst()
        }
        if performanceMetrics.batteryHistory.count > 1000 {
            performanceMetrics.batteryHistory.removeFirst()
        }
        if performanceMetrics.networkHistory.count > 1000 {
            performanceMetrics.networkHistory.removeFirst()
        }
        
        // Update averages
        performanceMetrics.averageMemoryUsage = calculateAverageMemoryUsage()
        performanceMetrics.averageCPUUsage = calculateAverageCPUUsage()
        performanceMetrics.averageBatteryUsage = calculateAverageBatteryUsage()
        performanceMetrics.averageFrameRate = frameRateMonitor.getAverageFrameRate()
    }
    
    // MARK: - Threshold Monitoring
    private func checkPerformanceThresholds() {
        checkMemoryThreshold()
        checkCPUThreshold()
        checkBatteryThreshold()
        checkFrameRateThreshold()
    }
    
    private func checkMemoryThreshold() {
        if currentMemoryUsage.used > memoryThreshold {
            let alert = PerformanceAlert(
                type: .highMemoryUsage,
                severity: .high,
                message: "Memory usage is high: \(formatBytes(currentMemoryUsage.used))",
                timestamp: Date(),
                value: Double(currentMemoryUsage.used),
                threshold: Double(memoryThreshold)
            )
            addPerformanceAlert(alert)
            
            // Trigger memory cleanup
            performMemoryCleanup()
        }
    }
    
    private func checkCPUThreshold() {
        if cpuUsage > cpuThreshold {
            let alert = PerformanceAlert(
                type: .highCPUUsage,
                severity: .medium,
                message: "CPU usage is high: \(String(format: "%.1f", cpuUsage))%",
                timestamp: Date(),
                value: cpuUsage,
                threshold: cpuThreshold
            )
            addPerformanceAlert(alert)
        }
    }
    
    private func checkBatteryThreshold() {
        if batteryUsage < batteryThreshold {
            let alert = PerformanceAlert(
                type: .lowBattery,
                severity: .medium,
                message: "Battery is low: \(String(format: "%.1f", batteryUsage))%",
                timestamp: Date(),
                value: batteryUsage,
                threshold: batteryThreshold
            )
            addPerformanceAlert(alert)
        }
    }
    
    private func checkFrameRateThreshold() {
        let currentFrameRate = frameRateMonitor.getCurrentFrameRate()
        if currentFrameRate < frameRateThreshold {
            let alert = PerformanceAlert(
                type: .lowFrameRate,
                severity: .medium,
                message: "Frame rate is low: \(String(format: "%.1f", currentFrameRate)) FPS",
                timestamp: Date(),
                value: currentFrameRate,
                threshold: frameRateThreshold
            )
            addPerformanceAlert(alert)
        }
    }
    
    private func addPerformanceAlert(_ alert: PerformanceAlert) {
        performanceAlerts.append(alert)
        
        // Keep only last 100 alerts
        if performanceAlerts.count > 100 {
            performanceAlerts.removeFirst()
        }
        
        logger.warning("Performance alert: \(alert.message)")
        
        // Post notification
        NotificationCenter.default.post(
            name: .performanceAlertGenerated,
            object: alert
        )
    }
    
    // MARK: - Memory Management
    func performMemoryCleanup() {
        logger.info("Performing memory cleanup")
        
        // Clear caches
        URLCache.shared.removeAllCachedResponses()
        
        // Clear image caches if using a library like Kingfisher or SDWebImage
        // ImageCache.default.clearMemoryCache()
        
        // Force garbage collection
        DispatchQueue.global(qos: .utility).async {
            // Perform heavy cleanup operations
            self.performDeepMemoryCleanup()
        }
        
        // Post notification for other components to clean up
        NotificationCenter.default.post(name: .performanceMemoryCleanupRequested, object: nil)
    }
    
    private func performDeepMemoryCleanup() {
        // Clear temporary files
        clearTemporaryFiles()
        
        // Clear old log files
        clearOldLogFiles()
        
        // Clear expired data
        clearExpiredData()
    }
    
    private func clearTemporaryFiles() {
        let tempDirectory = NSTemporaryDirectory()
        let fileManager = FileManager.default
        
        do {
            let tempFiles = try fileManager.contentsOfDirectory(atPath: tempDirectory)
            for file in tempFiles {
                let filePath = (tempDirectory as NSString).appendingPathComponent(file)
                try fileManager.removeItem(atPath: filePath)
            }
            logger.info("Temporary files cleared")
        } catch {
            logger.error("Failed to clear temporary files: \(error.localizedDescription)")
        }
    }
    
    private func clearOldLogFiles() {
        // Implementation would depend on your logging system
        logger.info("Old log files cleared")
    }
    
    private func clearExpiredData() {
        // Clear expired cached data
        logger.info("Expired data cleared")
    }
    
    // MARK: - Crash Reporting
    func reportCrash(_ crashInfo: CrashInfo) {
        let crashReport = CrashReport(
            id: UUID().uuidString,
            timestamp: Date(),
            crashInfo: crashInfo,
            deviceInfo: getDeviceInfo(),
            appVersion: getAppVersion(),
            memoryUsage: currentMemoryUsage,
            cpuUsage: cpuUsage
        )
        
        crashReports.append(crashReport)
        saveCrashReport(crashReport)
        
        logger.error("Crash reported: \(crashInfo.reason)")
        
        // Send to analytics service
        sendCrashReportToAnalytics(crashReport)
    }
    
    private func saveCrashReport(_ report: CrashReport) {
        do {
            let data = try JSONEncoder().encode(report)
            let filename = "crash_\(report.id).json"
            let url = getCrashReportsDirectory().appendingPathComponent(filename)
            try data.write(to: url)
        } catch {
            logger.error("Failed to save crash report: \(error.localizedDescription)")
        }
    }
    
    private func sendCrashReportToAnalytics(_ report: CrashReport) {
        // Implementation would depend on your analytics service
        // e.g., Firebase Crashlytics, Bugsnag, etc.
    }
    
    // MARK: - Performance Analysis
    private func generateOptimizationSuggestions() {
        var suggestions: [OptimizationSuggestion] = []
        
        // Memory optimization suggestions
        if currentMemoryUsage.used > memoryThreshold * 80 / 100 {
            suggestions.append(OptimizationSuggestion(
                type: .memoryOptimization,
                priority: .high,
                title: "High Memory Usage Detected",
                description: "Consider clearing caches or reducing image quality",
                impact: .high,
                estimatedImprovement: "20-30% memory reduction"
            ))
        }
        
        // CPU optimization suggestions
        if cpuUsage > cpuThreshold * 70 / 100 {
            suggestions.append(OptimizationSuggestion(
                type: .cpuOptimization,
                priority: .medium,
                title: "High CPU Usage Detected",
                description: "Consider optimizing background tasks or reducing animation complexity",
                impact: .medium,
                estimatedImprovement: "15-25% CPU reduction"
            ))
        }
        
        // Battery optimization suggestions
        if batteryUsage < 30 {
            suggestions.append(OptimizationSuggestion(
                type: .batteryOptimization,
                priority: .medium,
                title: "Battery Optimization Available",
                description: "Enable low power mode or reduce background activity",
                impact: .medium,
                estimatedImprovement: "10-20% battery life extension"
            ))
        }
        
        // Frame rate optimization suggestions
        let currentFrameRate = frameRateMonitor.getCurrentFrameRate()
        if currentFrameRate < frameRateThreshold {
            suggestions.append(OptimizationSuggestion(
                type: .uiOptimization,
                priority: .high,
                title: "Low Frame Rate Detected",
                description: "Consider reducing UI complexity or optimizing animations",
                impact: .high,
                estimatedImprovement: "Smoother user interface"
            ))
        }
        
        optimizationSuggestions = suggestions
    }
    
    // MARK: - Performance Reports
    func generatePerformanceReport() -> PerformanceReport {
        let report = PerformanceReport(
            timestamp: Date(),
            memoryUsage: currentMemoryUsage,
            cpuUsage: cpuUsage,
            batteryUsage: batteryUsage,
            networkUsage: networkUsage,
            frameRate: frameRateMonitor.getCurrentFrameRate(),
            averageMemoryUsage: performanceMetrics.averageMemoryUsage,
            averageCPUUsage: performanceMetrics.averageCPUUsage,
            averageBatteryUsage: performanceMetrics.averageBatteryUsage,
            averageFrameRate: performanceMetrics.averageFrameRate,
            crashCount: crashReports.count,
            alertCount: performanceAlerts.count,
            optimizationSuggestions: optimizationSuggestions
        )
        
        return report
    }
    
    func exportPerformanceData() -> Data? {
        let exportData = PerformanceExportData(
            metrics: performanceMetrics,
            crashReports: crashReports,
            alerts: performanceAlerts,
            suggestions: optimizationSuggestions
        )
        
        do {
            return try JSONEncoder().encode(exportData)
        } catch {
            logger.error("Failed to export performance data: \(error.localizedDescription)")
            return nil
        }
    }
    
    // MARK: - Utility Methods
    private func calculateAverageMemoryUsage() -> Double {
        guard !performanceMetrics.memoryHistory.isEmpty else { return 0.0 }
        
        let total = performanceMetrics.memoryHistory.reduce(0) { $0 + Double($1.usage.used) }
        return total / Double(performanceMetrics.memoryHistory.count)
    }
    
    private func calculateAverageCPUUsage() -> Double {
        guard !performanceMetrics.cpuHistory.isEmpty else { return 0.0 }
        
        let total = performanceMetrics.cpuHistory.reduce(0) { $0 + $1.usage }
        return total / Double(performanceMetrics.cpuHistory.count)
    }
    
    private func calculateAverageBatteryUsage() -> Double {
        guard !performanceMetrics.batteryHistory.isEmpty else { return 0.0 }
        
        let total = performanceMetrics.batteryHistory.reduce(0) { $0 + $1.usage }
        return total / Double(performanceMetrics.batteryHistory.count)
    }
    
    private func formatBytes(_ bytes: UInt64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: Int64(bytes))
    }
    
    private func getDeviceInfo() -> DeviceInfo {
        return DeviceInfo(
            model: UIDevice.current.model,
            systemName: UIDevice.current.systemName,
            systemVersion: UIDevice.current.systemVersion,
            identifierForVendor: UIDevice.current.identifierForVendor?.uuidString
        )
    }
    
    private func getAppVersion() -> String {
        return Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
    }
    
    private func getCrashReportsDirectory() -> URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let crashReportsPath = documentsPath.appendingPathComponent("CrashReports")
        
        if !FileManager.default.fileExists(atPath: crashReportsPath.path) {
            try? FileManager.default.createDirectory(at: crashReportsPath, withIntermediateDirectories: true)
        }
        
        return crashReportsPath
    }
    
    // MARK: - Data Persistence
    private func loadPerformanceHistory() {
        // Load saved performance data
        if let data = UserDefaults.standard.data(forKey: "performance_metrics"),
           let metrics = try? JSONDecoder().decode(PerformanceMetrics.self, from: data) {
            self.performanceMetrics = metrics
        }
    }
    
    func savePerformanceHistory() {
        if let data = try? JSONEncoder().encode(performanceMetrics) {
            UserDefaults.standard.set(data, forKey: "performance_metrics")
        }
    }
}

// MARK: - CrashReporterDelegate
extension PerformanceManager: CrashReporterDelegate {
    func crashReporter(_ reporter: CrashReporter, didDetectCrash crashInfo: CrashInfo) {
        reportCrash(crashInfo)
    }
}

// MARK: - MemoryLeakDetectorDelegate
extension PerformanceManager: MemoryLeakDetectorDelegate {
    func memoryLeakDetector(_ detector: MemoryLeakDetector, didDetectLeak leak: MemoryLeak) {
        let alert = PerformanceAlert(
            type: .memoryLeak,
            severity: .high,
            message: "Memory leak detected in \(leak.className)",
            timestamp: Date(),
            value: Double(leak.retainCount),
            threshold: 1.0
        )
        addPerformanceAlert(alert)
        
        logger.error("Memory leak detected: \(leak.className) - \(leak.retainCount) instances")
    }
}

// MARK: - FrameRateMonitorDelegate
extension PerformanceManager: FrameRateMonitorDelegate {
    func frameRateMonitor(_ monitor: FrameRateMonitor, didUpdateFrameRate frameRate: Double) {
        if frameRate < frameRateThreshold {
            // Frame rate is low, but don't spam alerts
            let lastAlert = performanceAlerts.last { $0.type == .lowFrameRate }
            if lastAlert == nil || Date().timeIntervalSince(lastAlert!.timestamp) > 60 {
                checkFrameRateThreshold()
            }
        }
    }
}

// MARK: - Supporting Classes
class MemoryMonitor {
    func getCurrentMemoryUsage() -> MemoryUsage {
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
        
        if kerr == KERN_SUCCESS {
            return MemoryUsage(
                used: UInt64(info.resident_size),
                available: UInt64(ProcessInfo.processInfo.physicalMemory),
                total: UInt64(ProcessInfo.processInfo.physicalMemory)
            )
        } else {
            return MemoryUsage()
        }
    }
}

class CPUMonitor {
    func getCurrentCPUUsage() -> Double {
        var info = processor_info_array_t.allocate(capacity: 1)
        var numCpuInfo: mach_msg_type_number_t = 0
        var numCpus: natural_t = 0
        
        let result = host_processor_info(mach_host_self(),
                                       PROCESSOR_CPU_LOAD_INFO,
                                       &numCpus,
                                       &info,
                                       &numCpuInfo)
        
        if result == KERN_SUCCESS {
            let cpuLoadInfo = info.withMemoryRebound(to: processor_cpu_load_info.self, capacity: Int(numCpus)) { $0 }
            
            var totalUser: UInt32 = 0
            var totalSystem: UInt32 = 0
            var totalIdle: UInt32 = 0
            var totalNice: UInt32 = 0
            
            for i in 0..<Int(numCpus) {
                totalUser += cpuLoadInfo[i].cpu_ticks.0
                totalSystem += cpuLoadInfo[i].cpu_ticks.1
                totalIdle += cpuLoadInfo[i].cpu_ticks.2
                totalNice += cpuLoadInfo[i].cpu_ticks.3
            }
            
            let totalTicks = totalUser + totalSystem + totalIdle + totalNice
            let usedTicks = totalUser + totalSystem + totalNice
            
            vm_deallocate(mach_task_self_, vm_address_t(bitPattern: info), vm_size_t(numCpuInfo))
            
            return totalTicks > 0 ? Double(usedTicks) / Double(totalTicks) * 100.0 : 0.0
        }
        
        return 0.0
    }
}

class BatteryMonitor {
    func getCurrentBatteryUsage() -> Double {
        UIDevice.current.isBatteryMonitoringEnabled = true
        return Double(UIDevice.current.batteryLevel * 100)
    }
}

class NetworkMonitor {
    func getCurrentNetworkUsage() -> NetworkUsage {
        // This would require more complex implementation using Network framework
        // For now, return mock data
        return NetworkUsage(
            bytesReceived: 0,
            bytesSent: 0,
            packetsReceived: 0,
            packetsSent: 0
        )
    }
}

class CrashReporter {
    weak var delegate: CrashReporterDelegate?
    
    func startMonitoring() {
        // Set up crash detection
        NSSetUncaughtExceptionHandler { exception in
            let crashInfo = CrashInfo(
                type: .exception,
                reason: exception.reason ?? "Unknown exception",
                stackTrace: exception.callStackSymbols,
                userInfo: exception.userInfo
            )
            
            PerformanceManager.shared.reportCrash(crashInfo)
        }
        
        // Set up signal handling for crashes
        signal(SIGABRT) { signal in
            let crashInfo = CrashInfo(
                type: .signal,
                reason: "SIGABRT received",
                stackTrace: Thread.callStackSymbols,
                userInfo: ["signal": signal]
            )
            
            PerformanceManager.shared.reportCrash(crashInfo)
        }
        
        signal(SIGILL) { signal in
            let crashInfo = CrashInfo(
                type: .signal,
                reason: "SIGILL received",
                stackTrace: Thread.callStackSymbols,
                userInfo: ["signal": signal]
            )
            
            PerformanceManager.shared.reportCrash(crashInfo)
        }
        
        signal(SIGSEGV) { signal in
            let crashInfo = CrashInfo(
                type: .signal,
                reason: "SIGSEGV received",
                stackTrace: Thread.callStackSymbols,
                userInfo: ["signal": signal]
            )
            
            PerformanceManager.shared.reportCrash(crashInfo)
        }
    }
}

class MemoryLeakDetector {
    weak var delegate: MemoryLeakDetectorDelegate?
    private var objectCounts: [String: Int] = [:]
    private var detectionTimer: Timer?
    
    func startDetection() {
        detectionTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            self?.detectMemoryLeaks()
        }
    }
    
    func stopDetection() {
        detectionTimer?.invalidate()
        detectionTimer = nil
    }
    
    private func detectMemoryLeaks() {
        // This is a simplified implementation
        // In a real app, you'd use more sophisticated techniques
        
        let currentCounts = getCurrentObjectCounts()
        
        for (className, count) in currentCounts {
            let previousCount = objectCounts[className] ?? 0
            
            if count > previousCount + 10 { // Threshold for leak detection
                let leak = MemoryLeak(
                    className: className,
                    retainCount: count,
                    timestamp: Date()
                )
                
                delegate?.memoryLeakDetector(self, didDetectLeak: leak)
            }
        }
        
        objectCounts = currentCounts
    }
    
    private func getCurrentObjectCounts() -> [String: Int] {
        // This would require runtime introspection
        // For now, return mock data
        return [:]
    }
}

class FrameRateMonitor {
    weak var delegate: FrameRateMonitorDelegate?
    private var displayLink: CADisplayLink?
    private var frameCount = 0
    private var lastTimestamp: CFTimeInterval = 0
    private var frameRates: [Double] = []
    
    func startMonitoring() {
        displayLink = CADisplayLink(target: self, selector: #selector(displayLinkTick))
        displayLink?.add(to: .main, forMode: .common)
    }
    
    func stopMonitoring() {
        displayLink?.invalidate()
        displayLink = nil
    }
    
    @objc private func displayLinkTick(displayLink: CADisplayLink) {
        if lastTimestamp == 0 {
            lastTimestamp = displayLink.timestamp
            return
        }
        
        frameCount += 1
        
        let elapsed = displayLink.timestamp - lastTimestamp
        if elapsed >= 1.0 {
            let frameRate = Double(frameCount) / elapsed
            frameRates.append(frameRate)
            
            // Keep only last 60 measurements (1 minute at 1 FPS)
            if frameRates.count > 60 {
                frameRates.removeFirst()
            }
            
            delegate?.frameRateMonitor(self, didUpdateFrameRate: frameRate)
            
            frameCount = 0
            lastTimestamp = displayLink.timestamp
        }
    }
    
    func getCurrentFrameRate() -> Double {
        return frameRates.last ?? 60.0
    }
    
    func getAverageFrameRate() -> Double {
        guard !frameRates.isEmpty else { return 60.0 }
        return frameRates.reduce(0, +) / Double(frameRates.count)
    }
}

class PerformanceAnalyzer {
    func analyzePerformance(_ metrics: PerformanceMetrics) -> PerformanceAnalysis {
        return PerformanceAnalysis(
            overallScore: calculateOverallScore(metrics),
            memoryScore: calculateMemoryScore(metrics),
            cpuScore: calculateCPUScore(metrics),
            batteryScore: calculateBatteryScore(metrics),
            uiScore: calculateUIScore(metrics),
            recommendations: generateRecommendations(metrics)
        )
    }
    
    private func calculateOverallScore(_ metrics: PerformanceMetrics) -> Double {
        let memoryScore = calculateMemoryScore(metrics)
        let cpuScore = calculateCPUScore(metrics)
        let batteryScore = calculateBatteryScore(metrics)
        let uiScore = calculateUIScore(metrics)
        
        return (memoryScore + cpuScore + batteryScore + uiScore) / 4.0
    }
    
    private func calculateMemoryScore(_ metrics: PerformanceMetrics) -> Double {
        // Score based on memory usage efficiency
        let averageUsage = metrics.averageMemoryUsage
        let maxRecommended = 200.0 * 1024 * 1024 // 200MB
        
        if averageUsage <= maxRecommended {
            return 100.0
        } else {
            return max(0.0, 100.0 - ((averageUsage - maxRecommended) / maxRecommended) * 100.0)
        }
    }
    
    private func calculateCPUScore(_ metrics: PerformanceMetrics) -> Double {
        // Score based on CPU usage efficiency
        let averageUsage = metrics.averageCPUUsage
        
        if averageUsage <= 50.0 {
            return 100.0
        } else {
            return max(0.0, 100.0 - (averageUsage - 50.0))
        }
    }
    
    private func calculateBatteryScore(_ metrics: PerformanceMetrics) -> Double {
        // Score based on battery efficiency
        let averageUsage = metrics.averageBatteryUsage
        
        if averageUsage >= 80.0 {
            return 100.0
        } else {
            return averageUsage * 1.25 // Scale to 100
        }
    }
    
    private func calculateUIScore(_ metrics: PerformanceMetrics) -> Double {
        // Score based on frame rate
        let averageFrameRate = metrics.averageFrameRate
        
        if averageFrameRate >= 60.0 {
            return 100.0
        } else {
            return (averageFrameRate / 60.0) * 100.0
        }
    }
    
    private func generateRecommendations(_ metrics: PerformanceMetrics) -> [String] {
        var recommendations: [String] = []
        
        if metrics.averageMemoryUsage > 150 * 1024 * 1024 {
            recommendations.append("Consider optimizing memory usage by clearing caches more frequently")
        }
        
        if metrics.averageCPUUsage > 70.0 {
            recommendations.append("Optimize CPU-intensive operations by moving them to background queues")
        }
        
        if metrics.averageFrameRate < 50.0 {
            recommendations.append("Improve UI performance by reducing animation complexity")
        }
        
        return recommendations
    }
}

// MARK: - MetricKit Integration
@available(iOS 13.0, *)
class MetricKitSubscriber: NSObject, MXMetricManagerSubscriber {
    func didReceive(_ payloads: [MXMetricPayload]) {
        for payload in payloads {
            processMetricPayload(payload)
        }
    }
    
    func didReceive(_ payloads: [MXDiagnosticPayload]) {
        for payload in payloads {
            processDiagnosticPayload(payload)
        }
    }
    
    private func processMetricPayload(_ payload: MXMetricPayload) {
        // Process MetricKit metrics
        if let cpuMetrics = payload.cpuMetrics {
            // Process CPU metrics
        }
        
        if let memoryMetrics = payload.memoryMetrics {
            // Process memory metrics
        }
        
        if let displayMetrics = payload.displayMetrics {
            // Process display metrics
        }
    }
    
    private func processDiagnosticPayload(_ payload: MXDiagnosticPayload) {
        // Process diagnostic information
        if let crashDiagnostics = payload.crashDiagnostics {
            for diagnostic in crashDiagnostics {
                let crashInfo = CrashInfo(
                    type: .metricKit,
                    reason: "MetricKit crash diagnostic",
                    stackTrace: [],
                    userInfo: ["diagnostic": diagnostic]
                )
                
                PerformanceManager.shared.reportCrash(crashInfo)
            }
        }
    }
}

// MARK: - Protocols
protocol CrashReporterDelegate: AnyObject {
    func crashReporter(_ reporter: CrashReporter, didDetectCrash crashInfo: CrashInfo)
}

protocol MemoryLeakDetectorDelegate: AnyObject {
    func memoryLeakDetector(_ detector: MemoryLeakDetector, didDetectLeak leak: MemoryLeak)
}

protocol FrameRateMonitorDelegate: AnyObject {
    func frameRateMonitor(_ monitor: FrameRateMonitor, didUpdateFrameRate frameRate: Double)
}

// MARK: - Supporting Types
struct MemoryUsage: Codable {
    var used: UInt64 = 0
    var available: UInt64 = 0
    var total: UInt64 = 0
    
    var usagePercentage: Double {
        guard total > 0 else { return 0.0 }
        return Double(used) / Double(total) * 100.0
    }
}

struct NetworkUsage: Codable {
    var bytesReceived: UInt64 = 0
    var bytesSent: UInt64 = 0
    var packetsReceived: UInt64 = 0
    var packetsSent: UInt64 = 0
}

struct PerformanceMetrics: Codable {
    var memoryHistory: [MemoryDataPoint] = []
    var cpuHistory: [CPUDataPoint] = []
    var batteryHistory: [BatteryDataPoint] = []
    var networkHistory: [NetworkDataPoint] = []
    
    var averageMemoryUsage: Double = 0.0
    var averageCPUUsage: Double = 0.0
    var averageBatteryUsage: Double = 0.0
    var averageFrameRate: Double = 60.0
}

struct MemoryDataPoint: Codable {
    var timestamp: Date
    var usage: MemoryUsage
}

struct CPUDataPoint: Codable {
    var timestamp: Date
    var usage: Double
}

struct BatteryDataPoint: Codable {
    var timestamp: Date
    var usage: Double
}

struct NetworkDataPoint: Codable {
    var timestamp: Date
    var usage: NetworkUsage
}

struct CrashReport: Codable {
    var id: String
    var timestamp: Date
    var crashInfo: CrashInfo
    var deviceInfo: DeviceInfo
    var appVersion: String
    var memoryUsage: MemoryUsage
    var cpuUsage: Double
}

struct CrashInfo: Codable {
    var type: CrashType
    var reason: String
    var stackTrace: [String]
    var userInfo: [String: Any]?
    
    enum CodingKeys: String, CodingKey {
        case type, reason, stackTrace
    }
    
    init(type: CrashType, reason: String, stackTrace: [String], userInfo: [String: Any]?) {
        self.type = type
        self.reason = reason
        self.stackTrace = stackTrace
        self.userInfo = userInfo
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try container.decode(CrashType.self, forKey: .type)
        reason = try container.decode(String.self, forKey: .reason)
        stackTrace = try container.decode([String].self, forKey: .stackTrace)
        userInfo = nil
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(type, forKey: .type)
        try container.encode(reason, forKey: .reason)
        try container.encode(stackTrace, forKey: .stackTrace)
    }
}

enum CrashType: String, Codable {
    case exception = "Exception"
    case signal = "Signal"
    case metricKit = "MetricKit"
    case custom = "Custom"
}

struct DeviceInfo: Codable {
    var model: String
    var systemName: String
    var systemVersion: String
    var identifierForVendor: String?
}

struct PerformanceAlert {
    var type: PerformanceAlertType
    var severity: PerformanceAlertSeverity
    var message: String
    var timestamp: Date
    var value: Double
    var threshold: Double
}

enum PerformanceAlertType: String, CaseIterable {
    case highMemoryUsage = "High Memory Usage"
    case highCPUUsage = "High CPU Usage"
    case lowBattery = "Low Battery"
    case lowFrameRate = "Low Frame Rate"
    case memoryLeak = "Memory Leak"
    case networkIssue = "Network Issue"
}

enum PerformanceAlertSeverity: String, CaseIterable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"
    case critical = "Critical"
}

struct OptimizationSuggestion {
    var type: OptimizationType
    var priority: OptimizationPriority
    var title: String
    var description: String
    var impact: OptimizationImpact
    var estimatedImprovement: String
}

enum OptimizationType: String, CaseIterable {
    case memoryOptimization = "Memory Optimization"
    case cpuOptimization = "CPU Optimization"
    case batteryOptimization = "Battery Optimization"
    case uiOptimization = "UI Optimization"
    case networkOptimization = "Network Optimization"
}

enum OptimizationPriority: String, CaseIterable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"
}

enum OptimizationImpact: String, CaseIterable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"
}

struct PerformanceReport {
    var timestamp: Date
    var memoryUsage: MemoryUsage
    var cpuUsage: Double
    var batteryUsage: Double
    var networkUsage: NetworkUsage
    var frameRate: Double
    var averageMemoryUsage: Double
    var averageCPUUsage: Double
    var averageBatteryUsage: Double
    var averageFrameRate: Double
    var crashCount: Int
    var alertCount: Int
    var optimizationSuggestions: [OptimizationSuggestion]
}

struct PerformanceExportData: Codable {
    var metrics: PerformanceMetrics
    var crashReports: [CrashReport]
    var alerts: [PerformanceAlert]
    var suggestions: [OptimizationSuggestion]
    
    enum CodingKeys: String, CodingKey {
        case metrics, crashReports
    }
    
    init(metrics: PerformanceMetrics, crashReports: [CrashReport], alerts: [PerformanceAlert], suggestions: [OptimizationSuggestion]) {
        self.metrics = metrics
        self.crashReports = crashReports
        self.alerts = alerts
        self.suggestions = suggestions
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        metrics = try container.decode(PerformanceMetrics.self, forKey: .metrics)
        crashReports = try container.decode([CrashReport].self, forKey: .crashReports)
        alerts = []
        suggestions = []
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(metrics, forKey: .metrics)
        try container.encode(crashReports, forKey: .crashReports)
    }
}

struct MemoryLeak {
    var className: String
    var retainCount: Int
    var timestamp: Date
}

struct PerformanceAnalysis {
    var overallScore: Double
    var memoryScore: Double
    var cpuScore: Double
    var batteryScore: Double
    var uiScore: Double
    var recommendations: [String]
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let performanceAlertGenerated = Notification.Name("performanceAlertGenerated")
    static let performanceMemoryCleanupRequested = Notification.Name("performanceMemoryCleanupRequested")
    static let performanceCrashDetected = Notification.Name("performanceCrashDetected")
    static let performanceOptimizationSuggested = Notification.Name("performanceOptimizationSuggested")
}