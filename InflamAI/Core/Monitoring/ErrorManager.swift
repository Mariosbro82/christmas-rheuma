//
//  ErrorManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import OSLog
import Combine
import Network
import UIKit

// MARK: - Error Manager
@MainActor
class ErrorManager: ObservableObject {
    
    // MARK: - Singleton
    static let shared = ErrorManager()
    
    // MARK: - Published Properties
    @Published var recentErrors: [AppError] = []
    @Published var crashReports: [CrashReport] = []
    @Published var performanceMetrics: PerformanceMetrics = PerformanceMetrics()
    @Published var isMonitoring = false
    @Published var networkStatus: NetworkStatus = .unknown
    @Published var memoryWarnings: [MemoryWarning] = []
    
    // MARK: - Private Properties
    private let logger = Logger(subsystem: "com.inflamai", category: "ErrorManager")
    private let crashLogger = Logger(subsystem: "com.inflamai", category: "CrashReporting")
    private let performanceLogger = Logger(subsystem: "com.inflamai", category: "Performance")
    
    private var cancellables = Set<AnyCancellable>()
    private let networkMonitor = NWPathMonitor()
    private var performanceTimer: Timer?
    private var memoryMonitorTimer: Timer?
    
    // MARK: - Error Tracking
    private var errorCounts: [String: Int] = [:]
    private var lastErrorTime: [String: Date] = [:]
    private let maxRecentErrors = 100
    private let errorThrottleInterval: TimeInterval = 1.0
    
    // MARK: - Performance Monitoring
    private var startupTime: Date?
    private var memoryBaseline: UInt64 = 0
    private var cpuUsageHistory: [Double] = []
    private var memoryUsageHistory: [UInt64] = []
    
    // MARK: - Initialization
    private init() {
        setupErrorHandling()
        setupNetworkMonitoring()
        setupPerformanceMonitoring()
        setupMemoryMonitoring()
        setupCrashReporting()
        
        startupTime = Date()
        memoryBaseline = getCurrentMemoryUsage()
    }
    
    deinit {
        performanceTimer?.invalidate()
        memoryMonitorTimer?.invalidate()
        networkMonitor.cancel()
    }
    
    // MARK: - Public Methods
    
    func startMonitoring() {
        isMonitoring = true
        
        performanceTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.updatePerformanceMetrics()
            }
        }
        
        memoryMonitorTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.checkMemoryUsage()
            }
        }
        
        logger.info("Error monitoring started")
    }
    
    func stopMonitoring() {
        isMonitoring = false
        performanceTimer?.invalidate()
        memoryMonitorTimer?.invalidate()
        
        logger.info("Error monitoring stopped")
    }
    
    func logError(_ error: Error, context: ErrorContext = ErrorContext()) {
        let appError = AppError(
            error: error,
            context: context,
            timestamp: Date(),
            stackTrace: Thread.callStackSymbols
        )
        
        // Throttle similar errors
        let errorKey = "\(type(of: error))_\(error.localizedDescription)"
        if shouldThrottleError(errorKey) {
            return
        }
        
        recentErrors.append(appError)
        
        // Keep only recent errors
        if recentErrors.count > maxRecentErrors {
            recentErrors.removeFirst(recentErrors.count - maxRecentErrors)
        }
        
        // Log to system
        logger.error("\(appError.description)")
        
        // Send to analytics if enabled
        sendErrorToAnalytics(appError)
        
        // Update error counts
        errorCounts[errorKey, default: 0] += 1
        lastErrorTime[errorKey] = Date()
    }
    
    func logCrash(_ crashInfo: CrashInfo) {
        let crashReport = CrashReport(
            crashInfo: crashInfo,
            timestamp: Date(),
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown",
            osVersion: UIDevice.current.systemVersion,
            deviceModel: UIDevice.current.model,
            memoryUsage: getCurrentMemoryUsage(),
            performanceMetrics: performanceMetrics
        )
        
        crashReports.append(crashReport)
        crashLogger.critical("Crash reported: \(crashReport.description)")
        
        // Send crash report to analytics
        sendCrashReportToAnalytics(crashReport)
    }
    
    func logPerformanceIssue(_ issue: PerformanceIssue) {
        performanceLogger.warning("Performance issue: \(issue.description)")
        
        // Add to performance metrics
        performanceMetrics.issues.append(issue)
        
        // Send to analytics
        sendPerformanceIssueToAnalytics(issue)
    }
    
    func getErrorSummary() -> ErrorSummary {
        let totalErrors = recentErrors.count
        let uniqueErrors = Set(recentErrors.map { $0.errorType }).count
        let criticalErrors = recentErrors.filter { $0.severity == .critical }.count
        let recentCrashes = crashReports.filter { $0.timestamp > Date().addingTimeInterval(-86400) }.count
        
        return ErrorSummary(
            totalErrors: totalErrors,
            uniqueErrors: uniqueErrors,
            criticalErrors: criticalErrors,
            recentCrashes: recentCrashes,
            averageMemoryUsage: memoryUsageHistory.isEmpty ? 0 : memoryUsageHistory.reduce(0, +) / UInt64(memoryUsageHistory.count),
            averageCPUUsage: cpuUsageHistory.isEmpty ? 0 : cpuUsageHistory.reduce(0, +) / Double(cpuUsageHistory.count)
        )
    }
    
    func exportErrorLogs() -> Data? {
        let exportData = ErrorExportData(
            errors: recentErrors,
            crashes: crashReports,
            performanceMetrics: performanceMetrics,
            memoryWarnings: memoryWarnings,
            exportDate: Date()
        )
        
        do {
            return try JSONEncoder().encode(exportData)
        } catch {
            logger.error("Failed to export error logs: \(error)")
            return nil
        }
    }
    
    func clearErrorLogs() {
        recentErrors.removeAll()
        crashReports.removeAll()
        memoryWarnings.removeAll()
        errorCounts.removeAll()
        lastErrorTime.removeAll()
        
        logger.info("Error logs cleared")
    }
    
    func enableCrashReporting(_ enabled: Bool) {
        UserDefaults.standard.set(enabled, forKey: "CrashReportingEnabled")
        
        if enabled {
            setupCrashReporting()
        }
    }
    
    func enableAnalytics(_ enabled: Bool) {
        UserDefaults.standard.set(enabled, forKey: "AnalyticsEnabled")
    }
    
    // MARK: - Performance Monitoring
    
    func measurePerformance<T>(_ operation: String, block: () throws -> T) rethrows -> T {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        performanceLogger.info("\(operation) took \(timeElapsed * 1000, specifier: "%.2f")ms")
        
        // Log slow operations
        if timeElapsed > 1.0 {
            let issue = PerformanceIssue(
                type: .slowOperation,
                operation: operation,
                duration: timeElapsed,
                timestamp: Date(),
                context: [:]
            )
            logPerformanceIssue(issue)
        }
        
        return result
    }
    
    func measureAsyncPerformance<T>(_ operation: String, block: () async throws -> T) async rethrows -> T {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await block()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        performanceLogger.info("\(operation) took \(timeElapsed * 1000, specifier: "%.2f")ms")
        
        // Log slow operations
        if timeElapsed > 1.0 {
            let issue = PerformanceIssue(
                type: .slowOperation,
                operation: operation,
                duration: timeElapsed,
                timestamp: Date(),
                context: [:]
            )
            logPerformanceIssue(issue)
        }
        
        return result
    }
    
    func trackMemoryUsage(_ context: String) {
        let currentUsage = getCurrentMemoryUsage()
        let usageIncrease = currentUsage > memoryBaseline ? currentUsage - memoryBaseline : 0
        
        performanceLogger.info("Memory usage in \(context): \(currentUsage / 1024 / 1024)MB (increase: \(usageIncrease / 1024 / 1024)MB)")
        
        // Check for memory leaks
        if usageIncrease > 100 * 1024 * 1024 { // 100MB increase
            let issue = PerformanceIssue(
                type: .memoryLeak,
                operation: context,
                duration: 0,
                timestamp: Date(),
                context: ["memoryUsage": currentUsage, "baseline": memoryBaseline]
            )
            logPerformanceIssue(issue)
        }
    }
    
    // MARK: - Private Methods
    
    private func setupErrorHandling() {
        // Set up global error handling
        NSSetUncaughtExceptionHandler { exception in
            let crashInfo = CrashInfo(
                type: .exception,
                reason: exception.reason ?? "Unknown exception",
                stackTrace: exception.callStackSymbols,
                userInfo: exception.userInfo
            )
            
            Task { @MainActor in
                ErrorManager.shared.logCrash(crashInfo)
            }
        }
        
        // Set up signal handling for crashes
        signal(SIGABRT) { signal in
            let crashInfo = CrashInfo(
                type: .signal,
                reason: "SIGABRT received",
                stackTrace: Thread.callStackSymbols,
                userInfo: ["signal": signal]
            )
            
            Task { @MainActor in
                ErrorManager.shared.logCrash(crashInfo)
            }
        }
        
        signal(SIGSEGV) { signal in
            let crashInfo = CrashInfo(
                type: .signal,
                reason: "SIGSEGV received",
                stackTrace: Thread.callStackSymbols,
                userInfo: ["signal": signal]
            )
            
            Task { @MainActor in
                ErrorManager.shared.logCrash(crashInfo)
            }
        }
    }
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.networkStatus = NetworkStatus(from: path)
            }
        }
        
        let queue = DispatchQueue(label: "NetworkMonitor")
        networkMonitor.start(queue: queue)
    }
    
    private func setupPerformanceMonitoring() {
        // Monitor app lifecycle events
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                self?.trackAppLaunchTime()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)
            .sink { [weak self] _ in
                self?.handleMemoryWarning()
            }
            .store(in: &cancellables)
    }
    
    private func setupMemoryMonitoring() {
        // Set up memory monitoring
    }
    
    private func setupCrashReporting() {
        guard UserDefaults.standard.bool(forKey: "CrashReportingEnabled") else { return }
        
        // Additional crash reporting setup
    }
    
    private func updatePerformanceMetrics() {
        let currentMemory = getCurrentMemoryUsage()
        let currentCPU = getCurrentCPUUsage()
        
        memoryUsageHistory.append(currentMemory)
        cpuUsageHistory.append(currentCPU)
        
        // Keep only recent history
        if memoryUsageHistory.count > 100 {
            memoryUsageHistory.removeFirst()
        }
        if cpuUsageHistory.count > 100 {
            cpuUsageHistory.removeFirst()
        }
        
        performanceMetrics.currentMemoryUsage = currentMemory
        performanceMetrics.currentCPUUsage = currentCPU
        performanceMetrics.peakMemoryUsage = max(performanceMetrics.peakMemoryUsage, currentMemory)
        performanceMetrics.averageMemoryUsage = memoryUsageHistory.reduce(0, +) / UInt64(memoryUsageHistory.count)
        performanceMetrics.averageCPUUsage = cpuUsageHistory.reduce(0, +) / Double(cpuUsageHistory.count)
    }
    
    private func checkMemoryUsage() {
        let currentMemory = getCurrentMemoryUsage()
        let memoryThreshold: UInt64 = 500 * 1024 * 1024 // 500MB
        
        if currentMemory > memoryThreshold {
            let warning = MemoryWarning(
                timestamp: Date(),
                memoryUsage: currentMemory,
                threshold: memoryThreshold,
                context: "Automatic memory check"
            )
            
            memoryWarnings.append(warning)
            
            let issue = PerformanceIssue(
                type: .highMemoryUsage,
                operation: "Memory monitoring",
                duration: 0,
                timestamp: Date(),
                context: ["memoryUsage": currentMemory, "threshold": memoryThreshold]
            )
            
            logPerformanceIssue(issue)
        }
    }
    
    private func shouldThrottleError(_ errorKey: String) -> Bool {
        guard let lastTime = lastErrorTime[errorKey] else { return false }
        return Date().timeIntervalSince(lastTime) < errorThrottleInterval
    }
    
    private func trackAppLaunchTime() {
        guard let startTime = startupTime else { return }
        
        let launchTime = Date().timeIntervalSince(startTime)
        performanceMetrics.appLaunchTime = launchTime
        
        performanceLogger.info("App launch time: \(launchTime * 1000, specifier: "%.2f")ms")
        
        if launchTime > 3.0 { // Slow launch
            let issue = PerformanceIssue(
                type: .slowLaunch,
                operation: "App launch",
                duration: launchTime,
                timestamp: Date(),
                context: [:]
            )
            logPerformanceIssue(issue)
        }
    }
    
    private func handleMemoryWarning() {
        let warning = MemoryWarning(
            timestamp: Date(),
            memoryUsage: getCurrentMemoryUsage(),
            threshold: 0,
            context: "System memory warning"
        )
        
        memoryWarnings.append(warning)
        
        logger.warning("Memory warning received")
        
        // Trigger cleanup
        performCleanup()
    }
    
    private func performCleanup() {
        // Clear caches
        URLCache.shared.removeAllCachedResponses()
        
        // Clear old error logs
        if recentErrors.count > 50 {
            recentErrors.removeFirst(recentErrors.count - 50)
        }
        
        // Clear old performance history
        if memoryUsageHistory.count > 50 {
            memoryUsageHistory.removeFirst(memoryUsageHistory.count - 50)
        }
        if cpuUsageHistory.count > 50 {
            cpuUsageHistory.removeFirst(cpuUsageHistory.count - 50)
        }
        
        logger.info("Cleanup performed due to memory pressure")
    }
    
    private func getCurrentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return info.resident_size
        }
        
        return 0
    }
    
    private func getCurrentCPUUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / Double(1024 * 1024) // Convert to percentage
        }
        
        return 0.0
    }
    
    // MARK: - Analytics Methods
    
    private func sendErrorToAnalytics(_ error: AppError) {
        guard UserDefaults.standard.bool(forKey: "AnalyticsEnabled") else { return }
        
        // Send error to analytics service
        // Implementation depends on your analytics provider
    }
    
    private func sendCrashReportToAnalytics(_ crashReport: CrashReport) {
        guard UserDefaults.standard.bool(forKey: "AnalyticsEnabled") else { return }
        
        // Send crash report to analytics service
        // Implementation depends on your analytics provider
    }
    
    private func sendPerformanceIssueToAnalytics(_ issue: PerformanceIssue) {
        guard UserDefaults.standard.bool(forKey: "AnalyticsEnabled") else { return }
        
        // Send performance issue to analytics service
        // Implementation depends on your analytics provider
    }
}

// MARK: - Supporting Types

struct AppError: Identifiable, Codable {
    let id = UUID()
    let errorType: String
    let message: String
    let context: ErrorContext
    let timestamp: Date
    let stackTrace: [String]
    let severity: ErrorSeverity
    
    init(error: Error, context: ErrorContext, timestamp: Date, stackTrace: [String]) {
        self.errorType = String(describing: type(of: error))
        self.message = error.localizedDescription
        self.context = context
        self.timestamp = timestamp
        self.stackTrace = stackTrace
        
        // Determine severity based on error type
        if error is CancellationError {
            self.severity = .low
        } else if error.localizedDescription.contains("network") {
            self.severity = .medium
        } else {
            self.severity = .high
        }
    }
    
    var description: String {
        return "[\(severity.rawValue.uppercased())] \(errorType): \(message) at \(timestamp)"
    }
}

struct ErrorContext: Codable {
    let viewController: String?
    let userAction: String?
    let additionalInfo: [String: String]
    
    init(viewController: String? = nil, userAction: String? = nil, additionalInfo: [String: String] = [:]) {
        self.viewController = viewController
        self.userAction = userAction
        self.additionalInfo = additionalInfo
    }
}

enum ErrorSeverity: String, Codable, CaseIterable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

struct CrashReport: Identifiable, Codable {
    let id = UUID()
    let crashInfo: CrashInfo
    let timestamp: Date
    let appVersion: String
    let osVersion: String
    let deviceModel: String
    let memoryUsage: UInt64
    let performanceMetrics: PerformanceMetrics
    
    var description: String {
        return "Crash: \(crashInfo.reason) at \(timestamp) (\(appVersion) on \(deviceModel))"
    }
}

struct CrashInfo: Codable {
    let type: CrashType
    let reason: String
    let stackTrace: [String]
    let userInfo: [AnyHashable: Any]?
    
    enum CodingKeys: String, CodingKey {
        case type, reason, stackTrace
    }
    
    init(type: CrashType, reason: String, stackTrace: [String], userInfo: [AnyHashable: Any]?) {
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
    case exception = "exception"
    case signal = "signal"
    case outOfMemory = "outOfMemory"
    case unknown = "unknown"
}

struct PerformanceMetrics: Codable {
    var appLaunchTime: TimeInterval = 0
    var currentMemoryUsage: UInt64 = 0
    var peakMemoryUsage: UInt64 = 0
    var averageMemoryUsage: UInt64 = 0
    var currentCPUUsage: Double = 0
    var averageCPUUsage: Double = 0
    var issues: [PerformanceIssue] = []
}

struct PerformanceIssue: Identifiable, Codable {
    let id = UUID()
    let type: PerformanceIssueType
    let operation: String
    let duration: TimeInterval
    let timestamp: Date
    let context: [String: Any]
    
    enum CodingKeys: String, CodingKey {
        case id, type, operation, duration, timestamp
    }
    
    init(type: PerformanceIssueType, operation: String, duration: TimeInterval, timestamp: Date, context: [String: Any]) {
        self.type = type
        self.operation = operation
        self.duration = duration
        self.timestamp = timestamp
        self.context = context
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        type = try container.decode(PerformanceIssueType.self, forKey: .type)
        operation = try container.decode(String.self, forKey: .operation)
        duration = try container.decode(TimeInterval.self, forKey: .duration)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        context = [:]
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(type, forKey: .type)
        try container.encode(operation, forKey: .operation)
        try container.encode(duration, forKey: .duration)
        try container.encode(timestamp, forKey: .timestamp)
    }
    
    var description: String {
        return "\(type.rawValue): \(operation) (\(duration * 1000, specifier: "%.2f")ms)"
    }
}

enum PerformanceIssueType: String, Codable {
    case slowOperation = "slowOperation"
    case slowLaunch = "slowLaunch"
    case memoryLeak = "memoryLeak"
    case highMemoryUsage = "highMemoryUsage"
    case highCPUUsage = "highCPUUsage"
    case networkTimeout = "networkTimeout"
}

struct MemoryWarning: Identifiable, Codable {
    let id = UUID()
    let timestamp: Date
    let memoryUsage: UInt64
    let threshold: UInt64
    let context: String
}

enum NetworkStatus: String, Codable {
    case satisfied = "satisfied"
    case unsatisfied = "unsatisfied"
    case requiresConnection = "requiresConnection"
    case unknown = "unknown"
    
    init(from path: NWPath) {
        switch path.status {
        case .satisfied:
            self = .satisfied
        case .unsatisfied:
            self = .unsatisfied
        case .requiresConnection:
            self = .requiresConnection
        @unknown default:
            self = .unknown
        }
    }
}

struct ErrorSummary {
    let totalErrors: Int
    let uniqueErrors: Int
    let criticalErrors: Int
    let recentCrashes: Int
    let averageMemoryUsage: UInt64
    let averageCPUUsage: Double
}

struct ErrorExportData: Codable {
    let errors: [AppError]
    let crashes: [CrashReport]
    let performanceMetrics: PerformanceMetrics
    let memoryWarnings: [MemoryWarning]
    let exportDate: Date
}

// MARK: - Error Extensions

extension Error {
    func log(context: ErrorContext = ErrorContext()) {
        ErrorManager.shared.logError(self, context: context)
    }
}

// MARK: - Performance Measurement Extensions

extension ErrorManager {
    func measure<T>(_ operation: String, _ block: () throws -> T) rethrows -> T {
        return try measurePerformance(operation, block: block)
    }
    
    func measureAsync<T>(_ operation: String, _ block: () async throws -> T) async rethrows -> T {
        return try await measureAsyncPerformance(operation, block: block)
    }
}