//
//  CrashReportingManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import os.log
import UIKit
import Network
import CryptoKit

// MARK: - Crash Reporting Manager
class CrashReportingManager: NSObject, ObservableObject {
    
    static let shared = CrashReportingManager()
    
    private let logger = Logger(subsystem: "InflamAI", category: "CrashReporting")
    private let fileManager = FileManager.default
    private let networkMonitor = NWPathMonitor()
    private let dispatchQueue = DispatchQueue(label: "crash-reporting", qos: .utility)
    
    // Configuration
    private let maxCrashReports = 50
    private let maxLogEntries = 1000
    private let reportRetentionDays = 30
    private let uploadRetryLimit = 3
    
    // Storage paths
    private lazy var crashReportsDirectory: URL = {
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let crashDir = documentsPath.appendingPathComponent("CrashReports")
        try? fileManager.createDirectory(at: crashDir, withIntermediateDirectories: true)
        return crashDir
    }()
    
    private lazy var analyticsDirectory: URL = {
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let analyticsDir = documentsPath.appendingPathComponent("Analytics")
        try? fileManager.createDirectory(at: analyticsDir, withIntermediateDirectories: true)
        return analyticsDir
    }()
    
    // Published properties
    @Published var isEnabled = true
    @Published var crashReports: [CrashReport] = []
    @Published var performanceMetrics: PerformanceMetrics?
    @Published var analyticsEvents: [AnalyticsEvent] = []
    @Published var networkStatus: NetworkStatus = .unknown
    
    // Internal state
    private var isNetworkAvailable = false
    private var pendingUploads: [String] = []
    private var performanceTimer: Timer?
    private var memoryWarningObserver: NSObjectProtocol?
    private var appStateObserver: NSObjectProtocol?
    
    override init() {
        super.init()
        setupCrashReporting()
        setupPerformanceMonitoring()
        setupNetworkMonitoring()
        setupAppStateObservers()
        loadStoredData()
        cleanupOldReports()
    }
    
    deinit {
        performanceTimer?.invalidate()
        if let observer = memoryWarningObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        if let observer = appStateObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        networkMonitor.cancel()
    }
    
    // MARK: - Public Methods
    
    func reportCrash(_ error: Error, context: CrashContext? = nil) {
        guard isEnabled else { return }
        
        dispatchQueue.async { [weak self] in
            self?.handleCrash(error, context: context)
        }
    }
    
    func reportError(_ error: Error, severity: ErrorSeverity = .medium, context: [String: Any] = [:]) {
        guard isEnabled else { return }
        
        let errorReport = ErrorReport(
            id: UUID().uuidString,
            timestamp: Date(),
            error: error,
            severity: severity,
            context: context,
            deviceInfo: collectDeviceInfo(),
            appInfo: collectAppInfo(),
            userInfo: collectUserInfo()
        )
        
        dispatchQueue.async { [weak self] in
            self?.storeErrorReport(errorReport)
            self?.uploadErrorReportIfPossible(errorReport)
        }
        
        logger.error("Error reported: \(error.localizedDescription)")
    }
    
    func trackEvent(_ event: String, parameters: [String: Any] = [:]) {
        guard isEnabled else { return }
        
        let analyticsEvent = AnalyticsEvent(
            id: UUID().uuidString,
            name: event,
            parameters: parameters,
            timestamp: Date(),
            sessionId: getCurrentSessionId(),
            userId: getCurrentUserId()
        )
        
        DispatchQueue.main.async {
            self.analyticsEvents.append(analyticsEvent)
        }
        
        dispatchQueue.async { [weak self] in
            self?.storeAnalyticsEvent(analyticsEvent)
            self?.uploadAnalyticsEventIfPossible(analyticsEvent)
        }
    }
    
    func trackScreenView(_ screenName: String, parameters: [String: Any] = [:]) {
        var params = parameters
        params["screen_name"] = screenName
        trackEvent("screen_view", parameters: params)
    }
    
    func trackUserAction(_ action: String, target: String? = nil, parameters: [String: Any] = [:]) {
        var params = parameters
        params["action"] = action
        if let target = target {
            params["target"] = target
        }
        trackEvent("user_action", parameters: params)
    }
    
    func trackPerformanceMetric(_ metric: PerformanceMetric) {
        guard isEnabled else { return }
        
        dispatchQueue.async { [weak self] in
            self?.storePerformanceMetric(metric)
            self?.updatePerformanceMetrics(metric)
        }
    }
    
    func generateDiagnosticReport() -> DiagnosticReport {
        let deviceInfo = collectDeviceInfo()
        let appInfo = collectAppInfo()
        let memoryInfo = collectMemoryInfo()
        let storageInfo = collectStorageInfo()
        
        return DiagnosticReport(
            timestamp: Date(),
            deviceInfo: deviceInfo,
            appInfo: appInfo,
            memoryInfo: memoryInfo,
            storageInfo: storageInfo,
            networkStatus: networkStatus,
            recentCrashes: Array(crashReports.prefix(10)),
            performanceMetrics: performanceMetrics
        )
    }
    
    func exportCrashReports() -> Data? {
        do {
            let reports = crashReports.map { report in
                CrashReportExport(
                    id: report.id,
                    timestamp: report.timestamp,
                    errorDescription: report.error.localizedDescription,
                    stackTrace: report.stackTrace,
                    deviceInfo: report.deviceInfo,
                    appInfo: report.appInfo
                )
            }
            
            return try JSONEncoder().encode(reports)
        } catch {
            logger.error("Failed to export crash reports: \(error.localizedDescription)")
            return nil
        }
    }
    
    func clearAllData() {
        dispatchQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Clear stored files
            try? self.fileManager.removeItem(at: self.crashReportsDirectory)
            try? self.fileManager.removeItem(at: self.analyticsDirectory)
            
            // Recreate directories
            try? self.fileManager.createDirectory(at: self.crashReportsDirectory, withIntermediateDirectories: true)
            try? self.fileManager.createDirectory(at: self.analyticsDirectory, withIntermediateDirectories: true)
            
            DispatchQueue.main.async {
                self.crashReports.removeAll()
                self.analyticsEvents.removeAll()
                self.performanceMetrics = nil
            }
            
            self.logger.info("All crash reporting data cleared")
        }
    }
    
    // MARK: - Private Methods
    
    private func setupCrashReporting() {
        // Set up uncaught exception handler
        NSSetUncaughtExceptionHandler { exception in
            CrashReportingManager.shared.handleUncaughtException(exception)
        }
        
        // Set up signal handler for crashes
        signal(SIGABRT) { signal in
            CrashReportingManager.shared.handleSignal(signal)
        }
        signal(SIGILL) { signal in
            CrashReportingManager.shared.handleSignal(signal)
        }
        signal(SIGSEGV) { signal in
            CrashReportingManager.shared.handleSignal(signal)
        }
        signal(SIGFPE) { signal in
            CrashReportingManager.shared.handleSignal(signal)
        }
        signal(SIGBUS) { signal in
            CrashReportingManager.shared.handleSignal(signal)
        }
        signal(SIGPIPE) { signal in
            CrashReportingManager.shared.handleSignal(signal)
        }
    }
    
    private func setupPerformanceMonitoring() {
        performanceTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            self?.collectPerformanceMetrics()
        }
    }
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isNetworkAvailable = path.status == .satisfied
                self?.networkStatus = NetworkStatus(from: path)
                
                if self?.isNetworkAvailable == true {
                    self?.uploadPendingReports()
                }
            }
        }
        
        networkMonitor.start(queue: dispatchQueue)
    }
    
    private func setupAppStateObservers() {
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryWarning()
        }
        
        appStateObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppTermination()
        }
    }
    
    private func handleCrash(_ error: Error, context: CrashContext?) {
        let crashReport = CrashReport(
            id: UUID().uuidString,
            timestamp: Date(),
            error: error,
            context: context,
            stackTrace: collectStackTrace(),
            deviceInfo: collectDeviceInfo(),
            appInfo: collectAppInfo(),
            userInfo: collectUserInfo(),
            memoryInfo: collectMemoryInfo()
        )
        
        storeCrashReport(crashReport)
        
        DispatchQueue.main.async {
            self.crashReports.append(crashReport)
        }
        
        uploadCrashReportIfPossible(crashReport)
        
        logger.critical("Crash reported: \(error.localizedDescription)")
    }
    
    private func handleUncaughtException(_ exception: NSException) {
        let error = NSError(
            domain: "UncaughtException",
            code: -1,
            userInfo: [
                NSLocalizedDescriptionKey: exception.reason ?? "Unknown exception",
                "exception_name": exception.name.rawValue,
                "call_stack": exception.callStackSymbols
            ]
        )
        
        let context = CrashContext(
            type: .exception,
            additionalInfo: [
                "exception_name": exception.name.rawValue,
                "reason": exception.reason ?? "Unknown"
            ]
        )
        
        handleCrash(error, context: context)
    }
    
    private func handleSignal(_ signal: Int32) {
        let error = NSError(
            domain: "Signal",
            code: Int(signal),
            userInfo: [
                NSLocalizedDescriptionKey: "Application received signal \(signal)",
                "signal_number": signal
            ]
        )
        
        let context = CrashContext(
            type: .signal,
            additionalInfo: ["signal_number": signal]
        )
        
        handleCrash(error, context: context)
    }
    
    private func handleMemoryWarning() {
        let memoryInfo = collectMemoryInfo()
        
        trackEvent("memory_warning", parameters: [
            "available_memory": memoryInfo.availableMemory,
            "used_memory": memoryInfo.usedMemory,
            "memory_pressure": memoryInfo.memoryPressure.rawValue
        ])
        
        logger.warning("Memory warning received")
    }
    
    private func handleAppTermination() {
        // Save any pending data
        saveCurrentState()
        logger.info("App termination handled")
    }
    
    private func collectStackTrace() -> [String] {
        return Thread.callStackSymbols
    }
    
    private func collectDeviceInfo() -> DeviceInfo {
        let device = UIDevice.current
        
        return DeviceInfo(
            model: device.model,
            systemName: device.systemName,
            systemVersion: device.systemVersion,
            identifierForVendor: device.identifierForVendor?.uuidString,
            batteryLevel: device.batteryLevel,
            batteryState: device.batteryState.rawValue,
            orientation: device.orientation.rawValue,
            userInterfaceIdiom: device.userInterfaceIdiom.rawValue
        )
    }
    
    private func collectAppInfo() -> AppInfo {
        let bundle = Bundle.main
        
        return AppInfo(
            bundleIdentifier: bundle.bundleIdentifier ?? "Unknown",
            version: bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Unknown",
            buildNumber: bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "Unknown",
            launchTime: Date(), // Approximate
            isDebugBuild: isDebugBuild()
        )
    }
    
    private func collectUserInfo() -> UserInfo {
        return UserInfo(
            userId: getCurrentUserId(),
            sessionId: getCurrentSessionId(),
            isFirstLaunch: isFirstLaunch(),
            appUsageTime: getAppUsageTime()
        )
    }
    
    private func collectMemoryInfo() -> MemoryInfo {
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
        let availableMemory = totalMemory - usedMemory
        
        return MemoryInfo(
            totalMemory: totalMemory,
            availableMemory: availableMemory,
            usedMemory: usedMemory,
            memoryPressure: getMemoryPressure()
        )
    }
    
    private func collectStorageInfo() -> StorageInfo {
        do {
            let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
            let resourceValues = try documentsPath.resourceValues(forKeys: [
                .volumeAvailableCapacityKey,
                .volumeTotalCapacityKey
            ])
            
            let availableCapacity = resourceValues.volumeAvailableCapacity ?? 0
            let totalCapacity = resourceValues.volumeTotalCapacity ?? 0
            let usedCapacity = totalCapacity - availableCapacity
            
            return StorageInfo(
                totalStorage: UInt64(totalCapacity),
                availableStorage: UInt64(availableCapacity),
                usedStorage: UInt64(usedCapacity)
            )
        } catch {
            logger.error("Failed to collect storage info: \(error.localizedDescription)")
            return StorageInfo(totalStorage: 0, availableStorage: 0, usedStorage: 0)
        }
    }
    
    private func collectPerformanceMetrics() {
        let memoryInfo = collectMemoryInfo()
        let cpuUsage = getCPUUsage()
        
        let metric = PerformanceMetric(
            timestamp: Date(),
            cpuUsage: cpuUsage,
            memoryUsage: Double(memoryInfo.usedMemory) / Double(memoryInfo.totalMemory),
            diskUsage: 0.0, // TODO: Implement disk usage calculation
            networkLatency: 0.0, // TODO: Implement network latency measurement
            frameRate: 60.0 // TODO: Implement actual frame rate measurement
        )
        
        trackPerformanceMetric(metric)
    }
    
    private func getCPUUsage() -> Double {
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
        
        return kerr == KERN_SUCCESS ? Double(info.user_time.seconds + info.system_time.seconds) : 0.0
    }
    
    private func getMemoryPressure() -> MemoryPressure {
        let memoryInfo = collectMemoryInfo()
        let usageRatio = Double(memoryInfo.usedMemory) / Double(memoryInfo.totalMemory)
        
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
    
    private func isDebugBuild() -> Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }
    
    private func getCurrentUserId() -> String {
        // Return anonymized user ID or device identifier
        return UIDevice.current.identifierForVendor?.uuidString ?? "unknown"
    }
    
    private func getCurrentSessionId() -> String {
        // Generate or retrieve current session ID
        if let sessionId = UserDefaults.standard.string(forKey: "current_session_id") {
            return sessionId
        } else {
            let newSessionId = UUID().uuidString
            UserDefaults.standard.set(newSessionId, forKey: "current_session_id")
            return newSessionId
        }
    }
    
    private func isFirstLaunch() -> Bool {
        return !UserDefaults.standard.bool(forKey: "has_launched_before")
    }
    
    private func getAppUsageTime() -> TimeInterval {
        // Return approximate app usage time
        return Date().timeIntervalSince(Date()) // Placeholder
    }
    
    private func storeCrashReport(_ report: CrashReport) {
        do {
            let data = try JSONEncoder().encode(report)
            let filename = "crash_\(report.id).json"
            let fileURL = crashReportsDirectory.appendingPathComponent(filename)
            try data.write(to: fileURL)
            
            logger.info("Crash report stored: \(filename)")
        } catch {
            logger.error("Failed to store crash report: \(error.localizedDescription)")
        }
    }
    
    private func storeErrorReport(_ report: ErrorReport) {
        do {
            let data = try JSONEncoder().encode(report)
            let filename = "error_\(report.id).json"
            let fileURL = crashReportsDirectory.appendingPathComponent(filename)
            try data.write(to: fileURL)
            
            logger.info("Error report stored: \(filename)")
        } catch {
            logger.error("Failed to store error report: \(error.localizedDescription)")
        }
    }
    
    private func storeAnalyticsEvent(_ event: AnalyticsEvent) {
        do {
            let data = try JSONEncoder().encode(event)
            let filename = "event_\(event.id).json"
            let fileURL = analyticsDirectory.appendingPathComponent(filename)
            try data.write(to: fileURL)
        } catch {
            logger.error("Failed to store analytics event: \(error.localizedDescription)")
        }
    }
    
    private func storePerformanceMetric(_ metric: PerformanceMetric) {
        // Store performance metrics in a rolling file
        let filename = "performance_metrics.json"
        let fileURL = analyticsDirectory.appendingPathComponent(filename)
        
        var metrics: [PerformanceMetric] = []
        
        if let data = try? Data(contentsOf: fileURL),
           let existingMetrics = try? JSONDecoder().decode([PerformanceMetric].self, from: data) {
            metrics = existingMetrics
        }
        
        metrics.append(metric)
        
        // Keep only recent metrics
        if metrics.count > maxLogEntries {
            metrics = Array(metrics.suffix(maxLogEntries))
        }
        
        do {
            let data = try JSONEncoder().encode(metrics)
            try data.write(to: fileURL)
        } catch {
            logger.error("Failed to store performance metric: \(error.localizedDescription)")
        }
    }
    
    private func updatePerformanceMetrics(_ metric: PerformanceMetric) {
        DispatchQueue.main.async {
            if self.performanceMetrics == nil {
                self.performanceMetrics = PerformanceMetrics(
                    averageCPUUsage: metric.cpuUsage,
                    averageMemoryUsage: metric.memoryUsage,
                    averageDiskUsage: metric.diskUsage,
                    averageNetworkLatency: metric.networkLatency,
                    averageFrameRate: metric.frameRate,
                    sampleCount: 1
                )
            } else {
                let current = self.performanceMetrics!
                let newCount = current.sampleCount + 1
                
                self.performanceMetrics = PerformanceMetrics(
                    averageCPUUsage: (current.averageCPUUsage * Double(current.sampleCount) + metric.cpuUsage) / Double(newCount),
                    averageMemoryUsage: (current.averageMemoryUsage * Double(current.sampleCount) + metric.memoryUsage) / Double(newCount),
                    averageDiskUsage: (current.averageDiskUsage * Double(current.sampleCount) + metric.diskUsage) / Double(newCount),
                    averageNetworkLatency: (current.averageNetworkLatency * Double(current.sampleCount) + metric.networkLatency) / Double(newCount),
                    averageFrameRate: (current.averageFrameRate * Double(current.sampleCount) + metric.frameRate) / Double(newCount),
                    sampleCount: newCount
                )
            }
        }
    }
    
    private func loadStoredData() {
        dispatchQueue.async { [weak self] in
            self?.loadCrashReports()
            self?.loadAnalyticsEvents()
        }
    }
    
    private func loadCrashReports() {
        do {
            let files = try fileManager.contentsOfDirectory(at: crashReportsDirectory, includingPropertiesForKeys: nil)
            let crashFiles = files.filter { $0.pathExtension == "json" && $0.lastPathComponent.hasPrefix("crash_") }
            
            var reports: [CrashReport] = []
            
            for file in crashFiles {
                if let data = try? Data(contentsOf: file),
                   let report = try? JSONDecoder().decode(CrashReport.self, from: data) {
                    reports.append(report)
                }
            }
            
            DispatchQueue.main.async {
                self.crashReports = reports.sorted { $0.timestamp > $1.timestamp }
            }
            
            logger.info("Loaded \(reports.count) crash reports")
        } catch {
            logger.error("Failed to load crash reports: \(error.localizedDescription)")
        }
    }
    
    private func loadAnalyticsEvents() {
        do {
            let files = try fileManager.contentsOfDirectory(at: analyticsDirectory, includingPropertiesForKeys: nil)
            let eventFiles = files.filter { $0.pathExtension == "json" && $0.lastPathComponent.hasPrefix("event_") }
            
            var events: [AnalyticsEvent] = []
            
            for file in eventFiles {
                if let data = try? Data(contentsOf: file),
                   let event = try? JSONDecoder().decode(AnalyticsEvent.self, from: data) {
                    events.append(event)
                }
            }
            
            DispatchQueue.main.async {
                self.analyticsEvents = events.sorted { $0.timestamp > $1.timestamp }
            }
            
            logger.info("Loaded \(events.count) analytics events")
        } catch {
            logger.error("Failed to load analytics events: \(error.localizedDescription)")
        }
    }
    
    private func cleanupOldReports() {
        dispatchQueue.async { [weak self] in
            guard let self = self else { return }
            
            let cutoffDate = Calendar.current.date(byAdding: .day, value: -self.reportRetentionDays, to: Date()) ?? Date()
            
            // Clean up crash reports
            do {
                let files = try self.fileManager.contentsOfDirectory(at: self.crashReportsDirectory, includingPropertiesForKeys: [.creationDateKey])
                
                for file in files {
                    if let resourceValues = try? file.resourceValues(forKeys: [.creationDateKey]),
                       let creationDate = resourceValues.creationDate,
                       creationDate < cutoffDate {
                        try? self.fileManager.removeItem(at: file)
                    }
                }
            } catch {
                self.logger.error("Failed to cleanup old crash reports: \(error.localizedDescription)")
            }
            
            // Clean up analytics events
            do {
                let files = try self.fileManager.contentsOfDirectory(at: self.analyticsDirectory, includingPropertiesForKeys: [.creationDateKey])
                
                for file in files {
                    if let resourceValues = try? file.resourceValues(forKeys: [.creationDateKey]),
                       let creationDate = resourceValues.creationDate,
                       creationDate < cutoffDate {
                        try? self.fileManager.removeItem(at: file)
                    }
                }
            } catch {
                self.logger.error("Failed to cleanup old analytics events: \(error.localizedDescription)")
            }
        }
    }
    
    private func uploadCrashReportIfPossible(_ report: CrashReport) {
        guard isNetworkAvailable else {
            pendingUploads.append(report.id)
            return
        }
        
        uploadCrashReport(report)
    }
    
    private func uploadErrorReportIfPossible(_ report: ErrorReport) {
        guard isNetworkAvailable else {
            pendingUploads.append(report.id)
            return
        }
        
        uploadErrorReport(report)
    }
    
    private func uploadAnalyticsEventIfPossible(_ event: AnalyticsEvent) {
        guard isNetworkAvailable else { return }
        
        uploadAnalyticsEvent(event)
    }
    
    private func uploadCrashReport(_ report: CrashReport) {
        // TODO: Implement actual upload to crash reporting service
        logger.info("Uploading crash report: \(report.id)")
    }
    
    private func uploadErrorReport(_ report: ErrorReport) {
        // TODO: Implement actual upload to error reporting service
        logger.info("Uploading error report: \(report.id)")
    }
    
    private func uploadAnalyticsEvent(_ event: AnalyticsEvent) {
        // TODO: Implement actual upload to analytics service
        logger.info("Uploading analytics event: \(event.id)")
    }
    
    private func uploadPendingReports() {
        guard !pendingUploads.isEmpty else { return }
        
        dispatchQueue.async { [weak self] in
            guard let self = self else { return }
            
            for reportId in self.pendingUploads {
                // Find and upload the report
                if let crashReport = self.crashReports.first(where: { $0.id == reportId }) {
                    self.uploadCrashReport(crashReport)
                }
            }
            
            self.pendingUploads.removeAll()
        }
    }
    
    private func saveCurrentState() {
        // Save any important state before app termination
        UserDefaults.standard.synchronize()
    }
}

// MARK: - Supporting Types

struct CrashReport: Codable {
    let id: String
    let timestamp: Date
    let error: CodableError
    let context: CrashContext?
    let stackTrace: [String]
    let deviceInfo: DeviceInfo
    let appInfo: AppInfo
    let userInfo: UserInfo
    let memoryInfo: MemoryInfo
}

struct ErrorReport: Codable {
    let id: String
    let timestamp: Date
    let error: CodableError
    let severity: ErrorSeverity
    let context: [String: String] // Simplified for Codable
    let deviceInfo: DeviceInfo
    let appInfo: AppInfo
    let userInfo: UserInfo
}

struct CodableError: Codable {
    let domain: String
    let code: Int
    let localizedDescription: String
    let userInfo: [String: String]
    
    init(from error: Error) {
        if let nsError = error as NSError? {
            self.domain = nsError.domain
            self.code = nsError.code
            self.localizedDescription = nsError.localizedDescription
            self.userInfo = nsError.userInfo.compactMapValues { "\($0)" }
        } else {
            self.domain = "Unknown"
            self.code = -1
            self.localizedDescription = error.localizedDescription
            self.userInfo = [:]
        }
    }
}

struct CrashContext: Codable {
    let type: CrashType
    let additionalInfo: [String: String]
}

enum CrashType: String, Codable {
    case exception = "exception"
    case signal = "signal"
    case error = "error"
    case memoryWarning = "memory_warning"
}

enum ErrorSeverity: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

struct DeviceInfo: Codable {
    let model: String
    let systemName: String
    let systemVersion: String
    let identifierForVendor: String?
    let batteryLevel: Float
    let batteryState: Int
    let orientation: Int
    let userInterfaceIdiom: Int
}

struct AppInfo: Codable {
    let bundleIdentifier: String
    let version: String
    let buildNumber: String
    let launchTime: Date
    let isDebugBuild: Bool
}

struct UserInfo: Codable {
    let userId: String
    let sessionId: String
    let isFirstLaunch: Bool
    let appUsageTime: TimeInterval
}

struct MemoryInfo: Codable {
    let totalMemory: UInt64
    let availableMemory: UInt64
    let usedMemory: UInt64
    let memoryPressure: MemoryPressure
}

enum MemoryPressure: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

struct StorageInfo: Codable {
    let totalStorage: UInt64
    let availableStorage: UInt64
    let usedStorage: UInt64
}

struct AnalyticsEvent: Codable {
    let id: String
    let name: String
    let parameters: [String: String] // Simplified for Codable
    let timestamp: Date
    let sessionId: String
    let userId: String
}

struct PerformanceMetric: Codable {
    let timestamp: Date
    let cpuUsage: Double
    let memoryUsage: Double
    let diskUsage: Double
    let networkLatency: Double
    let frameRate: Double
}

struct PerformanceMetrics {
    let averageCPUUsage: Double
    let averageMemoryUsage: Double
    let averageDiskUsage: Double
    let averageNetworkLatency: Double
    let averageFrameRate: Double
    let sampleCount: Int
}

struct DiagnosticReport {
    let timestamp: Date
    let deviceInfo: DeviceInfo
    let appInfo: AppInfo
    let memoryInfo: MemoryInfo
    let storageInfo: StorageInfo
    let networkStatus: NetworkStatus
    let recentCrashes: [CrashReport]
    let performanceMetrics: PerformanceMetrics?
}

struct CrashReportExport: Codable {
    let id: String
    let timestamp: Date
    let errorDescription: String
    let stackTrace: [String]
    let deviceInfo: DeviceInfo
    let appInfo: AppInfo
}

enum NetworkStatus {
    case unknown
    case unavailable
    case wifi
    case cellular
    case ethernet
    
    init(from path: NWPath) {
        if path.status != .satisfied {
            self = .unavailable
        } else if path.usesInterfaceType(.wifi) {
            self = .wifi
        } else if path.usesInterfaceType(.cellular) {
            self = .cellular
        } else if path.usesInterfaceType(.wiredEthernet) {
            self = .ethernet
        } else {
            self = .unknown
        }
    }
}