//
//  InflamAIApp.swift
//  InflamAI
//
//  Production-grade iOS app for Ankylosing Spondylitis management
//  Fortune 100 quality: Privacy-first, accessible, clinically accurate
//

import SwiftUI
import LocalAuthentication
import CoreData
import BackgroundTasks
import UserNotifications
import WidgetKit

@main
struct InflamAIApp: App {
    // MARK: - State

    private let persistenceController = InflamAIPersistenceController.shared
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @AppStorage("biometricLockEnabled") private var biometricLockEnabled = false
    @State private var isUnlocked = false
    @State private var isFirstLaunch = true
    @State private var showOnboarding: Bool
    @State private var isSeedingData = false  // Loading state for demo data seeding
    @State private var seedingComplete = false  // Track if seeding check completed
    @Environment(\.scenePhase) private var scenePhase

    init() {
        _showOnboarding = State(initialValue: !UserDefaults.standard.bool(forKey: "hasCompletedOnboarding"))

        // Register background tasks
        registerBackgroundTasks()
    }

    // MARK: - Scene

    var body: some Scene {
        WindowGroup {
            ZStack {
                if isSeedingData {
                    // Show loading screen while seeding demo data
                    DemoDataLoadingView()
                } else if !biometricLockEnabled || isUnlocked {
                    MainTabView()
                        .environment(\.managedObjectContext, persistenceController.container.viewContext)
                        .onAppear {
                            // Only run checkFirstLaunch once
                            if !seedingComplete {
                                seedingComplete = true
                                performFirstLaunchCheck()
                            }
                            requestPermissions()
                            synchronizeOnboardingState()
                            scheduleWeatherMonitoring()

                            // Initialize flare notifications
                            initializeFlareNotifications()

                            // Sync data to widgets on app launch
                            syncWidgetData()
                        }
                } else {
                    BiometricLockScreen {
                        isUnlocked = true
                        synchronizeOnboardingState()
                    }
                }
            }
            .onAppear(perform: synchronizeOnboardingState)
            .onChange(of: hasCompletedOnboarding) { _ in
                synchronizeOnboardingState()
            }
            .onChange(of: scenePhase) { newPhase in
                if newPhase == .active {
                    synchronizeOnboardingState()
                    // Refresh widget data when app becomes active
                    syncWidgetData()
                }
            }
            .fullScreenCover(isPresented: $showOnboarding, onDismiss: synchronizeOnboardingState) {
                // Premium redesigned onboarding with animated mascot
                OnboardingRedesignFlow()
                    .interactiveDismissDisabled()
            }
            .onReceive(NotificationCenter.default.publisher(for: .onboardingCompleted)) { _ in
                // Handle onboarding completion notification
                hasCompletedOnboarding = true
                synchronizeOnboardingState()
            }
        }
    }

    // MARK: - First Launch

    /// Synchronous wrapper that kicks off async first launch check
    private func performFirstLaunchCheck() {
        let hasLaunchedBefore = UserDefaults.standard.bool(forKey: "hasLaunchedBefore")
        let hasDemoData = UserDefaults.standard.hasDemoDataBeenSeeded

        // Only seed if truly first launch AND not already seeded
        if !hasLaunchedBefore && !hasDemoData {
            isFirstLaunch = true
            isSeedingData = true  // Show loading screen

            Task { @MainActor in
                await seedDemoDataAsync()
            }
        }
    }

    /// Async function to seed demo data with proper state management
    @MainActor
    private func seedDemoDataAsync() async {
        print("🚀 Starting demo data seeding process...")

        do {
            let context = persistenceController.container.viewContext
            try await DemoDataSeeder.shared.seedDemoData(context: context)

            // Mark as completed AFTER successful seeding
            UserDefaults.standard.set(true, forKey: "hasLaunchedBefore")
            UserDefaults.standard.hasDemoDataBeenSeeded = true
            print("✅ Demo data seeded successfully for Anna")

        } catch {
            print("❌ Failed to seed demo data: \(error)")

            // Still mark as launched to prevent infinite retry loop
            UserDefaults.standard.set(true, forKey: "hasLaunchedBefore")

            // Fallback: Create default user profile
            do {
                let _ = try persistenceController.fetchUserProfile()
                print("✅ User profile created successfully (fallback)")
            } catch {
                print("❌ CRITICAL: Failed to create user profile: \(error)")
            }
        }

        // Hide loading screen - UI will now show MainTabView
        isSeedingData = false
    }

    // MARK: - Permissions

    private func requestPermissions() {
        // Run permission requests in background without blocking UI
        Task {
            // CRITICAL FIX: Check if already authorized before requesting
            // This prevents race conditions with OnboardingFlow and HealthKitManager

            // Check existing authorization first (from previous session or onboarding)
            let hasExistingAuth = HealthKitService.shared.isAuthorized ||
                                  HealthKitService.shared.checkExistingAuthorization()

            if hasExistingAuth {
                print("✅ [InflamAIApp] HealthKit already authorized, fetching data...")
                await fetchAndStoreHealthKitData()
            } else {
                // Only request if not already authorized
                // Use ensureAuthorization for retry logic
                print("🔄 [InflamAIApp] HealthKit not authorized, requesting...")
                let authorized = await HealthKitService.shared.ensureAuthorization(maxRetries: 2)

                if authorized {
                    print("✅ [InflamAIApp] HealthKit authorization completed")
                    await fetchAndStoreHealthKitData()
                } else {
                    print("⚠️ [InflamAIApp] HealthKit authorization failed - user must enable in Settings")
                    // App continues to work without HealthKit data
                }
            }

            // Request location/weather authorization (non-blocking)
            await OpenMeteoService.shared.requestAuthorization()
            print("✅ OpenMeteo location authorization completed")

            // Request notification permissions
            do {
                let granted = try await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound])
                print(granted ? "✅ Notification permissions granted" : "⚠️ Notification permissions denied")
            } catch {
                print("⚠️ Notification permission request failed: \(error.localizedDescription)")
            }
        }
    }

    /// CRITICAL FIX: Fetch HealthKit data and persist to Core Data
    private func fetchAndStoreHealthKitData() async {
        guard HealthKitService.shared.isAuthorized else {
            print("⚠️ HealthKit not authorized, skipping data fetch")
            return
        }

        do {
            let biometrics = try await HealthKitService.shared.fetchAllBiometrics(for: Date())
            print("✅ Fetched biometrics: HRV=\(biometrics.hrvValue)ms, HR=\(biometrics.restingHeartRate)bpm, Steps=\(biometrics.stepCount)")

            // Store to Core Data via ContextSnapshot on next symptom log
            // The biometrics are now available for DailyCheckInViewModel to use
            UserDefaults.standard.set(true, forKey: "hasValidHealthKitData")
            UserDefaults.standard.set(Date(), forKey: "lastHealthKitFetch")
        } catch {
            print("⚠️ Failed to fetch HealthKit data: \(error.localizedDescription)")
        }
    }

    // MARK: - Timeout Helper

    /// Executes an async operation with a timeout
    private func withTimeout(seconds: TimeInterval, operation: @escaping () async -> Void) async {
        await withTaskGroup(of: Void.self) { group in
            // Add the actual operation
            group.addTask {
                await operation()
            }

            // Add timeout task
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            }

            // Wait for first to complete (either operation or timeout)
            await group.next()

            // Cancel any remaining tasks
            group.cancelAll()
        }
    }

    // MARK: - Onboarding

    private func synchronizeOnboardingState() {
        if biometricLockEnabled && !isUnlocked {
            showOnboarding = false
        } else {
            showOnboarding = !hasCompletedOnboarding
        }
    }

    // MARK: - Background Tasks

    private func registerBackgroundTasks() {
        // Register weather monitoring task
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.inflamai.weatherMonitoring",
            using: nil
        ) { task in
            guard let processingTask = task as? BGProcessingTask else {
                print("❌ weatherMonitoring: Expected BGProcessingTask, got \(type(of: task))")
                task.setTaskCompleted(success: false)
                return
            }
            self.handleWeatherMonitoring(task: processingTask)
        }

        // Register ML prediction refresh task
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.inflamai.predictionRefresh",
            using: nil
        ) { task in
            guard let refreshTask = task as? BGAppRefreshTask else {
                print("❌ predictionRefresh: Expected BGAppRefreshTask, got \(type(of: task))")
                task.setTaskCompleted(success: false)
                return
            }
            self.handlePredictionRefresh(task: refreshTask)
        }

        // Schedule initial prediction refresh
        schedulePredictionRefresh()
    }

    private func handleWeatherMonitoring(task: BGProcessingTask) {
        task.expirationHandler = {
            task.setTaskCompleted(success: false)
        }

        Task {
            // Schedule weather monitoring using WeatherKit
            await WeatherNotificationService.shared.scheduleWeatherMonitoring()
            task.setTaskCompleted(success: true)

            // Schedule next run (12 hours)
            scheduleWeatherMonitoring()
        }
    }

    private func scheduleWeatherMonitoring() {
        let request = BGProcessingTaskRequest(identifier: "com.inflamai.weatherMonitoring")
        request.earliestBeginDate = Calendar.current.date(byAdding: .hour, value: 12, to: Date())
        request.requiresNetworkConnectivity = true

        do {
            try BGTaskScheduler.shared.submit(request)
            print("✅ Scheduled weather monitoring background task")
        } catch {
            print("❌ Failed to schedule weather monitoring: \(error)")
        }
    }

    // MARK: - ML Prediction Background Refresh

    private func handlePredictionRefresh(task: BGAppRefreshTask) {
        task.expirationHandler = {
            task.setTaskCompleted(success: false)
        }

        Task { @MainActor in
            // Refresh neural engine prediction
            await UnifiedNeuralEngine.shared.refresh()

            // Check if we should send a flare warning notification
            await FlareNotificationService.shared.checkAndNotify()

            // Sync updated prediction to widgets
            await SharedDataSyncService.shared.syncFlareData()

            task.setTaskCompleted(success: true)

            // Schedule next refresh (every 4 hours)
            schedulePredictionRefresh()

            print("✅ Background ML prediction refresh completed")
        }
    }

    private func schedulePredictionRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: "com.inflamai.predictionRefresh")
        request.earliestBeginDate = Calendar.current.date(byAdding: .hour, value: 4, to: Date())

        do {
            try BGTaskScheduler.shared.submit(request)
            print("✅ Scheduled ML prediction refresh (4 hours)")
        } catch {
            print("❌ Failed to schedule prediction refresh: \(error)")
        }
    }

    // MARK: - Flare Notifications

    private func initializeFlareNotifications() {
        // Set notification delegate to handle action responses
        UNUserNotificationCenter.current().delegate = FlareNotificationDelegate.shared

        Task {
            // Register notification categories
            FlareNotificationService.shared.registerNotificationCategories()

            // Schedule daily check-in reminder
            FlareNotificationService.shared.scheduleDailyCheck()

            // Refresh neural engine prediction
            await UnifiedNeuralEngine.shared.refresh()

            // Check if we should send a flare warning
            await FlareNotificationService.shared.checkAndNotify()

            print("✅ Flare notification service initialized")
        }
    }

    // MARK: - Widget Sync

    /// Sync data to widgets via App Group shared UserDefaults
    private func syncWidgetData() {
        Task {
            await performWidgetSync()
        }
    }

    @MainActor
    private func performWidgetSync() async {
        let context = persistenceController.container.viewContext
        guard let defaults = UserDefaults(suiteName: "group.com.inflamai.InflamAI") else {
            print("⚠️ Could not access App Group UserDefaults")
            return
        }

        do {
            // Sync BASDAI data
            let basdaiRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            basdaiRequest.predicate = NSPredicate(format: "basdaiScore > 0")
            basdaiRequest.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
            basdaiRequest.fetchLimit = 30

            let logs = try context.fetch(basdaiRequest)

            if let latestLog = logs.first {
                let score = latestLog.basdaiScore
                let category: String
                switch score {
                case 0..<2: category = "Remission"
                case 2..<4: category = "Low"
                case 4..<6: category = "Moderate"
                case 6..<8: category = "High"
                default: category = "Very High"
                }

                // Calculate trend
                var trend = "stable"
                if logs.count >= 6 {
                    let recent = Array(logs.prefix(3))
                    let previous = Array(logs.dropFirst(3).prefix(3))
                    let recentAvg = recent.reduce(0.0) { $0 + $1.basdaiScore } / Double(recent.count)
                    let previousAvg = previous.reduce(0.0) { $0 + $1.basdaiScore } / Double(previous.count)
                    let diff = recentAvg - previousAvg
                    if diff < -0.5 { trend = "improving" }
                    else if diff > 0.5 { trend = "worsening" }
                }

                defaults.set(score, forKey: "widget.basdai.score")
                defaults.set(category, forKey: "widget.basdai.category")
                defaults.set(trend, forKey: "widget.basdai.trend")
                defaults.set(Date(), forKey: "widget.basdai.updated")
            }

            // Sync streak data
            let streakRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            streakRequest.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
            let allLogs = try context.fetch(streakRequest)

            let calendar = Calendar.current
            var streak = 0
            var currentDate = calendar.startOfDay(for: Date())

            // Check if logged today
            let todayLogs = allLogs.filter { log in
                guard let timestamp = log.timestamp else { return false }
                return calendar.isDate(timestamp, inSameDayAs: currentDate)
            }

            if todayLogs.isEmpty {
                currentDate = calendar.date(byAdding: .day, value: -1, to: currentDate) ?? currentDate
            }

            for log in allLogs {
                guard let timestamp = log.timestamp else { continue }
                let logDay = calendar.startOfDay(for: timestamp)
                if calendar.isDate(logDay, inSameDayAs: currentDate) {
                    streak += 1
                    currentDate = calendar.date(byAdding: .day, value: -1, to: currentDate) ?? currentDate
                } else if logDay < currentDate {
                    break
                }
            }

            defaults.set(streak, forKey: "widget.streak.days")
            defaults.set(Date(), forKey: "widget.streak.updated")

            // Sync flare risk (simplified)
            let recentRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            let threeDaysAgo = calendar.date(byAdding: .day, value: -3, to: Date()) ?? Date()
            recentRequest.predicate = NSPredicate(format: "timestamp >= %@", threeDaysAgo as NSDate)
            let recentLogs = try context.fetch(recentRequest)

            var riskPercentage = 25
            if !recentLogs.isEmpty {
                let avgBASDAI = recentLogs.reduce(0.0) { $0 + $1.basdaiScore } / Double(recentLogs.count)
                let avgFatigue = recentLogs.reduce(0.0) { $0 + Double($1.fatigueLevel) } / Double(recentLogs.count)
                riskPercentage = min(100, Int((avgBASDAI * 8) + (avgFatigue * 2)))
            }

            let riskLevel: String
            switch riskPercentage {
            case 0..<25: riskLevel = "low"
            case 25..<50: riskLevel = "moderate"
            case 50..<75: riskLevel = "high"
            default: riskLevel = "veryHigh"
            }

            defaults.set(riskPercentage, forKey: "widget.flareRisk.percentage")
            defaults.set(riskLevel, forKey: "widget.flareRisk.level")
            defaults.set(Date(), forKey: "widget.flareRisk.updated")

            // Sync medications
            let medRequest: NSFetchRequest<Medication> = Medication.fetchRequest()
            medRequest.predicate = NSPredicate(format: "isActive == YES")
            let medications = try context.fetch(medRequest)

            var medReminders: [[String: Any]] = []
            for med in medications {
                guard let name = med.name else { continue }
                let nextDose = calendar.date(byAdding: .hour, value: 1, to: Date()) ?? Date()
                medReminders.append([
                    "id": (med.id ?? UUID()).uuidString,
                    "name": name,
                    "dosage": med.dosage ?? "",
                    "nextDoseTime": nextDose.timeIntervalSince1970,
                    "frequency": med.frequency ?? "Daily"
                ])
            }

            if let medData = try? JSONSerialization.data(withJSONObject: medReminders) {
                defaults.set(medData, forKey: "widget.medications.next")
            }
            defaults.set(Date(), forKey: "widget.medications.updated")

            // Trigger widget reload
            WidgetCenter.shared.reloadAllTimelines()
            print("✅ Widget data synced successfully")

        } catch {
            print("❌ Failed to sync widget data: \(error)")
        }
    }
}

// MARK: - Main Tab View

struct MainTabView: View {
    @State private var selectedTab = 0
    @State private var previousTab = 0
    @Environment(\.managedObjectContext) private var context

    var body: some View {
        TabView(selection: $selectedTab) {
            // Tab 0: Home - Dashboard with QuickFlare
            NavigationView {
                HomeView(context: context)
            }
            .navigationViewStyle(.stack)
            .tabItem {
                Label("Home", systemImage: "house.fill")
            }
            .tag(0)

            // Tab 1: Track - Body Map + Journal
            TrackHubView()
                .environment(\.managedObjectContext, context)
                .tabItem {
                    Label("Track", systemImage: "figure.stand")
                }
                .tag(1)

            // Tab 2: Analytics - Trends, ML, Triggers, Meds, Flares
            AnalyticsHubView()
                .environment(\.managedObjectContext, context)
                .tabItem {
                    Label("Analytics", systemImage: "chart.xyaxis.line")
                }
                .tag(2)

            // Tab 3: Wellness - Exercise, Meditation, Routines, Library
            WellnessHubView()
                .environment(\.managedObjectContext, context)
                .tabItem {
                    Label("Wellness", systemImage: "heart.fill")
                }
                .tag(3)

            // Tab 4: Settings - App settings + Export
            NavigationView {
                SettingsView()
            }
            .navigationViewStyle(.stack)
            .tabItem {
                Label("Settings", systemImage: "gearshape.fill")
            }
            .tag(4)
        }
        .tint(Colors.Primary.p500)
        .onChange(of: selectedTab) { newTab in
            // Haptic feedback on tab change
            if newTab != previousTab {
                HapticFeedback.selection()
                previousTab = newTab
            }
        }
    }
}

// MARK: - More View

struct MoreView: View {
    let context: NSManagedObjectContext

    var body: some View {
        List {
            // Library Section - moved from separate tab to avoid iOS auto "More" menu
            Section {
                NavigationLink(destination: LibraryView()) {
                    HStack {
                        Image(systemName: "books.vertical.fill")
                            .foregroundColor(.indigo)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Library")
                                .font(.body)
                            Text("Educational content & resources")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            } header: {
                Text("Learn")
            }

            Section {
                NavigationLink(destination: AssessmentsView().environment(\.managedObjectContext, context)) {
                    HStack {
                        Image(systemName: "list.clipboard.fill")
                            .foregroundColor(.purple)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Assessments")
                                .font(.body)
                            Text("Complete questionnaires & surveys")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                NavigationLink(destination: MeditationHomeView()) {
                    HStack {
                        Image(systemName: "leaf.fill")
                            .foregroundColor(.purple)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Meditation")
                                .font(.body)
                            Text("Guided sessions & progress tracking")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                NavigationLink(destination: BASSDAIView().environment(\.managedObjectContext, context)) {
                    HStack {
                        Image(systemName: "chart.bar.doc.horizontal")
                            .foregroundColor(.blue)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("BASDAI Assessment")
                                .font(.body)
                            Text("Track disease activity")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                NavigationLink(destination: TrendsView(context: context)) {
                    HStack {
                        Image(systemName: "chart.xyaxis.line")
                            .foregroundColor(.green)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Trends")
                                .font(.body)
                            Text("Visualize your health data")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                NavigationLink(destination: FlareTimelineView(context: context)) {
                    HStack {
                        Image(systemName: "flame.fill")
                            .foregroundColor(.orange)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Flare Timeline")
                                .font(.body)
                            Text("Track flare events")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                NavigationLink(destination: AIInsightsView(context: context)) {
                    HStack {
                        Image(systemName: "brain.head.profile")
                            .foregroundColor(.purple)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Pattern Insights")
                                .font(.body)
                            Text("AI predictions & trigger analysis")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                NavigationLink(destination: TriggerInsightsView()) {
                    HStack {
                        Image(systemName: "waveform.path.ecg.rectangle")
                            .foregroundColor(.teal)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Trigger Insights")
                                .font(.body)
                            Text("Analyze your personalized flare triggers")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            } header: {
                Text("Health Tracking")
            }

            Section {
                NavigationLink(destination: ExportDataView().environment(\.managedObjectContext, context)) {
                    HStack {
                        Image(systemName: "square.and.arrow.up")
                            .foregroundColor(.blue)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Export Data")
                                .font(.body)
                            Text("PDF, CSV, JSON formats")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                NavigationLink(destination: SettingsView()) {
                    HStack {
                        Image(systemName: "gearshape")
                            .foregroundColor(.gray)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Settings")
                                .font(.body)
                            Text("App preferences")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            } header: {
                Text("App")
            }
        }
        .navigationTitle("More")
    }
}

// MARK: - Biometric Lock Screen

struct BiometricLockScreen: View {
    let onUnlock: () -> Void

    @State private var showingError = false
    @State private var errorMessage = ""

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            Image(systemName: "lock.shield.fill")
                .font(.system(size: 80))
                .foregroundColor(.blue)

            VStack(spacing: 8) {
                Text("InflamAI is Locked")
                    .font(.title)
                    .fontWeight(.bold)

                Text("Your health data is protected")
                    .font(.body)
                    .foregroundColor(.secondary)
            }

            Button {
                authenticate()
            } label: {
                HStack {
                    Image(systemName: "faceid")
                    Text("Unlock with Face ID")
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .cornerRadius(12)
            }
            .padding(.horizontal, 40)

            Spacer()
        }
        .padding()
        .onAppear {
            // Auto-authenticate on appear
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                authenticate()
            }
        }
        .alert("Authentication Failed", isPresented: $showingError) {
            Button("Try Again") {
                authenticate()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(errorMessage)
        }
    }

    private func authenticate() {
        let context = LAContext()
        var error: NSError?

        // Check biometric availability
        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            errorMessage = "Biometric authentication not available"
            showingError = true
            return
        }

        // Perform biometric authentication
        context.evaluatePolicy(
            .deviceOwnerAuthenticationWithBiometrics,
            localizedReason: "Unlock InflamAI to access your health data"
        ) { success, authError in
            DispatchQueue.main.async {
                if success {
                    // Success haptic
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                    onUnlock()
                } else {
                    errorMessage = authError?.localizedDescription ?? "Authentication failed"
                    showingError = true
                }
            }
        }
    }
}

// MARK: - Settings View
// Note: Full SettingsView is defined in SettingsView.swift

// MARK: - Demo Data Loading View

struct DemoDataLoadingView: View {
    @State private var progress: Double = 0
    @State private var statusMessage = "Preparing Anna's health journey..."

    private let timer = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            // App icon placeholder
            Image(systemName: "waveform.path.ecg.rectangle")
                .font(.system(size: 80))
                .foregroundColor(.blue)
                .symbolEffect(.pulse)

            VStack(spacing: 12) {
                Text("Setting Up InflamAI")
                    .font(.title)
                    .fontWeight(.bold)

                Text("Creating 200 days of demo data...")
                    .font(.body)
                    .foregroundColor(.secondary)
            }

            VStack(spacing: 8) {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .frame(width: 250)
                    .tint(.blue)

                Text(statusMessage)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text("This only happens once")
                .font(.footnote)
                .foregroundColor(.secondary)
                .padding(.bottom, 40)
        }
        .padding()
        .onReceive(timer) { _ in
            // Simulate progress
            if progress < 0.95 {
                progress += 0.02
            }

            // Update status message based on progress
            switch progress {
            case 0..<0.2:
                statusMessage = "Creating Anna's profile..."
            case 0.2..<0.4:
                statusMessage = "Adding medications..."
            case 0.4..<0.6:
                statusMessage = "Generating symptom logs..."
            case 0.6..<0.8:
                statusMessage = "Adding exercise sessions..."
            case 0.8..<0.95:
                statusMessage = "Creating journal entries..."
            default:
                statusMessage = "Finalizing..."
            }
        }
    }
}
