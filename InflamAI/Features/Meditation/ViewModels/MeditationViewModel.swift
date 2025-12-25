//
//  MeditationViewModel.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import Foundation
import Combine
import CoreData
import AVFoundation
import HealthKit

@MainActor
class MeditationViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var availableSessions: [MeditationSessionModel] = []
    @Published var currentSession: MeditationSessionModel?
    @Published var recentSessions: [MeditationSession] = []
    @Published var streak: MeditationStreak?

    @Published var isPlaying = false
    @Published var isPaused = false
    @Published var currentTime: TimeInterval = 0
    @Published var progress: Double = 0

    @Published var stressLevelBefore: Int?
    @Published var painLevelBefore: Int?
    @Published var moodBefore: Int?

    @Published var isLoading = false
    @Published var errorMessage: String?

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private var persistenceHelper: MeditationPersistenceHelper
    private var cancellables = Set<AnyCancellable>()
    private var timer: Timer?

    // MARK: - Initialization

    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
        self.persistenceHelper = MeditationPersistenceHelper(
            context: persistenceController.container.viewContext
        )

        loadData()
    }

    // MARK: - Data Loading

    func loadData() {
        isLoading = true
        defer { isLoading = false }

        // Load available sessions (default AS-specific sessions)
        availableSessions = MeditationSessionModel.asSpecificSessions

        // Load recent sessions from Core Data
        do {
            recentSessions = try persistenceHelper.fetchRecentSessions(days: 30)
            streak = try persistenceHelper.getOrCreateStreak()
        } catch {
            errorMessage = "Failed to load meditation data: \(error.localizedDescription)"
            print("Error loading meditation data: \(error)")
        }
    }

    func refreshData() {
        loadData()
    }

    // MARK: - Session Management

    func startSession(_ session: MeditationSessionModel, recordMetrics: Bool = true) {
        guard !isPlaying else { return }

        currentSession = session
        currentTime = 0
        progress = 0
        isPlaying = true
        isPaused = false

        // Start timer
        startTimer()

        // TODO: Start audio playback if audioURL is provided
        // TODO: Start HealthKit monitoring if enabled
    }

    func pauseSession() {
        guard isPlaying, !isPaused else { return }
        isPaused = true
        stopTimer()
    }

    func resumeSession() {
        guard isPlaying, isPaused else { return }
        isPaused = false
        startTimer()
    }

    func stopSession() {
        isPlaying = false
        isPaused = false
        stopTimer()
        currentSession = nil
        currentTime = 0
        progress = 0
    }

    func completeSession(
        stressAfter: Int? = nil,
        painAfter: Int? = nil,
        moodAfter: Int? = nil,
        energyAfter: Int? = nil,
        notes: String? = nil
    ) async throws {
        guard let session = currentSession else {
            throw NSError(domain: "MeditationViewModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "No active session"])
        }

        stopTimer()

        let completed = currentTime >= (session.duration * 0.8) // 80% completion = completed

        // Save to Core Data
        do {
            _ = try persistenceHelper.saveMeditationSession(
                sessionType: session.type.rawValue,
                title: session.title,
                description: session.description,
                category: session.category.rawValue,
                durationSeconds: Int(session.duration),
                completedDuration: Int(currentTime),
                isCompleted: completed,
                stressLevelBefore: stressLevelBefore,
                stressLevelAfter: stressAfter,
                painLevelBefore: painLevelBefore,
                painLevelAfter: painAfter,
                moodBefore: moodBefore,
                moodAfter: moodAfter,
                energyAfter: energyAfter,
                breathingPattern: session.breathingPattern?.technique.rawValue,
                breathingTechnique: session.breathingPattern?.technique.rawValue,
                notes: notes,
                audioURL: session.audioURL,
                difficulty: session.difficulty.rawValue,
                targetSymptoms: session.targetSymptoms.map { $0.rawValue },
                recommendedTime: session.recommendedTime.first?.rawValue
            )

            // Reload data
            loadData()

            // Reset session state
            currentSession = nil
            isPlaying = false
            isPaused = false
            currentTime = 0
            progress = 0
            stressLevelBefore = nil
            painLevelBefore = nil
            moodBefore = nil

        } catch {
            errorMessage = "Failed to save session: \(error.localizedDescription)"
            throw error
        }
    }

    // MARK: - Timer Management

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self, self.isPlaying, !self.isPaused else { return }

                self.currentTime += 1

                if let session = self.currentSession {
                    self.progress = self.currentTime / session.duration

                    // Auto-complete when time is up
                    if self.currentTime >= session.duration {
                        self.stopTimer()
                        self.isPlaying = false
                        // User should call completeSession() manually to provide metrics
                    }
                }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    // MARK: - Recommendations

    func getRecommendedSessions() -> [MeditationSessionModel] {
        // Get sessions recommended for current time of day
        let timeBasedSessions = MeditationSessionModel.recommendedForCurrentTime()

        // TODO: Get user's current symptoms from latest SymptomLog
        // TODO: Filter by symptom relevance

        return Array(timeBasedSessions.prefix(5))
    }

    func getQuickSessions() -> [MeditationSessionModel] {
        MeditationSessionModel.quickSessions
    }

    func getSessions(for category: MeditationCategory) -> [MeditationSessionModel] {
        MeditationSessionModel.sessions(for: category)
    }

    func getBeginnerSessions() -> [MeditationSessionModel] {
        MeditationSessionModel.beginnerSessions
    }

    // MARK: - Analytics

    func getTotalMinutesThisWeek() -> Double {
        do {
            let weekStart = Calendar.current.date(
                from: Calendar.current.dateComponents([.yearForWeekOfYear, .weekOfYear], from: Date())
            )!
            return try persistenceHelper.getTotalMinutes(startDate: weekStart, endDate: Date())
        } catch {
            print("Error getting total minutes: \(error)")
            return 0
        }
    }

    func getTotalMinutesThisMonth() -> Double {
        do {
            let monthStart = Calendar.current.date(
                from: Calendar.current.dateComponents([.year, .month], from: Date())
            )!
            return try persistenceHelper.getTotalMinutes(startDate: monthStart, endDate: Date())
        } catch {
            print("Error getting total minutes: \(error)")
            return 0
        }
    }

    func getAveragePainReduction() -> Double? {
        do {
            return try persistenceHelper.getAveragePainReduction()
        } catch {
            print("Error getting pain reduction: \(error)")
            return nil
        }
    }

    func getFavoriteTypes() -> [(type: String, count: Int)] {
        do {
            return try persistenceHelper.getFavoriteTypes(limit: 3)
        } catch {
            print("Error getting favorite types: \(error)")
            return []
        }
    }

    // MARK: - Streak Management

    func checkStreakStatus() {
        do {
            streak = try persistenceHelper.getOrCreateStreak()
        } catch {
            print("Error checking streak: \(error)")
        }
    }

    // MARK: - Favorites Management

    func toggleFavorite(_ session: MeditationSessionModel) {
        // TODO: Implement favorites persistence
        // For now, just update local state
        if let index = availableSessions.firstIndex(where: { $0.id == session.id }) {
            var updatedSession = availableSessions[index]
            // Note: This requires MeditationSessionModel to be mutable
            // In production, store favorites separately
        }
    }

    // MARK: - Search & Filter

    func searchSessions(_ query: String) -> [MeditationSessionModel] {
        guard !query.isEmpty else { return availableSessions }

        let lowercasedQuery = query.lowercased()
        return availableSessions.filter { session in
            session.title.lowercased().contains(lowercasedQuery) ||
            session.description.lowercased().contains(lowercasedQuery) ||
            session.tags.contains(where: { $0.lowercased().contains(lowercasedQuery) })
        }
    }

    func filterSessions(
        by category: MeditationCategory? = nil,
        type: MeditationType? = nil,
        difficulty: DifficultyLevel? = nil,
        maxDuration: TimeInterval? = nil
    ) -> [MeditationSessionModel] {
        var filtered = availableSessions

        if let category = category {
            filtered = filtered.filter { $0.category == category }
        }

        if let type = type {
            filtered = filtered.filter { $0.type == type }
        }

        if let difficulty = difficulty {
            filtered = filtered.filter { $0.difficulty == difficulty }
        }

        if let maxDuration = maxDuration {
            filtered = filtered.filter { $0.duration <= maxDuration }
        }

        return filtered
    }

    // MARK: - Cleanup

    deinit {
        timer?.invalidate()
        timer = nil
        cancellables.removeAll()
    }
}

// MARK: - Formatted Time

extension MeditationViewModel {
    var currentTimeFormatted: String {
        let minutes = Int(currentTime) / 60
        let seconds = Int(currentTime) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    var remainingTimeFormatted: String {
        guard let session = currentSession else { return "0:00" }
        let remaining = max(0, session.duration - currentTime)
        let minutes = Int(remaining) / 60
        let seconds = Int(remaining) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    var totalTimeFormatted: String {
        guard let session = currentSession else { return "0:00" }
        let minutes = Int(session.duration) / 60
        let seconds = Int(session.duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}
