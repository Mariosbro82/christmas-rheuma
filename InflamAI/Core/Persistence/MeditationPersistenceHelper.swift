//
//  MeditationPersistenceHelper.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import Foundation
import CoreData

/// Helper for persisting and fetching meditation data from Core Data
@MainActor
class MeditationPersistenceHelper {
    let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    // MARK: - Meditation Session Operations

    /// Save a completed meditation session
    func saveMeditationSession(
        sessionType: String,
        title: String,
        description: String? = nil,
        category: String? = nil,
        durationSeconds: Int,
        completedDuration: Int,
        isCompleted: Bool = true,
        stressLevelBefore: Int? = nil,
        stressLevelAfter: Int? = nil,
        painLevelBefore: Int? = nil,
        painLevelAfter: Int? = nil,
        moodBefore: Int? = nil,
        moodAfter: Int? = nil,
        energyBefore: Int? = nil,
        energyAfter: Int? = nil,
        breathingPattern: String? = nil,
        breathingTechnique: String? = nil,
        avgHeartRate: Double? = nil,
        hrvValue: Double? = nil,
        notes: String? = nil,
        audioURL: String? = nil,
        difficulty: String? = nil,
        targetSymptoms: [String]? = nil,
        recommendedTime: String? = nil,
        symptomLogID: UUID? = nil
    ) throws -> MeditationSession {
        let session = MeditationSession(context: context)
        session.id = UUID()
        session.timestamp = Date()
        session.sessionType = sessionType
        session.title = title
        session.sessionDescription = description
        session.category = category
        session.durationSeconds = Int32(durationSeconds)
        session.completedDuration = Int32(completedDuration)
        session.isCompleted = isCompleted

        // Before/After Metrics
        if let stress = stressLevelBefore {
            session.stressLevelBefore = Int16(stress)
        }
        if let stress = stressLevelAfter {
            session.stressLevelAfter = Int16(stress)
        }
        if let pain = painLevelBefore {
            session.painLevelBefore = Int16(pain)
        }
        if let pain = painLevelAfter {
            session.painLevelAfter = Int16(pain)
        }
        if let mood = moodBefore {
            session.moodBefore = Int16(mood)
        }
        if let mood = moodAfter {
            session.moodAfter = Int16(mood)
        }
        if let energy = energyBefore {
            session.energyBefore = Int16(energy)
        }
        if let energy = energyAfter {
            session.energyAfter = Int16(energy)
        }

        // Session Details
        session.breathingPattern = breathingPattern
        session.breathingTechnique = breathingTechnique
        session.avgHeartRate = avgHeartRate ?? 0
        session.hrvValue = hrvValue ?? 0
        session.notes = notes
        session.audioURL = audioURL
        session.difficulty = difficulty
        session.recommendedTime = recommendedTime

        // Target Symptoms (encode as JSON)
        if let symptoms = targetSymptoms {
            session.targetSymptoms = try? JSONEncoder().encode(symptoms)
        }

        // Link to SymptomLog if provided
        if let symptomID = symptomLogID {
            let fetchRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            fetchRequest.predicate = NSPredicate(format: "id == %@", symptomID as CVarArg)
            if let symptomLog = try? context.fetch(fetchRequest).first {
                session.symptomLog = symptomLog
            }
        }

        try context.save()

        // Update streak after saving session
        try updateStreakAfterSession()

        return session
    }

    /// Fetch recent meditation sessions
    func fetchRecentSessions(days: Int = 30) throws -> [MeditationSession] {
        let request: NSFetchRequest<MeditationSession> = MeditationSession.fetchRequest()

        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        request.predicate = NSPredicate(format: "timestamp >= %@", startDate as CVarArg)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \MeditationSession.timestamp, ascending: false)]

        return try context.fetch(request)
    }

    /// Fetch meditation sessions by type
    func fetchSessions(ofType type: String, limit: Int? = nil) throws -> [MeditationSession] {
        let request: NSFetchRequest<MeditationSession> = MeditationSession.fetchRequest()
        request.predicate = NSPredicate(format: "sessionType == %@", type)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \MeditationSession.timestamp, ascending: false)]

        if let limit = limit {
            request.fetchLimit = limit
        }

        return try context.fetch(request)
    }

    /// Fetch completed sessions only
    func fetchCompletedSessions(days: Int = 30) throws -> [MeditationSession] {
        let request: NSFetchRequest<MeditationSession> = MeditationSession.fetchRequest()

        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND isCompleted == YES",
            startDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \MeditationSession.timestamp, ascending: false)]

        return try context.fetch(request)
    }

    /// Get total meditation minutes for a date range
    func getTotalMinutes(startDate: Date, endDate: Date) throws -> Double {
        let request: NSFetchRequest<MeditationSession> = MeditationSession.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp <= %@ AND isCompleted == YES",
            startDate as CVarArg,
            endDate as CVarArg
        )

        let sessions = try context.fetch(request)
        let totalSeconds = sessions.reduce(0) { $0 + Int($1.completedDuration) }
        return Double(totalSeconds) / 60.0
    }

    // MARK: - Streak Operations

    /// Get or create the meditation streak (singleton)
    func getOrCreateStreak() throws -> MeditationStreak {
        let request: NSFetchRequest<MeditationStreak> = MeditationStreak.fetchRequest()
        request.fetchLimit = 1

        if let streak = try context.fetch(request).first {
            return streak
        }

        // Create new streak
        let newStreak = MeditationStreak(context: context)
        newStreak.id = UUID()
        newStreak.currentStreak = 0
        newStreak.longestStreak = 0
        newStreak.totalSessions = 0
        newStreak.totalMinutes = 0
        newStreak.weeklyGoal = 7
        newStreak.monthlyGoal = 30
        newStreak.weeklyProgress = 0
        newStreak.monthlyProgress = 0
        newStreak.createdAt = Date()
        newStreak.lastUpdated = Date()

        try context.save()
        return newStreak
    }

    /// Update streak after completing a session
    func updateStreakAfterSession() throws {
        let streak = try getOrCreateStreak()
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())

        // Increment total sessions
        streak.totalSessions += 1

        // Update total minutes
        let todaySessions = try fetchRecentSessions(days: 1)
        let todayMinutes = todaySessions.reduce(0.0) { $0 + (Double($1.completedDuration) / 60.0) }
        streak.totalMinutes += todayMinutes

        // Update streak logic
        if let lastSessionDate = streak.lastSessionDate {
            let lastDay = calendar.startOfDay(for: lastSessionDate)
            let daysDifference = calendar.dateComponents([.day], from: lastDay, to: today).day ?? 0

            if daysDifference == 0 {
                // Same day - don't increment streak
            } else if daysDifference == 1 {
                // Consecutive day - increment streak
                streak.currentStreak += 1
                if streak.currentStreak > streak.longestStreak {
                    streak.longestStreak = streak.currentStreak
                }
            } else {
                // Streak broken - reset
                streak.currentStreak = 1
            }
        } else {
            // First ever session
            streak.currentStreak = 1
            streak.longestStreak = 1
        }

        streak.lastSessionDate = Date()
        streak.lastUpdated = Date()

        // Update weekly/monthly progress
        try updateWeeklyMonthlyProgress(streak: streak)

        try context.save()
    }

    /// Update weekly and monthly progress counts
    private func updateWeeklyMonthlyProgress(streak: MeditationStreak) throws {
        let calendar = Calendar.current
        let now = Date()

        // Weekly progress
        let weekStart = calendar.date(from: calendar.dateComponents([.yearForWeekOfYear, .weekOfYear], from: now))!
        let weeklySessions = try fetchCompletedSessions(days: 7).filter {
            guard let timestamp = $0.timestamp else { return false }
            return timestamp >= weekStart
        }
        streak.weeklyProgress = Int16(weeklySessions.count)

        // Monthly progress
        let monthStart = calendar.date(from: calendar.dateComponents([.year, .month], from: now))!
        let monthlySessions = try fetchCompletedSessions(days: 31).filter {
            guard let timestamp = $0.timestamp else { return false }
            return timestamp >= monthStart
        }
        streak.monthlyProgress = Int16(monthlySessions.count)
    }

    /// Reset streak manually (if needed)
    func resetStreak() throws {
        guard let streak = try? getOrCreateStreak() else { return }
        streak.currentStreak = 0
        streak.lastSessionDate = nil
        streak.lastUpdated = Date()
        try context.save()
    }

    // MARK: - Analytics

    /// Get meditation sessions grouped by day
    func getSessionsGroupedByDay(days: Int = 30) throws -> [Date: [MeditationSession]] {
        let sessions = try fetchRecentSessions(days: days)
        let calendar = Calendar.current

        var grouped: [Date: [MeditationSession]] = [:]
        for session in sessions {
            guard let timestamp = session.timestamp else { continue }
            let day = calendar.startOfDay(for: timestamp)
            grouped[day, default: []].append(session)
        }

        return grouped
    }

    /// Get average pain reduction from meditation
    func getAveragePainReduction() throws -> Double? {
        let request: NSFetchRequest<MeditationSession> = MeditationSession.fetchRequest()
        request.predicate = NSPredicate(
            format: "painLevelBefore != nil AND painLevelAfter != nil AND isCompleted == YES"
        )

        let sessions = try context.fetch(request)

        guard !sessions.isEmpty else { return nil }

        let totalReduction = sessions.reduce(0.0) { sum, session in
            let before = Double(session.painLevelBefore)
            let after = Double(session.painLevelAfter)
            return sum + (before - after)
        }

        return totalReduction / Double(sessions.count)
    }

    /// Get favorite meditation types (most completed)
    func getFavoriteTypes(limit: Int = 5) throws -> [(type: String, count: Int)] {
        let request: NSFetchRequest<MeditationSession> = MeditationSession.fetchRequest()
        request.predicate = NSPredicate(format: "isCompleted == YES AND sessionType != nil")

        let sessions = try context.fetch(request)

        var typeCounts: [String: Int] = [:]
        for session in sessions {
            guard let type = session.sessionType else { continue }
            typeCounts[type, default: 0] += 1
        }

        return typeCounts
            .sorted { $0.value > $1.value }
            .prefix(limit)
            .map { (type: $0.key, count: $0.value) }
    }

    // MARK: - Deletion

    /// Delete a meditation session
    func deleteSession(_ session: MeditationSession) throws {
        context.delete(session)
        try context.save()

        // Update streak after deletion
        try updateStreakAfterSession()
    }

    /// Delete all meditation data (for settings/privacy)
    func deleteAllMeditationData() throws {
        // Delete all sessions
        let sessionRequest: NSFetchRequest<NSFetchRequestResult> = MeditationSession.fetchRequest()
        let sessionDelete = NSBatchDeleteRequest(fetchRequest: sessionRequest)
        try context.execute(sessionDelete)

        // Delete streak
        let streakRequest: NSFetchRequest<NSFetchRequestResult> = MeditationStreak.fetchRequest()
        let streakDelete = NSBatchDeleteRequest(fetchRequest: streakRequest)
        try context.execute(streakDelete)

        try context.save()
    }
}
