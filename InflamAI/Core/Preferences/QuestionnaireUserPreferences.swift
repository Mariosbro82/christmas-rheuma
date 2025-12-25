//
//  QuestionnaireUserPreferences.swift
//  InflamAI-Swift
//
//  User preferences for questionnaire management
//  Handles enabled questionnaires, custom schedules, and notification settings
//

import Foundation
import Combine

// MARK: - User Preferences Model

class QuestionnaireUserPreferences: ObservableObject {
    static let shared = QuestionnaireUserPreferences()

    // UserDefaults keys
    private enum Keys {
        static let enabledQuestionnaires = "questionnaire_enabled_set"
        static let customSchedules = "questionnaire_custom_schedules"
        static let notificationSettings = "questionnaire_notifications"
        static let hasCompletedSetup = "questionnaire_setup_completed"
    }

    // MARK: - Published Properties

    @Published private(set) var enabledQuestionnaires: Set<QuestionnaireID> {
        didSet {
            saveEnabledQuestionnaires()
        }
    }

    @Published private(set) var customSchedules: [QuestionnaireID: QuestionnaireSchedule] {
        didSet {
            saveCustomSchedules()
        }
    }

    @Published private(set) var notificationSettings: [QuestionnaireID: Bool] {
        didSet {
            saveNotificationSettings()
        }
    }

    @Published var hasCompletedSetup: Bool {
        didSet {
            UserDefaults.standard.set(hasCompletedSetup, forKey: Keys.hasCompletedSetup)
        }
    }

    private let userDefaults: UserDefaults

    // MARK: - Initialization

    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults

        // Load enabled questionnaires (default: only BASDAI)
        if let data = userDefaults.data(forKey: Keys.enabledQuestionnaires),
           let decoded = try? JSONDecoder().decode(Set<QuestionnaireID>.self, from: data) {
            self.enabledQuestionnaires = decoded
        } else {
            // First launch: enable BASDAI by default
            self.enabledQuestionnaires = [.basdai]
        }

        // Load custom schedules
        if let data = userDefaults.data(forKey: Keys.customSchedules),
           let decoded = try? JSONDecoder().decode([QuestionnaireID: QuestionnaireSchedule].self, from: data) {
            self.customSchedules = decoded
        } else {
            self.customSchedules = [:]
        }

        // Load notification settings (default: all enabled)
        if let data = userDefaults.data(forKey: Keys.notificationSettings),
           let decoded = try? JSONDecoder().decode([QuestionnaireID: Bool].self, from: data) {
            self.notificationSettings = decoded
        } else {
            self.notificationSettings = [:]
        }

        // Setup status
        self.hasCompletedSetup = userDefaults.bool(forKey: Keys.hasCompletedSetup)
    }

    // MARK: - Public Methods

    /// Enable a questionnaire
    func enableQuestionnaire(_ id: QuestionnaireID) {
        enabledQuestionnaires.insert(id)
        // Enable notifications by default
        if notificationSettings[id] == nil {
            notificationSettings[id] = true
        }
    }

    /// Disable a questionnaire
    func disableQuestionnaire(_ id: QuestionnaireID) {
        enabledQuestionnaires.remove(id)
    }

    /// Toggle questionnaire enabled state
    func toggleQuestionnaire(_ id: QuestionnaireID) {
        if enabledQuestionnaires.contains(id) {
            disableQuestionnaire(id)
        } else {
            enableQuestionnaire(id)
        }
    }

    /// Check if questionnaire is enabled
    func isEnabled(_ id: QuestionnaireID) -> Bool {
        return enabledQuestionnaires.contains(id)
    }

    /// Set custom schedule for a questionnaire
    func setSchedule(_ schedule: QuestionnaireSchedule, for id: QuestionnaireID) {
        customSchedules[id] = schedule
    }

    /// Get schedule for a questionnaire (custom or default)
    func getSchedule(for id: QuestionnaireID) -> QuestionnaireSchedule {
        return customSchedules[id] ?? id.defaultSchedule
    }

    /// Reset to default schedule
    func resetToDefaultSchedule(for id: QuestionnaireID) {
        customSchedules.removeValue(forKey: id)
    }

    /// Enable/disable notifications for a questionnaire
    func setNotificationsEnabled(_ enabled: Bool, for id: QuestionnaireID) {
        notificationSettings[id] = enabled
    }

    /// Check if notifications are enabled for a questionnaire
    func areNotificationsEnabled(for id: QuestionnaireID) -> Bool {
        return notificationSettings[id] ?? true // Default: enabled
    }

    /// Get all enabled questionnaires sorted by category
    func getEnabledQuestionnairesByCategory() -> [DiseaseCategory: [QuestionnaireID]] {
        var grouped: [DiseaseCategory: [QuestionnaireID]] = [:]
        for id in enabledQuestionnaires {
            grouped[id.category, default: []].append(id)
        }
        // Sort each category's questionnaires
        for (category, ids) in grouped {
            grouped[category] = ids.sorted { $0.rawValue < $1.rawValue }
        }
        return grouped
    }

    /// Reset all preferences to defaults
    func resetToDefaults() {
        enabledQuestionnaires = [.basdai]
        customSchedules = [:]
        notificationSettings = [:]
        hasCompletedSetup = false
    }

    // MARK: - Persistence

    private func saveEnabledQuestionnaires() {
        if let encoded = try? JSONEncoder().encode(enabledQuestionnaires) {
            userDefaults.set(encoded, forKey: Keys.enabledQuestionnaires)
        }
    }

    private func saveCustomSchedules() {
        if let encoded = try? JSONEncoder().encode(customSchedules) {
            userDefaults.set(encoded, forKey: Keys.customSchedules)
        }
    }

    private func saveNotificationSettings() {
        if let encoded = try? JSONEncoder().encode(notificationSettings) {
            userDefaults.set(encoded, forKey: Keys.notificationSettings)
        }
    }
}

// MARK: - Convenience Extensions

extension QuestionnaireUserPreferences {
    /// Get all questionnaires grouped by category with enabled status
    func getAllQuestionnairesByCategory() -> [(category: DiseaseCategory, questionnaires: [(id: QuestionnaireID, isEnabled: Bool)])] {
        var result: [DiseaseCategory: [(id: QuestionnaireID, isEnabled: Bool)]] = [:]

        for id in QuestionnaireID.allCases {
            let category = id.category
            let isEnabled = enabledQuestionnaires.contains(id)
            result[category, default: []].append((id: id, isEnabled: isEnabled))
        }

        // Sort and return
        return result.sorted { $0.key.rawValue < $1.key.rawValue }.map { category, questionnaires in
            (
                category: category,
                questionnaires: questionnaires.sorted { $0.id.rawValue < $1.id.rawValue }
            )
        }
    }

    /// Get count of enabled questionnaires
    var enabledCount: Int {
        return enabledQuestionnaires.count
    }

    /// Get due questionnaires for today
    func getDueQuestionnaires(using manager: QuestionnaireManager) -> [QuestionnaireID] {
        let allStates = manager.dueStates()
        return allStates
            .filter { state in
                enabledQuestionnaires.contains(state.questionnaireID) && state.isDue
            }
            .map { $0.questionnaireID }
    }
}
