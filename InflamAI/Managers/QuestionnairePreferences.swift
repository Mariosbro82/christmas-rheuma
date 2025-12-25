//
//  QuestionnairePreferences.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-06-09.
//

import Foundation

// Note: This file provides backward compatibility and bridges to the new QuestionnaireUserPreferences system

struct QuestionnairePreferences {
    private static let scheduleKeyPrefix = "questionnaire_schedule_"

    private static func storageKey(for identifier: String) -> String {
        scheduleKeyPrefix + identifier
    }

    // MARK: - Bridge to New System

    static func schedule(for id: QuestionnaireID) -> QuestionnaireSchedule? {
        return QuestionnaireUserPreferences.shared.customSchedules[id]
    }

    static func save(schedule: QuestionnaireSchedule, for id: QuestionnaireID) {
        QuestionnaireUserPreferences.shared.setSchedule(schedule, for: id)
    }
}
