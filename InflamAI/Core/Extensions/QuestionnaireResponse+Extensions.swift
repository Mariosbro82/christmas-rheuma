//
//  QuestionnaireResponse+Extensions.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-06-09.
//

import Foundation
import CoreData

extension QuestionnaireResponse {
    var questionnaireIDEnum: QuestionnaireID? {
        QuestionnaireID(rawValue: questionnaireID ?? "")
    }
    
    var answers: QuestionnaireAnswerSet {
        get {
            guard
                let data = answersData,
                let values = try? JSONDecoder().decode([String: Double].self, from: data)
            else {
                return QuestionnaireAnswerSet(values: [:])
            }
            return QuestionnaireAnswerSet(values: values)
        }
        set {
            answersData = try? JSONEncoder().encode(newValue.values)
        }
    }
    
    var meta: QuestionnaireMetaPayload? {
        get {
            guard
                let data = metaData,
                let payload = try? JSONDecoder().decode(QuestionnaireMetaPayload.self, from: data)
            else {
                return nil
            }
            return payload
        }
        set {
            metaData = try? JSONEncoder().encode(newValue)
        }
    }
    
    func configure(
        questionnaireID: QuestionnaireID,
        answers: QuestionnaireAnswerSet,
        score: Double,
        note: String?,
        timezone: TimeZone,
        durationMs: Double,
        meta: QuestionnaireMetaPayload?,
        userID: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id ?? UUID()
        self.questionnaireID = questionnaireID.rawValue
        self.answers = answers
        self.score = score
        self.note = note
        self.createdAt = createdAt
        self.timezoneIdentifier = timezone.identifier
        self.localDate = Self.formatLocalDate(createdAt, timezone: timezone)
        self.durationMs = durationMs
        self.meta = meta
        self.userID = userID
    }
    
    private static func formatLocalDate(_ date: Date, timezone: TimeZone) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timezone
        let components = calendar.dateComponents([.year, .month, .day], from: date)
        guard
            let year = components.year,
            let month = components.month,
            let day = components.day
        else {
            return ISO8601DateFormatter().string(from: date)
        }
        return String(format: "%04d-%02d-%02d", year, month, day)
    }
}
