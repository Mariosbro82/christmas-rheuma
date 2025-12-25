//
//  QuestionnaireManager.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-06-09.
//

import Foundation
import CoreData

struct QuestionnaireSaveOutcome {
    let responseID: NSManagedObjectID
    let requiresSpeedConfirmation: Bool
}

struct QuestionnaireDueState: Identifiable {
    let questionnaireID: QuestionnaireID
    let isDue: Bool
    let windowOpens: Date
    let windowCloses: Date
    let lastSubmission: QuestionnaireResponse?
    let pendingCarryOverCount: Int
    
    var id: QuestionnaireID { questionnaireID }
}

final class QuestionnaireManager: ObservableObject {
    private let viewContext: NSManagedObjectContext
    private var schedules: [QuestionnaireID: QuestionnaireSchedule]
    private var calendar: Calendar
    
    init(
        viewContext: NSManagedObjectContext,
        schedules: [QuestionnaireID: QuestionnaireSchedule] = [:],
        calendar: Calendar = Calendar(identifier: .gregorian)
    ) {
        self.viewContext = viewContext
        self.schedules = QuestionnaireID.allCases.reduce(into: [:]) { result, questionnaireID in
            if let override = schedules[questionnaireID] {
                result[questionnaireID] = override
            } else if let stored = QuestionnairePreferences.schedule(for: questionnaireID) {
                result[questionnaireID] = stored
            } else {
                result[questionnaireID] = questionnaireID.defaultSchedule
            }
        }
        self.calendar = calendar
    }
    
    func updateSchedule(_ schedule: QuestionnaireSchedule, for questionnaireID: QuestionnaireID) {
        schedules[questionnaireID] = schedule
        QuestionnairePreferences.save(schedule: schedule, for: questionnaireID)
    }
    
    func fetchSchedule(for questionnaireID: QuestionnaireID) -> QuestionnaireSchedule {
        schedules[questionnaireID] ?? questionnaireID.defaultSchedule
    }
    
    @discardableResult
    func recordResponse(
        for questionnaireID: QuestionnaireID,
        answers: QuestionnaireAnswerSet,
        note: String?,
        duration: TimeInterval,
        appVersion: String,
        deviceLocale: Locale = .current,
        userID: String? = nil,
        isDraft: Bool = false,
        createdAt: Date = Date()
    ) throws -> QuestionnaireSaveOutcome {
        let definition = definition(for: questionnaireID)
        let expectedItemCount = definition.items.count
        let answeredCount = answers.values.count
        
        if !isDraft && answeredCount != expectedItemCount {
            throw QuestionnaireError.incompleteAnswers(expected: expectedItemCount, actual: answeredCount)
        }
        
        let score = QuestionnaireScoring.score(for: questionnaireID, answers: answers.values)
        if !isDraft && !score.isFinite {
            throw QuestionnaireError.invalidScore
        }
        
        let schedule = fetchSchedule(for: questionnaireID)
        let timezone = schedule.timezone
        let durationMs = duration * 1000.0
        let meta = QuestionnaireMetaPayload(
            appVersion: appVersion,
            durationMs: durationMs,
            isDraft: isDraft,
            deviceLocale: deviceLocale.identifier
        )
        
        let response: QuestionnaireResponse = try createResponse()
        response.configure(
            questionnaireID: questionnaireID,
            answers: answers,
            score: score.isFinite ? score : 0.0,
            note: note,
            timezone: timezone,
            durationMs: durationMs,
            meta: meta,
            userID: userID,
            createdAt: createdAt
        )
        
        try viewContext.save()
        
        let requiresSpeedConfirmation = !isDraft && duration < 10
        return QuestionnaireSaveOutcome(
            responseID: response.objectID,
            requiresSpeedConfirmation: requiresSpeedConfirmation
        )
    }
    
    func updateDraft(
        _ response: QuestionnaireResponse,
        answers: QuestionnaireAnswerSet,
        note: String?,
        duration: TimeInterval
    ) throws {
        guard let questionnaireID = response.questionnaireIDEnum else {
            throw QuestionnaireError.unknownQuestionnaire
        }
        let score = QuestionnaireScoring.score(for: questionnaireID, answers: answers.values)
        response.answers = answers
        response.note = note
        response.score = score.isFinite ? score : response.score
        response.durationMs = duration * 1000.0
        if var meta = response.meta {
            meta.durationMs = response.durationMs
            response.meta = meta
        }
        try viewContext.save()
    }
    
    func fetchRecentResponses(for questionnaireID: QuestionnaireID, limit: Int = 30) -> [QuestionnaireResponse] {
        let request: NSFetchRequest<QuestionnaireResponse> = QuestionnaireResponse.fetchRequest()
        request.predicate = NSPredicate(format: "questionnaireID == %@", questionnaireID.rawValue)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \QuestionnaireResponse.createdAt, ascending: false)]
        request.fetchLimit = limit
        
        do {
            return try viewContext.fetch(request)
        } catch {
            print("Failed to fetch questionnaire responses: \(error)")
            return []
        }
    }
    
    func fetchResponses(for questionnaireID: QuestionnaireID, in interval: DateInterval) -> [QuestionnaireResponse] {
        let request: NSFetchRequest<QuestionnaireResponse> = QuestionnaireResponse.fetchRequest()
        request.predicate = NSPredicate(
            format: "questionnaireID == %@ AND createdAt >= %@ AND createdAt <= %@",
            questionnaireID.rawValue,
            interval.start as NSDate,
            interval.end as NSDate
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \QuestionnaireResponse.createdAt, ascending: true)]
        
        do {
            return try viewContext.fetch(request)
        } catch {
            print("Failed to fetch responses for interval: \(error)")
            return []
        }
    }
    
    func state(for questionnaireID: QuestionnaireID, referenceDate: Date = Date()) -> QuestionnaireDueState {
        let schedule = fetchSchedule(for: questionnaireID)
        let timezone = schedule.timezone
        var calendar = self.calendar
        calendar.timeZone = timezone
        
        let now = referenceDate
        let lastSubmission = fetchRecentResponses(for: questionnaireID, limit: 1).first
        let window = computeWindow(for: questionnaireID, schedule: schedule, calendar: calendar, referenceDate: now)
        let pendingCarryOverCount: Int
        switch questionnaireID {
        case .basdai:
            pendingCarryOverCount = min(2, missingDailyCount(for: questionnaireID, schedule: schedule, referenceDate: now))
        default:
            pendingCarryOverCount = 0
        }
        
        let isWithinWindow = window.contains(now)
        let alreadySubmitted = hasSubmission(for: questionnaireID, in: window, calendar: calendar)
        let prerequisitesMet = prerequisitesSatisfied(for: questionnaireID, calendar: calendar, referenceDate: now)
        
        let due = !alreadySubmitted && isWithinWindow && prerequisitesMet
        return QuestionnaireDueState(
            questionnaireID: questionnaireID,
            isDue: due,
            windowOpens: window.start,
            windowCloses: window.end,
            lastSubmission: lastSubmission,
            pendingCarryOverCount: pendingCarryOverCount
        )
    }
    
    func dueStates(referenceDate: Date = Date()) -> [QuestionnaireDueState] {
        QuestionnaireID.allCases.map { state(for: $0, referenceDate: referenceDate) }
    }
    
    func deleteResponse(_ response: QuestionnaireResponse) throws {
        viewContext.delete(response)
        try viewContext.save()
    }
    
    // MARK: - Helpers
    
    private func createResponse() throws -> QuestionnaireResponse {
        guard let entity = NSEntityDescription.entity(forEntityName: "QuestionnaireResponse", in: viewContext) else {
            throw QuestionnaireError.modelUnavailable
        }
        let response = QuestionnaireResponse(entity: entity, insertInto: viewContext)
        response.id = UUID()
        return response
    }
    
    private func definition(for questionnaireID: QuestionnaireID) -> QuestionnaireDefinition {
        // Use the static helper method from QuestionnaireDefinition
        if let def = QuestionnaireDefinition.definition(for: questionnaireID) {
            return def
        }
        // Fallback for questionnaires not yet fully implemented
        // Return a minimal placeholder definition
        return QuestionnaireDefinition(
            id: questionnaireID,
            version: "0.0.1-placeholder",
            items: [],
            periodDescriptionKey: "questionnaire.placeholder.period",
            notesAllowed: true
        )
    }
    
    private func computeWindow(
        for questionnaireID: QuestionnaireID,
        schedule: QuestionnaireSchedule,
        calendar: Calendar,
        referenceDate: Date
    ) -> DateInterval {
        switch schedule.frequency {
        case .daily(let time):
            let components = DateComponents(
                year: calendar.component(.year, from: referenceDate),
                month: calendar.component(.month, from: referenceDate),
                day: calendar.component(.day, from: referenceDate),
                hour: time.hour ?? 0,
                minute: time.minute ?? 0,
                second: 0
            )
            let start = calendar.date(from: components) ?? referenceDate
            let nextMidnight = calendar.nextDate(after: start, matching: DateComponents(hour: 0, minute: 0), matchingPolicy: .nextTimePreservingSmallerComponents) ?? calendar.date(byAdding: .day, value: 1, to: start) ?? start.addingTimeInterval(24 * 3600)
            let end = min(start.addingTimeInterval(TimeInterval(schedule.windowHours) * 3600), nextMidnight)
            return DateInterval(start: start, end: end)

        case .weekly(let weekday, let time):
            let referenceWeek = calendar.dateInterval(of: .weekOfYear, for: referenceDate) ?? DateInterval(start: referenceDate, duration: 7 * 24 * 3600)
            var startComponents = calendar.dateComponents([.yearForWeekOfYear, .weekOfYear], from: referenceWeek.start)
            startComponents.weekday = weekday
            startComponents.hour = time.hour
            startComponents.minute = time.minute
            let start = calendar.date(from: startComponents) ?? referenceWeek.start
            let end = start.addingTimeInterval(TimeInterval(schedule.windowHours) * 3600)
            return DateInterval(start: start, end: end)

        case .monthly(let day, let time):
            var startComponents = calendar.dateComponents([.year, .month], from: referenceDate)
            startComponents.day = min(day, 28) // Safety: avoid invalid dates
            startComponents.hour = time.hour
            startComponents.minute = time.minute
            let start = calendar.date(from: startComponents) ?? referenceDate
            let end = start.addingTimeInterval(TimeInterval(schedule.windowHours) * 3600)
            return DateInterval(start: start, end: end)

        case .onDemand:
            // On-demand questionnaires have no scheduled window
            // Return a window that's always in the past so it never shows as "due"
            let past = referenceDate.addingTimeInterval(-365 * 24 * 3600)
            return DateInterval(start: past, end: past)
        }
    }
    
    private func hasSubmission(
        for questionnaireID: QuestionnaireID,
        in window: DateInterval,
        calendar: Calendar
    ) -> Bool {
        let request: NSFetchRequest<QuestionnaireResponse> = QuestionnaireResponse.fetchRequest()
        request.predicate = NSPredicate(
            format: "questionnaireID == %@ AND createdAt >= %@ AND createdAt <= %@",
            questionnaireID.rawValue,
            window.start as NSDate,
            window.end as NSDate
        )
        request.fetchLimit = 1
        do {
            return try viewContext.count(for: request) > 0
        } catch {
            print("Failed to count submissions: \(error)")
            return false
        }
    }
    
    private func missingDailyCount(
        for questionnaireID: QuestionnaireID,
        schedule: QuestionnaireSchedule,
        referenceDate: Date
    ) -> Int {
        var calendar = self.calendar
        calendar.timeZone = schedule.timezone
        let todayLocalStart = calendar.startOfDay(for: referenceDate)
        guard let previousDay = calendar.date(byAdding: .day, value: -1, to: todayLocalStart) else {
            return 0
        }
        let twoDaysAgo = calendar.date(byAdding: .day, value: -1, to: previousDay) ?? previousDay
        
        let intervals = [
            DateInterval(start: calendar.date(bySettingHour: 0, minute: 0, second: 0, of: previousDay) ?? previousDay,
                         end: calendar.date(bySettingHour: 23, minute: 59, second: 59, of: previousDay) ?? previousDay),
            DateInterval(start: calendar.date(bySettingHour: 0, minute: 0, second: 0, of: twoDaysAgo) ?? twoDaysAgo,
                         end: calendar.date(bySettingHour: 23, minute: 59, second: 59, of: twoDaysAgo) ?? twoDaysAgo)
        ]
        
        let missing = intervals.filter { interval in
            !hasSubmission(for: questionnaireID, in: interval, calendar: calendar)
        }.count
        return missing
    }
    
    private func prerequisitesSatisfied(
        for questionnaireID: QuestionnaireID,
        calendar: Calendar,
        referenceDate: Date
    ) -> Bool {
        let schedule = fetchSchedule(for: questionnaireID)
        guard !schedule.prerequisites.isEmpty else { return true }
        for prerequisite in schedule.prerequisites {
            let prerequisiteSchedule = fetchSchedule(for: prerequisite)
            let window = computeWindow(for: prerequisite, schedule: prerequisiteSchedule, calendar: calendar, referenceDate: referenceDate)
            if !hasSubmission(for: prerequisite, in: window, calendar: calendar) {
                return false
            }
        }
        return true
    }
}

enum QuestionnaireError: LocalizedError {
    case incompleteAnswers(expected: Int, actual: Int)
    case invalidScore
    case modelUnavailable
    case unknownQuestionnaire
    
    var errorDescription: String? {
        switch self {
        case let .incompleteAnswers(expected, actual):
            return "Incomplete questionnaire. Expected \(expected) responses, received \(actual)."
        case .invalidScore:
            return "Unable to calculate score. Please review answers."
        case .modelUnavailable:
            return "Questionnaire data model is unavailable."
        case .unknownQuestionnaire:
            return "Unknown questionnaire identifier."
        }
    }
}
