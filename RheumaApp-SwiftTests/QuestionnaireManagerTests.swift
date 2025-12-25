import XCTest
import CoreData
@testable import RheumaApp_Swift

final class QuestionnaireManagerTests: XCTestCase {
    private var persistenceController: PersistenceController!
    private var context: NSManagedObjectContext!
    private var preferenceBackup: [QuestionnaireID: Data?] = [:]
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        persistenceController = PersistenceController(inMemory: true)
        context = persistenceController.container.viewContext
        backupPreferences()
        clearPreferences()
    }
    
    override func tearDownWithError() throws {
        restorePreferences()
        preferenceBackup.removeAll()
        context = nil
        persistenceController = nil
        try super.tearDownWithError()
    }
    
    func testRecordResponseRequiresAllAnswers() throws {
        let manager = QuestionnaireManager(viewContext: context)
        let incompleteAnswers = QuestionnaireAnswerSet(values: ["Q1": 5])
        
        XCTAssertThrowsError(
            try manager.recordResponse(
                for: .basfi,
                answers: incompleteAnswers,
                note: nil,
                duration: 12,
                appVersion: "1.0",
                deviceLocale: Locale(identifier: "en_US"),
                isDraft: false
            )
        ) { error in
            guard case let QuestionnaireError.incompleteAnswers(expected, actual) = error else {
                return XCTFail("Expected incompleteAnswers error, received \(error)")
            }
            XCTAssertEqual(expected, QuestionnaireDefinition.basfi.items.count)
            XCTAssertEqual(actual, 1)
        }
    }
    
    func testRecordResponseFlagsFastCompletion() throws {
        let manager = QuestionnaireManager(viewContext: context)
        let answers = QuestionnaireAnswerSet(values: [
            "Q1": 4,
            "Q2": 5,
            "Q3": 6,
            "Q4": 3,
            "Q5": 2,
            "Q6": 1
        ])
        
        let outcome = try manager.recordResponse(
            for: .basdai,
            answers: answers,
            note: "Felt okay today",
            duration: 5, // seconds
            appVersion: "1.0",
            deviceLocale: Locale(identifier: "en_US"),
            isDraft: false
        )
        
        XCTAssertTrue(outcome.requiresSpeedConfirmation, "Fast submissions should require confirmation.")
        
        let request: NSFetchRequest<QuestionnaireResponse> = QuestionnaireResponse.fetchRequest()
        request.predicate = NSPredicate(format: "questionnaireID == %@", QuestionnaireID.basdai.rawValue)
        let savedResponses = try context.fetch(request)
        XCTAssertEqual(savedResponses.count, 1)
        XCTAssertEqual(savedResponses.first?.score ?? -1, QuestionnaireScoring.score(for: .basdai, answers: answers.values), accuracy: 0.0001)
    }
    
    func testDailyScheduleHandlesEuropeBerlinDST() throws {
        let timezoneIdentifier = "Europe/Berlin"
        let baselineComponents = DateComponents(hour: 20, minute: 0)
        let dailySchedule = QuestionnaireSchedule(
            frequency: .daily(time: baselineComponents),
            windowHours: 4,
            timezoneIdentifier: timezoneIdentifier
        )
        QuestionnairePreferences.save(schedule: dailySchedule, for: .basdai)
        
        let manager = QuestionnaireManager(viewContext: context)
        let reference = makeDate(year: 2024, month: 3, day: 31, hour: 20, minute: 30, timezoneIdentifier: timezoneIdentifier)
        
        let state = manager.state(for: .basdai, referenceDate: reference)
        XCTAssertTrue(state.isDue, "Daily questionnaire should be due during the reminder window.")
        
        let expectedStart = makeDate(year: 2024, month: 3, day: 31, hour: 20, minute: 0, timezoneIdentifier: timezoneIdentifier)
        XCTAssertEqual(state.windowOpens.timeIntervalSinceReferenceDate, expectedStart.timeIntervalSinceReferenceDate, accuracy: 1.0)
        
        let fourHoursLater = expectedStart.addingTimeInterval(4 * 3600)
        XCTAssertEqual(state.windowCloses.timeIntervalSinceReferenceDate, fourHoursLater.timeIntervalSinceReferenceDate, accuracy: 1.0)
    }
    
    // MARK: - Helpers
    
    private func backupPreferences() {
        QuestionnaireID.allCases.forEach { id in
            let key = QuestionnairePreferences.storageKey(for: id)
            preferenceBackup[id] = UserDefaults.standard.data(forKey: key)
        }
    }
    
    private func clearPreferences() {
        QuestionnaireID.allCases.forEach { id in
            QuestionnairePreferences.clear(for: id)
        }
    }
    
    private func restorePreferences() {
        QuestionnaireID.allCases.forEach { id in
            let key = QuestionnairePreferences.storageKey(for: id)
            if let data = preferenceBackup[id], let data = data {
                UserDefaults.standard.set(data, forKey: key)
            } else {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }
    }
    
    private func makeDate(year: Int, month: Int, day: Int, hour: Int, minute: Int, timezoneIdentifier: String) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: timezoneIdentifier) ?? .current
        let components = DateComponents(year: year, month: month, day: day, hour: hour, minute: minute)
        return calendar.date(from: components) ?? Date()
    }
}
