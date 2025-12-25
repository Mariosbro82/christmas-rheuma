//
//  MedicationViewModel.swift
//  InflamAI
//
//  ViewModel for medication management with adherence tracking
//

import Foundation
import CoreData
import UserNotifications
import UIKit

@MainActor
class MedicationViewModel: ObservableObject {
    @Published var activeMedications: [MedicationData] = []
    @Published var inactiveMedications: [MedicationData] = []
    @Published var todaysDoses: [ScheduledDose] = []
    @Published var last30DaysDoses: [DoseLog] = []
    @Published var weeklyAdherence: Double = 0
    @Published var monthlyAdherence: Double = 0
    // CRIT-005: Add loading and error states
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    var takenCount: Int {
        todaysDoses.filter { $0.isTaken }.count
    }

    private let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    // MARK: - Loading

    func loadMedications() {
        isLoading = true
        errorMessage = nil

        Task {
            do {
                let medications = try await context.perform {
                    let request: NSFetchRequest<Medication> = Medication.fetchRequest()
                    request.sortDescriptors = [NSSortDescriptor(keyPath: \Medication.name, ascending: true)]
                    return try self.context.fetch(request)
                }

                let medicationData = medications.map { medication in
                    MedicationData(
                        id: medication.id ?? UUID(),
                        name: medication.name ?? "",
                        category: medication.category,
                        dosage: medication.dosage,
                        unit: medication.dosageUnit ?? "mg",
                        frequency: medication.frequency ?? "Daily",
                        route: medication.route ?? "Oral",
                        isBiologic: medication.isBiologic,
                        isActive: medication.isActive,
                        reminderEnabled: medication.reminderEnabled
                    )
                }

                activeMedications = medicationData.filter { $0.isActive }
                inactiveMedications = medicationData.filter { !$0.isActive }
                isLoading = false

            } catch {
                print("Error loading medications: \(error)")
                errorMessage = "Failed to load medications. Please try again."
                isLoading = false
            }
        }
    }

    func loadTodaysDoses() {
        Task {
            // Generate scheduled doses for today
            let calendar = Calendar.current
            let today = calendar.startOfDay(for: Date())

            var scheduledDoses: [ScheduledDose] = []

            for medication in activeMedications {
                // Parse reminder times
                // Simplified: assume one dose per day at 8 AM
                let doseTime = calendar.date(bySettingHour: 8, minute: 0, second: 0, of: today)!

                // Check if already logged
                let isTaken = false // Would check DoseLog

                scheduledDoses.append(ScheduledDose(
                    id: UUID(),
                    medicationName: medication.name,
                    dosage: medication.dosage,
                    unit: medication.unit,
                    scheduledTime: doseTime,
                    isTaken: isTaken,
                    isBiologic: medication.isBiologic
                ))
            }

            todaysDoses = scheduledDoses.sorted { $0.scheduledTime < $1.scheduledTime }

            // Load historical doses
            await loadLast30DaysDoses()

            // Calculate adherence
            calculateAdherence()
        }
    }

    private func loadLast30DaysDoses() async {
        do {
            let thirtyDaysAgo = Calendar.current.date(byAdding: .day, value: -30, to: Date())!

            let doses = try await context.perform {
                let request: NSFetchRequest<DoseLog> = DoseLog.fetchRequest()
                request.predicate = NSPredicate(format: "timestamp >= %@", thirtyDaysAgo as NSDate)
                request.sortDescriptors = [NSSortDescriptor(keyPath: \DoseLog.timestamp, ascending: false)]
                return try self.context.fetch(request)
            }

            last30DaysDoses = doses
        } catch {
            print("Error loading dose history: \(error)")
        }
    }

    private func calculateAdherence() {
        let calendar = Calendar.current

        // Weekly adherence
        let oneWeekAgo = calendar.date(byAdding: .day, value: -7, to: Date())!
        let weekDoses = last30DaysDoses.filter { ($0.timestamp ?? Date.distantPast) >= oneWeekAgo }

        if !weekDoses.isEmpty {
            let takenCount = weekDoses.filter { !$0.wasSkipped }.count
            weeklyAdherence = Double(takenCount) / Double(weekDoses.count) * 100
        }

        // Monthly adherence
        if !last30DaysDoses.isEmpty {
            let takenCount = last30DaysDoses.filter { !$0.wasSkipped }.count
            monthlyAdherence = Double(takenCount) / Double(last30DaysDoses.count) * 100
        }
    }

    // MARK: - Actions

    func addMedication(
        name: String,
        category: String,
        dosage: Double,
        unit: String,
        frequency: String,
        route: String,
        isBiologic: Bool,
        reminderEnabled: Bool,
        reminderTimes: [Date]
    ) async {
        await context.perform {
            let medication = Medication(context: self.context)
            medication.id = UUID()
            medication.name = name
            medication.category = category
            medication.dosage = dosage
            medication.dosageUnit = unit
            medication.frequency = frequency
            medication.route = route
            medication.isBiologic = isBiologic
            medication.isActive = true
            medication.reminderEnabled = reminderEnabled
            medication.startDate = Date()

            // Encode reminder times
            if let timesData = try? JSONEncoder().encode(reminderTimes) {
                medication.reminderTimes = timesData
            }

            do {
                try self.context.save()
            } catch {
                print("Error saving medication: \(error)")
            }
        }

        // Reload
        loadMedications()

        // Schedule notifications
        if reminderEnabled {
            scheduleReminders(name: name, times: reminderTimes)
        }
    }

    func markDoseTaken(_ dose: ScheduledDose) {
        Task {
            await context.perform {
                let doseLog = DoseLog(context: self.context)
                doseLog.id = UUID()
                doseLog.timestamp = Date()
                doseLog.scheduledTime = dose.scheduledTime
                doseLog.dosageTaken = dose.dosage
                doseLog.wasSkipped = false

                do {
                    try self.context.save()
                } catch {
                    print("Error saving dose log: \(error)")
                }
            }

            // Reload
            loadTodaysDoses()

            // Haptic feedback
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        }
    }

    func markDoseSkipped(_ dose: ScheduledDose) {
        Task {
            await context.perform {
                let doseLog = DoseLog(context: self.context)
                doseLog.id = UUID()
                doseLog.timestamp = Date()
                doseLog.scheduledTime = dose.scheduledTime
                doseLog.dosageTaken = 0
                doseLog.wasSkipped = true
                doseLog.skipReason = "Manually skipped"

                do {
                    try self.context.save()
                } catch {
                    print("Error saving skipped dose: \(error)")
                }
            }

            loadTodaysDoses()
        }
    }

    func toggleMedicationActive(_ medication: MedicationData) {
        Task {
            await context.perform {
                let request: NSFetchRequest<Medication> = Medication.fetchRequest()
                request.predicate = NSPredicate(format: "id == %@", medication.id as CVarArg)

                if let med = try? self.context.fetch(request).first {
                    med.isActive.toggle()
                    // FIXED: Proper error handling instead of silent try?
                    do {
                        try self.context.save()
                    } catch {
                        print("❌ CRITICAL: Failed to save medication toggle: \(error)")
                    }
                }
            }

            loadMedications()
            loadTodaysDoses()
        }
    }

    func deleteMedication(_ medication: MedicationData) {
        Task {
            await context.perform {
                let request: NSFetchRequest<Medication> = Medication.fetchRequest()
                request.predicate = NSPredicate(format: "id == %@", medication.id as CVarArg)

                if let med = try? self.context.fetch(request).first {
                    self.context.delete(med)
                    // FIXED: Proper error handling instead of silent try?
                    do {
                        try self.context.save()
                    } catch {
                        print("❌ CRITICAL: Failed to delete medication: \(error)")
                    }
                }
            }

            loadMedications()
        }
    }

    // MARK: - Notifications

    private func scheduleReminders(name: String, times: [Date]) {
        let center = UNUserNotificationCenter.current()

        for time in times {
            let content = UNMutableNotificationContent()
            content.title = "Medication Reminder"
            content.body = "Time to take \(name)"
            content.sound = .default
            content.categoryIdentifier = "MEDICATION_REMINDER"

            let components = Calendar.current.dateComponents([.hour, .minute], from: time)
            let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)

            let identifier = "medication_\(name)_\(components.hour ?? 0)_\(components.minute ?? 0)"
            let request = UNNotificationRequest(identifier: identifier, content: content, trigger: trigger)

            center.add(request) { error in
                if let error = error {
                    print("Error scheduling reminder: \(error)")
                }
            }
        }
    }
}

// MARK: - Supporting Models

struct MedicationData: Identifiable {
    let id: UUID
    let name: String
    let category: String?
    let dosage: Double
    let unit: String
    let frequency: String
    let route: String
    let isBiologic: Bool
    var isActive: Bool
    let reminderEnabled: Bool
}

struct ScheduledDose: Identifiable {
    let id: UUID
    let medicationName: String
    let dosage: Double
    let unit: String
    let scheduledTime: Date
    var isTaken: Bool
    let isBiologic: Bool
}
