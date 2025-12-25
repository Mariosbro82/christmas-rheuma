//
//  QuestionnaireConfigView.swift
//  InflamAI-Swift
//
//  Individual questionnaire configuration screen
//  Configure schedule, notifications, and view questionnaire details
//

import SwiftUI

struct QuestionnaireConfigView: View {
    let questionnaireID: QuestionnaireID

    @StateObject private var preferences = QuestionnaireUserPreferences.shared
    @Environment(\.dismiss) private var dismiss

    @State private var isEnabled: Bool = false
    @State private var notificationsEnabled: Bool = true
    @State private var selectedFrequency: FrequencyOption = .default
    @State private var selectedTime: Date = Date()
    @State private var selectedWeekday: Int = 1 // Monday
    @State private var selectedMonthDay: Int = 1
    @State private var showingPreview = false

    enum FrequencyOption: String, CaseIterable, Identifiable {
        case `default` = "Default"
        case daily = "Daily"
        case weekly = "Weekly"
        case monthly = "Monthly"
        case onDemand = "On-Demand"

        var id: String { rawValue }
    }

    var body: some View {
        List {
            // Questionnaire Info Section
            Section {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text(NSLocalizedString(questionnaireID.titleKey, comment: ""))
                            .font(.title2)
                            .fontWeight(.bold)

                        if questionnaireID.isDefault {
                            Text("DEFAULT")
                                .font(.caption)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.blue)
                                .cornerRadius(6)
                        }
                    }

                    Text(NSLocalizedString(questionnaireID.descriptionKey, comment: ""))
                        .font(.body)
                        .foregroundColor(.secondary)

                    HStack(spacing: 16) {
                        Label(questionnaireID.category.rawValue, systemImage: questionnaireID.category.icon)
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if let definition = QuestionnaireDefinition.definition(for: questionnaireID) {
                            Label("\(definition.items.count) questions", systemImage: "list.bullet")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(.vertical, 8)
            }

            // Enable/Disable Toggle
            Section {
                Toggle(isOn: $isEnabled) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Enable Questionnaire")
                            .font(.headline)
                        Text("Include this questionnaire in your routine")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .tint(.blue)
            }

            // Schedule Configuration
            if isEnabled {
                Section(header: Text("Schedule")) {
                    Picker("Frequency", selection: $selectedFrequency) {
                        ForEach(FrequencyOption.allCases) { option in
                            Text(option.rawValue).tag(option)
                        }
                    }
                    .pickerStyle(.menu)

                    if selectedFrequency != .default && selectedFrequency != .onDemand {
                        DatePicker("Time", selection: $selectedTime, displayedComponents: .hourAndMinute)

                        switch selectedFrequency {
                        case .weekly:
                            Picker("Day of Week", selection: $selectedWeekday) {
                                Text("Monday").tag(1)
                                Text("Tuesday").tag(2)
                                Text("Wednesday").tag(3)
                                Text("Thursday").tag(4)
                                Text("Friday").tag(5)
                                Text("Saturday").tag(6)
                                Text("Sunday").tag(7)
                            }
                            .pickerStyle(.menu)

                        case .monthly:
                            Picker("Day of Month", selection: $selectedMonthDay) {
                                ForEach(1...28, id: \.self) { day in
                                    Text("\(day)").tag(day)
                                }
                            }
                            .pickerStyle(.menu)

                        default:
                            EmptyView()
                        }
                    }

                    if selectedFrequency == .default {
                        HStack {
                            Text("Using default schedule:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(defaultScheduleDescription)
                                .font(.caption)
                                .foregroundColor(.blue)
                        }
                    }
                }

                // Notifications
                Section(header: Text("Notifications")) {
                    Toggle(isOn: $notificationsEnabled) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Enable Reminders")
                                .font(.body)
                            Text("Get notified when it's time to complete this questionnaire")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .tint(.blue)
                }

                // Preview
                Section {
                    Button {
                        showingPreview = true
                    } label: {
                        HStack {
                            Image(systemName: "eye")
                            Text("Preview Questions")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }

            // Reset to Default
            if selectedFrequency != .default {
                Section {
                    Button("Reset to Default Schedule") {
                        resetToDefaults()
                    }
                    .foregroundColor(.blue)
                }
            }
        }
        .navigationTitle("Configure")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Done") {
                    saveChanges()
                    dismiss()
                }
                .fontWeight(.semibold)
            }
        }
        .onAppear {
            loadCurrentSettings()
        }
        .sheet(isPresented: $showingPreview) {
            if QuestionnaireDefinition.definition(for: questionnaireID) != nil {
                NavigationView {
                    QuestionnaireFormView(questionnaireID: questionnaireID)
                }
            } else {
                Text("Preview not available")
            }
        }
    }

    // MARK: - Helper Methods

    private func loadCurrentSettings() {
        isEnabled = preferences.isEnabled(questionnaireID)
        notificationsEnabled = preferences.areNotificationsEnabled(for: questionnaireID)

        let schedule = preferences.getSchedule(for: questionnaireID)
        let defaultSchedule = questionnaireID.defaultSchedule

        // Determine if using custom or default schedule
        if preferences.customSchedules[questionnaireID] == nil {
            selectedFrequency = .default
        } else {
            switch schedule.frequency {
            case .daily:
                selectedFrequency = .daily
            case .weekly:
                selectedFrequency = .weekly
            case .monthly:
                selectedFrequency = .monthly
            case .onDemand:
                selectedFrequency = .onDemand
            }
        }

        // Load time components
        switch schedule.frequency {
        case .daily(let time):
            if let hour = time.hour, let minute = time.minute {
                var components = DateComponents()
                components.hour = hour
                components.minute = minute
                selectedTime = Calendar.current.date(from: components) ?? Date()
            }
        case .weekly(let weekday, let time):
            selectedWeekday = weekday
            if let hour = time.hour, let minute = time.minute {
                var components = DateComponents()
                components.hour = hour
                components.minute = minute
                selectedTime = Calendar.current.date(from: components) ?? Date()
            }
        case .monthly(let day, let time):
            selectedMonthDay = day
            if let hour = time.hour, let minute = time.minute {
                var components = DateComponents()
                components.hour = hour
                components.minute = minute
                selectedTime = Calendar.current.date(from: components) ?? Date()
            }
        case .onDemand:
            break
        }
    }

    private func saveChanges() {
        // Update enabled state
        if isEnabled {
            preferences.enableQuestionnaire(questionnaireID)
        } else {
            preferences.disableQuestionnaire(questionnaireID)
        }

        // Update notifications
        preferences.setNotificationsEnabled(notificationsEnabled, for: questionnaireID)

        // Update schedule
        if selectedFrequency == .default {
            preferences.resetToDefaultSchedule(for: questionnaireID)
        } else {
            let components = Calendar.current.dateComponents([.hour, .minute], from: selectedTime)
            let timeComponents = DateComponents(hour: components.hour, minute: components.minute)

            let frequency: QuestionnaireSchedule.Frequency
            switch selectedFrequency {
            case .daily:
                frequency = .daily(time: timeComponents)
            case .weekly:
                frequency = .weekly(weekday: selectedWeekday, time: timeComponents)
            case .monthly:
                frequency = .monthly(day: selectedMonthDay, time: timeComponents)
            case .onDemand:
                frequency = .onDemand
            case .default:
                return // Already handled above
            }

            let schedule = QuestionnaireSchedule(
                frequency: frequency,
                windowHours: 4, // Default window
                timezoneIdentifier: TimeZone.current.identifier
            )
            preferences.setSchedule(schedule, for: questionnaireID)
        }
    }

    private func resetToDefaults() {
        selectedFrequency = .default
        preferences.resetToDefaultSchedule(for: questionnaireID)
    }

    private var defaultScheduleDescription: String {
        let schedule = questionnaireID.defaultSchedule
        switch schedule.frequency {
        case .daily(let time):
            return "Daily at \(formatTime(time))"
        case .weekly(let weekday, let time):
            return "\(weekdayName(weekday)) at \(formatTime(time))"
        case .monthly(let day, let time):
            return "Monthly on day \(day) at \(formatTime(time))"
        case .onDemand:
            return "On-Demand only"
        }
    }

    private func formatTime(_ components: DateComponents) -> String {
        guard let hour = components.hour, let minute = components.minute else {
            return "N/A"
        }
        return String(format: "%02d:%02d", hour, minute)
    }

    private func weekdayName(_ weekday: Int) -> String {
        switch weekday {
        case 1: return "Monday"
        case 2: return "Tuesday"
        case 3: return "Wednesday"
        case 4: return "Thursday"
        case 5: return "Friday"
        case 6: return "Saturday"
        case 7: return "Sunday"
        default: return "Unknown"
        }
    }
}

// MARK: - Preview

struct QuestionnaireConfigView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            QuestionnaireConfigView(questionnaireID: .basdai)
        }
    }
}
