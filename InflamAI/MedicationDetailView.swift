//
//  MedicationDetailView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData
import Foundation
import UIKit

struct MedicationDetailView: View {
    let medication: Medication
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss
    
    @State private var showingEditView = false
    @State private var showingDeleteAlert = false
    @State private var intakeHistory: [MedicationIntake] = []
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header Card
                    VStack(spacing: 15) {
                        HStack {
                            ZStack {
                                Circle()
                                    .fill(Color.blue)
                                    .frame(width: 60, height: 60)
                                
                                Image(systemName: "pills")
                                    .foregroundColor(.white)
                                    .font(.title)
                            }
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text(medication.name ?? "Unknown Medication")
                                    .font(.title2)
                                    .fontWeight(.bold)
                                
                                Text("\(Int(medication.dosage)) \(medication.unit ?? "mg")")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                
                                if let frequency = medication.frequency {
                                    Text(frequency)
                                        .font(.subheadline)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Color.blue.opacity(0.1))
                                        .foregroundColor(.blue)
                                        .cornerRadius(8)
                                }
                            }
                            
                            Spacer()
                        }
                        
                        if let instructions = medication.instructions, !instructions.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Instructions")
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                
                                Text(instructions)
                                    .font(.body)
                                    .foregroundColor(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding()
                    .background(Color(.systemBackground))
                    .cornerRadius(12)
                    .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
                    
                    // Schedule Information
                    VStack(alignment: .leading, spacing: 15) {
                        Text("Schedule")
                            .font(.headline)
                            .fontWeight(.semibold)
                        
                        VStack(spacing: 12) {
                            InfoRow(title: "Start Date", value: formatDate(medication.startDate))
                            
                            if let endDate = medication.endDate {
                                InfoRow(title: "End Date", value: formatDate(endDate))
                            }
                            
                            InfoRow(title: "Reminders", value: medication.reminderEnabled ? "Enabled" : "Disabled")
                            
                            if medication.reminderEnabled, let reminderTimes = getReminderTimes() {
                                InfoRow(title: "Reminder Times", value: reminderTimes.joined(separator: ", "))
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemBackground))
                    .cornerRadius(12)
                    .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
                    
                    // Recent Intake History
                    VStack(alignment: .leading, spacing: 15) {
                        HStack {
                            Text("Recent Intake History")
                                .font(.headline)
                                .fontWeight(.semibold)
                            
                            Spacer()
                            
                            Text("Last 7 days")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        if intakeHistory.isEmpty {
                            VStack(spacing: 10) {
                                Image(systemName: "clock")
                                    .font(.system(size: 30))
                                    .foregroundColor(.gray)
                                Text("No intake history yet")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 20)
                        } else {
                            LazyVStack(spacing: 8) {
                                ForEach(intakeHistory, id: \.objectID) { intake in
                                    IntakeHistoryRow(intake: intake)
                                }
                            }
                        }
                    }
                    .padding()
                    .background(Color(UIColor.systemBackground))
                    .cornerRadius(12)
                    .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
                }
                .padding()
            }
            .navigationTitle("Medication Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("Edit") {
                            showingEditView = true
                        }
                        
                        Button("Delete", role: .destructive) {
                            showingDeleteAlert = true
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
        .onAppear {
            loadIntakeHistory()
        }
        .sheet(isPresented: $showingEditView) {
            EditMedicationView(medication: medication)
        }
        .alert("Delete Medication", isPresented: $showingDeleteAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                deleteMedication()
            }
        } message: {
            Text("Are you sure you want to delete this medication? This action cannot be undone.")
        }
    }
    
    private func formatDate(_ date: Date?) -> String {
        guard let date = date else { return "Not set" }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
    
    private func getReminderTimes() -> [String]? {
        guard let reminderTimesString = medication.reminderTimes,
              let data = reminderTimesString.data(using: .utf8),
              let timeStrings = try? JSONSerialization.jsonObject(with: data) as? [String] else {
            return nil
        }
        return timeStrings
    }
    
    private func loadIntakeHistory() {
        let calendar = Calendar.current
        let sevenDaysAgo = calendar.date(byAdding: .day, value: -7, to: Date())!
        
        let request = NSFetchRequest<MedicationIntake>(entityName: "MedicationIntake")
        request.predicate = NSPredicate(format: "medicationName == %@ AND timestamp >= %@", medication.name ?? "", sevenDaysAgo as NSDate)
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        
        do {
            intakeHistory = try viewContext.fetch(request)
        } catch {
            print("Error loading intake history: \(error)")
        }
    }
    
    private func deleteMedication() {
        viewContext.delete(medication)
        
        do {
            try viewContext.save()
            dismiss()
        } catch {
            print("Error deleting medication: \(error)")
        }
    }
}

struct InfoRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
    }
}

struct IntakeHistoryRow: View {
    let intake: MedicationIntake
    
    var body: some View {
        HStack {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
                .font(.title3)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(formatDate(intake.timestamp))
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text("\(Int(intake.dosage)) mg")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(formatTime(intake.timestamp))
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
    
    private func formatDate(_ date: Date?) -> String {
        guard let date = date else { return "" }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
    
    private func formatTime(_ date: Date?) -> String {
        guard let date = date else { return "" }
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

struct EditMedicationView: View {
    let medication: Medication
    @Environment(\.dismiss) private var dismiss
    @Environment(\.managedObjectContext) private var viewContext
    @StateObject private var errorHandler = CoreDataErrorHandler.shared
    
    @State private var isLoading = false
    @State private var showingSuccessAlert = false
    
    @State private var name = ""
    @State private var dosage = ""
    @State private var unit = "mg"
    @State private var frequency = "Daily"
    @State private var instructions = ""
    @State private var startDate = Date()
    @State private var endDate: Date?
    @State private var hasEndDate = false
    @State private var reminderEnabled = true
    @State private var reminderTimes: [Date] = [Date()]
    
    private let units = ["mg", "ml", "tablets", "capsules", "drops", "puffs"]
    private let frequencies = ["Daily", "Twice Daily", "Three Times Daily", "Four Times Daily", "Weekly", "As Needed"]
    
    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section(header: Text("Medication Details")) {
                    TextField("Medication Name", text: $name)
                    
                    HStack {
                        TextField("Dosage", text: $dosage)
                        
                        Picker("Unit", selection: $unit) {
                            ForEach(units, id: \.self) { unit in
                                Text(unit).tag(unit)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                    }
                    
                    Picker("Frequency", selection: $frequency) {
                        ForEach(frequencies, id: \.self) { freq in
                            Text(freq).tag(freq)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    
                    TextField("Instructions (optional)", text: $instructions)
                        .lineLimit(6)
                }
                
                Section(header: Text("Schedule")) {
                    DatePicker("Start Date", selection: $startDate, displayedComponents: .date)
                    
                    Toggle("Set End Date", isOn: $hasEndDate)
                    
                    if hasEndDate {
                        DatePicker("End Date", selection: Binding(
                            get: { endDate ?? Date() },
                            set: { endDate = $0 }
                        ), displayedComponents: .date)
                    }
                }
                
                Section(header: Text("Reminders")) {
                    Toggle("Enable Reminders", isOn: $reminderEnabled)
                    
                    if reminderEnabled {
                        ForEach(reminderTimes.indices, id: \.self) { index in
                            HStack {
                                DatePicker("Reminder \(index + 1)", selection: $reminderTimes[index], displayedComponents: .hourAndMinute)
                                
                                if reminderTimes.count > 1 {
                                    Button("Remove") {
                                        reminderTimes.remove(at: index)
                                    }
                                    .foregroundColor(.red)
                                }
                            }
                        }
                        
                        if reminderTimes.count < 4 {
                            Button("Add Reminder Time") {
                                reminderTimes.append(Date())
                            }
                        }
                    }
                }
            }
            
            // Bottom Save Button
            VStack(spacing: 16) {
                Divider()
                
                Button(action: {
                    updateMedication()
                }) {
                    HStack {
                        if isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        }
                        Text(isLoading ? "Updating..." : "Update Medication")
                            .font(.headline)
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(name.isEmpty || isLoading ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .disabled(name.isEmpty || isLoading)
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
            }
            .background(Color(UIColor.systemGroupedBackground))
        }
        .navigationTitle("Edit Medication")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button("Cancel") {
                    dismiss()
                }
            }
            
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Save") {
                    updateMedication()
                }
                .disabled(name.isEmpty || isLoading)
            }
        }
        .onAppear {
            loadMedicationData()
        }
        .coreDataErrorAlert()
        .alert("Success", isPresented: $showingSuccessAlert) {
            Button("OK") {
                dismiss()
            }
        } message: {
            Text("Medication has been updated successfully.")
        }
    }
    
    private func loadMedicationData() {
        name = medication.name ?? ""
        dosage = String(medication.dosage)
        unit = medication.unit ?? "mg"
        frequency = medication.frequency ?? "Daily"
        instructions = medication.instructions ?? ""
        startDate = medication.startDate ?? Date()
        endDate = medication.endDate
        hasEndDate = endDate != nil
        reminderEnabled = medication.reminderEnabled
        
        // Load reminder times
        if let reminderTimesString = medication.reminderTimes,
           let data = reminderTimesString.data(using: .utf8),
           let timeStrings = try? JSONSerialization.jsonObject(with: data) as? [String] {
            
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            
            reminderTimes = timeStrings.compactMap { timeString in
                formatter.date(from: timeString)
            }
            
            if reminderTimes.isEmpty {
                reminderTimes = [Date()]
            }
        }
    }
    
    private func updateMedication() {
        // Validate input
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorHandler.handle(.invalidData("Medication name is required"))
            return
        }
        
        guard let dosageValue = Double(dosage), dosageValue > 0 else {
            errorHandler.handle(.invalidData("Please enter a valid dosage"))
            return
        }
        
        isLoading = true
        
        // Update medication properties
        medication.name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        medication.dosage = dosageValue
        medication.unit = unit
        medication.frequency = frequency
        medication.instructions = instructions.isEmpty ? nil : instructions
        medication.startDate = startDate
        medication.endDate = hasEndDate ? endDate : nil
        medication.reminderEnabled = reminderEnabled
        
        // Save reminder times as a JSON string
        if reminderEnabled {
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            let timeStrings = reminderTimes.map { formatter.string(from: $0) }
            if let jsonData = try? JSONSerialization.data(withJSONObject: timeStrings),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                medication.reminderTimes = jsonString
            }
        }
        
        // Save with proper error handling
        CoreDataOperations.safeSave(context: viewContext) { saveResult in
            DispatchQueue.main.async {
                self.isLoading = false
                
                switch saveResult {
                case .success:
                    // Remove old notifications and schedule new ones if reminders are enabled
                    self.removeOldNotifications()
                    if self.reminderEnabled {
                        self.scheduleNotifications()
                    }
                    self.showingSuccessAlert = true
                    
                case .failure:
                    // Error already handled by CoreDataOperations
                    break
                }
            }
        }
    }
    
    private func removeOldNotifications() {
        let identifier = medication.objectID.uriRepresentation().absoluteString
        UNUserNotificationCenter.current().getPendingNotificationRequests { requests in
            let idsToRemove = requests.filter { $0.identifier.hasPrefix(identifier) }.map { $0.identifier }
            UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: idsToRemove)
        }
    }
    
    private func scheduleNotifications() {
        let dosage = medication.dosage
        let unit = medication.unit ?? ""
        
        let content = UNMutableNotificationContent()
        content.title = "Medication Reminder"
        content.body = "Time to take \(name) - \(Int(dosage)) \(unit)"
        content.sound = .default
        
        for (index, time) in reminderTimes.enumerated() {
            let calendar = Calendar.current
            let components = calendar.dateComponents([.hour, .minute], from: time)
            
            let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)
            let identifier = "\(medication.objectID.uriRepresentation().absoluteString)_\(index)"
            let request = UNNotificationRequest(identifier: identifier, content: content, trigger: trigger)
            
            UNUserNotificationCenter.current().add(request) { error in
                if let error = error {
                    print("Error scheduling notification: \(error)")
                }
            }
        }
    }
}

struct MedicationDetailView_Previews: PreviewProvider {
    static var previews: some View {
        let context = InflamAIPersistenceController.preview.container.viewContext
        
        let medication = Medication(context: context)
        medication.name = "Sample Medication"
        medication.dosage = 10.0
        medication.frequency = "Daily"
        
        return MedicationDetailView(medication: medication)
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}