//
//  AddMedicationView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import Foundation
import CoreData
import UserNotifications
// UIKit not available on macOS





struct AddMedicationViewLegacy: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss
    @Environment(\.presentationMode) var presentationMode
    @State private var showingErrorAlert = false
    @State private var errorMessage = ""
    
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
    @State private var selectedPredefinedMedication: PredefinedMedication?
    @State private var showingQuickSelect = true
    @State private var searchText = ""
    
    private let medicationsDB = StandardizedMedicationsDatabase.shared
    
    private let units = ["mg", "ml", "tablets", "capsules", "drops", "puffs"]
    private let frequencies = ["Daily", "Twice Daily", "Three Times Daily", "Four Times Daily", "Weekly", "As Needed"]
    
    var body: some View {
        NavigationView {
            formContent
        }
    }
    
    private var formContent: some View {
        VStack(spacing: 0) {
            Form {
                quickSelectSection
                medicationDetailsSection
                scheduleSection
                reminderSection
            }
            
            // Bottom Save Button
            VStack(spacing: 16) {
                Divider()
                
                Button(action: {
                    saveMedication()
                }) {
                    HStack {
                        if isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        }
                        Text(isLoading ? "Saving..." : "Save Medication")
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
            .background(Color.gray.opacity(0.1))
        }
        .navigationTitle("Add Medication")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                }
            }
            
            ToolbarItem(placement: .primaryAction) {
                Button("Save") {
                    saveMedication()
                }
                .disabled(name.isEmpty || dosage.isEmpty)
            }
        }
        .alert("Error", isPresented: $showingErrorAlert) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
        .alert("Success", isPresented: $showingSuccessAlert) {
            Button("OK") {
                dismiss()
            }
        } message: {
            Text("Medication has been added successfully.")
        }
    }
    
    private var quickSelectSection: some View {
        Group {
            if showingQuickSelect {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Quick Select")
                        .font(.headline)
                        .padding(.horizontal)

                    TextField("Search medications...", text: $searchText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    let filteredMedications = searchText.isEmpty ? 
                        medicationsDB.predefinedMedications : 
                        medicationsDB.searchMedications(query: searchText)
                    
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 8) {
                        ForEach(filteredMedications.prefix(8), id: \.id) { medication in
                            Button(action: {
                                selectPredefinedMedication(medication)
                            }) {
                                VStack(spacing: 4) {
                                    Text(medication.emoji)
                                        .font(.title2)
                                    Text(medication.name)
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .multilineTextAlignment(.center)
                                    Text(medication.category.rawValue.capitalized)
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 2)
                                        .background(Color.blue.opacity(0.1))
                                        .cornerRadius(8)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(8)
                                .background(selectedPredefinedMedication?.id == medication.id ? 
                                           Color.blue.opacity(0.1) : Color.gray.opacity(0.1))
                                .cornerRadius(8)
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }
    
    private var medicationDetailsSection: some View {
        Section(header: Text("Medication Details")) {
            TextField("Medication Name", text: $name)
            
            HStack {
                if let selected = selectedPredefinedMedication, !selected.standardDosages.isEmpty {
                    Picker("Dosage", selection: $dosage) {
                        ForEach(selected.standardDosages, id: \.self) { dose in
                            Text(String(dose)).tag(String(dose))
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                } else {
                    TextField("Dosage", text: $dosage)
                }
                
                Picker("Unit", selection: $unit) {
                    ForEach(units, id: \.self) { unit in
                        Text(unit).tag(unit)
                    }
                }
                .pickerStyle(MenuPickerStyle())
            }
            
            if let selected = selectedPredefinedMedication, !selected.standardFrequencies.isEmpty {
                Picker("Frequency", selection: $frequency) {
                    ForEach(selected.standardFrequencies, id: \.self) { freq in
                        Text(freq).tag(freq)
                    }
                }
                .pickerStyle(MenuPickerStyle())
            } else {
                Picker("Frequency", selection: $frequency) {
                    ForEach(frequencies, id: \.self) { freq in
                        Text(freq).tag(freq)
                    }
                }
                .pickerStyle(MenuPickerStyle())
            }
            
            TextField("Instructions (optional)", text: $instructions)
                .lineLimit(6)
        }
    }
    
    private var scheduleSection: some View {
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
    }
    
    private var reminderSection: some View {
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
    
    private func selectPredefinedMedication(_ medication: PredefinedMedication) {
        selectedPredefinedMedication = medication
        name = medication.name
        
        // Auto-populate with first standard dosage if available
        if !medication.standardDosages.isEmpty {
            dosage = medication.standardDosages[0]
        }
        
        // Auto-populate with first standard frequency if available
        if !medication.standardFrequencies.isEmpty {
            frequency = medication.standardFrequencies[0]
        }
        
        // Set appropriate unit based on medication type
        if medication.category == .biologic {
            unit = "mg" // Most biologics are dosed in mg
        } else {
            unit = "mg" // Default to mg for most medications
        }
    }
    
    private func saveMedication() {
        // Validate input
        guard !name.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty else {
            errorMessage = "Medication name is required"
            showingErrorAlert = true
            return
        }

        guard let dosageValue = Double(dosage), dosageValue > 0 else {
            errorMessage = "Please enter a valid dosage"
            showingErrorAlert = true
            return
        }

        isLoading = true

        // Create medication using Core Data
        let newMedication = Medication(context: viewContext)
        newMedication.id = UUID()
        newMedication.name = name
        newMedication.dosage = dosageValue
        newMedication.unit = unit
        newMedication.dosageUnit = unit // For compatibility
        newMedication.frequency = frequency
        newMedication.category = selectedPredefinedMedication?.category.rawValue ?? "other"
        newMedication.startDate = startDate
        newMedication.endDate = hasEndDate ? endDate : nil
        newMedication.isActive = true
        newMedication.reminderEnabled = reminderEnabled

        // Save reminder times as binary data
        if reminderEnabled {
            if let encoded = try? JSONEncoder().encode(reminderTimes) {
                newMedication.reminderTimes = encoded
            }

            // Schedule notifications
            scheduleNotifications(for: name)
        }

        // Save to Core Data
        do {
            try viewContext.save()
            isLoading = false
            showingSuccessAlert = true
        } catch {
            isLoading = false
            errorMessage = "Failed to save medication: \(error.localizedDescription)"
            showingErrorAlert = true
        }
    }
    
    private func scheduleNotifications(for medicationName: String) {
        // Implementation for scheduling notifications would go here
        // This would use UserNotifications framework
        print("Scheduling notifications for: \(medicationName)")
    }
}

#if DEBUG
struct AddMedicationViewLegacy_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            AddMedicationView()
        }
    }
}
#endif