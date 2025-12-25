//
//  EmergencyView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import CoreLocation
import Contacts

struct EmergencyView: View {
    @StateObject private var emergencyManager = EmergencyManager.shared
    @State private var showingAddContact = false
    @State private var showingMedicalInfo = false
    @State private var showingSettings = false
    @State private var showingEmergencyServices = false
    @State private var showingContactImport = false
    @State private var showingEmergencyConfirmation = false
    @State private var selectedContact: EmergencyContact?
    @State private var importedContacts: [EmergencyContact] = []
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Emergency Status Card
                    emergencyStatusCard
                    
                    // Quick Actions
                    quickActionsSection
                    
                    // Emergency Contacts
                    emergencyContactsSection
                    
                    // Medical Information
                    medicalInformationSection
                    
                    // Emergency Services
                    emergencyServicesSection
                    
                    // Emergency History
                    emergencyHistorySection
                }
                .padding()
            }
            .navigationTitle("Emergency")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Settings") {
                        showingSettings = true
                    }
                }
            }
        }
        .sheet(isPresented: $showingAddContact) {
            AddEmergencyContactView(emergencyManager: emergencyManager)
        }
        .sheet(isPresented: $showingMedicalInfo) {
            MedicalInformationView(emergencyManager: emergencyManager)
        }
        .sheet(isPresented: $showingSettings) {
            EmergencySettingsView(emergencyManager: emergencyManager)
        }
        .sheet(isPresented: $showingEmergencyServices) {
            EmergencyServicesView(emergencyManager: emergencyManager)
        }
        .sheet(isPresented: $showingContactImport) {
            ContactImportView(importedContacts: $importedContacts, emergencyManager: emergencyManager)
        }
        .alert("Emergency Activation", isPresented: $showingEmergencyConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Activate Emergency", role: .destructive) {
                emergencyManager.activateEmergencyMode()
            }
        } message: {
            Text("This will notify your emergency contacts and activate emergency protocols. Are you sure?")
        }
        .onAppear {
            emergencyManager.requestLocationPermission()
        }
    }
    
    // MARK: - Emergency Status Card
    private var emergencyStatusCard: some View {
        VStack(spacing: 16) {
            HStack {
                Image(systemName: emergencyManager.isEmergencyModeActive ? "exclamationmark.triangle.fill" : "shield.checkered")
                    .font(.title2)
                    .foregroundColor(emergencyManager.isEmergencyModeActive ? .red : .green)
                
                VStack(alignment: .leading) {
                    Text(emergencyManager.isEmergencyModeActive ? "Emergency Mode Active" : "Emergency System Ready")
                        .font(.headline)
                        .foregroundColor(emergencyManager.isEmergencyModeActive ? .red : .primary)
                    
                    if emergencyManager.isEmergencyModeActive {
                        if let lastAlert = emergencyManager.lastEmergencyAlert {
                            Text("Activated: \(lastAlert.formatted(date: .omitted, time: .shortened))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        Text("\(emergencyManager.emergencyContacts.count) contacts configured")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                if emergencyManager.isEmergencyModeActive {
                    Button("Resolve") {
                        emergencyManager.resolveEmergency(reason: "Manually resolved by user")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                    .controlSize(.small)
                }
            }
            
            if !emergencyManager.isLocationServicesEnabled {
                HStack {
                    Image(systemName: "location.slash")
                        .foregroundColor(.orange)
                    Text("Location services disabled. Enable for better emergency response.")
                        .font(.caption)
                        .foregroundColor(.orange)
                    Spacer()
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    // MARK: - Quick Actions
    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Actions")
                .font(.headline)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                QuickActionButton(
                    title: "Emergency",
                    subtitle: "Activate now",
                    icon: "exclamationmark.triangle.fill",
                    color: .red
                ) {
                    showingEmergencyConfirmation = true
                }
                
                QuickActionButton(
                    title: "Call 911",
                    subtitle: "Emergency services",
                    icon: "phone.fill",
                    color: .red
                ) {
                    callEmergencyServices()
                }
                
                QuickActionButton(
                    title: "Find Services",
                    subtitle: "Nearby emergency",
                    icon: "location.fill",
                    color: .blue
                ) {
                    showingEmergencyServices = true
                }
                
                QuickActionButton(
                    title: "Medical ID",
                    subtitle: "View card",
                    icon: "cross.fill",
                    color: .green
                ) {
                    showMedicalInformationCard()
                }
            }
        }
    }
    
    // MARK: - Emergency Contacts
    private var emergencyContactsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Emergency Contacts")
                    .font(.headline)
                
                Spacer()
                
                Menu {
                    Button("Add Contact") {
                        showingAddContact = true
                    }
                    
                    Button("Import from Contacts") {
                        importContactsFromAddressBook()
                    }
                } label: {
                    Image(systemName: "plus.circle.fill")
                        .font(.title2)
                        .foregroundColor(.blue)
                }
            }
            
            if emergencyManager.emergencyContacts.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "person.2.slash")
                        .font(.largeTitle)
                        .foregroundColor(.gray)
                    
                    Text("No Emergency Contacts")
                        .font(.headline)
                        .foregroundColor(.gray)
                    
                    Text("Add contacts who should be notified in case of emergency")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    
                    Button("Add First Contact") {
                        showingAddContact = true
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                    .controlSize(.small)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 20)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(emergencyManager.emergencyContacts) { contact in
                        EmergencyContactRow(contact: contact) {
                            selectedContact = contact
                            showingAddContact = true
                        } onDelete: {
                            emergencyManager.removeEmergencyContact(contact)
                        } onCall: {
                            callContact(contact)
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Medical Information
    private var medicalInformationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Medical Information")
                    .font(.headline)
                
                Spacer()
                
                Button("Edit") {
                    showingMedicalInfo = true
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                if !emergencyManager.medicalInformation.firstName.isEmpty {
                    HStack {
                        Text("Name:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("\(emergencyManager.medicalInformation.firstName) \(emergencyManager.medicalInformation.lastName)")
                            .font(.caption)
                    }
                }
                
                if !emergencyManager.medicalInformation.bloodType.isEmpty {
                    HStack {
                        Text("Blood Type:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(emergencyManager.medicalInformation.bloodType)
                            .font(.caption)
                    }
                }
                
                if !emergencyManager.medicalInformation.allergies.isEmpty {
                    HStack {
                        Text("Allergies:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(emergencyManager.medicalInformation.allergies.joined(separator: ", "))
                            .font(.caption)
                    }
                }
                
                if !emergencyManager.medicalInformation.medications.isEmpty {
                    HStack {
                        Text("Medications:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(emergencyManager.medicalInformation.medications.joined(separator: ", "))
                            .font(.caption)
                    }
                }
                
                if emergencyManager.medicalInformation.firstName.isEmpty &&
                   emergencyManager.medicalInformation.bloodType.isEmpty &&
                   emergencyManager.medicalInformation.allergies.isEmpty &&
                   emergencyManager.medicalInformation.medications.isEmpty {
                    Text("No medical information configured")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
    }
    
    // MARK: - Emergency Services
    private var emergencyServicesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Nearby Emergency Services")
                    .font(.headline)
                
                Spacer()
                
                Button("View All") {
                    showingEmergencyServices = true
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
            
            if emergencyManager.nearbyEmergencyServices.isEmpty {
                Text("No emergency services found nearby")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(emergencyManager.nearbyEmergencyServices.prefix(3)) { service in
                        EmergencyServiceRow(service: service) {
                            emergencyManager.callEmergencyService(service)
                        } onDirections: {
                            emergencyManager.getDirectionsToEmergencyService(service)
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Emergency History
    private var emergencyHistorySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Emergency Events")
                .font(.headline)
            
            if emergencyManager.emergencyHistory.isEmpty {
                Text("No emergency events recorded")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(emergencyManager.emergencyHistory.suffix(5).reversed(), id: \.id) { event in
                        EmergencyEventRow(event: event)
                    }
                }
            }
        }
    }
    
    // MARK: - Actions
    private func callEmergencyServices() {
        guard let url = URL(string: "tel://911") else { return }
        
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
        }
    }
    
    private func callContact(_ contact: EmergencyContact) {
        let phoneNumber = contact.phoneNumber.replacingOccurrences(of: " ", with: "")
        guard let url = URL(string: "tel://\(phoneNumber)") else { return }
        
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
        }
    }
    
    private func showMedicalInformationCard() {
        let medicalCard = emergencyManager.generateMedicalInformationCard()
        
        // In a real app, this could show a modal with the medical card
        print("Medical Information Card: \(medicalCard)")
    }
    
    private func importContactsFromAddressBook() {
        Task {
            let contacts = await emergencyManager.importContactsFromAddressBook()
            await MainActor.run {
                importedContacts = contacts
                showingContactImport = true
            }
        }
    }
}

// MARK: - Quick Action Button
struct QuickActionButton: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(color)
                
                VStack(spacing: 2) {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text(subtitle)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Emergency Contact Row
struct EmergencyContactRow: View {
    let contact: EmergencyContact
    let onEdit: () -> Void
    let onDelete: () -> Void
    let onCall: () -> Void
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(contact.name)
                        .font(.headline)
                    
                    if contact.isPrimary {
                        Text("PRIMARY")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(4)
                    }
                }
                
                Text(contact.relationship.displayName)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(contact.phoneNumber)
                    .font(.caption)
                    .foregroundColor(.blue)
            }
            
            Spacer()
            
            HStack(spacing: 12) {
                Button(action: onCall) {
                    Image(systemName: "phone.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                }
                
                Button(action: onEdit) {
                    Image(systemName: "pencil")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
                
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundColor(.red)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Emergency Service Row
struct EmergencyServiceRow: View {
    let service: EmergencyService
    let onCall: () -> Void
    let onDirections: () -> Void
    
    var body: some View {
        HStack {
            Image(systemName: service.type.icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(service.name)
                    .font(.headline)
                
                Text(service.type.displayName)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack {
                    Text(String(format: "%.1f km", service.distance))
                        .font(.caption)
                        .foregroundColor(.blue)
                    
                    if service.isOpen24Hours {
                        Text("• 24/7")
                            .font(.caption)
                            .foregroundColor(.green)
                    }
                }
            }
            
            Spacer()
            
            HStack(spacing: 12) {
                Button(action: onCall) {
                    Image(systemName: "phone.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                }
                
                Button(action: onDirections) {
                    Image(systemName: "location.fill")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Emergency Event Row
struct EmergencyEventRow: View {
    let event: EmergencyEvent
    
    var body: some View {
        HStack {
            Image(systemName: event.wasAutoDetected ? "exclamationmark.triangle.fill" : "hand.raised.fill")
                .font(.title3)
                .foregroundColor(event.wasResolved ? .green : .red)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(event.reason.description)
                    .font(.headline)
                    .lineLimit(2)
                
                Text(event.timestamp.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack {
                    Text(event.wasAutoDetected ? "Auto-detected" : "Manual")
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(event.wasAutoDetected ? Color.orange : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(4)
                    
                    Text(event.wasResolved ? "Resolved" : "Active")
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(event.wasResolved ? Color.green : Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(4)
                }
            }
            
            Spacer()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Add Emergency Contact View
struct AddEmergencyContactView: View {
    @ObservedObject var emergencyManager: EmergencyManager
    @Environment(\.dismiss) private var dismiss
    
    @State private var name = ""
    @State private var phoneNumber = ""
    @State private var email = ""
    @State private var relationship = ContactRelationship.other
    @State private var isPrimary = false
    @State private var canReceiveSMS = true
    @State private var canReceiveCall = true
    @State private var notes = ""
    
    var body: some View {
        NavigationView {
            Form {
                Section("Contact Information") {
                    TextField("Name", text: $name)
                    TextField("Phone Number", text: $phoneNumber)
                        .keyboardType(.phonePad)
                    TextField("Email (Optional)", text: $email)
                        .keyboardType(.emailAddress)
                }
                
                Section("Relationship") {
                    Picker("Relationship", selection: $relationship) {
                        ForEach(ContactRelationship.allCases, id: \.self) { relationship in
                            Text(relationship.displayName).tag(relationship)
                        }
                    }
                    .pickerStyle(.menu)
                }
                
                Section("Settings") {
                    Toggle("Primary Contact", isOn: $isPrimary)
                    Toggle("Can Receive SMS", isOn: $canReceiveSMS)
                    Toggle("Can Receive Calls", isOn: $canReceiveCall)
                }
                
                Section("Notes") {
                    TextField("Additional notes", text: $notes, axis: .vertical)
                        .lineLimit(3...6)
                }
            }
            .navigationTitle("Add Emergency Contact")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveContact()
                    }
                    .disabled(name.isEmpty || phoneNumber.isEmpty)
                }
            }
        }
    }
    
    private func saveContact() {
        let contact = EmergencyContact(
            id: UUID(),
            name: name,
            phoneNumber: phoneNumber,
            email: email.isEmpty ? nil : email,
            relationship: relationship,
            isPrimary: isPrimary,
            canReceiveSMS: canReceiveSMS,
            canReceiveCall: canReceiveCall,
            notes: notes
        )
        
        emergencyManager.addEmergencyContact(contact)
        dismiss()
    }
}

// MARK: - Medical Information View
struct MedicalInformationView: View {
    @ObservedObject var emergencyManager: EmergencyManager
    @Environment(\.dismiss) private var dismiss
    
    @State private var firstName: String
    @State private var lastName: String
    @State private var dateOfBirth: Date?
    @State private var bloodType: String
    @State private var allergies: String
    @State private var medications: String
    @State private var medicalConditions: String
    @State private var emergencyNotes: String
    @State private var insuranceInformation: String
    @State private var preferredHospital: String
    @State private var primaryPhysician: String
    
    init(emergencyManager: EmergencyManager) {
        self.emergencyManager = emergencyManager
        let medInfo = emergencyManager.medicalInformation
        
        _firstName = State(initialValue: medInfo.firstName)
        _lastName = State(initialValue: medInfo.lastName)
        _dateOfBirth = State(initialValue: medInfo.dateOfBirth)
        _bloodType = State(initialValue: medInfo.bloodType)
        _allergies = State(initialValue: medInfo.allergies.joined(separator: ", "))
        _medications = State(initialValue: medInfo.medications.joined(separator: ", "))
        _medicalConditions = State(initialValue: medInfo.medicalConditions.joined(separator: ", "))
        _emergencyNotes = State(initialValue: medInfo.emergencyNotes)
        _insuranceInformation = State(initialValue: medInfo.insuranceInformation)
        _preferredHospital = State(initialValue: medInfo.preferredHospital)
        _primaryPhysician = State(initialValue: medInfo.primaryPhysician)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section("Personal Information") {
                    TextField("First Name", text: $firstName)
                    TextField("Last Name", text: $lastName)
                    
                    DatePicker("Date of Birth", selection: Binding(
                        get: { dateOfBirth ?? Date() },
                        set: { dateOfBirth = $0 }
                    ), displayedComponents: .date)
                    
                    TextField("Blood Type", text: $bloodType)
                        .textInputAutocapitalization(.characters)
                }
                
                Section("Medical Information") {
                    TextField("Allergies (comma separated)", text: $allergies, axis: .vertical)
                        .lineLimit(2...4)
                    
                    TextField("Current Medications (comma separated)", text: $medications, axis: .vertical)
                        .lineLimit(2...4)
                    
                    TextField("Medical Conditions (comma separated)", text: $medicalConditions, axis: .vertical)
                        .lineLimit(2...4)
                }
                
                Section("Emergency Information") {
                    TextField("Emergency Notes", text: $emergencyNotes, axis: .vertical)
                        .lineLimit(3...6)
                    
                    TextField("Insurance Information", text: $insuranceInformation)
                    TextField("Preferred Hospital", text: $preferredHospital)
                    TextField("Primary Physician", text: $primaryPhysician)
                }
            }
            .navigationTitle("Medical Information")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveMedicalInformation()
                    }
                }
            }
        }
    }
    
    private func saveMedicalInformation() {
        var medicalInfo = emergencyManager.medicalInformation
        
        medicalInfo.firstName = firstName
        medicalInfo.lastName = lastName
        medicalInfo.dateOfBirth = dateOfBirth
        medicalInfo.bloodType = bloodType
        medicalInfo.allergies = allergies.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        medicalInfo.medications = medications.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        medicalInfo.medicalConditions = medicalConditions.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        medicalInfo.emergencyNotes = emergencyNotes
        medicalInfo.insuranceInformation = insuranceInformation
        medicalInfo.preferredHospital = preferredHospital
        medicalInfo.primaryPhysician = primaryPhysician
        
        emergencyManager.updateMedicalInformation(medicalInfo)
        dismiss()
    }
}

// MARK: - Emergency Settings View
struct EmergencySettingsView: View {
    @ObservedObject var emergencyManager: EmergencyManager
    @Environment(\.dismiss) private var dismiss
    
    @State private var settings: EmergencySettings
    
    init(emergencyManager: EmergencyManager) {
        self.emergencyManager = emergencyManager
        _settings = State(initialValue: emergencyManager.emergencySettings)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section("Automatic Detection") {
                    Toggle("Enable Automatic Detection", isOn: $settings.enableAutomaticDetection)
                    Toggle("Require Confirmation", isOn: $settings.requireConfirmation)
                    
                    if settings.requireConfirmation {
                        Stepper("Confirmation Timeout: \(settings.confirmationTimeoutSeconds)s", value: $settings.confirmationTimeoutSeconds, in: 30...300, step: 30)
                    }
                }
                
                Section("Emergency Response") {
                    Toggle("Enable Automatic Calling", isOn: $settings.enableAutomaticCalling)
                    Toggle("Share Location in Emergency", isOn: $settings.shareLocationInEmergency)
                    
                    Stepper("Auto-resolve Timeout: \(settings.autoResolveTimeoutMinutes) min", value: $settings.autoResolveTimeoutMinutes, in: 30...480, step: 30)
                }
                
                Section("Monitoring") {
                    Toggle("Monitor Vital Signs", isOn: $settings.monitorVitalSigns)
                    Toggle("Severe Pain Detection", isOn: $settings.enableSeverePainDetection)
                    Toggle("Prolonged Pain Detection", isOn: $settings.enableProlongedPainDetection)
                    
                    if settings.enableSeverePainDetection {
                        Stepper("Pain Threshold: \(settings.painThreshold, specifier: "%.1f")/10", value: $settings.painThreshold, in: 6.0...10.0, step: 0.5)
                    }
                }
            }
            .navigationTitle("Emergency Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        emergencyManager.updateEmergencySettings(settings)
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Emergency Services View
struct EmergencyServicesView: View {
    @ObservedObject var emergencyManager: EmergencyManager
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                ForEach(emergencyManager.nearbyEmergencyServices) { service in
                    EmergencyServiceRow(service: service) {
                        emergencyManager.callEmergencyService(service)
                    } onDirections: {
                        emergencyManager.getDirectionsToEmergencyService(service)
                    }
                }
            }
            .navigationTitle("Emergency Services")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Contact Import View
struct ContactImportView: View {
    @Binding var importedContacts: [EmergencyContact]
    @ObservedObject var emergencyManager: EmergencyManager
    @Environment(\.dismiss) private var dismiss
    
    @State private var selectedContacts: Set<UUID> = []
    
    var body: some View {
        NavigationView {
            List {
                ForEach(importedContacts) { contact in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(contact.name)
                                .font(.headline)
                            Text(contact.phoneNumber)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        if selectedContacts.contains(contact.id) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.blue)
                        } else {
                            Image(systemName: "circle")
                                .foregroundColor(.gray)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        if selectedContacts.contains(contact.id) {
                            selectedContacts.remove(contact.id)
                        } else {
                            selectedContacts.insert(contact.id)
                        }
                    }
                }
            }
            .navigationTitle("Import Contacts")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Import") {
                        importSelectedContacts()
                    }
                    .disabled(selectedContacts.isEmpty)
                }
            }
        }
    }
    
    private func importSelectedContacts() {
        let contactsToImport = importedContacts.filter { selectedContacts.contains($0.id) }
        
        for contact in contactsToImport {
            emergencyManager.addEmergencyContact(contact)
        }
        
        dismiss()
    }
}

#Preview {
    EmergencyView()
}