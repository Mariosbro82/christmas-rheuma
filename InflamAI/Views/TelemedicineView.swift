//
//  TelemedicineView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import AVFoundation
import CallKit

struct TelemedicineView: View {
    @StateObject private var telemedicineManager = TelemedicineManager()
    @State private var selectedTab = 0
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // Providers Tab
                ProvidersView()
                    .environmentObject(telemedicineManager)
                    .tabItem {
                        Image(systemName: "person.2.fill")
                        Text("Providers")
                    }
                    .tag(0)
                
                // Appointments Tab
                AppointmentsView()
                    .environmentObject(telemedicineManager)
                    .tabItem {
                        Image(systemName: "calendar")
                        Text("Appointments")
                    }
                    .tag(1)
                
                // Health Data Tab
                HealthDataView()
                    .environmentObject(telemedicineManager)
                    .tabItem {
                        Image(systemName: "heart.text.square")
                        Text("Health Data")
                    }
                    .tag(2)
                
                // Prescriptions Tab
                PrescriptionsView()
                    .environmentObject(telemedicineManager)
                    .tabItem {
                        Image(systemName: "pills")
                        Text("Prescriptions")
                    }
                    .tag(3)
            }
            .navigationTitle("Telemedicine")
        }
        .fullScreenCover(isPresented: $telemedicineManager.isInCall) {
            VideoCallView()
                .environmentObject(telemedicineManager)
        }
    }
}

// MARK: - Providers View

struct ProvidersView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    @State private var searchText = ""
    @State private var selectedSpecialty: MedicalSpecialty?
    @State private var showingFilters = false
    @State private var showingProviderDetail = false
    @State private var selectedProvider: HealthcareProvider?
    
    var filteredProviders: [HealthcareProvider] {
        telemedicineManager.providers.filter { provider in
            (searchText.isEmpty || provider.name.localizedCaseInsensitiveContains(searchText)) &&
            (selectedSpecialty == nil || provider.specialty == selectedSpecialty)
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Search and Filter Bar
            VStack(spacing: 12) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.gray)
                    
                    TextField("Search providers...", text: $searchText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    Button(action: { showingFilters.toggle() }) {
                        Image(systemName: "slider.horizontal.3")
                            .foregroundColor(.blue)
                    }
                }
                .padding(.horizontal)
                
                if selectedSpecialty != nil {
                    HStack {
                        Text("Filtered by: \(selectedSpecialty?.rawValue ?? "")")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Button("Clear") {
                            selectedSpecialty = nil
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                    .padding(.horizontal)
                }
            }
            .padding(.vertical)
            .background(Color(.systemGray6))
            
            // Providers List
            if telemedicineManager.isLoading {
                ProgressView("Loading providers...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if filteredProviders.isEmpty {
                EmptyProvidersView()
            } else {
                List(filteredProviders, id: \.id) { provider in
                    ProviderRowView(provider: provider) {
                        selectedProvider = provider
                        showingProviderDetail = true
                    }
                }
                .listStyle(PlainListStyle())
            }
        }
        .sheet(isPresented: $showingFilters) {
            ProviderFiltersView(selectedSpecialty: $selectedSpecialty)
        }
        .sheet(isPresented: $showingProviderDetail) {
            if let provider = selectedProvider {
                ProviderDetailView(provider: provider)
                    .environmentObject(telemedicineManager)
            }
        }
        .onAppear {
            telemedicineManager.searchProviders()
        }
    }
}

struct ProviderRowView: View {
    let provider: HealthcareProvider
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                // Provider Image
                AsyncImage(url: provider.profileImageURL) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } placeholder: {
                    Image(systemName: "person.circle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.gray)
                }
                .frame(width: 60, height: 60)
                .clipShape(Circle())
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(provider.name)
                            .font(.headline)
                            .foregroundColor(.primary)
                        
                        if provider.isVerified {
                            Image(systemName: "checkmark.seal.fill")
                                .foregroundColor(.blue)
                                .font(.caption)
                        }
                        
                        Spacer()
                        
                        Circle()
                            .fill(provider.isOnline ? Color.green : Color.gray)
                            .frame(width: 8, height: 8)
                    }
                    
                    Text(provider.specialty.rawValue)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    HStack {
                        HStack(spacing: 2) {
                            Image(systemName: "star.fill")
                                .foregroundColor(.yellow)
                                .font(.caption)
                            Text(String(format: "%.1f", provider.rating))
                                .font(.caption)
                            Text("(\(provider.reviewCount))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Text("$\(provider.consultationFee.amount, specifier: "%.0f")")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.green)
                    }
                }
                
                Image(systemName: "chevron.right")
                    .foregroundColor(.gray)
                    .font(.caption)
            }
            .padding(.vertical, 8)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct ProviderDetailView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    @Environment(\.presentationMode) var presentationMode
    let provider: HealthcareProvider
    @State private var showingBooking = false
    @State private var selectedDate = Date()
    @State private var availableSlots: [Date] = []
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Provider Header
                    VStack(spacing: 16) {
                        AsyncImage(url: provider.profileImageURL) { image in
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        } placeholder: {
                            Image(systemName: "person.circle.fill")
                                .font(.system(size: 80))
                                .foregroundColor(.gray)
                        }
                        .frame(width: 100, height: 100)
                        .clipShape(Circle())
                        
                        VStack(spacing: 4) {
                            HStack {
                                Text(provider.name)
                                    .font(.title2)
                                    .fontWeight(.bold)
                                
                                if provider.isVerified {
                                    Image(systemName: "checkmark.seal.fill")
                                        .foregroundColor(.blue)
                                }
                            }
                            
                            Text(provider.title)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            
                            Text(provider.specialty.rawValue)
                                .font(.subheadline)
                                .foregroundColor(.blue)
                        }
                        
                        HStack(spacing: 20) {
                            VStack {
                                HStack(spacing: 2) {
                                    Image(systemName: "star.fill")
                                        .foregroundColor(.yellow)
                                    Text(String(format: "%.1f", provider.rating))
                                        .fontWeight(.semibold)
                                }
                                Text("\(provider.reviewCount) reviews")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            VStack {
                                Text("\(provider.experience) years")
                                    .fontWeight(.semibold)
                                Text("Experience")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            VStack {
                                Text("$\(provider.consultationFee.amount, specifier: "%.0f")")
                                    .fontWeight(.semibold)
                                    .foregroundColor(.green)
                                Text("Per session")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Bio Section
                    VStack(alignment: .leading, spacing: 8) {
                        Text("About")
                            .font(.headline)
                        
                        Text(provider.bio)
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    
                    // Credentials Section
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Credentials")
                            .font(.headline)
                        
                        ForEach(provider.credentials, id: \.self) { credential in
                            HStack {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text(credential)
                                    .font(.body)
                                Spacer()
                            }
                        }
                    }
                    
                    // Languages Section
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Languages")
                            .font(.headline)
                        
                        HStack {
                            ForEach(provider.languages, id: \.self) { language in
                                Text(language)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(Color.blue.opacity(0.1))
                                    .foregroundColor(.blue)
                                    .cornerRadius(8)
                            }
                            Spacer()
                        }
                    }
                    
                    // Insurance Section
                    if provider.acceptsInsurance {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Insurance")
                                .font(.headline)
                            
                            Text("Accepts insurance with $\(provider.consultationFee.insuranceCopay ?? 0, specifier: "%.0f") copay")
                                .font(.body)
                                .foregroundColor(.secondary)
                            
                            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 8) {
                                ForEach(provider.insuranceNetworks, id: \.self) { network in
                                    Text(network)
                                        .font(.caption)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Color.green.opacity(0.1))
                                        .foregroundColor(.green)
                                        .cornerRadius(6)
                                }
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Provider Details")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Close") {
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Book Appointment") {
                    showingBooking = true
                }
                .fontWeight(.semibold)
            )
        }
        .sheet(isPresented: $showingBooking) {
            BookAppointmentView(provider: provider)
                .environmentObject(telemedicineManager)
        }
    }
}

struct BookAppointmentView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    @Environment(\.presentationMode) var presentationMode
    let provider: HealthcareProvider
    
    @State private var selectedDate = Date()
    @State private var selectedTime: Date?
    @State private var consultationType: Consultation.ConsultationType = .videoCall
    @State private var reason = ""
    @State private var symptoms: [String] = []
    @State private var newSymptom = ""
    @State private var urgency: Consultation.UrgencyLevel = .medium
    @State private var availableSlots: [Date] = []
    @State private var isBooking = false
    @State private var showingSuccess = false
    
    var body: some View {
        NavigationView {
            Form {
                Section("Appointment Type") {
                    Picker("Type", selection: $consultationType) {
                        ForEach(Consultation.ConsultationType.allCases, id: \.self) { type in
                            HStack {
                                Image(systemName: type.icon)
                                Text(type.rawValue)
                            }
                            .tag(type)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
                
                Section("Date & Time") {
                    DatePicker("Date", selection: $selectedDate, in: Date()..., displayedComponents: .date)
                        .onChange(of: selectedDate) { _ in
                            loadAvailableSlots()
                        }
                    
                    if !availableSlots.isEmpty {
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 8) {
                            ForEach(availableSlots, id: \.self) { slot in
                                Button(action: { selectedTime = slot }) {
                                    Text(DateFormatter.timeFormatter.string(from: slot))
                                        .font(.caption)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 8)
                                        .background(selectedTime == slot ? Color.blue : Color(.systemGray5))
                                        .foregroundColor(selectedTime == slot ? .white : .primary)
                                        .cornerRadius(8)
                                }
                            }
                        }
                    } else {
                        Text("No available slots for this date")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Reason for Visit") {
                    TextField("Describe your symptoms or concerns", text: $reason, axis: .vertical)
                        .lineLimit(3...6)
                }
                
                Section("Symptoms") {
                    ForEach(symptoms, id: \.self) { symptom in
                        HStack {
                            Text(symptom)
                            Spacer()
                            Button(action: { removeSymptom(symptom) }) {
                                Image(systemName: "minus.circle.fill")
                                    .foregroundColor(.red)
                            }
                        }
                    }
                    
                    HStack {
                        TextField("Add symptom", text: $newSymptom)
                        Button("Add") {
                            addSymptom()
                        }
                        .disabled(newSymptom.isEmpty)
                    }
                }
                
                Section("Urgency Level") {
                    Picker("Urgency", selection: $urgency) {
                        ForEach(Consultation.UrgencyLevel.allCases, id: \.self) { level in
                            HStack {
                                Circle()
                                    .fill(Color(level.color))
                                    .frame(width: 12, height: 12)
                                Text(level.rawValue)
                            }
                            .tag(level)
                        }
                    }
                }
                
                Section("Cost") {
                    HStack {
                        Text("Consultation Fee")
                        Spacer()
                        Text("$\(provider.consultationFee.amount, specifier: "%.2f")")
                            .fontWeight(.semibold)
                    }
                    
                    if provider.acceptsInsurance {
                        HStack {
                            Text("Insurance Copay")
                            Spacer()
                            Text("$\(provider.consultationFee.insuranceCopay ?? 0, specifier: "%.2f")")
                                .fontWeight(.semibold)
                                .foregroundColor(.green)
                        }
                    }
                }
            }
            .navigationTitle("Book Appointment")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Book") {
                    bookAppointment()
                }
                .disabled(selectedTime == nil || reason.isEmpty || isBooking)
            )
        }
        .onAppear {
            loadAvailableSlots()
        }
        .alert("Appointment Booked!", isPresented: $showingSuccess) {
            Button("OK") {
                presentationMode.wrappedValue.dismiss()
            }
        } message: {
            Text("Your appointment has been successfully scheduled.")
        }
    }
    
    private func loadAvailableSlots() {
        Task {
            let slots = await telemedicineManager.getProviderAvailability(providerId: provider.id, date: selectedDate)
            await MainActor.run {
                self.availableSlots = slots
            }
        }
    }
    
    private func addSymptom() {
        if !newSymptom.isEmpty {
            symptoms.append(newSymptom)
            newSymptom = ""
        }
    }
    
    private func removeSymptom(_ symptom: String) {
        symptoms.removeAll { $0 == symptom }
    }
    
    private func bookAppointment() {
        guard let selectedTime = selectedTime else { return }
        
        isBooking = true
        
        Task {
            do {
                _ = try await telemedicineManager.scheduleConsultation(
                    providerId: provider.id,
                    date: selectedTime,
                    type: consultationType,
                    reason: reason,
                    symptoms: symptoms,
                    urgency: urgency
                )
                
                await MainActor.run {
                    self.isBooking = false
                    self.showingSuccess = true
                }
            } catch {
                await MainActor.run {
                    self.isBooking = false
                    // Handle error
                }
            }
        }
    }
}

// MARK: - Appointments View

struct AppointmentsView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    @State private var selectedSegment = 0
    
    var upcomingAppointments: [Consultation] {
        telemedicineManager.consultations.filter { $0.status == .scheduled && $0.scheduledDate > Date() }
    }
    
    var pastAppointments: [Consultation] {
        telemedicineManager.consultations.filter { $0.status == .completed || $0.scheduledDate < Date() }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            Picker("Appointments", selection: $selectedSegment) {
                Text("Upcoming").tag(0)
                Text("Past").tag(1)
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding()
            
            if selectedSegment == 0 {
                if upcomingAppointments.isEmpty {
                    EmptyAppointmentsView(isUpcoming: true)
                } else {
                    List(upcomingAppointments, id: \.id) { appointment in
                        AppointmentRowView(appointment: appointment)
                            .environmentObject(telemedicineManager)
                    }
                }
            } else {
                if pastAppointments.isEmpty {
                    EmptyAppointmentsView(isUpcoming: false)
                } else {
                    List(pastAppointments, id: \.id) { appointment in
                        AppointmentRowView(appointment: appointment)
                            .environmentObject(telemedicineManager)
                    }
                }
            }
        }
    }
}

struct AppointmentRowView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    let appointment: Consultation
    @State private var showingDetail = false
    
    var provider: HealthcareProvider? {
        telemedicineManager.providers.first { $0.id == appointment.providerId }
    }
    
    var body: some View {
        Button(action: { showingDetail = true }) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(provider?.name ?? "Unknown Provider")
                            .font(.headline)
                            .foregroundColor(.primary)
                        
                        Text(provider?.specialty.rawValue ?? "")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .trailing, spacing: 4) {
                        Text(DateFormatter.dateFormatter.string(from: appointment.scheduledDate))
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Text(DateFormatter.timeFormatter.string(from: appointment.scheduledDate))
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                }
                
                HStack {
                    HStack(spacing: 4) {
                        Image(systemName: appointment.type.icon)
                            .font(.caption)
                        Text(appointment.type.rawValue)
                            .font(.caption)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.blue.opacity(0.1))
                    .foregroundColor(.blue)
                    .cornerRadius(6)
                    
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Color(appointment.status.color))
                            .frame(width: 8, height: 8)
                        Text(appointment.status.rawValue)
                            .font(.caption)
                    }
                    
                    Spacer()
                    
                    if appointment.status == .scheduled && appointment.scheduledDate.timeIntervalSinceNow < 900 { // 15 minutes
                        Button("Join") {
                            joinCall()
                        }
                        .font(.caption)
                        .fontWeight(.semibold)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                }
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(PlainButtonStyle())
        .sheet(isPresented: $showingDetail) {
            AppointmentDetailView(appointment: appointment)
                .environmentObject(telemedicineManager)
        }
    }
    
    private func joinCall() {
        Task {
            try? await telemedicineManager.startVideoCall(consultationId: appointment.id)
        }
    }
}

struct AppointmentDetailView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    @Environment(\.presentationMode) var presentationMode
    let appointment: Consultation
    @State private var showingCancelAlert = false
    @State private var showingReschedule = false
    
    var provider: HealthcareProvider? {
        telemedicineManager.providers.first { $0.id == appointment.providerId }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Provider Info
                    if let provider = provider {
                        HStack(spacing: 12) {
                            AsyncImage(url: provider.profileImageURL) { image in
                                image
                                    .resizable()
                                    .aspectRatio(contentMode: .fill)
                            } placeholder: {
                                Image(systemName: "person.circle.fill")
                                    .font(.system(size: 40))
                                    .foregroundColor(.gray)
                            }
                            .frame(width: 60, height: 60)
                            .clipShape(Circle())
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text(provider.name)
                                    .font(.headline)
                                
                                Text(provider.specialty.rawValue)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                        }
                    }
                    
                    // Appointment Details
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Appointment Details")
                            .font(.headline)
                        
                        DetailRow(title: "Date", value: DateFormatter.dateFormatter.string(from: appointment.scheduledDate))
                        DetailRow(title: "Time", value: DateFormatter.timeFormatter.string(from: appointment.scheduledDate))
                        DetailRow(title: "Type", value: appointment.type.rawValue)
                        DetailRow(title: "Status", value: appointment.status.rawValue)
                        DetailRow(title: "Duration", value: "\(Int(appointment.duration / 60)) minutes")
                    }
                    
                    // Reason and Symptoms
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Reason for Visit")
                            .font(.headline)
                        
                        Text(appointment.reason)
                            .font(.body)
                            .foregroundColor(.secondary)
                        
                        if !appointment.symptoms.isEmpty {
                            Text("Symptoms")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                            
                            ForEach(appointment.symptoms, id: \.self) { symptom in
                                HStack {
                                    Image(systemName: "circle.fill")
                                        .font(.system(size: 6))
                                        .foregroundColor(.secondary)
                                    Text(symptom)
                                        .font(.body)
                                }
                            }
                        }
                    }
                    
                    // Actions
                    if appointment.status == .scheduled {
                        VStack(spacing: 12) {
                            if appointment.scheduledDate.timeIntervalSinceNow < 900 { // 15 minutes
                                Button(action: joinCall) {
                                    HStack {
                                        Image(systemName: "video")
                                        Text("Join Call")
                                    }
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                                }
                            }
                            
                            HStack(spacing: 12) {
                                Button("Reschedule") {
                                    showingReschedule = true
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                                
                                Button("Cancel") {
                                    showingCancelAlert = true
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.red)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Appointment")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                trailing: Button("Done") {
                    presentationMode.wrappedValue.dismiss()
                }
            )
        }
        .alert("Cancel Appointment", isPresented: $showingCancelAlert) {
            Button("Cancel Appointment", role: .destructive) {
                cancelAppointment()
            }
            Button("Keep Appointment", role: .cancel) { }
        } message: {
            Text("Are you sure you want to cancel this appointment?")
        }
    }
    
    private func joinCall() {
        Task {
            try? await telemedicineManager.startVideoCall(consultationId: appointment.id)
        }
    }
    
    private func cancelAppointment() {
        Task {
            try? await telemedicineManager.cancelConsultation(consultationId: appointment.id)
            await MainActor.run {
                presentationMode.wrappedValue.dismiss()
            }
        }
    }
}

// MARK: - Video Call View

struct VideoCallView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    @State private var showingControls = true
    @State private var controlsTimer: Timer?
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // Video Views
            VStack {
                // Remote Video (Provider)
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .overlay(
                        VStack {
                            Image(systemName: "person.fill")
                                .font(.system(size: 60))
                                .foregroundColor(.white)
                            Text("Dr. Sarah Johnson")
                                .foregroundColor(.white)
                                .font(.headline)
                        }
                    )
                
                Spacer()
            }
            
            // Local Video (Self)
            VStack {
                HStack {
                    Spacer()
                    
                    Rectangle()
                        .fill(Color.blue.opacity(0.3))
                        .frame(width: 120, height: 160)
                        .overlay(
                            VStack {
                                Image(systemName: "person.fill")
                                    .font(.system(size: 30))
                                    .foregroundColor(.white)
                                Text("You")
                                    .foregroundColor(.white)
                                    .font(.caption)
                            }
                        )
                        .cornerRadius(12)
                        .padding()
                }
                
                Spacer()
            }
            
            // Connection Quality Indicator
            VStack {
                HStack {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Color(telemedicineManager.connectionQuality.color))
                            .frame(width: 8, height: 8)
                        Text(telemedicineManager.connectionQuality.displayName)
                            .font(.caption)
                            .foregroundColor(.white)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(20)
                    .padding()
                    
                    Spacer()
                }
                
                Spacer()
            }
            
            // Call Controls
            if showingControls {
                VStack {
                    Spacer()
                    
                    HStack(spacing: 30) {
                        // Mute Button
                        Button(action: { telemedicineManager.toggleMute() }) {
                            Image(systemName: telemedicineManager.videoCallManager.isMuted ? "mic.slash.fill" : "mic.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(telemedicineManager.videoCallManager.isMuted ? Color.red : Color.black.opacity(0.5))
                                .clipShape(Circle())
                        }
                        
                        // Video Toggle Button
                        Button(action: { telemedicineManager.toggleVideo() }) {
                            Image(systemName: telemedicineManager.videoCallManager.isVideoEnabled ? "video.fill" : "video.slash.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(telemedicineManager.videoCallManager.isVideoEnabled ? Color.black.opacity(0.5) : Color.red)
                                .clipShape(Circle())
                        }
                        
                        // End Call Button
                        Button(action: endCall) {
                            Image(systemName: "phone.down.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Color.red)
                                .clipShape(Circle())
                        }
                        
                        // Switch Camera Button
                        Button(action: { telemedicineManager.switchCamera() }) {
                            Image(systemName: "camera.rotate.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Color.black.opacity(0.5))
                                .clipShape(Circle())
                        }
                    }
                    .padding(.bottom, 50)
                }
            }
        }
        .onTapGesture {
            toggleControls()
        }
        .onAppear {
            startControlsTimer()
        }
    }
    
    private func toggleControls() {
        withAnimation(.easeInOut(duration: 0.3)) {
            showingControls.toggle()
        }
        
        if showingControls {
            startControlsTimer()
        }
    }
    
    private func startControlsTimer() {
        controlsTimer?.invalidate()
        controlsTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: false) { _ in
            withAnimation(.easeInOut(duration: 0.3)) {
                showingControls = false
            }
        }
    }
    
    private func endCall() {
        Task {
            await telemedicineManager.endVideoCall()
        }
    }
}

// MARK: - Health Data View

struct HealthDataView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    @State private var selectedTimeRange = 0
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Time Range Selector
                Picker("Time Range", selection: $selectedTimeRange) {
                    Text("Last Week").tag(0)
                    Text("Last Month").tag(1)
                    Text("Last 3 Months").tag(2)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
                
                if let healthData = telemedicineManager.healthDataSummary {
                    VStack(alignment: .leading, spacing: 16) {
                        // Vital Signs
                        if !healthData.vitalSigns.isEmpty {
                            HealthDataSection(title: "Vital Signs", icon: "heart.fill") {
                                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                                    ForEach(healthData.vitalSigns, id: \.timestamp) { vital in
                                        VitalSignCard(vital: vital)
                                    }
                                }
                            }
                        }
                        
                        // Symptoms
                        if !healthData.symptoms.isEmpty {
                            HealthDataSection(title: "Recent Symptoms", icon: "exclamationmark.triangle.fill") {
                                ForEach(healthData.symptoms, id: \.timestamp) { symptom in
                                    SymptomCard(symptom: symptom)
                                }
                            }
                        }
                        
                        // Medications
                        if !healthData.medications.isEmpty {
                            HealthDataSection(title: "Medications", icon: "pills.fill") {
                                ForEach(healthData.medications, id: \.name) { medication in
                                    MedicationCard(medication: medication)
                                }
                            }
                        }
                        
                        // Sleep Data
                        if !healthData.sleepData.isEmpty {
                            HealthDataSection(title: "Sleep", icon: "moon.fill") {
                                ForEach(healthData.sleepData, id: \.bedtime) { sleep in
                                    SleepCard(sleep: sleep)
                                }
                            }
                        }
                    }
                    .padding(.horizontal)
                } else {
                    VStack(spacing: 16) {
                        Image(systemName: "chart.bar.doc.horizontal")
                            .font(.system(size: 60))
                            .foregroundColor(.gray)
                        
                        Text("No Health Data Available")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        
                        Text("Connect your health apps to see your data here")
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                        
                        Button("Connect Health Apps") {
                            // Connect health apps
                        }
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding()
                }
            }
        }
        .onAppear {
            telemedicineManager.generateHealthDataSummary()
        }
    }
}

// MARK: - Prescriptions View

struct PrescriptionsView: View {
    @EnvironmentObject var telemedicineManager: TelemedicineManager
    
    var body: some View {
        if telemedicineManager.prescriptions.isEmpty {
            VStack(spacing: 16) {
                Image(systemName: "pills")
                    .font(.system(size: 60))
                    .foregroundColor(.gray)
                
                Text("No Prescriptions")
                    .font(.headline)
                    .foregroundColor(.secondary)
                
                Text("Your prescriptions from consultations will appear here")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding()
        } else {
            List(telemedicineManager.prescriptions, id: \.id) { prescription in
                PrescriptionRowView(prescription: prescription)
            }
        }
    }
}

struct PrescriptionRowView: View {
    let prescription: Prescription
    @State private var showingDetail = false
    
    var body: some View {
        Button(action: { showingDetail = true }) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(prescription.medicationName)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Spacer()
                    
                    Text(DateFormatter.dateFormatter.string(from: prescription.prescribedDate))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                HStack {
                    Text(prescription.dosage)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Text("•")
                        .foregroundColor(.secondary)
                    
                    Text(prescription.frequency)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text("\(prescription.refills) refills")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.green.opacity(0.1))
                        .foregroundColor(.green)
                        .cornerRadius(6)
                }
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(PlainButtonStyle())
        .sheet(isPresented: $showingDetail) {
            PrescriptionDetailView(prescription: prescription)
        }
    }
}

struct PrescriptionDetailView: View {
    @Environment(\.presentationMode) var presentationMode
    let prescription: Prescription
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Medication Info
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Medication Information")
                            .font(.headline)
                        
                        DetailRow(title: "Name", value: prescription.medicationName)
                        DetailRow(title: "Dosage", value: prescription.dosage)
                        DetailRow(title: "Frequency", value: prescription.frequency)
                        DetailRow(title: "Duration", value: prescription.duration)
                        DetailRow(title: "Refills", value: "\(prescription.refills)")
                        DetailRow(title: "Prescribed Date", value: DateFormatter.dateFormatter.string(from: prescription.prescribedDate))
                    }
                    
                    // Instructions
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Instructions")
                            .font(.headline)
                        
                        Text(prescription.instructions)
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    
                    // Side Effects
                    if !prescription.sideEffects.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Possible Side Effects")
                                .font(.headline)
                            
                            ForEach(prescription.sideEffects, id: \.self) { sideEffect in
                                HStack {
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .foregroundColor(.orange)
                                        .font(.caption)
                                    Text(sideEffect)
                                        .font(.body)
                                }
                            }
                        }
                    }
                    
                    // Interactions
                    if !prescription.interactions.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Drug Interactions")
                                .font(.headline)
                            
                            ForEach(prescription.interactions, id: \.self) { interaction in
                                HStack {
                                    Image(systemName: "exclamationmark.circle.fill")
                                        .foregroundColor(.red)
                                        .font(.caption)
                                    Text(interaction)
                                        .font(.body)
                                }
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Prescription Details")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                trailing: Button("Done") {
                    presentationMode.wrappedValue.dismiss()
                }
            )
        }
    }
}

// MARK: - Helper Views

struct ProviderFiltersView: View {
    @Binding var selectedSpecialty: MedicalSpecialty?
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            Form {
                Section("Specialty") {
                    ForEach(MedicalSpecialty.allCases, id: \.self) { specialty in
                        Button(action: {
                            selectedSpecialty = specialty
                            presentationMode.wrappedValue.dismiss()
                        }) {
                            HStack {
                                Image(systemName: specialty.icon)
                                Text(specialty.rawValue)
                                Spacer()
                                if selectedSpecialty == specialty {
                                    Image(systemName: "checkmark")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                        .foregroundColor(.primary)
                    }
                }
            }
            .navigationTitle("Filters")
            .navigationBarItems(
                leading: Button("Clear") {
                    selectedSpecialty = nil
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Done") {
                    presentationMode.wrappedValue.dismiss()
                }
            )
        }
    }
}

struct EmptyProvidersView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "person.2")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No Providers Found")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Try adjusting your search criteria or filters")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

struct EmptyAppointmentsView: View {
    let isUpcoming: Bool
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "calendar")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text(isUpcoming ? "No Upcoming Appointments" : "No Past Appointments")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text(isUpcoming ? "Book your first appointment with a healthcare provider" : "Your completed appointments will appear here")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

struct DetailRow: View {
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

struct HealthDataSection<Content: View>: View {
    let title: String
    let icon: String
    let content: Content
    
    init(title: String, icon: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                Text(title)
                    .font(.headline)
            }
            
            content
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct VitalSignCard: View {
    let vital: HealthDataSummary.VitalSignReading
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(vital.type.rawValue)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text("\(vital.value, specifier: "%.1f") \(vital.unit)")
                .font(.headline)
                .fontWeight(.semibold)
            
            Text(DateFormatter.timeFormatter.string(from: vital.timestamp))
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.white)
        .cornerRadius(8)
        .shadow(radius: 1)
    }
}

struct SymptomCard: View {
    let symptom: HealthDataSummary.SymptomEntry
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(symptom.symptom)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text("Severity: \(symptom.severity)/10")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(DateFormatter.timeFormatter.string(from: symptom.timestamp))
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.white)
        .cornerRadius(8)
        .shadow(radius: 1)
    }
}

struct MedicationCard: View {
    let medication: HealthDataSummary.MedicationEntry
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(medication.name)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text("\(Int(medication.adherence * 100))%")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(medication.adherence > 0.8 ? .green : .orange)
            }
            
            Text("\(medication.dosage) - \(medication.frequency)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.white)
        .cornerRadius(8)
        .shadow(radius: 1)
    }
}

struct SleepCard: View {
    let sleep: HealthDataSummary.SleepEntry
    
    var sleepDuration: TimeInterval {
        sleep.wakeTime.timeIntervalSince(sleep.bedtime)
    }
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Sleep Duration")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text("\(Int(sleepDuration / 3600))h \(Int((sleepDuration.truncatingRemainder(dividingBy: 3600)) / 60))m")
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text("Quality")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text("\(sleep.quality)/10")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(sleep.quality >= 7 ? .green : sleep.quality >= 5 ? .orange : .red)
            }
        }
        .padding()
        .background(Color.white)
        .cornerRadius(8)
        .shadow(radius: 1)
    }
}

// MARK: - Extensions

extension DateFormatter {
    static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter
    }()
    
    static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter
    }()
}

#Preview {
    TelemedicineView()
}