//
//  JournalView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData

struct JournalView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @StateObject private var errorHandler = CoreDataErrorHandler.shared
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: false)],
        animation: .default)
    private var journalEntries: FetchedResults<JournalEntry>
    
    @State private var showingAddEntry = false
    @State private var selectedEntry: JournalEntry?
    @State private var showingEntryDetail = false
    @State private var showingHistory = false
    @State private var isLoading = false
    @State private var showingSuccessAlert = false

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from TrackHubView,
        // which is already wrapped in NavigationView in MainTabView.
        ScrollView {
            VStack(spacing: 20) {
                // Quick Stats Card
                if !journalEntries.isEmpty {
                    QuickStatsCard(entries: Array(journalEntries.prefix(7)))
                        .padding(.horizontal)
                }

                // Journal Entries
                LazyVStack(spacing: 12) {
                    if journalEntries.isEmpty {
                        EmptyJournalView {
                            showingAddEntry = true
                        }
                    } else {
                        ForEach(journalEntries, id: \.objectID) { entry in
                            JournalEntryCard(entry: entry) {
                                selectedEntry = entry
                                showingEntryDetail = true
                            }
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
        .navigationTitle("Health Journal")
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    showingHistory = true
                } label: {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.title2)
                }
            }

            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showingAddEntry = true
                } label: {
                    Image(systemName: "plus.circle.fill")
                        .font(.title2)
                }
            }
        }
        .sheet(isPresented: $showingAddEntry) {
            AddJournalEntryView()
        }
        .sheet(isPresented: $showingEntryDetail) {
            if let entry = selectedEntry {
                JournalEntryDetailView(entry: entry)
            }
        }
        .sheet(isPresented: $showingHistory) {
            JournalHistoryView()
        }
        .coreDataErrorAlert()
        .alert("Success", isPresented: $showingSuccessAlert) {
            Button("OK") { }
        } message: {
            Text("Journal entry has been saved successfully.")
        }
    }
}

// MARK: - Quick Stats Card
struct QuickStatsCard: View {
    let entries: [JournalEntry]
    
    var averageMood: Double {
        guard !entries.isEmpty else { return 0 }
        let moodValues = entries.compactMap { moodToValue($0.mood ?? "") }
        return moodValues.reduce(0, +) / Double(moodValues.count)
    }
    
    var averageEnergyLevel: Double {
        guard !entries.isEmpty else { return 0 }
        return entries.reduce(0) { $0 + $1.energyLevel } / Double(entries.count)
    }
    
    var body: some View {
        VStack(spacing: 15) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(.blue)
                Text("7-Day Overview")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }
            
            HStack(spacing: 20) {
                StatItem(title: "Avg Mood", value: String(format: "%.1f", averageMood), icon: "face.smiling", color: .green)
                StatItem(title: "Avg Energy", value: String(format: "%.1f", averageEnergyLevel), icon: "bolt.fill", color: .orange)
                StatItem(title: "Entries", value: "\(entries.count)", icon: "book.fill", color: .blue)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func moodToValue(_ mood: String) -> Double? {
        switch mood.lowercased() {
        case "excellent": return 5.0
        case "good": return 4.0
        case "fair": return 3.0
        case "poor": return 2.0
        case "very poor": return 1.0
        default: return nil
        }
    }
}

struct StatItem: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Empty Journal View
struct EmptyJournalView: View {
    let onAddEntry: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "book")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            VStack(spacing: 8) {
                Text("Start Your Health Journey")
                    .font(.title2)
                    .fontWeight(.semibold)
                Text("Track your daily symptoms, mood, and progress")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            
            Button("Add Your First Entry") {
                onAddEntry()
            }
            .buttonStyle(.borderedProminent)
            .tint(Colors.Primary.p500)
        }
        .padding(40)
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Journal Entry Card
struct JournalEntryCard: View {
    let entry: JournalEntry
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 12) {
                // Header
                HStack {
                    Text(entry.date ?? Date(), style: .date)
                        .font(.headline)
                        .fontWeight(.semibold)
                    Spacer()
                    MoodBadge(mood: entry.mood ?? "")
                }
                
                // Quick Stats
                HStack(spacing: 20) {
                    StatRow(icon: "bolt.fill", label: "Energy", value: "\(Int(entry.energyLevel))/10", color: .orange)
                    StatRow(icon: "bed.double.fill", label: "Sleep", value: "\(Int(entry.sleepQuality))/10", color: .purple)
                    if entry.painLevel > 0 {
                        StatRow(icon: "exclamationmark.triangle.fill", label: "Pain", value: "\(Int(entry.painLevel))/10", color: .red)
                    }
                }
                
                // Symptoms Preview
                if let symptoms = entry.symptoms, !symptoms.isEmpty {
                    Text(symptoms)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
                
                // Notes Preview
                if let notes = entry.notes, !notes.isEmpty {
                    Text(notes)
                        .font(.body)
                        .lineLimit(2)
                }
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.systemBackground))
            .cornerRadius(12)
            .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct StatRow: View {
    let icon: String
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundColor(color)
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
}

struct MoodBadge: View {
    let mood: String
    
    var body: some View {
        Text(mood)
            .font(.caption)
            .fontWeight(.medium)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(moodColor(for: mood))
            .foregroundColor(.white)
            .cornerRadius(12)
    }
    
    private func moodColor(for mood: String) -> Color {
        switch mood.lowercased() {
        case "excellent":
            return .green
        case "good":
            return .blue
        case "fair":
            return .orange
        case "poor":
            return .red
        case "very poor":
            return .purple
        default:
            return .gray
        }
    }
}

// MARK: - Add Journal Entry View
struct AddJournalEntryView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss
    @StateObject private var errorHandler = CoreDataErrorHandler.shared
    
    @State private var selectedDate = Date()
    @State private var mood = "Good"
    @State private var energyLevel: Double = 5
    @State private var sleepQuality: Double = 5
    @State private var painLevel: Double = 0
    @State private var symptoms = ""
    @State private var notes = ""
    @State private var activities = ""
    @State private var medications = ""
    @State private var isLoading = false
    
    private let moods = ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
    
    var body: some View {
        NavigationView {
            Form {
                Section("Entry Details") {
                    DatePicker("Date", selection: $selectedDate, displayedComponents: .date)
                    
                    Picker("Mood", selection: $mood) {
                        ForEach(moods, id: \.self) { mood in
                            Text(mood).tag(mood)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                
                Section("Health Metrics") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "bolt.fill")
                                .foregroundColor(.orange)
                            Text("Energy Level")
                            Spacer()
                            Text("\(Int(energyLevel))/10")
                                .fontWeight(.medium)
                        }
                        Slider(value: $energyLevel, in: 1...10, step: 1)
                    }
                    
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "bed.double.fill")
                                .foregroundColor(.purple)
                            Text("Sleep Quality")
                            Spacer()
                            Text("\(Int(sleepQuality))/10")
                                .fontWeight(.medium)
                        }
                        Slider(value: $sleepQuality, in: 1...10, step: 1)
                    }
                    
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                            Text("Pain Level")
                            Spacer()
                            Text("\(Int(painLevel))/10")
                                .fontWeight(.medium)
                        }
                        Slider(value: $painLevel, in: 0...10, step: 1)
                    }
                }
                
                Section("Symptoms") {
                    TextEditor(text: $symptoms)
                        .frame(minHeight: 80)
                }
                
                Section("Activities & Exercise") {
                    TextEditor(text: $activities)
                        .frame(minHeight: 60)
                }
                
                Section("Medications Taken") {
                    TextEditor(text: $medications)
                        .frame(minHeight: 60)
                }
                
                Section("Additional Notes") {
                    TextEditor(text: $notes)
                        .frame(minHeight: 80)
                }
            }
            .navigationTitle("New Journal Entry")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveEntry()
                    }
                    .disabled(symptoms.isEmpty && notes.isEmpty || isLoading)
                }
            }
        }
    }
    
    private func saveEntry() {
        // Validate input
        guard !symptoms.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        
        isLoading = true
        
        // Create journal entry using safe operations
        let result: Result<JournalEntry, CoreDataError> = CoreDataOperations.createEntity(
            entityName: "JournalEntry",
            context: viewContext
        )
        
        switch result {
        case .success(let newEntry):
            // Set journal entry properties
            newEntry.id = UUID()
            newEntry.date = selectedDate
            newEntry.mood = mood
            newEntry.energyLevel = energyLevel
            newEntry.sleepQuality = sleepQuality
            newEntry.painLevel = painLevel
            newEntry.symptoms = symptoms.isEmpty ? nil : symptoms
            newEntry.notes = notes.isEmpty ? nil : notes
            newEntry.activities = activities.isEmpty ? nil : activities
            newEntry.medications = medications.isEmpty ? nil : medications
            
            // Save with proper error handling
            CoreDataOperations.safeSave(context: viewContext) { saveResult in
                DispatchQueue.main.async {
                    self.isLoading = false
                    
                    switch saveResult {
                    case .success:
                        self.dismiss()
                        
                    case .failure:
                        // Error already handled by CoreDataOperations
                        break
                    }
                }
            }
            
        case .failure:
            isLoading = false
            // Error already handled by CoreDataOperations
        }
    }
}

// MARK: - Journal Entry Detail View
struct JournalEntryDetailView: View {
    let entry: JournalEntry
    @Environment(\.dismiss) private var dismiss
    @Environment(\.managedObjectContext) private var viewContext
    
    @State private var showingDeleteAlert = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header Card
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text(entry.date ?? Date(), style: .date)
                                .font(.title2)
                                .fontWeight(.bold)
                            Spacer()
                            MoodBadge(mood: entry.mood ?? "")
                        }
                        
                        Text(entry.date ?? Date(), style: .time)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Health Metrics
                    VStack(alignment: .leading, spacing: 15) {
                        Text("Health Metrics")
                            .font(.headline)
                            .fontWeight(.semibold)
                        
                        VStack(spacing: 12) {
                            MetricRow(icon: "bolt.fill", label: "Energy Level", value: "\(Int(entry.energyLevel))/10", color: .orange)
                            MetricRow(icon: "bed.double.fill", label: "Sleep Quality", value: "\(Int(entry.sleepQuality))/10", color: .purple)
                            if entry.painLevel > 0 {
                                MetricRow(icon: "exclamationmark.triangle.fill", label: "Pain Level", value: "\(Int(entry.painLevel))/10", color: .red)
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Symptoms
                    if let symptoms = entry.symptoms, !symptoms.isEmpty {
                        DetailSection(title: "Symptoms", content: symptoms, icon: "stethoscope")
                    }
                    
                    // Activities
                    if let activities = entry.activities, !activities.isEmpty {
                        DetailSection(title: "Activities & Exercise", content: activities, icon: "figure.walk")
                    }
                    
                    // Medications
                    if let medications = entry.medications, !medications.isEmpty {
                        DetailSection(title: "Medications Taken", content: medications, icon: "pills")
                    }
                    
                    // Notes
                    if let notes = entry.notes, !notes.isEmpty {
                        DetailSection(title: "Additional Notes", content: notes, icon: "note.text")
                    }
                }
                .padding()
            }
            .navigationTitle("Journal Entry")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Delete") {
                        showingDeleteAlert = true
                    }
                    .foregroundColor(.red)
                }
            }
            .alert("Delete Entry", isPresented: $showingDeleteAlert) {
                Button("Cancel", role: .cancel) { }
                Button("Delete", role: .destructive) {
                    deleteEntry()
                }
            } message: {
                Text("Are you sure you want to delete this journal entry? This action cannot be undone.")
            }
        }
    }
    
    private func deleteEntry() {
        CoreDataOperations.safeDelete(object: entry, context: viewContext) { result in
            DispatchQueue.main.async {
                switch result {
                case .success:
                    self.dismiss()
                case .failure:
                    // Error already handled by CoreDataOperations
                    break
                }
            }
        }
    }
}

struct MetricRow: View {
    let icon: String
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
                .frame(width: 24)
            Text(label)
                .font(.subheadline)
            Spacer()
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
        }
    }
}

struct DetailSection: View {
    let title: String
    let content: String
    let icon: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                Text(title)
                    .font(.headline)
                    .fontWeight(.semibold)
            }
            
            Text(content)
                .font(.body)
                .lineSpacing(4)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct JournalView_Previews: PreviewProvider {
    static var previews: some View {
        JournalView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}