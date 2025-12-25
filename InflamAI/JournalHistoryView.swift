//
//  JournalHistoryView.swift
//  InflamAI-Swift
//
//  Created by Assistant on 2024
//

import SwiftUI
import CoreData

struct JournalHistoryView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: false)],
        animation: .default)
    private var allEntries: FetchedResults<JournalEntry>
    
    @State private var searchText = ""
    @State private var selectedDateRange: DateRange = .all
    @State private var showingDatePicker = false
    @State private var customStartDate = Calendar.current.date(byAdding: .month, value: -1, to: Date()) ?? Date()
    @State private var customEndDate = Date()
    @State private var showingExportSheet = false
    @State private var selectedEntry: JournalEntry?
    
    enum DateRange: String, CaseIterable {
        case all = "All Time"
        case week = "Last 7 Days"
        case month = "Last 30 Days"
        case threeMonths = "Last 3 Months"
        case custom = "Custom Range"
    }
    
    var filteredEntries: [JournalEntry] {
        var entries = Array(allEntries)
        
        // Filter by date range
        let now = Date()
        switch selectedDateRange {
        case .all:
            break
        case .week:
            let weekAgo = Calendar.current.date(byAdding: .day, value: -7, to: now) ?? now
            entries = entries.filter { ($0.date ?? Date()) >= weekAgo }
        case .month:
            let monthAgo = Calendar.current.date(byAdding: .day, value: -30, to: now) ?? now
            entries = entries.filter { ($0.date ?? Date()) >= monthAgo }
        case .threeMonths:
            let threeMonthsAgo = Calendar.current.date(byAdding: .month, value: -3, to: now) ?? now
            entries = entries.filter { ($0.date ?? Date()) >= threeMonthsAgo }
        case .custom:
            entries = entries.filter { 
                let entryDate = $0.date ?? Date()
                return entryDate >= customStartDate && entryDate <= customEndDate
            }
        }
        
        // Filter by search text
        if !searchText.isEmpty {
            entries = entries.filter { entry in
                let symptoms = entry.symptoms ?? ""
                let notes = entry.notes ?? ""
                let activities = entry.activities ?? ""
                let medications = entry.medications ?? ""
                let mood = entry.mood ?? ""
                
                return symptoms.localizedCaseInsensitiveContains(searchText) ||
                       notes.localizedCaseInsensitiveContains(searchText) ||
                       activities.localizedCaseInsensitiveContains(searchText) ||
                       medications.localizedCaseInsensitiveContains(searchText) ||
                       mood.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        return entries
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search and Filter Section
                VStack(spacing: 12) {
                    // Search Bar
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(.secondary)
                        TextField("Search entries...", text: $searchText)
                            .textFieldStyle(PlainTextFieldStyle())
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    
                    // Date Range Filter
                    HStack {
                        Text("Filter:")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        Picker("Date Range", selection: $selectedDateRange) {
                            ForEach(DateRange.allCases, id: \.self) { range in
                                Text(range.rawValue).tag(range)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        
                        Spacer()
                        
                        if selectedDateRange == .custom {
                            Button("Set Dates") {
                                showingDatePicker = true
                            }
                            .font(.caption)
                            .foregroundColor(.blue)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)
                .padding(.bottom, 12)
                .background(Color(.systemBackground))
                
                Divider()
                
                // Results Summary
                HStack {
                    Text("\(filteredEntries.count) entries found")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    if !filteredEntries.isEmpty {
                        Button("Export") {
                            showingExportSheet = true
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                
                // Entries List
                if filteredEntries.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "doc.text")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("No entries found")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        if !searchText.isEmpty || selectedDateRange != .all {
                            Text("Try adjusting your search or filter criteria")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        ForEach(filteredEntries, id: \.id) { entry in
                            JournalHistoryCard(entry: entry)
                                .onTapGesture {
                                    selectedEntry = entry
                                }
                        }
                        .onDelete(perform: deleteEntries)
                    }
                    .listStyle(PlainListStyle())
                }
            }
            .navigationTitle("Journal History")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .sheet(isPresented: $showingDatePicker) {
                CustomDateRangeView(
                    startDate: $customStartDate,
                    endDate: $customEndDate,
                    isPresented: $showingDatePicker
                )
            }
            .sheet(item: $selectedEntry) { entry in
                JournalEntryDetailView(entry: entry)
            }
            .actionSheet(isPresented: $showingExportSheet) {
                ActionSheet(
                    title: Text("Export Journal History"),
                    message: Text("Choose export format"),
                    buttons: [
                        .default(Text("Export as Text")) {
                            exportAsText()
                        },
                        .default(Text("Export as CSV")) {
                            exportAsCSV()
                        },
                        .cancel()
                    ]
                )
            }
        }
    }
    
    private func deleteEntries(offsets: IndexSet) {
        withAnimation {
            for index in offsets {
                let entry = filteredEntries[index]
                CoreDataOperations.safeDelete(object: entry, context: viewContext) { result in
                    // Error handling is done within CoreDataOperations
                }
            }
        }
    }
    
    private func exportAsText() {
        let content = generateTextExport()
        shareContent(content, filename: "journal_history.txt")
    }
    
    private func exportAsCSV() {
        let content = generateCSVExport()
        shareContent(content, filename: "journal_history.csv")
    }
    
    private func generateTextExport() -> String {
        var content = "Journal History Export\n"
        content += "Generated on: \(DateFormatter.localizedString(from: Date(), dateStyle: .full, timeStyle: .short))\n\n"
        
        for entry in filteredEntries {
            content += "Date: \(DateFormatter.localizedString(from: entry.date ?? Date(), dateStyle: .full, timeStyle: .short))\n"
            content += "Mood: \(entry.mood ?? "Not specified")\n"
            content += "Energy Level: \(Int(entry.energyLevel))/10\n"
            content += "Sleep Quality: \(Int(entry.sleepQuality))/10\n"
            if entry.painLevel > 0 {
                content += "Pain Level: \(Int(entry.painLevel))/10\n"
            }
            if let symptoms = entry.symptoms, !symptoms.isEmpty {
                content += "Symptoms: \(symptoms)\n"
            }
            if let activities = entry.activities, !activities.isEmpty {
                content += "Activities: \(activities)\n"
            }
            if let medications = entry.medications, !medications.isEmpty {
                content += "Medications: \(medications)\n"
            }
            if let notes = entry.notes, !notes.isEmpty {
                content += "Notes: \(notes)\n"
            }
            content += "\n---\n\n"
        }
        
        return content
    }
    
    private func generateCSVExport() -> String {
        var content = "Date,Time,Mood,Energy Level,Sleep Quality,Pain Level,Symptoms,Activities,Medications,Notes\n"
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .short
        let timeFormatter = DateFormatter()
        timeFormatter.timeStyle = .short
        
        for entry in filteredEntries {
            let date = dateFormatter.string(from: entry.date ?? Date())
            let time = timeFormatter.string(from: entry.date ?? Date())
            let mood = entry.mood ?? ""
            let energy = "\(Int(entry.energyLevel))"
            let sleep = "\(Int(entry.sleepQuality))"
            let pain = entry.painLevel > 0 ? "\(Int(entry.painLevel))" : ""
            let symptoms = (entry.symptoms ?? "").replacingOccurrences(of: ",", with: ";")
            let activities = (entry.activities ?? "").replacingOccurrences(of: ",", with: ";")
            let medications = (entry.medications ?? "").replacingOccurrences(of: ",", with: ";")
            let notes = (entry.notes ?? "").replacingOccurrences(of: ",", with: ";")
            
            content += "\(date),\(time),\(mood),\(energy),\(sleep),\(pain),\(symptoms),\(activities),\(medications),\(notes)\n"
        }
        
        return content
    }
    
    private func shareContent(_ content: String, filename: String) {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        
        do {
            try content.write(to: tempURL, atomically: true, encoding: .utf8)
            
            let activityViewController = UIActivityViewController(activityItems: [tempURL], applicationActivities: nil)
            
            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
               let window = windowScene.windows.first {
                window.rootViewController?.present(activityViewController, animated: true)
            }
        } catch {
            print("Error sharing content: \(error)")
        }
    }
}

// MARK: - Journal History Card
struct JournalHistoryCard: View {
    let entry: JournalEntry
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(entry.date ?? Date(), style: .date)
                        .font(.headline)
                        .fontWeight(.semibold)
                    Text(entry.date ?? Date(), style: .time)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                if let mood = entry.mood, !mood.isEmpty {
                    MoodBadge(mood: mood)
                }
            }
            
            // Quick Metrics
            HStack(spacing: 16) {
                MetricBadge(icon: "bolt.fill", value: "\(Int(entry.energyLevel))", color: .orange)
                MetricBadge(icon: "bed.double.fill", value: "\(Int(entry.sleepQuality))", color: .purple)
                if entry.painLevel > 0 {
                    MetricBadge(icon: "exclamationmark.triangle.fill", value: "\(Int(entry.painLevel))", color: .red)
                }
            }
            
            // Content Preview
            if let symptoms = entry.symptoms, !symptoms.isEmpty {
                Text(symptoms)
                    .font(.subheadline)
                    .lineLimit(2)
                    .foregroundColor(.secondary)
            } else if let notes = entry.notes, !notes.isEmpty {
                Text(notes)
                    .font(.subheadline)
                    .lineLimit(2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct MetricBadge: View {
    let icon: String
    let value: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundColor(color)
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.1))
        .cornerRadius(6)
    }
}

// MARK: - Custom Date Range View
struct CustomDateRangeView: View {
    @Binding var startDate: Date
    @Binding var endDate: Date
    @Binding var isPresented: Bool
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                DatePicker("Start Date", selection: $startDate, displayedComponents: .date)
                    .datePickerStyle(GraphicalDatePickerStyle())
                
                DatePicker("End Date", selection: $endDate, displayedComponents: .date)
                    .datePickerStyle(GraphicalDatePickerStyle())
                
                Spacer()
            }
            .padding()
            .navigationTitle("Select Date Range")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        isPresented = false
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
}

struct JournalHistoryView_Previews: PreviewProvider {
    static var previews: some View {
        JournalHistoryView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}