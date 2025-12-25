//
//  PainTrackingHistoryView.swift
//  InflamAI-Swift
//
//  Created by Assistant on 2024
//

import SwiftUI
import CoreData

struct PainTrackingHistoryView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss

    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)],
        predicate: NSPredicate(format: "source == %@ OR source == %@", "pain_tracking", "pain_map"),
        animation: .default)
    private var allEntries: FetchedResults<SymptomLog>
    
    @State private var searchText = ""
    @State private var selectedDateRange: DateRange = .all
    @State private var selectedPainLevelFilter: PainLevelFilter = .all
    @State private var showingDatePicker = false
    @State private var customStartDate = Calendar.current.date(byAdding: .month, value: -1, to: Date()) ?? Date()
    @State private var customEndDate = Date()
    @State private var showingExportSheet = false
    @State private var selectedEntry: SymptomLog?
    
    enum DateRange: String, CaseIterable {
        case all = "All Time"
        case week = "Last 7 Days"
        case month = "Last 30 Days"
        case threeMonths = "Last 3 Months"
        case custom = "Custom Range"
    }
    
    enum PainLevelFilter: String, CaseIterable {
        case all = "All Levels"
        case mild = "Mild (1-3)"
        case moderate = "Moderate (4-6)"
        case severe = "Severe (7-10)"
    }
    
    var filteredEntries: [SymptomLog] {
        var entries = Array(allEntries)

        // Filter by date range
        let now = Date()
        switch selectedDateRange {
        case .all:
            break
        case .week:
            let weekAgo = Calendar.current.date(byAdding: .day, value: -7, to: now) ?? now
            entries = entries.filter { ($0.timestamp ?? Date()) >= weekAgo }
        case .month:
            let monthAgo = Calendar.current.date(byAdding: .day, value: -30, to: now) ?? now
            entries = entries.filter { ($0.timestamp ?? Date()) >= monthAgo }
        case .threeMonths:
            let threeMonthsAgo = Calendar.current.date(byAdding: .month, value: -3, to: now) ?? now
            entries = entries.filter { ($0.timestamp ?? Date()) >= threeMonthsAgo }
        case .custom:
            entries = entries.filter {
                let entryDate = $0.timestamp ?? Date()
                return entryDate >= customStartDate && entryDate <= customEndDate
            }
        }

        // Filter by pain level (using max pain from body regions)
        switch selectedPainLevelFilter {
        case .all:
            break
        case .mild:
            entries = entries.filter { maxPainLevel(for: $0) >= 1 && maxPainLevel(for: $0) <= 3 }
        case .moderate:
            entries = entries.filter { maxPainLevel(for: $0) >= 4 && maxPainLevel(for: $0) <= 6 }
        case .severe:
            entries = entries.filter { maxPainLevel(for: $0) >= 7 && maxPainLevel(for: $0) <= 10 }
        }

        // Filter by search text
        if !searchText.isEmpty {
            entries = entries.filter { entry in
                let bodyRegions = bodyRegionNames(for: entry)
                let trigger = entry.painTriggers ?? ""
                let notes = entry.notes ?? ""

                return bodyRegions.localizedCaseInsensitiveContains(searchText) ||
                       trigger.localizedCaseInsensitiveContains(searchText) ||
                       notes.localizedCaseInsensitiveContains(searchText)
            }
        }

        return entries
    }

    var averagePainLevel: Double {
        guard !filteredEntries.isEmpty else { return 0 }
        let total = filteredEntries.reduce(0.0) { $0 + Double(maxPainLevel(for: $1)) }
        return total / Double(filteredEntries.count)
    }

    // Helper to get max pain level from body region logs
    private func maxPainLevel(for entry: SymptomLog) -> Int16 {
        guard let regions = entry.bodyRegionLogs as? Set<BodyRegionLog>, !regions.isEmpty else {
            return 0
        }
        return regions.map { $0.painLevel }.max() ?? 0
    }

    // Helper to get body region names as comma-separated string
    private func bodyRegionNames(for entry: SymptomLog) -> String {
        guard let regions = entry.bodyRegionLogs as? Set<BodyRegionLog> else {
            return ""
        }
        return regions.compactMap { $0.regionID }.joined(separator: ", ")
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
                        TextField("Search pain entries...", text: $searchText)
                            .textFieldStyle(PlainTextFieldStyle())
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    
                    // Filters
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Date Range")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Picker("Date Range", selection: $selectedDateRange) {
                                ForEach(DateRange.allCases, id: \.self) { range in
                                    Text(range.rawValue).tag(range)
                                }
                            }
                            .pickerStyle(MenuPickerStyle())
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Pain Level")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Picker("Pain Level", selection: $selectedPainLevelFilter) {
                                ForEach(PainLevelFilter.allCases, id: \.self) { filter in
                                    Text(filter.rawValue).tag(filter)
                                }
                            }
                            .pickerStyle(MenuPickerStyle())
                        }
                        
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
                
                // Statistics Summary
                if !filteredEntries.isEmpty {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(filteredEntries.count) entries")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Avg Pain: \(String(format: "%.1f", averagePainLevel))/10")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Button("Export") {
                            showingExportSheet = true
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    
                    Divider()
                }
                
                // Entries List
                if filteredEntries.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("No pain entries found")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        if !searchText.isEmpty || selectedDateRange != .all || selectedPainLevelFilter != .all {
                            Text("Try adjusting your search or filter criteria")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        ForEach(filteredEntries, id: \.objectID) { entry in
                            PainHistoryCard(entry: entry)
                                .onTapGesture {
                                    selectedEntry = entry
                                }
                        }
                        .onDelete(perform: deleteEntries)
                    }
                    .listStyle(PlainListStyle())
                }
            }
            .navigationTitle("Pain History")
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
                PainEntryDetailView(entry: entry)
            }
            .actionSheet(isPresented: $showingExportSheet) {
                ActionSheet(
                    title: Text("Export Pain History"),
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
        shareContent(content, filename: "pain_history.txt")
    }
    
    private func exportAsCSV() {
        let content = generateCSVExport()
        shareContent(content, filename: "pain_history.csv")
    }
    
    private func generateTextExport() -> String {
        var content = "Pain Tracking History Export\n"
        content += "Generated on: \(DateFormatter.localizedString(from: Date(), dateStyle: .full, timeStyle: .short))\n"
        content += "Total Entries: \(filteredEntries.count)\n"
        content += "Average Pain Level: \(String(format: "%.1f", averagePainLevel))/10\n\n"

        for entry in filteredEntries {
            content += "Date: \(DateFormatter.localizedString(from: entry.timestamp ?? Date(), dateStyle: .full, timeStyle: .short))\n"
            content += "Max Pain Level: \(maxPainLevel(for: entry))/10\n"

            // Export body regions with individual pain levels
            if let regions = entry.bodyRegionLogs as? Set<BodyRegionLog>, !regions.isEmpty {
                content += "Body Regions:\n"
                for region in regions.sorted(by: { ($0.regionID ?? "") < ($1.regionID ?? "") }) {
                    content += "  - \(region.regionID ?? "Unknown"): Pain \(region.painLevel)/10"
                    if region.stiffnessDuration > 0 {
                        content += ", Stiffness \(region.stiffnessDuration) min"
                    }
                    if region.swelling { content += ", Swelling" }
                    if region.warmth { content += ", Warmth" }
                    content += "\n"
                }
            }

            if let painType = entry.painType, !painType.isEmpty {
                content += "Pain Type: \(painType)\n"
            }
            if let trigger = entry.painTriggers, !trigger.isEmpty {
                content += "Trigger: \(trigger)\n"
            }
            if let notes = entry.notes, !notes.isEmpty {
                content += "Notes: \(notes)\n"
            }
            content += "\n---\n\n"
        }

        return content
    }

    private func generateCSVExport() -> String {
        var content = "Date,Time,Region,Pain Level,Stiffness (min),Swelling,Warmth,Pain Type,Trigger,Notes\n"

        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .short
        let timeFormatter = DateFormatter()
        timeFormatter.timeStyle = .short

        for entry in filteredEntries {
            let date = dateFormatter.string(from: entry.timestamp ?? Date())
            let time = timeFormatter.string(from: entry.timestamp ?? Date())
            let painType = (entry.painType ?? "").replacingOccurrences(of: ",", with: ";")
            let trigger = (entry.painTriggers ?? "").replacingOccurrences(of: ",", with: ";")
            let notes = (entry.notes ?? "").replacingOccurrences(of: ",", with: ";")

            // Export each body region as a separate row
            if let regions = entry.bodyRegionLogs as? Set<BodyRegionLog>, !regions.isEmpty {
                for region in regions.sorted(by: { ($0.regionID ?? "") < ($1.regionID ?? "") }) {
                    let regionName = (region.regionID ?? "Unknown").replacingOccurrences(of: ",", with: ";")
                    let painLevel = "\(region.painLevel)"
                    let stiffness = "\(region.stiffnessDuration)"
                    let swelling = region.swelling ? "Yes" : "No"
                    let warmth = region.warmth ? "Yes" : "No"

                    content += "\(date),\(time),\(regionName),\(painLevel),\(stiffness),\(swelling),\(warmth),\(painType),\(trigger),\(notes)\n"
                }
            }
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

// MARK: - Pain History Card
struct PainHistoryCard: View {
    let entry: SymptomLog

    // Get body region logs
    private var bodyRegions: [BodyRegionLog] {
        (entry.bodyRegionLogs as? Set<BodyRegionLog>)?.sorted { ($0.regionID ?? "") < ($1.regionID ?? "") } ?? []
    }

    // Get max pain level from body regions
    private var maxPainLevel: Int16 {
        bodyRegions.map { $0.painLevel }.max() ?? 0
    }

    var painLevelColor: Color {
        switch maxPainLevel {
        case 0...2:
            return .green
        case 3...5:
            return .yellow
        case 6...7:
            return .orange
        case 8...10:
            return .red
        default:
            return .gray
        }
    }

    var painLevelDescription: String {
        switch maxPainLevel {
        case 0:
            return "No Pain"
        case 1...2:
            return "Mild"
        case 3...4:
            return "Moderate"
        case 5...6:
            return "Moderately Severe"
        case 7...8:
            return "Severe"
        case 9...10:
            return "Very Severe"
        default:
            return "Unknown"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(entry.timestamp ?? Date(), style: .date)
                        .font(.headline)
                        .fontWeight(.semibold)
                    Text(entry.timestamp ?? Date(), style: .time)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Pain Level Badge (shows max pain)
                VStack(spacing: 2) {
                    Text("\(maxPainLevel)")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(painLevelColor)
                    Text(painLevelDescription)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(painLevelColor.opacity(0.1))
                .cornerRadius(8)
            }

            // Body Regions with individual pain levels
            if !bodyRegions.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Image(systemName: "figure.arms.open")
                            .foregroundColor(.blue)
                            .font(.caption)
                        Text("\(bodyRegions.count) region(s)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    // Show each region with its pain level
                    ForEach(bodyRegions.prefix(3), id: \.id) { region in
                        HStack {
                            Text(region.regionID ?? "Unknown")
                                .font(.subheadline)
                            Spacer()
                            Text("Pain: \(region.painLevel)/10")
                                .font(.caption)
                                .foregroundColor(colorForPainLevel(region.painLevel))
                                .fontWeight(.medium)
                        }
                    }

                    if bodyRegions.count > 3 {
                        Text("+ \(bodyRegions.count - 3) more regions...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            // Pain Type and Trigger
            HStack(spacing: 8) {
                if let painType = entry.painType, !painType.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "waveform.path")
                            .foregroundColor(.orange)
                            .font(.caption)
                        Text(painType)
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(6)
                }

                if let trigger = entry.painTriggers, !trigger.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "bolt")
                            .foregroundColor(.purple)
                            .font(.caption)
                        // CRIT-003 FIX: Apply displayName to convert snake_case to Title Case
                        Text(trigger.displayName)
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.purple.opacity(0.1))
                    .cornerRadius(6)
                }
            }

            // Notes Preview
            if let notes = entry.notes, !notes.isEmpty {
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

    private func colorForPainLevel(_ level: Int16) -> Color {
        switch level {
        case 0...2: return .green
        case 3...5: return .yellow
        case 6...7: return .orange
        case 8...10: return .red
        default: return .gray
        }
    }
}

// MARK: - Pain Entry Detail View
struct PainEntryDetailView: View {
    let entry: SymptomLog
    @Environment(\.dismiss) private var dismiss
    @Environment(\.managedObjectContext) private var viewContext

    @State private var showingDeleteAlert = false

    // Get body region logs
    private var bodyRegions: [BodyRegionLog] {
        (entry.bodyRegionLogs as? Set<BodyRegionLog>)?.sorted { ($0.regionID ?? "") < ($1.regionID ?? "") } ?? []
    }

    // Get max pain level from body regions
    private var maxPainLevel: Int16 {
        bodyRegions.map { $0.painLevel }.max() ?? 0
    }

    var painLevelColor: Color {
        switch maxPainLevel {
        case 0...2:
            return .green
        case 3...5:
            return .yellow
        case 6...7:
            return .orange
        case 8...10:
            return .red
        default:
            return .gray
        }
    }

    var painLevelDescription: String {
        switch maxPainLevel {
        case 0:
            return "No Pain"
        case 1...2:
            return "Mild Pain"
        case 3...4:
            return "Moderate Pain"
        case 5...6:
            return "Moderately Severe Pain"
        case 7...8:
            return "Severe Pain"
        case 9...10:
            return "Very Severe Pain"
        default:
            return "Unknown"
        }
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header Card
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text(entry.timestamp ?? Date(), style: .date)
                                .font(.title2)
                                .fontWeight(.bold)
                            Spacer()
                        }

                        Text(entry.timestamp ?? Date(), style: .time)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Overall Pain Level Summary
                    VStack(alignment: .leading, spacing: 15) {
                        Text("Overall Pain Level")
                            .font(.headline)
                            .fontWeight(.semibold)

                        HStack {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("\(maxPainLevel)/10")
                                    .font(.system(size: 36, weight: .bold, design: .rounded))
                                    .foregroundColor(painLevelColor)
                                Text(painLevelDescription)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Body Regions with Individual Pain Levels
                    if !bodyRegions.isEmpty {
                        VStack(alignment: .leading, spacing: 15) {
                            HStack {
                                Image(systemName: "figure.arms.open")
                                    .foregroundColor(.blue)
                                Text("Body Regions (\(bodyRegions.count))")
                                    .font(.headline)
                                    .fontWeight(.semibold)
                            }

                            ForEach(bodyRegions, id: \.id) { region in
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(region.regionID ?? "Unknown")
                                            .font(.body)
                                            .fontWeight(.medium)

                                        if region.stiffnessDuration > 0 {
                                            Text("Stiffness: \(region.stiffnessDuration) min")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }

                                        HStack(spacing: 8) {
                                            if region.swelling {
                                                Label("Swelling", systemImage: "drop.fill")
                                                    .font(.caption2)
                                                    .foregroundColor(.blue)
                                            }
                                            if region.warmth {
                                                Label("Warmth", systemImage: "flame.fill")
                                                    .font(.caption2)
                                                    .foregroundColor(.orange)
                                            }
                                        }
                                    }

                                    Spacer()

                                    // Pain level indicator
                                    VStack(spacing: 2) {
                                        Text("\(region.painLevel)")
                                            .font(.title3)
                                            .fontWeight(.bold)
                                            .foregroundColor(colorForPainLevel(region.painLevel))
                                        Text("/10")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(colorForPainLevel(region.painLevel).opacity(0.15))
                                    .cornerRadius(8)
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 12)
                                .background(Color(.systemGray5))
                                .cornerRadius(10)
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }

                    // Pain Type
                    if let painType = entry.painType, !painType.isEmpty {
                        DetailSection(title: "Pain Type", content: painType, icon: "waveform.path")
                    }

                    // Trigger
                    if let trigger = entry.painTriggers, !trigger.isEmpty {
                        DetailSection(title: "Trigger", content: trigger, icon: "bolt")
                    }

                    // Notes
                    if let notes = entry.notes, !notes.isEmpty {
                        DetailSection(title: "Notes", content: notes, icon: "note.text")
                    }
                }
                .padding()
            }
            .navigationTitle("Pain Entry")
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
                Text("Are you sure you want to delete this pain entry? This action cannot be undone.")
            }
        }
    }

    private func colorForPainLevel(_ level: Int16) -> Color {
        switch level {
        case 0...2: return .green
        case 3...5: return .yellow
        case 6...7: return .orange
        case 8...10: return .red
        default: return .gray
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

struct PainTrackingHistoryView_Previews: PreviewProvider {
    static var previews: some View {
        PainTrackingHistoryView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}