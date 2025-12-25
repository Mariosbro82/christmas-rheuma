//
//  ExportDataView.swift
//  InflamAI
//
//  Data export in multiple formats
//

import SwiftUI
import CoreData
import UniformTypeIdentifiers

struct ExportDataView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var selectedFormat: ExportFormat = .pdf
    @State private var isExporting = false
    @State private var exportedFileURL: URL?
    @State private var showShareSheet = false
    @State private var errorMessage: String?
    @State private var showError = false

    enum ExportFormat: String, CaseIterable {
        case pdf = "PDF Report"
        case json = "JSON Data"
        case csv = "CSV Spreadsheet"

        var icon: String {
            switch self {
            case .pdf: return "doc.text.fill"
            case .json: return "curlybraces"
            case .csv: return "tablecells.fill"
            }
        }

        var description: String {
            switch self {
            case .pdf:
                return "Perfect for doctors - formatted report with charts and summaries"
            case .json:
                return "Full backup - all your data in developer-friendly format"
            case .csv:
                return "For spreadsheets - open in Excel, Numbers, or Google Sheets"
            }
        }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 12) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 50))
                        .foregroundColor(.blue)
                        .padding(.top, 20)

                    Text("Export Your Data")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Choose how you'd like to export your health data. All exports are private and stored locally on your device.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 20)
                }
                .padding(.bottom, 8)

                // Format Selection
                VStack(alignment: .leading, spacing: 16) {
                    Text("Select Export Format")
                        .font(.headline)
                        .padding(.horizontal, 20)

                    ForEach(ExportFormat.allCases, id: \.self) { format in
                        ExportFormatCard(
                            format: format,
                            isSelected: selectedFormat == format,
                            action: {
                                withAnimation(.spring(response: 0.3)) {
                                    selectedFormat = format
                                }
                            }
                        )
                    }
                }

                // Export Button
                Button {
                    exportData()
                } label: {
                    HStack(spacing: 12) {
                        if isExporting {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            Text("Preparing Export...")
                                .fontWeight(.semibold)
                        } else {
                            Image(systemName: "arrow.down.doc.fill")
                                .font(.title3)
                            Text("Export as \(selectedFormat.rawValue)")
                                .fontWeight(.semibold)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(isExporting ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .disabled(isExporting)
                .padding(.horizontal, 20)
                .padding(.top, 8)

                // Privacy Notice
                HStack(spacing: 12) {
                    Image(systemName: "lock.shield.fill")
                        .font(.title2)
                        .foregroundColor(.green)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Your Privacy Matters")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text("Exported data stays on your device. You choose who to share it with.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
                .padding(16)
                .background(Color.green.opacity(0.1))
                .cornerRadius(12)
                .padding(.horizontal, 20)

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("Export Data")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showShareSheet) {
            if let url = exportedFileURL {
                ShareSheet(items: [url])
            }
        }
        .alert("Export Error", isPresented: $showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage ?? "An unknown error occurred")
        }
    }

    // MARK: - Export Functions

    private func exportData() {
        isExporting = true
        errorMessage = nil

        Task {
            do {
                let url: URL

                switch selectedFormat {
                case .pdf:
                    url = try await exportPDF()
                case .json:
                    url = try await exportJSON()
                case .csv:
                    url = try await exportCSV()
                }

                await MainActor.run {
                    exportedFileURL = url
                    isExporting = false
                    showShareSheet = true
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    showError = true
                    isExporting = false
                }
            }
        }
    }

    private func exportPDF() async throws -> URL {
        // Fetch data needed for PDF
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -30, to: endDate) ?? endDate

        let (logs, flares, meds) = await context.perform {
            let logRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            logRequest.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp <= %@",
                                              startDate as NSDate, endDate as NSDate)
            let logs = (try? self.context.fetch(logRequest)) ?? []

            let flareRequest: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
            flareRequest.predicate = NSPredicate(format: "startDate >= %@ AND startDate <= %@",
                                                startDate as NSDate, endDate as NSDate)
            let flares = (try? self.context.fetch(flareRequest)) ?? []

            let medRequest: NSFetchRequest<Medication> = Medication.fetchRequest()
            let meds = (try? self.context.fetch(medRequest)) ?? []

            return (logs, flares, meds)
        }

        // Generate PDF
        let service = PDFExportService()
        let pdfData = service.generateReport(
            patientName: nil,
            dateRange: (start: startDate, end: endDate),
            logs: logs,
            flareEvents: flares,
            medications: meds
        )

        // Save to temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let pdfURL = tempDir.appendingPathComponent("health_export_\(Date().timeIntervalSince1970).pdf")
        try pdfData.write(to: pdfURL)
        return pdfURL
    }

    private func exportJSON() async throws -> URL {
        let jsonData = try await fetchAllData()
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let data = try encoder.encode(jsonData)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("spinalytics_export_\(Date().timeIntervalSince1970).json")

        try data.write(to: tempURL)
        return tempURL
    }

    private func exportCSV() async throws -> URL {
        let data = try await fetchAllData()
        var csv = ""

        // Header
        csv += "Date,Type,BASDAI,Fatigue,Morning Stiffness,Notes\n"

        // Rows
        for log in data.symptomLogs {
            let date = ISO8601DateFormatter().string(from: log.date)
            csv += "\"\(date)\","
            csv += "\"Symptom Log\","
            csv += "\(log.basdaiScore),"
            csv += "\(log.fatigueLevel),"
            csv += "\(log.morningStiffnessMinutes),"
            csv += "\"\(log.notes.replacingOccurrences(of: "\"", with: "\"\""))\"\n"
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("spinalytics_export_\(Date().timeIntervalSince1970).csv")

        try csv.write(to: tempURL, atomically: true, encoding: .utf8)
        return tempURL
    }

    private func fetchAllData() async throws -> ExportData {
        return try await context.perform {
            // Fetch symptom logs
            let logRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            logRequest.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
            let logs = try context.fetch(logRequest)

            // Fetch medications
            let medRequest: NSFetchRequest<Medication> = Medication.fetchRequest()
            let medications = try context.fetch(medRequest)

            // Fetch body region logs
            let regionRequest: NSFetchRequest<BodyRegionLog> = BodyRegionLog.fetchRequest()
            let regions = try context.fetch(regionRequest)

            return ExportData(
                exportDate: Date(),
                symptomLogs: logs.map { ExportableSymptomLog(from: $0) },
                medications: medications.map { ExportableMedication(from: $0) },
                bodyRegions: regions.map { ExportableBodyRegion(from: $0) }
            )
        }
    }
}

// MARK: - Export Format Card

struct ExportFormatCard: View {
    let format: ExportDataView.ExportFormat
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 0) {
                HStack(spacing: 16) {
                    // Icon Circle
                    ZStack {
                        Circle()
                            .fill(isSelected ? Color.blue.opacity(0.15) : Color(.systemGray6))
                            .frame(width: 56, height: 56)

                        Image(systemName: format.icon)
                            .font(.system(size: 24))
                            .foregroundColor(isSelected ? .blue : .gray)
                    }

                    // Content
                    VStack(alignment: .leading, spacing: 6) {
                        Text(format.rawValue)
                            .font(.headline)
                            .fontWeight(.semibold)
                            .foregroundColor(.primary)

                        Text(format.description)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                    }

                    Spacer()

                    // Selection indicator
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .font(.title2)
                        .foregroundColor(isSelected ? .blue : Color(.systemGray4))
                }
                .padding(18)
            }
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color(.systemBackground))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .strokeBorder(isSelected ? Color.blue : Color(.systemGray5), lineWidth: isSelected ? 2.5 : 1)
                    )
                    .shadow(color: isSelected ? Color.blue.opacity(0.15) : Color.black.opacity(0.05),
                            radius: isSelected ? 8 : 3,
                            x: 0,
                            y: isSelected ? 4 : 2)
            )
            .padding(.horizontal, 20)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Export Data Models

struct ExportData: Codable {
    let exportDate: Date
    let symptomLogs: [ExportableSymptomLog]
    let medications: [ExportableMedication]
    let bodyRegions: [ExportableBodyRegion]
}

struct ExportableSymptomLog: Codable {
    let date: Date
    let basdaiScore: Double
    let fatigueLevel: Int
    let morningStiffnessMinutes: Int
    let notes: String

    init(from log: SymptomLog) {
        self.date = log.timestamp ?? Date()
        self.basdaiScore = log.basdaiScore
        self.fatigueLevel = Int(log.fatigueLevel)
        self.morningStiffnessMinutes = Int(log.morningStiffnessMinutes)
        self.notes = log.notes ?? ""
    }
}

struct ExportableMedication: Codable {
    let name: String
    let dosage: Double
    let frequency: String

    init(from medication: Medication) {
        self.name = medication.name ?? ""
        self.dosage = medication.dosage
        self.frequency = medication.frequency ?? ""
    }
}

struct ExportableBodyRegion: Codable {
    let regionID: String
    let painLevel: Int

    init(from region: BodyRegionLog) {
        self.regionID = region.regionID ?? ""
        self.painLevel = Int(region.painLevel)
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Preview

#Preview {
    NavigationView {
        ExportDataView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}
