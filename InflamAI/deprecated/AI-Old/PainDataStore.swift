//
//  PainDataStore.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import Foundation
import Combine
import CoreData
import HealthKit

class PainDataStore: ObservableObject {
    static let shared = PainDataStore()
    
    @Published var painEntries: [PainEntry] = []
    @Published var medicationEntries: [MedicationEntry] = []
    @Published var currentPainRegions: Set<BodyRegion> = []
    @Published var currentPainIntensity: [BodyRegion: Double] = [:]
    @Published var globalPainLevel: Double = 0.0
    @Published var isLoading = false
    @Published var lastSyncDate: Date?
    
    private var cancellables = Set<AnyCancellable>()
    private let userDefaults = UserDefaults.standard
    private let fileManager = FileManager.default
    
    // Core Data
    private lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "PainDataModel")
        container.loadPersistentStores { _, error in
            if let error = error {
                print("Core Data error: \(error)")
            }
        }
        return container
    }
    
    private var context: NSManagedObjectContext {
        return persistentContainer.viewContext
    }
    
    // File URLs
    private var documentsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
    }
    
    private var painEntriesURL: URL {
        documentsDirectory.appendingPathComponent("painEntries.json")
    }
    
    private var medicationEntriesURL: URL {
        documentsDirectory.appendingPathComponent("medicationEntries.json")
    }
    
    private var currentStateURL: URL {
        documentsDirectory.appendingPathComponent("currentPainState.json")
    }
    
    private init() {
        loadData()
        setupNotificationObservers()
        startPeriodicSync()
    }
    
    // MARK: - Setup
    
    private func setupNotificationObservers() {
        // Voice command notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleVoiceCommandSetPainLevel(_:)),
            name: .voiceCommandSetPainLevel,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleVoiceCommandAddPainRegion(_:)),
            name: .voiceCommandAddPainRegion,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleVoiceCommandRemovePainRegion(_:)),
            name: .voiceCommandRemovePainRegion,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleVoiceCommandSaveEntry(_:)),
            name: .voiceCommandSaveEntry,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleVoiceCommandLogMedication(_:)),
            name: .voiceCommandLogMedication,
            object: nil
        )
        
        // App lifecycle notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(saveDataOnBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(loadDataOnForeground),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )
    }
    
    private func startPeriodicSync() {
        Timer.publish(every: 300, on: .main, in: .common) // Every 5 minutes
            .autoconnect()
            .sink { [weak self] _ in
                self?.saveData()
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Voice Command Handlers
    
    @objc private func handleVoiceCommandSetPainLevel(_ notification: Notification) {
        guard let level = notification.userInfo?["level"] as? Int else { return }
        setGlobalPainLevel(Double(level))
    }
    
    @objc private func handleVoiceCommandAddPainRegion(_ notification: Notification) {
        guard let region = notification.userInfo?["region"] as? BodyRegion else { return }
        let level = notification.userInfo?["level"] as? Int ?? 5
        
        addPainRegion(region, intensity: Double(level))
    }
    
    @objc private func handleVoiceCommandRemovePainRegion(_ notification: Notification) {
        guard let region = notification.userInfo?["region"] as? BodyRegion else { return }
        removePainRegion(region)
    }
    
    @objc private func handleVoiceCommandSaveEntry(_ notification: Notification) {
        savePainEntry()
    }
    
    @objc private func handleVoiceCommandLogMedication(_ notification: Notification) {
        let medication = notification.userInfo?["medication"] as? String ?? "Unknown"
        let dose = notification.userInfo?["dose"] as? String ?? ""
        
        logMedication(name: medication, dose: dose)
    }
    
    @objc private func saveDataOnBackground() {
        saveData()
    }
    
    @objc private func loadDataOnForeground() {
        loadData()
    }
    
    // MARK: - Pain Management
    
    func setGlobalPainLevel(_ level: Double) {
        globalPainLevel = max(0, min(10, level))
        saveCurrentState()
    }
    
    func addPainRegion(_ region: BodyRegion, intensity: Double = 5.0) {
        currentPainRegions.insert(region)
        currentPainIntensity[region] = max(0, min(10, intensity))
        saveCurrentState()
    }
    
    func removePainRegion(_ region: BodyRegion) {
        currentPainRegions.remove(region)
        currentPainIntensity.removeValue(forKey: region)
        saveCurrentState()
    }
    
    func updatePainIntensity(for region: BodyRegion, intensity: Double) {
        if currentPainRegions.contains(region) {
            currentPainIntensity[region] = max(0, min(10, intensity))
            saveCurrentState()
        }
    }
    
    func clearAllPain() {
        currentPainRegions.removeAll()
        currentPainIntensity.removeAll()
        globalPainLevel = 0.0
        saveCurrentState()
    }
    
    func savePainEntry(notes: String = "", triggers: [String] = [], painType: String = "Aching", mood: String = "") {
        let entry = PainEntry(
            id: UUID(),
            timestamp: Date(),
            painLevel: globalPainLevel,
            affectedRegions: Array(currentPainRegions),
            regionIntensities: currentPainIntensity,
            notes: notes,
            triggers: triggers,
            painType: painType,
            mood: mood,
            weather: getCurrentWeather(),
            medications: getRecentMedications()
        )
        
        painEntries.append(entry)
        saveData()
        
        // Trigger AI analysis
        AIMLEngine.shared.analyzePainPatterns(painData: painEntries) { _ in
            // Analysis complete
        }
    }
    
    // MARK: - Medication Management
    
    func logMedication(name: String, dose: String = "", notes: String = "") {
        let entry = MedicationEntry(
            id: UUID(),
            timestamp: Date(),
            medicationName: name,
            dose: dose,
            notes: notes,
            painLevelBefore: globalPainLevel,
            painLevelAfter: nil // Will be updated later
        )
        
        medicationEntries.append(entry)
        saveData()
    }
    
    func updateMedicationEffectiveness(medicationId: UUID, painLevelAfter: Double, effectiveness: Int) {
        if let index = medicationEntries.firstIndex(where: { $0.id == medicationId }) {
            medicationEntries[index].painLevelAfter = painLevelAfter
            medicationEntries[index].effectiveness = effectiveness
            saveData()
        }
    }
    
    private func getRecentMedications() -> [String] {
        let recentCutoff = Date().addingTimeInterval(-24 * 60 * 60) // Last 24 hours
        return medicationEntries
            .filter { $0.timestamp >= recentCutoff }
            .map { "\($0.medicationName) \($0.dose)" }
    }
    
    // MARK: - Data Persistence
    
    private func saveData() {
        savePainEntries()
        saveMedicationEntries()
        saveCurrentState()
        lastSyncDate = Date()
    }
    
    private func loadData() {
        isLoading = true
        
        loadPainEntries()
        loadMedicationEntries()
        loadCurrentState()
        
        isLoading = false
    }
    
    private func savePainEntries() {
        do {
            let data = try JSONEncoder().encode(painEntries)
            try data.write(to: painEntriesURL)
        } catch {
            print("Failed to save pain entries: \(error)")
        }
    }
    
    private func loadPainEntries() {
        do {
            let data = try Data(contentsOf: painEntriesURL)
            painEntries = try JSONDecoder().decode([PainEntry].self, from: data)
        } catch {
            print("Failed to load pain entries: \(error)")
            painEntries = []
        }
    }
    
    private func saveMedicationEntries() {
        do {
            let data = try JSONEncoder().encode(medicationEntries)
            try data.write(to: medicationEntriesURL)
        } catch {
            print("Failed to save medication entries: \(error)")
        }
    }
    
    private func loadMedicationEntries() {
        do {
            let data = try Data(contentsOf: medicationEntriesURL)
            medicationEntries = try JSONDecoder().decode([MedicationEntry].self, from: data)
        } catch {
            print("Failed to load medication entries: \(error)")
            medicationEntries = []
        }
    }
    
    private func saveCurrentState() {
        let state = CurrentPainState(
            painRegions: Array(currentPainRegions),
            painIntensity: currentPainIntensity,
            globalPainLevel: globalPainLevel,
            lastUpdated: Date()
        )
        
        do {
            let data = try JSONEncoder().encode(state)
            try data.write(to: currentStateURL)
        } catch {
            print("Failed to save current state: \(error)")
        }
    }
    
    private func loadCurrentState() {
        do {
            let data = try Data(contentsOf: currentStateURL)
            let state = try JSONDecoder().decode(CurrentPainState.self, from: data)
            
            currentPainRegions = Set(state.painRegions)
            currentPainIntensity = state.painIntensity
            globalPainLevel = state.globalPainLevel
        } catch {
            print("Failed to load current state: \(error)")
            // Initialize with default values
            currentPainRegions = []
            currentPainIntensity = [:]
            globalPainLevel = 0.0
        }
    }
    
    // MARK: - Data Analysis
    
    func getPainTrend(days: Int = 7) -> [PainDataPoint] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()
        
        return painEntries
            .filter { $0.timestamp >= cutoffDate }
            .map { entry in
                PainDataPoint(
                    date: entry.timestamp,
                    painLevel: entry.painLevel,
                    affectedRegions: entry.affectedRegions.count
                )
            }
            .sorted { $0.date < $1.date }
    }
    
    func getAveragePainLevel(days: Int = 7) -> Double {
        let recentEntries = getPainTrend(days: days)
        guard !recentEntries.isEmpty else { return 0.0 }
        
        let total = recentEntries.reduce(0) { $0 + $1.painLevel }
        return total / Double(recentEntries.count)
    }
    
    func getMostAffectedRegions(days: Int = 30) -> [(BodyRegion, Int)] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()
        
        var regionCounts: [BodyRegion: Int] = [:]
        
        for entry in painEntries.filter({ $0.timestamp >= cutoffDate }) {
            for region in entry.affectedRegions {
                regionCounts[region, default: 0] += 1
            }
        }
        
        return regionCounts.sorted { $0.value > $1.value }
    }
    
    func getMedicationEffectiveness() -> [(String, Double)] {
        var medicationEffects: [String: [Double]] = [:]
        
        for entry in medicationEntries {
            if let afterPain = entry.painLevelAfter {
                let improvement = entry.painLevelBefore - afterPain
                medicationEffects[entry.medicationName, default: []].append(improvement)
            }
        }
        
        return medicationEffects.compactMap { name, improvements in
            guard !improvements.isEmpty else { return nil }
            let average = improvements.reduce(0, +) / Double(improvements.count)
            return (name, average)
        }.sorted { $0.1 > $1.1 }
    }
    
    // MARK: - Export and Import
    
    func exportData() -> Data? {
        let exportData = PainDataExport(
            painEntries: painEntries,
            medicationEntries: medicationEntries,
            exportDate: Date(),
            version: "1.0"
        )
        
        do {
            return try JSONEncoder().encode(exportData)
        } catch {
            print("Failed to export data: \(error)")
            return nil
        }
    }
    
    func importData(_ data: Data) -> Bool {
        do {
            let importData = try JSONDecoder().decode(PainDataExport.self, from: data)
            
            // Merge data (avoid duplicates)
            let existingIds = Set(painEntries.map { $0.id })
            let newPainEntries = importData.painEntries.filter { !existingIds.contains($0.id) }
            painEntries.append(contentsOf: newPainEntries)
            
            let existingMedicationIds = Set(medicationEntries.map { $0.id })
            let newMedicationEntries = importData.medicationEntries.filter { !existingMedicationIds.contains($0.id) }
            medicationEntries.append(contentsOf: newMedicationEntries)
            
            saveData()
            return true
        } catch {
            print("Failed to import data: \(error)")
            return false
        }
    }
    
    // MARK: - Utility Methods
    
    private func getCurrentWeather() -> String {
        // This would integrate with a weather API
        // For now, return a placeholder
        return "Unknown"
    }
    
    func clearAllData() {
        painEntries.removeAll()
        medicationEntries.removeAll()
        clearAllPain()
        saveData()
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}

// MARK: - Data Models

struct PainEntry: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let painLevel: Double
    let affectedRegions: [BodyRegion]
    let regionIntensities: [BodyRegion: Double]
    let notes: String
    let triggers: [String]
    let painType: String
    let mood: String
    let weather: String
    let medications: [String]
}

struct MedicationEntry: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let medicationName: String
    let dose: String
    let notes: String
    let painLevelBefore: Double
    var painLevelAfter: Double?
    var effectiveness: Int? // 1-5 scale
}

struct CurrentPainState: Codable {
    let painRegions: [BodyRegion]
    let painIntensity: [BodyRegion: Double]
    let globalPainLevel: Double
    let lastUpdated: Date
}

struct PainDataExport: Codable {
    let painEntries: [PainEntry]
    let medicationEntries: [MedicationEntry]
    let exportDate: Date
    let version: String
}

struct PainDataPoint: Codable {
    let date: Date
    let painLevel: Double
    let affectedRegions: Int
}