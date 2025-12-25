//
//  BodyMapViewModel.swift
//  InflamAI
//
//  ViewModel for body map with pain analytics
//

import Foundation
import CoreData
import Combine

@MainActor
class BodyMapViewModel: ObservableObject {
    @Published var painData: [String: RegionPainData] = [:]
    @Published var isLoading = false

    private let context: NSManagedObjectContext
    private let timeRange: TimeRange

    init(context: NSManagedObjectContext, timeRange: TimeRange = .week) {
        self.context = context
        self.timeRange = timeRange
    }

    // MARK: - Data Loading

    func loadPainData() {
        isLoading = true

        Task {
            do {
                let startDate = timeRange.startDate
                // FIXED: Use start of tomorrow instead of current Date() to avoid race conditions
                let calendar = Calendar.current
                let tomorrow = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: Date())) ?? Date()

                // FIXED: Use context.perform to ensure Core Data operations happen on correct thread
                let newPainData: [String: RegionPainData] = try await context.perform {
                    // Fetch symptom logs in range
                    let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
                    // FIXED: Use < tomorrow instead of <= Date() to avoid off-by-one issues
                    request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp < %@", startDate as NSDate, tomorrow as NSDate)
                    request.relationshipKeyPathsForPrefetching = ["bodyRegionLogs"]

                    let logs = try self.context.fetch(request)

                    // DEBUG: Log what we found
                    print("🔍 BodyMapViewModel: Fetched \(logs.count) symptom logs from \(startDate) to \(tomorrow)")

                    // Aggregate pain data by region
                    var regionData: [String: [Double]] = [:]

                    for log in logs {
                        if let bodyRegions = log.bodyRegionLogs as? Set<BodyRegionLog> {
                            for regionLog in bodyRegions {
                                let regionID = regionLog.regionID ?? ""
                                let pain = Double(regionLog.painLevel)

                                if regionData[regionID] == nil {
                                    regionData[regionID] = []
                                }
                                regionData[regionID]?.append(pain)
                            }
                        }
                    }

                    // DEBUG: Log region data found
                    print("🔍 BodyMapViewModel: Found \(regionData.count) unique regions with pain data")
                    for (regionID, pains) in regionData.prefix(5) {
                        print("   - Region '\(regionID)': \(pains.count) entries, avg: \(pains.reduce(0.0, +) / Double(pains.count))")
                    }

                    // Calculate averages
                    var result: [String: RegionPainData] = [:]
                    for (regionID, pains) in regionData {
                        let average = pains.reduce(0.0, +) / Double(pains.count)
                        let max = pains.max() ?? 0
                        let count = pains.count

                        result[regionID] = RegionPainData(
                            regionID: regionID,
                            averagePain: average,
                            maxPain: max,
                            entryCount: count
                        )
                    }

                    print("🔍 BodyMapViewModel: Returning \(result.count) region pain entries")
                    return result
                }

                // FIXED: Explicitly update @Published properties on MainActor
                await MainActor.run {
                    self.painData = newPainData
                    self.isLoading = false
                }

            } catch {
                print("Error loading pain data: \(error)")
                await MainActor.run {
                    self.isLoading = false
                }
            }
        }
    }

    func refreshData() {
        loadPainData()
    }

    // MARK: - Logging

    func logPain(region: BodyRegion, painLevel: Int16, stiffness: Int16, swelling: Bool, warmth: Bool, notes: String?) async throws {
        // Create new symptom log
        let log = SymptomLog(context: context)
        log.id = UUID()
        log.timestamp = Date()
        log.source = "manual"

        // Create body region log
        let regionLog = BodyRegionLog(context: context)
        regionLog.id = UUID()
        regionLog.regionID = region.rawValue
        regionLog.painLevel = painLevel
        regionLog.stiffnessDuration = stiffness
        regionLog.swelling = swelling
        regionLog.warmth = warmth
        regionLog.notes = notes
        regionLog.symptomLog = log

        try context.save()

        // Reload data
        loadPainData()
    }
}

// MARK: - Supporting Models

struct RegionPainData {
    let regionID: String
    let averagePain: Double
    let maxPain: Double
    let entryCount: Int
}

enum TimeRange {
    case week
    case month
    case threeMonths

    var startDate: Date {
        let calendar = Calendar.current
        switch self {
        case .week:
            return calendar.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        case .month:
            return calendar.date(byAdding: .day, value: -30, to: Date()) ?? Date()
        case .threeMonths:
            return calendar.date(byAdding: .day, value: -90, to: Date()) ?? Date()
        }
    }
}
