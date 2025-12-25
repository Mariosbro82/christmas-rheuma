//
//  PainMapViewModel.swift
//  InflamAI
//
//  ViewModel for pain location tracking
//

import Foundation
import CoreData
import SwiftUI

@MainActor
class PainMapViewModel: ObservableObject {
    @Published var selectedLocations: Set<Int> = []
    @Published var intensityMap: [Int: Int] = [:] // location ID -> intensity (1-10)

    private let context: NSManagedObjectContext

    var hasSelections: Bool {
        !selectedLocations.isEmpty
    }

    init(context: NSManagedObjectContext) {
        self.context = context
        loadTodaysPainMap()
    }

    func isLocationSelected(_ id: Int) -> Bool {
        selectedLocations.contains(id)
    }

    func getIntensity(_ id: Int) -> Int {
        // FIXED: Return 0 if intensity not explicitly set
        // 0 = "not rated yet" - UI should prompt user to set intensity
        intensityMap[id] ?? 0
    }

    func toggleLocation(_ id: Int) {
        if selectedLocations.contains(id) {
            // If already selected, show intensity picker
            // For now, just toggle selection
            selectedLocations.remove(id)
            intensityMap.removeValue(forKey: id)
        } else {
            selectedLocations.insert(id)
            // FIXED: Don't auto-assign fake "moderate" pain
            // User must explicitly set intensity via setIntensity()
            // intensityMap not set here - getIntensity returns 0 until user rates
        }
    }

    func setIntensity(_ id: Int, intensity: Int) {
        intensityMap[id] = intensity
    }

    func savePainMap() async {
        guard !selectedLocations.isEmpty else { return }

        // First, create the symptom log and body region logs synchronously
        let log: SymptomLog = await context.perform {
            let today = Calendar.current.startOfDay(for: Date())
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.predicate = NSPredicate(format: "timestamp >= %@", today as NSDate)
            request.fetchLimit = 1

            let symptomLog: SymptomLog
            if let existing = try? self.context.fetch(request).first {
                symptomLog = existing
            } else {
                symptomLog = SymptomLog(context: self.context)
                symptomLog.id = UUID()
                symptomLog.timestamp = Date()
                symptomLog.source = "pain_map"
            }

            // Delete existing body region logs
            if let regions = symptomLog.bodyRegionLogs as? Set<BodyRegionLog> {
                for region in regions {
                    self.context.delete(region)
                }
            }

            // Create new body region logs
            for locationID in self.selectedLocations {
                let regionLog = BodyRegionLog(context: self.context)
                regionLog.id = UUID()
                regionLog.regionID = self.regionNameForID(locationID)
                // FIXED: Use 0 if intensity not set, not fake "5"
                // 0 = "location selected but not rated" - honest data
                regionLog.painLevel = Int16(self.intensityMap[locationID] ?? 0)
                regionLog.symptomLog = symptomLog
            }

            return symptomLog
        }

        // Attach context data asynchronously (outside of perform block)
        await attachContextData(to: log)

        // Save the context
        await context.perform {
            do {
                try self.context.save()
                print("✅ Saved pain map with \(self.selectedLocations.count) locations + context snapshot")
            } catch {
                print("❌ CRITICAL: Failed to save pain map: \(error)")
            }
        }
    }

    /// FIXED: Attach environmental and biometric context to symptom log
    private func attachContextData(to log: SymptomLog) async {
        let snapshot = ContextSnapshot(context: context)
        snapshot.id = UUID()
        snapshot.timestamp = Date()

        let today = Date()

        // Fetch weather data (Open-Meteo - FREE, no API key)
        do {
            let weather = try await OpenMeteoService.shared.fetchCurrentWeather()
            snapshot.barometricPressure = weather.pressure
            snapshot.humidity = Int16(weather.humidity)
            snapshot.temperature = weather.temperature
            snapshot.pressureChange12h = weather.pressureChange12h
        } catch {
            print("⚠️ PainMap: Weather fetch failed")
        }

        // Fetch HealthKit data
        do {
            let biometrics = try await HealthKitService.shared.fetchAllBiometrics(for: today)
            snapshot.restingHeartRate = Int16(biometrics.restingHeartRate)
            snapshot.stepCount = Int32(biometrics.stepCount)
            snapshot.hrvValue = biometrics.hrvValue
            snapshot.sleepEfficiency = biometrics.sleep.efficiency
        } catch {
            print("⚠️ PainMap: HealthKit fetch failed")
        }

        log.contextSnapshot = snapshot
    }

    private func loadTodaysPainMap() {
        Task {
            let today = Calendar.current.startOfDay(for: Date())
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.predicate = NSPredicate(format: "timestamp >= %@", today as NSDate)
            request.fetchLimit = 1
            request.relationshipKeyPathsForPrefetching = ["bodyRegionLogs"]

            let result = await context.perform {
                guard let log = try? self.context.fetch(request).first,
                      let regions = log.bodyRegionLogs as? Set<BodyRegionLog> else {
                    return (Set<Int>(), [Int: Int]())
                }

                var locations = Set<Int>()
                var intensities = [Int: Int]()

                for regionLog in regions {
                    if let regionName = regionLog.regionID,
                       let locationID = self.locationIDForRegionName(regionName) {
                        locations.insert(locationID)
                        intensities[locationID] = Int(regionLog.painLevel)
                    }
                }

                return (locations, intensities)
            }

            await MainActor.run {
                self.selectedLocations = result.0
                self.intensityMap = result.1
                self.objectWillChange.send()
            }
        }
    }

    // Map location IDs to anatomical region names
    nonisolated private func regionNameForID(_ id: Int) -> String {
        switch id {
        // Front view
        case 1: return "head"
        case 2: return "neck_upper"
        case 3: return "neck_mid"
        case 4: return "neck_lower"
        case 5: return "left_shoulder_front"
        case 6: return "right_shoulder_front"
        case 7: return "left_shoulder_lower"
        case 8: return "right_shoulder_lower"
        case 9...12: return "left_arm_\(id - 8)"
        case 13...16: return "right_arm_\(id - 12)"
        case 17: return "chest_upper"
        case 18: return "chest_mid"
        case 19: return "abdomen_upper"
        case 20: return "abdomen_mid"
        case 21: return "abdomen_lower"
        case 22: return "pelvis"
        case 23...26: return "left_leg_front_\(id - 22)"
        case 27...30: return "right_leg_front_\(id - 26)"

        // Back view
        case 31: return "head_back"
        case 32: return "neck_back_upper"
        case 33: return "upper_back_thoracic"
        case 34: return "mid_back"
        case 35: return "left_shoulder_back"
        case 36: return "right_shoulder_back"
        case 37: return "left_shoulder_blade"
        case 38: return "right_shoulder_blade"
        case 39...42: return "left_arm_back_\(id - 38)"
        case 43...46: return "right_arm_back_\(id - 42)"
        case 47: return "thoracic_spine_upper"
        case 48: return "thoracic_spine_mid"
        case 49: return "lumbar_spine_upper"
        case 50: return "lumbar_spine_mid"
        case 51: return "lumbar_spine_lower"
        case 52: return "sacral_spine"
        case 53...56: return "left_leg_back_\(id - 52)"
        case 57...60: return "right_leg_back_\(id - 56)"

        default: return "unknown_\(id)"
        }
    }

    nonisolated private func locationIDForRegionName(_ name: String) -> Int? {
        // FIXED: Complete reverse mapping for all 60 locations
        switch name {
        // Front view
        case "head": return 1
        case "neck_upper": return 2
        case "neck_mid": return 3
        case "neck_lower": return 4
        case "left_shoulder_front": return 5
        case "right_shoulder_front": return 6
        case "left_shoulder_lower": return 7
        case "right_shoulder_lower": return 8
        case "left_arm_1": return 9
        case "left_arm_2": return 10
        case "left_arm_3": return 11
        case "left_arm_4": return 12
        case "right_arm_1": return 13
        case "right_arm_2": return 14
        case "right_arm_3": return 15
        case "right_arm_4": return 16
        case "chest_upper": return 17
        case "chest_mid": return 18
        case "abdomen_upper": return 19
        case "abdomen_mid": return 20
        case "abdomen_lower": return 21
        case "pelvis": return 22
        case "left_leg_front_1": return 23
        case "left_leg_front_2": return 24
        case "left_leg_front_3": return 25
        case "left_leg_front_4": return 26
        case "right_leg_front_1": return 27
        case "right_leg_front_2": return 28
        case "right_leg_front_3": return 29
        case "right_leg_front_4": return 30

        // Back view
        case "head_back": return 31
        case "neck_back_upper": return 32
        case "upper_back_thoracic": return 33
        case "mid_back": return 34
        case "left_shoulder_back": return 35
        case "right_shoulder_back": return 36
        case "left_shoulder_blade": return 37
        case "right_shoulder_blade": return 38
        case "left_arm_back_1": return 39
        case "left_arm_back_2": return 40
        case "left_arm_back_3": return 41
        case "left_arm_back_4": return 42
        case "right_arm_back_1": return 43
        case "right_arm_back_2": return 44
        case "right_arm_back_3": return 45
        case "right_arm_back_4": return 46
        case "thoracic_spine_upper": return 47
        case "thoracic_spine_mid": return 48
        case "lumbar_spine_upper": return 49
        case "lumbar_spine_mid": return 50
        case "lumbar_spine_lower": return 51
        case "sacral_spine": return 52
        case "left_leg_back_1": return 53
        case "left_leg_back_2": return 54
        case "left_leg_back_3": return 55
        case "left_leg_back_4": return 56
        case "right_leg_back_1": return 57
        case "right_leg_back_2": return 58
        case "right_leg_back_3": return 59
        case "right_leg_back_4": return 60

        default: return nil
        }
    }
}
