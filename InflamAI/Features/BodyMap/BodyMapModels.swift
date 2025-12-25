//
//  BodyMapModels.swift
//  InflamAI
//
//  Models for body map pain tracking
//

import Foundation
import CoreData
import SwiftUI

/// ViewModel for InteractiveBodyMapView - uses simplified 14-region SpineBodyRegion enum
/// Note: This is SEPARATE from BodyMapViewModel.swift which uses the full 47-region BodyRegion enum
@MainActor
class InteractiveBodyMapViewModel: ObservableObject {
    @Published var activeRegions: [(region: SpineBodyRegion, painLevel: Int)] = []

    private let context: NSManagedObjectContext
    private var painMap: [SpineBodyRegion: Int] = [:]

    init(context: NSManagedObjectContext) {
        self.context = context
        loadTodaysPain()
    }

    func getPainLevel(for region: SpineBodyRegion) -> Int {
        return painMap[region] ?? 0
    }

    func setPain(for region: SpineBodyRegion, level: Int) {
        if level > 0 {
            painMap[region] = level
        } else {
            painMap.removeValue(forKey: region)
        }
        updateActiveRegions()
    }

    func removePain(for region: SpineBodyRegion) {
        painMap.removeValue(forKey: region)
        updateActiveRegions()
    }

    func clearAllRegions() {
        painMap.removeAll()
        updateActiveRegions()
    }

    private func updateActiveRegions() {
        activeRegions = painMap.map { (region: $0.key, painLevel: $0.value) }
            .sorted { $0.painLevel > $1.painLevel }
    }

    func saveToday() async {
        guard !painMap.isEmpty else { return }

        await context.perform {
            // Create or update today's symptom log
            let calendar = Calendar.current
            let today = calendar.startOfDay(for: Date())
            // FIXED: Use proper date boundary (< tomorrow instead of open-ended >= today)
            let tomorrow = calendar.date(byAdding: .day, value: 1, to: today) ?? Date()
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp < %@", today as NSDate, tomorrow as NSDate)
            request.fetchLimit = 1

            let log: SymptomLog
            // FIXED: Proper error handling instead of silent try?
            do {
                if let existing = try self.context.fetch(request).first {
                    log = existing
                } else {
                    log = SymptomLog(context: self.context)
                    log.id = UUID()
                    log.timestamp = Date()
                    log.source = "body_map"
                }
            } catch {
                print("❌ Failed to fetch existing log: \(error)")
                // Create new log on fetch failure
                log = SymptomLog(context: self.context)
                log.id = UUID()
                log.timestamp = Date()
                log.source = "body_map"
            }

            // Delete existing body region logs for today
            if let regions = log.bodyRegionLogs as? Set<BodyRegionLog> {
                for region in regions {
                    self.context.delete(region)
                }
            }

            // Create new body region logs
            for (region, painLevel) in self.painMap {
                let regionLog = BodyRegionLog(context: self.context)
                regionLog.id = UUID()
                regionLog.regionID = region.rawValue  // FIXED: was 'region', should be 'regionID'
                regionLog.painLevel = Int16(painLevel)
                regionLog.symptomLog = log
            }

            // FIXED: Attach ContextSnapshot for ML feature extraction
            await self.attachContextData(to: log)

            // FIXED: Proper error handling instead of silent try?
            do {
                try self.context.save()
                print("✅ Saved body map with \(self.painMap.count) regions + context snapshot")
            } catch {
                print("❌ CRITICAL: Failed to save body map data: \(error)")
                // In production, trigger error reporting here
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
            print("⚠️ BodyMap: Weather fetch failed")
        }

        // Fetch HealthKit data
        do {
            let biometrics = try await HealthKitService.shared.fetchAllBiometrics(for: today)
            snapshot.restingHeartRate = Int16(biometrics.restingHeartRate)
            snapshot.stepCount = Int32(biometrics.stepCount)
            snapshot.hrvValue = biometrics.hrvValue
            snapshot.sleepEfficiency = biometrics.sleep.efficiency
        } catch {
            print("⚠️ BodyMap: HealthKit fetch failed")
        }

        log.contextSnapshot = snapshot
    }

    private func loadTodaysPain() {
        Task {
            await context.perform {
                let calendar = Calendar.current
                let today = calendar.startOfDay(for: Date())
                // FIXED: Use proper date boundary
                let tomorrow = calendar.date(byAdding: .day, value: 1, to: today) ?? Date()
                let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
                request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp < %@", today as NSDate, tomorrow as NSDate)
                request.fetchLimit = 1
                request.relationshipKeyPathsForPrefetching = ["bodyRegionLogs"]

                // FIXED: Proper error handling instead of silent try?
                do {
                    if let log = try self.context.fetch(request).first,
                       let regions = log.bodyRegionLogs as? Set<BodyRegionLog> {
                        for regionLog in regions {
                            if let regionName = regionLog.regionID,
                               let region = SpineBodyRegion(rawValue: regionName) {
                                self.painMap[region] = Int(regionLog.painLevel)
                            }
                        }
                    }
                } catch {
                    print("⚠️ Failed to load today's pain data: \(error)")
                }

                await MainActor.run {
                    self.updateActiveRegions()
                }
            }
        }
    }
}

// MARK: - Body View Selection

enum BodyView: String, CaseIterable {
    case front
    case back
}

// MARK: - Body Region Model

enum SpineBodyRegion: String, CaseIterable, Identifiable {
    // Spine
    case cervicalSpine = "cervical_spine"
    case thoracicSpine = "thoracic_spine"
    case lumbarSpine = "lumbar_spine"
    case sacroiliac = "sacroiliac"

    // Peripheral
    case leftShoulder = "left_shoulder"
    case rightShoulder = "right_shoulder"
    case leftHip = "left_hip"
    case rightHip = "right_hip"
    case leftKnee = "left_knee"
    case rightKnee = "right_knee"
    case leftAnkle = "left_ankle"
    case rightAnkle = "right_ankle"

    // Chest
    case chest = "chest"
    case ribs = "ribs"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .cervicalSpine: return "Neck (Cervical Spine)"
        case .thoracicSpine: return "Upper Back (Thoracic)"
        case .lumbarSpine: return "Lower Back (Lumbar)"
        case .sacroiliac: return "SI Joints"
        case .leftShoulder: return "Left Shoulder"
        case .rightShoulder: return "Right Shoulder"
        case .leftHip: return "Left Hip"
        case .rightHip: return "Right Hip"
        case .leftKnee: return "Left Knee"
        case .rightKnee: return "Right Knee"
        case .leftAnkle: return "Left Ankle"
        case .rightAnkle: return "Right Ankle"
        case .chest: return "Chest"
        case .ribs: return "Ribs"
        }
    }

    var isFrontView: Bool {
        switch self {
        case .cervicalSpine, .chest, .ribs, .leftShoulder, .rightShoulder,
             .leftHip, .rightHip, .leftKnee, .rightKnee, .leftAnkle, .rightAnkle:
            return true
        default:
            return false
        }
    }

    var isBackView: Bool {
        switch self {
        case .cervicalSpine, .thoracicSpine, .lumbarSpine, .sacroiliac,
             .leftShoulder, .rightShoulder, .leftHip, .rightHip:
            return true
        default:
            return false
        }
    }

    // Position on the 2D body model (normalized 0-1)
    var position: CGPoint {
        switch self {
        // Front view positions
        case .cervicalSpine: return CGPoint(x: 0.5, y: 0.15)
        case .chest: return CGPoint(x: 0.5, y: 0.30)
        case .ribs: return CGPoint(x: 0.5, y: 0.35)
        case .leftShoulder: return CGPoint(x: 0.25, y: 0.22)
        case .rightShoulder: return CGPoint(x: 0.75, y: 0.22)
        case .leftHip: return CGPoint(x: 0.40, y: 0.55)
        case .rightHip: return CGPoint(x: 0.60, y: 0.55)
        case .leftKnee: return CGPoint(x: 0.42, y: 0.75)
        case .rightKnee: return CGPoint(x: 0.58, y: 0.75)
        case .leftAnkle: return CGPoint(x: 0.42, y: 0.95)
        case .rightAnkle: return CGPoint(x: 0.58, y: 0.95)

        // Back view positions
        case .thoracicSpine: return CGPoint(x: 0.5, y: 0.30)
        case .lumbarSpine: return CGPoint(x: 0.5, y: 0.45)
        case .sacroiliac: return CGPoint(x: 0.5, y: 0.55)
        }
    }
}
