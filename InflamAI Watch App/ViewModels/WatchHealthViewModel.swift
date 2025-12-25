//
//  WatchHealthViewModel.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//

import Foundation
import Combine
import HealthKit

@MainActor
class WatchHealthViewModel: ObservableObject {
    @Published var heartRate: Double?
    @Published var hrv: Double?
    @Published var steps: Double?
    @Published var activeEnergy: Double?

    private let healthStore = HKHealthStore()
    private var heartRateQuery: HKAnchoredObjectQuery?

    func startMonitoring() async {
        await requestAuthorization()
        await refresh()
        startHeartRateQuery()
    }

    func refresh() async {
        async let hr = fetchLatestHeartRate()
        async let hrvValue = fetchLatestHRV()
        async let stepsValue = fetchTodaySteps()
        async let energyValue = fetchTodayActiveEnergy()

        heartRate = await hr
        hrv = await hrvValue
        steps = await stepsValue
        activeEnergy = await energyValue
    }

    private func requestAuthorization() async {
        let types: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!
        ]

        try? await healthStore.requestAuthorization(toShare: [], read: types)
    }

    private func fetchLatestHeartRate() async -> Double? {
        let type = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let predicate = HKQuery.predicateForSamples(
            withStart: Date().addingTimeInterval(-3600),  // Last hour
            end: Date()
        )

        return await withCheckedContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: type,
                predicate: predicate,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
            ) { _, samples, _ in
                if let sample = samples?.first as? HKQuantitySample {
                    let value = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
                    continuation.resume(returning: value)
                } else {
                    continuation.resume(returning: nil)
                }
            }
            healthStore.execute(query)
        }
    }

    private func fetchLatestHRV() async -> Double? {
        let type = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        let predicate = HKQuery.predicateForSamples(
            withStart: Date().addingTimeInterval(-86400),  // Last 24 hours
            end: Date()
        )

        return await withCheckedContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: type,
                predicate: predicate,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
            ) { _, samples, _ in
                if let sample = samples?.first as? HKQuantitySample {
                    let value = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                    continuation.resume(returning: value)
                } else {
                    continuation.resume(returning: nil)
                }
            }
            healthStore.execute(query)
        }
    }

    private func fetchTodaySteps() async -> Double? {
        let type = HKQuantityType.quantityType(forIdentifier: .stepCount)!
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: Date())

        let predicate = HKQuery.predicateForSamples(
            withStart: startOfDay,
            end: Date()
        )

        return await withCheckedContinuation { continuation in
            let query = HKStatisticsQuery(
                quantityType: type,
                quantitySamplePredicate: predicate,
                options: .cumulativeSum
            ) { _, statistics, _ in
                if let sum = statistics?.sumQuantity() {
                    let value = sum.doubleValue(for: HKUnit.count())
                    continuation.resume(returning: value)
                } else {
                    continuation.resume(returning: nil)
                }
            }
            healthStore.execute(query)
        }
    }

    private func fetchTodayActiveEnergy() async -> Double? {
        let type = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: Date())

        let predicate = HKQuery.predicateForSamples(
            withStart: startOfDay,
            end: Date()
        )

        return await withCheckedContinuation { continuation in
            let query = HKStatisticsQuery(
                quantityType: type,
                quantitySamplePredicate: predicate,
                options: .cumulativeSum
            ) { _, statistics, _ in
                if let sum = statistics?.sumQuantity() {
                    let value = sum.doubleValue(for: HKUnit.kilocalorie())
                    continuation.resume(returning: value)
                } else {
                    continuation.resume(returning: nil)
                }
            }
            healthStore.execute(query)
        }
    }

    private func startHeartRateQuery() {
        // Create anchor query for continuous updates
        let type = HKQuantityType.quantityType(forIdentifier: .heartRate)!

        let query = HKAnchoredObjectQuery(
            type: type,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] _, samples, _, _, _ in
            guard let self = self,
                  let samples = samples as? [HKQuantitySample],
                  let latest = samples.last else { return }

            Task { @MainActor in
                self.heartRate = latest.quantity.doubleValue(for: HKUnit(from: "count/min"))
            }
        }

        query.updateHandler = { [weak self] _, samples, _, _, _ in
            guard let self = self,
                  let samples = samples as? [HKQuantitySample],
                  let latest = samples.last else { return }

            Task { @MainActor in
                self.heartRate = latest.quantity.doubleValue(for: HKUnit(from: "count/min"))
            }
        }

        heartRateQuery = query
        healthStore.execute(query)
    }
}
