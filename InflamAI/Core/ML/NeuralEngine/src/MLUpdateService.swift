//
//  MLUpdateService.swift
//  InflamAI
//
//  Created by OVERLORD Neural Engine
//

import Foundation
import BackgroundTasks
import CoreML
import CoreData

/// Service responsible for orchestrating on-device model updates
class MLUpdateService: ObservableObject {
    static let shared = MLUpdateService()
    
    private let predictor = UpdatableFlarePredictor()
    private let backgroundTaskID = "com.inflamai.modelupdate"
    
    @Published var lastUpdateDate: Date?
    @Published var isUpdating = false
    
    private init() {
        registerBackgroundTask()
    }
    
    /// Register background task for model updates
    private func registerBackgroundTask() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: backgroundTaskID, using: nil) { task in
            self.handleBackgroundTask(task: task as! BGProcessingTask)
        }
    }
    
    /// Schedule a background update
    func scheduleBackgroundUpdate() {
        let request = BGProcessingTaskRequest(identifier: backgroundTaskID)
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = true // Only update when charging
        
        // Schedule for 2 AM or later
        let calendar = Calendar.current
        let now = Date()
        var components = calendar.dateComponents([.year, .month, .day], from: now)
        components.hour = 2
        components.minute = 0
        components.second = 0
        
        guard let scheduledDate = calendar.date(from: components) else { return }
        let nextUpdate = scheduledDate < now ? scheduledDate.addingTimeInterval(86400) : scheduledDate
        
        request.earliestBeginDate = nextUpdate
        
        do {
            try BGTaskScheduler.shared.submit(request)
            print("✅ Model update scheduled for \(nextUpdate)")
        } catch {
            print("❌ Could not schedule model update: \(error)")
        }
    }
    
    /// Handle the background task
    private func handleBackgroundTask(task: BGProcessingTask) {
        // Schedule next update
        scheduleBackgroundUpdate()
        
        task.expirationHandler = {
            // Cancel operations if running out of time
            print("⚠️ Background update time expired")
        }
        
        Task {
            do {
                try await performUpdate()
                task.setTaskCompleted(success: true)
            } catch {
                print("❌ Background update failed: \(error)")
                task.setTaskCompleted(success: false)
            }
        }
    }
    
    /// Perform the actual model update
    func performUpdate() async throws {
        guard !isUpdating else { return }
        
        DispatchQueue.main.async { self.isUpdating = true }
        defer { DispatchQueue.main.async { self.isUpdating = false } }
        
        // 1. Collect training data
        let trainingData = try await collectTrainingData()
        
        guard !trainingData.isEmpty else {
            print("ℹ️ No new data for training")
            return
        }
        
        // 2. Run update
        try await predictor.updateModel(with: trainingData)
        
        // 3. Update timestamp
        DispatchQueue.main.async {
            self.lastUpdateDate = Date()
            UserDefaults.standard.set(Date(), forKey: "LastModelUpdate")
        }
        
        print("✅ Background model update completed")
    }
    
    /// Collect data from local storage for training
    /// Uses shared TrainingDataCollector for consistency
    private func collectTrainingData() async throws -> [(features: [[Float]], label: Int)] {
        let context = InflamAIPersistenceController.shared.container.viewContext
        let featureExtractor = FeatureExtractor()

        print("🔄 [MLUpdateService] Collecting training data...")

        return try await TrainingDataCollector.collectTrainingData(
            context: context,
            featureExtractor: featureExtractor
        )
    }
}
