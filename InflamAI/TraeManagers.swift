//
//  TraeManagers.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//

import Foundation
import Combine
import CryptoKit

/// Manages offline caching for recipes, media, and timers.
final class TraeOfflineCacheManager {
    enum CacheType: String {
        case recipeLibrary
        case tutorialVideos
        case cookTimers
        case conversions
    }
    
    private let fileManager: FileManager
    private let cacheDirectory: URL
    private var cacheIndex: [CacheType: Set<String>] = [:]
    private let queue = DispatchQueue(label: "com.trae.cache", qos: .utility)
    
    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
        let base = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
        cacheDirectory = base.appendingPathComponent("TraeOfflineCache", isDirectory: true)
        prepareCacheDirectory()
    }
    
    func cache<T: Encodable>(_ value: T, for key: String, type: CacheType) {
        queue.async {
            do {
                try self.ensureIndex(for: type)
                let url = self.url(for: key, type: type)
                let data = try JSONEncoder().encode(value)
                try data.write(to: url, options: [.atomic])
                self.cacheIndex[type, default: []].insert(key)
            } catch {
                print("Cache error: \(error.localizedDescription)")
            }
        }
    }
    
    func retrieve<T: Decodable>(_ type: T.Type, for key: String, cacheType: CacheType) -> T? {
        queue.sync {
            let url = self.url(for: key, type: cacheType)
            guard fileManager.fileExists(atPath: url.path) else { return nil }
            do {
                let data = try Data(contentsOf: url)
                return try JSONDecoder().decode(type, from: data)
            } catch {
                print("Cache retrieval error: \(error.localizedDescription)")
                return nil
            }
        }
    }
    
    func clear(type: CacheType? = nil) {
        queue.async {
            if let type {
                let directory = self.directory(for: type)
                try? self.fileManager.removeItem(at: directory)
                self.cacheIndex[type] = []
            } else {
                try? self.fileManager.removeItem(at: self.cacheDirectory)
                self.prepareCacheDirectory()
                self.cacheIndex.removeAll()
            }
        }
    }
    
    func cachedKeys(for type: CacheType) -> [String] {
        queue.sync {
            Array(cacheIndex[type] ?? [])
        }
    }
    
    private func prepareCacheDirectory() {
        if !fileManager.fileExists(atPath: cacheDirectory.path) {
            try? fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        }
    }
    
    private func ensureIndex(for type: CacheType) throws {
        let directory = self.directory(for: type)
        if !fileManager.fileExists(atPath: directory.path) {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }
    
    private func directory(for type: CacheType) -> URL {
        cacheDirectory.appendingPathComponent(type.rawValue, isDirectory: true)
    }
    
    private func url(for key: String, type: CacheType) -> URL {
        directory(for: type).appendingPathComponent("\(key).json")
    }
}

/// Sync manager that replays pending updates after reconnecting.
final class TraeSyncEngine: ObservableObject {
    enum SyncState {
        case idle
        case syncing(progress: Double)
        case success(timestamp: Date)
        case failed(error: String)
    }
    
    @Published private(set) var state: SyncState = .idle
    private var pendingOperations = [() -> Void]()
    private let queue = DispatchQueue(label: "com.trae.sync", qos: .utility)
    
    func enqueue(operation: @escaping () -> Void) {
        queue.async {
            self.pendingOperations.append(operation)
        }
    }
    
    func replayPendingOperations() {
        queue.async {
            guard !self.pendingOperations.isEmpty else {
                DispatchQueue.main.async { self.state = .success(timestamp: Date()) }
                return
            }
            
            DispatchQueue.main.async { self.state = .syncing(progress: 0) }
            for (index, operation) in self.pendingOperations.enumerated() {
                operation()
                let progress = Double(index + 1) / Double(self.pendingOperations.count)
                DispatchQueue.main.async { self.state = .syncing(progress: progress) }
                Thread.sleep(forTimeInterval: 0.05)
            }
            
            self.pendingOperations.removeAll()
            DispatchQueue.main.async { self.state = .success(timestamp: Date()) }
        }
    }
}

/// Analytics and personalization engine powering adaptive suggestions.
final class TraeAnalyticsEngine: ObservableObject {
    struct Insight: Identifiable, Codable {
        let id = UUID()
        let title: String
        let detail: String
        let action: String
    }
    
    @Published private(set) var insights: [Insight] = []
    @Published private(set) var adherenceScore: Double = 0.0
    
    func refreshInsights(for profile: TraeProfile, pantry: [PantryItem], plans: [MealPlan]) {
        var newInsights: [Insight] = []
        newInsights.append(Insight(title: NSLocalizedString("insights.card.pacing.title", comment: ""),
                                   detail: NSLocalizedString("insights.card.pacing.detail", comment: ""),
                                   action: NSLocalizedString("insights.card.pacing.action", comment: "")))
        newInsights.append(Insight(title: NSLocalizedString("insights.card.mobility.title", comment: ""),
                                   detail: NSLocalizedString("insights.card.mobility.detail", comment: ""),
                                   action: NSLocalizedString("insights.card.mobility.action", comment: "")))
        newInsights.append(Insight(title: NSLocalizedString("insights.card.medication.title", comment: ""),
                                   detail: NSLocalizedString("insights.card.medication.detail", comment: ""),
                                   action: NSLocalizedString("insights.card.medication.action", comment: "")))
        insights = newInsights
    }
    
    func updateTrending(with recipes: [Recipe]) {
        adherenceScore = 0.0
    }
}

/// Security center handling encryption of favorites, allergen filters, and telemetry.
final class TraeSecurityCenter: ObservableObject {
    enum SecurityEvent {
        case favoritesEncrypted
        case profileLockAdded
        case allergenFilterUpdated
        case telemetryConsentUpdated(Bool)
    }
    
    @Published private(set) var recentEvents: [SecurityEvent] = []
    private(set) var favoritesKey: SymmetricKey
    private(set) var telemetryOptIn = true
    
    init() {
        favoritesKey = SymmetricKey(size: .bits256)
    }
    
    func encryptFavorites(_ recipes: [Recipe]) -> Data? {
        do {
            let data = try JSONEncoder().encode(recipes)
            let sealedBox = try ChaChaPoly.seal(data, using: favoritesKey)
            append(.favoritesEncrypted)
            return sealedBox.combined
        } catch {
            print("Encryption error: \(error.localizedDescription)")
            return nil
        }
    }
    
    func updateTelemetryConsent(_ consented: Bool) {
        telemetryOptIn = consented
        append(.telemetryConsentUpdated(consented))
    }
    
    func markAllergenFilterUpdate() {
        append(.allergenFilterUpdated)
    }
    
    func recordProfileLock() {
        append(.profileLockAdded)
    }
    
    private func append(_ event: SecurityEvent) {
        DispatchQueue.main.async {
            self.recentEvents.append(event)
            if self.recentEvents.count > 5 {
                self.recentEvents.removeFirst()
            }
        }
    }
}

/// Pantry intelligence for proactive alerts and stock reminders.
final class PantryIntelligence: ObservableObject {
    @Published private(set) var pantryItems: [PantryItem]
    @Published private(set) var alerts: [PantryAlert] = []
    
    init(pantryItems: [PantryItem]) {
        self.pantryItems = pantryItems
        refreshAlerts()
    }
    
    func update(item: PantryItem) {
        if let index = pantryItems.firstIndex(where: { $0.id == item.id }) {
            pantryItems[index] = item
        } else {
            pantryItems.append(item)
        }
        refreshAlerts()
    }
    
    func refreshAlerts() {
        var newAlerts: [PantryAlert] = []
        for item in pantryItems {
            switch item.stockStatus {
            case .runningLow:
                newAlerts.append(PantryAlert(title: "\(item.name) running low",
                                             message: "Top up before the weekend to stay prepared for mid-week cooking.",
                                             type: .restock,
                                             created: Date()))
            case .outOfStock:
                newAlerts.append(PantryAlert(title: "\(item.name) out of stock",
                                             message: "Add to shopping list or adjust upcoming recipes.",
                                             type: .restock,
                                             created: Date()))
            case .expiringSoon(let date):
                let formatter = DateFormatter()
                formatter.dateStyle = .short
                newAlerts.append(PantryAlert(title: "\(item.name) expires soon",
                                             message: "Use by \(formatter.string(from: date)) to avoid waste.",
                                             type: .spoilage,
                                             created: Date()))
            case .inStock:
                continue
            }
        }
        alerts = newAlerts
    }
}

/// Recipe data store with sample fixtures and filtering logic.
final class RecipeLibrary: ObservableObject {
    @Published private(set) var recipes: [Recipe]
    @Published private(set) var filtered: [Recipe]
    @Published var selectedCategories: Set<String> = []
    @Published var searchTerm: String = ""
    @Published var showPinned = false
    
    init(recipes: [Recipe]) {
        self.recipes = recipes
        filtered = recipes
    }
    
    func applyFilters() {
        filtered = recipes.filter { recipe in
            var matchesCategory = true
            if !selectedCategories.isEmpty {
                matchesCategory = !selectedCategories.isDisjoint(with: recipe.categories)
            }
            let matchesSearch = searchTerm.isEmpty ||
            recipe.title.localizedCaseInsensitiveContains(searchTerm) ||
            recipe.subtitle.localizedCaseInsensitiveContains(searchTerm)
            let matchesPin = !showPinned || recipe.isPinned
            return matchesCategory && matchesSearch && matchesPin
        }
    }
}

/// Meal plan collaboration logic with live comments.
final class MealPlanCollaboration: ObservableObject {
    @Published private(set) var plans: [MealPlan]
    @Published private(set) var recentComments: [MealPlanComment] = []
    
    init(plans: [MealPlan]) {
        self.plans = plans
        aggregateComments()
    }
    
    func add(comment: MealPlanComment, to plan: MealPlan) {
        guard let index = plans.firstIndex(where: { $0.id == plan.id }) else { return }
        plans[index].comments.append(comment)
        aggregateComments()
    }
    
    private func aggregateComments() {
        recentComments = plans.flatMap { $0.comments }.sorted(by: { $0.timestamp > $1.timestamp }).prefix(5).map { $0 }
    }
}
