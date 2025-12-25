//
//  AIInsightsViewModel.swift
//  InflamAI-Swift
//
//  Created by Claude Code on 2025-01-25.
//

import Foundation
import CoreData
import Combine

@MainActor
class AIInsightsViewModel: ObservableObject {

    // MARK: - Published Properties
    @Published var flareRiskPercentage: Double = 0.0
    @Published var riskLevel: AIRiskLevel = .unknown
    @Published var contributingFactors: [AIContributingFactor] = []
    @Published var suggestions: [String] = []
    @Published var loadingState: LoadingState = .idle
    @Published var errorMessage: String?
    @Published var trainingStatus: TrainingStatus = .insufficientData
    @Published var daysOfDataAvailable: Int = 0

    // MARK: - Dependencies
    private let persistenceController: InflamAIPersistenceController
    private let flarePredictor: FlarePredictor
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Constants
    private let minimumDataDays = 7  // FlarePredictor uses 7 days minimum

    // MARK: - Initialization
    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
        self.flarePredictor = FlarePredictor(context: persistenceController.container.viewContext)
        setupBindings()
    }

    // MARK: - Private Setup

    private func setupBindings() {
        // Observe FlarePredictor's published properties
        flarePredictor.$riskPercentage
            .receive(on: DispatchQueue.main)
            .sink { [weak self] percentage in
                self?.flareRiskPercentage = percentage
            }
            .store(in: &cancellables)

        flarePredictor.$flareRiskLevel
            .receive(on: DispatchQueue.main)
            .sink { [weak self] level in
                self?.riskLevel = AIRiskLevel.from(flarePredictorLevel: level)
            }
            .store(in: &cancellables)

        flarePredictor.$contributingFactors
            .receive(on: DispatchQueue.main)
            .sink { [weak self] factors in
                self?.contributingFactors = factors.map { AIContributingFactor.from(flarePredictorFactor: $0) }
            }
            .store(in: &cancellables)

        flarePredictor.$isModelTrained
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isTrained in
                if isTrained {
                    self?.trainingStatus = .ready
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Public Methods

    func analyzeFlareRisk() async {
        loadingState = .loading

        do {
            // Check if we have enough data
            daysOfDataAvailable = try await countAvailableDataDays()

            guard daysOfDataAvailable >= minimumDataDays else {
                trainingStatus = .insufficientData
                loadingState = .loaded
                suggestions = [
                    "Track your symptoms daily to enable pattern analysis",
                    "Record at least \(minimumDataDays) days of data for accurate insights",
                    "Include pain levels, stiffness, and activity in your daily logs"
                ]
                return
            }

            trainingStatus = .analyzing

            // Use FlarePredictor's updatePrediction method
            await flarePredictor.updatePrediction()

            // Generate suggestions based on current state
            suggestions = generateSuggestions()

            loadingState = .loaded
            trainingStatus = flarePredictor.isModelTrained ? .ready : .insufficientData

        } catch {
            loadingState = .error(error)
            errorMessage = "Analysis failed: \(error.localizedDescription)"
            trainingStatus = .error
        }
    }

    func refresh() async {
        await analyzeFlareRisk()
    }

    // MARK: - Private Methods

    private func countAvailableDataDays() async throws -> Int {
        let context = persistenceController.container.viewContext
        return try await context.perform {
            let request = SymptomLog.fetchRequest()
            request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: true)]

            let logs = try context.fetch(request)

            guard let firstDate = logs.first?.timestamp,
                  let lastDate = logs.last?.timestamp else {
                return 0
            }

            let calendar = Calendar.current
            let components = calendar.dateComponents([.day], from: firstDate, to: lastDate)
            return (components.day ?? 0) + 1
        }
    }

    private func generateSuggestions() -> [String] {
        var suggestions: [String] = []

        // Risk-based suggestions
        switch riskLevel {
        case .low:
            suggestions.append("Your current pattern suggests low flare risk. Keep up your current routine!")
        case .moderate:
            suggestions.append("Elevated flare risk detected. Consider increasing rest and gentle movement.")
        case .high:
            suggestions.append("High flare risk indicated. Consult your rheumatologist if symptoms worsen.")
        case .unknown:
            suggestions.append("Continue tracking to enable personalized insights.")
        }

        // Factor-based suggestions from FlarePredictor
        for factor in flarePredictor.contributingFactors.prefix(3) {
            suggestions.append(factor.recommendation)
        }

        // Always include disclaimer
        suggestions.append("⚠️ This is statistical analysis, not medical advice. Always consult your rheumatologist.")

        return suggestions
    }
}

// MARK: - Supporting Types

enum AIRiskLevel {
    case unknown, low, moderate, high

    static func from(score: Double) -> AIRiskLevel {
        switch score {
        case 0..<0.3:
            return .low
        case 0.3..<0.6:
            return .moderate
        case 0.6...1.0:
            return .high
        default:
            return .unknown
        }
    }

    static func from(flarePredictorLevel: FlarePredictorRiskLevel) -> AIRiskLevel {
        switch flarePredictorLevel {
        case .unknown: return .unknown
        case .low: return .low
        case .moderate: return .moderate
        case .high, .critical: return .high
        }
    }

    var color: String {
        switch self {
        case .unknown: return "gray"
        case .low: return "green"
        case .moderate: return "yellow"
        case .high: return "red"
        }
    }

    var displayName: String {
        switch self {
        case .unknown: return "Unknown"
        case .low: return "Low Risk"
        case .moderate: return "Moderate Risk"
        case .high: return "High Risk"
        }
    }
}

enum TrainingStatus {
    case insufficientData
    case ready
    case analyzing
    case error

    var message: String {
        switch self {
        case .insufficientData:
            return "Need more data for analysis (minimum 30 days)"
        case .ready:
            return "Analysis ready"
        case .analyzing:
            return "Analyzing patterns..."
        case .error:
            return "Analysis unavailable"
        }
    }
}

// MARK: - Loading State
enum LoadingState: Equatable {
    case idle
    case loading
    case loaded
    case error(Error)

    static func == (lhs: LoadingState, rhs: LoadingState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.loading, .loading), (.loaded, .loaded):
            return true
        case (.error, .error):
            return true
        default:
            return false
        }
    }
}

// MARK: - AI Contributing Factor
struct AIContributingFactor: Identifiable {
    let id = UUID()
    let name: String
    let type: AIFactorType
    let value: Double
    let impact: Double
    let recommendation: String

    enum AIFactorType {
        case painLevel, stiffness, weather, activity, medication, other

        var displayName: String {
            switch self {
            case .painLevel: return "Pain Level"
            case .stiffness: return "Morning Stiffness"
            case .weather: return "Weather"
            case .activity: return "Activity Level"
            case .medication: return "Medication"
            case .other: return "Other Factor"
            }
        }
    }

    static func from(flarePredictorFactor: FlarePredictorFactor) -> AIContributingFactor {
        // Map factor name to type
        let factorType: AIFactorType
        let name = flarePredictorFactor.name.lowercased()

        if name.contains("pain") {
            factorType = .painLevel
        } else if name.contains("stiffness") {
            factorType = .stiffness
        } else if name.contains("weather") || name.contains("pressure") || name.contains("humidity") || name.contains("storm") {
            factorType = .weather
        } else if name.contains("activity") || name.contains("exercise") {
            factorType = .activity
        } else if name.contains("medication") {
            factorType = .medication
        } else {
            factorType = .other
        }

        // Map impact
        let impactValue: Double
        switch flarePredictorFactor.impact {
        case .low: impactValue = 0.3
        case .medium: impactValue = 0.6
        case .high: impactValue = 1.0
        }

        return AIContributingFactor(
            name: flarePredictorFactor.name,
            type: factorType,
            value: flarePredictorFactor.value,
            impact: impactValue,
            recommendation: flarePredictorFactor.recommendation
        )
    }
}
