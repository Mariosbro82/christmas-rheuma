//
//  TriggerInsightsViewModel.swift
//  InflamAI
//
//  ViewModel for TriggerInsightsView
//  Coordinates with TriggerAnalysisService for trigger detection
//

import Foundation
import Combine

@MainActor
class TriggerInsightsViewModel: ObservableObject {

    // MARK: - Published State

    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    @Published var currentPhase: ActivationPhase = .statistical
    @Published var daysOfData: Int = 0
    @Published var progressToNextPhase: Double?
    @Published var daysUntilNextPhase: Int?

    @Published var topTriggers: [UnifiedTriggerResult] = []
    @Published var allTriggers: [UnifiedTriggerResult] = []
    @Published var recommendations: [TriggerRecommendation] = []
    @Published var tomorrowPrediction: TomorrowPrediction?

    // MARK: - Computed Properties

    var activeEngines: [EngineType] {
        currentPhase.activeEngines
    }

    var nextPhaseName: String {
        switch currentPhase {
        case .statistical: return "Pattern Matching"
        case .knn: return "Neural Network"
        case .neural: return "Maximum Analysis"
        }
    }

    // MARK: - Dependencies

    private let analysisService: TriggerAnalysisService
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init(analysisService: TriggerAnalysisService = .shared) {
        self.analysisService = analysisService
        setupBindings()
    }

    private func setupBindings() {
        analysisService.$currentPhase
            .receive(on: DispatchQueue.main)
            .assign(to: &$currentPhase)

        analysisService.$daysOfData
            .receive(on: DispatchQueue.main)
            .assign(to: &$daysOfData)

        analysisService.$isAnalyzing
            .receive(on: DispatchQueue.main)
            .assign(to: &$isLoading)

        analysisService.$topTriggers
            .receive(on: DispatchQueue.main)
            .sink { [weak self] triggers in
                self?.topTriggers = Array(triggers.prefix(5))
                self?.allTriggers = triggers
            }
            .store(in: &cancellables)

        analysisService.$recommendations
            .receive(on: DispatchQueue.main)
            .assign(to: &$recommendations)

        analysisService.$errorMessage
            .receive(on: DispatchQueue.main)
            .assign(to: &$errorMessage)
    }

    // MARK: - Public Methods

    func loadData() async {
        isLoading = true

        // Get insights summary
        let insights = await analysisService.getQuickInsights()
        progressToNextPhase = insights.progressToNextPhase
        daysUntilNextPhase = insights.daysUntilNextPhase

        // Run full analysis
        _ = await analysisService.analyzeAllTriggers()

        // Get tomorrow prediction
        tomorrowPrediction = await analysisService.predictTomorrow()

        isLoading = false
    }

    func refresh() async {
        await analysisService.invalidateCaches()
        await loadData()
    }

    func enableNeuralEngine() async {
        await analysisService.enableNeuralEngine()
        await loadData()
    }

    func disableNeuralEngine() async {
        await analysisService.disableNeuralEngine()
        await loadData()
    }

    func getTriggers(for category: TriggerCategory) -> [UnifiedTriggerResult] {
        allTriggers.filter { $0.triggerCategory == category }
    }

    func getExplanation(for triggerName: String) async -> TriggerExplanation? {
        await analysisService.getExplanation(for: triggerName)
    }
}
