//
//  DIContainer.swift
//  InflamAI-Swift
//
//  Created by Claude Code on 2025-01-25.
//
//  Dependency Injection Container for managing app-wide dependencies
//

import Foundation
import CoreData

/// Main dependency injection container for the application
/// Provides singleton access to core services and factory methods for feature dependencies
@MainActor
class DIContainer {

    // MARK: - Singleton
    static let shared = DIContainer()

    private init() {}

    // MARK: - Core Dependencies (Singletons)

    /// Primary Core Data persistence controller
    lazy var persistenceController: InflamAIPersistenceController = {
        return .shared
    }()

    /// Unified Neural Engine - THE single source of truth for all ML predictions
    /// All UI components MUST use this for flare predictions
    var neuralEngine: UnifiedNeuralEngine {
        return UnifiedNeuralEngine.shared
    }

    /// Error handling and logging manager
    lazy var errorManager: ErrorManager = {
        return ErrorManager()
    }()

    /// Security and encryption manager
    /// Note: Currently multiple SecurityManager files exist - using Core/Security version
    lazy var securityManager: SecurityManager? = {
        // Placeholder - actual implementation depends on which SecurityManager is active
        return nil
    }()

    /// Theme and appearance manager
    /// Note: Currently multiple ThemeManager files exist - needs consolidation
    lazy var themeManager: ThemeManager? = {
        // Placeholder - actual implementation depends on which ThemeManager is active
        return nil
    }()

    // MARK: - Service Factories

    /// Creates a CloudSync manager instance
    /// Note: Currently multiple CloudSyncManager files exist - using Core/Cloud version
    func makeCloudSyncManager() -> CloudSyncManager? {
        // Placeholder - returns nil until duplicates are consolidated
        return nil
    }

    /// Creates a FlarePredictor for statistical analysis
    /// DEPRECATED: Use neuralEngine instead for unified predictions
    @available(*, deprecated, message: "Use DIContainer.shared.neuralEngine instead")
    func makeFlarePredictor() -> FlarePredictor {
        return FlarePredictor(context: persistenceController.viewContext)
    }

    /// Creates a PDF export service
    func makePDFExportService() -> PDFExportService {
        return PDFExportService(context: persistenceController.viewContext)
    }

    // MARK: - ViewModel Factories

    /// Creates a FlareTimelineViewModel
    func makeFlareTimelineViewModel() -> FlareTimelineViewModel {
        return FlareTimelineViewModel(persistenceController: persistenceController)
    }

    /// Creates an AIInsightsViewModel
    func makeAIInsightsViewModel() -> AIInsightsViewModel {
        return AIInsightsViewModel(persistenceController: persistenceController)
    }

    /// Creates an ExerciseLibraryViewModel
    func makeExerciseLibraryViewModel() -> ExerciseLibraryViewModel {
        return ExerciseLibraryViewModel(persistenceController: persistenceController)
    }

    /// Creates a TrendsViewModel
    func makeTrendsViewModel() -> TrendsViewModel {
        return TrendsViewModel(persistenceController: persistenceController)
    }

    /// Creates a MedicationViewModel
    func makeMedicationViewModel() -> MedicationViewModel {
        return MedicationViewModel(persistenceController: persistenceController)
    }

    /// Creates a PainMapViewModel
    func makePainMapViewModel() -> PainMapViewModel {
        return PainMapViewModel(context: persistenceController.container.viewContext)
    }

    /// Creates a BodyMapViewModel
    func makeBodyMapViewModel() -> BodyMapViewModel {
        return BodyMapViewModel(context: persistenceController.container.viewContext)
    }
}

// MARK: - Protocol-Based DI (Future Enhancement)

/// Protocol for dependency injection
/// ViewModels can conform to this to declare their dependencies explicitly
protocol Injectable {
    associatedtype Dependencies
    init(dependencies: Dependencies)
}

/// Example of how to use protocol-based DI
/// Uncomment and implement when refactoring existing ViewModels
/*
struct FlareTimelineViewModelDependencies {
    let persistenceController: InflamAIPersistenceController
    let flarePredictor: FlarePredictor
}

extension FlareTimelineViewModel: Injectable {
    convenience init(dependencies: FlareTimelineViewModelDependencies) {
        self.init(
            persistenceController: dependencies.persistenceController,
            flarePredictor: dependencies.flarePredictor
        )
    }
}
*/

// MARK: - SwiftUI Environment Extension

import SwiftUI

/// Environment key for DIContainer
struct DIContainerKey: EnvironmentKey {
    static let defaultValue = DIContainer.shared
}

extension EnvironmentValues {
    var diContainer: DIContainer {
        get { self[DIContainerKey.self] }
        set { self[DIContainerKey.self] = newValue }
    }
}

/// Environment key for UnifiedNeuralEngine
/// Use this for direct access to the neural engine in views
struct NeuralEngineKey: EnvironmentKey {
    @MainActor
    static let defaultValue = UnifiedNeuralEngine.shared
}

extension EnvironmentValues {
    @MainActor
    var neuralEngine: UnifiedNeuralEngine {
        get { self[NeuralEngineKey.self] }
        set { self[NeuralEngineKey.self] = newValue }
    }
}

/// Usage in SwiftUI Views:
/// @Environment(\.neuralEngine) var neuralEngine  // Direct access to unified ML
/// @Environment(\.diContainer) var container      // Access to all services
///
/// Example:
/// struct MyView: View {
///     @Environment(\.neuralEngine) var neuralEngine
///
///     var body: some View {
///         Text(neuralEngine.currentPrediction?.summary ?? "No prediction")
///     }
/// }
