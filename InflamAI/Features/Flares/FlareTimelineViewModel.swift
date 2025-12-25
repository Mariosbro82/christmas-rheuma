//
//  FlareTimelineViewModel.swift
//  InflamAI-Swift
//
//  Created by Claude Code on 2025-01-25.
//

import Foundation
import CoreData
import Combine

@MainActor
class FlareTimelineViewModel: ObservableObject {

    // MARK: - Published Properties
    @Published var flares: [FlareEvent] = []
    @Published var loadingState: LoadingState = .idle
    @Published var selectedTimeRange: TimeRange = .all
    @Published var errorMessage: String?

    // MARK: - Dependencies
    private let persistenceController: InflamAIPersistenceController
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization
    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
    }

    // MARK: - Public Methods

    func loadFlares() async {
        loadingState = .loading

        do {
            let context = persistenceController.viewContext
            let request = FlareEvent.fetchRequest()
            request.sortDescriptors = [NSSortDescriptor(keyPath: \FlareEvent.startDate, ascending: false)]

            // Apply time range filter
            if let predicate = timeRangePredicate() {
                request.predicate = predicate
            }

            let results = try context.fetch(request)
            flares = results
            loadingState = .loaded
        } catch {
            loadingState = .error(error)
            errorMessage = "Failed to load flare history: \(error.localizedDescription)"
        }
    }

    func deleteFlare(_ flare: FlareEvent) async {
        let context = persistenceController.viewContext
        context.delete(flare)

        do {
            try context.save()
            await loadFlares()
        } catch {
            errorMessage = "Failed to delete flare: \(error.localizedDescription)"
        }
    }

    func setTimeRange(_ range: TimeRange) {
        selectedTimeRange = range
        Task {
            await loadFlares()
        }
    }

    // MARK: - Private Methods

    private func timeRangePredicate() -> NSPredicate? {
        guard selectedTimeRange != .all else { return nil }

        let calendar = Calendar.current
        let now = Date()
        var startDate: Date?

        switch selectedTimeRange {
        case .week:
            startDate = calendar.date(byAdding: .day, value: -7, to: now)
        case .month:
            startDate = calendar.date(byAdding: .month, value: -1, to: now)
        case .threeMonths:
            startDate = calendar.date(byAdding: .month, value: -3, to: now)
        case .sixMonths:
            startDate = calendar.date(byAdding: .month, value: -6, to: now)
        case .year:
            startDate = calendar.date(byAdding: .year, value: -1, to: now)
        case .all:
            return nil
        }

        if let start = startDate {
            return NSPredicate(format: "startDate >= %@", start as NSDate)
        }

        return nil
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
