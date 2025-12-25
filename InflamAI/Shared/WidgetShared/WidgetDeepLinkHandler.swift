//
//  WidgetDeepLinkHandler.swift
//  InflamAI
//
//  Handles deep links from widgets to navigate to appropriate views
//

import Foundation
import SwiftUI

/// Widget deep link URL scheme: spinalytics://widget/{destination}
public enum WidgetDeepLink: String, CaseIterable {
    case quicklog = "quicklog"
    case sosflare = "sosflare"
    case medication = "medication"
    case exercise = "exercise"
    case trends = "trends"
    case flare = "flare"
    case basdai = "basdai"
    case dashboard = "dashboard"

    /// Full URL for this deep link
    public var url: URL {
        URL(string: "spinalytics://widget/\(rawValue)")!
    }

    /// Parse a URL into a WidgetDeepLink
    public static func from(url: URL) -> WidgetDeepLink? {
        guard url.scheme == "spinalytics",
              url.host == "widget",
              let destination = url.pathComponents.last else {
            return nil
        }
        return WidgetDeepLink(rawValue: destination)
    }
}

/// Navigation state manager for widget deep links
@MainActor
public class WidgetNavigationState: ObservableObject {

    public static let shared = WidgetNavigationState()

    /// The current navigation destination triggered by a widget
    @Published public var destination: WidgetDeepLink?

    /// Whether to show the quick log sheet
    @Published public var showQuickLogSheet: Bool = false

    /// Whether to show the SOS flare sheet
    @Published public var showSOSFlareSheet: Bool = false

    /// Whether to show the medication sheet
    @Published public var showMedicationSheet: Bool = false

    /// Whether to show the exercise sheet
    @Published public var showExerciseSheet: Bool = false

    private init() {}

    /// Handle a deep link URL from a widget
    public func handle(url: URL) {
        guard let deepLink = WidgetDeepLink.from(url: url) else {
            return
        }

        destination = deepLink

        switch deepLink {
        case .quicklog:
            showQuickLogSheet = true
        case .sosflare:
            showSOSFlareSheet = true
        case .medication:
            showMedicationSheet = true
        case .exercise:
            showExerciseSheet = true
        case .trends, .flare, .basdai, .dashboard:
            // These navigate to tabs/views rather than sheets
            // The main app should observe `destination` and navigate accordingly
            break
        }
    }

    /// Clear the navigation state
    public func clearNavigation() {
        destination = nil
        showQuickLogSheet = false
        showSOSFlareSheet = false
        showMedicationSheet = false
        showExerciseSheet = false
    }
}

// MARK: - View Extension for Deep Link Handling

public extension View {
    /// Add widget deep link handling to a view
    func handleWidgetDeepLinks() -> some View {
        self
            .onOpenURL { url in
                WidgetNavigationState.shared.handle(url: url)
            }
    }
}
