//
//  StringExtensions.swift
//  InflamAI
//
//  String utility extensions for UI/UX improvements
//  CRIT-002, CRIT-003 fixes: Localization fallback and displayName formatting
//

import Foundation

// MARK: - String Extensions for UI Formatting

extension String {
    /// Converts snake_case to Title Case for user-facing display (CRIT-003 Fix)
    /// Examples:
    /// - "poor_sleep" → "Poor Sleep"
    /// - "weather_change" → "Weather Change"
    /// - "menstrual_cycle" → "Menstrual Cycle"
    var displayName: String {
        return self
            .replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    /// Provides localized string with fallback to display name if key not found (CRIT-002 Fix)
    /// This prevents raw localization keys from being displayed to users
    var localizedWithFallback: String {
        let localized = NSLocalizedString(self, comment: "")
        if localized == self {
            // Key not found, return formatted version
            #if DEBUG
            print("⚠️ Missing translation for key: \(self)")
            #endif
            return self.friendlyFallback
        }
        return localized
    }

    /// Creates a user-friendly fallback from a localization key
    /// Examples:
    /// - "questionnaire.gaq_2.title" → "GAQ 2"
    /// - "questionnaire.basdai.description" → "BASDAI"
    /// - "some_snake_case_key" → "Some Snake Case Key"
    private var friendlyFallback: String {
        // Handle questionnaire keys specially: "questionnaire.xxx.title" or "questionnaire.xxx.description"
        if self.hasPrefix("questionnaire.") {
            let components = self.components(separatedBy: ".")
            if components.count >= 2 {
                // Extract the questionnaire ID (e.g., "gaq_2", "basdai")
                let questionnaireID = components[1]
                // Format: replace underscores, uppercase known acronyms
                return questionnaireID
                    .replacingOccurrences(of: "_", with: " ")
                    .uppercased()
            }
        }
        // Default: use displayName for other keys
        return self.displayName
    }
}
