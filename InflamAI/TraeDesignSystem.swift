//
//  TraeDesignSystem.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//

import Foundation
import SwiftUI
import CoreHaptics

/// Centralizes brand colors, typography, spacing, and haptic patterns for Trae am Kochen.
enum TraePalette {
    static let traeOrange = Color(red: 1.0, green: 0.48, blue: 0.12)
    static let forestGreen = Color(red: 0.18, green: 0.37, blue: 0.31)
    static let saffron = Color(red: 1.0, green: 0.70, blue: 0.28)
    static let snow = Color(red: 0.97, green: 0.97, blue: 0.96)
    static let graphite = Color(red: 0.12, green: 0.12, blue: 0.12)
    
    static let success = Color(red: 0.16, green: 0.60, blue: 0.38)
    static let warning = Color(red: 0.93, green: 0.60, blue: 0.16)
    static let danger = Color(red: 0.84, green: 0.16, blue: 0.20)
}

/// Defines the typography scale used across the application.
enum TraeTypography {
    static let largeTitle = Font.system(size: 34, weight: .bold, design: .serif)
    static let title = Font.system(size: 28, weight: .semibold, design: .serif)
    static let title2 = Font.system(size: 22, weight: .semibold, design: .serif)
    static let headline = Font.system(size: 17, weight: .semibold, design: .rounded)
    static let body = Font.system(size: 17, weight: .regular, design: .default)
    static let subheadline = Font.system(size: 15, weight: .medium, design: .rounded)
    static let footnote = Font.system(size: 13, weight: .regular, design: .default)
}

/// Spacing constants for consistent layout rhythm.
enum TraeSpacing {
    static let xs: CGFloat = 4
    static let sm: CGFloat = 8
    static let md: CGFloat = 16
    static let lg: CGFloat = 24
    static let xl: CGFloat = 32
}

/// Haptic designer for stage-based cooking experiences.
final class TraeHaptics {
    static let shared = TraeHaptics()
    
    private var engine: CHHapticEngine?
    
    private init() {
        prepareEngine()
    }
    
    func performStageChange() {
        guard let engine else { return }
        
        let intensity = CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.75)
        let sharpness = CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.6)
        let event = CHHapticEvent(eventType: .hapticTransient,
                                  parameters: [intensity, sharpness],
                                  relativeTime: 0)
        
        do {
            let pattern = try CHHapticPattern(events: [event], parameters: [])
            let player = try engine.makePlayer(with: pattern)
            try player.start(atTime: 0)
        } catch {
            // Using print instead of os_log to avoid pulling in os.log for release builds.
            print("Haptic error: \(error.localizedDescription)")
        }
    }
    
    func performGentleProgress() {
        guard let engine else { return }
        
        let intensity = CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.4)
        let sharpness = CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.2)
        let event = CHHapticEvent(eventType: .hapticContinuous,
                                  parameters: [intensity, sharpness],
                                  relativeTime: 0,
                                  duration: 0.4)
        do {
            let pattern = try CHHapticPattern(events: [event], parameters: [])
            let player = try engine.makePlayer(with: pattern)
            try player.start(atTime: 0)
        } catch {
            print("Haptic progress error: \(error.localizedDescription)")
        }
    }
    
    func performAlert() {
        guard let engine else { return }
        let intensity = CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0)
        let sharpness = CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.8)
        let event = CHHapticEvent(eventType: .hapticTransient,
                                  parameters: [intensity, sharpness],
                                  relativeTime: 0)
        do {
            let pattern = try CHHapticPattern(events: [event], parameters: [])
            let player = try engine.makePlayer(with: pattern)
            try player.start(atTime: 0)
        } catch {
            print("Haptic alert error: \(error.localizedDescription)")
        }
    }
    
    private func prepareEngine() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
        do {
            engine = try CHHapticEngine()
            try engine?.start()
        } catch {
            print("Failed to start haptic engine: \(error.localizedDescription)")
        }
    }
}
