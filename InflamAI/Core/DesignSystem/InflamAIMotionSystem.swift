//
//  InflamAIMotionSystem.swift
//  InflamAI
//
//  Motion Design System - Intuitive like riding a bike
//  Direction = Meaning. No explanation needed.
//
//  Rules:
//  - HORIZONTAL = Same level (tabs, siblings)
//  - VERTICAL = Different level (detail up, dismiss down)
//  - SCALE UP = Selected / Important
//  - SCALE DOWN = Dismissed / Deselected
//  - SHAKE = Error / Wrong
//  - PULSE = Loading / Waiting
//  - BLOOM = Success / Complete
//

import SwiftUI
#if os(iOS)
import UIKit
#endif

// MARK: - Motion Timing

enum MotionTiming {
    static let instant: Double = 0.1      // Micro-feedback
    static let fast: Double = 0.15        // Tap response
    static let normal: Double = 0.25      // Standard transitions
    static let smooth: Double = 0.35      // Detail views
    static let slow: Double = 0.5         // Major transitions
}

// MARK: - Motion Springs

enum MotionSpring {
    /// Snappy - for buttons, selections
    static var snappy: Animation {
        .spring(response: 0.3, dampingFraction: 0.7, blendDuration: 0)
    }

    /// Smooth - for sheets, detail views
    static var smooth: Animation {
        .spring(response: 0.4, dampingFraction: 0.8, blendDuration: 0)
    }

    /// Bouncy - for success, celebration
    static var bouncy: Animation {
        .spring(response: 0.5, dampingFraction: 0.6, blendDuration: 0)
    }

    /// Gentle - for loading, subtle motion
    static var gentle: Animation {
        .spring(response: 0.6, dampingFraction: 0.9, blendDuration: 0)
    }
}

// MARK: - Directional Transitions

enum MotionDirection {
    /// Detail view rises from below (drill in)
    static var riseUp: AnyTransition {
        .asymmetric(
            insertion: .move(edge: .bottom).combined(with: .opacity),
            removal: .move(edge: .bottom).combined(with: .opacity)
        )
    }

    /// Content sinks down (dismiss)
    static var sinkDown: AnyTransition {
        .asymmetric(
            insertion: .move(edge: .top).combined(with: .opacity),
            removal: .move(edge: .bottom).combined(with: .opacity)
        )
    }

    /// Slide from right (next item)
    static var slideFromRight: AnyTransition {
        .asymmetric(
            insertion: .move(edge: .trailing),
            removal: .move(edge: .leading)
        )
    }

    /// Slide from left (previous item)
    static var slideFromLeft: AnyTransition {
        .asymmetric(
            insertion: .move(edge: .leading),
            removal: .move(edge: .trailing)
        )
    }

    /// Scale up from center (modal, important)
    static var scaleUp: AnyTransition {
        .scale(scale: 0.9).combined(with: .opacity)
    }

    /// Fade only (subtle changes)
    static var fade: AnyTransition {
        .opacity
    }
}

// MARK: - Interactive Animations

struct PressableButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .opacity(configuration.isPressed ? 0.9 : 1.0)
            .animation(MotionSpring.snappy, value: configuration.isPressed)
    }
}

struct CardPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .shadow(
                color: Color.black.opacity(configuration.isPressed ? 0.15 : 0.08),
                radius: configuration.isPressed ? 4 : 8,
                y: configuration.isPressed ? 2 : 4
            )
            .animation(MotionSpring.snappy, value: configuration.isPressed)
    }
}

struct SOSButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 1.05 : 1.0)
            .shadow(
                color: Color.red.opacity(configuration.isPressed ? 0.4 : 0.2),
                radius: configuration.isPressed ? 16 : 8,
                y: 0
            )
            .animation(MotionSpring.bouncy, value: configuration.isPressed)
    }
}

// MARK: - Haptic Feedback

enum HapticFeedback {
    static func light() {
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        #endif
    }

    static func medium() {
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        #endif
    }

    static func heavy() {
        #if os(iOS)
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        #endif
    }

    static func success() {
        #if os(iOS)
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        #endif
    }

    static func error() {
        #if os(iOS)
        UINotificationFeedbackGenerator().notificationOccurred(.error)
        #endif
    }

    static func warning() {
        #if os(iOS)
        UINotificationFeedbackGenerator().notificationOccurred(.warning)
        #endif
    }

    static func selection() {
        #if os(iOS)
        UISelectionFeedbackGenerator().selectionChanged()
        #endif
    }
}

// MARK: - Shake Animation (Error)

struct ShakeEffect: GeometryEffect {
    var amount: CGFloat = 10
    var shakesPerUnit = 3
    var animatableData: CGFloat

    func effectValue(size: CGSize) -> ProjectionTransform {
        ProjectionTransform(CGAffineTransform(translationX:
            amount * sin(animatableData * .pi * CGFloat(shakesPerUnit)),
            y: 0))
    }
}

// MARK: - Pulse Animation (Loading)

struct PulseEffect: ViewModifier {
    @State private var isPulsing = false

    func body(content: Content) -> some View {
        content
            .opacity(isPulsing ? 0.6 : 1.0)
            .scaleEffect(isPulsing ? 0.98 : 1.0)
            .animation(
                Animation.easeInOut(duration: 0.8)
                    .repeatForever(autoreverses: true),
                value: isPulsing
            )
            .onAppear { isPulsing = true }
    }
}

// MARK: - Bloom Animation (Success)

struct BloomEffect: ViewModifier {
    let isActive: Bool

    func body(content: Content) -> some View {
        content
            .scaleEffect(isActive ? 1.0 : 0.8)
            .opacity(isActive ? 1.0 : 0.0)
            .animation(MotionSpring.bouncy, value: isActive)
    }
}

// MARK: - Staggered Appearance

struct StaggeredAppearance: ViewModifier {
    let index: Int
    let isVisible: Bool

    func body(content: Content) -> some View {
        content
            .opacity(isVisible ? 1 : 0)
            .offset(y: isVisible ? 0 : 20)
            .animation(
                MotionSpring.smooth.delay(Double(index) * 0.05),
                value: isVisible
            )
    }
}

// MARK: - Tab Transition

struct TabTransition: ViewModifier {
    let direction: Edge

    func body(content: Content) -> some View {
        content
            .transition(.asymmetric(
                insertion: .move(edge: direction).combined(with: .opacity),
                removal: .move(edge: direction == .leading ? .trailing : .leading).combined(with: .opacity)
            ))
    }
}

// MARK: - View Extensions

extension View {
    /// Apply pressable button animation
    func pressable() -> some View {
        buttonStyle(PressableButtonStyle())
    }

    /// Apply card press animation
    func cardPress() -> some View {
        buttonStyle(CardPressStyle())
    }

    /// Apply SOS button animation
    func sosStyle() -> some View {
        buttonStyle(SOSButtonStyle())
    }

    /// Apply shake animation for errors
    func shake(trigger: Bool) -> some View {
        modifier(ShakeModifier(trigger: trigger))
    }

    /// Apply pulse animation for loading
    func pulse() -> some View {
        modifier(PulseEffect())
    }

    /// Apply bloom animation for success
    func bloom(isActive: Bool) -> some View {
        modifier(BloomEffect(isActive: isActive))
    }

    /// Apply staggered appearance
    func staggered(index: Int, isVisible: Bool) -> some View {
        modifier(StaggeredAppearance(index: index, isVisible: isVisible))
    }

    /// Animate view appearing from bottom (detail view)
    func riseTransition() -> some View {
        transition(MotionDirection.riseUp)
    }

    /// Animate view disappearing to bottom (dismiss)
    func sinkTransition() -> some View {
        transition(MotionDirection.sinkDown)
    }

    /// Standard spring animation
    func smoothAnimation() -> some View {
        animation(MotionSpring.smooth, value: UUID())
    }
}

// MARK: - Shake Modifier

struct ShakeModifier: ViewModifier {
    let trigger: Bool
    @State private var shakeAmount: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .modifier(ShakeEffect(animatableData: shakeAmount))
            .onChange(of: trigger) { newValue in
                if newValue {
                    HapticFeedback.error()
                    withAnimation(.linear(duration: 0.4)) {
                        shakeAmount = 1
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                        shakeAmount = 0
                    }
                }
            }
    }
}

// MARK: - Pain Level Colors (Consistent across app)

enum PainColors {
    static func color(for level: Int) -> Color {
        switch level {
        case 0: return .green                     // Green - No pain
        case 1: return Color.green.opacity(0.7)   // Light green
        case 2: return .yellow                    // Yellow
        case 3: return Color.yellow.opacity(0.8)  // Light yellow
        case 4: return .orange                    // Orange
        case 5: return Color.orange               // Dark orange
        case 6: return .red                       // Red
        case 7: return Color.red                  // Dark red
        case 8: return .purple                    // Purple
        case 9: return Color.purple               // Dark purple
        case 10: return Color.black               // Black
        default: return Color.gray
        }
    }

    static func gradient(for level: Int) -> LinearGradient {
        let baseColor = color(for: level)
        return LinearGradient(
            colors: [baseColor.opacity(0.8), baseColor],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
}

// MARK: - Navigation Hub Card Style

struct HubCardStyle: ViewModifier {
    let accentColor: Color

    func body(content: Content) -> some View {
        content
            .padding(24)  // Spacing.lg
            .background(
                RoundedRectangle(cornerRadius: 16)  // Radii.xl
                    .fill(Color.white)
                    .shadow(color: accentColor.opacity(0.1), radius: 8, y: 4)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16)  // Radii.xl
                    .stroke(accentColor.opacity(0.2), lineWidth: 1)
            )
    }
}

extension View {
    func hubCard(accent: Color) -> some View {
        modifier(HubCardStyle(accentColor: accent))
    }
}

// MARK: - Sheet Detent Helper

enum SheetHeight {
    case small      // 25% - Quick actions
    case medium     // 50% - Quick log
    case large      // 85% - Full form
    case full       // 100% - Modal

    var fraction: CGFloat {
        switch self {
        case .small: return 0.25
        case .medium: return 0.5
        case .large: return 0.85
        case .full: return 1.0
        }
    }
}
