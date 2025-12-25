//
//  ThemeManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import UIKit
import Combine

// MARK: - Theme Models

enum AppTheme: String, CaseIterable, Identifiable {
    case system = "system"
    case light = "light"
    case dark = "dark"
    case highContrast = "high_contrast"
    case colorBlind = "color_blind"
    
    var id: String { rawValue }
    
    var displayName: String {
        switch self {
        case .system: return "System"
        case .light: return "Light"
        case .dark: return "Dark"
        case .highContrast: return "High Contrast"
        case .colorBlind: return "Color Blind Friendly"
        }
    }
    
    var iconName: String {
        switch self {
        case .system: return "gear"
        case .light: return "sun.max"
        case .dark: return "moon"
        case .highContrast: return "circle.lefthalf.filled"
        case .colorBlind: return "eye"
        }
    }
}

struct ThemeColors {
    // Primary Colors
    let primary: Color
    let primaryVariant: Color
    let secondary: Color
    let secondaryVariant: Color
    
    // Background Colors
    let background: Color
    let surface: Color
    let surfaceVariant: Color
    
    // Text Colors
    let onPrimary: Color
    let onSecondary: Color
    let onBackground: Color
    let onSurface: Color
    
    // Status Colors
    let success: Color
    let warning: Color
    let error: Color
    let info: Color
    
    // Pain Level Colors
    let painLow: Color
    let painMedium: Color
    let painHigh: Color
    let painSevere: Color
    
    // Chart Colors
    let chartPrimary: Color
    let chartSecondary: Color
    let chartTertiary: Color
    let chartQuaternary: Color
    
    // Interactive Elements
    let buttonPrimary: Color
    let buttonSecondary: Color
    let buttonDisabled: Color
    let link: Color
    
    // Borders and Dividers
    let border: Color
    let divider: Color
    
    // Shadows
    let shadow: Color
}

struct ThemeTypography {
    let largeTitle: Font
    let title1: Font
    let title2: Font
    let title3: Font
    let headline: Font
    let body: Font
    let callout: Font
    let subheadline: Font
    let footnote: Font
    let caption1: Font
    let caption2: Font
}

struct ThemeSpacing {
    let xs: CGFloat = 4
    let sm: CGFloat = 8
    let md: CGFloat = 16
    let lg: CGFloat = 24
    let xl: CGFloat = 32
    let xxl: CGFloat = 48
}

struct ThemeCornerRadius {
    let small: CGFloat = 4
    let medium: CGFloat = 8
    let large: CGFloat = 12
    let extraLarge: CGFloat = 16
    let round: CGFloat = 50
}

struct ThemeShadows {
    let small: (color: Color, radius: CGFloat, x: CGFloat, y: CGFloat)
    let medium: (color: Color, radius: CGFloat, x: CGFloat, y: CGFloat)
    let large: (color: Color, radius: CGFloat, x: CGFloat, y: CGFloat)
}

// MARK: - Theme Manager

class ThemeManager: ObservableObject {
    // MARK: - Published Properties
    
    @Published var currentTheme: AppTheme {
        didSet {
            saveTheme()
            updateSystemAppearance()
            triggerHapticFeedback()
        }
    }
    
    @Published var colors: ThemeColors
    @Published var typography: ThemeTypography
    @Published var spacing = ThemeSpacing()
    @Published var cornerRadius = ThemeCornerRadius()
    @Published var shadows: ThemeShadows
    
    @Published var isHighContrastEnabled = false
    @Published var isReduceMotionEnabled = false
    @Published var isLargeTextEnabled = false
    @Published var isDynamicTypeEnabled = true
    
    // MARK: - Private Properties
    
    private let userDefaults = UserDefaults.standard
    private let themeKey = "selected_theme"
    private let hapticGenerator = UIImpactFeedbackGenerator(style: .light)
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    
    init() {
        // Load saved theme or default to system
        let savedTheme = userDefaults.string(forKey: themeKey) ?? AppTheme.system.rawValue
        self.currentTheme = AppTheme(rawValue: savedTheme) ?? .system
        
        // Initialize colors and typography based on current theme
        self.colors = Self.createColors(for: currentTheme)
        self.typography = Self.createTypography()
        self.shadows = Self.createShadows(for: currentTheme)
        
        setupAccessibilityObservers()
        updateSystemAppearance()
    }
    
    // MARK: - Public Methods
    
    func setTheme(_ theme: AppTheme) {
        currentTheme = theme
        colors = Self.createColors(for: theme)
        shadows = Self.createShadows(for: theme)
    }
    
    func toggleTheme() {
        switch currentTheme {
        case .light:
            setTheme(.dark)
        case .dark:
            setTheme(.light)
        case .system:
            // Toggle based on current system appearance
            let isDarkMode = UITraitCollection.current.userInterfaceStyle == .dark
            setTheme(isDarkMode ? .light : .dark)
        case .highContrast, .colorBlind:
            setTheme(.system)
        }
    }
    
    func enableHighContrast(_ enabled: Bool) {
        isHighContrastEnabled = enabled
        if enabled {
            setTheme(.highContrast)
        } else {
            setTheme(.system)
        }
    }
    
    func updateForAccessibility() {
        isReduceMotionEnabled = UIAccessibility.isReduceMotionEnabled
        isLargeTextEnabled = UIApplication.shared.preferredContentSizeCategory.isAccessibilityCategory
        
        if isLargeTextEnabled {
            typography = Self.createAccessibleTypography()
        } else {
            typography = Self.createTypography()
        }
    }
    
    func getPainLevelColor(for level: Double) -> Color {
        switch level {
        case 0..<3:
            return colors.painLow
        case 3..<6:
            return colors.painMedium
        case 6..<8:
            return colors.painHigh
        default:
            return colors.painSevere
        }
    }
    
    func getChartColor(for index: Int) -> Color {
        let chartColors = [colors.chartPrimary, colors.chartSecondary, colors.chartTertiary, colors.chartQuaternary]
        return chartColors[index % chartColors.count]
    }
    
    func triggerHapticFeedback(_ style: UIImpactFeedbackGenerator.FeedbackStyle = .light) {
        guard !isReduceMotionEnabled else { return }
        
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.impactOccurred()
    }
    
    func triggerSelectionFeedback() {
        guard !isReduceMotionEnabled else { return }
        
        let generator = UISelectionFeedbackGenerator()
        generator.selectionChanged()
    }
    
    func triggerNotificationFeedback(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        guard !isReduceMotionEnabled else { return }
        
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(type)
    }
    
    // MARK: - Private Methods
    
    private func saveTheme() {
        userDefaults.set(currentTheme.rawValue, forKey: themeKey)
    }
    
    private func updateSystemAppearance() {
        DispatchQueue.main.async {
            guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                  let window = windowScene.windows.first else { return }
            
            switch self.currentTheme {
            case .light:
                window.overrideUserInterfaceStyle = .light
            case .dark, .highContrast:
                window.overrideUserInterfaceStyle = .dark
            case .system, .colorBlind:
                window.overrideUserInterfaceStyle = .unspecified
            }
        }
    }
    
    private func setupAccessibilityObservers() {
        NotificationCenter.default.publisher(for: UIAccessibility.reduceMotionStatusDidChangeNotification)
            .sink { [weak self] _ in
                self?.updateForAccessibility()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIContentSizeCategory.didChangeNotification)
            .sink { [weak self] _ in
                self?.updateForAccessibility()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIAccessibility.darkerSystemColorsStatusDidChangeNotification)
            .sink { [weak self] _ in
                if UIAccessibility.isDarkerSystemColorsEnabled {
                    self?.setTheme(.highContrast)
                }
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Static Factory Methods
    
    private static func createColors(for theme: AppTheme) -> ThemeColors {
        switch theme {
        case .light:
            return createLightThemeColors()
        case .dark:
            return createDarkThemeColors()
        case .system:
            return UITraitCollection.current.userInterfaceStyle == .dark ? createDarkThemeColors() : createLightThemeColors()
        case .highContrast:
            return createHighContrastColors()
        case .colorBlind:
            return createColorBlindFriendlyColors()
        }
    }
    
    private static func createLightThemeColors() -> ThemeColors {
        return ThemeColors(
            primary: Color(red: 0.2, green: 0.4, blue: 0.8),
            primaryVariant: Color(red: 0.1, green: 0.3, blue: 0.7),
            secondary: Color(red: 0.9, green: 0.5, blue: 0.2),
            secondaryVariant: Color(red: 0.8, green: 0.4, blue: 0.1),
            background: Color(UIColor.systemBackground),
            surface: Color(UIColor.secondarySystemBackground),
            surfaceVariant: Color(UIColor.tertiarySystemBackground),
            onPrimary: Color.white,
            onSecondary: Color.white,
            onBackground: Color(UIColor.label),
            onSurface: Color(UIColor.label),
            success: Color.green,
            warning: Color.orange,
            error: Color.red,
            info: Color.blue,
            painLow: Color.green,
            painMedium: Color.yellow,
            painHigh: Color.orange,
            painSevere: Color.red,
            chartPrimary: Color.blue,
            chartSecondary: Color.green,
            chartTertiary: Color.orange,
            chartQuaternary: Color.purple,
            buttonPrimary: Color(red: 0.2, green: 0.4, blue: 0.8),
            buttonSecondary: Color(UIColor.secondarySystemFill),
            buttonDisabled: Color(UIColor.quaternaryLabel),
            link: Color.blue,
            border: Color(UIColor.separator),
            divider: Color(UIColor.separator),
            shadow: Color.black.opacity(0.1)
        )
    }
    
    private static func createDarkThemeColors() -> ThemeColors {
        return ThemeColors(
            primary: Color(red: 0.4, green: 0.6, blue: 1.0),
            primaryVariant: Color(red: 0.3, green: 0.5, blue: 0.9),
            secondary: Color(red: 1.0, green: 0.7, blue: 0.4),
            secondaryVariant: Color(red: 0.9, green: 0.6, blue: 0.3),
            background: Color(UIColor.systemBackground),
            surface: Color(UIColor.secondarySystemBackground),
            surfaceVariant: Color(UIColor.tertiarySystemBackground),
            onPrimary: Color.black,
            onSecondary: Color.black,
            onBackground: Color(UIColor.label),
            onSurface: Color(UIColor.label),
            success: Color.green,
            warning: Color.orange,
            error: Color.red,
            info: Color.blue,
            painLow: Color.green,
            painMedium: Color.yellow,
            painHigh: Color.orange,
            painSevere: Color.red,
            chartPrimary: Color.cyan,
            chartSecondary: Color.mint,
            chartTertiary: Color.orange,
            chartQuaternary: Color.purple,
            buttonPrimary: Color(red: 0.4, green: 0.6, blue: 1.0),
            buttonSecondary: Color(UIColor.secondarySystemFill),
            buttonDisabled: Color(UIColor.quaternaryLabel),
            link: Color.cyan,
            border: Color(UIColor.separator),
            divider: Color(UIColor.separator),
            shadow: Color.black.opacity(0.3)
        )
    }
    
    private static func createHighContrastColors() -> ThemeColors {
        return ThemeColors(
            primary: Color.white,
            primaryVariant: Color.gray,
            secondary: Color.yellow,
            secondaryVariant: Color.orange,
            background: Color.black,
            surface: Color(red: 0.1, green: 0.1, blue: 0.1),
            surfaceVariant: Color(red: 0.2, green: 0.2, blue: 0.2),
            onPrimary: Color.black,
            onSecondary: Color.black,
            onBackground: Color.white,
            onSurface: Color.white,
            success: Color.green,
            warning: Color.yellow,
            error: Color.red,
            info: Color.cyan,
            painLow: Color.green,
            painMedium: Color.yellow,
            painHigh: Color.orange,
            painSevere: Color.red,
            chartPrimary: Color.white,
            chartSecondary: Color.yellow,
            chartTertiary: Color.cyan,
            chartQuaternary: Color.green,
            buttonPrimary: Color.white,
            buttonSecondary: Color.gray,
            buttonDisabled: Color(red: 0.3, green: 0.3, blue: 0.3),
            link: Color.cyan,
            border: Color.white,
            divider: Color.white,
            shadow: Color.clear
        )
    }
    
    private static func createColorBlindFriendlyColors() -> ThemeColors {
        // Using colors that are distinguishable for color blind users
        return ThemeColors(
            primary: Color(red: 0.0, green: 0.45, blue: 0.7), // Blue
            primaryVariant: Color(red: 0.0, green: 0.35, blue: 0.6),
            secondary: Color(red: 0.9, green: 0.6, blue: 0.0), // Orange
            secondaryVariant: Color(red: 0.8, green: 0.5, blue: 0.0),
            background: Color(UIColor.systemBackground),
            surface: Color(UIColor.secondarySystemBackground),
            surfaceVariant: Color(UIColor.tertiarySystemBackground),
            onPrimary: Color.white,
            onSecondary: Color.black,
            onBackground: Color(UIColor.label),
            onSurface: Color(UIColor.label),
            success: Color(red: 0.0, green: 0.6, blue: 0.5), // Teal
            warning: Color(red: 0.9, green: 0.6, blue: 0.0), // Orange
            error: Color(red: 0.8, green: 0.4, blue: 0.4), // Muted red
            info: Color(red: 0.0, green: 0.45, blue: 0.7), // Blue
            painLow: Color(red: 0.0, green: 0.6, blue: 0.5),
            painMedium: Color(red: 0.9, green: 0.6, blue: 0.0),
            painHigh: Color(red: 0.7, green: 0.3, blue: 0.0),
            painSevere: Color(red: 0.8, green: 0.4, blue: 0.4),
            chartPrimary: Color(red: 0.0, green: 0.45, blue: 0.7),
            chartSecondary: Color(red: 0.9, green: 0.6, blue: 0.0),
            chartTertiary: Color(red: 0.0, green: 0.6, blue: 0.5),
            chartQuaternary: Color(red: 0.35, green: 0.7, blue: 0.9),
            buttonPrimary: Color(red: 0.0, green: 0.45, blue: 0.7),
            buttonSecondary: Color(UIColor.secondarySystemFill),
            buttonDisabled: Color(UIColor.quaternaryLabel),
            link: Color(red: 0.0, green: 0.45, blue: 0.7),
            border: Color(UIColor.separator),
            divider: Color(UIColor.separator),
            shadow: Color.black.opacity(0.1)
        )
    }
    
    private static func createTypography() -> ThemeTypography {
        return ThemeTypography(
            largeTitle: .largeTitle,
            title1: .title,
            title2: .title2,
            title3: .title3,
            headline: .headline,
            body: .body,
            callout: .callout,
            subheadline: .subheadline,
            footnote: .footnote,
            caption1: .caption,
            caption2: .caption2
        )
    }
    
    private static func createAccessibleTypography() -> ThemeTypography {
        return ThemeTypography(
            largeTitle: .system(size: 40, weight: .bold),
            title1: .system(size: 32, weight: .bold),
            title2: .system(size: 28, weight: .bold),
            title3: .system(size: 24, weight: .semibold),
            headline: .system(size: 20, weight: .semibold),
            body: .system(size: 18, weight: .regular),
            callout: .system(size: 18, weight: .regular),
            subheadline: .system(size: 16, weight: .regular),
            footnote: .system(size: 14, weight: .regular),
            caption1: .system(size: 14, weight: .regular),
            caption2: .system(size: 12, weight: .regular)
        )
    }
    
    private static func createShadows(for theme: AppTheme) -> ThemeShadows {
        let shadowColor = theme == .highContrast ? Color.clear : Color.black.opacity(theme == .dark ? 0.3 : 0.1)
        
        return ThemeShadows(
            small: (color: shadowColor, radius: 2, x: 0, y: 1),
            medium: (color: shadowColor, radius: 4, x: 0, y: 2),
            large: (color: shadowColor, radius: 8, x: 0, y: 4)
        )
    }
}

// MARK: - Theme Environment

struct ThemeEnvironmentKey: EnvironmentKey {
    static let defaultValue = ThemeManager()
}

extension EnvironmentValues {
    var theme: ThemeManager {
        get { self[ThemeEnvironmentKey.self] }
        set { self[ThemeEnvironmentKey.self] = newValue }
    }
}

// MARK: - View Extensions

extension View {
    func themedBackground() -> some View {
        self.background(Color(UIColor.systemBackground))
    }
    
    func themedSurface() -> some View {
        self.background(Color(UIColor.secondarySystemBackground))
    }
    
    func themedShadow(_ size: String = "medium") -> some View {
        self.modifier(ThemedShadowModifier(size: size))
    }
    
    func themedCornerRadius(_ size: String = "medium") -> some View {
        self.modifier(ThemedCornerRadiusModifier(size: size))
    }
    
    func hapticFeedback(on gesture: some Gesture, style: UIImpactFeedbackGenerator.FeedbackStyle = .light) -> some View {
        self.modifier(HapticFeedbackModifier(gesture: gesture, style: style))
    }
}

// MARK: - View Modifiers

struct ThemedShadowModifier: ViewModifier {
    @Environment(\.theme) var theme
    let size: String
    
    func body(content: Content) -> some View {
        let shadow: (color: Color, radius: CGFloat, x: CGFloat, y: CGFloat)
        
        switch size {
        case "small":
            shadow = theme.shadows.small
        case "large":
            shadow = theme.shadows.large
        default:
            shadow = theme.shadows.medium
        }
        
        return content
            .shadow(color: shadow.color, radius: shadow.radius, x: shadow.x, y: shadow.y)
    }
}

struct ThemedCornerRadiusModifier: ViewModifier {
    @Environment(\.theme) var theme
    let size: String
    
    func body(content: Content) -> some View {
        let radius: CGFloat
        
        switch size {
        case "small":
            radius = theme.cornerRadius.small
        case "large":
            radius = theme.cornerRadius.large
        case "extraLarge":
            radius = theme.cornerRadius.extraLarge
        case "round":
            radius = theme.cornerRadius.round
        default:
            radius = theme.cornerRadius.medium
        }
        
        return content
            .cornerRadius(radius)
    }
}

struct HapticFeedbackModifier<G: Gesture>: ViewModifier {
    @Environment(\.theme) var theme
    let gesture: G
    let style: UIImpactFeedbackGenerator.FeedbackStyle
    
    func body(content: Content) -> some View {
        content
            .gesture(
                gesture.onEnded { _ in
                    theme.triggerHapticFeedback(style)
                }
            )
    }
}