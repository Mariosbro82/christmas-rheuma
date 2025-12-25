//
//  ThemeManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import SwiftUI
import UIKit
import Combine

// MARK: - Theme Manager
@MainActor
class ThemeManager: ObservableObject {
    
    // MARK: - Published Properties
    @Published var currentTheme: AppTheme = .system
    @Published var isDarkMode: Bool = false
    @Published var accentColor: AccentColor = .blue
    @Published var fontSize: FontSize = .medium
    @Published var isHighContrast: Bool = false
    @Published var reducedMotion: Bool = false
    @Published var hapticFeedbackEnabled: Bool = true
    
    // MARK: - Private Properties
    private var cancellables = Set<AnyCancellable>()
    private let userDefaults = UserDefaults.standard
    private let hapticManager = HapticFeedbackManager()
    
    // MARK: - Keys for UserDefaults
    private enum Keys {
        static let theme = "app_theme"
        static let accentColor = "accent_color"
        static let fontSize = "font_size"
        static let highContrast = "high_contrast"
        static let reducedMotion = "reduced_motion"
        static let hapticFeedback = "haptic_feedback"
    }
    
    // MARK: - Initialization
    init() {
        loadSettings()
        setupSystemThemeObserver()
        setupAccessibilityObservers()
    }
    
    // MARK: - Public Methods
    
    func setTheme(_ theme: AppTheme) {
        currentTheme = theme
        updateDarkModeState()
        saveSettings()
        
        if hapticFeedbackEnabled {
            hapticManager.selectionChanged()
        }
    }
    
    func setAccentColor(_ color: AccentColor) {
        accentColor = color
        saveSettings()
        
        if hapticFeedbackEnabled {
            hapticManager.selectionChanged()
        }
    }
    
    func setFontSize(_ size: FontSize) {
        fontSize = size
        saveSettings()
        
        if hapticFeedbackEnabled {
            hapticManager.selectionChanged()
        }
    }
    
    func toggleHighContrast() {
        isHighContrast.toggle()
        saveSettings()
        
        if hapticFeedbackEnabled {
            hapticManager.impactOccurred(.medium)
        }
    }
    
    func toggleReducedMotion() {
        reducedMotion.toggle()
        saveSettings()
        
        if hapticFeedbackEnabled {
            hapticManager.impactOccurred(.light)
        }
    }
    
    func toggleHapticFeedback() {
        hapticFeedbackEnabled.toggle()
        saveSettings()
        
        // Give feedback for enabling, but not for disabling
        if hapticFeedbackEnabled {
            hapticManager.impactOccurred(.medium)
        }
    }
    
    func getColorScheme() -> ColorScheme? {
        switch currentTheme {
        case .light:
            return .light
        case .dark:
            return .dark
        case .system:
            return nil // Let system decide
        }
    }
    
    func getCurrentColors() -> AppColors {
        if isDarkMode {
            return isHighContrast ? AppColors.darkHighContrast : AppColors.dark
        } else {
            return isHighContrast ? AppColors.lightHighContrast : AppColors.light
        }
    }
    
    func getCurrentFonts() -> AppFonts {
        return AppFonts.forSize(fontSize)
    }
    
    func getAccentUIColor() -> UIColor {
        return accentColor.uiColor
    }
    
    func getAccentSwiftUIColor() -> Color {
        return accentColor.swiftUIColor
    }
    
    // MARK: - Private Methods
    
    private func loadSettings() {
        if let themeRawValue = userDefaults.object(forKey: Keys.theme) as? String,
           let theme = AppTheme(rawValue: themeRawValue) {
            currentTheme = theme
        }
        
        if let accentRawValue = userDefaults.object(forKey: Keys.accentColor) as? String,
           let accent = AccentColor(rawValue: accentRawValue) {
            accentColor = accent
        }
        
        if let fontRawValue = userDefaults.object(forKey: Keys.fontSize) as? String,
           let font = FontSize(rawValue: fontRawValue) {
            fontSize = font
        }
        
        isHighContrast = userDefaults.bool(forKey: Keys.highContrast)
        reducedMotion = userDefaults.bool(forKey: Keys.reducedMotion)
        hapticFeedbackEnabled = userDefaults.object(forKey: Keys.hapticFeedback) as? Bool ?? true
        
        updateDarkModeState()
    }
    
    private func saveSettings() {
        userDefaults.set(currentTheme.rawValue, forKey: Keys.theme)
        userDefaults.set(accentColor.rawValue, forKey: Keys.accentColor)
        userDefaults.set(fontSize.rawValue, forKey: Keys.fontSize)
        userDefaults.set(isHighContrast, forKey: Keys.highContrast)
        userDefaults.set(reducedMotion, forKey: Keys.reducedMotion)
        userDefaults.set(hapticFeedbackEnabled, forKey: Keys.hapticFeedback)
    }
    
    private func updateDarkModeState() {
        switch currentTheme {
        case .light:
            isDarkMode = false
        case .dark:
            isDarkMode = true
        case .system:
            isDarkMode = UITraitCollection.current.userInterfaceStyle == .dark
        }
    }
    
    private func setupSystemThemeObserver() {
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                self?.updateDarkModeState()
            }
            .store(in: &cancellables)
    }
    
    private func setupAccessibilityObservers() {
        // Observe system accessibility settings
        NotificationCenter.default.publisher(for: UIAccessibility.reduceMotionStatusDidChangeNotification)
            .sink { [weak self] _ in
                if UIAccessibility.isReduceMotionEnabled {
                    self?.reducedMotion = true
                    self?.saveSettings()
                }
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIAccessibility.darkerSystemColorsStatusDidChangeNotification)
            .sink { [weak self] _ in
                if UIAccessibility.isDarkerSystemColorsEnabled {
                    self?.isHighContrast = true
                    self?.saveSettings()
                }
            }
            .store(in: &cancellables)
    }
}

// MARK: - Haptic Feedback Manager
class HapticFeedbackManager {
    
    private let impactLight = UIImpactFeedbackGenerator(style: .light)
    private let impactMedium = UIImpactFeedbackGenerator(style: .medium)
    private let impactHeavy = UIImpactFeedbackGenerator(style: .heavy)
    private let selection = UISelectionFeedbackGenerator()
    private let notification = UINotificationFeedbackGenerator()
    
    init() {
        prepareGenerators()
    }
    
    func impactOccurred(_ style: UIImpactFeedbackGenerator.FeedbackStyle) {
        switch style {
        case .light:
            impactLight.impactOccurred()
        case .medium:
            impactMedium.impactOccurred()
        case .heavy:
            impactHeavy.impactOccurred()
        @unknown default:
            impactMedium.impactOccurred()
        }
    }
    
    func selectionChanged() {
        selection.selectionChanged()
    }
    
    func notificationOccurred(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        notification.notificationOccurred(type)
    }
    
    private func prepareGenerators() {
        impactLight.prepare()
        impactMedium.prepare()
        impactHeavy.prepare()
        selection.prepare()
        notification.prepare()
    }
}

// MARK: - Theme Types

enum AppTheme: String, CaseIterable {
    case light = "light"
    case dark = "dark"
    case system = "system"
    
    var displayName: String {
        switch self {
        case .light:
            return "Light"
        case .dark:
            return "Dark"
        case .system:
            return "System"
        }
    }
    
    var icon: String {
        switch self {
        case .light:
            return "sun.max"
        case .dark:
            return "moon"
        case .system:
            return "circle.lefthalf.filled"
        }
    }
}

enum AccentColor: String, CaseIterable {
    case blue = "blue"
    case green = "green"
    case orange = "orange"
    case red = "red"
    case purple = "purple"
    case pink = "pink"
    case teal = "teal"
    case indigo = "indigo"
    
    var displayName: String {
        return rawValue.capitalized
    }
    
    var swiftUIColor: Color {
        switch self {
        case .blue:
            return .blue
        case .green:
            return .green
        case .orange:
            return .orange
        case .red:
            return .red
        case .purple:
            return .purple
        case .pink:
            return .pink
        case .teal:
            return .teal
        case .indigo:
            return .indigo
        }
    }
    
    var uiColor: UIColor {
        switch self {
        case .blue:
            return .systemBlue
        case .green:
            return .systemGreen
        case .orange:
            return .systemOrange
        case .red:
            return .systemRed
        case .purple:
            return .systemPurple
        case .pink:
            return .systemPink
        case .teal:
            return .systemTeal
        case .indigo:
            return .systemIndigo
        }
    }
}

enum FontSize: String, CaseIterable {
    case small = "small"
    case medium = "medium"
    case large = "large"
    case extraLarge = "extraLarge"
    
    var displayName: String {
        switch self {
        case .small:
            return "Small"
        case .medium:
            return "Medium"
        case .large:
            return "Large"
        case .extraLarge:
            return "Extra Large"
        }
    }
    
    var scaleFactor: CGFloat {
        switch self {
        case .small:
            return 0.85
        case .medium:
            return 1.0
        case .large:
            return 1.15
        case .extraLarge:
            return 1.3
        }
    }
}

// MARK: - Color Schemes

struct AppColors {
    let primary: Color
    let secondary: Color
    let background: Color
    let surface: Color
    let onPrimary: Color
    let onSecondary: Color
    let onBackground: Color
    let onSurface: Color
    let error: Color
    let onError: Color
    let success: Color
    let warning: Color
    let info: Color
    
    static let light = AppColors(
        primary: Color(red: 0.2, green: 0.4, blue: 0.8),
        secondary: Color(red: 0.6, green: 0.6, blue: 0.6),
        background: Color(red: 0.98, green: 0.98, blue: 0.98),
        surface: Color.white,
        onPrimary: Color.white,
        onSecondary: Color.white,
        onBackground: Color.black,
        onSurface: Color.black,
        error: Color.red,
        onError: Color.white,
        success: Color.green,
        warning: Color.orange,
        info: Color.blue
    )
    
    static let dark = AppColors(
        primary: Color(red: 0.4, green: 0.6, blue: 1.0),
        secondary: Color(red: 0.7, green: 0.7, blue: 0.7),
        background: Color(red: 0.05, green: 0.05, blue: 0.05),
        surface: Color(red: 0.1, green: 0.1, blue: 0.1),
        onPrimary: Color.black,
        onSecondary: Color.black,
        onBackground: Color.white,
        onSurface: Color.white,
        error: Color(red: 1.0, green: 0.3, blue: 0.3),
        onError: Color.black,
        success: Color(red: 0.3, green: 1.0, blue: 0.3),
        warning: Color(red: 1.0, green: 0.7, blue: 0.3),
        info: Color(red: 0.3, green: 0.7, blue: 1.0)
    )
    
    static let lightHighContrast = AppColors(
        primary: Color.black,
        secondary: Color(red: 0.3, green: 0.3, blue: 0.3),
        background: Color.white,
        surface: Color.white,
        onPrimary: Color.white,
        onSecondary: Color.white,
        onBackground: Color.black,
        onSurface: Color.black,
        error: Color(red: 0.8, green: 0.0, blue: 0.0),
        onError: Color.white,
        success: Color(red: 0.0, green: 0.6, blue: 0.0),
        warning: Color(red: 0.8, green: 0.4, blue: 0.0),
        info: Color(red: 0.0, green: 0.0, blue: 0.8)
    )
    
    static let darkHighContrast = AppColors(
        primary: Color.white,
        secondary: Color(red: 0.8, green: 0.8, blue: 0.8),
        background: Color.black,
        surface: Color.black,
        onPrimary: Color.black,
        onSecondary: Color.black,
        onBackground: Color.white,
        onSurface: Color.white,
        error: Color(red: 1.0, green: 0.2, blue: 0.2),
        onError: Color.black,
        success: Color(red: 0.2, green: 1.0, blue: 0.2),
        warning: Color(red: 1.0, green: 0.8, blue: 0.2),
        info: Color(red: 0.2, green: 0.8, blue: 1.0)
    )
}

// MARK: - Font Schemes

struct AppFonts {
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
    
    static func forSize(_ size: FontSize) -> AppFonts {
        let scale = size.scaleFactor
        
        return AppFonts(
            largeTitle: .system(size: 34 * scale, weight: .regular, design: .default),
            title1: .system(size: 28 * scale, weight: .regular, design: .default),
            title2: .system(size: 22 * scale, weight: .regular, design: .default),
            title3: .system(size: 20 * scale, weight: .regular, design: .default),
            headline: .system(size: 17 * scale, weight: .semibold, design: .default),
            body: .system(size: 17 * scale, weight: .regular, design: .default),
            callout: .system(size: 16 * scale, weight: .regular, design: .default),
            subheadline: .system(size: 15 * scale, weight: .regular, design: .default),
            footnote: .system(size: 13 * scale, weight: .regular, design: .default),
            caption1: .system(size: 12 * scale, weight: .regular, design: .default),
            caption2: .system(size: 11 * scale, weight: .regular, design: .default)
        )
    }
}

// MARK: - Theme Environment Key

struct ThemeEnvironmentKey: EnvironmentKey {
    static let defaultValue = ThemeManager()
}

extension EnvironmentValues {
    var themeManager: ThemeManager {
        get { self[ThemeEnvironmentKey.self] }
        set { self[ThemeEnvironmentKey.self] = newValue }
    }
}

// MARK: - View Extensions

extension View {
    func themedBackground() -> some View {
        self.modifier(ThemedBackgroundModifier())
    }
    
    func themedSurface() -> some View {
        self.modifier(ThemedSurfaceModifier())
    }
    
    func themedText(_ style: ThemedTextStyle = .body) -> some View {
        self.modifier(ThemedTextModifier(style: style))
    }
    
    func hapticFeedback(_ type: HapticFeedbackType, enabled: Bool = true) -> some View {
        self.modifier(HapticFeedbackModifier(type: type, enabled: enabled))
    }
    
    func adaptiveAnimation() -> some View {
        self.modifier(AdaptiveAnimationModifier())
    }
}

// MARK: - View Modifiers

struct ThemedBackgroundModifier: ViewModifier {
    @Environment(\.themeManager) var themeManager
    
    func body(content: Content) -> some View {
        content
            .background(themeManager.getCurrentColors().background)
    }
}

struct ThemedSurfaceModifier: ViewModifier {
    @Environment(\.themeManager) var themeManager
    
    func body(content: Content) -> some View {
        content
            .background(themeManager.getCurrentColors().surface)
            .foregroundColor(themeManager.getCurrentColors().onSurface)
    }
}

enum ThemedTextStyle {
    case largeTitle, title1, title2, title3, headline, body, callout, subheadline, footnote, caption1, caption2
}

struct ThemedTextModifier: ViewModifier {
    @Environment(\.themeManager) var themeManager
    let style: ThemedTextStyle
    
    func body(content: Content) -> some View {
        let fonts = themeManager.getCurrentFonts()
        let colors = themeManager.getCurrentColors()
        
        let font: Font
        switch style {
        case .largeTitle: font = fonts.largeTitle
        case .title1: font = fonts.title1
        case .title2: font = fonts.title2
        case .title3: font = fonts.title3
        case .headline: font = fonts.headline
        case .body: font = fonts.body
        case .callout: font = fonts.callout
        case .subheadline: font = fonts.subheadline
        case .footnote: font = fonts.footnote
        case .caption1: font = fonts.caption1
        case .caption2: font = fonts.caption2
        }
        
        return content
            .font(font)
            .foregroundColor(colors.onBackground)
    }
}

enum HapticFeedbackType {
    case light, medium, heavy, selection, success, warning, error
}

struct HapticFeedbackModifier: ViewModifier {
    @Environment(\.themeManager) var themeManager
    let type: HapticFeedbackType
    let enabled: Bool
    
    func body(content: Content) -> some View {
        content
            .onTapGesture {
                guard enabled && themeManager.hapticFeedbackEnabled else { return }
                
                let hapticManager = HapticFeedbackManager()
                
                switch type {
                case .light:
                    hapticManager.impactOccurred(.light)
                case .medium:
                    hapticManager.impactOccurred(.medium)
                case .heavy:
                    hapticManager.impactOccurred(.heavy)
                case .selection:
                    hapticManager.selectionChanged()
                case .success:
                    hapticManager.notificationOccurred(.success)
                case .warning:
                    hapticManager.notificationOccurred(.warning)
                case .error:
                    hapticManager.notificationOccurred(.error)
                }
            }
    }
}

struct AdaptiveAnimationModifier: ViewModifier {
    @Environment(\.themeManager) var themeManager
    
    func body(content: Content) -> some View {
        content
            .animation(
                themeManager.reducedMotion ? .none : .easeInOut(duration: 0.3),
                value: themeManager.isDarkMode
            )
    }
}

// MARK: - Themed Components

struct ThemedButton: View {
    let title: String
    let action: () -> Void
    let style: ButtonStyle
    
    @Environment(\.themeManager) var themeManager
    
    enum ButtonStyle {
        case primary, secondary, outline, text
    }
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(themeManager.getCurrentFonts().headline)
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(backgroundColor)
                .foregroundColor(foregroundColor)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(borderColor, lineWidth: borderWidth)
                )
                .cornerRadius(8)
        }
        .hapticFeedback(.selection)
        .adaptiveAnimation()
    }
    
    private var backgroundColor: Color {
        let colors = themeManager.getCurrentColors()
        
        switch style {
        case .primary:
            return themeManager.getAccentSwiftUIColor()
        case .secondary:
            return colors.secondary
        case .outline, .text:
            return Color.clear
        }
    }
    
    private var foregroundColor: Color {
        let colors = themeManager.getCurrentColors()
        
        switch style {
        case .primary:
            return colors.onPrimary
        case .secondary:
            return colors.onSecondary
        case .outline, .text:
            return themeManager.getAccentSwiftUIColor()
        }
    }
    
    private var borderColor: Color {
        switch style {
        case .outline:
            return themeManager.getAccentSwiftUIColor()
        default:
            return Color.clear
        }
    }
    
    private var borderWidth: CGFloat {
        switch style {
        case .outline:
            return 1
        default:
            return 0
        }
    }
}

struct ThemedCard: View {
    let content: () -> AnyView
    
    @Environment(\.themeManager) var themeManager
    
    init<Content: View>(@ViewBuilder content: @escaping () -> Content) {
        self.content = { AnyView(content()) }
    }
    
    var body: some View {
        VStack {
            content()
        }
        .padding()
        .background(themeManager.getCurrentColors().surface)
        .cornerRadius(12)
        .shadow(
            color: themeManager.isDarkMode ? Color.clear : Color.black.opacity(0.1),
            radius: themeManager.isHighContrast ? 0 : 4,
            x: 0,
            y: 2
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(
                    themeManager.isHighContrast ? themeManager.getCurrentColors().onBackground : Color.clear,
                    lineWidth: themeManager.isHighContrast ? 1 : 0
                )
        )
        .adaptiveAnimation()
    }
}