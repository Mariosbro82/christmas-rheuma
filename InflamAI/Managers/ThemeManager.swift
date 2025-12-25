//
//  ThemeManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Combine

// MARK: - Theme Manager

class ThemeManager: ObservableObject {
    static let shared = ThemeManager()
    
    @Published var currentTheme: AppTheme = .system
    @Published var isDarkMode: Bool = false
    @Published var useHighContrast: Bool = false
    @Published var textScale: TextScale = .medium
    @Published var reduceMotion: Bool = false
    @Published var hapticFeedback: Bool = true
    
    // Color schemes
    @Published var colors: ColorScheme = ColorScheme()
    @Published var typography: Typography = Typography()
    @Published var spacing: Spacing = Spacing()
    @Published var animations: AnimationSettings = AnimationSettings()
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        loadSettings()
        setupObservers()
        updateColorScheme()
    }
    
    // MARK: - Theme Management
    
    func setTheme(_ theme: AppTheme) {
        currentTheme = theme
        updateColorScheme()
        saveSettings()
    }
    
    func toggleDarkMode() {
        if currentTheme == .system {
            currentTheme = isDarkMode ? .light : .dark
        } else {
            currentTheme = currentTheme == .dark ? .light : .dark
        }
        updateColorScheme()
        saveSettings()
        
        if hapticFeedback {
            HapticManager.shared.impact(.light)
        }
    }
    
    func setHighContrast(_ enabled: Bool) {
        useHighContrast = enabled
        updateColorScheme()
        saveSettings()
    }
    
    func setTextScale(_ scale: TextScale) {
        textScale = scale
        updateTypography()
        saveSettings()
    }
    
    func setReduceMotion(_ enabled: Bool) {
        reduceMotion = enabled
        updateAnimations()
        saveSettings()
    }
    
    func setHapticFeedback(_ enabled: Bool) {
        hapticFeedback = enabled
        saveSettings()
    }
    
    // MARK: - Private Methods
    
    private func setupObservers() {
        // Observe system color scheme changes
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                self?.updateColorScheme()
            }
            .store(in: &cancellables)
        
        // Observe accessibility settings
        NotificationCenter.default.publisher(for: UIAccessibility.reduceMotionStatusDidChangeNotification)
            .sink { [weak self] _ in
                self?.updateAccessibilitySettings()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIAccessibility.darkerSystemColorsStatusDidChangeNotification)
            .sink { [weak self] _ in
                self?.updateAccessibilitySettings()
            }
            .store(in: &cancellables)
    }
    
    private func updateColorScheme() {
        let systemIsDark = UITraitCollection.current.userInterfaceStyle == .dark
        
        switch currentTheme {
        case .light:
            isDarkMode = false
        case .dark:
            isDarkMode = true
        case .system:
            isDarkMode = systemIsDark
        }
        
        if useHighContrast {
            colors = isDarkMode ? HighContrastDarkColorScheme() : HighContrastLightColorScheme()
        } else {
            colors = isDarkMode ? DarkColorScheme() : LightColorScheme()
        }
    }
    
    private func updateTypography() {
        typography = Typography(scale: textScale)
    }
    
    private func updateAnimations() {
        animations = AnimationSettings(reduceMotion: reduceMotion)
    }
    
    private func updateAccessibilitySettings() {
        let systemReduceMotion = UIAccessibility.isReduceMotionEnabled
        let systemHighContrast = UIAccessibility.isDarkerSystemColorsEnabled
        
        if systemReduceMotion != reduceMotion {
            reduceMotion = systemReduceMotion
            updateAnimations()
        }
        
        if systemHighContrast != useHighContrast {
            useHighContrast = systemHighContrast
            updateColorScheme()
        }
    }
    
    // MARK: - Persistence
    
    private func saveSettings() {
        UserDefaults.standard.set(currentTheme.rawValue, forKey: "app_theme")
        UserDefaults.standard.set(useHighContrast, forKey: "high_contrast")
        UserDefaults.standard.set(textScale.rawValue, forKey: "text_scale")
        UserDefaults.standard.set(reduceMotion, forKey: "reduce_motion")
        UserDefaults.standard.set(hapticFeedback, forKey: "haptic_feedback")
    }
    
    private func loadSettings() {
        if let themeRaw = UserDefaults.standard.object(forKey: "app_theme") as? String,
           let theme = AppTheme(rawValue: themeRaw) {
            currentTheme = theme
        }
        
        useHighContrast = UserDefaults.standard.bool(forKey: "high_contrast")
        reduceMotion = UserDefaults.standard.bool(forKey: "reduce_motion")
        hapticFeedback = UserDefaults.standard.object(forKey: "haptic_feedback") as? Bool ?? true
        
        if let scaleRaw = UserDefaults.standard.object(forKey: "text_scale") as? String,
           let scale = TextScale(rawValue: scaleRaw) {
            textScale = scale
        }
    }
}

// MARK: - Theme Enums

enum AppTheme: String, CaseIterable {
    case light = "light"
    case dark = "dark"
    case system = "system"
    
    var displayName: String {
        switch self {
        case .light: return "Light"
        case .dark: return "Dark"
        case .system: return "System"
        }
    }
    
    var icon: String {
        switch self {
        case .light: return "sun.max"
        case .dark: return "moon"
        case .system: return "circle.lefthalf.filled"
        }
    }
}

enum TextScale: String, CaseIterable {
    case small = "small"
    case medium = "medium"
    case large = "large"
    case extraLarge = "extraLarge"
    
    var displayName: String {
        switch self {
        case .small: return "Small"
        case .medium: return "Medium"
        case .large: return "Large"
        case .extraLarge: return "Extra Large"
        }
    }
    
    var multiplier: CGFloat {
        switch self {
        case .small: return 0.85
        case .medium: return 1.0
        case .large: return 1.15
        case .extraLarge: return 1.3
        }
    }
}

// MARK: - Color Schemes

protocol ColorSchemeProtocol {
    var primary: Color { get }
    var secondary: Color { get }
    var accent: Color { get }
    var background: Color { get }
    var surface: Color { get }
    var textPrimary: Color { get }
    var textSecondary: Color { get }
    var textTertiary: Color { get }
    var border: Color { get }
    var error: Color { get }
    var warning: Color { get }
    var success: Color { get }
    var info: Color { get }
    var shadow: Color { get }
}

struct ColorScheme: ColorSchemeProtocol {
    var primary: Color = .blue
    var secondary: Color = .gray
    var accent: Color = .orange
    var background: Color = .white
    var surface: Color = .gray.opacity(0.1)
    var textPrimary: Color = .black
    var textSecondary: Color = .gray
    var textTertiary: Color = .gray.opacity(0.6)
    var border: Color = .gray.opacity(0.3)
    var error: Color = .red
    var warning: Color = .orange
    var success: Color = .green
    var info: Color = .blue
    var shadow: Color = .black.opacity(0.1)
}

struct LightColorScheme: ColorSchemeProtocol {
    let primary = Color(red: 0.2, green: 0.4, blue: 0.8)
    let secondary = Color(red: 0.5, green: 0.5, blue: 0.5)
    let accent = Color(red: 1.0, green: 0.6, blue: 0.0)
    let background = Color.white
    let surface = Color(red: 0.98, green: 0.98, blue: 0.98)
    let textPrimary = Color(red: 0.1, green: 0.1, blue: 0.1)
    let textSecondary = Color(red: 0.4, green: 0.4, blue: 0.4)
    let textTertiary = Color(red: 0.6, green: 0.6, blue: 0.6)
    let border = Color(red: 0.9, green: 0.9, blue: 0.9)
    let error = Color(red: 0.9, green: 0.2, blue: 0.2)
    let warning = Color(red: 1.0, green: 0.6, blue: 0.0)
    let success = Color(red: 0.2, green: 0.7, blue: 0.3)
    let info = Color(red: 0.2, green: 0.4, blue: 0.8)
    let shadow = Color.black.opacity(0.1)
}

struct DarkColorScheme: ColorSchemeProtocol {
    let primary = Color(red: 0.3, green: 0.5, blue: 0.9)
    let secondary = Color(red: 0.6, green: 0.6, blue: 0.6)
    let accent = Color(red: 1.0, green: 0.7, blue: 0.2)
    let background = Color(red: 0.05, green: 0.05, blue: 0.05)
    let surface = Color(red: 0.1, green: 0.1, blue: 0.1)
    let textPrimary = Color(red: 0.95, green: 0.95, blue: 0.95)
    let textSecondary = Color(red: 0.7, green: 0.7, blue: 0.7)
    let textTertiary = Color(red: 0.5, green: 0.5, blue: 0.5)
    let border = Color(red: 0.2, green: 0.2, blue: 0.2)
    let error = Color(red: 1.0, green: 0.3, blue: 0.3)
    let warning = Color(red: 1.0, green: 0.7, blue: 0.2)
    let success = Color(red: 0.3, green: 0.8, blue: 0.4)
    let info = Color(red: 0.3, green: 0.5, blue: 0.9)
    let shadow = Color.black.opacity(0.3)
}

struct HighContrastLightColorScheme: ColorSchemeProtocol {
    let primary = Color.blue
    let secondary = Color.black
    let accent = Color.orange
    let background = Color.white
    let surface = Color(red: 0.95, green: 0.95, blue: 0.95)
    let textPrimary = Color.black
    let textSecondary = Color(red: 0.2, green: 0.2, blue: 0.2)
    let textTertiary = Color(red: 0.4, green: 0.4, blue: 0.4)
    let border = Color.black
    let error = Color.red
    let warning = Color.orange
    let success = Color.green
    let info = Color.blue
    let shadow = Color.black.opacity(0.2)
}

struct HighContrastDarkColorScheme: ColorSchemeProtocol {
    let primary = Color.cyan
    let secondary = Color.white
    let accent = Color.yellow
    let background = Color.black
    let surface = Color(red: 0.1, green: 0.1, blue: 0.1)
    let textPrimary = Color.white
    let textSecondary = Color(red: 0.9, green: 0.9, blue: 0.9)
    let textTertiary = Color(red: 0.7, green: 0.7, blue: 0.7)
    let border = Color.white
    let error = Color.red
    let warning = Color.yellow
    let success = Color.green
    let info = Color.cyan
    let shadow = Color.white.opacity(0.1)
}

// MARK: - Typography

struct Typography {
    let scale: TextScale
    
    init(scale: TextScale = .medium) {
        self.scale = scale
    }
    
    var largeTitle: Font {
        .system(size: 34 * scale.multiplier, weight: .bold, design: .default)
    }
    
    var title1: Font {
        .system(size: 28 * scale.multiplier, weight: .bold, design: .default)
    }
    
    var title2: Font {
        .system(size: 22 * scale.multiplier, weight: .bold, design: .default)
    }
    
    var title3: Font {
        .system(size: 20 * scale.multiplier, weight: .semibold, design: .default)
    }
    
    var headline: Font {
        .system(size: 17 * scale.multiplier, weight: .semibold, design: .default)
    }
    
    var body: Font {
        .system(size: 17 * scale.multiplier, weight: .regular, design: .default)
    }
    
    var callout: Font {
        .system(size: 16 * scale.multiplier, weight: .regular, design: .default)
    }
    
    var subheadline: Font {
        .system(size: 15 * scale.multiplier, weight: .regular, design: .default)
    }
    
    var footnote: Font {
        .system(size: 13 * scale.multiplier, weight: .regular, design: .default)
    }
    
    var caption: Font {
        .system(size: 12 * scale.multiplier, weight: .regular, design: .default)
    }
    
    var caption2: Font {
        .system(size: 11 * scale.multiplier, weight: .regular, design: .default)
    }
}

// MARK: - Spacing

struct Spacing {
    let xs: CGFloat = 4
    let sm: CGFloat = 8
    let md: CGFloat = 16
    let lg: CGFloat = 24
    let xl: CGFloat = 32
    let xxl: CGFloat = 48
    
    // Semantic spacing
    let cardPadding: CGFloat = 16
    let sectionSpacing: CGFloat = 24
    let itemSpacing: CGFloat = 12
    let borderRadius: CGFloat = 12
    let buttonHeight: CGFloat = 44
}

// MARK: - Animation Settings

struct AnimationSettings {
    let reduceMotion: Bool
    
    init(reduceMotion: Bool = false) {
        self.reduceMotion = reduceMotion
    }
    
    var standard: Animation {
        reduceMotion ? .none : .easeInOut(duration: 0.3)
    }
    
    var quick: Animation {
        reduceMotion ? .none : .easeInOut(duration: 0.2)
    }
    
    var slow: Animation {
        reduceMotion ? .none : .easeInOut(duration: 0.5)
    }
    
    var spring: Animation {
        reduceMotion ? .none : .spring(response: 0.5, dampingFraction: 0.8)
    }
    
    var bounce: Animation {
        reduceMotion ? .none : .spring(response: 0.3, dampingFraction: 0.6)
    }
}

// MARK: - Haptic Manager

class HapticManager {
    static let shared = HapticManager()
    
    private init() {}
    
    func impact(_ style: UIImpactFeedbackGenerator.FeedbackStyle) {
        guard ThemeManager.shared.hapticFeedback else { return }
        
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.impactOccurred()
    }
    
    func notification(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        guard ThemeManager.shared.hapticFeedback else { return }
        
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(type)
    }
    
    func selection() {
        guard ThemeManager.shared.hapticFeedback else { return }
        
        let generator = UISelectionFeedbackGenerator()
        generator.selectionChanged()
    }
}

// MARK: - Theme Extensions

extension View {
    func themedBackground() -> some View {
        self.background(ThemeManager.shared.colors.background)
    }
    
    func themedSurface() -> some View {
        self.background(
            RoundedRectangle(cornerRadius: ThemeManager.shared.spacing.borderRadius)
                .fill(ThemeManager.shared.colors.surface)
        )
    }
    
    func themedCard() -> some View {
        self
            .padding(ThemeManager.shared.spacing.cardPadding)
            .background(
                RoundedRectangle(cornerRadius: ThemeManager.shared.spacing.borderRadius)
                    .fill(ThemeManager.shared.colors.surface)
                    .shadow(
                        color: ThemeManager.shared.colors.shadow,
                        radius: 4,
                        x: 0,
                        y: 2
                    )
            )
    }
    
    func themedButton(style: ButtonStyle = .primary) -> some View {
        self.modifier(ThemedButtonModifier(style: style))
    }
    
    func hapticFeedback(_ style: UIImpactFeedbackGenerator.FeedbackStyle = .light) -> some View {
        self.onTapGesture {
            HapticManager.shared.impact(style)
        }
    }
}

// MARK: - Button Styles

enum ButtonStyle {
    case primary
    case secondary
    case tertiary
    case destructive
}

struct ThemedButtonModifier: ViewModifier {
    @EnvironmentObject private var themeManager: ThemeManager
    let style: ButtonStyle
    
    func body(content: Content) -> some View {
        content
            .font(themeManager.typography.body)
            .fontWeight(.medium)
            .foregroundColor(textColor)
            .frame(height: themeManager.spacing.buttonHeight)
            .background(
                RoundedRectangle(cornerRadius: themeManager.spacing.borderRadius)
                    .fill(backgroundColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: themeManager.spacing.borderRadius)
                    .stroke(borderColor, lineWidth: borderWidth)
            )
    }
    
    private var backgroundColor: Color {
        switch style {
        case .primary:
            return themeManager.colors.primary
        case .secondary:
            return themeManager.colors.surface
        case .tertiary:
            return Color.clear
        case .destructive:
            return themeManager.colors.error
        }
    }
    
    private var textColor: Color {
        switch style {
        case .primary, .destructive:
            return .white
        case .secondary:
            return themeManager.colors.textPrimary
        case .tertiary:
            return themeManager.colors.primary
        }
    }
    
    private var borderColor: Color {
        switch style {
        case .primary, .destructive:
            return Color.clear
        case .secondary:
            return themeManager.colors.border
        case .tertiary:
            return themeManager.colors.primary
        }
    }
    
    private var borderWidth: CGFloat {
        switch style {
        case .primary, .destructive:
            return 0
        case .secondary, .tertiary:
            return 1
        }
    }
}

// MARK: - Theme Preview

struct ThemePreview: View {
    @StateObject private var themeManager = ThemeManager()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Theme Preview")
                .font(themeManager.typography.title1)
                .foregroundColor(themeManager.colors.textPrimary)
            
            HStack(spacing: 16) {
                ForEach(AppTheme.allCases, id: \.self) { theme in
                    Button(theme.displayName) {
                        themeManager.setTheme(theme)
                    }
                    .themedButton(style: theme == themeManager.currentTheme ? .primary : .secondary)
                }
            }
            
            VStack(spacing: 12) {
                Text("Sample Text")
                    .font(themeManager.typography.body)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Text("Secondary Text")
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
            }
            .themedCard()
            
            HStack(spacing: 12) {
                Button("Primary") {}
                    .themedButton(style: .primary)
                
                Button("Secondary") {}
                    .themedButton(style: .secondary)
                
                Button("Tertiary") {}
                    .themedButton(style: .tertiary)
            }
        }
        .padding()
        .themedBackground()
        .environmentObject(themeManager)
    }
}

#Preview {
    ThemePreview()
}