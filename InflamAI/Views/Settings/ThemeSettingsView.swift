//
//  ThemeSettingsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI

struct ThemeSettingsView: View {
    @Environment(\.theme) var theme
    @Environment(\.presentationMode) var presentationMode
    
    @State private var selectedTheme: AppTheme
    @State private var isHighContrastEnabled: Bool
    @State private var isHapticFeedbackEnabled = true
    @State private var isReduceMotionEnabled: Bool
    @State private var isDynamicTypeEnabled: Bool
    @State private var showingThemePreview = false
    
    init() {
        let themeManager = ThemeManager()
        _selectedTheme = State(initialValue: themeManager.currentTheme)
        _isHighContrastEnabled = State(initialValue: themeManager.isHighContrastEnabled)
        _isReduceMotionEnabled = State(initialValue: themeManager.isReduceMotionEnabled)
        _isDynamicTypeEnabled = State(initialValue: themeManager.isDynamicTypeEnabled)
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: theme.spacing.lg) {
                    // Theme Selection Section
                    themeSelectionSection
                    
                    // Accessibility Section
                    accessibilitySection
                    
                    // Haptic Feedback Section
                    hapticFeedbackSection
                    
                    // Preview Section
                    previewSection
                    
                    Spacer(minLength: theme.spacing.xl)
                }
                .padding(theme.spacing.md)
            }
            .navigationTitle("Theme & Accessibility")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        theme.triggerHapticFeedback(.light)
                        presentationMode.wrappedValue.dismiss()
                    }
                    .foregroundColor(theme.colors.primary)
                }
            }
        }
        .themedBackground()
        .onAppear {
            updateStateFromTheme()
        }
    }
    
    // MARK: - Theme Selection Section
    
    private var themeSelectionSection: some View {
        ThemedCard {
            VStack(alignment: .leading, spacing: theme.spacing.md) {
                HStack {
                    Image(systemName: "paintbrush.fill")
                        .foregroundColor(theme.colors.primary)
                        .font(.title2)
                    
                    Text("Appearance")
                        .font(theme.typography.headline)
                        .foregroundColor(theme.colors.onSurface)
                    
                    Spacer()
                }
                
                ThemedDivider()
                
                VStack(spacing: theme.spacing.sm) {
                    ForEach(AppTheme.allCases) { themeOption in
                        ThemeOptionRow(
                            theme: themeOption,
                            isSelected: selectedTheme == themeOption
                        ) {
                            selectTheme(themeOption)
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Accessibility Section
    
    private var accessibilitySection: some View {
        ThemedCard {
            VStack(alignment: .leading, spacing: theme.spacing.md) {
                HStack {
                    Image(systemName: "accessibility")
                        .foregroundColor(theme.colors.primary)
                        .font(.title2)
                    
                    Text("Accessibility")
                        .font(theme.typography.headline)
                        .foregroundColor(theme.colors.onSurface)
                    
                    Spacer()
                }
                
                ThemedDivider()
                
                VStack(spacing: theme.spacing.md) {
                    ThemedToggle(
                        "High Contrast",
                        isOn: $isHighContrastEnabled,
                        description: "Increases contrast for better visibility"
                    )
                    .onChange(of: isHighContrastEnabled) { enabled in
                        theme.enableHighContrast(enabled)
                        updateSelectedTheme()
                    }
                    
                    ThemedToggle(
                        "Reduce Motion",
                        isOn: $isReduceMotionEnabled,
                        description: "Minimizes animations and transitions"
                    )
                    .onChange(of: isReduceMotionEnabled) { _ in
                        theme.updateForAccessibility()
                    }
                    
                    ThemedToggle(
                        "Dynamic Type",
                        isOn: $isDynamicTypeEnabled,
                        description: "Adjusts text size based on system settings"
                    )
                    .onChange(of: isDynamicTypeEnabled) { enabled in
                        // Update theme typography based on dynamic type setting
                        if enabled {
                            theme.updateForAccessibility()
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Haptic Feedback Section
    
    private var hapticFeedbackSection: some View {
        ThemedCard {
            VStack(alignment: .leading, spacing: theme.spacing.md) {
                HStack {
                    Image(systemName: "iphone.radiowaves.left.and.right")
                        .foregroundColor(theme.colors.primary)
                        .font(.title2)
                    
                    Text("Haptic Feedback")
                        .font(theme.typography.headline)
                        .foregroundColor(theme.colors.onSurface)
                    
                    Spacer()
                }
                
                ThemedDivider()
                
                VStack(spacing: theme.spacing.md) {
                    ThemedToggle(
                        "Enable Haptic Feedback",
                        isOn: $isHapticFeedbackEnabled,
                        description: "Provides tactile feedback for interactions"
                    )
                    
                    if isHapticFeedbackEnabled {
                        VStack(spacing: theme.spacing.sm) {
                            Text("Test Haptic Feedback")
                                .font(theme.typography.subheadline)
                                .foregroundColor(theme.colors.onSurface)
                            
                            HStack(spacing: theme.spacing.sm) {
                                ThemedButton("Light", size: .small) {
                                    theme.triggerHapticFeedback(.light)
                                }
                                
                                ThemedButton("Medium", size: .small) {
                                    theme.triggerHapticFeedback(.medium)
                                }
                                
                                ThemedButton("Heavy", size: .small) {
                                    theme.triggerHapticFeedback(.heavy)
                                }
                            }
                            
                            HStack(spacing: theme.spacing.sm) {
                                ThemedButton("Success", style: .secondary, size: .small) {
                                    theme.triggerNotificationFeedback(.success)
                                }
                                
                                ThemedButton("Warning", style: .secondary, size: .small) {
                                    theme.triggerNotificationFeedback(.warning)
                                }
                                
                                ThemedButton("Error", style: .secondary, size: .small) {
                                    theme.triggerNotificationFeedback(.error)
                                }
                            }
                        }
                        .padding(.top, theme.spacing.sm)
                    }
                }
            }
        }
    }
    
    // MARK: - Preview Section
    
    private var previewSection: some View {
        ThemedCard {
            VStack(alignment: .leading, spacing: theme.spacing.md) {
                HStack {
                    Image(systemName: "eye")
                        .foregroundColor(theme.colors.primary)
                        .font(.title2)
                    
                    Text("Preview")
                        .font(theme.typography.headline)
                        .foregroundColor(theme.colors.onSurface)
                    
                    Spacer()
                    
                    Button("Show Full Preview") {
                        showingThemePreview = true
                        theme.triggerHapticFeedback(.light)
                    }
                    .font(theme.typography.caption1)
                    .foregroundColor(theme.colors.primary)
                }
                
                ThemedDivider()
                
                // Mini preview of current theme
                VStack(spacing: theme.spacing.sm) {
                    HStack {
                        ThemedBadge("Primary", style: .primary)
                        ThemedBadge("Success", style: .success)
                        ThemedBadge("Warning", style: .warning)
                        ThemedBadge("Error", style: .error)
                        Spacer()
                    }
                    
                    ThemedProgressBar(value: 0.7, showPercentage: true)
                    
                    HStack {
                        ThemedButton("Primary", size: .small) {}
                        ThemedButton("Secondary", style: .secondary, size: .small) {}
                        Spacer()
                    }
                }
            }
        }
        .sheet(isPresented: $showingThemePreview) {
            ThemePreviewView()
        }
    }
    
    // MARK: - Helper Methods
    
    private func selectTheme(_ newTheme: AppTheme) {
        selectedTheme = newTheme
        theme.setTheme(newTheme)
        theme.triggerSelectionFeedback()
    }
    
    private func updateStateFromTheme() {
        selectedTheme = theme.currentTheme
        isHighContrastEnabled = theme.isHighContrastEnabled
        isReduceMotionEnabled = theme.isReduceMotionEnabled
        isDynamicTypeEnabled = theme.isDynamicTypeEnabled
    }
    
    private func updateSelectedTheme() {
        selectedTheme = theme.currentTheme
    }
}

// MARK: - Theme Option Row

struct ThemeOptionRow: View {
    @Environment(\.theme) var themeManager
    
    let theme: AppTheme
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: themeManager.spacing.md) {
                Image(systemName: theme.iconName)
                    .foregroundColor(isSelected ? themeManager.colors.primary : themeManager.colors.onSurface.opacity(0.7))
                    .font(.title3)
                    .frame(width: 24)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(theme.displayName)
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.onSurface)
                    
                    Text(themeDescription(for: theme))
                        .font(themeManager.typography.caption1)
                        .foregroundColor(themeManager.colors.onSurface.opacity(0.7))
                }
                
                Spacer()
                
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(themeManager.colors.primary)
                        .font(.title3)
                }
            }
            .padding(.vertical, themeManager.spacing.xs)
        }
        .buttonStyle(PlainButtonStyle())
    }
    
    private func themeDescription(for theme: AppTheme) -> String {
        switch theme {
        case .system:
            return "Follows system appearance"
        case .light:
            return "Light colors and backgrounds"
        case .dark:
            return "Dark colors and backgrounds"
        case .highContrast:
            return "High contrast for better visibility"
        case .colorBlind:
            return "Optimized for color blind users"
        }
    }
}

// MARK: - Theme Preview View

struct ThemePreviewView: View {
    @Environment(\.theme) var theme
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: theme.spacing.lg) {
                    // Sample Dashboard Content
                    ThemedCard {
                        VStack(alignment: .leading, spacing: theme.spacing.md) {
                            Text("Pain Tracking")
                                .font(theme.typography.headline)
                                .foregroundColor(theme.colors.onSurface)
                            
                            HStack {
                                VStack(alignment: .leading) {
                                    Text("Current Level")
                                        .font(theme.typography.caption1)
                                        .foregroundColor(theme.colors.onSurface.opacity(0.7))
                                    
                                    Text("6/10")
                                        .font(theme.typography.title2)
                                        .foregroundColor(theme.getPainLevelColor(for: 6))
                                }
                                
                                Spacer()
                                
                                Circle()
                                    .fill(theme.getPainLevelColor(for: 6))
                                    .frame(width: 40, height: 40)
                            }
                            
                            ThemedProgressBar(value: 0.6, color: theme.getPainLevelColor(for: 6), showPercentage: true)
                        }
                    }
                    
                    // Sample Medication Card
                    ThemedCard {
                        VStack(alignment: .leading, spacing: theme.spacing.md) {
                            Text("Medications")
                                .font(theme.typography.headline)
                                .foregroundColor(theme.colors.onSurface)
                            
                            HStack {
                                VStack(alignment: .leading) {
                                    Text("Next Dose")
                                        .font(theme.typography.caption1)
                                        .foregroundColor(theme.colors.onSurface.opacity(0.7))
                                    
                                    Text("Methotrexate")
                                        .font(theme.typography.body)
                                        .foregroundColor(theme.colors.onSurface)
                                    
                                    Text("2:00 PM")
                                        .font(theme.typography.caption1)
                                        .foregroundColor(theme.colors.primary)
                                }
                                
                                Spacer()
                                
                                ThemedBadge("Due Soon", style: .warning)
                            }
                        }
                    }
                    
                    // Sample Buttons
                    VStack(spacing: theme.spacing.md) {
                        ThemedButton("Log Pain Entry") {}
                        ThemedButton("View Analytics", style: .secondary) {}
                        ThemedButton("Emergency Contact", style: .outline) {}
                    }
                    
                    // Sample Alerts
                    VStack(spacing: theme.spacing.sm) {
                        ThemedAlert(title: "Reminder", message: "Time to take your medication", type: .info)
                        ThemedAlert(title: "Success", message: "Pain entry logged successfully", type: .success)
                        ThemedAlert(title: "Warning", message: "High pain level detected", type: .warning)
                    }
                }
                .padding(theme.spacing.md)
            }
            .navigationTitle("Theme Preview")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        theme.triggerHapticFeedback(.light)
                        presentationMode.wrappedValue.dismiss()
                    }
                    .foregroundColor(theme.colors.primary)
                }
            }
        }
        .themedBackground()
    }
}

// MARK: - Preview

#if DEBUG
struct ThemeSettingsView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            ThemeSettingsView()
                .environmentObject(ThemeManager())
                .preferredColorScheme(.light)
            
            ThemeSettingsView()
                .environmentObject(ThemeManager())
                .preferredColorScheme(.dark)
        }
    }
}
#endif