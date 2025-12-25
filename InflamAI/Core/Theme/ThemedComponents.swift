//
//  ThemedComponents.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI

// MARK: - Themed Button

struct ThemedButton: View {
    @Environment(\.theme) var theme
    
    let title: String
    let style: ButtonStyle
    let size: ButtonSize
    let action: () -> Void
    
    enum ButtonStyle {
        case primary
        case secondary
        case outline
        case text
        case destructive
    }
    
    enum ButtonSize {
        case small
        case medium
        case large
        
        var padding: EdgeInsets {
            switch self {
            case .small:
                return EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16)
            case .medium:
                return EdgeInsets(top: 12, leading: 20, bottom: 12, trailing: 20)
            case .large:
                return EdgeInsets(top: 16, leading: 24, bottom: 16, trailing: 24)
            }
        }
        
        var font: Font {
            switch self {
            case .small:
                return .caption
            case .medium:
                return .body
            case .large:
                return .headline
            }
        }
    }
    
    init(_ title: String, style: ButtonStyle = .primary, size: ButtonSize = .medium, action: @escaping () -> Void) {
        self.title = title
        self.style = style
        self.size = size
        self.action = action
    }
    
    var body: some View {
        Button(action: {
            theme.triggerHapticFeedback(.light)
            action()
        }) {
            Text(title)
                .font(size.font)
                .fontWeight(.medium)
                .foregroundColor(textColor)
                .padding(size.padding)
                .frame(maxWidth: .infinity)
                .background(backgroundColor)
                .overlay(
                    RoundedRectangle(cornerRadius: theme.cornerRadius.medium)
                        .stroke(borderColor, lineWidth: borderWidth)
                )
                .cornerRadius(theme.cornerRadius.medium)
        }
        .buttonStyle(PlainButtonStyle())
        .scaleEffect(theme.isReduceMotionEnabled ? 1.0 : 0.98)
        .animation(.easeInOut(duration: 0.1), value: theme.isReduceMotionEnabled)
    }
    
    private var backgroundColor: Color {
        switch style {
        case .primary:
            return theme.colors.buttonPrimary
        case .secondary:
            return theme.colors.buttonSecondary
        case .outline, .text:
            return Color.clear
        case .destructive:
            return theme.colors.error
        }
    }
    
    private var textColor: Color {
        switch style {
        case .primary:
            return theme.colors.onPrimary
        case .secondary:
            return theme.colors.onSurface
        case .outline:
            return theme.colors.primary
        case .text:
            return theme.colors.link
        case .destructive:
            return Color.white
        }
    }
    
    private var borderColor: Color {
        switch style {
        case .outline:
            return theme.colors.primary
        default:
            return Color.clear
        }
    }
    
    private var borderWidth: CGFloat {
        style == .outline ? 1 : 0
    }
}

// MARK: - Themed Card

struct ThemedCard<Content: View>: View {
    @Environment(\.theme) var theme
    
    let content: Content
    let padding: EdgeInsets
    let shadowSize: String
    
    init(padding: EdgeInsets = EdgeInsets(top: 16, leading: 16, bottom: 16, trailing: 16), shadowSize: String = "medium", @ViewBuilder content: () -> Content) {
        self.content = content()
        self.padding = padding
        self.shadowSize = shadowSize
    }
    
    var body: some View {
        content
            .padding(padding)
            .background(theme.colors.surface)
            .cornerRadius(theme.cornerRadius.large)
            .themedShadow(shadowSize)
    }
}

// MARK: - Themed Text Field

struct ThemedTextField: View {
    @Environment(\.theme) var theme
    @Binding var text: String
    
    let placeholder: String
    let isSecure: Bool
    let keyboardType: UIKeyboardType
    let autocapitalization: TextInputAutocapitalization
    
    @State private var isEditing = false
    
    init(_ placeholder: String, text: Binding<String>, isSecure: Bool = false, keyboardType: UIKeyboardType = .default, autocapitalization: TextInputAutocapitalization = .sentences) {
        self.placeholder = placeholder
        self._text = text
        self.isSecure = isSecure
        self.keyboardType = keyboardType
        self.autocapitalization = autocapitalization
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.xs) {
            if isSecure {
                SecureField(placeholder, text: $text)
                    .textFieldStyle()
            } else {
                TextField(placeholder, text: $text)
                    .keyboardType(keyboardType)
                    .textInputAutocapitalization(autocapitalization)
                    .textFieldStyle()
            }
        }
    }
    
    private func textFieldStyle() -> some View {
        return AnyView(
            EmptyView()
                .padding(theme.spacing.md)
                .background(theme.colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: theme.cornerRadius.medium)
                        .stroke(isEditing ? theme.colors.primary : theme.colors.border, lineWidth: isEditing ? 2 : 1)
                )
                .cornerRadius(theme.cornerRadius.medium)
                .onTapGesture {
                    isEditing = true
                    theme.triggerSelectionFeedback()
                }
        )
    }
}

// MARK: - Themed Progress Bar

struct ThemedProgressBar: View {
    @Environment(\.theme) var theme
    
    let value: Double
    let total: Double
    let color: Color?
    let height: CGFloat
    let showPercentage: Bool
    
    init(value: Double, total: Double = 1.0, color: Color? = nil, height: CGFloat = 8, showPercentage: Bool = false) {
        self.value = value
        self.total = total
        self.color = color
        self.height = height
        self.showPercentage = showPercentage
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.xs) {
            if showPercentage {
                HStack {
                    Spacer()
                    Text("\(Int((value / total) * 100))%")
                        .font(theme.typography.caption1)
                        .foregroundColor(theme.colors.onSurface)
                }
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(theme.colors.surfaceVariant)
                        .frame(height: height)
                        .cornerRadius(height / 2)
                    
                    Rectangle()
                        .fill(color ?? theme.colors.primary)
                        .frame(width: geometry.size.width * CGFloat(value / total), height: height)
                        .cornerRadius(height / 2)
                        .animation(.easeInOut(duration: theme.isReduceMotionEnabled ? 0 : 0.3), value: value)
                }
            }
            .frame(height: height)
        }
    }
}

// MARK: - Themed Toggle

struct ThemedToggle: View {
    @Environment(\.theme) var theme
    @Binding var isOn: Bool
    
    let label: String
    let description: String?
    
    init(_ label: String, isOn: Binding<Bool>, description: String? = nil) {
        self.label = label
        self._isOn = isOn
        self.description = description
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.xs) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(label)
                        .font(theme.typography.body)
                        .foregroundColor(theme.colors.onSurface)
                    
                    if let description = description {
                        Text(description)
                            .font(theme.typography.caption1)
                            .foregroundColor(theme.colors.onSurface.opacity(0.7))
                    }
                }
                
                Spacer()
                
                Toggle("", isOn: $isOn)
                    .toggleStyle(SwitchToggleStyle(tint: theme.colors.primary))
                    .onChange(of: isOn) { _ in
                        theme.triggerHapticFeedback(.light)
                    }
            }
        }
    }
}

// MARK: - Themed Picker

struct ThemedPicker<SelectionValue: Hashable, Content: View>: View {
    @Environment(\.theme) var theme
    @Binding var selection: SelectionValue
    
    let label: String
    let content: Content
    
    init(_ label: String, selection: Binding<SelectionValue>, @ViewBuilder content: () -> Content) {
        self.label = label
        self._selection = selection
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.xs) {
            Text(label)
                .font(theme.typography.headline)
                .foregroundColor(theme.colors.onSurface)
            
            Picker(label, selection: $selection) {
                content
            }
            .pickerStyle(SegmentedPickerStyle())
            .onChange(of: selection) { _ in
                theme.triggerSelectionFeedback()
            }
        }
    }
}

// MARK: - Themed Alert

struct ThemedAlert: View {
    @Environment(\.theme) var theme
    
    let title: String
    let message: String
    let type: AlertType
    let action: (() -> Void)?
    
    enum AlertType {
        case info
        case success
        case warning
        case error
        
        var iconName: String {
            switch self {
            case .info: return "info.circle.fill"
            case .success: return "checkmark.circle.fill"
            case .warning: return "exclamationmark.triangle.fill"
            case .error: return "xmark.circle.fill"
            }
        }
    }
    
    init(title: String, message: String, type: AlertType = .info, action: (() -> Void)? = nil) {
        self.title = title
        self.message = message
        self.type = type
        self.action = action
    }
    
    var body: some View {
        HStack(spacing: theme.spacing.md) {
            Image(systemName: type.iconName)
                .foregroundColor(iconColor)
                .font(.title2)
            
            VStack(alignment: .leading, spacing: theme.spacing.xs) {
                Text(title)
                    .font(theme.typography.headline)
                    .foregroundColor(theme.colors.onSurface)
                
                Text(message)
                    .font(theme.typography.body)
                    .foregroundColor(theme.colors.onSurface.opacity(0.8))
            }
            
            Spacer()
            
            if let action = action {
                Button("Dismiss") {
                    theme.triggerHapticFeedback(.light)
                    action()
                }
                .font(theme.typography.caption1)
                .foregroundColor(theme.colors.primary)
            }
        }
        .padding(theme.spacing.md)
        .background(backgroundColor)
        .cornerRadius(theme.cornerRadius.medium)
        .themedShadow("small")
    }
    
    private var iconColor: Color {
        switch type {
        case .info: return theme.colors.info
        case .success: return theme.colors.success
        case .warning: return theme.colors.warning
        case .error: return theme.colors.error
        }
    }
    
    private var backgroundColor: Color {
        switch type {
        case .info: return theme.colors.info.opacity(0.1)
        case .success: return theme.colors.success.opacity(0.1)
        case .warning: return theme.colors.warning.opacity(0.1)
        case .error: return theme.colors.error.opacity(0.1)
        }
    }
}

// MARK: - Themed Loading Indicator

struct ThemedLoadingIndicator: View {
    @Environment(\.theme) var theme
    @State private var isAnimating = false
    
    let size: CGFloat
    let color: Color?
    
    init(size: CGFloat = 24, color: Color? = nil) {
        self.size = size
        self.color = color
    }
    
    var body: some View {
        Circle()
            .trim(from: 0, to: 0.7)
            .stroke(color ?? theme.colors.primary, lineWidth: 3)
            .frame(width: size, height: size)
            .rotationEffect(Angle(degrees: isAnimating ? 360 : 0))
            .animation(
                theme.isReduceMotionEnabled ? .none : Animation.linear(duration: 1).repeatForever(autoreverses: false),
                value: isAnimating
            )
            .onAppear {
                isAnimating = true
            }
    }
}

// MARK: - Themed Divider

struct ThemedDivider: View {
    @Environment(\.theme) var theme
    
    let thickness: CGFloat
    let color: Color?
    
    init(thickness: CGFloat = 1, color: Color? = nil) {
        self.thickness = thickness
        self.color = color
    }
    
    var body: some View {
        Rectangle()
            .fill(color ?? theme.colors.divider)
            .frame(height: thickness)
    }
}

// MARK: - Themed Badge

struct ThemedBadge: View {
    @Environment(\.theme) var theme
    
    let text: String
    let style: BadgeStyle
    
    enum BadgeStyle {
        case primary
        case secondary
        case success
        case warning
        case error
        case info
        
        func backgroundColor(for theme: ThemeManager) -> Color {
            switch self {
            case .primary: return theme.colors.primary
            case .secondary: return theme.colors.secondary
            case .success: return theme.colors.success
            case .warning: return theme.colors.warning
            case .error: return theme.colors.error
            case .info: return theme.colors.info
            }
        }
        
        func textColor(for theme: ThemeManager) -> Color {
            switch self {
            case .primary: return theme.colors.onPrimary
            case .secondary: return theme.colors.onSecondary
            default: return Color.white
            }
        }
    }
    
    init(_ text: String, style: BadgeStyle = .primary) {
        self.text = text
        self.style = style
    }
    
    var body: some View {
        Text(text)
            .font(theme.typography.caption1)
            .fontWeight(.medium)
            .foregroundColor(style.textColor(for: theme))
            .padding(.horizontal, theme.spacing.sm)
            .padding(.vertical, theme.spacing.xs)
            .background(style.backgroundColor(for: theme))
            .cornerRadius(theme.cornerRadius.round)
    }
}

// MARK: - Preview Helpers

#if DEBUG
struct ThemedComponents_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            ThemedButton("Primary Button") {}
            ThemedButton("Secondary Button", style: .secondary) {}
            ThemedButton("Outline Button", style: .outline) {}
            
            ThemedCard {
                VStack {
                    Text("Card Content")
                    ThemedProgressBar(value: 0.7, showPercentage: true)
                }
            }
            
            ThemedAlert(title: "Success", message: "Operation completed successfully", type: .success)
            
            HStack {
                ThemedBadge("Primary")
                ThemedBadge("Success", style: .success)
                ThemedBadge("Warning", style: .warning)
                ThemedBadge("Error", style: .error)
            }
        }
        .padding()
        .environmentObject(ThemeManager())
    }
}
#endif