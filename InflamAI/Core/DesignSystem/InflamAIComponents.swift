//
//  InflamAIComponents.swift
//  InflamAI
//
//  Reusable UI Components per UI/UX Audit Report Section 8
//  These components enforce design consistency across the app
//
//  IMPORTANT: Use these components instead of creating custom buttons/cards
//

import SwiftUI

// MARK: - Primary Button (Section 8.1)
/// Standard primary CTA button - use for main actions
/// Properties:
/// - Height: 48pt (minimum touch target)
/// - Horizontal Padding: 24pt
/// - Font: 15pt, Semibold
/// - Border Radius: full (pill shape)
/// - Background: Primary-500 (#3B82F6)
struct PrimaryButton: View {
    let title: String
    let action: () -> Void
    var isDisabled: Bool = false
    var isLoading: Bool = false
    var fullWidth: Bool = true

    @State private var isPressed: Bool = false

    var body: some View {
        Button(action: {
            guard !isDisabled && !isLoading else { return }
            // Haptic feedback
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            action()
        }) {
            HStack(spacing: Spacing.xs) {
                if isLoading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(0.8)
                }
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(.white)
            }
            .frame(maxWidth: fullWidth ? .infinity : nil)
            .frame(height: 48)
            .padding(.horizontal, fullWidth ? 0 : Spacing.lg)
        }
        .background(isDisabled ? Colors.Gray.g200 : Colors.Primary.p500)
        .cornerRadius(Radii.full)
        .disabled(isDisabled || isLoading)
        .scaleEffect(isPressed ? 0.98 : 1.0)
        .animation(.easeOut(duration: 0.2), value: isPressed)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in isPressed = true }
                .onEnded { _ in isPressed = false }
        )
        .accessibilityLabel(title)
        .accessibilityAddTraits(isDisabled ? .isButton : [.isButton])
        .accessibilityHint(isDisabled ? "Button is disabled" : "")
    }
}

// MARK: - Secondary Button (Section 8.2)
/// Secondary action button (outlined style)
/// Reference: UI/UX Audit Section 8.2
/// Properties:
/// - Height: 48pt (minimum touch target)
/// - Horizontal Padding: 24pt
/// - Font: 15pt, Semibold
/// - Border Radius: full (pill shape)
/// - Border: 1pt Gray-300 (enabled) / Gray-200 (disabled)
struct SecondaryButton: View {
    let title: String
    let action: () -> Void
    var isDisabled: Bool = false
    var fullWidth: Bool = true

    @State private var isPressed: Bool = false

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: Typography.base, weight: .semibold))
                .foregroundColor(isDisabled ? Colors.Gray.g400 : Colors.Gray.g700)
                .frame(maxWidth: fullWidth ? .infinity : nil)
                .frame(height: 48)
                .padding(.horizontal, fullWidth ? 0 : Spacing.lg)
        }
        .background(isPressed ? Colors.Gray.g50 : Color.clear)
        .overlay(
            RoundedRectangle(cornerRadius: Radii.full)
                .stroke(isDisabled ? Colors.Gray.g200 : Colors.Gray.g300, lineWidth: 1)
        )
        .cornerRadius(Radii.full)
        .disabled(isDisabled)
        .scaleEffect(isPressed ? 0.98 : 1.0)
        .animation(.easeOut(duration: 0.2), value: isPressed)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in isPressed = true }
                .onEnded { _ in isPressed = false }
        )
    }
}

// MARK: - Ghost Button
/// Text-only button for tertiary actions
struct GhostButton: View {
    let title: String
    let action: () -> Void
    var color: Color = Colors.Primary.p500

    var body: some View {
        Button(action: {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            action()
        }) {
            Text(title)
                .font(.system(size: 15, weight: .medium))
                .foregroundColor(color)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Card View (Section 8.3)
/// Consistent card container
/// Reference: UI/UX Audit Section 8.3
/// Properties:
/// - Background: White
/// - Border Radius: 12pt (customizable)
/// - Shadow: shadows.sm
/// - Padding: 16pt (md)
struct CardView<Content: View>: View {
    let content: Content
    var padding: CGFloat = Spacing.md
    var cornerRadius: CGFloat = Radii.lg

    init(
        padding: CGFloat = Spacing.md,
        cornerRadius: CGFloat = Radii.lg,
        @ViewBuilder content: () -> Content
    ) {
        self.padding = padding
        self.cornerRadius = cornerRadius
        self.content = content()
    }

    var body: some View {
        content
            .padding(padding)
            .background(Color.white)
            .cornerRadius(cornerRadius)
            .dshadow(Shadows.sm)
    }
}

// MARK: - Tappable Card View (Section 8.3)
/// Tappable card variant with press animation
/// Reference: UI/UX Audit Section 8.3
struct TappableCardView<Content: View>: View {
    let action: () -> Void
    let content: Content
    var padding: CGFloat = Spacing.md

    @State private var isPressed: Bool = false

    init(
        padding: CGFloat = Spacing.md,
        action: @escaping () -> Void,
        @ViewBuilder content: () -> Content
    ) {
        self.padding = padding
        self.action = action
        self.content = content()
    }

    var body: some View {
        Button(action: action) {
            content
                .padding(padding)
                .background(Color.white)
                .cornerRadius(Radii.lg)
                .dshadow(isPressed ? Shadows.xs : Shadows.sm)
                .scaleEffect(isPressed ? 0.98 : 1.0)
        }
        .buttonStyle(PlainButtonStyle())
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    withAnimation(.easeOut(duration: 0.1)) {
                        isPressed = true
                    }
                }
                .onEnded { _ in
                    withAnimation(.easeOut(duration: 0.1)) {
                        isPressed = false
                    }
                }
        )
    }
}

// MARK: - Interactive Card (Legacy)
/// Card with press state animation - elevates shadow on press
/// Note: Consider using TappableCardView for new code
struct InteractiveCard<Content: View>: View {
    let content: Content
    let action: () -> Void
    var padding: CGFloat = Spacing.md

    @State private var isPressed: Bool = false

    init(padding: CGFloat = Spacing.md, action: @escaping () -> Void, @ViewBuilder content: () -> Content) {
        self.content = content()
        self.action = action
        self.padding = padding
    }

    var body: some View {
        Button(action: {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            action()
        }) {
            content
                .padding(padding)
                .background(Color.white)
                .cornerRadius(Radii.lg)
                .dshadow(isPressed ? Shadows.md : Shadows.sm)
                .scaleEffect(isPressed ? 0.98 : 1.0)
        }
        .buttonStyle(.plain)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    withAnimation(Animations.easeOut) {
                        isPressed = true
                    }
                }
                .onEnded { _ in
                    withAnimation(Animations.easeOut) {
                        isPressed = false
                    }
                }
        )
    }
}

// MARK: - Input Field (Section 8.4)
/// Consistent text input field
/// Reference: UI/UX Audit Section 8.4
/// Properties:
/// - Height: 48pt
/// - Padding: 0 16pt
/// - Border: 1pt solid Gray-200 (default), 2pt Primary-500 (focus), error (error)
/// - Border Radius: 8pt
struct InputField: View {
    let placeholder: String
    @Binding var text: String
    var label: String? = nil
    var errorMessage: String? = nil
    var isSecure: Bool = false
    var keyboardType: UIKeyboardType = .default

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xxs) {
            if let label = label {
                Text(label)
                    .font(.system(size: Typography.sm, weight: .medium))
                    .foregroundColor(Colors.Gray.g700)
            }

            Group {
                if isSecure {
                    SecureField(placeholder, text: $text)
                } else {
                    TextField(placeholder, text: $text)
                        .keyboardType(keyboardType)
                }
            }
            .font(.system(size: Typography.base))
            .foregroundColor(Colors.Gray.g900)
            .padding(.horizontal, Spacing.md)
            .frame(height: 48)
            .background(Color.white)
            .cornerRadius(Radii.md)
            .overlay(
                RoundedRectangle(cornerRadius: Radii.md)
                    .stroke(borderColor, lineWidth: isFocused ? 2 : 1)
            )
            .focused($isFocused)

            if let error = errorMessage {
                Text(error)
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Semantic.error)
            }
        }
    }

    private var borderColor: Color {
        if errorMessage != nil {
            return Colors.Semantic.error
        } else if isFocused {
            return Colors.Primary.p500
        } else {
            return Colors.Gray.g200
        }
    }
}

// MARK: - Chip/Tag (Section 8.5)
/// Tag component for filters, categories, etc.
/// Properties:
/// - Height: 28pt
/// - Padding: 0 12pt
/// - Border Radius: full (pill)
/// - Font: 13pt, Medium
struct Chip: View {
    let label: String
    var icon: String? = nil
    var isSelected: Bool = false
    var style: ChipStyle = .primary
    let action: () -> Void

    enum ChipStyle {
        case primary   // Blue background when selected
        case outline   // Border only
        case subtle    // Light background
    }

    var body: some View {
        Button(action: {
            UISelectionFeedbackGenerator().selectionChanged()
            action()
        }) {
            HStack(spacing: Spacing.xxs) {
                if let icon = icon {
                    Image(systemName: icon)
                        .font(.system(size: 12))
                }
                Text(label)
                    .font(.system(size: Typography.sm, weight: .medium))
            }
            .padding(.horizontal, Spacing.sm)
            .frame(height: 28)
            .background(backgroundColor)
            .foregroundColor(foregroundColor)
            .overlay(
                Capsule()
                    .stroke(borderColor, lineWidth: style == .outline ? 1 : 0)
            )
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    private var backgroundColor: Color {
        switch style {
        case .primary:
            return isSelected ? Colors.Primary.p500 : Colors.Gray.g100
        case .outline:
            return isSelected ? Colors.Primary.p50 : Color.clear
        case .subtle:
            return isSelected ? Colors.Primary.p100 : Colors.Gray.g50
        }
    }

    private var foregroundColor: Color {
        switch style {
        case .primary:
            return isSelected ? .white : Colors.Gray.g700
        case .outline, .subtle:
            return isSelected ? Colors.Primary.p600 : Colors.Gray.g600
        }
    }

    private var borderColor: Color {
        switch style {
        case .outline:
            return isSelected ? Colors.Primary.p500 : Colors.Gray.g300
        default:
            return Color.clear
        }
    }
}

// MARK: - Section Header
/// Consistent section header styling
struct SectionHeader: View {
    let title: String
    var icon: String? = nil
    var iconColor: Color = Colors.Primary.p500
    var action: (() -> Void)? = nil
    var actionLabel: String = "See All"

    var body: some View {
        HStack {
            if let icon = icon {
                Image(systemName: icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(iconColor)
            }

            Text(title)
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Spacer()

            if let action = action {
                Button(action: action) {
                    Text(actionLabel)
                        .font(.system(size: Typography.sm, weight: .medium))
                        .foregroundColor(Colors.Primary.p500)
                }
            }
        }
    }
}

// MARK: - Navigation Header (MOD-005)
/// Unified navigation header component
/// Fixes: MOD-005 Navigation Header Inconsistency
/// Reference: UI/UX Audit - Consistent navigation headers across all screens
struct NavigationHeader: View {
    let title: String
    var subtitle: String? = nil
    var leadingAction: HeaderAction? = nil
    var trailingActions: [HeaderAction] = []

    struct HeaderAction: Identifiable {
        let id = UUID()
        let icon: String
        let action: () -> Void
        var accessibilityLabel: String = ""
    }

    var body: some View {
        HStack(spacing: Spacing.md) {
            // Leading action (back button)
            if let leading = leadingAction {
                Button(action: leading.action) {
                    Image(systemName: leading.icon)
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)
                        .frame(width: 44, height: 44)
                }
                .accessibilityLabel(leading.accessibilityLabel.isEmpty ? "Back" : leading.accessibilityLabel)
            }

            Spacer()

            // Title (centered)
            VStack(spacing: Spacing.xxxs) {
                Text(title)
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                }
            }

            Spacer()

            // Trailing actions
            HStack(spacing: Spacing.xs) {
                ForEach(trailingActions) { action in
                    Button(action: action.action) {
                        Image(systemName: action.icon)
                            .font(.system(size: 17))
                            .foregroundColor(Colors.Gray.g900)
                            .frame(width: 44, height: 44)
                    }
                    .accessibilityLabel(action.accessibilityLabel)
                }
            }

            // Invisible spacer to balance leading button when no trailing actions
            if leadingAction != nil && trailingActions.isEmpty {
                Color.clear.frame(width: 44, height: 44)
            }
        }
        .frame(height: 44)
        .padding(.horizontal, Spacing.md)
    }
}

// MARK: - NavigationHeader Convenience Initializers

extension NavigationHeader {
    /// Header with back button (chevron.left)
    static func withBackButton(
        title: String,
        subtitle: String? = nil,
        dismiss: @escaping () -> Void,
        trailingActions: [HeaderAction] = []
    ) -> NavigationHeader {
        NavigationHeader(
            title: title,
            subtitle: subtitle,
            leadingAction: HeaderAction(
                icon: "chevron.left",
                action: dismiss,
                accessibilityLabel: "Go back"
            ),
            trailingActions: trailingActions
        )
    }

    /// Header with close button (xmark) for modals
    static func withCloseButton(
        title: String,
        subtitle: String? = nil,
        dismiss: @escaping () -> Void
    ) -> NavigationHeader {
        NavigationHeader(
            title: title,
            subtitle: subtitle,
            trailingActions: [
                HeaderAction(
                    icon: "xmark",
                    action: dismiss,
                    accessibilityLabel: "Close"
                )
            ]
        )
    }

    /// Simple title-only header (no actions)
    static func titleOnly(_ title: String, subtitle: String? = nil) -> NavigationHeader {
        NavigationHeader(title: title, subtitle: subtitle)
    }
}

// MARK: - Loading Skeleton
/// Skeleton placeholder for loading states
struct SkeletonView: View {
    var width: CGFloat? = nil
    var height: CGFloat = 16
    var cornerRadius: CGFloat = Radii.sm

    @State private var isAnimating = false

    var body: some View {
        Rectangle()
            .fill(
                LinearGradient(
                    gradient: Gradient(colors: [
                        Colors.Gray.g200,
                        Colors.Gray.g100,
                        Colors.Gray.g200
                    ]),
                    startPoint: isAnimating ? .trailing : .leading,
                    endPoint: isAnimating ? .leading : .trailing
                )
            )
            .frame(width: width, height: height)
            .cornerRadius(cornerRadius)
            .onAppear {
                withAnimation(
                    Animation.linear(duration: 1.5)
                        .repeatForever(autoreverses: false)
                ) {
                    isAnimating = true
                }
            }
    }
}

// MARK: - Empty State View
/// Standard empty state with guidance
struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String
    var actionTitle: String? = nil
    var action: (() -> Void)? = nil

    var body: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: icon)
                .font(.system(size: 60))
                .foregroundColor(Colors.Gray.g300)

            Text(title)
                .font(.system(size: Typography.lg, weight: .semibold))
                .foregroundColor(Colors.Gray.g700)

            Text(message)
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g500)
                .multilineTextAlignment(.center)

            if let actionTitle = actionTitle, let action = action {
                PrimaryButton(title: actionTitle, action: action, fullWidth: false)
                    .padding(.top, Spacing.sm)
            }
        }
        .padding(Spacing.xl)
    }
}

// MARK: - Error State View
/// Standard error state with retry
struct ErrorStateView: View {
    let title: String
    let message: String
    var retryAction: (() -> Void)? = nil

    var body: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 50))
                .foregroundColor(Colors.Semantic.warning)

            Text(title)
                .font(.system(size: Typography.lg, weight: .semibold))
                .foregroundColor(Colors.Gray.g700)

            Text(message)
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g500)
                .multilineTextAlignment(.center)

            if let retry = retryAction {
                PrimaryButton(title: "Try Again", action: retry, fullWidth: false)
                    .padding(.top, Spacing.sm)
            }
        }
        .padding(Spacing.xl)
    }
}

// MARK: - Previews
#if DEBUG
struct DesignSystemPreviews: PreviewProvider {
    static var previews: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Buttons
                VStack(spacing: 12) {
                    Text("Buttons").font(.headline)
                    PrimaryButton(title: "Primary Button") {}
                    PrimaryButton(title: "Loading...", action: {}, isLoading: true)
                    PrimaryButton(title: "Disabled", action: {}, isDisabled: true)
                    SecondaryButton(title: "Cancel") {}
                    SecondaryButton(title: "Skip", action: {}, isDisabled: true)
                    GhostButton(title: "Ghost Button") {}
                }

                // Cards
                VStack(spacing: 12) {
                    Text("Cards").font(.headline)
                    CardView {
                        Text("Static Card Content")
                    }

                    TappableCardView(action: { print("tapped") }) {
                        Text("Tappable Card Content")
                    }
                }

                // Navigation Headers
                VStack(spacing: 12) {
                    Text("Navigation Headers").font(.headline)

                    NavigationHeader.withBackButton(title: "Exercise Detail", dismiss: { })

                    Divider()

                    NavigationHeader.withCloseButton(title: "Quick Log", dismiss: { })

                    Divider()

                    NavigationHeader(
                        title: "Trends",
                        subtitle: "Last 30 days",
                        leadingAction: .init(icon: "chevron.left", action: {}, accessibilityLabel: "Back"),
                        trailingActions: [
                            .init(icon: "square.and.arrow.up", action: {}, accessibilityLabel: "Share"),
                            .init(icon: "gearshape", action: {}, accessibilityLabel: "Settings")
                        ]
                    )
                }
                .background(Color(.systemBackground))

                // Chips
                VStack(spacing: 12) {
                    Text("Chips").font(.headline)
                    HStack {
                        Chip(label: "Selected", isSelected: true) {}
                        Chip(label: "Unselected", isSelected: false) {}
                        Chip(label: "Outline", isSelected: true, style: .outline) {}
                    }
                }

                // Input
                VStack(spacing: 12) {
                    Text("Input").font(.headline)
                    InputField(placeholder: "Enter email", text: .constant(""))

                    InputField(
                        placeholder: "Enter email",
                        text: .constant("test@example.com"),
                        label: "Email"
                    )

                    InputField(
                        placeholder: "Enter email",
                        text: .constant("invalid"),
                        label: "Email",
                        errorMessage: "Please enter a valid email"
                    )

                    InputField(
                        placeholder: "Password",
                        text: .constant(""),
                        label: "Password",
                        isSecure: true
                    )
                }

                // States
                EmptyStateView(
                    icon: "chart.bar.doc.horizontal",
                    title: "No Data Yet",
                    message: "Start logging to see insights",
                    actionTitle: "Get Started"
                ) {}
            }
            .padding()
        }
        .background(Colors.Gray.g50)
    }
}
#endif
