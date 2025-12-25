//
//  MascotHelper.swift
//  InflamAI
//
//  Official mascot helper with contextual tips and encouragement
//  Integrates Anky the Ankylosaurus throughout the app
//

import SwiftUI

/// Contextual mascot helper that provides tips, encouragement, and guidance
struct MascotHelper: View {
    enum Context {
        case welcome
        case firstPainLog
        case flareDetected
        case improvement
        case medicationReminder
        case exerciseTime
        case dataInsight
        case celebration
        case sympathy
        case encouragement

        var message: String {
            switch self {
            case .welcome:
                return "Hi! I'm Anky, your AS companion. I'm here to help you track symptoms and understand patterns!"
            case .firstPainLog:
                return "Great start! Consistent tracking helps us spot patterns that matter to you."
            case .flareDetected:
                return "I noticed increased symptoms. Remember to rest, take meds on schedule, and contact your doctor if needed."
            case .improvement:
                return "Your symptoms are trending down! Keep up with your treatment plan."
            case .medicationReminder:
                return "Time for your medication. Consistency is key to managing AS!"
            case .exerciseTime:
                return "Gentle movement can help with stiffness. Ready for some exercises?"
            case .dataInsight:
                return "I've been analyzing your patterns. Want to see what I found?"
            case .celebration:
                return "Amazing progress! You're doing great managing your AS!"
            case .sympathy:
                return "I know AS can be tough. Take it one day at a time. You've got this!"
            case .encouragement:
                return "Every log helps build a clearer picture. Keep going!"
            }
        }

        var expression: AnkylosaurusMascot.Expression {
            switch self {
            case .welcome, .firstPainLog, .encouragement:
                return .waving
            case .flareDetected, .sympathy:
                return .encouraging
            case .improvement, .celebration:
                return .excited
            case .medicationReminder, .exerciseTime, .dataInsight:
                return .happy
            }
        }
    }

    let context: Context
    var size: CGFloat = 120
    var showDismissButton: Bool = true
    var onDismiss: (() -> Void)?

    var body: some View {
        VStack(spacing: AssetsManager.Spacing.md) {
            // Mascot
            AnkylosaurusMascot(
                expression: context.expression,
                size: size
            )
            .bouncing()

            // Speech bubble
            VStack(alignment: .leading, spacing: AssetsManager.Spacing.sm) {
                Text(context.message)
                    .font(.subheadline)
                    .foregroundColor(AssetsManager.Colors.primaryText)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(AssetsManager.Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                    .fill(AssetsManager.Colors.cardBackground)
                    .shadow(
                        color: AssetsManager.Shadow.small.color,
                        radius: AssetsManager.Shadow.small.radius,
                        x: AssetsManager.Shadow.small.x,
                        y: AssetsManager.Shadow.small.y
                    )
            )
            .overlay(
                // Speech bubble pointer
                Triangle()
                    .fill(AssetsManager.Colors.cardBackground)
                    .frame(width: 20, height: 10)
                    .rotationEffect(.degrees(180))
                    .offset(y: -10),
                alignment: .top
            )

            // Dismiss button
            if showDismissButton {
                Button {
                    onDismiss?()
                } label: {
                    Text("Got it!")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(AssetsManager.Colors.primary)
                }
            }
        }
        .padding(AssetsManager.Spacing.lg)
    }
}

// MARK: - Triangle Shape for Speech Bubble

private struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}

// MARK: - Mascot Tip Card

/// Compact mascot tip card for inline use
struct MascotTipCard: View {
    let icon: String
    let title: String
    let message: String
    var color: Color = AssetsManager.Colors.primary

    var body: some View {
        HStack(spacing: AssetsManager.Spacing.md) {
            // Icon
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 50, height: 50)

                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(color)
            }

            // Content
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(AssetsManager.Colors.primaryText)

                Text(message)
                    .font(.caption)
                    .foregroundColor(AssetsManager.Colors.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding(AssetsManager.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                .fill(color.opacity(0.05))
        )
    }
}

// MARK: - Mascot Banner

/// Top banner with mascot for important messages
struct MascotBanner: View {
    enum BannerType {
        case info
        case success
        case warning
        case tip

        var color: Color {
            switch self {
            case .info: return AssetsManager.Colors.info
            case .success: return AssetsManager.Colors.success
            case .warning: return AssetsManager.Colors.warning
            case .tip: return AssetsManager.Colors.primary
            }
        }

        var icon: String {
            switch self {
            case .info: return AssetsManager.Symbols.info
            case .success: return AssetsManager.Symbols.success
            case .warning: return AssetsManager.Symbols.warning
            case .tip: return AssetsManager.Symbols.lightbulb
            }
        }
    }

    let type: BannerType
    let message: String
    var onDismiss: (() -> Void)?

    var body: some View {
        HStack(spacing: AssetsManager.Spacing.sm) {
            // Icon
            Image(systemName: type.icon)
                .font(.title3)
                .foregroundColor(type.color)

            // Message
            Text(message)
                .font(.subheadline)
                .foregroundColor(AssetsManager.Colors.primaryText)
                .fixedSize(horizontal: false, vertical: true)

            Spacer()

            // Dismiss
            if let onDismiss = onDismiss {
                Button {
                    onDismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                }
            }
        }
        .padding(AssetsManager.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                .fill(type.color.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                        .strokeBorder(type.color.opacity(0.3), lineWidth: 1)
                )
        )
        .padding(.horizontal, AssetsManager.Spacing.md)
    }
}

// MARK: - Empty State with Mascot

/// Empty state view with mascot encouragement
struct MascotEmptyState: View {
    let icon: String
    let title: String
    let message: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: AssetsManager.Spacing.lg) {
            // Mascot
            AnkylosaurusMascot(
                expression: .encouraging,
                size: 160
            )
            .bouncing()

            // Icon
            Image(systemName: icon)
                .font(.system(size: 50))
                .foregroundColor(AssetsManager.Colors.secondaryText)

            // Text
            VStack(spacing: AssetsManager.Spacing.sm) {
                Text(title)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(AssetsManager.Colors.primaryText)

                Text(message)
                    .font(.subheadline)
                    .foregroundColor(AssetsManager.Colors.secondaryText)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, AssetsManager.Spacing.xl)

            // Action button
            if let actionTitle = actionTitle, let action = action {
                Button {
                    action()
                } label: {
                    Text(actionTitle)
                        .primaryButtonStyle()
                }
                .padding(.top, AssetsManager.Spacing.sm)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(AssetsManager.Spacing.xl)
    }
}

// MARK: - Mascot Celebration

/// Celebration view with animated mascot
struct MascotCelebration: View {
    let title: String
    let message: String
    var onDismiss: (() -> Void)?

    var body: some View {
        ZStack {
            // Backdrop
            Color.black.opacity(0.4)
                .ignoresSafeArea()
                .onTapGesture {
                    onDismiss?()
                }

            // Celebration card
            VStack(spacing: AssetsManager.Spacing.lg) {
                // Confetti emoji header
                Text("🎉 🎊 ✨")
                    .font(.largeTitle)

                // Animated mascot
                AnkylosaurusMascot(
                    expression: .excited,
                    size: 180
                )
                .bouncing()

                // Text
                VStack(spacing: AssetsManager.Spacing.sm) {
                    Text(title)
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(AssetsManager.Colors.primaryText)

                    Text(message)
                        .font(.body)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal, AssetsManager.Spacing.lg)

                // Dismiss button
                if let onDismiss = onDismiss {
                    Button {
                        onDismiss()
                    } label: {
                        Text("Awesome!")
                            .primaryButtonStyle()
                    }
                }
            }
            .padding(AssetsManager.Spacing.xl)
            .background(
                RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.lg)
                    .fill(AssetsManager.Colors.cardBackground)
                    .shadow(
                        color: AssetsManager.Shadow.large.color,
                        radius: AssetsManager.Shadow.large.radius,
                        x: AssetsManager.Shadow.large.x,
                        y: AssetsManager.Shadow.large.y
                    )
            )
            .padding(AssetsManager.Spacing.xl)
        }
    }
}

// MARK: - Preview

#Preview("Mascot Helper - Welcome") {
    MascotHelper(context: .welcome)
}

#Preview("Mascot Helper - Flare") {
    MascotHelper(context: .flareDetected)
}

#Preview("Mascot Tip Card") {
    VStack(spacing: 16) {
        MascotTipCard(
            icon: AssetsManager.Symbols.lightbulb,
            title: "Daily Tracking Tip",
            message: "Log symptoms at the same time each day for more accurate patterns!"
        )

        MascotTipCard(
            icon: AssetsManager.Symbols.medication,
            title: "Medication Reminder",
            message: "Taking biologics on schedule helps maintain steady inflammation control.",
            color: AssetsManager.Colors.medication
        )
    }
    .padding()
}

#Preview("Mascot Banner") {
    VStack(spacing: 16) {
        MascotBanner(
            type: .success,
            message: "7-day streak! Keep logging symptoms daily.",
            onDismiss: {}
        )

        MascotBanner(
            type: .warning,
            message: "Your BASDAI score is high today. Consider rest and contact your doctor.",
            onDismiss: {}
        )

        MascotBanner(
            type: .tip,
            message: "Morning stiffness lasting over 30 minutes? This is important data for your rheumatologist!",
            onDismiss: {}
        )
    }
}

#Preview("Mascot Empty State") {
    MascotEmptyState(
        icon: AssetsManager.Symbols.chart,
        title: "No Data Yet",
        message: "Start logging your symptoms to see trends and insights!",
        actionTitle: "Log Symptoms",
        action: {}
    )
}

#Preview("Mascot Celebration") {
    MascotCelebration(
        title: "First Week Complete!",
        message: "You've logged symptoms for 7 days straight. This data will help identify patterns!",
        onDismiss: {}
    )
}
