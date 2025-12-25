//
//  InflamAIDesignSystem.swift
//  InflamAI
//
//  Design System Foundation per UI/UX Audit Report
//  Implements exact specifications from Section 7: Design System Foundation
//
//  IMPORTANT: All values in this file are EXACT specifications from the audit.
//  Do NOT modify without clinical validation and design review.
//

import SwiftUI

// MARK: - Spacing System (Section 7.1)
/// Consistent spacing scale based on 4px base unit
/// Usage Rules:
/// - Container padding: md (16px) minimum
/// - Between related items: xs (8px) or sm (12px)
/// - Between sections: lg (24px) or xl (32px)
/// - Card internal padding: md (16px) or lg (24px)
/// - Page margins: xl (32px) on larger screens
enum Spacing {
    static let xxxs: CGFloat = 2   // Micro: icon padding
    static let xxs: CGFloat = 4    // Tight: inline elements
    static let xs: CGFloat = 8     // Small: related items
    static let sm: CGFloat = 12    // Compact: form fields
    static let md: CGFloat = 16    // Default: card padding
    static let lg: CGFloat = 24    // Large: section gaps
    static let xl: CGFloat = 32    // XL: major sections
    static let xxl: CGFloat = 48   // XXL: page margins
    static let xxxl: CGFloat = 64  // Hero: special layouts
}

// MARK: - Typography System (Section 7.2)
/// Font sizes in points with recommended weights
enum Typography {
    // Font Family
    static let primaryFamily = "SF Pro Display"
    static let secondaryFamily = "SF Pro Text"

    // Font Sizes (in points)
    static let xxs: CGFloat = 9    // Micro labels
    static let xs: CGFloat = 11    // Captions
    static let sm: CGFloat = 13    // Secondary text
    static let base: CGFloat = 15  // Body
    static let md: CGFloat = 17    // Emphasized body
    static let lg: CGFloat = 20    // Subheadings
    static let xl: CGFloat = 24    // Section titles
    static let xxl: CGFloat = 28   // Page titles
    static let xxxl: CGFloat = 34  // Large titles

    // Line Heights
    static let tight: CGFloat = 1.2   // Headlines
    static let normal: CGFloat = 1.5  // Body
    static let relaxed: CGFloat = 1.7 // Long text

    // Typography Hierarchy Helpers
    static var pageTitle: Font {
        .system(size: xxl, weight: .semibold)
    }

    static var sectionHeading: Font {
        .system(size: xl, weight: .semibold)
    }

    static var cardTitle: Font {
        .system(size: lg, weight: .medium)
    }

    static var body: Font {
        .system(size: base, weight: .regular)
    }

    static var secondaryText: Font {
        .system(size: sm, weight: .regular)
    }

    static var caption: Font {
        .system(size: xs, weight: .regular)
    }
}

// MARK: - Color System (Section 7.3)
/// Semantic color system with exact hex values from audit
enum Colors {
    // Primary Brand Colors
    enum Primary {
        static let p50 = Color(hex: "#EBF5FF")
        static let p100 = Color(hex: "#E1EFFE")
        static let p200 = Color(hex: "#C3DDFD")
        static let p300 = Color(hex: "#A4CAFE")
        static let p400 = Color(hex: "#76A9FA")
        static let p500 = Color(hex: "#3B82F6")  // Main Primary
        static let p600 = Color(hex: "#1D4ED8")  // Hover
        static let p700 = Color(hex: "#1E40AF")  // Active
    }

    // Gray Scale
    enum Gray {
        static let g50 = Color(hex: "#F9FAFB")   // Background
        static let g100 = Color(hex: "#F3F4F6")  // Surface
        static let g200 = Color(hex: "#E5E7EB")  // Border
        static let g300 = Color(hex: "#D1D5DB")  // Disabled
        static let g400 = Color(hex: "#9CA3AF")  // Placeholder
        static let g500 = Color(hex: "#6B7280")  // Secondary text
        static let g600 = Color(hex: "#4B5563")  // Primary text
        static let g700 = Color(hex: "#374151")
        static let g800 = Color(hex: "#1F2937")
        static let g900 = Color(hex: "#111827")  // Headlines
    }

    // Semantic Colors
    enum Semantic {
        static let success = Color(hex: "#10B981")
        static let successLight = Color(hex: "#D1FAE5")
        static let warning = Color(hex: "#F59E0B")
        static let warningLight = Color(hex: "#FEF3C7")
        static let error = Color(hex: "#EF4444")
        static let errorLight = Color(hex: "#FEE2E2")
        static let info = Color(hex: "#3B82F6")
        static let infoLight = Color(hex: "#DBEAFE")
    }

    // Accent Colors (for AI/special features)
    enum Accent {
        static let purple = Color(hex: "#8B5CF6")
        static let purpleLight = Color(hex: "#EDE9FE")
        static let teal = Color(hex: "#14B8A6")     // Teal accent
        static let tealLight = Color(hex: "#CCFBF1") // Light teal
    }
}

// MARK: - Shadow System (Section 7.4)
/// Shadow usage:
/// - Cards at rest: sm
/// - Cards on hover/press: md
/// - Modals/dialogs: xl
/// - Dropdowns/popovers: md
/// - Floating action buttons: lg
enum Shadows {
    struct ShadowConfig {
        let color: Color
        let radius: CGFloat
        let x: CGFloat
        let y: CGFloat
    }

    static let xs = ShadowConfig(
        color: Color.black.opacity(0.05),
        radius: 2,
        x: 0,
        y: 1
    )

    static let sm = ShadowConfig(
        color: Color.black.opacity(0.08),
        radius: 8,
        x: 0,
        y: 2
    )

    static let md = ShadowConfig(
        color: Color.black.opacity(0.1),
        radius: 16,
        x: 0,
        y: 4
    )

    static let lg = ShadowConfig(
        color: Color.black.opacity(0.12),
        radius: 24,
        x: 0,
        y: 8
    )

    static let xl = ShadowConfig(
        color: Color.black.opacity(0.15),
        radius: 50,
        x: 0,
        y: 25
    )
}

// MARK: - Border Radius System (Section 7.5)
enum Radii {
    static let none: CGFloat = 0
    static let xs: CGFloat = 2     // Micro radius
    static let sm: CGFloat = 4     // Tags, badges
    static let md: CGFloat = 8     // Buttons, inputs
    static let lg: CGFloat = 12    // Cards
    static let xl: CGFloat = 16    // Large cards, modals
    static let xxl: CGFloat = 24   // Feature cards
    static let full: CGFloat = 9999 // Pills, avatars
}

// MARK: - Animation System (Section 10.1)
enum Animations {
    static let fast: Double = 0.15    // Hover states
    static let normal: Double = 0.2   // Most interactions
    static let slow: Double = 0.3     // Page transitions

    static var spring: Animation {
        .spring(response: 0.35, dampingFraction: 0.7)
    }

    static var easeOut: Animation {
        .easeOut(duration: normal)
    }
}

// MARK: - Color Extension for Hex Support
extension Color {
    /// Initialize Color from hex string
    /// Supports 3, 6, or 8 character hex codes (with/without #)
    /// Examples: "#FFF", "#FFFFFF", "#FFFFFFFF"
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3:
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

// MARK: - View Modifiers for Design System
extension View {
    /// Apply shadow from design system
    func dshadow(_ config: Shadows.ShadowConfig) -> some View {
        self.shadow(color: config.color, radius: config.radius, x: config.x, y: config.y)
    }

    /// Apply card styling per audit Section 8.3
    /// NOTE: Using dsCardStyle to avoid conflict with AssetsManager.cardStyle()
    func dsCardStyle() -> some View {
        self
            .padding(Spacing.md)
            .background(Color.white)
            .cornerRadius(Radii.lg)
            .dshadow(Shadows.sm)
    }

    /// Apply input field styling per audit Section 8.4
    func dsInputFieldStyle(isFocused: Bool = false, hasError: Bool = false) -> some View {
        self
            .padding(.horizontal, Spacing.md)
            .frame(height: 48)
            .background(Color.white)
            .cornerRadius(Radii.md)
            .overlay(
                RoundedRectangle(cornerRadius: Radii.md)
                    .stroke(
                        hasError ? Colors.Semantic.error :
                        isFocused ? Colors.Primary.p500 : Colors.Gray.g200,
                        lineWidth: isFocused ? 2 : 1
                    )
            )
    }

    /// Apply shimmer loading effect
    func shimmer() -> some View {
        self.modifier(ShimmerModifier())
    }
}

// MARK: - Shimmer Modifier
struct ShimmerModifier: ViewModifier {
    @State private var isAnimating = false

    func body(content: Content) -> some View {
        content
            .overlay(
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color.white.opacity(0),
                        Color.white.opacity(0.4),
                        Color.white.opacity(0)
                    ]),
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .offset(x: isAnimating ? 400 : -400)
                .animation(
                    Animation.linear(duration: 1.5)
                        .repeatForever(autoreverses: false),
                    value: isAnimating
                )
            )
            .onAppear {
                isAnimating = true
            }
    }
}

// MARK: - String Extensions
// NOTE: String extensions (displayName, localizedWithFallback) are defined in
// StringExtensions.swift for project-wide use
