//
//  AssetsManager.swift
//  InflamAI
//
//  Centralized assets management for colors, images, and resources
//  Type-safe access to all app assets with compile-time checking
//

import SwiftUI
import UIKit

/// Centralized manager for all app assets (colors, images, icons)
enum AssetsManager {

    // MARK: - Colors

    /// App color palette
    /// Uses asset catalog colors with system fallbacks via helper function
    enum Colors {

        // MARK: - Private Helper

        /// Safely load a color from asset catalog with fallback
        /// Color(name:bundle:) returns non-optional but may be clear if not found
        private static func color(_ name: String, fallback: Color) -> Color {
            if let uiColor = UIColor(named: name) {
                return Color(uiColor)
            }
            return fallback
        }

        // MARK: Brand Colors

        static let primary = color("Primary", fallback: .blue)
        static let secondary = color("Secondary", fallback: .purple)
        static let accent = color("Accent", fallback: .green)

        // MARK: UI Colors

        static let background = color("Background", fallback: Color(.systemBackground))
        static let secondaryBackground = color("SecondaryBackground", fallback: Color(.secondarySystemBackground))
        static let cardBackground = color("CardBackground", fallback: Color(.systemBackground))

        // MARK: Text Colors

        static let primaryText = color("PrimaryText", fallback: Color(.label))
        static let secondaryText = color("SecondaryText", fallback: Color(.secondaryLabel))
        static let tertiaryText = color("TertiaryText", fallback: Color(.tertiaryLabel))

        // MARK: Semantic Colors

        /// Success/Low risk/Improving
        static let success = color("Success", fallback: .green)

        /// Warning/Moderate risk
        static let warning = color("Warning", fallback: .orange)

        /// Error/High risk/Critical
        static let error = color("Error", fallback: .red)

        /// Info/Neutral
        static let info = color("Info", fallback: .blue)

        // MARK: Pain Levels

        /// Pain level 0-2 (green)
        static let painLow = color("PainLow", fallback: .green)

        /// Pain level 3-4 (yellow)
        static let painMild = color("PainMild", fallback: .yellow)

        /// Pain level 5-6 (orange)
        static let painModerate = color("PainModerate", fallback: .orange)

        /// Pain level 7-8 (red-orange)
        static let painSevere = color("PainSevere", fallback: Color(red: 1.0, green: 0.3, blue: 0.0))

        /// Pain level 9-10 (dark red)
        static let painCritical = color("PainCritical", fallback: Color(red: 0.8, green: 0.0, blue: 0.0))

        // MARK: BASDAI Colors

        /// BASDAI 0-2 (remission)
        static let basdaiLow = color("BASDAILow", fallback: .green)

        /// BASDAI 2-4 (mild)
        static let basdaiMild = color("BASDAIMild", fallback: .yellow)

        /// BASDAI 4-6 (moderate)
        static let basdaiModerate = color("BASDAIModerate", fallback: .orange)

        /// BASDAI 6+ (high)
        static let basdaiHigh = color("BASDAIHigh", fallback: .red)

        // MARK: Feature Colors

        static let medication = color("Medication", fallback: .purple)
        static let exercise = color("Exercise", fallback: .green)
        static let flare = color("Flare", fallback: .red)
        static let trends = color("Trends", fallback: .blue)
        static let ai = color("AI", fallback: .purple)

        // MARK: Chart Colors

        static let chartPrimary = color("ChartPrimary", fallback: .blue)
        static let chartSecondary = color("ChartSecondary", fallback: .orange)
        static let chartTertiary = color("ChartTertiary", fallback: .green)
        static let chartGrid = color("ChartGrid", fallback: Color.gray.opacity(0.2))

        // MARK: Helper Functions

        /// Get color for pain level (0-10)
        static func forPainLevel(_ level: Double) -> Color {
            switch level {
            case 0..<3: return painLow
            case 3..<5: return painMild
            case 5..<7: return painModerate
            case 7..<9: return painSevere
            default: return painCritical
            }
        }

        /// Get color for BASDAI score (0-10)
        static func forBASDAI(_ score: Double) -> Color {
            switch score {
            case 0..<2: return basdaiLow
            case 2..<4: return basdaiMild
            case 4..<6: return basdaiModerate
            default: return basdaiHigh
            }
        }

        /// Get color for trend direction
        static func forTrend(_ isImproving: Bool) -> Color {
            isImproving ? success : error
        }
    }

    // MARK: - Images

    /// App images and icons
    enum Images {

        // MARK: App Branding

        static let appIcon = Image("AppIcon", bundle: nil)
        static let logo = Image("Logo", bundle: nil)
        static let logoText = Image("LogoText", bundle: nil)

        // MARK: Illustrations

        static let onboardingWelcome = Image("OnboardingWelcome", bundle: nil)
        static let onboardingFeatures = Image("OnboardingFeatures", bundle: nil)
        static let emptyState = Image("EmptyState", bundle: nil)
        static let successAnimation = Image("SuccessAnimation", bundle: nil)

        // MARK: Body Diagrams

        static let bodyFront = Image("BodyFront", bundle: nil)
        static let bodyBack = Image("BodyBack", bundle: nil)
        static let spineDiagram = Image("SpineDiagram", bundle: nil)

        // MARK: Exercise Thumbnails

        static func exerciseThumbnail(_ name: String) -> Image {
            Image("Exercise-\(name)", bundle: nil)
        }

        // MARK: Placeholder Images

        static let placeholder = Image(systemName: "photo")
        static let userPlaceholder = Image(systemName: "person.circle.fill")
    }

    // MARK: - SF Symbols

    /// Commonly used SF Symbols organized by category
    enum Symbols {

        // MARK: Navigation

        static let home = "house.fill"
        static let settings = "gearshape.fill"
        static let profile = "person.circle.fill"
        static let back = "chevron.left"
        static let forward = "chevron.right"
        static let close = "xmark"

        // MARK: Health & Medical

        static let pain = "bolt.fill"
        static let medication = "pills.fill"
        static let flare = "flame.fill"
        static let heart = "heart.fill"
        static let healthData = "heart.text.square.fill"
        static let stethoscope = "stethoscope"
        static let bandage = "bandage.fill"
        static let thermometer = "thermometer"

        // MARK: Body & Exercise

        static let body = "figure.stand"
        static let bodyFront = "figure.arms.open"
        static let spine = "figure.stand.line.dotted.figure.stand"
        static let exercise = "figure.flexibility"
        static let stretching = "figure.flexibility"
        static let strengthening = "figure.strengthtraining.traditional"
        static let walking = "figure.walk"
        static let breathing = "wind"

        // MARK: Tracking & Analytics

        static let chart = "chart.xyaxis.line"
        static let chartBar = "chart.bar.fill"
        static let trend = "chart.line.uptrend.xyaxis"
        static let calendar = "calendar"
        static let clock = "clock.fill"
        static let timer = "timer"

        // MARK: AI & Intelligence

        static let ai = "brain.head.profile"
        static let sparkles = "sparkles"
        static let wand = "wand.and.stars"
        static let lightbulb = "lightbulb.fill"

        // MARK: Actions

        static let add = "plus.circle.fill"
        static let edit = "pencil.circle.fill"
        static let delete = "trash.fill"
        static let save = "checkmark.circle.fill"
        static let share = "square.and.arrow.up"
        static let export = "arrow.down.doc.fill"
        static let search = "magnifyingglass"

        // MARK: Status

        static let success = "checkmark.circle.fill"
        static let warning = "exclamationmark.triangle.fill"
        static let error = "xmark.circle.fill"
        static let info = "info.circle.fill"

        // MARK: Notifications

        static let bell = "bell.fill"
        static let bellBadge = "bell.badge.fill"
        static let bellSlash = "bell.slash.fill"

        // MARK: Weather

        static let sun = "sun.max.fill"
        static let cloud = "cloud.fill"
        static let rain = "cloud.rain.fill"
        static let snow = "cloud.snow.fill"
        static let thermometerHigh = "thermometer.sun.fill"
        static let thermometerLow = "thermometer.snowflake"
        static let humidity = "humidity.fill"
        static let wind = "wind"

        // MARK: Data

        static let download = "arrow.down.circle.fill"
        static let upload = "arrow.up.circle.fill"
        static let sync = "arrow.triangle.2.circlepath"
        static let refresh = "arrow.clockwise"

        // MARK: Security

        static let lock = "lock.fill"
        static let unlock = "lock.open.fill"
        static let faceID = "faceid"
        static let touchID = "touchid"
        static let shield = "shield.fill"

        // MARK: Helper Function

        /// Get SF Symbol Image
        static func image(_ name: String, size: CGFloat = 20, weight: Font.Weight = .regular) -> Image {
            Image(systemName: name)
        }
    }

    // MARK: - Gradients

    enum Gradients {

        /// Primary brand gradient (blue to purple)
        static let primary = LinearGradient(
            colors: [Colors.primary, Colors.secondary],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )

        /// Success gradient (green shades)
        static let success = LinearGradient(
            colors: [Color.green, Color.mint],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )

        /// Warning gradient (orange to yellow)
        static let warning = LinearGradient(
            colors: [Color.orange, Color.yellow],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )

        /// Error gradient (red shades)
        static let error = LinearGradient(
            colors: [Color.red, Color.orange],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )

        /// BASDAI gradient (green to red spectrum)
        static let basdai = LinearGradient(
            colors: [Colors.basdaiLow, Colors.basdaiMild, Colors.basdaiModerate, Colors.basdaiHigh],
            startPoint: .leading,
            endPoint: .trailing
        )

        /// Pain gradient (green to dark red)
        static let pain = LinearGradient(
            colors: [Colors.painLow, Colors.painMild, Colors.painModerate, Colors.painSevere, Colors.painCritical],
            startPoint: .leading,
            endPoint: .trailing
        )

        /// Background gradient (subtle)
        static let background = LinearGradient(
            colors: [Color.blue.opacity(0.1), Color.purple.opacity(0.1)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    // MARK: - Spacing

    enum Spacing {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 16
        static let lg: CGFloat = 24
        static let xl: CGFloat = 32
        static let xxl: CGFloat = 48
    }

    // MARK: - Corner Radius

    enum CornerRadius {
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 20
        static let full: CGFloat = 999
    }

    // MARK: - Shadow

    enum Shadow {
        static let small = (color: Color.black.opacity(0.05), radius: CGFloat(3), x: CGFloat(0), y: CGFloat(1))
        static let medium = (color: Color.black.opacity(0.1), radius: CGFloat(5), x: CGFloat(0), y: CGFloat(2))
        static let large = (color: Color.black.opacity(0.15), radius: CGFloat(10), x: CGFloat(0), y: CGFloat(5))
    }

    // MARK: - Animations

    enum Animation {
        static let quick = SwiftUI.Animation.easeInOut(duration: 0.2)
        static let standard = SwiftUI.Animation.easeInOut(duration: 0.3)
        static let slow = SwiftUI.Animation.easeInOut(duration: 0.5)
        static let spring = SwiftUI.Animation.spring(response: 0.4, dampingFraction: 0.7)
    }
}

// MARK: - View Extensions

extension View {

    /// Apply standard card styling
    func cardStyle() -> some View {
        self
            .padding()
            .background(AssetsManager.Colors.cardBackground)
            .cornerRadius(AssetsManager.CornerRadius.md)
            .shadow(
                color: AssetsManager.Shadow.medium.color,
                radius: AssetsManager.Shadow.medium.radius,
                x: AssetsManager.Shadow.medium.x,
                y: AssetsManager.Shadow.medium.y
            )
    }

    /// Apply primary button styling
    func primaryButtonStyle() -> some View {
        self
            .font(.headline)
            .foregroundColor(.white)
            .padding(.horizontal, AssetsManager.Spacing.lg)
            .padding(.vertical, AssetsManager.Spacing.md)
            .background(AssetsManager.Gradients.primary)
            .cornerRadius(AssetsManager.CornerRadius.md)
    }

    /// Apply secondary button styling
    func secondaryButtonStyle() -> some View {
        self
            .font(.subheadline)
            .foregroundColor(AssetsManager.Colors.primary)
            .padding(.horizontal, AssetsManager.Spacing.md)
            .padding(.vertical, AssetsManager.Spacing.sm)
            .background(AssetsManager.Colors.primary.opacity(0.1))
            .cornerRadius(AssetsManager.CornerRadius.sm)
    }
}

// Note: Color(hex:) extension already exists in PainLocationSelector.swift
