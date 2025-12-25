//
//  MedicalDisclaimerView.swift
//  InflamAI
//
//  Reusable medical disclaimer component for all prediction/analysis displays
//  CRITICAL: This MUST be displayed on ALL screens that show health predictions,
//  pattern analysis, or AI-generated insights.
//
//  Legal requirement for medical liability protection.
//

import SwiftUI

/// Medical disclaimer banner for prediction and analysis screens
///
/// ## Usage
/// Add this view ABOVE any prediction, risk score, or pattern analysis:
/// ```swift
/// var body: some View {
///     VStack {
///         MedicalDisclaimerView()  // Always first
///         FlareRiskCard()
///         PatternAnalysisSection()
///     }
/// }
/// ```
///
/// ## Accessibility
/// - VoiceOver reads the full disclaimer
/// - High contrast colors for visibility
/// - Semantic grouping for screen readers
struct MedicalDisclaimerView: View {
    // MARK: - Configuration

    enum Style {
        case banner      // Full-width banner (default)
        case compact     // Single-line for space-constrained areas
        case footer      // Smaller text for bottom of screens
        case prominent   // Large, attention-grabbing for critical displays
    }

    let style: Style

    // MARK: - Initialization

    init(style: Style = .banner) {
        self.style = style
    }

    // MARK: - Body

    var body: some View {
        Group {
            switch style {
            case .banner:
                bannerView
            case .compact:
                compactView
            case .footer:
                footerView
            case .prominent:
                prominentView
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Medical Disclaimer")
        .accessibilityValue("This is not medical advice. Statistical pattern analysis only. Always consult your rheumatologist for medical decisions.")
        .accessibilityHint("Important health information disclaimer")
    }

    // MARK: - Banner Style (Default)

    private var bannerView: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.title3)
                .foregroundColor(.orange)

            VStack(alignment: .leading, spacing: 4) {
                Text("NOT MEDICAL ADVICE")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.orange)

                Text("Statistical pattern analysis only. Always consult your rheumatologist for medical decisions.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.orange.opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
    }

    // MARK: - Compact Style

    private var compactView: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundColor(.orange)

            Text("Not medical advice - consult your doctor")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.orange.opacity(0.05))
        .cornerRadius(4)
    }

    // MARK: - Footer Style

    private var footerView: some View {
        VStack(spacing: 4) {
            Divider()
                .padding(.bottom, 8)

            HStack(spacing: 4) {
                Image(systemName: "info.circle")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Text("Wellness tracking only. Not intended for diagnosis or treatment.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }

            Text("Consult your rheumatologist for medical decisions.")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    // MARK: - Prominent Style

    private var prominentView: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title2)
                    .foregroundColor(.orange)

                Text("IMPORTANT HEALTH DISCLAIMER")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(.orange)

                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title2)
                    .foregroundColor(.orange)
            }

            VStack(alignment: .leading, spacing: 8) {
                DisclaimerBullet(
                    icon: "chart.bar.xaxis",
                    text: "This app uses statistical pattern analysis, NOT medical diagnosis"
                )

                DisclaimerBullet(
                    icon: "person.crop.circle.badge.questionmark",
                    text: "Predictions are based on YOUR personal data patterns only"
                )

                DisclaimerBullet(
                    icon: "stethoscope",
                    text: "Always consult your rheumatologist before changing treatment"
                )

                DisclaimerBullet(
                    icon: "exclamationmark.shield",
                    text: "This is NOT a medical device and should not replace professional care"
                )
            }
            .padding(.horizontal, 8)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.orange.opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.orange, lineWidth: 2)
        )
    }
}

// MARK: - Supporting Views

private struct DisclaimerBullet: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.subheadline)
                .foregroundColor(.orange)
                .frame(width: 20)

            Text(text)
                .font(.subheadline)
                .foregroundColor(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Preview

#if DEBUG
struct MedicalDisclaimerView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 24) {
            Text("Banner Style").font(.headline)
            MedicalDisclaimerView(style: .banner)

            Text("Compact Style").font(.headline)
            MedicalDisclaimerView(style: .compact)

            Text("Prominent Style").font(.headline)
            MedicalDisclaimerView(style: .prominent)

            Text("Footer Style").font(.headline)
            MedicalDisclaimerView(style: .footer)
        }
        .padding()
        .previewLayout(.sizeThatFits)
    }
}
#endif

// MARK: - View Modifier for Easy Integration

extension View {
    /// Adds a medical disclaimer banner above the view
    /// - Parameter style: The disclaimer style to use
    func withMedicalDisclaimer(_ style: MedicalDisclaimerView.Style = .banner) -> some View {
        VStack(spacing: 16) {
            MedicalDisclaimerView(style: style)
            self
        }
    }

    /// Adds a medical disclaimer footer below the view
    func withMedicalFooter() -> some View {
        VStack {
            self
            Spacer()
            MedicalDisclaimerView(style: .footer)
        }
    }
}
