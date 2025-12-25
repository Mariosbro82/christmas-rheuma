//
//  AboutView.swift
//  InflamAI
//
//  App information, version, and credits
//

import SwiftUI

struct AboutView: View {
    @State private var showCredits = false

    private let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
    private let buildNumber = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"

    var body: some View {
        ScrollView {
            VStack(spacing: AssetsManager.Spacing.xl) {
                // App Icon & Name
                VStack(spacing: AssetsManager.Spacing.md) {
                    // App Icon Placeholder
                    ZStack {
                        RoundedRectangle(cornerRadius: 20)
                            .fill(
                                LinearGradient(
                                    colors: [AssetsManager.Colors.primary, AssetsManager.Colors.secondary],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 100, height: 100)

                        AnkylosaurusMascot(expression: .happy, size: 80)
                    }
                    .shadow(
                        color: AssetsManager.Shadow.medium.color,
                        radius: AssetsManager.Shadow.medium.radius,
                        x: AssetsManager.Shadow.medium.x,
                        y: AssetsManager.Shadow.medium.y
                    )

                    Text("InflamAI")
                        .font(.title)
                        .fontWeight(.bold)

                    Text("Version \(appVersion) (\(buildNumber))")
                        .font(.subheadline)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                }
                .padding(.top, AssetsManager.Spacing.xl)

                // Tagline
                Text("Your Personal AS Companion")
                    .font(.headline)
                    .foregroundColor(AssetsManager.Colors.primary)

                // Description
                Text("InflamAI helps you track and document your ankylosing spondylitis journey with privacy-first tools.")
                    .font(.body)
                    .foregroundColor(AssetsManager.Colors.secondaryText)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, AssetsManager.Spacing.xl)

                // Features
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("What Makes Us Different")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    FeatureRow(
                        icon: "lock.shield.fill",
                        title: "Privacy-First",
                        description: "100% on-device. Zero tracking. Your data never leaves your iPhone.",
                        color: AssetsManager.Colors.success
                    )

                    FeatureRow(
                        icon: "stethoscope",
                        title: "Research-Based Questionnaires",
                        description: "BASDAI, ASAS, and ASDAS scoring based on published research.",
                        color: AssetsManager.Colors.primary
                    )

                    FeatureRow(
                        icon: "brain.head.profile",
                        title: "Pattern Visualization",
                        description: "View correlations and trends from your logged data.",
                        color: AssetsManager.Colors.info
                    )

                    FeatureRow(
                        icon: "heart.fill",
                        title: "Made with Care",
                        description: "Built by someone who understands living with chronic illness.",
                        color: AssetsManager.Colors.error
                    )
                }

                // Credits Button
                Button {
                    showCredits = true
                } label: {
                    HStack {
                        Image(systemName: "person.2.fill")
                        Text("Credits & Acknowledgments")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(AssetsManager.Colors.primary.opacity(0.1))
                    .foregroundColor(AssetsManager.Colors.primary)
                    .cornerRadius(AssetsManager.CornerRadius.md)
                }
                .padding(.horizontal, AssetsManager.Spacing.md)

                // Legal Links
                VStack(spacing: AssetsManager.Spacing.sm) {
                    Button("Privacy Policy") {
                        // Open privacy policy
                        if let url = URL(string: "https://spinalytics.app/privacy") {
                            UIApplication.shared.open(url)
                        }
                    }
                    .font(.subheadline)
                    .foregroundColor(AssetsManager.Colors.primary)

                    Button("Terms of Service") {
                        // Open terms
                        if let url = URL(string: "https://spinalytics.app/terms") {
                            UIApplication.shared.open(url)
                        }
                    }
                    .font(.subheadline)
                    .foregroundColor(AssetsManager.Colors.primary)
                }

                // Copyright
                Text("© 2024 InflamAI. All rights reserved.")
                    .font(.caption)
                    .foregroundColor(AssetsManager.Colors.secondaryText)
                    .padding(.bottom, 40)
            }
        }
        .navigationTitle("About")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showCredits) {
            CreditsView()
        }
    }
}

// MARK: - Feature Row

struct FeatureRow: View {
    let icon: String
    let title: String
    let description: String
    let color: Color

    var body: some View {
        HStack(spacing: AssetsManager.Spacing.md) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 44, height: 44)

                Image(systemName: icon)
                    .foregroundColor(color)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Text(description)
                    .font(.caption)
                    .foregroundColor(AssetsManager.Colors.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
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
        .padding(.horizontal, AssetsManager.Spacing.md)
    }
}

// MARK: - Credits View

struct CreditsView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.lg) {
                    // Development
                    CreditSection(
                        title: "Development",
                        credits: [
                            Credit(role: "Lead Developer", name: "Your Name"),
                            Credit(role: "UI/UX Design", name: "Your Name"),
                            Credit(role: "Mascot Design", name: "Anky the Ankylosaurus")
                        ]
                    )

                    // Medical Advisors
                    CreditSection(
                        title: "Medical Guidance",
                        credits: [
                            Credit(role: "BASDAI Scoring", name: "Bath Institute for Rheumatic Diseases"),
                            Credit(role: "ASAS Criteria", name: "Assessment of SpondyloArthritis international Society"),
                            Credit(role: "Clinical Validation", name: "Published AS Research Literature")
                        ]
                    )

                    // Community
                    CreditSection(
                        title: "Community & Support",
                        credits: [
                            Credit(role: "Patient Advocacy", name: "Spondylitis Association of America"),
                            Credit(role: "Community Feedback", name: "r/ankylosingspondylitis"),
                            Credit(role: "Beta Testers", name: "AS Community Members")
                        ]
                    )

                    // Technology
                    CreditSection(
                        title: "Technology",
                        credits: [
                            Credit(role: "Framework", name: "SwiftUI & iOS SDK"),
                            Credit(role: "Data Storage", name: "Core Data"),
                            Credit(role: "Charts", name: "Swift Charts"),
                            Credit(role: "Health Integration", name: "HealthKit & WeatherKit")
                        ]
                    )

                    // Special Thanks
                    VStack(alignment: .leading, spacing: AssetsManager.Spacing.sm) {
                        Text("Special Thanks")
                            .font(.headline)

                        Text("To everyone living with ankylosing spondylitis - your resilience inspires this app. To the rheumatologists advancing AS research. To the caregivers and loved ones providing support. And to Anky, our mascot, for bringing smiles to difficult days.")
                            .font(.body)
                            .foregroundColor(AssetsManager.Colors.secondaryText)
                    }
                    .padding(AssetsManager.Spacing.md)
                    .background(
                        RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                            .fill(AssetsManager.Colors.primary.opacity(0.1))
                    )
                    .padding(.horizontal, AssetsManager.Spacing.md)

                    Spacer(minLength: 40)
                }
                .padding(.vertical, AssetsManager.Spacing.lg)
            }
            .navigationTitle("Credits")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Credit Section

struct CreditSection: View {
    let title: String
    let credits: [Credit]

    var body: some View {
        VStack(alignment: .leading, spacing: AssetsManager.Spacing.sm) {
            Text(title)
                .font(.headline)
                .padding(.horizontal, AssetsManager.Spacing.md)

            VStack(spacing: 0) {
                ForEach(credits.indices, id: \.self) { index in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(credits[index].role)
                                .font(.caption)
                                .foregroundColor(AssetsManager.Colors.secondaryText)

                            Text(credits[index].name)
                                .font(.subheadline)
                                .fontWeight(.medium)
                        }

                        Spacer()
                    }
                    .padding(AssetsManager.Spacing.md)

                    if index < credits.count - 1 {
                        Divider()
                            .padding(.leading, AssetsManager.Spacing.md)
                    }
                }
            }
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
            .padding(.horizontal, AssetsManager.Spacing.md)
        }
    }
}

struct Credit {
    let role: String
    let name: String
}

// MARK: - Preview

#Preview("About") {
    NavigationView {
        AboutView()
    }
}

#Preview("Credits") {
    CreditsView()
}
