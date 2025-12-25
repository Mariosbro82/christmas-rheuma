//
//  CommunityView.swift
//  InflamAI
//
//  Community support resources and connections
//

import SwiftUI

struct CommunityView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: AssetsManager.Spacing.lg) {
                // Header with Anky
                VStack(spacing: AssetsManager.Spacing.md) {
                    AnkylosaurusMascot(expression: .happy, size: 140)
                        .bouncing()

                    Text("You're Not Alone")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Connect with others managing ankylosing spondylitis")
                        .font(.subheadline)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, AssetsManager.Spacing.xl)

                // Support Groups
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Support Communities")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    CommunityResourceCard(
                        icon: "person.3.fill",
                        title: "SAA - Spondylitis Association",
                        description: "The leading AS patient organization with local support groups and online forums",
                        url: "https://spondylitis.org",
                        color: AssetsManager.Colors.primary
                    )

                    CommunityResourceCard(
                        icon: "bubble.left.and.bubble.right.fill",
                        title: "Reddit r/ankylosingspondylitis",
                        description: "Active community of 20,000+ AS patients sharing experiences",
                        url: "https://reddit.com/r/ankylosingspondylitis",
                        color: Color.orange
                    )

                    CommunityResourceCard(
                        icon: "person.2.fill",
                        title: "Inspire AS Community",
                        description: "Moderated forum with expert medical input",
                        url: "https://inspire.com/groups/spondylitis-association-of-america",
                        color: Color.green
                    )

                    CommunityResourceCard(
                        icon: "message.fill",
                        title: "CreakyJoints Community",
                        description: "Arthritis support network including AS",
                        url: "https://creakyjoints.org",
                        color: Color.purple
                    )
                }

                // Educational Resources
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Educational Resources")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    CommunityResourceCard(
                        icon: "book.fill",
                        title: "Spondylitis Plus Podcast",
                        description: "Weekly podcast covering AS research, treatments, and patient stories",
                        url: "https://spondylitis.org/podcast",
                        color: AssetsManager.Colors.info
                    )

                    CommunityResourceCard(
                        icon: "video.fill",
                        title: "SAA YouTube Channel",
                        description: "Educational videos, webinars, and exercise demos",
                        url: "https://youtube.com/spondylitis",
                        color: Color.red
                    )

                    CommunityResourceCard(
                        icon: "doc.text.fill",
                        title: "AS Patient Guide",
                        description: "Comprehensive guide to diagnosis, treatment, and living with AS",
                        url: "https://spondylitis.org/learn",
                        color: AssetsManager.Colors.secondary
                    )
                }

                // Crisis Support
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Need Immediate Help?")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    MascotTipCard(
                        icon: AssetsManager.Symbols.warning,
                        title: "Crisis Resources",
                        message: "If you're experiencing a mental health crisis, call 988 (US) or your local emergency services. Your mental health matters.",
                        color: AssetsManager.Colors.error
                    )
                    .padding(.horizontal, AssetsManager.Spacing.md)
                }

                // Disclaimer
                VStack(spacing: AssetsManager.Spacing.sm) {
                    Text("Medical Disclaimer")
                        .font(.caption)
                        .fontWeight(.semibold)

                    Text("These resources are for support and education only. Always consult your rheumatologist for medical advice. InflamAI is not affiliated with these organizations.")
                        .font(.caption2)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                        .multilineTextAlignment(.center)
                }
                .padding(AssetsManager.Spacing.md)
                .background(
                    RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                        .fill(AssetsManager.Colors.warning.opacity(0.1))
                )
                .padding(.horizontal, AssetsManager.Spacing.md)

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("Community")
        .navigationBarTitleDisplayMode(.large)
    }
}

// MARK: - Community Resource Card

struct CommunityResourceCard: View {
    let icon: String
    let title: String
    let description: String
    let url: String
    let color: Color

    var body: some View {
        Button {
            if let url = URL(string: url) {
                UIApplication.shared.open(url)
            }
        } label: {
            HStack(spacing: AssetsManager.Spacing.md) {
                // Icon
                ZStack {
                    Circle()
                        .fill(color.opacity(0.15))
                        .frame(width: 50, height: 50)

                    Image(systemName: icon)
                        .font(.title3)
                        .foregroundColor(color)
                }

                // Content
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(AssetsManager.Colors.primaryText)

                    Text(description)
                        .font(.caption)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()

                // External link icon
                Image(systemName: "arrow.up.right.circle.fill")
                    .font(.title3)
                    .foregroundColor(color.opacity(0.7))
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
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Preview

#Preview {
    NavigationView {
        CommunityView()
    }
}
