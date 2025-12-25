//
//  HelpView.swift
//  InflamAI
//
//  Searchable help and FAQ
//

import SwiftUI

struct HelpView: View {
    @State private var searchText = ""
    @State private var expandedQuestions: Set<UUID> = []

    var filteredFAQs: [FAQSection] {
        if searchText.isEmpty {
            return faqSections
        }

        return faqSections.compactMap { section in
            let filteredQuestions = section.questions.filter { question in
                question.question.localizedCaseInsensitiveContains(searchText) ||
                question.answer.localizedCaseInsensitiveContains(searchText)
            }

            return filteredQuestions.isEmpty ? nil : FAQSection(
                title: section.title,
                icon: section.icon,
                questions: filteredQuestions
            )
        }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: AssetsManager.Spacing.lg) {
                // Header
                VStack(spacing: AssetsManager.Spacing.md) {
                    AnkylosaurusMascot(expression: .happy, size: 120)
                        .bouncing()

                    Text("How Can I Help?")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Find answers to common questions")
                        .font(.subheadline)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                }
                .padding(.top, AssetsManager.Spacing.md)

                // Search Bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.gray)

                    TextField("Search help topics...", text: $searchText)
                        .textFieldStyle(PlainTextFieldStyle())
                }
                .padding(AssetsManager.Spacing.md)
                .background(
                    RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                        .fill(Color(.systemGray6))
                )
                .padding(.horizontal, AssetsManager.Spacing.md)

                // FAQ Sections
                if filteredFAQs.isEmpty {
                    MascotEmptyState(
                        icon: "magnifyingglass",
                        title: "No Results",
                        message: "Try a different search term",
                        actionTitle: nil,
                        action: nil
                    )
                    .padding(.top, 40)
                } else {
                    ForEach(filteredFAQs) { section in
                        FAQSectionView(
                            section: section,
                            expandedQuestions: $expandedQuestions
                        )
                    }
                }

                // Contact Support
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Still Need Help?")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    Button {
                        // Email support
                        if let url = URL(string: "mailto:support@spinalytics.app?subject=Help%20Request") {
                            UIApplication.shared.open(url)
                        }
                    } label: {
                        HStack {
                            Image(systemName: "envelope.fill")
                                .font(.title3)
                                .foregroundColor(AssetsManager.Colors.primary)

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Email Support")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)

                                Text("support@spinalytics.app")
                                    .font(.caption)
                                    .foregroundColor(AssetsManager.Colors.secondaryText)
                            }

                            Spacer()

                            Image(systemName: "arrow.right.circle.fill")
                                .foregroundColor(AssetsManager.Colors.primary.opacity(0.7))
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
                .padding(.top, AssetsManager.Spacing.lg)

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("Help & FAQ")
        .navigationBarTitleDisplayMode(.large)
    }

    // MARK: - FAQ Data

    private let faqSections: [FAQSection] = [
        FAQSection(
            title: "Getting Started",
            icon: "star.fill",
            questions: [
                FAQItem(
                    question: "How do I log my first symptoms?",
                    answer: "Tap the Home tab, then use the Quick Symptom Log card. Answer 3 simple questions about pain, stiffness, and fatigue. Your BASDAI score is calculated automatically!"
                ),
                FAQItem(
                    question: "What is BASDAI?",
                    answer: "BASDAI (Bath Ankylosing Spondylitis Disease Activity Index) is a validated 0-10 scale measuring AS disease activity. Scores under 4 indicate controlled disease, while scores above 4 suggest active disease requiring treatment adjustment."
                ),
                FAQItem(
                    question: "Why should I track daily?",
                    answer: "Consistent daily tracking reveals patterns your doctor needs to see - like morning stiffness duration, pain trends, and medication effectiveness. This data leads to better treatment decisions."
                )
            ]
        ),

        FAQSection(
            title: "Pain Map",
            icon: "figure.stand",
            questions: [
                FAQItem(
                    question: "How do I use the pain map?",
                    answer: "Tap the Pain Map tab and select the body locations where you feel pain. Tap the View button to toggle between front and back body views. Selected locations turn blue."
                ),
                FAQItem(
                    question: "Can I track pain intensity per location?",
                    answer: "Yes! After selecting a location, you can adjust the pain intensity for each specific body region using the slider. This helps identify which areas are most affected."
                )
            ]
        ),

        FAQSection(
            title: "AI Insights",
            icon: "brain.head.profile",
            questions: [
                FAQItem(
                    question: "How does the flare predictor work?",
                    answer: "Our statistical pattern analyzer compares your current symptoms to your historical data. It identifies patterns that preceded past flares and calculates risk based on correlation strength. This is NOT machine learning - it's comparative statistical analysis using only YOUR data."
                ),
                FAQItem(
                    question: "Is my data used to train AI?",
                    answer: "NO. All analysis happens on YOUR device using only YOUR data. We never collect, upload, or share your health information. Your privacy is absolute."
                ),
                FAQItem(
                    question: "Can AI predict my flares?",
                    answer: "The pattern analyzer can identify trends that MAY indicate increased flare risk, but it's NOT medical advice. Always consult your rheumatologist for medical decisions."
                )
            ]
        ),

        FAQSection(
            title: "Medications",
            icon: "pills.fill",
            questions: [
                FAQItem(
                    question: "How do I add a medication?",
                    answer: "Go to the Medications tab, tap the + button, and enter your medication details. You can set reminders, track adherence, and log side effects."
                ),
                FAQItem(
                    question: "Will I get medication reminders?",
                    answer: "Yes! Enable notifications in Settings to receive reminders at your scheduled medication times. You can customize reminder timing for each medication."
                )
            ]
        ),

        FAQSection(
            title: "Data & Privacy",
            icon: "lock.shield.fill",
            questions: [
                FAQItem(
                    question: "Where is my data stored?",
                    answer: "100% on YOUR device. InflamAI uses Core Data for local storage. Your health information never leaves your iPhone unless YOU explicitly choose to export it."
                ),
                FAQItem(
                    question: "Can I export my data?",
                    answer: "Yes! Go to Settings > Export Data. Choose PDF (for your doctor), JSON (complete backup), or CSV (spreadsheet format). YOU control who sees your data."
                ),
                FAQItem(
                    question: "Does InflamAI collect any data?",
                    answer: "NO. We have zero analytics, zero tracking, and zero data collection. Your privacy is non-negotiable. We never see your health information."
                )
            ]
        ),

        FAQSection(
            title: "Troubleshooting",
            icon: "wrench.fill",
            questions: [
                FAQItem(
                    question: "My data isn't syncing",
                    answer: "InflamAI stores data locally - there's no cloud sync by design for privacy. If you're switching devices, use Export Data to create a backup and transfer it manually."
                ),
                FAQItem(
                    question: "How do I delete my data?",
                    answer: "Go to Settings > Privacy Settings > Delete All Data. This permanently removes all health information from your device. This action cannot be undone."
                ),
                FAQItem(
                    question: "The app is crashing",
                    answer: "Try restarting your iPhone. If crashes persist, email support@spinalytics.app with your iOS version and device model."
                )
            ]
        )
    ]
}

// MARK: - FAQ Section View

struct FAQSectionView: View {
    let section: FAQSection
    @Binding var expandedQuestions: Set<UUID>

    var body: some View {
        VStack(alignment: .leading, spacing: AssetsManager.Spacing.sm) {
            // Section Header
            HStack(spacing: AssetsManager.Spacing.sm) {
                Image(systemName: section.icon)
                    .foregroundColor(AssetsManager.Colors.primary)
                    .font(.title3)

                Text(section.title)
                    .font(.headline)
            }
            .padding(.horizontal, AssetsManager.Spacing.md)
            .padding(.top, AssetsManager.Spacing.md)

            // Questions
            VStack(spacing: AssetsManager.Spacing.xs) {
                ForEach(section.questions) { question in
                    FAQQuestionCard(
                        question: question,
                        isExpanded: expandedQuestions.contains(question.id),
                        toggle: {
                            if expandedQuestions.contains(question.id) {
                                expandedQuestions.remove(question.id)
                            } else {
                                expandedQuestions.insert(question.id)
                            }
                        }
                    )
                }
            }
        }
    }
}

// MARK: - FAQ Question Card

struct FAQQuestionCard: View {
    let question: FAQItem
    let isExpanded: Bool
    let toggle: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Question
            Button(action: toggle) {
                HStack {
                    Text(question.question)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(AssetsManager.Colors.primaryText)
                        .multilineTextAlignment(.leading)

                    Spacer()

                    Image(systemName: isExpanded ? "chevron.up.circle.fill" : "chevron.down.circle")
                        .foregroundColor(AssetsManager.Colors.primary)
                }
                .padding(AssetsManager.Spacing.md)
            }
            .buttonStyle(PlainButtonStyle())

            // Answer
            if isExpanded {
                Text(question.answer)
                    .font(.caption)
                    .foregroundColor(AssetsManager.Colors.secondaryText)
                    .padding(.horizontal, AssetsManager.Spacing.md)
                    .padding(.bottom, AssetsManager.Spacing.md)
                    .transition(.opacity.combined(with: .move(edge: .top)))
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
        .animation(.easeInOut(duration: 0.2), value: isExpanded)
    }
}

// MARK: - Models

struct FAQSection: Identifiable {
    let id = UUID()
    let title: String
    let icon: String
    let questions: [FAQItem]
}

struct FAQItem: Identifiable {
    let id = UUID()
    let question: String
    let answer: String
}

// MARK: - Preview

#Preview {
    NavigationView {
        HelpView()
    }
}
