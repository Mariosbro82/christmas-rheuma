//
//  MeditationHomeView.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import SwiftUI

struct MeditationHomeView: View {
    @StateObject private var viewModel = MeditationViewModel()
    @State private var searchText = ""
    @State private var selectedCategory: MeditationCategory?
    @State private var showingPlayer = false
    @State private var selectedSession: MeditationSessionModel?

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from MoreView,
        // which is already wrapped in NavigationView in MainTabView.
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.xl) {
                // Header with streak
                headerSection

                // Search bar
                searchBar

                // Quick actions
                quickActionsSection

                // Recommended sessions
                recommendedSection

                // Categories
                categoriesSection

                // All sessions
                allSessionsSection
            }
            .padding()
        }
        .navigationTitle("Meditation")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                NavigationLink(destination: MeditationProgressView(viewModel: viewModel)) {
                    Image(systemName: "chart.bar.fill")
                }
            }
        }
        .sheet(isPresented: $showingPlayer) {
            if let session = selectedSession {
                MeditationPlayerView(session: session, viewModel: viewModel)
            }
        }
    }

    // MARK: - Header Section

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            if let streak = viewModel.streak, streak.currentStreak > 0 {
                HStack {
                    Image(systemName: "flame.fill")
                        .foregroundColor(.orange)
                        .font(.system(size: Typography.xl))

                    VStack(alignment: .leading, spacing: Spacing.xxs) {
                        Text("\(streak.currentStreak) Day Streak")
                            .font(.system(size: Typography.md, weight: .semibold))
                            .foregroundColor(Colors.Gray.g900)
                        Text("Keep it going!")
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)
                    }

                    Spacer()

                    VStack(alignment: .trailing, spacing: Spacing.xxs) {
                        Text("\(Int(streak.totalMinutes)) min")
                            .font(.system(size: Typography.md, weight: .semibold))
                            .foregroundColor(Colors.Gray.g900)
                        Text("Total")
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }
                .padding(Spacing.md)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(Radii.lg)
            }
        }
    }

    // MARK: - Search Bar

    private var searchBar: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: "magnifyingglass")
                .foregroundColor(Colors.Gray.g400)

            TextField("Search meditations...", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: Typography.base))

            if !searchText.isEmpty {
                Button(action: { searchText = "" }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(Colors.Gray.g400)
                }
            }
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.md)
    }

    // MARK: - Quick Actions

    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Quick Start")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: Spacing.md) {
                    ForEach(viewModel.getQuickSessions().prefix(4)) { session in
                        MeditationQuickActionCard(session: session) {
                            UISelectionFeedbackGenerator().selectionChanged()
                            selectedSession = session
                            showingPlayer = true
                        }
                    }
                }
            }
        }
    }

    // MARK: - Recommended Section

    private var recommendedSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            HStack(spacing: Spacing.sm) {
                Image(systemName: "star.fill")
                    .foregroundColor(Colors.Semantic.warning)
                Text("Recommended for You")
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)
            }

            ForEach(viewModel.getRecommendedSessions().prefix(3)) { session in
                SessionRow(session: session) {
                    UISelectionFeedbackGenerator().selectionChanged()
                    selectedSession = session
                    showingPlayer = true
                }
            }
        }
    }

    // MARK: - Categories Section

    private var categoriesSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Categories")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: Spacing.md) {
                    ForEach(MeditationCategory.allCases, id: \.self) { category in
                        MeditationCategoryChip(
                            category: category,
                            isSelected: selectedCategory == category
                        ) {
                            UISelectionFeedbackGenerator().selectionChanged()
                            selectedCategory = selectedCategory == category ? nil : category
                        }
                    }
                }
            }
        }
    }

    // MARK: - All Sessions Section

    private var allSessionsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("All Sessions")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            ForEach(filteredSessions) { session in
                SessionRow(session: session) {
                    UISelectionFeedbackGenerator().selectionChanged()
                    selectedSession = session
                    showingPlayer = true
                }
            }
        }
    }

    // MARK: - Filtered Sessions

    private var filteredSessions: [MeditationSessionModel] {
        var sessions = viewModel.availableSessions

        if !searchText.isEmpty {
            sessions = viewModel.searchSessions(searchText)
        }

        if let category = selectedCategory {
            sessions = sessions.filter { $0.category == category }
        }

        return sessions
    }
}

// MARK: - Meditation Quick Action Card

struct MeditationQuickActionCard: View {
    let session: MeditationSessionModel
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: Spacing.sm) {
                // Custom dino character image
                Image(session.category.dinoImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 50, height: 50)

                Text(session.title)
                    .font(.system(size: Typography.sm, weight: .medium))
                    .foregroundColor(Colors.Gray.g900)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)

                Text("\(session.durationMinutes) min")
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)
            }
            .frame(width: 140, height: 120)
            .padding(Spacing.md)
            .background(Color(.systemBackground))
            .cornerRadius(Radii.lg)
            .dshadow(Shadows.sm)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Session Row

struct SessionRow: View {
    let session: MeditationSessionModel
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Spacing.md) {
                // Custom dino character
                Image(session.category.dinoImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 50, height: 50)

                // Content
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(session.title)
                        .font(.system(size: Typography.sm, weight: .medium))
                        .foregroundColor(Colors.Gray.g900)

                    Text(session.category.displayName)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)

                    HStack(spacing: Spacing.sm) {
                        Label(session.durationFormatted, systemImage: "clock")
                            .font(.system(size: Typography.xxs))
                            .foregroundColor(Colors.Gray.g500)

                        Label(session.difficulty.displayName, systemImage: "chart.bar")
                            .font(.system(size: Typography.xxs))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }

                Spacer()

                Image(systemName: "play.circle.fill")
                    .font(.system(size: Typography.xl))
                    .foregroundColor(Colors.Accent.purple)
            }
            .padding(Spacing.md)
            .background(Color(.systemBackground))
            .cornerRadius(Radii.lg)
            .dshadow(Shadows.xs)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Meditation Category Chip

struct MeditationCategoryChip: View {
    let category: MeditationCategory
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Spacing.xs) {
                Image(category.dinoImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 20, height: 20)

                Text(category.displayName)
                    .font(.system(size: Typography.xs, weight: .medium))
            }
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.sm)
            .background(isSelected ? Colors.Accent.purple : Colors.Gray.g100)
            .foregroundColor(isSelected ? .white : Colors.Gray.g700)
            .cornerRadius(Radii.full)
        }
    }
}

// MARK: - Preview

#Preview {
    MeditationHomeView()
}
