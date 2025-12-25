//
//  WellnessHubView.swift
//  InflamAI
//
//  Wellness Tab - Self-care activities
//  Exercise, Meditation, Breathing, Routines, Coach, Library
//

import SwiftUI
import CoreData

#if os(iOS)
struct WellnessHubView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var animateCards = false

    // Featured (large cards)
    private let featuredSections: [(title: String, description: String, icon: String, color: Color, destination: WellnessDestination)] = [
        ("Exercise Library", "AS-specific physio exercises", "figure.run", .green, .exercises),
        ("Meditation", "Guided sessions & breathing", "leaf.fill", .teal, .meditation)
    ]

    // Grid (small cards)
    private let gridSections: [(title: String, icon: String, color: Color, destination: WellnessDestination)] = [
        ("My Routines", "calendar.badge.clock", .blue, .routines),
        ("Personal Coach", "sparkles", .purple, .coach),
        ("Knowledge Library", "books.vertical.fill", .indigo, .library)
    ]

    enum WellnessDestination {
        case exercises, routines, coach, meditation, library
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Take Care of Yourself")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.primary)

                        Text("Exercises, meditation & self-care")
                            .font(.system(size: 15))
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .opacity(animateCards ? 1 : 0)
                    .offset(y: animateCards ? 0 : 20)

                    // Featured Cards (Large)
                    VStack(spacing: 16) {
                        ForEach(Array(featuredSections.enumerated()), id: \.offset) { index, section in
                            NavigationLink(destination: destinationView(for: section.destination)) {
                                WellnessFeatureCard(
                                    title: section.title,
                                    description: section.description,
                                    icon: section.icon,
                                    color: section.color
                                )
                            }
                            .buttonStyle(WellnessCardButtonStyle())
                            .opacity(animateCards ? 1 : 0)
                            .offset(y: animateCards ? 0 : 20)
                            .animation(.spring(response: 0.4, dampingFraction: 0.8).delay(Double(index) * 0.05), value: animateCards)
                        }
                    }
                    .padding(.horizontal, 16)

                    // Grid Cards (Small)
                    LazyVGrid(columns: [
                        GridItem(.flexible(), spacing: 16),
                        GridItem(.flexible(), spacing: 16),
                        GridItem(.flexible(), spacing: 16)
                    ], spacing: 16) {
                        ForEach(Array(gridSections.enumerated()), id: \.offset) { index, section in
                            NavigationLink(destination: destinationView(for: section.destination)) {
                                WellnessGridCard(
                                    title: section.title,
                                    icon: section.icon,
                                    color: section.color
                                )
                            }
                            .buttonStyle(WellnessCardButtonStyle())
                            .opacity(animateCards ? 1 : 0)
                            .offset(y: animateCards ? 0 : 20)
                            .animation(.spring(response: 0.4, dampingFraction: 0.8).delay(Double(index + 2) * 0.05), value: animateCards)
                        }
                    }
                    .padding(.horizontal, 16)

                    Spacer(minLength: 48)
                }
                .padding(.top, 16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Wellness")
            .navigationBarTitleDisplayMode(.large)
        }
        .navigationViewStyle(.stack)
        .onAppear {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                animateCards = true
            }
        }
    }

    @ViewBuilder
    private func destinationView(for destination: WellnessDestination) -> some View {
        switch destination {
        case .exercises:
            ExerciseLibraryView()
                .environment(\.managedObjectContext, context)
        case .routines:
            RoutineManagementView()
                .environment(\.managedObjectContext, context)
        case .coach:
            CoachCompositorView()
                .environment(\.managedObjectContext, context)
        case .meditation:
            MeditationHomeView()
        case .library:
            LibraryView()
        }
    }
}

// MARK: - Wellness Feature Card (Large)

struct WellnessFeatureCard: View {
    let title: String
    let description: String
    let icon: String
    let color: Color

    var body: some View {
        HStack(spacing: 24) {
            // Icon
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(
                        LinearGradient(
                            colors: [color.opacity(0.2), color.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 64, height: 64)

                Image(systemName: icon)
                    .font(.system(size: 28, weight: .medium))
                    .foregroundColor(color)
            }

            // Text
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundColor(.primary)

                Text(description)
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Chevron
            Image(systemName: "chevron.right")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(Color(.tertiaryLabel))
        }
        .padding(24)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(color.opacity(0.2), lineWidth: 1)
        )
        .shadow(color: color.opacity(0.15), radius: 12, y: 6)
    }
}

// MARK: - Wellness Grid Card (Small)

struct WellnessGridCard: View {
    let title: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(spacing: 12) {
            // Icon
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 44, height: 44)

                Image(systemName: icon)
                    .font(.system(size: 20, weight: .medium))
                    .foregroundColor(color)
            }

            // Text
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.primary)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(color.opacity(0.15), lineWidth: 1)
        )
        .shadow(color: color.opacity(0.08), radius: 6, y: 3)
    }
}

// MARK: - Button Style

struct WellnessCardButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

#Preview {
    WellnessHubView()
}
#endif
