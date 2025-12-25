//
//  AnalyticsHubView.swift
//  InflamAI
//
//  Analytics Tab - All ANALYSIS activities
//  Trends, ML Predictions, Triggers, Medications, Flares, Assessments
//

import SwiftUI
import CoreData

#if os(iOS)
struct AnalyticsHubView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var animateCards = false

    private let sections: [(title: String, description: String, icon: String, color: Color, destination: AnalyticsDestination)] = [
        ("Trends", "Visualize your health data", "chart.xyaxis.line", .green, .trends),
        ("AI Predictions", "ML-powered flare predictions", "brain.head.profile", .purple, .predictions),
        ("Trigger Insights", "Analyze your flare triggers", "waveform.path.ecg.rectangle", .teal, .triggers),
        ("Medications", "Track adherence & history", "pills.fill", .blue, .medications),
        ("Flare Timeline", "View flare event timeline", "flame.fill", .orange, .flares),
        ("Assessments", "BASDAI, BASFI & surveys", "list.clipboard.fill", .indigo, .assessments)
    ]

    enum AnalyticsDestination {
        case trends, predictions, triggers, medications, flares, assessments
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Understand Your Patterns")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.primary)

                        Text("Insights, trends, and predictions")
                            .font(.system(size: 15))
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .opacity(animateCards ? 1 : 0)
                    .offset(y: animateCards ? 0 : 20)

                    // Cards Grid
                    LazyVGrid(columns: [
                        GridItem(.flexible(), spacing: 16),
                        GridItem(.flexible(), spacing: 16)
                    ], spacing: 16) {
                        ForEach(Array(sections.enumerated()), id: \.offset) { index, section in
                            NavigationLink(destination: destinationView(for: section.destination)) {
                                AnalyticsCard(
                                    title: section.title,
                                    description: section.description,
                                    icon: section.icon,
                                    color: section.color
                                )
                            }
                            .buttonStyle(AnalyticsCardButtonStyle())
                            .opacity(animateCards ? 1 : 0)
                            .offset(y: animateCards ? 0 : 20)
                            .animation(.spring(response: 0.4, dampingFraction: 0.8).delay(Double(index) * 0.05), value: animateCards)
                        }
                    }
                    .padding(.horizontal, 16)

                    Spacer(minLength: 48)
                }
                .padding(.top, 16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Analytics")
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
    private func destinationView(for destination: AnalyticsDestination) -> some View {
        switch destination {
        case .trends:
            TrendsView(context: context)
        case .predictions:
            AIInsightsView(context: context)
        case .triggers:
            TriggerInsightsView()
        case .medications:
            MedicationManagementView(context: context)
        case .flares:
            FlareTimelineView(context: context)
        case .assessments:
            AssessmentsView()
                .environment(\.managedObjectContext, context)
        }
    }
}

// MARK: - Analytics Card

struct AnalyticsCard: View {
    let title: String
    let description: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(spacing: 16) {
            // Icon
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 48, height: 48)

                Image(systemName: icon)
                    .font(.system(size: 22, weight: .medium))
                    .foregroundColor(color)
            }

            // Text
            VStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.primary)
                    .multilineTextAlignment(.center)

                Text(description)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(color.opacity(0.2), lineWidth: 1)
        )
        .shadow(color: color.opacity(0.1), radius: 8, y: 4)
    }
}

// MARK: - Button Style

struct AnalyticsCardButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

#Preview {
    AnalyticsHubView()
}
#endif
