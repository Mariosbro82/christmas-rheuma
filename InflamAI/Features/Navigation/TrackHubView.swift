//
//  TrackHubView.swift
//  InflamAI
//
//  Track Tab - All INPUT activities
//  Body Map + Journal
//

import SwiftUI
import CoreData

#if os(iOS)
struct TrackHubView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var animateCards = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Log Your Symptoms")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.primary)

                        Text("Track pain locations and journal entries")
                            .font(.system(size: 15))
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .opacity(animateCards ? 1 : 0)
                    .offset(y: animateCards ? 0 : 20)

                    // Body Map Card
                    NavigationLink(destination: PainTrackingView().environment(\.managedObjectContext, context)) {
                        TrackCard(
                            title: "Body Map",
                            description: "Log pain by body region",
                            icon: "figure.stand",
                            color: .blue
                        )
                    }
                    .buttonStyle(TrackCardButtonStyle())
                    .padding(.horizontal, 16)
                    .opacity(animateCards ? 1 : 0)
                    .offset(y: animateCards ? 0 : 20)

                    // Journal Card
                    NavigationLink(destination: JournalView().environment(\.managedObjectContext, context)) {
                        TrackCard(
                            title: "Journal",
                            description: "Write your health diary",
                            icon: "book.fill",
                            color: .purple
                        )
                    }
                    .buttonStyle(TrackCardButtonStyle())
                    .padding(.horizontal, 16)
                    .opacity(animateCards ? 1 : 0)
                    .offset(y: animateCards ? 0 : 20)

                    Spacer(minLength: 48)
                }
                .padding(.top, 16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Track")
            .navigationBarTitleDisplayMode(.large)
        }
        .navigationViewStyle(.stack)
        .onAppear {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                animateCards = true
            }
        }
    }
}

// MARK: - Track Card

struct TrackCard: View {
    let title: String
    let description: String
    let icon: String
    let color: Color

    var body: some View {
        HStack(spacing: 16) {
            // Icon
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 56, height: 56)

                Image(systemName: icon)
                    .font(.system(size: 24, weight: .medium))
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
        .shadow(color: color.opacity(0.1), radius: 8, y: 4)
    }
}

// MARK: - Button Style

struct TrackCardButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

#Preview {
    TrackHubView()
}
#endif
