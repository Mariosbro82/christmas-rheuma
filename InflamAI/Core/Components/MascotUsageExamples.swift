//
//  MascotUsageExamples.swift
//  InflamAI-Swift
//
//  Examples of how to use Anky the Ankylosaurus throughout the app
//

import SwiftUI

// MARK: - Empty State Example

struct EmptyStateWithMascot: View {
    var body: some View {
        VStack(spacing: 24) {
            AnkylosaurusMascot(expression: .encouraging, size: 180)
                .bouncing()

            VStack(spacing: 8) {
                Text("No Entries Yet")
                    .font(.title2)
                    .fontWeight(.semibold)

                Text("Tap the + button to log your first symptom!")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
    }
}

// MARK: - Success Message Example

struct SuccessMessageWithMascot: View {
    var body: some View {
        VStack(spacing: 20) {
            AnkylosaurusMascot(expression: .excited, size: 120)
                .waving()

            VStack(spacing: 8) {
                Text("Great Job!")
                    .font(.title3)
                    .fontWeight(.bold)

                Text("Your symptom has been logged successfully")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(red: 0.95, green: 0.98, blue: 0.97))
        )
        .padding()
    }
}

// MARK: - Reminder Card Example

struct ReminderCardWithMascot: View {
    var body: some View {
        HStack(spacing: 16) {
            AnkylosaurusMascot(expression: .happy, size: 60)

            VStack(alignment: .leading, spacing: 4) {
                Text("Time for your medication!")
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Text("Don't forget your evening dose")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Button {
                // Mark as taken
            } label: {
                Image(systemName: "checkmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(Color(red: 0.4, green: 0.7, blue: 0.6))
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(.white)
                .shadow(color: .black.opacity(0.05), radius: 10, y: 4)
        )
        .padding(.horizontal)
    }
}

// MARK: - Loading State Example

struct LoadingStateWithMascot: View {
    @State private var isAnimating = false

    var body: some View {
        VStack(spacing: 20) {
            AnkylosaurusMascot(expression: .happy, size: 150)
                .rotationEffect(.degrees(isAnimating ? 360 : 0))
                .animation(
                    .linear(duration: 2)
                    .repeatForever(autoreverses: false),
                    value: isAnimating
                )
                .onAppear {
                    isAnimating = true
                }

            Text("Analyzing your data...")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Achievement Badge Example

struct AchievementBadgeWithMascot: View {
    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 1.0, green: 0.84, blue: 0.0),
                                Color(red: 1.0, green: 0.65, blue: 0.0)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 120, height: 120)
                    .shadow(color: .orange.opacity(0.3), radius: 15, y: 5)

                AnkylosaurusMascot(expression: .excited, size: 80)
            }

            VStack(spacing: 4) {
                Text("7 Day Streak!")
                    .font(.title3)
                    .fontWeight(.bold)

                Text("You've logged symptoms for 7 days in a row")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
    }
}

// MARK: - Inline Helper Example

struct InlineHelperWithMascot: View {
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            AnkylosaurusMascot(expression: .encouraging, size: 50)

            VStack(alignment: .leading, spacing: 4) {
                Text("Pro Tip!")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(Color(red: 0.4, green: 0.7, blue: 0.6))

                Text("Logging symptoms at the same time each day helps us find better patterns.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(red: 0.4, green: 0.7, blue: 0.6).opacity(0.1))
        )
        .padding(.horizontal)
    }
}

// MARK: - Preview

#Preview("All Examples") {
    ScrollView {
        VStack(spacing: 40) {
            Text("Mascot Usage Examples")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding(.top)

            Group {
                Text("Empty State")
                    .font(.headline)
                EmptyStateWithMascot()
            }

            Divider()

            Group {
                Text("Success Message")
                    .font(.headline)
                SuccessMessageWithMascot()
            }

            Divider()

            Group {
                Text("Reminder Card")
                    .font(.headline)
                ReminderCardWithMascot()
            }

            Divider()

            Group {
                Text("Loading State")
                    .font(.headline)
                LoadingStateWithMascot()
            }

            Divider()

            Group {
                Text("Achievement Badge")
                    .font(.headline)
                AchievementBadgeWithMascot()
            }

            Divider()

            Group {
                Text("Inline Helper")
                    .font(.headline)
                InlineHelperWithMascot()
            }
        }
        .padding(.bottom, 40)
    }
}
