//
//  ExerciseFeedbackView.swift
//  InflamAI-Swift
//
//  Feedback collection view after exercise completion
//

import SwiftUI

struct ExerciseFeedbackView: View {
    let exerciseName: String
    let onSubmit: (ExerciseFeedback, String?) -> Void
    let onSkip: () -> Void

    @State private var selectedFeedback: ExerciseFeedback?
    @State private var notes: String = ""
    @State private var showingNotes: Bool = false

    var body: some View {
        VStack(spacing: 24) {
            // Header
            VStack(spacing: 12) {
                Text("How was it?")
                    .font(.title)
                    .fontWeight(.bold)

                Text(exerciseName)
                    .font(.headline)
                    .foregroundColor(.secondary)
            }
            .padding(.top, 32)

            // Emoji feedback buttons
            VStack(spacing: 16) {
                HStack(spacing: 16) {
                    feedbackButton(.easy)
                    feedbackButton(.manageable)
                }

                HStack(spacing: 16) {
                    feedbackButton(.difficult)
                    feedbackButton(.unbearable)
                }
            }
            .padding(.horizontal)

            // Optional notes section
            if showingNotes || selectedFeedback != nil {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Notes (optional)")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    TextEditor(text: $notes)
                        .frame(height: 100)
                        .padding(8)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color(.systemGray4), lineWidth: 1)
                        )
                }
                .padding(.horizontal)
                .transition(.move(edge: .top).combined(with: .opacity))
            }

            Spacer()

            // Action buttons
            VStack(spacing: 12) {
                if let feedback = selectedFeedback {
                    Button(action: {
                        onSubmit(feedback, notes.isEmpty ? nil : notes)
                    }) {
                        Text("Submit Feedback")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                }

                Button(action: onSkip) {
                    Text("Skip")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
        }
        .animation(.easeInOut, value: selectedFeedback)
        .animation(.easeInOut, value: showingNotes)
    }

    private func feedbackButton(_ feedback: ExerciseFeedback) -> some View {
        Button(action: {
            selectedFeedback = feedback
            showingNotes = true
        }) {
            VStack(spacing: 12) {
                Text(feedback.emoji)
                    .font(.system(size: 60))

                Text(feedback.description)
                    .font(.caption)
                    .fontWeight(.medium)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.primary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 20)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(selectedFeedback == feedback ? Color.blue.opacity(0.1) : Color(.secondarySystemGroupedBackground))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(selectedFeedback == feedback ? Color.blue : Color.clear, lineWidth: 3)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

#Preview {
    NavigationView {
        ExerciseFeedbackView(
            exerciseName: "Cat-Cow Stretch",
            onSubmit: { feedback, notes in
                print("Feedback: \(feedback.description)")
                if let notes = notes {
                    print("Notes: \(notes)")
                }
            },
            onSkip: {
                print("Skipped feedback")
            }
        )
    }
}
