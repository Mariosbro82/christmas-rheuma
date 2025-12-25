//
//  MorningCheckInView.swift
//  InflamAI
//
//  Quick Morning Check-in for ML feature collection
//  Collects 6 key data points for Phase 2 ML features
//
//  Features collected:
//  - pain_current (index 18)
//  - nocturnal_pain (index 21)
//  - morning_stiffness_duration (index 22)
//  - morning_stiffness_severity (index 23)
//  - pain_burning (index 25)
//  - pain_aching (index 26)
//  - pain_sharp (index 27)
//  - pain_interference_sleep (index 28)
//  - pain_interference_activity (index 29)
//  - breakthrough_pain (index 31)
//  - mood_current (index 64)
//  - stress_level (index 68)
//  - universal_assessment (index 88)
//

import SwiftUI
import CoreData

struct MorningCheckInView: View {
    @StateObject private var viewModel: MorningCheckInViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) var reduceMotion

    init(context: NSManagedObjectContext) {
        _viewModel = StateObject(wrappedValue: MorningCheckInViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    headerSection

                    // Question Cards
                    VStack(spacing: 20) {
                        // 1. Universal Assessment - "How do you feel overall?"
                        overallFeelingCard

                        // 2. Pain Current
                        painCard

                        // 2b. Pain Details (if pain > 0)
                        if viewModel.painCurrent > 0 {
                            painDetailsCard
                        }

                        // 3. Morning Stiffness Duration & Severity
                        stiffnessCard

                        // 4. Mood
                        moodCard

                        // 5. Stress Level
                        stressCard
                    }
                    .padding(.horizontal)

                    // Save Button
                    saveButton
                        .padding(.horizontal)
                        .padding(.bottom, 32)
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Morning Check-In")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .alert("Saved!", isPresented: $viewModel.showingSaveConfirmation) {
                Button("OK") {
                    dismiss()
                }
            } message: {
                Text("Your morning check-in has been recorded.")
            }
            .alert("Error", isPresented: $viewModel.showingError) {
                Button("OK") {}
            } message: {
                Text(viewModel.errorMessage)
            }
        }
    }

    // MARK: - Header Section

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "sun.horizon.fill")
                .font(.system(size: 48))
                .foregroundColor(.orange)
                .padding(.top, 16)

            Text("Good Morning")
                .font(.title2)
                .fontWeight(.bold)

            Text("Quick check-in takes ~30 seconds")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.bottom, 8)
    }

    // MARK: - Overall Feeling Card (Universal Assessment)

    private var overallFeelingCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "heart.circle.fill")
                    .foregroundColor(.pink)
                    .font(.title2)
                Text("How do you feel overall?")
                    .font(.headline)
            }

            // Emoji feedback
            Text(viewModel.overallFeelingEmoji)
                .font(.system(size: 48))
                .frame(maxWidth: .infinity)

            // Value display
            HStack {
                Text(String(format: "%.0f", viewModel.overallFeeling))
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundColor(colorForFeeling(viewModel.overallFeeling))
                Text("/ 10")
                    .font(.title3)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity)

            Slider(value: $viewModel.overallFeeling, in: 0...10, step: 1)
                .tint(colorForFeeling(viewModel.overallFeeling))
                .onChange(of: viewModel.overallFeeling) { _ in
                    if !reduceMotion {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    }
                }

            HStack {
                Text("Very poor")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Excellent")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Overall feeling: \(Int(viewModel.overallFeeling)) out of 10")
    }

    // MARK: - Pain Card

    private var painCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "bolt.heart.fill")
                    .foregroundColor(.red)
                    .font(.title2)
                Text("Current Pain Level")
                    .font(.headline)
            }

            HStack {
                Text(String(format: "%.0f", viewModel.painCurrent))
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundColor(colorForPain(viewModel.painCurrent))
                Text("/ 10")
                    .font(.title3)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity)

            Slider(value: $viewModel.painCurrent, in: 0...10, step: 1)
                .tint(colorForPain(viewModel.painCurrent))
                .onChange(of: viewModel.painCurrent) { _ in
                    if !reduceMotion {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    }
                }

            HStack {
                Text("No pain")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Worst pain")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Pain level: \(Int(viewModel.painCurrent)) out of 10")
    }

    // MARK: - Pain Details Card (Nocturnal, Types, Interference)

    private var painDetailsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "moon.stars.fill")
                    .foregroundColor(.indigo)
                    .font(.title2)
                Text("Pain Details")
                    .font(.headline)
            }

            // Nocturnal Pain Toggle
            Toggle(isOn: $viewModel.nocturnalPain) {
                HStack {
                    Image(systemName: "bed.double.fill")
                        .foregroundColor(.purple)
                    Text("Did you have pain during the night?")
                        .font(.subheadline)
                }
            }
            .onChange(of: viewModel.nocturnalPain) { _ in
                if !reduceMotion {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                }
            }

            Divider()

            // Pain Type Checkboxes
            VStack(alignment: .leading, spacing: 8) {
                Text("What type of pain are you experiencing?")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                HStack(spacing: 12) {
                    painTypeButton(
                        title: "Burning",
                        icon: "flame.fill",
                        isSelected: viewModel.painBurning,
                        color: .orange
                    ) {
                        viewModel.painBurning.toggle()
                    }

                    painTypeButton(
                        title: "Aching",
                        icon: "waveform.path",
                        isSelected: viewModel.painAching,
                        color: .blue
                    ) {
                        viewModel.painAching.toggle()
                    }

                    painTypeButton(
                        title: "Sharp",
                        icon: "bolt.fill",
                        isSelected: viewModel.painSharp,
                        color: .red
                    ) {
                        viewModel.painSharp.toggle()
                    }
                }
            }

            Divider()

            // Pain Interference Sliders
            VStack(alignment: .leading, spacing: 12) {
                Text("How much does pain interfere with...")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                // Sleep interference
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "moon.zzz.fill")
                            .foregroundColor(.indigo)
                        Text("Sleep")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.0f", viewModel.painInterferenceSleep))
                            .font(.headline)
                            .foregroundColor(colorForInterference(viewModel.painInterferenceSleep))
                    }

                    Slider(value: $viewModel.painInterferenceSleep, in: 0...10, step: 1)
                        .tint(colorForInterference(viewModel.painInterferenceSleep))

                    HStack {
                        Text("Not at all")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("Completely")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                // Activity interference
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "figure.walk")
                            .foregroundColor(.teal)
                        Text("Daily Activities")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.0f", viewModel.painInterferenceActivity))
                            .font(.headline)
                            .foregroundColor(colorForInterference(viewModel.painInterferenceActivity))
                    }

                    Slider(value: $viewModel.painInterferenceActivity, in: 0...10, step: 1)
                        .tint(colorForInterference(viewModel.painInterferenceActivity))

                    HStack {
                        Text("Not at all")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("Completely")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }

            Divider()

            // Breakthrough Pain Toggle
            Toggle(isOn: $viewModel.breakthroughPain) {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Breakthrough Pain")
                            .font(.subheadline)
                        Text("Sudden severe pain despite treatment")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .onChange(of: viewModel.breakthroughPain) { _ in
                if !reduceMotion {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
    }

    // MARK: - Pain Type Button Helper

    private func painTypeButton(title: String, icon: String, isSelected: Bool, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: {
            action()
            if !reduceMotion {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            }
        }) {
            VStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.title2)
                Text(title)
                    .font(.caption)
                    .fontWeight(isSelected ? .semibold : .regular)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(isSelected ? color.opacity(0.15) : Color(.systemGray6))
            .foregroundColor(isSelected ? color : .secondary)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isSelected ? color : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(title) pain type")
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    // MARK: - Stiffness Card (Duration + Severity)

    private var stiffnessCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "figure.stand")
                    .foregroundColor(.orange)
                    .font(.title2)
                Text("Morning Stiffness")
                    .font(.headline)
            }

            // Duration
            VStack(alignment: .leading, spacing: 8) {
                Text("How long did stiffness last?")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                HStack {
                    Text("\(Int(viewModel.stiffnessDuration))")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.orange)
                    Text("minutes")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Slider(value: $viewModel.stiffnessDuration, in: 0...180, step: 5)
                    .tint(.orange)

                HStack {
                    Text("None")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("3+ hours")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Divider()
                .padding(.vertical, 4)

            // Severity
            VStack(alignment: .leading, spacing: 8) {
                Text("How severe was the stiffness?")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                HStack {
                    Text(String(format: "%.0f", viewModel.stiffnessSeverity))
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(colorForSeverity(viewModel.stiffnessSeverity))
                    Text("/ 10")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Slider(value: $viewModel.stiffnessSeverity, in: 0...10, step: 1)
                    .tint(colorForSeverity(viewModel.stiffnessSeverity))

                HStack {
                    Text("None")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("Severe")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
    }

    // MARK: - Mood Card

    private var moodCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "face.smiling.fill")
                    .foregroundColor(.purple)
                    .font(.title2)
                Text("Current Mood")
                    .font(.headline)
            }

            // Emoji picker
            HStack(spacing: 12) {
                ForEach(MoodOption.allCases) { mood in
                    Button {
                        viewModel.moodCurrent = Double(mood.value)
                        if !reduceMotion {
                            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        }
                    } label: {
                        VStack(spacing: 4) {
                            Text(mood.emoji)
                                .font(.system(size: viewModel.moodCurrent == Double(mood.value) ? 36 : 28))

                            if viewModel.moodCurrent == Double(mood.value) {
                                Text(mood.label)
                                    .font(.caption2)
                                    .foregroundColor(.primary)
                            }
                        }
                        .padding(8)
                        .background(
                            viewModel.moodCurrent == Double(mood.value) ?
                            Color.purple.opacity(0.15) :
                            Color.clear
                        )
                        .cornerRadius(12)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(mood.accessibilityLabel)
                    .accessibilityAddTraits(viewModel.moodCurrent == Double(mood.value) ? [.isSelected] : [])
                }
            }
            .frame(maxWidth: .infinity)

            // Fine-tune slider
            Slider(value: $viewModel.moodCurrent, in: 0...10, step: 1)
                .tint(.purple)

            HStack {
                Text("Very low")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Value: \(Int(viewModel.moodCurrent))")
                    .font(.caption)
                    .foregroundColor(.purple)
                Spacer()
                Text("Great")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
    }

    // MARK: - Stress Card

    private var stressCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.teal)
                    .font(.title2)
                Text("Stress Level")
                    .font(.headline)
            }

            HStack {
                Text(String(format: "%.0f", viewModel.stressLevel))
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundColor(colorForStress(viewModel.stressLevel))
                Text("/ 10")
                    .font(.title3)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity)

            Slider(value: $viewModel.stressLevel, in: 0...10, step: 1)
                .tint(colorForStress(viewModel.stressLevel))
                .onChange(of: viewModel.stressLevel) { _ in
                    if !reduceMotion {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    }
                }

            HStack {
                Text("Relaxed")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Very stressed")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Stress level: \(Int(viewModel.stressLevel)) out of 10")
    }

    // MARK: - Save Button

    private var saveButton: some View {
        Button {
            Task {
                await viewModel.saveCheckIn()
            }
        } label: {
            HStack {
                if viewModel.isSaving {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Image(systemName: "checkmark.circle.fill")
                    Text("Save Check-In")
                        .fontWeight(.semibold)
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.accentColor)
            .foregroundColor(.white)
            .cornerRadius(16)
        }
        .disabled(viewModel.isSaving)
        .accessibilityLabel("Save morning check-in")
    }

    // MARK: - Color Helpers

    private func colorForFeeling(_ value: Double) -> Color {
        switch value {
        case 0..<3: return .red
        case 3..<5: return .orange
        case 5..<7: return .yellow
        case 7..<9: return .green
        default: return .green
        }
    }

    private func colorForPain(_ value: Double) -> Color {
        switch value {
        case 0..<3: return .green
        case 3..<5: return .yellow
        case 5..<7: return .orange
        default: return .red
        }
    }

    private func colorForSeverity(_ value: Double) -> Color {
        switch value {
        case 0..<3: return .green
        case 3..<5: return .yellow
        case 5..<7: return .orange
        default: return .red
        }
    }

    private func colorForStress(_ value: Double) -> Color {
        switch value {
        case 0..<3: return .green
        case 3..<5: return .yellow
        case 5..<7: return .orange
        default: return .red
        }
    }

    private func colorForInterference(_ value: Double) -> Color {
        switch value {
        case 0..<3: return .green
        case 3..<5: return .yellow
        case 5..<7: return .orange
        default: return .red
        }
    }
}

// MARK: - Mood Option Enum

enum MoodOption: Int, CaseIterable, Identifiable {
    case veryLow = 0
    case low = 2
    case neutral = 5
    case good = 7
    case great = 10

    var id: Int { rawValue }
    var value: Int { rawValue }

    var emoji: String {
        switch self {
        case .veryLow: return "😞"
        case .low: return "😕"
        case .neutral: return "😐"
        case .good: return "🙂"
        case .great: return "😊"
        }
    }

    var label: String {
        switch self {
        case .veryLow: return "Very Low"
        case .low: return "Low"
        case .neutral: return "Okay"
        case .good: return "Good"
        case .great: return "Great"
        }
    }

    var accessibilityLabel: String {
        "\(label) mood, value \(value)"
    }
}

// MARK: - Preview

struct MorningCheckInView_Previews: PreviewProvider {
    static var previews: some View {
        MorningCheckInView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
