//
//  FlareLogView.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//  Enhanced with beautiful flare logging UI
//

import SwiftUI
#if os(watchOS)
import WatchKit
#endif

struct FlareLogView: View {
    @StateObject private var viewModel = WatchFlareViewModel()
    @EnvironmentObject var connectivityManager: WatchConnectivityManager
    @Environment(\.dismiss) private var dismiss

    @State private var severity: Int = 5
    @State private var selectedBodyParts: Set<String> = []
    @State private var showConfirmation = false
    @State private var isSaving = false

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                // Header
                VStack(spacing: 6) {
                    ZStack {
                        Circle()
                            .fill(severityGradient)
                            .frame(width: 60, height: 60)

                        Image(systemName: "flame.fill")
                            .font(.system(size: 32))
                            .foregroundColor(.white)
                            .symbolRenderingMode(.hierarchical)
                    }

                    Text("Flare Log")
                        .font(.headline)
                        .fontWeight(.bold)

                    Text("Document your flare")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 8)

                // Severity Selector
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Image(systemName: "gauge")
                            .foregroundColor(severityColor)
                        Text("Severity")
                            .font(.system(size: 13, weight: .semibold))
                        Spacer()
                        Text(severityLabel)
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(severityColor)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(severityColor.opacity(0.2))
                            .cornerRadius(8)
                    }

                    // Severity Buttons
                    HStack(spacing: 6) {
                        ForEach([1, 2, 3, 4], id: \.self) { level in
                            Button(action: {
                                severity = level * 2 + 1
                                #if os(watchOS)
                                WKInterfaceDevice.current().play(.click)
                                #endif
                            }) {
                                VStack(spacing: 3) {
                                    Image(systemName: "flame.fill")
                                        .font(.system(size: 16))
                                        .foregroundColor(severity >= level * 2 ? severityColorForLevel(level) : .gray)

                                    Text(severityLabelForLevel(level))
                                        .font(.system(size: 9))
                                        .foregroundColor(severity >= level * 2 ? severityColorForLevel(level) : .secondary)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                        .fill(severity >= level * 2 ? severityColorForLevel(level).opacity(0.2) : Color(white: 0.15))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(severity >= level * 2 ? severityColorForLevel(level) : Color.clear, lineWidth: 1.5)
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color(white: 0.15))
                )

                // Quick Body Parts
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Image(systemName: "figure.stand")
                            .foregroundColor(.blue)
                            .font(.system(size: 12))
                        Text("Areas")
                            .font(.system(size: 13, weight: .semibold))
                    }

                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 6) {
                        ForEach(["Neck", "Spine", "Lower Back", "Hips", "Knees", "Shoulders"], id: \.self) { part in
                            Button(action: {
                                if selectedBodyParts.contains(part) {
                                    selectedBodyParts.remove(part)
                                } else {
                                    selectedBodyParts.insert(part)
                                }
                                #if os(watchOS)
                                WKInterfaceDevice.current().play(.click)
                                #endif
                            }) {
                                HStack(spacing: 4) {
                                    Image(systemName: selectedBodyParts.contains(part) ? "checkmark.circle.fill" : "circle")
                                        .foregroundColor(selectedBodyParts.contains(part) ? .blue : .secondary)
                                        .font(.system(size: 12))

                                    Text(part)
                                        .font(.system(size: 11, weight: .medium))
                                        .foregroundColor(selectedBodyParts.contains(part) ? .blue : .primary)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.8)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                        .fill(selectedBodyParts.contains(part) ? Color.blue.opacity(0.2) : Color(white: 0.15))
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color(white: 0.15))
                )

                // Save Button
                Button(action: saveFlare) {
                    HStack {
                        if isSaving {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Image(systemName: "flame.fill")
                            Text("Log Flare")
                        }
                    }
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(
                        LinearGradient(
                            colors: [.red, .orange],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(12)
                }
                .buttonStyle(.plain)
                .disabled(isSaving)
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 8)
        }
        .alert("Flare Logged", isPresented: $showConfirmation) {
            Button("Done") {}
        } message: {
            Text("Your flare has been recorded")
        }
    }

    private var severityGradient: LinearGradient {
        LinearGradient(
            colors: [severityColor, severityColor.opacity(0.7)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    private var severityColor: Color {
        switch severity {
        case 0...2: return .green
        case 3...5: return .yellow
        case 6...7: return .orange
        default: return .red
        }
    }

    private var severityLabel: String {
        switch severity {
        case 0...2: return "Mild"
        case 3...5: return "Moderate"
        case 6...7: return "Severe"
        default: return "Extreme"
        }
    }

    private func severityColorForLevel(_ level: Int) -> Color {
        switch level {
        case 1: return .green
        case 2: return .yellow
        case 3: return .orange
        default: return .red
        }
    }

    private func severityLabelForLevel(_ level: Int) -> String {
        switch level {
        case 1: return "Mild"
        case 2: return "Moderate"
        case 3: return "Severe"
        default: return "Extreme"
        }
    }

    private func saveFlare() {
        isSaving = true
        #if os(watchOS)
        WKInterfaceDevice.current().play(.start)
        #endif

        Task {
            let response = await connectivityManager.sendMessage([
                "type": "flare_quick_log",
                "severity": severity,
                "symptoms": Array(selectedBodyParts),
                "triggers": [],
                "timestamp": Date().timeIntervalSince1970
            ])

            if response?["success"] as? Bool == true {
                isSaving = false
                showConfirmation = true
                #if os(watchOS)
                WKInterfaceDevice.current().play(.success)
                #endif

                try? await Task.sleep(for: .seconds(1.5))
                severity = 5
                selectedBodyParts.removeAll()
            } else {
                isSaving = false
                #if os(watchOS)
                WKInterfaceDevice.current().play(.failure)
                #endif
            }
        }
    }
}

enum FlareSeverityOption: Int, CaseIterable, Identifiable {
    case mild = 4
    case moderate = 7
    case severe = 9

    var id: Int { rawValue }

    var displayName: String {
        switch self {
        case .mild: return "Mild"
        case .moderate: return "Moderate"
        case .severe: return "Severe"
        }
    }

    var severityValue: Int {
        rawValue
    }
}

enum FlareSymptomOption: String, CaseIterable, Identifiable {
    case pain
    case stiffness
    case fatigue
    case swelling

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .pain: return "Pain spike"
        case .stiffness: return "Stiffness"
        case .fatigue: return "Extreme fatigue"
        case .swelling: return "Swelling"
        }
    }

    var systemImage: String {
        switch self {
        case .pain: return "bolt.heart.fill"
        case .stiffness: return "figure.arms.open"
        case .fatigue: return "zzz"
        case .swelling: return "drop.triangle"
        }
    }
}

enum FlareTriggerOption: String, CaseIterable, Identifiable {
    case weather
    case stress
    case infection
    case activity

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .weather: return "Weather"
        case .stress: return "Stress"
        case .infection: return "Infection"
        case .activity: return "Activity"
        }
    }

    var systemImage: String {
        switch self {
        case .weather: return "cloud.drizzle.fill"
        case .stress: return "exclamationmark.triangle.fill"
        case .infection: return "bandage.fill"
        case .activity: return "figure.run"
        }
    }
}

#Preview {
    NavigationStack {
        FlareLogView()
    }
}
