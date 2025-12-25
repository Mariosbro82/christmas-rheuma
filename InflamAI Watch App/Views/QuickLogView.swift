//
//  QuickLogView.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//  Enhanced with beautiful quick log UI
//

import SwiftUI
#if os(watchOS)
import WatchKit
#endif

struct QuickLogView: View {
    @StateObject private var viewModel = WatchSymptomViewModel()
    @EnvironmentObject var connectivityManager: WatchConnectivityManager
    @Environment(\.dismiss) private var dismiss

    @State private var painLevel: Double = 0
    @State private var stiffnessLevel: Double = 0
    @State private var fatigueLevel: Double = 0
    @State private var markAsFlare = false
    @State private var showConfirmation = false
    @State private var isSaving = false

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                // Header
                VStack(spacing: 6) {
                    Image(systemName: "heart.text.square.fill")
                        .font(.system(size: 36))
                        .foregroundColor(.pink)
                        .symbolRenderingMode(.hierarchical)

                    Text("Quick Log")
                        .font(.headline)
                        .fontWeight(.bold)

                    Text("How are you feeling?")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 8)

                // Pain Level
                SymptomSliderCard(
                    title: "Pain Level",
                    icon: "bolt.fill",
                    value: $painLevel,
                    color: .red,
                    gradient: [.red, .pink]
                )

                // Stiffness
                SymptomSliderCard(
                    title: "Stiffness",
                    icon: "figure.walk.motion",
                    value: $stiffnessLevel,
                    color: .orange,
                    gradient: [.orange, .yellow]
                )

                // Fatigue
                SymptomSliderCard(
                    title: "Fatigue",
                    icon: "battery.25",
                    value: $fatigueLevel,
                    color: .purple,
                    gradient: [.purple, .indigo]
                )

                // Flare Toggle
                Toggle(isOn: $markAsFlare) {
                    HStack(spacing: 8) {
                        Image(systemName: "flame.fill")
                            .foregroundColor(markAsFlare ? .red : .gray)
                        Text("Mark as Flare")
                            .font(.system(size: 14, weight: .medium))
                    }
                }
                .tint(.red)
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(markAsFlare ? Color.red.opacity(0.2) : Color(white: 0.15))
                )

                // Save Button
                Button(action: saveLog) {
                    HStack {
                        if isSaving {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Image(systemName: "checkmark.circle.fill")
                            Text("Save Log")
                        }
                    }
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(
                        LinearGradient(
                            colors: canSave ? [.green, .mint] : [.gray, .gray],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(12)
                }
                .buttonStyle(.plain)
                .disabled(!canSave || isSaving)
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 8)
        }
        .alert("Success!", isPresented: $showConfirmation) {
            Button("Done") {
                // Will auto dismiss
            }
        } message: {
            Text("Your symptoms have been logged")
        }
    }

    private var canSave: Bool {
        painLevel > 0 || stiffnessLevel > 0 || fatigueLevel > 0
    }

    private func saveLog() {
        guard canSave else { return }

        isSaving = true
        #if os(watchOS)
        WKInterfaceDevice.current().play(.start)
        #endif

        Task {
            let response = await connectivityManager.sendMessage([
                "type": "symptom_log",
                "id": UUID().uuidString,
                "pain": Int(painLevel),
                "stiffness": Int(stiffnessLevel),
                "fatigue": Int(fatigueLevel),
                "isFlare": markAsFlare,
                "timestamp": Date().timeIntervalSince1970
            ])

            if response?["success"] as? Bool == true {
                isSaving = false
                showConfirmation = true
                #if os(watchOS)
                WKInterfaceDevice.current().play(.success)
                #endif

                // Reset after successful save
                try? await Task.sleep(for: .seconds(1.5))
                painLevel = 0
                stiffnessLevel = 0
                fatigueLevel = 0
                markAsFlare = false
            } else {
                isSaving = false
                #if os(watchOS)
                WKInterfaceDevice.current().play(.failure)
                #endif
            }
        }
    }
}

// MARK: - Symptom Slider Card

struct SymptomSliderCard: View {
    let title: String
    let icon: String
    @Binding var value: Double
    let color: Color
    let gradient: [Color]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.system(size: 16))

                Text(title)
                    .font(.system(size: 14, weight: .semibold))

                Spacer()

                // Value Display
                Text("\(Int(value))")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(color)
                    .frame(width: 32, height: 32)
                    .background(
                        Circle()
                            .fill(color.opacity(0.15))
                    )
            }

            // Slider
            VStack(spacing: 4) {
                Slider(value: $value, in: 0...10, step: 1)
                    .tint(color)
                    .onChange(of: value) { _ in
                        #if os(watchOS)
                        WKInterfaceDevice.current().play(.click)
                        #endif
                    }

                HStack {
                    Text("None")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("Severe")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                }
            }

            // Visual Intensity Indicator
            HStack(spacing: 4) {
                ForEach(0..<10, id: \.self) { index in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(index < Int(value) ? color : Color.gray.opacity(0.2))
                        .frame(height: 6)
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(value > 0 ? color.opacity(0.15) : Color(white: 0.15))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(value > 0 ? color.opacity(0.3) : Color.clear, lineWidth: 1)
        )
    }
}

#Preview {
    QuickLogView()
}
