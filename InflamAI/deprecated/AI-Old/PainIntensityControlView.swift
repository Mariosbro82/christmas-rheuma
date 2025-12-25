//
//  PainIntensityControlView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import SwiftUI
import CoreHaptics

struct PainIntensityControlView: View {
    @Binding var currentPainLevel: Double
    let selectedRegions: Set<BodyRegion>
    @Binding var painIntensity: [BodyRegion: Double]
    
    @StateObject private var aiEngine = AIMLEngine.shared
    @StateObject private var hapticEngine = HapticFeedbackEngine.shared
    
    @State private var isAdjustingIntensity = false
    @State private var showingIntensityHistory = false
    @State private var selectedIntensityRegion: BodyRegion?
    
    private let painLevels = [
        (0.0, "No Pain", Color.green),
        (1.0, "Minimal", Color.green.opacity(0.8)),
        (2.0, "Mild", Color.yellow),
        (3.0, "Uncomfortable", Color.yellow.opacity(0.8)),
        (4.0, "Moderate", Color.orange),
        (5.0, "Distracting", Color.orange.opacity(0.8)),
        (6.0, "Distressing", Color.red.opacity(0.6)),
        (7.0, "Unmanageable", Color.red.opacity(0.7)),
        (8.0, "Intense", Color.red.opacity(0.8)),
        (9.0, "Severe", Color.red.opacity(0.9)),
        (10.0, "Unable to Move", Color.red)
    ]
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Text("Pain Intensity")
                    .font(.headline)
                
                Spacer()
                
                if !selectedRegions.isEmpty {
                    Button("History") {
                        showingIntensityHistory = true
                    }
                    .font(.caption)
                    .foregroundColor(.blue)
                }
            }
            
            // Global Pain Level
            VStack(spacing: 15) {
                Text("Overall Pain Level")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                // Pain Level Slider with Visual Feedback
                VStack(spacing: 10) {
                    HStack {
                        Text("0")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Text("\(Int(currentPainLevel))")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(colorForPainLevel(currentPainLevel))
                        
                        Spacer()
                        
                        Text("10")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    // Custom Pain Slider
                    PainLevelSlider(
                        value: $currentPainLevel,
                        isAdjusting: $isAdjustingIntensity
                    )
                    .onChange(of: currentPainLevel) { newValue in
                        handlePainLevelChange(newValue)
                    }
                    
                    // Pain Description
                    Text(descriptionForPainLevel(currentPainLevel))
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .animation(.easeInOut, value: currentPainLevel)
                }
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(colorForPainLevel(currentPainLevel).opacity(0.1))
                    .animation(.easeInOut, value: currentPainLevel)
            )
            
            // Individual Region Intensity
            if !selectedRegions.isEmpty {
                VStack(spacing: 15) {
                    Text("Individual Region Intensity")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 10) {
                        ForEach(Array(selectedRegions), id: \.self) { region in
                            RegionIntensityCard(
                                region: region,
                                intensity: Binding(
                                    get: { painIntensity[region] ?? 0.0 },
                                    set: { painIntensity[region] = $0 }
                                ),
                                isSelected: selectedIntensityRegion == region
                            )
                            .onTapGesture {
                                selectedIntensityRegion = region
                            }
                        }
                    }
                }
            }
            
            // AI-Powered Suggestions
            if currentPainLevel > 3.0 {
                AIPainSuggestionsView(
                    painLevel: currentPainLevel,
                    selectedRegions: selectedRegions,
                    suggestions: aiEngine.getPainManagementSuggestions(
                        level: currentPainLevel,
                        regions: selectedRegions
                    )
                )
                .transition(.slide)
            }
            
            // Quick Action Buttons
            HStack(spacing: 15) {
                QuickActionButton(
                    title: "Log Pain",
                    icon: "plus.circle.fill",
                    color: .blue
                ) {
                    logCurrentPain()
                }
                
                QuickActionButton(
                    title: "Take Medication",
                    icon: "pills.fill",
                    color: .green
                ) {
                    suggestMedication()
                }
                
                QuickActionButton(
                    title: "Emergency",
                    icon: "exclamationmark.triangle.fill",
                    color: .red
                ) {
                    handleEmergency()
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .sheet(isPresented: $showingIntensityHistory) {
            PainIntensityHistoryView(selectedRegions: selectedRegions)
        }
    }
    
    // MARK: - Helper Methods
    
    private func colorForPainLevel(_ level: Double) -> Color {
        let index = min(Int(level), painLevels.count - 1)
        return painLevels[index].2
    }
    
    private func descriptionForPainLevel(_ level: Double) -> String {
        let index = min(Int(level), painLevels.count - 1)
        return painLevels[index].1
    }
    
    private func handlePainLevelChange(_ newValue: Double) {
        // Provide haptic feedback
        hapticEngine.providePainLevelFeedback(intensity: newValue)
        
        // Update AI engine with new pain data
        aiEngine.updatePainLevel(newValue, for: selectedRegions)
        
        // Auto-update individual regions if they're not set
        for region in selectedRegions {
            if painIntensity[region] == nil {
                painIntensity[region] = newValue
            }
        }
    }
    
    private func logCurrentPain() {
        let painEntry = PainEntry(
            level: currentPainLevel,
            regions: selectedRegions,
            intensity: painIntensity,
            timestamp: Date()
        )
        
        aiEngine.logPainEntry(painEntry)
        hapticEngine.provideSuccessFeedback()
    }
    
    private func suggestMedication() {
        let suggestion = aiEngine.getMedicationSuggestion(
            painLevel: currentPainLevel,
            regions: selectedRegions
        )
        
        // Present medication suggestion
    }
    
    private func handleEmergency() {
        if currentPainLevel >= 8.0 {
            // Trigger emergency protocol
            aiEngine.triggerEmergencyProtocol(
                painLevel: currentPainLevel,
                regions: selectedRegions
            )
        }
    }
}

// MARK: - Supporting Views

struct PainLevelSlider: View {
    @Binding var value: Double
    @Binding var isAdjusting: Bool
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background track
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemGray5))
                    .frame(height: 16)
                
                // Pain level gradient
                LinearGradient(
                    colors: [.green, .yellow, .orange, .red],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .mask(
                    RoundedRectangle(cornerRadius: 8)
                        .frame(width: geometry.size.width * (value / 10.0), height: 16)
                )
                
                // Thumb
                Circle()
                    .fill(Color.white)
                    .frame(width: 24, height: 24)
                    .shadow(color: .black.opacity(0.2), radius: 2, x: 0, y: 1)
                    .offset(x: geometry.size.width * (value / 10.0) - 12)
                    .scaleEffect(isAdjusting ? 1.2 : 1.0)
                    .animation(.spring(response: 0.3), value: isAdjusting)
            }
        }
        .frame(height: 24)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { gesture in
                    isAdjusting = true
                    let newValue = min(max(0, gesture.location.x / geometry.size.width * 10), 10)
                    value = newValue
                }
                .onEnded { _ in
                    isAdjusting = false
                }
        )
    }
    
    private var geometry: GeometryProxy {
        GeometryReader { proxy in
            Color.clear.preference(key: SizePreferenceKey.self, value: proxy.size)
        }
        .frame(height: 0)
        .onPreferenceChange(SizePreferenceKey.self) { _ in }
        as! GeometryProxy
    }
}

struct RegionIntensityCard: View {
    let region: BodyRegion
    @Binding var intensity: Double
    let isSelected: Bool
    
    var body: some View {
        VStack(spacing: 8) {
            Text(region.displayName)
                .font(.caption)
                .fontWeight(.medium)
                .multilineTextAlignment(.center)
            
            Text("\(Int(intensity))")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(colorForIntensity(intensity))
            
            // Mini intensity slider
            Slider(value: $intensity, in: 0...10, step: 1)
                .accentColor(colorForIntensity(intensity))
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isSelected ? Color.blue.opacity(0.1) : Color(.systemGray6))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
                )
        )
    }
    
    private func colorForIntensity(_ intensity: Double) -> Color {
        switch intensity {
        case 0...2: return .green
        case 3...4: return .yellow
        case 5...6: return .orange
        case 7...8: return .red.opacity(0.8)
        default: return .red
        }
    }
}

struct AIPainSuggestionsView: View {
    let painLevel: Double
    let selectedRegions: Set<BodyRegion>
    let suggestions: [PainManagementSuggestion]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.blue)
                Text("AI Suggestions")
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            
            ForEach(suggestions, id: \.id) { suggestion in
                HStack {
                    Image(systemName: suggestion.icon)
                        .foregroundColor(suggestion.color)
                        .frame(width: 20)
                    
                    Text(suggestion.text)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.blue.opacity(0.1))
        )
    }
}

struct QuickActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.title3)
                
                Text(title)
                    .font(.caption)
                    .fontWeight(.medium)
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(color)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Data Models

struct PainEntry {
    let level: Double
    let regions: Set<BodyRegion>
    let intensity: [BodyRegion: Double]
    let timestamp: Date
}

struct PainManagementSuggestion {
    let id = UUID()
    let text: String
    let icon: String
    let color: Color
    let priority: Int
}

// MARK: - Extensions

extension BodyRegion {
    var displayName: String {
        switch self {
        case .head: return "Head"
        case .neck: return "Neck"
        case .leftShoulder: return "L. Shoulder"
        case .rightShoulder: return "R. Shoulder"
        case .chest: return "Chest"
        case .leftArm: return "L. Arm"
        case .rightArm: return "R. Arm"
        case .leftElbow: return "L. Elbow"
        case .rightElbow: return "R. Elbow"
        case .leftForearm: return "L. Forearm"
        case .rightForearm: return "R. Forearm"
        case .leftWrist: return "L. Wrist"
        case .rightWrist: return "R. Wrist"
        case .leftHand: return "L. Hand"
        case .rightHand: return "R. Hand"
        case .abdomen: return "Abdomen"
        case .leftHip: return "L. Hip"
        case .rightHip: return "R. Hip"
        case .leftThigh: return "L. Thigh"
        case .rightThigh: return "R. Thigh"
        case .leftKnee: return "L. Knee"
        case .rightKnee: return "R. Knee"
        case .leftCalf: return "L. Calf"
        case .rightCalf: return "R. Calf"
        case .leftAnkle: return "L. Ankle"
        case .rightAnkle: return "R. Ankle"
        case .leftFoot: return "L. Foot"
        case .rightFoot: return "R. Foot"
        // Back regions
        case .cervicalC1: return "C1"
        case .cervicalC2: return "C2"
        case .cervicalC3: return "C3"
        case .cervicalC4: return "C4"
        case .cervicalC5: return "C5"
        case .cervicalC6: return "C6"
        case .cervicalC7: return "C7"
        case .thoracicT1: return "T1"
        case .thoracicT2: return "T2"
        case .thoracicT3: return "T3"
        case .thoracicT4: return "T4"
        case .thoracicT5: return "T5"
        case .thoracicT6: return "T6"
        case .thoracicT7: return "T7"
        case .thoracicT8: return "T8"
        case .thoracicT9: return "T9"
        case .thoracicT10: return "T10"
        case .thoracicT11: return "T11"
        case .thoracicT12: return "T12"
        case .lumbarL1: return "L1"
        case .lumbarL2: return "L2"
        case .lumbarL3: return "L3"
        case .lumbarL4: return "L4"
        case .lumbarL5: return "L5"
        case .sacralS1: return "S1"
        case .sacralS2: return "S2"
        case .sacralS3: return "S3"
        case .sacralS4: return "S4"
        case .sacralS5: return "S5"
        case .leftShoulderBlade: return "L. Shoulder Blade"
        case .rightShoulderBlade: return "R. Shoulder Blade"
        case .leftKidneyArea: return "L. Kidney"
        case .rightKidneyArea: return "R. Kidney"
        case .tailbone: return "Tailbone"
        case .leftHipBack: return "L. Hip (Back)"
        case .rightHipBack: return "R. Hip (Back)"
        case .upperTrapezius: return "Upper Trap"
        case .middleTrapezius: return "Middle Trap"
        case .lowerTrapezius: return "Lower Trap"
        case .leftRhomboid: return "L. Rhomboid"
        case .rightRhomboid: return "R. Rhomboid"
        case .leftLatissimus: return "L. Lat"
        case .rightLatissimus: return "R. Lat"
        case .leftArmBack: return "L. Arm (Back)"
        case .rightArmBack: return "R. Arm (Back)"
        case .leftLegBack: return "L. Leg (Back)"
        case .rightLegBack: return "R. Leg (Back)"
        }
    }
}

// MARK: - Preference Key

struct SizePreferenceKey: PreferenceKey {
    static var defaultValue: CGSize = .zero
    static func reduce(value: inout CGSize, nextValue: () -> CGSize) {
        value = nextValue()
    }
}