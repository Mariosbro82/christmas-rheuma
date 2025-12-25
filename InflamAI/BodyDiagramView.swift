//
//  BodyDiagramView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import Foundation
#if canImport(UIKit)
import UIKit
#endif

// Temporary stub implementations to resolve compilation errors
class AIMLEngine: ObservableObject {
    static let shared = AIMLEngine()
    
    @Published var isAnalyzing = false
    @Published var predictions: [String] = []
    @Published var patterns: [String] = []
    @Published var insights: [String] = []
    @Published var predictedPainAreas: Set<BodyDiagramRegion> = []

    private init() {}

    func analyzePainPatterns(selectedRegions: Set<BodyDiagramRegion>) {
        // Stub implementation
        print("Analyzing pain patterns for regions: \(selectedRegions)")
    }

    func analyzeSpecificRegion(_ region: BodyDiagramRegion) {
        // Stub implementation
        print("Analyzing specific region: \(region)")
    }

    func suggestRelatedPainAreas(for region: BodyDiagramRegion) -> [BodyDiagramRegion] {
        // Stub implementation
        print("Suggesting related areas for: \(region)")
        return []
    }
}

class VoiceCommandEngine: NSObject, ObservableObject {
    static let shared = VoiceCommandEngine()
    
    @Published var isListening = false
    @Published var lastCommand = ""
    
    override init() {
        super.init()
    }
    
    func startListening() {
        // Stub implementation
        isListening = true
        print("Voice command listening started")
    }
    
    func stopListening() {
        // Stub implementation
        isListening = false
        print("Voice command listening stopped")
    }
    
    func announcePainSelection(region: BodyDiagramRegion, selected: Bool) {
        // Stub implementation
        print("Announcing pain selection: \(region) - \(selected ? "selected" : "deselected")")
    }
}

enum BodyDiagramRegion: String, CaseIterable {
    case head = "Head"
    case neck = "Neck"
    case leftShoulder = "Left Shoulder"
    case rightShoulder = "Right Shoulder"
    case leftArm = "Left Arm"
    case rightArm = "Right Arm"
    case leftForearm = "Left Forearm"
    case rightForearm = "Right Forearm"
    case leftElbow = "Left Elbow"
    case rightElbow = "Right Elbow"
    case leftWrist = "Left Wrist"
    case rightWrist = "Right Wrist"
    case leftHand = "Left Hand"
    case rightHand = "Right Hand"
    case chest = "Chest"
    case upperBack = "Upper Back"
    case lowerBack = "Lower Back"
    case abdomen = "Abdomen"
    case leftHip = "Left Hip"
    case rightHip = "Right Hip"
    case leftThigh = "Left Thigh"
    case rightThigh = "Right Thigh"
    case leftKnee = "Left Knee"
    case rightKnee = "Right Knee"
    case leftCalf = "Left Calf"
    case rightCalf = "Right Calf"
    case leftAnkle = "Left Ankle"
    case rightAnkle = "Right Ankle"
    case leftFoot = "Left Foot"
    case rightFoot = "Right Foot"
    
    // Enhanced Back View Pain Points
    case cervicalC1 = "Cervical C1"
    case cervicalC2 = "Cervical C2"
    case cervicalC3 = "Cervical C3"
    case cervicalC4 = "Cervical C4"
    case cervicalC5 = "Cervical C5"
    case cervicalC6 = "Cervical C6"
    case cervicalC7 = "Cervical C7"
    case thoracicT1 = "Thoracic T1"
    case thoracicT2 = "Thoracic T2"
    case thoracicT3 = "Thoracic T3"
    case thoracicT4 = "Thoracic T4"
    case thoracicT5 = "Thoracic T5"
    case thoracicT6 = "Thoracic T6"
    case thoracicT7 = "Thoracic T7"
    case thoracicT8 = "Thoracic T8"
    case thoracicT9 = "Thoracic T9"
    case thoracicT10 = "Thoracic T10"
    case thoracicT11 = "Thoracic T11"
    case thoracicT12 = "Thoracic T12"
    case lumbarL1 = "Lumbar L1"
    case lumbarL2 = "Lumbar L2"
    case lumbarL3 = "Lumbar L3"
    case lumbarL4 = "Lumbar L4"
    case lumbarL5 = "Lumbar L5"
    case sacralS1 = "Sacral S1"
    case sacralS2 = "Sacral S2"
    case sacralS3 = "Sacral S3"
    case sacralS4 = "Sacral S4"
    case sacralS5 = "Sacral S5"
    case leftShoulderBlade = "Left Shoulder Blade"
    case rightShoulderBlade = "Right Shoulder Blade"
    case leftKidneyArea = "Left Kidney Area"
    case rightKidneyArea = "Right Kidney Area"
    case tailbone = "Tailbone/Coccyx"
    case leftHipBack = "Left Hip (Back)"
    case rightHipBack = "Right Hip (Back)"
    case upperTrapezius = "Upper Trapezius"
    case middleTrapezius = "Middle Trapezius"
    case lowerTrapezius = "Lower Trapezius"
    case leftRhomboid = "Left Rhomboid"
    case rightRhomboid = "Right Rhomboid"
    case leftLatissimus = "Left Latissimus Dorsi"
    case rightLatissimus = "Right Latissimus Dorsi"
    case leftErectorSpinae = "Left Erector Spinae"
    case rightErectorSpinae = "Right Erector Spinae"
    case leftQuadratus = "Left Quadratus Lumborum"
    case rightQuadratus = "Right Quadratus Lumborum"
}

struct BodyDiagramView: View {
    @Binding var selectedRegions: Set<BodyDiagramRegion>
    @Binding var regionPainLevels: [BodyDiagramRegion: Int]  // Pain levels from parent
    @State private var showingFrontView = true
    @State private var painIntensity: [BodyDiagramRegion: Double] = [:]
    @State private var showingHeatMap = false
    @State private var enableVoiceCommands = false
    @State private var enableHapticFeedback = true
    @StateObject private var aiEngine = AIMLEngine.shared
    @StateObject private var voiceManager = VoiceCommandEngine.shared
    
    var body: some View {
        VStack(spacing: 20) {
            // Front/Back Toggle
            Picker("View", selection: $showingFrontView) {
                Text("Front").tag(true)
                Text("Back").tag(false)
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding(.horizontal)
            
            // Body Diagram
            ZStack {
                Rectangle()
                    .fill(Color.gray.opacity(0.2))
                    .frame(width: 200, height: 400)
                    .cornerRadius(20)
                
                if showingFrontView {
                    FrontBodyView(selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                } else {
                    BackBodyView(selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                }
            }
            .frame(height: 420)
            
            // Quick Selection Buttons
            VStack(spacing: 10) {
                Text("Quick Select")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 8) {
                    ForEach(["Joints", "Spine", "Hands"], id: \.self) { category in
                        Button(category) {
                            selectCategory(category)
                        }
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.blue.opacity(0.1))
                        .foregroundColor(.blue)
                        .cornerRadius(8)
                    }
                }
                
                Button("Clear All") {
                    selectedRegions.removeAll()
                    painIntensity.removeAll()
                    regionPainLevels.removeAll()
                }
                .font(.caption)
                .foregroundColor(.red)
                .padding(.top, 5)
            }
        }
        .onChange(of: painIntensity) { newValue in
            // Sync pain intensity to regionPainLevels for PainTrackingView
            for (region, intensity) in newValue {
                regionPainLevels[region] = Int(intensity)
            }
        }
    }
    
    private func selectCategory(_ category: String) {
        switch category {
        case "Joints":
            selectedRegions.formUnion([
                .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
                .leftWrist, .rightWrist, .leftHip, .rightHip,
                .leftKnee, .rightKnee, .leftAnkle, .rightAnkle
            ])
        case "Spine":
            selectedRegions.formUnion([.neck, .upperBack, .lowerBack])
        case "Hands":
            selectedRegions.formUnion([.leftHand, .rightHand, .leftWrist, .rightWrist])
        default:
            break
        }
    }
}

struct FrontBodyView: View {
    @Binding var selectedRegions: Set<BodyDiagramRegion>
    @Binding var painIntensity: [BodyDiagramRegion: Double]
    @Binding var showingHeatMap: Bool
    @Binding var enableHapticFeedback: Bool
    @StateObject private var aiEngine = AIMLEngine.shared
    
    var body: some View {
        ZStack {
            // Head
            EnhancedBodyPartButton(region: .head, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 40)
            
            // Neck
            EnhancedBodyPartButton(region: .neck, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 70)
            
            // Shoulders
            EnhancedBodyPartButton(region: .leftShoulder, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 70, y: 90)
            EnhancedBodyPartButton(region: .rightShoulder, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 130, y: 90)
            
            // Arms
            EnhancedBodyPartButton(region: .leftArm, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 50, y: 120)
            EnhancedBodyPartButton(region: .rightArm, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 150, y: 120)
            
            // Elbows
            EnhancedBodyPartButton(region: .leftElbow, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 45, y: 150)
            EnhancedBodyPartButton(region: .rightElbow, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 155, y: 150)
            
            // Forearms
            EnhancedBodyPartButton(region: .leftForearm, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 40, y: 180)
            EnhancedBodyPartButton(region: .rightForearm, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 160, y: 180)
            
            // Chest
            EnhancedBodyPartButton(region: .chest, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 120)
            
            // Abdomen
            EnhancedBodyPartButton(region: .abdomen, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 160)
            
            // Wrists
            EnhancedBodyPartButton(region: .leftWrist, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 35, y: 210)
            EnhancedBodyPartButton(region: .rightWrist, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 165, y: 210)
            
            // Hands
            EnhancedBodyPartButton(region: .leftHand, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 30, y: 240)
            EnhancedBodyPartButton(region: .rightHand, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 170, y: 240)
            
            // Hips
            EnhancedBodyPartButton(region: .leftHip, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 200)
            EnhancedBodyPartButton(region: .rightHip, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 200)
            
            // Thighs
            EnhancedBodyPartButton(region: .leftThigh, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 240)
            EnhancedBodyPartButton(region: .rightThigh, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 240)
            
            // Knees
            EnhancedBodyPartButton(region: .leftKnee, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 280)
            EnhancedBodyPartButton(region: .rightKnee, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 280)
            
            // Calves
            EnhancedBodyPartButton(region: .leftCalf, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 320)
            EnhancedBodyPartButton(region: .rightCalf, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 320)
            
            // Ankles
            EnhancedBodyPartButton(region: .leftAnkle, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 360)
            EnhancedBodyPartButton(region: .rightAnkle, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 360)
            
            // Feet
            EnhancedBodyPartButton(region: .leftFoot, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 390)
            EnhancedBodyPartButton(region: .rightFoot, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 390)
        }
        .frame(width: 200, height: 400)
        .background(
            // Heat map overlay
            showingHeatMap ? 
            HeatMapOverlay(painIntensity: painIntensity, selectedRegions: selectedRegions)
                .opacity(0.3) : nil
        )
        .onAppear {
            // Initialize AI-powered pain pattern recognition
            aiEngine.analyzePainPatterns(selectedRegions: selectedRegions)
        }
    }
}

struct BackBodyView: View {
    @Binding var selectedRegions: Set<BodyDiagramRegion>
    @Binding var painIntensity: [BodyDiagramRegion: Double]
    @Binding var showingHeatMap: Bool
    @Binding var enableHapticFeedback: Bool
    @StateObject private var aiEngine = AIMLEngine.shared
    
    var body: some View {
        ZStack {
            // Head (Posterior)
            EnhancedBodyPartButton(region: .head, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 30)
            
            // Neck (Posterior)
            EnhancedBodyPartButton(region: .neck, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 40)
            
            // Cervical Spine (C1-C7)
            EnhancedBodyPartButton(region: .cervicalC1, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 45)
            EnhancedBodyPartButton(region: .cervicalC2, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 50)
            EnhancedBodyPartButton(region: .cervicalC3, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 55)
            EnhancedBodyPartButton(region: .cervicalC4, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 60)
            EnhancedBodyPartButton(region: .cervicalC5, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 65)
            EnhancedBodyPartButton(region: .cervicalC6, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 70)
            EnhancedBodyPartButton(region: .cervicalC7, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 75)
            
            // Shoulder Blades
            EnhancedBodyPartButton(region: .leftShoulderBlade, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 130, y: 95)
            EnhancedBodyPartButton(region: .rightShoulderBlade, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 70, y: 95)
            
            // Trapezius Muscles
            EnhancedBodyPartButton(region: .upperTrapezius, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 85)
            EnhancedBodyPartButton(region: .middleTrapezius, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 105)
            EnhancedBodyPartButton(region: .lowerTrapezius, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 125)
            
            // Rhomboid Muscles
            EnhancedBodyPartButton(region: .leftRhomboid, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 115, y: 110)
            EnhancedBodyPartButton(region: .rightRhomboid, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 85, y: 110)
            
            // Thoracic Spine (T1-T12)
            EnhancedBodyPartButton(region: .thoracicT1, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 80)
            EnhancedBodyPartButton(region: .thoracicT2, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 85)
            EnhancedBodyPartButton(region: .thoracicT3, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 90)
            EnhancedBodyPartButton(region: .thoracicT4, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 95)
            EnhancedBodyPartButton(region: .thoracicT5, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 100)
            EnhancedBodyPartButton(region: .thoracicT6, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 105)
            EnhancedBodyPartButton(region: .thoracicT7, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 110)
            EnhancedBodyPartButton(region: .thoracicT8, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 115)
            EnhancedBodyPartButton(region: .thoracicT9, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 120)
            EnhancedBodyPartButton(region: .thoracicT10, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 125)
            EnhancedBodyPartButton(region: .thoracicT11, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 130)
            EnhancedBodyPartButton(region: .thoracicT12, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 135)
            
            // Latissimus Dorsi
            EnhancedBodyPartButton(region: .leftLatissimus, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 140, y: 130)
            EnhancedBodyPartButton(region: .rightLatissimus, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 60, y: 130)
            
            // Kidney Areas
            EnhancedBodyPartButton(region: .leftKidneyArea, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 150)
            EnhancedBodyPartButton(region: .rightKidneyArea, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 150)
            
            // Lumbar Spine (L1-L5)
            EnhancedBodyPartButton(region: .lumbarL1, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 140)
            EnhancedBodyPartButton(region: .lumbarL2, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 150)
            EnhancedBodyPartButton(region: .lumbarL3, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 160)
            EnhancedBodyPartButton(region: .lumbarL4, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 170)
            EnhancedBodyPartButton(region: .lumbarL5, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 180)
            
            // Erector Spinae
            EnhancedBodyPartButton(region: .leftErectorSpinae, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 110, y: 155)
            EnhancedBodyPartButton(region: .rightErectorSpinae, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 90, y: 155)
            
            // Quadratus Lumborum
            EnhancedBodyPartButton(region: .leftQuadratus, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 115, y: 170)
            EnhancedBodyPartButton(region: .rightQuadratus, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 85, y: 170)
            
            // Sacral Region (S1-S5)
            EnhancedBodyPartButton(region: .sacralS1, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 190)
            EnhancedBodyPartButton(region: .sacralS2, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 195)
            EnhancedBodyPartButton(region: .sacralS3, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 200)
            EnhancedBodyPartButton(region: .sacralS4, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 205)
            EnhancedBodyPartButton(region: .sacralS5, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 210)
            
            // Tailbone/Coccyx
            EnhancedBodyPartButton(region: .tailbone, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 100, y: 220)
            
            // Hip Areas (Back View)
            EnhancedBodyPartButton(region: .leftHipBack, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 200)
            EnhancedBodyPartButton(region: .rightHipBack, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 200)
            
            // Posterior Shoulders
            EnhancedBodyPartButton(region: .leftShoulder, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 130, y: 85)
            EnhancedBodyPartButton(region: .rightShoulder, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 70, y: 85)
            
            // Arms (Posterior)
            EnhancedBodyPartButton(region: .leftArm, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 150, y: 120)
            EnhancedBodyPartButton(region: .rightArm, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 50, y: 120)
            
            // Elbows (Posterior)
            EnhancedBodyPartButton(region: .leftElbow, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 155, y: 150)
            EnhancedBodyPartButton(region: .rightElbow, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 45, y: 150)
            
            // Forearms (Posterior)
            EnhancedBodyPartButton(region: .leftForearm, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 160, y: 180)
            EnhancedBodyPartButton(region: .rightForearm, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 40, y: 180)
            
            // Wrists (Posterior)
            EnhancedBodyPartButton(region: .leftWrist, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 165, y: 210)
            EnhancedBodyPartButton(region: .rightWrist, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 35, y: 210)
            
            // Hands (Posterior)
            EnhancedBodyPartButton(region: .leftHand, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 170, y: 240)
            EnhancedBodyPartButton(region: .rightHand, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 30, y: 240)
            
            // Posterior Thighs (Hamstrings)
            EnhancedBodyPartButton(region: .leftThigh, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 240)
            EnhancedBodyPartButton(region: .rightThigh, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 240)
            
            // Posterior Knees
            EnhancedBodyPartButton(region: .leftKnee, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 280)
            EnhancedBodyPartButton(region: .rightKnee, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 280)
            
            // Posterior Calves
            EnhancedBodyPartButton(region: .leftCalf, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 320)
            EnhancedBodyPartButton(region: .rightCalf, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 320)
            
            // Posterior Ankles
            EnhancedBodyPartButton(region: .leftAnkle, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 360)
            EnhancedBodyPartButton(region: .rightAnkle, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 360)
            
            // Posterior Feet
            EnhancedBodyPartButton(region: .leftFoot, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 120, y: 390)
            EnhancedBodyPartButton(region: .rightFoot, selectedRegions: $selectedRegions, painIntensity: $painIntensity, showingHeatMap: $showingHeatMap, enableHapticFeedback: $enableHapticFeedback)
                .position(x: 80, y: 390)
        }
        .frame(width: 200, height: 400)
        .background(
            // Heat map overlay
            showingHeatMap ? 
            HeatMapOverlay(painIntensity: painIntensity, selectedRegions: selectedRegions)
                .opacity(0.3) : nil
        )
        .onAppear {
            // Initialize AI-powered pain pattern recognition
            aiEngine.analyzePainPatterns(selectedRegions: selectedRegions)
        }
    }
}

struct BodyPartButton: View {
    let region: BodyDiagramRegion
    @Binding var selectedRegions: Set<BodyDiagramRegion>
    
    private var isSelected: Bool {
        selectedRegions.contains(region)
    }
    
    var body: some View {
        Button(action: {
            if isSelected {
                selectedRegions.remove(region)
            } else {
                selectedRegions.insert(region)
            }
        }) {
            Circle()
                .fill(isSelected ? Color.red.opacity(0.8) : Color.blue.opacity(0.3))
                .frame(width: 20, height: 20)
                .overlay(
                    Circle()
                        .stroke(isSelected ? Color.red : Color.blue, lineWidth: 2)
                )
                .scaleEffect(isSelected ? 1.2 : 1.0)
                .animation(.easeInOut(duration: 0.2), value: isSelected)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// Enhanced Body Part Button with advanced features
struct EnhancedBodyPartButton: View {
    let region: BodyDiagramRegion
    @Binding var selectedRegions: Set<BodyDiagramRegion>
    @Binding var painIntensity: [BodyDiagramRegion: Double]
    @Binding var showingHeatMap: Bool
    @Binding var enableHapticFeedback: Bool
    
    @State private var isPulsing = false
    @State private var showingPainHistory = false
    @State private var showingPainPicker = false
    @StateObject private var aiEngine = AIMLEngine.shared
    @StateObject private var voiceEngine = VoiceCommandEngine.shared
    
    private var currentIntensity: Double {
        painIntensity[region] ?? 0.0
    }
    
    private var buttonColor: Color {
        if selectedRegions.contains(region) {
            // Color based on pain intensity
            switch currentIntensity {
            case 0.0..<2.0:
                return Color.green
            case 2.0..<4.0:
                return Color.yellow
            case 4.0..<6.0:
                return Color.orange
            case 6.0..<8.0:
                return Color.red
            case 8.0...10.0:
                return Color.purple
            default:
                return Color.blue
            }
        } else {
            return showingHeatMap ? Color.gray.opacity(0.3) : Color.blue.opacity(0.6)
        }
    }
    
    private var painColor: Color {
        let intensity = currentIntensity
        switch intensity {
        case 0:
            return .gray.opacity(0.3)
        case 0.1...1:
            return Color(red: 0.6, green: 0.9, blue: 0.6) // Light green
        case 1.1...2:
            return Color(red: 0.4, green: 0.8, blue: 0.4) // Green
        case 2.1...3:
            return Color(red: 0.8, green: 0.9, blue: 0.3) // Yellow-green
        case 3.1...4:
            return Color(red: 0.9, green: 0.8, blue: 0.2) // Yellow
        case 4.1...5:
            return Color(red: 0.9, green: 0.6, blue: 0.2) // Orange-yellow
        case 5.1...6:
            return Color(red: 0.9, green: 0.4, blue: 0.2) // Orange
        case 6.1...7:
            return Color(red: 0.9, green: 0.2, blue: 0.2) // Red-orange
        case 7.1...8:
            return Color(red: 0.8, green: 0.1, blue: 0.1) // Red
        case 8.1...9:
            return Color(red: 0.6, green: 0.1, blue: 0.4) // Dark red
        case 9.1...10:
            return Color(red: 0.4, green: 0.1, blue: 0.6) // Purple
        default:
            return .gray.opacity(0.3)
        }
    }
    
    private var painIntensityDescription: String {
        let intensity = currentIntensity
        switch intensity {
        case 0:
            return "No Pain"
        case 0.1...2:
            return "Mild"
        case 2.1...4:
            return "Moderate"
        case 4.1...6:
            return "Moderate-Severe"
        case 6.1...8:
            return "Severe"
        case 8.1...10:
            return "Extreme"
        default:
            return "Unknown"
        }
    }
    
    private var buttonSize: CGFloat {
        let baseSize: CGFloat = 12
        let intensityMultiplier = 1.0 + (currentIntensity / 10.0)
        return baseSize * intensityMultiplier
    }
    
    var body: some View {
        Button(action: {
            handleSelection()
        }) {
            ZStack {
                // Main pain bubble
                Circle()
                    .fill(
                        RadialGradient(
                            gradient: Gradient(colors: [
                                buttonColor.opacity(0.8),
                                buttonColor.opacity(0.4)
                            ]),
                            center: .center,
                            startRadius: 2,
                            endRadius: buttonSize / 2
                        )
                    )
                    .frame(width: buttonSize, height: buttonSize)
                    .overlay(
                        Circle()
                            .stroke(
                                selectedRegions.contains(region) ? Color.white : Color.gray,
                                lineWidth: selectedRegions.contains(region) ? 2 : 1
                            )
                    )
                    .scaleEffect(isPulsing ? 1.2 : 1.0)
                    .animation(
                        selectedRegions.contains(region) && currentIntensity > 5.0 ?
                        Animation.easeInOut(duration: 0.8).repeatForever(autoreverses: true) :
                        Animation.easeInOut(duration: 0.2),
                        value: isPulsing
                    )
                
                // Pain intensity indicator
                if selectedRegions.contains(region) && currentIntensity > 0 {
                    VStack(spacing: 1) {
                        Text("\(String(format: "%.1f", currentIntensity))")
                            .font(.system(size: 6, weight: .bold))
                            .foregroundColor(.white)
                            .shadow(color: .black, radius: 1)
                        
                        // Intensity level indicator dots
                        HStack(spacing: 1) {
                            ForEach(1...Int(min(currentIntensity, 10)), id: \.self) { _ in
                                Circle()
                                    .fill(Color.white)
                                    .frame(width: 1, height: 1)
                            }
                        }
                    }
                }
                
                // Glow effect for high pain levels
                if selectedRegions.contains(region) && currentIntensity > 7 {
                    Circle()
                        .fill(painColor.opacity(0.3))
                        .frame(width: buttonSize * 1.8, height: buttonSize * 1.8)
                        .blur(radius: 3)
                        .allowsHitTesting(false)
                }
                
                // AI prediction indicator
                if aiEngine.predictedPainAreas.contains(region) {
                    Circle()
                        .stroke(Color.cyan, lineWidth: 2)
                        .frame(width: buttonSize + 4, height: buttonSize + 4)
                        .opacity(0.7)
                }
                
                // Heat map overlay
                if showingHeatMap {
                    Circle()
                        .fill(
                            RadialGradient(
                                gradient: Gradient(colors: [
                                    Color.red.opacity(currentIntensity / 10.0),
                                    Color.clear
                                ]),
                                center: .center,
                                startRadius: 0,
                                endRadius: buttonSize
                            )
                        )
                        .frame(width: buttonSize * 2, height: buttonSize * 2)
                        .allowsHitTesting(false)
                }
            }
        }
        .buttonStyle(PlainButtonStyle())
        .onAppear {
            if selectedRegions.contains(region) && currentIntensity > 5.0 {
                isPulsing = true
            }
        }
        .onChange(of: selectedRegions) { _ in
            updatePulsingState()
        }
        .onChange(of: currentIntensity) { _ in
            updatePulsingState()
        }
        .contextMenu {
            Button("Set Pain Intensity") {
                showingPainPicker = true
            }
            Button("View Pain History") {
                showingPainHistory = true
            }
            Button("AI Analysis") {
                aiEngine.analyzeSpecificRegion(region)
            }
            if selectedRegions.contains(region) {
                Button("Remove Pain Point", role: .destructive) {
                    selectedRegions.remove(region)
                    painIntensity[region] = 0.0
                }
            }
        }
        .sheet(isPresented: $showingPainHistory) {
            PainHistoryView(region: region)
        }
        .sheet(isPresented: $showingPainPicker) {
            PainIntensityPicker(
                region: region,
                currentIntensity: Binding(
                    get: { painIntensity[region] ?? 0.0 },
                    set: { newValue in
                        if newValue > 0 {
                            selectedRegions.insert(region)
                            painIntensity[region] = newValue
                        } else {
                            selectedRegions.remove(region)
                            painIntensity[region] = 0.0
                        }
                    }
                ),
                selectedRegions: $selectedRegions
            )
        }
    }
    
    private func handleSelection() {
        // Haptic feedback
        if enableHapticFeedback {
            #if canImport(UIKit)
            let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
            impactFeedback.impactOccurred()
            #endif
        }
        
        // Show pain picker for setting intensity
        showingPainPicker = true
        
        // If not already selected, add to selection with default intensity
        if !selectedRegions.contains(region) {
            selectedRegions.insert(region)
            painIntensity[region] = 5.0 // Default intensity
        }
        
        // Voice feedback
        voiceEngine.announcePainSelection(region: region, selected: selectedRegions.contains(region))
        
        // AI analysis
        aiEngine.analyzePainPatterns(selectedRegions: selectedRegions)
        
        // Auto-suggest related areas
        if selectedRegions.contains(region) {
            let _ = aiEngine.suggestRelatedPainAreas(for: region)
            // Could show suggestions to user
        }
    }
    
    private func updatePulsingState() {
        isPulsing = selectedRegions.contains(region) && currentIntensity > 5.0
    }
}

// Heat Map Overlay Component
struct HeatMapOverlay: View {
    let painIntensity: [BodyDiagramRegion: Double]
    let selectedRegions: Set<BodyDiagramRegion>
    
    var body: some View {
        Canvas { context, size in
            for region in selectedRegions {
                let intensity = painIntensity[region] ?? 0.0
                if intensity > 0 {
                    let position = getRegionPosition(region, in: size)
                    let radius = 30.0 + (intensity * 5.0)
                    let opacity = intensity / 10.0
                    
                    let gradient = Gradient(colors: [
                        Color.red.opacity(opacity),
                        Color.orange.opacity(opacity * 0.7),
                        Color.yellow.opacity(opacity * 0.4),
                        Color.clear
                    ])
                    
                    context.fill(
                        Path(ellipseIn: CGRect(
                            x: position.x - radius,
                            y: position.y - radius,
                            width: radius * 2,
                            height: radius * 2
                        )),
                        with: .radialGradient(
                            gradient,
                            center: position,
                            startRadius: 0,
                            endRadius: radius
                        )
                    )
                }
            }
        }
    }
    
    private func getRegionPosition(_ region: BodyDiagramRegion, in size: CGSize) -> CGPoint {
        // This would map each region to its position in the view
        // For now, returning a default position
        return CGPoint(x: size.width / 2, y: size.height / 2)
    }
}

// Pain History View
struct PainHistoryView: View {
    let region: BodyDiagramRegion
    @Environment(\.dismiss) private var dismiss
    @StateObject private var aiEngine = AIMLEngine.shared
    
    var body: some View {
        NavigationView {
            VStack {
                Text("Pain History for \(region.rawValue)")
                    .font(.title2)
                    .padding()
                
                // Pain history chart would go here
                Text("Historical pain data and trends")
                    .foregroundColor(.secondary)
                
                Spacer()
            }
            .navigationTitle("Pain History")
            #if !os(macOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// Pain Intensity Picker Component
 struct PainIntensityPicker: View {
     let region: BodyDiagramRegion
     @Binding var currentIntensity: Double
     @Binding var selectedRegions: Set<BodyDiagramRegion>
     @Environment(\.dismiss) private var dismiss
     
     private var painColor: Color {
         let intensity = currentIntensity
         switch intensity {
         case 0:
             return .gray.opacity(0.3)
         case 0.1...1:
             return Color(red: 0.6, green: 0.9, blue: 0.6)
         case 1.1...2:
             return Color(red: 0.4, green: 0.8, blue: 0.4)
         case 2.1...3:
             return Color(red: 0.8, green: 0.9, blue: 0.3)
         case 3.1...4:
             return Color(red: 0.9, green: 0.8, blue: 0.2)
         case 4.1...5:
             return Color(red: 0.9, green: 0.6, blue: 0.2)
         case 5.1...6:
             return Color(red: 0.9, green: 0.4, blue: 0.2)
         case 6.1...7:
             return Color(red: 0.9, green: 0.2, blue: 0.2)
         case 7.1...8:
             return Color(red: 0.8, green: 0.1, blue: 0.1)
         case 8.1...9:
             return Color(red: 0.6, green: 0.1, blue: 0.4)
         case 9.1...10:
             return Color(red: 0.4, green: 0.1, blue: 0.6)
         default:
             return .gray.opacity(0.3)
         }
     }
     
     private var painDescription: String {
         let intensity = currentIntensity
         switch intensity {
         case 0:
             return "No Pain"
         case 0.1...2:
             return "Mild Pain"
         case 2.1...4:
             return "Moderate Pain"
         case 4.1...6:
             return "Moderate-Severe Pain"
         case 6.1...8:
             return "Severe Pain"
         case 8.1...10:
             return "Extreme Pain"
         default:
             return "Unknown"
         }
     }
     
     var body: some View {
         NavigationView {
             VStack(spacing: 20) {
                 Text("Pain Intensity for \(region.rawValue)")
                     .font(.title2)
                     .padding()
                 
                 // Visual pain indicator
                 VStack(spacing: 10) {
                     Circle()
                         .fill(painColor)
                         .frame(width: 60, height: 60)
                         .overlay(
                             Circle()
                                 .stroke(Color.white, lineWidth: 3)
                         )
                         .shadow(radius: 5)
                     
                     Text("\(String(format: "%.1f", currentIntensity))")
                         .font(.title)
                         .fontWeight(.bold)
                     
                     Text(painDescription)
                         .font(.headline)
                         .foregroundColor(painColor)
                 }
                 
                 VStack(spacing: 15) {
                     Slider(value: $currentIntensity, in: 0...10, step: 0.5) {
                         Text("Pain Level")
                     } minimumValueLabel: {
                         Text("0")
                             .font(.caption)
                     } maximumValueLabel: {
                         Text("10")
                             .font(.caption)
                     }
                     .accentColor(painColor)
                     .padding(.horizontal)
                     
                     HStack {
                         Text("No Pain")
                             .font(.caption)
                             .foregroundColor(.green)
                         Spacer()
                         Text("Extreme Pain")
                             .font(.caption)
                             .foregroundColor(.purple)
                     }
                     .padding(.horizontal)
                 }
                 .padding()
                 
                 // Quick selection buttons
                 VStack(spacing: 10) {
                     Text("Quick Select:")
                         .font(.subheadline)
                         .foregroundColor(.secondary)
                     
                     HStack(spacing: 10) {
                         ForEach([1.0, 3.0, 5.0, 7.0, 9.0], id: \.self) { level in
                             Button("\(Int(level))") {
                                 currentIntensity = level
                             }
                             .frame(width: 40, height: 40)
                             .background(getColorForLevel(level))
                             .foregroundColor(.white)
                             .clipShape(Circle())
                             .shadow(radius: 2)
                         }
                     }
                 }
                 
                 Button("Remove Pain Point") {
                     currentIntensity = 0.0
                     dismiss()
                 }
                 .foregroundColor(.red)
                 .padding()
                 
                 Spacer()
             }
             .navigationTitle("Pain Level")
             .navigationBarTitleDisplayMode(.inline)
             .toolbar {
                 ToolbarItem(placement: .primaryAction) {
                     Button("Done") {
                         dismiss()
                     }
                 }
             }
         }
     }
     
     private func getColorForLevel(_ level: Double) -> Color {
         switch level {
         case 1:
             return Color(red: 0.4, green: 0.8, blue: 0.4)
         case 3:
             return Color(red: 0.9, green: 0.8, blue: 0.2)
         case 5:
             return Color(red: 0.9, green: 0.4, blue: 0.2)
         case 7:
             return Color(red: 0.9, green: 0.2, blue: 0.2)
         case 9:
             return Color(red: 0.4, green: 0.1, blue: 0.6)
         default:
             return .gray
         }
     }
 }

struct BodyDiagramView_Previews: PreviewProvider {
    static var previews: some View {
        BodyDiagramView(selectedRegions: .constant(Set([.leftKnee, .rightKnee])), regionPainLevels: .constant([:]))
            .padding()
    }
}