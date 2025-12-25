//
//  AdvancedPainTrackingView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import SwiftUI
import CoreHaptics
import AVFoundation

struct AdvancedPainTrackingView: View {
    @StateObject private var aiEngine = AIMLEngine.shared
    @StateObject private var voiceEngine = VoiceCommandEngine.shared
    @StateObject private var healthMonitor = RealTimeHealthMonitor.shared
    @StateObject private var arModule = ARBodyScanningModule.shared
    
    @State private var selectedRegions: Set<BodyRegion> = []
    @State private var painIntensity: [BodyRegion: Double] = [:]
    @State private var showingHeatMap = false
    @State private var enableHapticFeedback = true
    @State private var showingARScanner = false
    @State private var showingPredictiveAnalysis = false
    @State private var showingVoiceCommands = false
    @State private var currentPainLevel: Double = 0.0
    @State private var painDescription = ""
    @State private var selectedTimeRange: TimeRange = .today
    @State private var showingPainHistory = false
    @State private var showingMedicationCorrelation = false
    @State private var isRecordingVoice = false
    
    // Advanced features
    @State private var painTriggers: [String] = []
    @State private var weatherCorrelation = false
    @State private var stressCorrelation = false
    @State private var sleepCorrelation = false
    @State private var activityCorrelation = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header with AI insights
                    PainTrackingHeaderView(
                        aiInsights: aiEngine.currentPainInsights,
                        predictedPainLevel: aiEngine.predictedPainLevel
                    )
                    
                    // Enhanced Body Diagram
                    VStack {
                        Text("Select Pain Areas")
                            .font(.headline)
                            .foregroundColor(.primary)
                        
                        BodyDiagramView(
                            selectedRegions: $selectedRegions,
                            painIntensity: $painIntensity,
                            showingHeatMap: $showingHeatMap,
                            enableHapticFeedback: $enableHapticFeedback
                        )
                        .frame(height: 450)
                        .background(
                            RoundedRectangle(cornerRadius: 15)
                                .fill(Color(.systemBackground))
                                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
                        )
                    }
                    
                    // Advanced Control Panel
                    AdvancedControlPanelView(
                        showingHeatMap: $showingHeatMap,
                        enableHapticFeedback: $enableHapticFeedback,
                        showingARScanner: $showingARScanner,
                        showingPredictiveAnalysis: $showingPredictiveAnalysis,
                        showingVoiceCommands: $showingVoiceCommands
                    )
                    
                    // Pain Intensity Slider with AI Enhancement
                    PainIntensityControlView(
                        currentPainLevel: $currentPainLevel,
                        selectedRegions: selectedRegions,
                        painIntensity: $painIntensity
                    )
                    
                    // Voice Command Integration
                    if showingVoiceCommands {
                        VoiceCommandPanelView(
                            isRecording: $isRecordingVoice,
                            painDescription: $painDescription,
                            selectedRegions: $selectedRegions
                        )
                        .transition(.slide)
                    }
                    
                    // AR Body Scanner
                    if showingARScanner {
                        ARBodyScannerView(
                            selectedRegions: $selectedRegions,
                            painIntensity: $painIntensity
                        )
                        .frame(height: 300)
                        .transition(.opacity)
                    }
                    
                    // Predictive Analysis Panel
                    if showingPredictiveAnalysis {
                        PredictiveAnalysisView(
                            selectedTimeRange: $selectedTimeRange,
                            painTriggers: $painTriggers,
                            correlationFactors: CorrelationFactors(
                                weather: weatherCorrelation,
                                stress: stressCorrelation,
                                sleep: sleepCorrelation,
                                activity: activityCorrelation
                            )
                        )
                        .transition(.move(edge: .bottom))
                    }
                    
                    // Quick Action Buttons
                    QuickActionButtonsView(
                        showingPainHistory: $showingPainHistory,
                        showingMedicationCorrelation: $showingMedicationCorrelation,
                        selectedRegions: selectedRegions,
                        currentPainLevel: currentPainLevel
                    )
                    
                    // Smart Suggestions
                    SmartSuggestionsView(
                        selectedRegions: selectedRegions,
                        currentPainLevel: currentPainLevel,
                        aiSuggestions: aiEngine.painManagementSuggestions
                    )
                }
                .padding()
            }
            .navigationTitle("Advanced Pain Tracking")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button("Export Pain Data") {
                            exportPainData()
                        }
                        Button("Share with Doctor") {
                            shareWithDoctor()
                        }
                        Button("AI Analysis Report") {
                            generateAIReport()
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
        .onAppear {
            initializeAdvancedFeatures()
        }
        .sheet(isPresented: $showingPainHistory) {
            PainHistoryDetailView(selectedRegions: selectedRegions)
        }
        .sheet(isPresented: $showingMedicationCorrelation) {
            MedicationCorrelationView(painData: generatePainData())
        }
    }
    
    // MARK: - Helper Methods
    
    private func initializeAdvancedFeatures() {
        // Initialize AI engine with current pain data
        aiEngine.initializePainTracking()
        
        // Setup voice commands
        voiceEngine.setupPainTrackingCommands()
        
        // Initialize health monitoring
        healthMonitor.startPainMonitoring()
        
        // Setup AR scanning
        arModule.initializeBodyScanning()
    }
    
    private func exportPainData() {
        // Export comprehensive pain data
        let painData = PainExportData(
            selectedRegions: selectedRegions,
            painIntensity: painIntensity,
            timestamp: Date(),
            aiInsights: aiEngine.currentPainInsights,
            correlationData: generateCorrelationData()
        )
        
        // Export logic here
    }
    
    private func shareWithDoctor() {
        // Generate doctor-friendly report
        let report = aiEngine.generateDoctorReport(
            painData: generatePainData(),
            timeRange: selectedTimeRange
        )
        
        // Share logic here
    }
    
    private func generateAIReport() {
        // Generate comprehensive AI analysis
        aiEngine.generateComprehensiveReport(
            painData: generatePainData(),
            includePredicitions: true,
            includeTreatmentSuggestions: true
        )
    }
    
    private func generatePainData() -> PainTrackingData {
        return PainTrackingData(
            selectedRegions: selectedRegions,
            painIntensity: painIntensity,
            currentLevel: currentPainLevel,
            description: painDescription,
            timestamp: Date(),
            triggers: painTriggers,
            correlationFactors: CorrelationFactors(
                weather: weatherCorrelation,
                stress: stressCorrelation,
                sleep: sleepCorrelation,
                activity: activityCorrelation
            )
        )
    }
    
    private func generateCorrelationData() -> CorrelationData {
        return CorrelationData(
            weatherImpact: weatherCorrelation ? 0.7 : 0.0,
            stressImpact: stressCorrelation ? 0.8 : 0.0,
            sleepImpact: sleepCorrelation ? 0.6 : 0.0,
            activityImpact: activityCorrelation ? 0.5 : 0.0,
            medicationEffectiveness: healthMonitor.medicationEffectiveness
        )
    }
}

// MARK: - Supporting Views

struct PainTrackingHeaderView: View {
    let aiInsights: String
    let predictedPainLevel: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.blue)
                Text("AI Pain Insights")
                    .font(.headline)
                Spacer()
                Text("Predicted: \(Int(predictedPainLevel))/10")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.orange.opacity(0.2))
                    .cornerRadius(8)
            }
            
            Text(aiInsights)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.leading)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.blue.opacity(0.1))
        )
    }
}

struct AdvancedControlPanelView: View {
    @Binding var showingHeatMap: Bool
    @Binding var enableHapticFeedback: Bool
    @Binding var showingARScanner: Bool
    @Binding var showingPredictiveAnalysis: Bool
    @Binding var showingVoiceCommands: Bool
    
    var body: some View {
        VStack(spacing: 15) {
            Text("Advanced Features")
                .font(.headline)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 10) {
                ControlToggleButton(
                    title: "Heat Map",
                    icon: "thermometer",
                    isOn: $showingHeatMap,
                    color: .red
                )
                
                ControlToggleButton(
                    title: "Haptic Feedback",
                    icon: "iphone.radiowaves.left.and.right",
                    isOn: $enableHapticFeedback,
                    color: .purple
                )
                
                ControlToggleButton(
                    title: "AR Scanner",
                    icon: "camera.viewfinder",
                    isOn: $showingARScanner,
                    color: .green
                )
                
                ControlToggleButton(
                    title: "AI Predictions",
                    icon: "brain",
                    isOn: $showingPredictiveAnalysis,
                    color: .blue
                )
                
                ControlToggleButton(
                    title: "Voice Commands",
                    icon: "mic",
                    isOn: $showingVoiceCommands,
                    color: .orange
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 3, x: 0, y: 1)
        )
    }
}

struct ControlToggleButton: View {
    let title: String
    let icon: String
    @Binding var isOn: Bool
    let color: Color
    
    var body: some View {
        Button(action: {
            withAnimation(.spring()) {
                isOn.toggle()
            }
        }) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(isOn ? .white : color)
                
                Text(title)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(isOn ? .white : .primary)
            }
            .frame(height: 60)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(isOn ? color : Color(.systemGray6))
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Data Models

struct PainTrackingData {
    let selectedRegions: Set<BodyRegion>
    let painIntensity: [BodyRegion: Double]
    let currentLevel: Double
    let description: String
    let timestamp: Date
    let triggers: [String]
    let correlationFactors: CorrelationFactors
}

struct CorrelationFactors {
    let weather: Bool
    let stress: Bool
    let sleep: Bool
    let activity: Bool
}

struct CorrelationData {
    let weatherImpact: Double
    let stressImpact: Double
    let sleepImpact: Double
    let activityImpact: Double
    let medicationEffectiveness: Double
}

struct PainExportData {
    let selectedRegions: Set<BodyRegion>
    let painIntensity: [BodyRegion: Double]
    let timestamp: Date
    let aiInsights: String
    let correlationData: CorrelationData
}

enum TimeRange: String, CaseIterable {
    case today = "Today"
    case week = "This Week"
    case month = "This Month"
    case quarter = "3 Months"
    case year = "This Year"
}