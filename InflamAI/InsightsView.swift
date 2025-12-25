//
//  InsightsView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData

struct InsightsView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @StateObject private var insightsEngine = HealthInsightsEngine.shared
    
    @State private var insights: [HealthInsightsEngine.PatternInsight] = []
    @State private var correlations: [HealthInsightsEngine.CorrelationResult] = []
    @State private var medicationEffects: [HealthInsightsEngine.MedicationEffect] = []
    @State private var isLoading = true
    @State private var selectedTab = 0
    @State private var showingExportSheet = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Tab Selector
                Picker("Insights Type", selection: $selectedTab) {
                    Text("Patterns").tag(0)
                    Text("Correlations").tag(1)
                    Text("Medications").tag(2)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                
                if isLoading {
                    LoadingView()
                } else {
                    TabView(selection: $selectedTab) {
                        // Patterns Tab
                        PatternsTabView(insights: insights)
                            .tag(0)
                        
                        // Correlations Tab
                        CorrelationsTabView(correlations: correlations)
                            .tag(1)
                        
                        // Medications Tab
                        MedicationsTabView(medicationEffects: medicationEffects)
                            .tag(2)
                    }
                    .tabViewStyle(PageTabViewStyle(indexDisplayMode: .never))
                }
            }
            .navigationTitle("Health Insights")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button("Refresh Analysis") {
                            refreshInsights()
                        }
                        Button("Export Insights") {
                            showingExportSheet = true
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .sheet(isPresented: $showingExportSheet) {
                ExportInsightsView(
                    insights: insights,
                    correlations: correlations,
                    medicationEffects: medicationEffects
                )
            }
        }
        .onAppear {
            refreshInsights()
        }
    }
    
    private func refreshInsights() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let newInsights = insightsEngine.analyzeHealthData(context: viewContext)
            let newCorrelations = insightsEngine.calculateCorrelations(context: viewContext)
            let newMedicationEffects = insightsEngine.analyzeMedicationEffectiveness(context: viewContext)
            
            DispatchQueue.main.async {
                self.insights = newInsights
                self.correlations = newCorrelations
                self.medicationEffects = newMedicationEffects
                self.isLoading = false
            }
        }
    }
}

// MARK: - Patterns Tab

struct PatternsTabView: View {
    let insights: [HealthInsightsEngine.PatternInsight]
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                if insights.isEmpty {
                    EmptyInsightsView(type: "patterns")
                } else {
                    ForEach(insights, id: \.id) { insight in
                        PatternInsightCard(insight: insight)
                    }
                }
            }
            .padding()
        }
    }
}

struct PatternInsightCard: View {
    let insight: HealthInsightsEngine.PatternInsight
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(insight.title)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text(insight.description)
                        .font(.body)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                
                Spacer()
                
                VStack {
                    ConfidenceIndicator(confidence: insight.confidence)
                    InsightTypeIcon(type: insight.type)
                }
            }
            
            if let recommendation = insight.recommendation {
                HStack {
                    Image(systemName: "lightbulb.fill")
                        .foregroundColor(.orange)
                        .font(.caption)
                    
                    Text(recommendation)
                        .font(.caption)
                        .foregroundColor(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 4)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

// MARK: - Correlations Tab

struct CorrelationsTabView: View {
    let correlations: [HealthInsightsEngine.CorrelationResult]
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                if correlations.isEmpty {
                    EmptyInsightsView(type: "correlations")
                } else {
                    ForEach(correlations, id: \.id) { correlation in
                        CorrelationCard(correlation: correlation)
                    }
                }
            }
            .padding()
        }
    }
}

struct CorrelationCard: View {
    let correlation: HealthInsightsEngine.CorrelationResult
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("\(correlation.factor1) ↔ \(correlation.factor2)")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text(correlation.description)
                        .font(.body)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    CorrelationStrengthBadge(strength: correlation.strength)
                    Text("r = \(String(format: "%.2f", correlation.correlation))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            HStack {
                Label("\(correlation.sampleSize) data points", systemImage: "chart.bar")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                ConfidenceIndicator(confidence: correlation.confidence)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

// MARK: - Medications Tab

struct MedicationsTabView: View {
    let medicationEffects: [HealthInsightsEngine.MedicationEffect]
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                if medicationEffects.isEmpty {
                    EmptyInsightsView(type: "medication effects")
                } else {
                    ForEach(medicationEffects, id: \.id) { effect in
                        MedicationEffectCard(effect: effect)
                    }
                }
            }
            .padding()
        }
    }
}

struct MedicationEffectCard: View {
    let effect: HealthInsightsEngine.MedicationEffect
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(effect.medicationName)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                ConfidenceIndicator(confidence: effect.confidence)
            }
            
            VStack(spacing: 8) {
                EffectRow(
                    title: "Pain Relief",
                    value: effect.effectOnPain,
                    icon: "heart.fill",
                    color: effect.effectOnPain > 0 ? .green : .red
                )
                
                EffectRow(
                    title: "Energy Level",
                    value: effect.effectOnEnergy,
                    icon: "bolt.fill",
                    color: effect.effectOnEnergy > 0 ? .green : .red
                )
                
                EffectRow(
                    title: "Sleep Quality",
                    value: effect.effectOnSleep,
                    icon: "moon.fill",
                    color: effect.effectOnSleep > 0 ? .green : .red
                )
            }
            
            HStack {
                Label("\(effect.sampleSize) doses analyzed", systemImage: "pills")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text("Effect in ~\(Int(effect.timeToEffect / 3600))h")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

struct EffectRow: View {
    let title: String
    let value: Double
    let icon: String
    let color: Color
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 20)
            
            Text(title)
                .font(.subheadline)
                .foregroundColor(.primary)
            
            Spacer()
            
            Text(formatEffect(value))
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(color)
        }
    }
    
    private func formatEffect(_ value: Double) -> String {
        let prefix = value > 0 ? "+" : ""
        return "\(prefix)\(String(format: "%.1f", value))"
    }
}

// MARK: - Supporting Views

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)
            
            Text("Analyzing your health data...")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct EmptyInsightsView: View {
    let type: String
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            
            Text("No \(type) found")
                .font(.headline)
                .foregroundColor(.primary)
            
            Text("Keep tracking your health data to discover meaningful \(type).")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

struct ConfidenceIndicator: View {
    let confidence: Double
    
    var body: some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text("\(Int(confidence * 100))%")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(confidenceColor)
            
            Text("confidence")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
    
    private var confidenceColor: Color {
        switch confidence {
        case 0.8...: return .green
        case 0.6..<0.8: return .orange
        default: return .red
        }
    }
}

struct InsightTypeIcon: View {
    let type: HealthInsightsEngine.InsightType
    
    var body: some View {
        Image(systemName: iconName)
            .foregroundColor(iconColor)
            .font(.title3)
    }
    
    private var iconName: String {
        switch type {
        case .medicationEffect: return "pills.fill"
        case .activityCorrelation: return "figure.walk"
        case .sleepPattern: return "moon.fill"
        case .painTrigger: return "exclamationmark.triangle.fill"
        case .energyBooster: return "bolt.fill"
        case .moodInfluencer: return "face.smiling.fill"
        }
    }
    
    private var iconColor: Color {
        switch type {
        case .medicationEffect: return .blue
        case .activityCorrelation: return .green
        case .sleepPattern: return .purple
        case .painTrigger: return .red
        case .energyBooster: return .orange
        case .moodInfluencer: return .pink
        }
    }
}

struct CorrelationStrengthBadge: View {
    let strength: HealthInsightsEngine.CorrelationStrength
    
    var body: some View {
        Text(strength.rawValue)
            .font(.caption)
            .fontWeight(.medium)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(strengthColor.opacity(0.2))
            .foregroundColor(strengthColor)
            .cornerRadius(8)
    }
    
    private var strengthColor: Color {
        switch strength {
        case .veryStrong: return .green
        case .strong: return .blue
        case .moderate: return .orange
        case .weak: return .yellow
        case .veryWeak: return .gray
        }
    }
}

// MARK: - Export View

struct ExportInsightsView: View {
    let insights: [HealthInsightsEngine.PatternInsight]
    let correlations: [HealthInsightsEngine.CorrelationResult]
    let medicationEffects: [HealthInsightsEngine.MedicationEffect]
    
    @Environment(\.dismiss) private var dismiss
    @State private var showingShareSheet = false
    @State private var exportText = ""
    
    var body: some View {
        NavigationView {
            VStack {
                Text("Export your health insights to share with healthcare providers or for personal records.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding()
                
                Button("Generate Report") {
                    generateExportText()
                    showingShareSheet = true
                }
                .buttonStyle(.borderedProminent)
                .tint(Colors.Primary.p500)

                Spacer()
            }
            .navigationTitle("Export Insights")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
        .sheet(isPresented: $showingShareSheet) {
            ShareSheet(items: [exportText])
        }
    }
    
    private func generateExportText() {
        var text = "Health Insights Report\n"
        text += "Generated on \(DateFormatter.mediumDate.string(from: Date()))\n\n"
        
        text += "PATTERN INSIGHTS (\(insights.count))\n"
        text += String(repeating: "=", count: 30) + "\n"
        for insight in insights {
            text += "• \(insight.title)\n"
            text += "  \(insight.description)\n"
            if let recommendation = insight.recommendation {
                text += "  Recommendation: \(recommendation)\n"
            }
            text += "  Confidence: \(Int(insight.confidence * 100))%\n\n"
        }
        
        text += "CORRELATIONS (\(correlations.count))\n"
        text += String(repeating: "=", count: 30) + "\n"
        for correlation in correlations {
            text += "• \(correlation.factor1) ↔ \(correlation.factor2)\n"
            text += "  \(correlation.description)\n"
            text += "  Strength: \(correlation.strength.rawValue)\n"
            text += "  Correlation: \(String(format: "%.2f", correlation.correlation))\n\n"
        }
        
        text += "MEDICATION EFFECTS (\(medicationEffects.count))\n"
        text += String(repeating: "=", count: 30) + "\n"
        for effect in medicationEffects {
            text += "• \(effect.medicationName)\n"
            text += "  Pain Relief: \(String(format: "%.1f", effect.effectOnPain))\n"
            text += "  Energy Effect: \(String(format: "%.1f", effect.effectOnEnergy))\n"
            text += "  Sleep Effect: \(String(format: "%.1f", effect.effectOnSleep))\n"
            text += "  Confidence: \(Int(effect.confidence * 100))%\n\n"
        }
        
        exportText = text
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

#Preview {
    InsightsView()
        let container = NSPersistentContainer(name: "InflamAI")
        container.persistentStoreDescriptions.first?.url = URL(fileURLWithPath: "/dev/null")
        container.loadPersistentStores { _, _ in }
        let context = container.viewContext
        
        return InsightsView()
            .environment(\.managedObjectContext, context)
}