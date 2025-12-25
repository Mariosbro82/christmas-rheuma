//
//  BodyMappingView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import SceneKit
import ARKit

struct BodyMappingView: View {
    @StateObject private var bodyMappingManager = BodyMappingManager()
    @State private var selectedTab = 0
    @State private var showingAddPain = false
    @State private var showingARView = false
    @State private var showingInsights = false
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // 3D Body View
                Body3DView(manager: bodyMappingManager)
                    .tabItem {
                        Image(systemName: "figure.stand")
                        Text("3D Body")
                    }
                    .tag(0)
                
                // Pain History
                PainHistoryView(manager: bodyMappingManager)
                    .tabItem {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                        Text("History")
                    }
                    .tag(1)
                
                // AR Scanning
                ARScanningView(manager: bodyMappingManager)
                    .tabItem {
                        Image(systemName: "camera.viewfinder")
                        Text("AR Scan")
                    }
                    .tag(2)
                
                // Insights
                BodyInsightsView(manager: bodyMappingManager)
                    .tabItem {
                        Image(systemName: "brain.head.profile")
                        Text("Insights")
                    }
                    .tag(3)
            }
            .navigationTitle("Body Mapping")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Add Pain") {
                        showingAddPain = true
                    }
                }
            }
        }
        .sheet(isPresented: $showingAddPain) {
            AddPainEntryView(manager: bodyMappingManager)
        }
    }
}

struct Body3DView: View {
    @ObservedObject var manager: BodyMappingManager
    @State private var selectedRegion: BodyRegion?
    @State private var rotationAngle: Float = 0
    @State private var showingRegionDetail = false
    
    var body: some View {
        VStack {
            // 3D Body Model
            SceneView(
                scene: createBodyScene(),
                pointOfView: nil,
                options: [.allowsCameraControl, .autoenablesDefaultLighting]
            )
            .frame(height: 400)
            .background(Color.black.opacity(0.1))
            .cornerRadius(12)
            .onTapGesture { location in
                handleBodyTap(at: location)
            }
            
            // Pain Level Legend
            PainLevelLegend()
            
            // Region List
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(manager.bodyRegions) { region in
                        BodyRegionCard(region: region) {
                            selectedRegion = region
                            showingRegionDetail = true
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
        .sheet(item: $selectedRegion) { region in
            RegionDetailView(region: region, manager: manager)
        }
    }
    
    private func createBodyScene() -> SCNScene {
        let scene = SCNScene()
        
        // Create body model
        let bodyNode = createBodyModel()
        scene.rootNode.addChildNode(bodyNode)
        
        // Add lighting
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.position = SCNVector3(0, 10, 10)
        scene.rootNode.addChildNode(lightNode)
        
        // Add ambient light
        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.color = UIColor.gray
        scene.rootNode.addChildNode(ambientLight)
        
        return scene
    }
    
    private func createBodyModel() -> SCNNode {
        let bodyNode = SCNNode()
        
        // Create simplified body parts with pain visualization
        for region in manager.bodyRegions {
            let regionNode = createRegionNode(for: region)
            bodyNode.addChildNode(regionNode)
        }
        
        return bodyNode
    }
    
    private func createRegionNode(for region: BodyRegion) -> SCNNode {
        let node = SCNNode()
        
        // Create geometry based on region
        let geometry: SCNGeometry
        switch region.category {
        case .head:
            geometry = SCNSphere(radius: 0.1)
        case .neck:
            geometry = SCNCylinder(radius: 0.05, height: 0.1)
        case .torso:
            geometry = SCNBox(width: 0.3, height: 0.4, length: 0.2, chamferRadius: 0.02)
        case .arms:
            geometry = SCNCylinder(radius: 0.03, height: 0.3)
        case .hands:
            geometry = SCNSphere(radius: 0.04)
        case .legs:
            geometry = SCNCylinder(radius: 0.04, height: 0.4)
        case .feet:
            geometry = SCNBox(width: 0.08, height: 0.03, length: 0.15, chamferRadius: 0.01)
        case .joints:
            geometry = SCNSphere(radius: 0.03)
        }
        
        // Apply pain level coloring
        let material = SCNMaterial()
        material.diffuse.contents = UIColor(region.painLevel.color)
        material.transparency = 0.8
        geometry.materials = [material]
        
        node.geometry = geometry
        node.position = SCNVector3(
            region.position.x,
            region.position.y,
            region.position.z
        )
        node.name = region.id.uuidString
        
        return node
    }
    
    private func handleBodyTap(at location: CGPoint) {
        // Handle tap on 3D body model
        // This would require more complex hit testing in a real implementation
    }
}

struct PainLevelLegend: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Pain Levels")
                .font(.headline)
                .padding(.bottom, 4)
            
            HStack {
                ForEach(PainLevel.allCases, id: \.self) { level in
                    HStack(spacing: 4) {
                        Circle()
                            .fill(level.color)
                            .frame(width: 12, height: 12)
                        Text(level.displayName)
                            .font(.caption)
                    }
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
        .padding(.horizontal)
    }
}

struct BodyRegionCard: View {
    let region: BodyRegion
    let onTap: () -> Void
    
    var body: some View {
        HStack {
            // Pain level indicator
            Circle()
                .fill(region.painLevel.color)
                .frame(width: 20, height: 20)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(region.name)
                    .font(.headline)
                Text(region.anatomicalName)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 2) {
                Text(region.painLevel.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                Text("Updated: \(region.lastUpdated, style: .relative)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Image(systemName: "chevron.right")
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
        .onTapGesture {
            onTap()
        }
    }
}

struct RegionDetailView: View {
    let region: BodyRegion
    @ObservedObject var manager: BodyMappingManager
    @Environment(\.dismiss) private var dismiss
    @State private var showingAddPain = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Region Info
                    VStack(alignment: .leading, spacing: 8) {
                        Text(region.name)
                            .font(.largeTitle)
                            .fontWeight(.bold)
                        
                        Text(region.anatomicalName)
                            .font(.title2)
                            .foregroundColor(.secondary)
                        
                        HStack {
                            Circle()
                                .fill(region.painLevel.color)
                                .frame(width: 24, height: 24)
                            Text(region.painLevel.displayName)
                                .font(.headline)
                        }
                    }
                    
                    Divider()
                    
                    // Pain History
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recent Pain Entries")
                            .font(.headline)
                        
                        let history = manager.getRegionHistory(for: region.id)
                        if history.isEmpty {
                            Text("No pain entries recorded")
                                .foregroundColor(.secondary)
                                .italic()
                        } else {
                            ForEach(history.prefix(5), id: \.id) { entry in
                                PainEntryRow(entry: entry)
                            }
                        }
                    }
                    
                    Divider()
                    
                    // Symptoms
                    if !region.symptoms.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Current Symptoms")
                                .font(.headline)
                            
                            ForEach(region.symptoms, id: \.self) { symptom in
                                Text("• \(symptom)")
                                    .font(.body)
                            }
                        }
                        
                        Divider()
                    }
                    
                    // Notes
                    if !region.notes.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Notes")
                                .font(.headline)
                            
                            Text(region.notes)
                                .font(.body)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Region Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Add Pain") {
                        showingAddPain = true
                    }
                }
            }
        }
        .sheet(isPresented: $showingAddPain) {
            AddPainEntryView(manager: manager, preselectedRegion: region)
        }
    }
}

struct PainEntryRow: View {
    let entry: PainEntry
    
    var body: some View {
        HStack {
            Circle()
                .fill(entry.painLevel.color)
                .frame(width: 16, height: 16)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.painLevel.displayName)
                    .font(.body)
                    .fontWeight(.medium)
                
                Text(entry.timestamp, style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if !entry.notes.isEmpty {
                Text(entry.notes)
                    .font(.caption)
                    .lineLimit(2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

struct AddPainEntryView: View {
    @ObservedObject var manager: BodyMappingManager
    var preselectedRegion: BodyRegion?
    
    @Environment(\.dismiss) private var dismiss
    @State private var selectedRegion: BodyRegion?
    @State private var painLevel: PainLevel = .none
    @State private var symptoms: [String] = []
    @State private var notes = ""
    @State private var newSymptom = ""
    
    var body: some View {
        NavigationView {
            Form {
                Section("Region") {
                    if let preselected = preselectedRegion {
                        Text(preselected.name)
                            .foregroundColor(.secondary)
                    } else {
                        Picker("Select Region", selection: $selectedRegion) {
                            Text("Select a region").tag(nil as BodyRegion?)
                            ForEach(manager.bodyRegions) { region in
                                Text(region.name).tag(region as BodyRegion?)
                            }
                        }
                    }
                }
                
                Section("Pain Level") {
                    Picker("Pain Level", selection: $painLevel) {
                        ForEach(PainLevel.allCases, id: \.self) { level in
                            HStack {
                                Circle()
                                    .fill(level.color)
                                    .frame(width: 12, height: 12)
                                Text(level.displayName)
                            }
                            .tag(level)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                
                Section("Symptoms") {
                    ForEach(symptoms, id: \.self) { symptom in
                        Text(symptom)
                    }
                    .onDelete { indexSet in
                        symptoms.remove(atOffsets: indexSet)
                    }
                    
                    HStack {
                        TextField("Add symptom", text: $newSymptom)
                        Button("Add") {
                            if !newSymptom.isEmpty {
                                symptoms.append(newSymptom)
                                newSymptom = ""
                            }
                        }
                        .disabled(newSymptom.isEmpty)
                    }
                }
                
                Section("Notes") {
                    TextEditor(text: $notes)
                        .frame(minHeight: 100)
                }
            }
            .navigationTitle("Add Pain Entry")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        savePainEntry()
                    }
                    .disabled(!canSave)
                }
            }
        }
        .onAppear {
            if let preselected = preselectedRegion {
                selectedRegion = preselected
            }
        }
    }
    
    private var canSave: Bool {
        (preselectedRegion != nil || selectedRegion != nil) && painLevel != .none
    }
    
    private func savePainEntry() {
        guard let region = preselectedRegion ?? selectedRegion else { return }
        
        manager.addPainEntry(
            for: region.id,
            painLevel: painLevel,
            symptoms: symptoms,
            notes: notes
        )
        
        dismiss()
    }
}

struct PainHistoryView: View {
    @ObservedObject var manager: BodyMappingManager
    @State private var selectedTimeRange: TimeRange = .week
    
    enum TimeRange: String, CaseIterable {
        case day = "24h"
        case week = "7d"
        case month = "30d"
        case year = "1y"
        
        var displayName: String {
            switch self {
            case .day: return "24 Hours"
            case .week: return "7 Days"
            case .month: return "30 Days"
            case .year: return "1 Year"
            }
        }
        
        var timeInterval: TimeInterval {
            switch self {
            case .day: return 24 * 60 * 60
            case .week: return 7 * 24 * 60 * 60
            case .month: return 30 * 24 * 60 * 60
            case .year: return 365 * 24 * 60 * 60
            }
        }
    }
    
    var body: some View {
        VStack {
            // Time range picker
            Picker("Time Range", selection: $selectedTimeRange) {
                ForEach(TimeRange.allCases, id: \.self) { range in
                    Text(range.displayName).tag(range)
                }
            }
            .pickerStyle(.segmented)
            .padding()
            
            // Pain chart
            PainChartView(entries: filteredEntries)
                .frame(height: 200)
                .padding()
            
            // Statistics
            PainStatisticsView(entries: filteredEntries)
                .padding()
            
            // Entry list
            List(filteredEntries, id: \.id) { entry in
                PainHistoryEntryRow(entry: entry, manager: manager)
            }
        }
        .navigationTitle("Pain History")
    }
    
    private var filteredEntries: [PainEntry] {
        let cutoffDate = Date().addingTimeInterval(-selectedTimeRange.timeInterval)
        return manager.painEntries.filter { $0.timestamp >= cutoffDate }
            .sorted { $0.timestamp > $1.timestamp }
    }
}

struct PainChartView: View {
    let entries: [PainEntry]
    
    var body: some View {
        // Simplified chart view - in a real app, use Charts framework
        VStack {
            Text("Pain Level Over Time")
                .font(.headline)
            
            if entries.isEmpty {
                Text("No data available")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // Simple line chart representation
                GeometryReader { geometry in
                    Path { path in
                        let width = geometry.size.width
                        let height = geometry.size.height
                        let maxPain = 4.0 // Max pain level
                        
                        for (index, entry) in entries.enumerated() {
                            let x = width * CGFloat(index) / CGFloat(max(entries.count - 1, 1))
                            let y = height * (1 - CGFloat(entry.painLevel.rawValue) / maxPain)
                            
                            if index == 0 {
                                path.move(to: CGPoint(x: x, y: y))
                            } else {
                                path.addLine(to: CGPoint(x: x, y: y))
                            }
                        }
                    }
                    .stroke(Color.blue, lineWidth: 2)
                }
            }
        }
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

struct PainStatisticsView: View {
    let entries: [PainEntry]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Statistics")
                .font(.headline)
            
            HStack {
                StatCard(title: "Total Entries", value: "\(entries.count)")
                StatCard(title: "Avg Pain", value: String(format: "%.1f", averagePain))
                StatCard(title: "Max Pain", value: "\(maxPain)")
            }
        }
    }
    
    private var averagePain: Double {
        guard !entries.isEmpty else { return 0 }
        let total = entries.reduce(0) { $0 + $1.painLevel.rawValue }
        return Double(total) / Double(entries.count)
    }
    
    private var maxPain: Int {
        entries.map { $0.painLevel.rawValue }.max() ?? 0
    }
}

struct StatCard: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

struct PainHistoryEntryRow: View {
    let entry: PainEntry
    @ObservedObject var manager: BodyMappingManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle()
                    .fill(entry.painLevel.color)
                    .frame(width: 16, height: 16)
                
                Text(regionName)
                    .font(.headline)
                
                Spacer()
                
                Text(entry.timestamp, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(entry.painLevel.displayName)
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            if !entry.symptoms.isEmpty {
                Text("Symptoms: \(entry.symptoms.joined(separator: ", "))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            if !entry.notes.isEmpty {
                Text(entry.notes)
                    .font(.caption)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }
    
    private var regionName: String {
        manager.bodyRegions.first { $0.id == entry.regionId }?.name ?? "Unknown Region"
    }
}

#Preview {
    BodyMappingView()
}