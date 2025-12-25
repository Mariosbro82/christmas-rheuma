//
//  Advanced3DVisualization.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-20.
//

import SwiftUI
import SceneKit
import Charts
import simd

struct Advanced3DVisualization: View {
    @StateObject private var visualizationManager = Visualization3DManager()
    @State private var selectedVisualizationType: VisualizationType = .painHeatMap
    @State private var selectedTimeRange: TimeRange3D = .week
    @State private var selectedBodyPart: BodyPart = .all
    @State private var showingStatistics = false
    @State private var isInteractionEnabled = true
    @State private var rotationAngle: Float = 0
    @State private var zoomLevel: Float = 1.0
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Control Panel
                controlPanel
                
                // 3D Visualization
                ZStack {
                    SceneView(
                        scene: visualizationManager.scene,
                        options: [.allowsCameraControl, .autoenablesDefaultLighting]
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.black)
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                if isInteractionEnabled {
                                    handleDragGesture(value)
                                }
                            }
                    )
                    .gesture(
                        MagnificationGesture()
                            .onChanged { value in
                                if isInteractionEnabled {
                                    handleZoomGesture(value)
                                }
                            }
                    )
                    
                    // Overlay Controls
                    VStack {
                        HStack {
                            Spacer()
                            
                            VStack(spacing: 12) {
                                // View Controls
                                viewControlButtons
                                
                                // Intensity Legend
                                intensityLegend
                            }
                            .padding()
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(12)
                            .padding(.trailing)
                        }
                        
                        Spacer()
                        
                        // Bottom Controls
                        bottomControls
                    }
                }
                
                // Statistics Panel
                if showingStatistics {
                    statisticsPanel
                        .transition(.move(edge: .bottom))
                }
            }
            .navigationTitle("3D Analysis")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button("Export 3D Model") {
                            exportModel()
                        }
                        
                        Button("Share Analysis") {
                            shareAnalysis()
                        }
                        
                        Divider()
                        
                        Button("Reset View") {
                            resetView()
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
        .onAppear {
            setupVisualization()
        }
        .onChange(of: selectedVisualizationType) { _ in
            updateVisualization()
        }
        .onChange(of: selectedTimeRange) { _ in
            updateVisualization()
        }
        .onChange(of: selectedBodyPart) { _ in
            updateVisualization()
        }
    }
    
    // MARK: - Control Panel
    
    private var controlPanel: some View {
        VStack(spacing: 16) {
            // Visualization Type Selector
            Picker("Visualization Type", selection: $selectedVisualizationType) {
                ForEach(VisualizationType.allCases, id: \.self) { type in
                    Text(type.displayName)
                        .tag(type)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            
            HStack {
                // Time Range Selector
                VStack(alignment: .leading, spacing: 4) {
                    Text("Time Range")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Picker("Time Range", selection: $selectedTimeRange) {
                        ForEach(TimeRange3D.allCases, id: \.self) { range in
                            Text(range.displayName)
                                .tag(range)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                Spacer()
                
                // Body Part Filter
                VStack(alignment: .leading, spacing: 4) {
                    Text("Focus Area")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Picker("Body Part", selection: $selectedBodyPart) {
                        ForEach(BodyPart.allCases, id: \.self) { part in
                            Text(part.displayName)
                                .tag(part)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                Spacer()
                
                // Statistics Toggle
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        showingStatistics.toggle()
                    }
                }) {
                    VStack(spacing: 4) {
                        Image(systemName: showingStatistics ? "chart.bar.fill" : "chart.bar")
                            .font(.title3)
                        
                        Text("Stats")
                            .font(.caption2)
                    }
                    .foregroundColor(showingStatistics ? .blue : .secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
    }
    
    // MARK: - View Control Buttons
    
    private var viewControlButtons: some View {
        VStack(spacing: 8) {
            Button(action: { rotateView(.front) }) {
                Image(systemName: "person.fill")
                    .foregroundColor(.white)
                    .font(.title3)
            }
            
            Button(action: { rotateView(.back) }) {
                Image(systemName: "person.fill")
                    .foregroundColor(.white)
                    .font(.title3)
                    .rotationEffect(.degrees(180))
            }
            
            Button(action: { rotateView(.left) }) {
                Image(systemName: "arrow.left")
                    .foregroundColor(.white)
                    .font(.title3)
            }
            
            Button(action: { rotateView(.right) }) {
                Image(systemName: "arrow.right")
                    .foregroundColor(.white)
                    .font(.title3)
            }
        }
    }
    
    // MARK: - Intensity Legend
    
    private var intensityLegend: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Intensity")
                .font(.caption)
                .foregroundColor(.white)
            
            VStack(spacing: 2) {
                ForEach(IntensityLevel.allCases.reversed(), id: \.self) { level in
                    HStack(spacing: 8) {
                        Rectangle()
                            .fill(level.color)
                            .frame(width: 20, height: 8)
                            .cornerRadius(2)
                        
                        Text(level.displayName)
                            .font(.caption2)
                            .foregroundColor(.white)
                    }
                }
            }
        }
    }
    
    // MARK: - Bottom Controls
    
    private var bottomControls: some View {
        HStack {
            // Interaction Toggle
            Button(action: {
                isInteractionEnabled.toggle()
            }) {
                HStack(spacing: 4) {
                    Image(systemName: isInteractionEnabled ? "hand.tap.fill" : "hand.tap")
                    Text(isInteractionEnabled ? "Interactive" : "Locked")
                }
                .font(.caption)
                .foregroundColor(isInteractionEnabled ? .blue : .gray)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.7))
                .cornerRadius(8)
            }
            
            Spacer()
            
            // Animation Controls
            HStack(spacing: 12) {
                Button(action: { playAnimation() }) {
                    Image(systemName: "play.fill")
                        .foregroundColor(.white)
                }
                
                Button(action: { pauseAnimation() }) {
                    Image(systemName: "pause.fill")
                        .foregroundColor(.white)
                }
                
                Button(action: { resetAnimation() }) {
                    Image(systemName: "gobackward")
                        .foregroundColor(.white)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.black.opacity(0.7))
            .cornerRadius(8)
        }
        .padding()
    }
    
    // MARK: - Statistics Panel
    
    private var statisticsPanel: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Advanced Analytics")
                    .font(.headline)
                
                Spacer()
                
                Button("Hide") {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        showingStatistics = false
                    }
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    // Pain Distribution Chart
                    StatisticsCard(
                        title: "Pain Distribution",
                        content: AnyView(painDistributionChart)
                    )
                    
                    // Trend Analysis
                    StatisticsCard(
                        title: "Trend Analysis",
                        content: AnyView(trendAnalysisChart)
                    )
                    
                    // Correlation Matrix
                    StatisticsCard(
                        title: "Correlations",
                        content: AnyView(correlationMatrix)
                    )
                    
                    // Predictive Model
                    StatisticsCard(
                        title: "Predictions",
                        content: AnyView(predictiveChart)
                    )
                }
                .padding(.horizontal)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12, corners: [.topLeft, .topRight])
        .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: -2)
    }
    
    // MARK: - Statistics Charts
    
    private var painDistributionChart: some View {
        Chart(visualizationManager.painDistributionData, id: \.bodyPart) { data in
            BarMark(
                x: .value("Body Part", data.bodyPart),
                y: .value("Average Pain", data.averagePain)
            )
            .foregroundStyle(data.color)
        }
        .frame(height: 120)
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXAxis {
            AxisMarks { _ in
                AxisValueLabel()
                    .font(.caption2)
            }
        }
    }
    
    private var trendAnalysisChart: some View {
        Chart(visualizationManager.trendData, id: \.date) { data in
            LineMark(
                x: .value("Date", data.date),
                y: .value("Pain Level", data.painLevel)
            )
            .foregroundStyle(.blue)
            .lineStyle(StrokeStyle(lineWidth: 2))
            
            AreaMark(
                x: .value("Date", data.date),
                y: .value("Pain Level", data.painLevel)
            )
            .foregroundStyle(.blue.opacity(0.1))
        }
        .frame(height: 120)
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXAxis {
            AxisMarks(values: .stride(by: .day)) { _ in
                AxisValueLabel(format: .dateTime.weekday(.abbreviated))
                    .font(.caption2)
            }
        }
    }
    
    private var correlationMatrix: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 4) {
            ForEach(visualizationManager.correlationData, id: \.id) { correlation in
                VStack(spacing: 2) {
                    Text(correlation.factor1)
                        .font(.caption2)
                        .lineLimit(1)
                    
                    Rectangle()
                        .fill(correlationColor(correlation.strength))
                        .frame(height: 20)
                        .cornerRadius(2)
                    
                    Text(correlation.factor2)
                        .font(.caption2)
                        .lineLimit(1)
                    
                    Text(String(format: "%.2f", correlation.strength))
                        .font(.caption2)
                        .fontWeight(.semibold)
                }
            }
        }
        .frame(height: 120)
    }
    
    private var predictiveChart: some View {
        Chart {
            ForEach(visualizationManager.historicalData, id: \.date) { data in
                LineMark(
                    x: .value("Date", data.date),
                    y: .value("Pain Level", data.painLevel)
                )
                .foregroundStyle(.blue)
                .lineStyle(StrokeStyle(lineWidth: 2))
            }
            
            ForEach(visualizationManager.predictedData, id: \.date) { data in
                LineMark(
                    x: .value("Date", data.date),
                    y: .value("Pain Level", data.painLevel)
                )
                .foregroundStyle(.orange)
                .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 5]))
            }
        }
        .frame(height: 120)
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXAxis {
            AxisMarks(values: .stride(by: .day)) { _ in
                AxisValueLabel(format: .dateTime.weekday(.abbreviated))
                    .font(.caption2)
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func setupVisualization() {
        visualizationManager.setupScene()
        updateVisualization()
    }
    
    private func updateVisualization() {
        Task {
            await visualizationManager.updateVisualization(
                type: selectedVisualizationType,
                timeRange: selectedTimeRange,
                bodyPart: selectedBodyPart
            )
        }
    }
    
    private func handleDragGesture(_ value: DragGesture.Value) {
        let sensitivity: Float = 0.01
        rotationAngle += Float(value.translation.x) * sensitivity
        visualizationManager.rotateModel(angle: rotationAngle)
    }
    
    private func handleZoomGesture(_ value: MagnificationGesture.Value) {
        zoomLevel = Float(value)
        visualizationManager.zoomModel(scale: zoomLevel)
    }
    
    private func rotateView(_ direction: ViewDirection) {
        visualizationManager.rotateToView(direction)
    }
    
    private func playAnimation() {
        visualizationManager.playTimelineAnimation()
    }
    
    private func pauseAnimation() {
        visualizationManager.pauseAnimation()
    }
    
    private func resetAnimation() {
        visualizationManager.resetAnimation()
    }
    
    private func resetView() {
        rotationAngle = 0
        zoomLevel = 1.0
        visualizationManager.resetView()
    }
    
    private func exportModel() {
        visualizationManager.exportModel()
    }
    
    private func shareAnalysis() {
        visualizationManager.shareAnalysis()
    }
    
    private func correlationColor(_ strength: Double) -> Color {
        let absStrength = abs(strength)
        if absStrength < 0.3 {
            return .gray
        } else if absStrength < 0.6 {
            return strength > 0 ? .green : .orange
        } else {
            return strength > 0 ? .blue : .red
        }
    }
}

// MARK: - Supporting Views

struct StatisticsCard: View {
    let title: String
    let content: AnyView
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.semibold)
            
            content
        }
        .padding()
        .frame(width: 200)
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - 3D Visualization Manager

@MainActor
class Visualization3DManager: ObservableObject {
    @Published var scene: SCNScene
    @Published var painDistributionData: [PainDistributionData] = []
    @Published var trendData: [TrendData] = []
    @Published var correlationData: [CorrelationData] = []
    @Published var historicalData: [PainData] = []
    @Published var predictedData: [PainData] = []
    
    private var bodyModel: SCNNode?
    private var heatMapNodes: [SCNNode] = []
    private var animationTimer: Timer?
    
    init() {
        self.scene = SCNScene()
        setupMockData()
    }
    
    func setupScene() {
        // Create camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 10)
        scene.rootNode.addChildNode(cameraNode)
        
        // Create lighting
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.position = SCNVector3(x: 0, y: 10, z: 10)
        scene.rootNode.addChildNode(lightNode)
        
        // Create ambient light
        let ambientLightNode = SCNNode()
        ambientLightNode.light = SCNLight()
        ambientLightNode.light?.type = .ambient
        ambientLightNode.light?.color = UIColor.darkGray
        scene.rootNode.addChildNode(ambientLightNode)
        
        // Create body model
        createBodyModel()
    }
    
    private func createBodyModel() {
        // Create a simplified human body model using basic shapes
        let bodyGroup = SCNNode()
        
        // Head
        let head = SCNSphere(radius: 0.5)
        let headNode = SCNNode(geometry: head)
        headNode.position = SCNVector3(0, 3, 0)
        headNode.geometry?.firstMaterial?.diffuse.contents = UIColor.systemBlue
        bodyGroup.addChildNode(headNode)
        
        // Torso
        let torso = SCNBox(width: 1.5, height: 2, length: 0.8, chamferRadius: 0.1)
        let torsoNode = SCNNode(geometry: torso)
        torsoNode.position = SCNVector3(0, 1, 0)
        torsoNode.geometry?.firstMaterial?.diffuse.contents = UIColor.systemBlue
        bodyGroup.addChildNode(torsoNode)
        
        // Arms
        createLimb(parent: bodyGroup, position: SCNVector3(-1.2, 1.5, 0), size: SCNVector3(0.3, 1.5, 0.3))
        createLimb(parent: bodyGroup, position: SCNVector3(1.2, 1.5, 0), size: SCNVector3(0.3, 1.5, 0.3))
        
        // Legs
        createLimb(parent: bodyGroup, position: SCNVector3(-0.4, -1, 0), size: SCNVector3(0.4, 2, 0.4))
        createLimb(parent: bodyGroup, position: SCNVector3(0.4, -1, 0), size: SCNVector3(0.4, 2, 0.4))
        
        bodyModel = bodyGroup
        scene.rootNode.addChildNode(bodyGroup)
    }
    
    private func createLimb(parent: SCNNode, position: SCNVector3, size: SCNVector3) {
        let limb = SCNBox(width: CGFloat(size.x), height: CGFloat(size.y), length: CGFloat(size.z), chamferRadius: 0.05)
        let limbNode = SCNNode(geometry: limb)
        limbNode.position = position
        limbNode.geometry?.firstMaterial?.diffuse.contents = UIColor.systemBlue
        parent.addChildNode(limbNode)
    }
    
    func updateVisualization(type: VisualizationType, timeRange: TimeRange3D, bodyPart: BodyPart) async {
        // Clear existing heat map
        clearHeatMap()
        
        switch type {
        case .painHeatMap:
            await createPainHeatMap(timeRange: timeRange, bodyPart: bodyPart)
        case .inflammationMap:
            await createInflammationMap(timeRange: timeRange, bodyPart: bodyPart)
        case .mobilityAnalysis:
            await createMobilityAnalysis(timeRange: timeRange, bodyPart: bodyPart)
        case .treatmentResponse:
            await createTreatmentResponse(timeRange: timeRange, bodyPart: bodyPart)
        }
        
        // Update statistics data
        updateStatisticsData(type: type, timeRange: timeRange)
    }
    
    private func clearHeatMap() {
        heatMapNodes.forEach { $0.removeFromParentNode() }
        heatMapNodes.removeAll()
    }
    
    private func createPainHeatMap(timeRange: TimeRange3D, bodyPart: BodyPart) async {
        // Simulate pain data points on the body
        let painPoints = generatePainPoints(for: bodyPart)
        
        for point in painPoints {
            let sphere = SCNSphere(radius: 0.1)
            let node = SCNNode(geometry: sphere)
            node.position = point.position
            
            // Color based on pain intensity
            let intensity = point.intensity
            let color = intensityToColor(intensity)
            node.geometry?.firstMaterial?.diffuse.contents = color
            node.geometry?.firstMaterial?.emission.contents = color
            
            // Add pulsing animation for high pain
            if intensity > 0.7 {
                let pulseAnimation = CABasicAnimation(keyPath: "transform.scale")
                pulseAnimation.fromValue = 1.0
                pulseAnimation.toValue = 1.3
                pulseAnimation.duration = 1.0
                pulseAnimation.autoreverses = true
                pulseAnimation.repeatCount = .infinity
                node.addAnimation(pulseAnimation, forKey: "pulse")
            }
            
            scene.rootNode.addChildNode(node)
            heatMapNodes.append(node)
        }
    }
    
    private func createInflammationMap(timeRange: TimeRange3D, bodyPart: BodyPart) async {
        // Similar to pain heat map but with different color scheme
        let inflammationPoints = generateInflammationPoints(for: bodyPart)
        
        for point in inflammationPoints {
            let sphere = SCNSphere(radius: 0.08)
            let node = SCNNode(geometry: sphere)
            node.position = point.position
            
            // Orange-red color scheme for inflammation
            let intensity = point.intensity
            let color = UIColor(red: 1.0, green: CGFloat(1.0 - intensity), blue: 0.0, alpha: 0.8)
            node.geometry?.firstMaterial?.diffuse.contents = color
            
            scene.rootNode.addChildNode(node)
            heatMapNodes.append(node)
        }
    }
    
    private func createMobilityAnalysis(timeRange: TimeRange3D, bodyPart: BodyPart) async {
        // Create mobility indicators with movement trails
        let mobilityData = generateMobilityData(for: bodyPart)
        
        for data in mobilityData {
            // Create trail effect
            let trail = createMovementTrail(from: data.startPosition, to: data.endPosition, mobility: data.mobilityScore)
            scene.rootNode.addChildNode(trail)
            heatMapNodes.append(trail)
        }
    }
    
    private func createTreatmentResponse(timeRange: TimeRange3D, bodyPart: BodyPart) async {
        // Show treatment effectiveness with color-coded regions
        let treatmentData = generateTreatmentData(for: bodyPart)
        
        for data in treatmentData {
            let sphere = SCNSphere(radius: 0.12)
            let node = SCNNode(geometry: sphere)
            node.position = data.position
            
            // Green for improvement, red for worsening
            let effectiveness = data.effectiveness
            let color = effectiveness > 0 ? 
                UIColor(red: CGFloat(1.0 - effectiveness), green: 1.0, blue: 0.0, alpha: 0.7) :
                UIColor(red: 1.0, green: CGFloat(1.0 + effectiveness), blue: 0.0, alpha: 0.7)
            
            node.geometry?.firstMaterial?.diffuse.contents = color
            
            scene.rootNode.addChildNode(node)
            heatMapNodes.append(node)
        }
    }
    
    private func createMovementTrail(from start: SCNVector3, to end: SCNVector3, mobility: Double) -> SCNNode {
        let trail = SCNNode()
        
        // Create line geometry
        let vertices = [start, end]
        let vertexSource = SCNGeometrySource(vertices: vertices)
        let indices: [Int32] = [0, 1]
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(data: indexData, primitiveType: .line, primitiveCount: 1, bytesPerIndex: MemoryLayout<Int32>.size)
        
        let geometry = SCNGeometry(sources: [vertexSource], elements: [element])
        
        // Color based on mobility score
        let color = mobility > 0.5 ? UIColor.green : UIColor.red
        geometry.firstMaterial?.diffuse.contents = color
        geometry.firstMaterial?.lineWidth = CGFloat(mobility * 10)
        
        trail.geometry = geometry
        return trail
    }
    
    private func updateStatisticsData(type: VisualizationType, timeRange: TimeRange3D) {
        // Update pain distribution data
        painDistributionData = BodyPart.allCases.filter { $0 != .all }.map { bodyPart in
            PainDistributionData(
                bodyPart: bodyPart.displayName,
                averagePain: Double.random(in: 1...10),
                color: bodyPart.color
            )
        }
        
        // Update trend data
        let calendar = Calendar.current
        let now = Date()
        trendData = (0..<7).map { dayOffset in
            let date = calendar.date(byAdding: .day, value: -dayOffset, to: now)!
            return TrendData(
                date: date,
                painLevel: Double.random(in: 1...10)
            )
        }.reversed()
        
        // Update correlation data
        correlationData = [
            CorrelationData(id: UUID(), factor1: "Pain", factor2: "Weather", strength: -0.6),
            CorrelationData(id: UUID(), factor1: "Pain", factor2: "Sleep", strength: -0.4),
            CorrelationData(id: UUID(), factor1: "Pain", factor2: "Stress", strength: 0.7),
            CorrelationData(id: UUID(), factor1: "Mobility", factor2: "Exercise", strength: 0.5),
            CorrelationData(id: UUID(), factor1: "Inflammation", factor2: "Diet", strength: -0.3),
            CorrelationData(id: UUID(), factor1: "Fatigue", factor2: "Pain", strength: 0.8)
        ]
        
        // Update historical and predicted data
        historicalData = (0..<14).map { dayOffset in
            let date = calendar.date(byAdding: .day, value: -dayOffset, to: now)!
            return PainData(
                date: date,
                painLevel: Double.random(in: 3...8)
            )
        }.reversed()
        
        predictedData = (1...7).map { dayOffset in
            let date = calendar.date(byAdding: .day, value: dayOffset, to: now)!
            return PainData(
                date: date,
                painLevel: Double.random(in: 2...7)
            )
        }
    }
    
    // MARK: - Animation and Interaction Methods
    
    func rotateModel(angle: Float) {
        bodyModel?.eulerAngles.y = angle
    }
    
    func zoomModel(scale: Float) {
        bodyModel?.scale = SCNVector3(scale, scale, scale)
    }
    
    func rotateToView(_ direction: ViewDirection) {
        guard let bodyModel = bodyModel else { return }
        
        let rotation: SCNVector3
        switch direction {
        case .front:
            rotation = SCNVector3(0, 0, 0)
        case .back:
            rotation = SCNVector3(0, Float.pi, 0)
        case .left:
            rotation = SCNVector3(0, Float.pi / 2, 0)
        case .right:
            rotation = SCNVector3(0, -Float.pi / 2, 0)
        }
        
        let action = SCNAction.rotateTo(x: CGFloat(rotation.x), y: CGFloat(rotation.y), z: CGFloat(rotation.z), duration: 0.5)
        bodyModel.runAction(action)
    }
    
    func playTimelineAnimation() {
        // Implement timeline animation showing pain progression over time
        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            // Update visualization for next time point
        }
    }
    
    func pauseAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
    }
    
    func resetAnimation() {
        pauseAnimation()
        // Reset to initial state
    }
    
    func resetView() {
        bodyModel?.eulerAngles = SCNVector3(0, 0, 0)
        bodyModel?.scale = SCNVector3(1, 1, 1)
    }
    
    func exportModel() {
        // Export 3D model functionality
    }
    
    func shareAnalysis() {
        // Share analysis functionality
    }
    
    // MARK: - Data Generation Methods
    
    private func setupMockData() {
        // Initialize with mock data
    }
    
    private func generatePainPoints(for bodyPart: BodyPart) -> [PainPoint] {
        // Generate realistic pain points based on body part
        var points: [PainPoint] = []
        
        switch bodyPart {
        case .all:
            points.append(contentsOf: generatePainPoints(for: .joints))
            points.append(contentsOf: generatePainPoints(for: .spine))
            points.append(contentsOf: generatePainPoints(for: .extremities))
        case .joints:
            points = [
                PainPoint(position: SCNVector3(-1.2, 1.5, 0), intensity: 0.8), // Left shoulder
                PainPoint(position: SCNVector3(1.2, 1.5, 0), intensity: 0.6),  // Right shoulder
                PainPoint(position: SCNVector3(-0.4, 0, 0), intensity: 0.9),   // Left hip
                PainPoint(position: SCNVector3(0.4, 0, 0), intensity: 0.7),    // Right hip
                PainPoint(position: SCNVector3(-0.4, -2, 0), intensity: 0.5),  // Left knee
                PainPoint(position: SCNVector3(0.4, -2, 0), intensity: 0.4)    // Right knee
            ]
        case .spine:
            points = [
                PainPoint(position: SCNVector3(0, 2.5, 0), intensity: 0.6),   // Upper spine
                PainPoint(position: SCNVector3(0, 1.5, 0), intensity: 0.8),   // Mid spine
                PainPoint(position: SCNVector3(0, 0.5, 0), intensity: 0.9)    // Lower spine
            ]
        case .extremities:
            points = [
                PainPoint(position: SCNVector3(-1.2, 0.5, 0), intensity: 0.4), // Left wrist
                PainPoint(position: SCNVector3(1.2, 0.5, 0), intensity: 0.3),  // Right wrist
                PainPoint(position: SCNVector3(-0.4, -3, 0), intensity: 0.5),  // Left ankle
                PainPoint(position: SCNVector3(0.4, -3, 0), intensity: 0.6)    // Right ankle
            ]
        }
        
        return points
    }
    
    private func generateInflammationPoints(for bodyPart: BodyPart) -> [PainPoint] {
        // Similar to pain points but representing inflammation
        return generatePainPoints(for: bodyPart).map { point in
            PainPoint(position: point.position, intensity: point.intensity * 0.8)
        }
    }
    
    private func generateMobilityData(for bodyPart: BodyPart) -> [MobilityData] {
        return [
            MobilityData(
                startPosition: SCNVector3(-1.2, 1.5, 0),
                endPosition: SCNVector3(-1.0, 1.7, 0),
                mobilityScore: 0.6
            ),
            MobilityData(
                startPosition: SCNVector3(0.4, 0, 0),
                endPosition: SCNVector3(0.6, 0.2, 0),
                mobilityScore: 0.4
            )
        ]
    }
    
    private func generateTreatmentData(for bodyPart: BodyPart) -> [TreatmentData] {
        return [
            TreatmentData(position: SCNVector3(-1.2, 1.5, 0), effectiveness: 0.3),
            TreatmentData(position: SCNVector3(0, 1.5, 0), effectiveness: -0.2),
            TreatmentData(position: SCNVector3(0.4, 0, 0), effectiveness: 0.7)
        ]
    }
    
    private func intensityToColor(_ intensity: Double) -> UIColor {
        // Convert intensity (0-1) to color (blue -> green -> yellow -> red)
        if intensity < 0.25 {
            return UIColor.blue
        } else if intensity < 0.5 {
            return UIColor.green
        } else if intensity < 0.75 {
            return UIColor.yellow
        } else {
            return UIColor.red
        }
    }
}

// MARK: - Supporting Types

enum VisualizationType: CaseIterable {
    case painHeatMap
    case inflammationMap
    case mobilityAnalysis
    case treatmentResponse
    
    var displayName: String {
        switch self {
        case .painHeatMap: return "Pain Heat Map"
        case .inflammationMap: return "Inflammation"
        case .mobilityAnalysis: return "Mobility"
        case .treatmentResponse: return "Treatment"
        }
    }
}

enum TimeRange3D: CaseIterable {
    case day
    case week
    case month
    case quarter
    
    var displayName: String {
        switch self {
        case .day: return "24 Hours"
        case .week: return "7 Days"
        case .month: return "30 Days"
        case .quarter: return "3 Months"
        }
    }
}

enum BodyPart: CaseIterable {
    case all
    case joints
    case spine
    case extremities
    
    var displayName: String {
        switch self {
        case .all: return "Full Body"
        case .joints: return "Joints"
        case .spine: return "Spine"
        case .extremities: return "Hands & Feet"
        }
    }
    
    var color: Color {
        switch self {
        case .all: return .blue
        case .joints: return .red
        case .spine: return .green
        case .extremities: return .orange
        }
    }
}

enum ViewDirection {
    case front
    case back
    case left
    case right
}

enum IntensityLevel: CaseIterable {
    case none
    case mild
    case moderate
    case severe
    case extreme
    
    var displayName: String {
        switch self {
        case .none: return "None"
        case .mild: return "Mild"
        case .moderate: return "Moderate"
        case .severe: return "Severe"
        case .extreme: return "Extreme"
        }
    }
    
    var color: Color {
        switch self {
        case .none: return .blue
        case .mild: return .green
        case .moderate: return .yellow
        case .severe: return .orange
        case .extreme: return .red
        }
    }
}

struct PainPoint {
    let position: SCNVector3
    let intensity: Double
}

struct MobilityData {
    let startPosition: SCNVector3
    let endPosition: SCNVector3
    let mobilityScore: Double
}

struct TreatmentData {
    let position: SCNVector3
    let effectiveness: Double // -1 to 1, negative means worsening
}

struct PainDistributionData {
    let bodyPart: String
    let averagePain: Double
    let color: Color
}

struct TrendData {
    let date: Date
    let painLevel: Double
}

struct CorrelationData {
    let id: UUID
    let factor1: String
    let factor2: String
    let strength: Double // -1 to 1
}

struct PainData {
    let date: Date
    let painLevel: Double
}

// MARK: - Extensions

extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners
    
    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}

#Preview {
    Advanced3DVisualization()
}