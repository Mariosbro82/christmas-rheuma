//
//  PainBodyDiagram.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import CoreData

struct PainBodyDiagram: View {
    let painEntries: [PainEntry]
    let showIntensity: Bool
    let isInteractive: Bool
    
    @State private var selectedRegion: String? = nil
    
    init(painEntries: [PainEntry], showIntensity: Bool = true, isInteractive: Bool = false) {
        self.painEntries = painEntries
        self.showIntensity = showIntensity
        self.isInteractive = isInteractive
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Human body outline
                BodyOutlineView()
                    .stroke(Color.gray, lineWidth: 2)
                
                // Pain bubbles for each body region
                ForEach(bodyRegions, id: \.self) { region in
                    if let painData = getPainDataForRegion(region) {
                        PainBubble(
                            region: region,
                            intensity: painData.averageIntensity,
                            frequency: painData.frequency,
                            position: getPositionForRegion(region, in: geometry.size),
                            showIntensity: showIntensity,
                            isSelected: selectedRegion == region
                        )
                        .onTapGesture {
                            if isInteractive {
                                selectedRegion = selectedRegion == region ? nil : region
                            }
                        }
                    }
                }
                
                // Legend
                if showIntensity {
                    VStack {
                        Spacer()
                        HStack {
                            PainIntensityLegend()
                            Spacer()
                        }
                        .padding()
                    }
                }
                
                // Selected region details
                if let selectedRegion = selectedRegion,
                   let painData = getPainDataForRegion(selectedRegion) {
                    VStack {
                        HStack {
                            Spacer()
                            PainRegionDetailCard(
                                region: selectedRegion,
                                painData: painData
                            )
                            .padding()
                        }
                        Spacer()
                    }
                }
            }
        }
        .aspectRatio(0.6, contentMode: .fit)
    }
    
    private var bodyRegions: [String] {
        [
            "head", "neck", "left_shoulder", "right_shoulder",
            "left_arm", "right_arm", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "chest", "upper_back",
            "lower_back", "abdomen", "left_hip", "right_hip",
            "left_thigh", "right_thigh", "left_knee", "right_knee",
            "left_calf", "right_calf", "left_ankle", "right_ankle",
            "left_foot", "right_foot"
        ]
    }
    
    private func getPainDataForRegion(_ region: String) -> PainRegionData? {
        let regionEntries = painEntries.filter { entry in
            entry.bodyRegions?.contains(region) == true
        }
        
        guard !regionEntries.isEmpty else { return nil }
        
        let totalIntensity = regionEntries.reduce(0) { $0 + $1.painLevel }
        let averageIntensity = Double(totalIntensity) / Double(regionEntries.count)
        
        return PainRegionData(
            region: region,
            averageIntensity: averageIntensity,
            frequency: regionEntries.count,
            entries: regionEntries
        )
    }
    
    private func getPositionForRegion(_ region: String, in size: CGSize) -> CGPoint {
        let centerX = size.width / 2
        let centerY = size.height / 2
        
        switch region {
        case "head":
            return CGPoint(x: centerX, y: size.height * 0.1)
        case "neck":
            return CGPoint(x: centerX, y: size.height * 0.18)
        case "left_shoulder":
            return CGPoint(x: centerX - size.width * 0.15, y: size.height * 0.25)
        case "right_shoulder":
            return CGPoint(x: centerX + size.width * 0.15, y: size.height * 0.25)
        case "left_arm":
            return CGPoint(x: centerX - size.width * 0.25, y: size.height * 0.35)
        case "right_arm":
            return CGPoint(x: centerX + size.width * 0.25, y: size.height * 0.35)
        case "left_elbow":
            return CGPoint(x: centerX - size.width * 0.3, y: size.height * 0.45)
        case "right_elbow":
            return CGPoint(x: centerX + size.width * 0.3, y: size.height * 0.45)
        case "left_wrist":
            return CGPoint(x: centerX - size.width * 0.35, y: size.height * 0.55)
        case "right_wrist":
            return CGPoint(x: centerX + size.width * 0.35, y: size.height * 0.55)
        case "chest":
            return CGPoint(x: centerX, y: size.height * 0.35)
        case "upper_back":
            return CGPoint(x: centerX, y: size.height * 0.35)
        case "lower_back":
            return CGPoint(x: centerX, y: size.height * 0.5)
        case "abdomen":
            return CGPoint(x: centerX, y: size.height * 0.45)
        case "left_hip":
            return CGPoint(x: centerX - size.width * 0.1, y: size.height * 0.55)
        case "right_hip":
            return CGPoint(x: centerX + size.width * 0.1, y: size.height * 0.55)
        case "left_thigh":
            return CGPoint(x: centerX - size.width * 0.08, y: size.height * 0.65)
        case "right_thigh":
            return CGPoint(x: centerX + size.width * 0.08, y: size.height * 0.65)
        case "left_knee":
            return CGPoint(x: centerX - size.width * 0.08, y: size.height * 0.75)
        case "right_knee":
            return CGPoint(x: centerX + size.width * 0.08, y: size.height * 0.75)
        case "left_calf":
            return CGPoint(x: centerX - size.width * 0.08, y: size.height * 0.85)
        case "right_calf":
            return CGPoint(x: centerX + size.width * 0.08, y: size.height * 0.85)
        case "left_ankle":
            return CGPoint(x: centerX - size.width * 0.08, y: size.height * 0.92)
        case "right_ankle":
            return CGPoint(x: centerX + size.width * 0.08, y: size.height * 0.92)
        case "left_foot":
            return CGPoint(x: centerX - size.width * 0.08, y: size.height * 0.98)
        case "right_foot":
            return CGPoint(x: centerX + size.width * 0.08, y: size.height * 0.98)
        default:
            return CGPoint(x: centerX, y: centerY)
        }
    }
}

struct PainRegionData {
    let region: String
    let averageIntensity: Double
    let frequency: Int
    let entries: [PainEntry]
}

struct PainBubble: View {
    let region: String
    let intensity: Double
    let frequency: Int
    let position: CGPoint
    let showIntensity: Bool
    let isSelected: Bool
    
    private var bubbleSize: CGFloat {
        let baseSize: CGFloat = 20
        let frequencyMultiplier = min(Double(frequency) / 10.0, 2.0)
        return baseSize * (1.0 + frequencyMultiplier)
    }
    
    private var bubbleColor: Color {
        switch intensity {
        case 0..<2:
            return .green
        case 2..<4:
            return .yellow
        case 4..<6:
            return .orange
        case 6..<8:
            return .red
        default:
            return .purple
        }
    }
    
    var body: some View {
        Circle()
            .fill(
                RadialGradient(
                    gradient: Gradient(colors: [
                        bubbleColor.opacity(0.8),
                        bubbleColor.opacity(0.3)
                    ]),
                    center: .center,
                    startRadius: 0,
                    endRadius: bubbleSize / 2
                )
            )
            .frame(width: bubbleSize, height: bubbleSize)
            .overlay(
                Circle()
                    .stroke(isSelected ? Color.blue : bubbleColor, lineWidth: isSelected ? 3 : 1)
            )
            .overlay(
                Group {
                    if showIntensity {
                        Text(String(format: "%.1f", intensity))
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                    }
                }
            )
            .position(position)
            .scaleEffect(isSelected ? 1.2 : 1.0)
            .animation(.easeInOut(duration: 0.2), value: isSelected)
    }
}

struct BodyOutlineView: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let width = rect.width
        let height = rect.height
        let centerX = width / 2
        
        // Head (circle)
        path.addEllipse(in: CGRect(
            x: centerX - width * 0.08,
            y: height * 0.05,
            width: width * 0.16,
            height: height * 0.12
        ))
        
        // Neck
        path.move(to: CGPoint(x: centerX, y: height * 0.17))
        path.addLine(to: CGPoint(x: centerX, y: height * 0.22))
        
        // Torso
        path.move(to: CGPoint(x: centerX - width * 0.12, y: height * 0.22))
        path.addLine(to: CGPoint(x: centerX + width * 0.12, y: height * 0.22))
        path.addLine(to: CGPoint(x: centerX + width * 0.1, y: height * 0.55))
        path.addLine(to: CGPoint(x: centerX - width * 0.1, y: height * 0.55))
        path.closeSubpath()
        
        // Left arm
        path.move(to: CGPoint(x: centerX - width * 0.12, y: height * 0.25))
        path.addLine(to: CGPoint(x: centerX - width * 0.35, y: height * 0.55))
        
        // Right arm
        path.move(to: CGPoint(x: centerX + width * 0.12, y: height * 0.25))
        path.addLine(to: CGPoint(x: centerX + width * 0.35, y: height * 0.55))
        
        // Left leg
        path.move(to: CGPoint(x: centerX - width * 0.08, y: height * 0.55))
        path.addLine(to: CGPoint(x: centerX - width * 0.08, y: height * 0.95))
        
        // Right leg
        path.move(to: CGPoint(x: centerX + width * 0.08, y: height * 0.55))
        path.addLine(to: CGPoint(x: centerX + width * 0.08, y: height * 0.95))
        
        return path
    }
}

struct PainIntensityLegend: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Pain Intensity")
                .font(.caption)
                .fontWeight(.semibold)
            
            HStack(spacing: 8) {
                LegendItem(color: .green, label: "0-2")
                LegendItem(color: .yellow, label: "2-4")
                LegendItem(color: .orange, label: "4-6")
                LegendItem(color: .red, label: "6-8")
                LegendItem(color: .purple, label: "8-10")
            }
        }
        .padding(8)
        .background(Color(.systemBackground).opacity(0.9))
        .cornerRadius(8)
        .shadow(radius: 2)
    }
}

struct LegendItem: View {
    let color: Color
    let label: String
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 12, height: 12)
            
            Text(label)
                .font(.caption2)
        }
    }
}

struct PainRegionDetailCard: View {
    let region: String
    let painData: PainRegionData
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(region.replacingOccurrences(of: "_", with: " ").capitalized)
                .font(.headline)
            
            HStack {
                Text("Avg Intensity:")
                    .font(.caption)
                Spacer()
                Text(String(format: "%.1f", painData.averageIntensity))
                    .font(.caption)
                    .fontWeight(.semibold)
            }
            
            HStack {
                Text("Frequency:")
                    .font(.caption)
                Spacer()
                Text("\(painData.frequency) times")
                    .font(.caption)
                    .fontWeight(.semibold)
            }
            
            if let lastEntry = painData.entries.last {
                HStack {
                    Text("Last Recorded:")
                        .font(.caption)
                    Spacer()
                    Text(lastEntry.timestamp?.formatted(date: .abbreviated, time: .omitted) ?? "N/A")
                        .font(.caption)
                        .fontWeight(.semibold)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 4)
        .frame(maxWidth: 200)
    }
}

struct PainBodyDiagram_Previews: PreviewProvider {
    static var previews: some View {
        PainBodyDiagram(painEntries: [], showIntensity: true, isInteractive: true)
            .frame(height: 400)
            .padding()
    }
}