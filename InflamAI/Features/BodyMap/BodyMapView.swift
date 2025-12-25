//
//  BodyMapView.swift
//  InflamAI
//
//  Interactive body map with 47 tappable regions
//  Production-grade: VoiceOver, Dynamic Type, Haptics
//

import SwiftUI
import CoreData

struct BodyMapView: View {
    @StateObject private var viewModel: BodyMapViewModel
    @State private var showingFront = false // Start with back (spine view)
    @State private var selectedRegion: BodyRegion?
    @State private var showHeatmap = true

    @Environment(\.accessibilityReduceMotion) var reduceMotion

    init(context: NSManagedObjectContext) {
        _viewModel = StateObject(wrappedValue: BodyMapViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                // View Toggle
                Picker("View", selection: $showingFront) {
                    Text("Back").tag(false)
                    Text("Front").tag(true)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .accessibilityLabel("Body view selection")

                // Heatmap Toggle
                Toggle(isOn: $showHeatmap) {
                    HStack {
                        Image(systemName: "chart.bar.fill")
                        Text("Show Pain Heatmap")
                    }
                }
                .padding(.horizontal)
                .accessibilityHint("Toggle to show or hide pain intensity overlay")

                // Body Map
                GeometryReader { geometry in
                    ZStack {
                        // Background body outline
                        bodyOutline(showingFront: showingFront)
                            .fill(Color(.systemGray6))
                            .frame(width: geometry.size.width * 0.7, height: geometry.size.height * 0.9)
                            .position(x: geometry.size.width / 2, y: geometry.size.height / 2)

                        // Heatmap overlay - FIXED: Coordinate transform to match body outline
                        if showHeatmap {
                            let bodyWidth = geometry.size.width * 0.7
                            let bodyHeight = geometry.size.height * 0.9
                            let bodyOffsetX = (geometry.size.width - bodyWidth) / 2
                            let bodyOffsetY = (geometry.size.height - bodyHeight) / 2

                            ForEach(currentRegions, id: \.id) { region in
                                if let painData = viewModel.painData[region.rawValue] {
                                    let pos = region.position(forFrontView: showingFront)
                                    Circle()
                                        .fill(painColor(painData.averagePain).opacity(0.3 + min(painData.averagePain / 10.0, 0.7)))
                                        .frame(width: 30, height: 30)
                                        .position(
                                            x: bodyOffsetX + pos.x * bodyWidth,
                                            y: bodyOffsetY + pos.y * bodyHeight
                                        )
                                        .animation(reduceMotion ? .none : .easeInOut, value: painData.averagePain)
                                }
                            }
                        }

                        // Tappable region buttons - FIXED: Same coordinate transform as heatmap
                        let btnBodyWidth = geometry.size.width * 0.7
                        let btnBodyHeight = geometry.size.height * 0.9
                        let btnOffsetX = (geometry.size.width - btnBodyWidth) / 2
                        let btnOffsetY = (geometry.size.height - btnBodyHeight) / 2

                        ForEach(currentRegions, id: \.id) { region in
                            let pos = region.position(forFrontView: showingFront)
                            RegionButton(
                                region: region,
                                painLevel: viewModel.painData[region.rawValue]?.averagePain ?? 0,
                                showLabel: !showHeatmap
                            ) {
                                selectedRegion = region
                                UIImpactFeedbackGenerator(style: .light).impactOccurred()
                            }
                            .position(
                                x: btnOffsetX + pos.x * btnBodyWidth,
                                y: btnOffsetY + pos.y * btnBodyHeight
                            )
                        }
                    }
                }
                .padding()

                // Legend
                PainLegendView()
                    .padding(.horizontal)
            }
            .navigationTitle("Body Map")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        viewModel.refreshData()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh pain data")
                }
            }
            .sheet(item: $selectedRegion) { region in
                RegionDetailView(region: region, viewModel: viewModel)
            }
            .onAppear {
                viewModel.loadPainData()
            }
        }
    }

    // MARK: - Helpers

    private var currentRegions: [BodyRegion] {
        BodyRegion.allCases.filter { region in
            showingFront ? region.isVisibleOnFrontView : region.isVisibleOnBackView
        }
    }

    private func bodyOutline(showingFront: Bool) -> Path {
        Path { path in
            // Simple humanoid outline (back view emphasizes spine)
            if showingFront {
                // Front view outline
                path.move(to: CGPoint(x: 0.5, y: 0.05)) // Head top
                path.addArc(center: CGPoint(x: 0.5, y: 0.1), radius: 0.08, startAngle: .degrees(0), endAngle: .degrees(360), clockwise: true)
                // Shoulders
                path.move(to: CGPoint(x: 0.3, y: 0.2))
                path.addLine(to: CGPoint(x: 0.7, y: 0.2))
                // Torso
                path.addLine(to: CGPoint(x: 0.65, y: 0.5))
                path.addLine(to: CGPoint(x: 0.35, y: 0.5))
                path.closeSubpath()
            } else {
                // Back view outline (emphasize spine)
                path.move(to: CGPoint(x: 0.5, y: 0.05))
                path.addArc(center: CGPoint(x: 0.5, y: 0.1), radius: 0.08, startAngle: .degrees(0), endAngle: .degrees(360), clockwise: true)
                // Spine centerline
                path.move(to: CGPoint(x: 0.5, y: 0.15))
                path.addLine(to: CGPoint(x: 0.5, y: 0.6))
                // Shoulders
                path.move(to: CGPoint(x: 0.3, y: 0.2))
                path.addLine(to: CGPoint(x: 0.7, y: 0.2))
                // Torso outline
                path.move(to: CGPoint(x: 0.35, y: 0.2))
                path.addLine(to: CGPoint(x: 0.35, y: 0.55))
                path.move(to: CGPoint(x: 0.65, y: 0.2))
                path.addLine(to: CGPoint(x: 0.65, y: 0.55))
            }
        }
    }

    private func painColor(_ pain: Double) -> Color {
        // FIXED: Removed case 0: .clear - was making dots invisible
        // Aligned ranges with legend (0-3, 4-6, 7-8, 9-10)
        switch pain {
        case ..<4: return .green      // 0-3: Low pain (green)
        case 4..<7: return .yellow    // 4-6: Moderate pain (yellow)
        case 7..<9: return .orange    // 7-8: High pain (orange)
        default: return .red          // 9-10: Severe pain (red)
        }
    }
}

// MARK: - Region Button

struct RegionButton: View {
    let region: BodyRegion
    let painLevel: Double
    let showLabel: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(buttonColor)
                    .frame(width: 44, height: 44) // Minimum hit target

                if showLabel {
                    Text(region.rawValue.uppercased())
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.white)
                }
            }
        }
        .accessibilityLabel("\(region.displayName)")
        .accessibilityValue(painLevel > 0 ? "Pain level \(Int(painLevel)) out of 10" : "No pain recorded")
        .accessibilityHint("Double tap to view details and log pain")
    }

    private var buttonColor: Color {
        if painLevel == 0 {
            return Color(.systemGray3)
        } else if painLevel < 3 {
            return .green.opacity(0.7)
        } else if painLevel < 6 {
            return .yellow.opacity(0.7)
        } else if painLevel < 8 {
            return .orange.opacity(0.7)
        } else {
            return .red.opacity(0.7)
        }
    }
}

// MARK: - Pain Legend

struct PainLegendView: View {
    var body: some View {
        HStack(spacing: 12) {
            LegendItem(color: .green, label: "0-3")
            LegendItem(color: .yellow, label: "4-6")
            LegendItem(color: .orange, label: "7-8")
            LegendItem(color: .red, label: "9-10")
        }
        .padding(.vertical, 8)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Pain intensity legend: Green 0 to 3, Yellow 4 to 6, Orange 7 to 8, Red 9 to 10")
    }
}

struct LegendItem: View {
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 16, height: 16)
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Preview

struct BodyMapView_Previews: PreviewProvider {
    static var previews: some View {
        BodyMapView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
