//
//  InteractiveBodyMapView.swift
//  InflamAI
//
//  Interactive 2D body visualization for AS pain tracking
//

import SwiftUI
import CoreData

struct InteractiveBodyMapView: View {
    @StateObject private var viewModel: InteractiveBodyMapViewModel
    @State private var selectedRegion: SpineBodyRegion?
    @State private var showingPainInput = false
    @State private var rotateView = false

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: InteractiveBodyMapViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    colors: [Color(.systemBackground), Color.blue.opacity(0.05)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 24) {
                        // Header
                        headerSection

                        // Body view toggle
                        viewToggle

                        // 2D Body Model
                        bodyModelSection

                        // Region List
                        regionListSection

                        // Quick Log Button
                        quickLogButton
                    }
                    .padding()
                }
            }
            .navigationTitle("Body Map")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        viewModel.clearAllRegions()
                    } label: {
                        Image(systemName: "trash")
                    }
                }
            }
            .sheet(isPresented: $showingPainInput) {
                if let region = selectedRegion {
                    PainInputSheet(region: region, viewModel: viewModel, isPresented: $showingPainInput)
                }
            }
        }
    }

    // MARK: - Header Section

    private var headerSection: some View {
        VStack(spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Tap body regions")
                        .font(.headline)
                    Text("to log pain intensity")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()

                // Active regions count
                ZStack {
                    Circle()
                        .fill(viewModel.activeRegions.isEmpty ? Color.gray.opacity(0.2) : Color.red.opacity(0.2))
                        .frame(width: 60, height: 60)

                    VStack(spacing: 2) {
                        Text("\(viewModel.activeRegions.count)")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(viewModel.activeRegions.isEmpty ? .gray : .red)
                        Text("areas")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
    }

    // MARK: - View Toggle

    private var viewToggle: some View {
        HStack(spacing: 16) {
            ForEach([false, true], id: \.self) { isBack in
                Button {
                    withAnimation(.spring()) {
                        rotateView = isBack
                    }
                } label: {
                    HStack {
                        Image(systemName: isBack ? "person.fill" : "person")
                        Text(isBack ? "Back" : "Front")
                    }
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(rotateView == isBack ? .white : .blue)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(rotateView == isBack ? Color.blue : Color.blue.opacity(0.1))
                    .cornerRadius(12)
                }
            }
        }
    }

    // MARK: - Body Model Section

    private var bodyModelSection: some View {
        VStack(spacing: 16) {
            Text("AS Common Pain Sites")
                .font(.caption)
                .foregroundColor(.secondary)

            ZStack {
                // Body outline
                BodyOutlineView(showBack: rotateView)
                    .frame(height: 500)

                // Pain regions overlay
                ForEach(SpineBodyRegion.allCases.filter { rotateView ? $0.isBackView : $0.isFrontView }) { region in
                    SpineBodyRegionButton(
                        region: region,
                        painLevel: viewModel.getPainLevel(for: region),
                        isSelected: selectedRegion == region
                    ) {
                        selectedRegion = region
                        showingPainInput = true
                    }
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(20)
            .shadow(color: Color.black.opacity(0.1), radius: 10)
        }
    }

    // MARK: - Region List

    private var regionListSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Active Pain Regions")
                .font(.headline)
                .padding(.horizontal)

            if viewModel.activeRegions.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 50))
                        .foregroundColor(.green)
                    Text("No pain logged today")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Text("Tap regions above to log")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 40)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(16)
            } else {
                ForEach(viewModel.activeRegions, id: \.region) { item in
                    RegionPainCard(region: item.region, painLevel: item.painLevel) {
                        selectedRegion = item.region
                        showingPainInput = true
                    }
                }
            }
        }
    }

    // MARK: - Quick Log Button

    private var quickLogButton: some View {
        Button {
            Task {
                await viewModel.saveToday()
            }
        } label: {
            HStack {
                Image(systemName: "checkmark.circle.fill")
                Text("Save Today's Pain Map")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(viewModel.activeRegions.isEmpty ? Color.gray : Color.blue)
            .foregroundColor(.white)
            .cornerRadius(16)
        }
        .disabled(viewModel.activeRegions.isEmpty)
    }
}

// MARK: - Body Outline View

struct BodyOutlineView: View {
    let showBack: Bool

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height

            if showBack {
                backBodyOutline(width: width, height: height)
            } else {
                frontBodyOutline(width: width, height: height)
            }
        }
    }

    private func frontBodyOutline(width: CGFloat, height: CGFloat) -> some View {
        Path { path in
            let centerX = width / 2
            let headRadius = width * 0.12
            let neckHeight = height * 0.05
            let shoulderWidth = width * 0.4
            let torsoHeight = height * 0.35
            let hipWidth = width * 0.35
            let legLength = height * 0.40

            // Head
            path.addEllipse(in: CGRect(x: centerX - headRadius, y: 0, width: headRadius * 2, height: headRadius * 1.3))

            // Neck
            let neckTop = headRadius * 1.3
            path.move(to: CGPoint(x: centerX - headRadius * 0.3, y: neckTop))
            path.addLine(to: CGPoint(x: centerX - headRadius * 0.3, y: neckTop + neckHeight))

            path.move(to: CGPoint(x: centerX + headRadius * 0.3, y: neckTop))
            path.addLine(to: CGPoint(x: centerX + headRadius * 0.3, y: neckTop + neckHeight))

            // Shoulders and torso
            let shoulderY = neckTop + neckHeight
            path.addArc(center: CGPoint(x: centerX - shoulderWidth / 2, y: shoulderY),
                       radius: headRadius * 0.5,
                       startAngle: .degrees(-90),
                       endAngle: .degrees(180),
                       clockwise: false)

            path.addArc(center: CGPoint(x: centerX + shoulderWidth / 2, y: shoulderY),
                       radius: headRadius * 0.5,
                       startAngle: .degrees(0),
                       endAngle: .degrees(270),
                       clockwise: false)

            // Torso
            let torsoTop = shoulderY + headRadius * 0.5
            path.move(to: CGPoint(x: centerX - shoulderWidth / 2, y: torsoTop))
            path.addCurve(
                to: CGPoint(x: centerX - hipWidth / 2, y: torsoTop + torsoHeight),
                control1: CGPoint(x: centerX - shoulderWidth / 2, y: torsoTop + torsoHeight * 0.3),
                control2: CGPoint(x: centerX - hipWidth / 2, y: torsoTop + torsoHeight * 0.7)
            )

            path.move(to: CGPoint(x: centerX + shoulderWidth / 2, y: torsoTop))
            path.addCurve(
                to: CGPoint(x: centerX + hipWidth / 2, y: torsoTop + torsoHeight),
                control1: CGPoint(x: centerX + shoulderWidth / 2, y: torsoTop + torsoHeight * 0.3),
                control2: CGPoint(x: centerX + hipWidth / 2, y: torsoTop + torsoHeight * 0.7)
            )

            // Legs
            let legsTop = torsoTop + torsoHeight
            path.move(to: CGPoint(x: centerX - hipWidth / 2, y: legsTop))
            path.addLine(to: CGPoint(x: centerX - hipWidth / 4, y: legsTop + legLength))

            path.move(to: CGPoint(x: centerX + hipWidth / 2, y: legsTop))
            path.addLine(to: CGPoint(x: centerX + hipWidth / 4, y: legsTop + legLength))
        }
        .stroke(Color.blue.opacity(0.3), lineWidth: 2)
    }

    private func backBodyOutline(width: CGFloat, height: CGFloat) -> some View {
        frontBodyOutline(width: width, height: height) // Similar outline for back
    }
}

// MARK: - Body Region Button

struct SpineBodyRegionButton: View {
    let region: SpineBodyRegion
    let painLevel: Int
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Circle()
                .fill(painLevel > 0 ? colorForPain(painLevel) : Color.blue.opacity(0.1))
                .frame(width: 40, height: 40)
                .overlay(
                    Circle()
                        .strokeBorder(isSelected ? Color.white : Color.clear, lineWidth: 3)
                )
                .overlay(
                    Text(painLevel > 0 ? "\(painLevel)" : "+")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(painLevel > 0 ? .white : .blue)
                )
        }
        .position(region.position)
        .shadow(color: painLevel > 0 ? colorForPain(painLevel).opacity(0.5) : Color.clear, radius: 8)
    }

    private func colorForPain(_ level: Int) -> Color {
        switch level {
        case 1...3: return .green
        case 4...6: return .yellow
        case 7...8: return .orange
        case 9...10: return .red
        default: return .gray
        }
    }
}

// MARK: - Region Pain Card

struct RegionPainCard: View {
    let region: SpineBodyRegion
    let painLevel: Int
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                // Region icon
                Circle()
                    .fill(colorForPain(painLevel))
                    .frame(width: 50, height: 50)
                    .overlay(
                        Text("\(painLevel)")
                            .font(.title3)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                    )

                VStack(alignment: .leading, spacing: 4) {
                    Text(region.displayName)
                        .font(.headline)
                    Text(painDescription(painLevel))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(12)
        }
        .buttonStyle(PlainButtonStyle())
    }

    private func colorForPain(_ level: Int) -> Color {
        switch level {
        case 1...3: return .green
        case 4...6: return .yellow
        case 7...8: return .orange
        case 9...10: return .red
        default: return .gray
        }
    }

    private func painDescription(_ level: Int) -> String {
        switch level {
        case 1...3: return "Mild discomfort"
        case 4...6: return "Moderate pain"
        case 7...8: return "Severe pain"
        case 9...10: return "Extreme pain"
        default: return ""
        }
    }
}

// MARK: - Pain Input Sheet

struct PainInputSheet: View {
    let region: SpineBodyRegion
    @ObservedObject var viewModel: InteractiveBodyMapViewModel
    @Binding var isPresented: Bool
    @State private var painLevel: Double

    init(region: SpineBodyRegion, viewModel: InteractiveBodyMapViewModel, isPresented: Binding<Bool>) {
        self.region = region
        self.viewModel = viewModel
        self._isPresented = isPresented
        self._painLevel = State(initialValue: Double(viewModel.getPainLevel(for: region)))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Visual indicator
                Circle()
                    .fill(colorForPain(Int(painLevel)))
                    .frame(width: 100, height: 100)
                    .overlay(
                        Text("\(Int(painLevel))")
                            .font(.system(size: 50, weight: .bold))
                            .foregroundColor(.white)
                    )
                    .padding(.top, 40)

                Text(region.displayName)
                    .font(.title2)
                    .fontWeight(.bold)

                Text(painDescription(Int(painLevel)))
                    .font(.headline)
                    .foregroundColor(.secondary)

                // Slider
                VStack(spacing: 12) {
                    Slider(value: $painLevel, in: 0...10, step: 1)
                        .accentColor(colorForPain(Int(painLevel)))

                    HStack {
                        Text("No Pain")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("Worst Pain")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()

                Spacer()

                // Action buttons
                HStack(spacing: 16) {
                    if painLevel > 0 {
                        Button {
                            viewModel.removePain(for: region)
                            isPresented = false
                        } label: {
                            Text("Clear")
                                .fontWeight(.semibold)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.red.opacity(0.1))
                                .foregroundColor(.red)
                                .cornerRadius(12)
                        }
                    }

                    Button {
                        viewModel.setPain(for: region, level: Int(painLevel))
                        isPresented = false
                    } label: {
                        Text("Save")
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                }
                .padding(.horizontal)
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        isPresented = false
                    }
                }
            }
        }
    }

    private func colorForPain(_ level: Int) -> Color {
        switch level {
        case 1...3: return .green
        case 4...6: return .yellow
        case 7...8: return .orange
        case 9...10: return .red
        default: return .gray
        }
    }

    private func painDescription(_ level: Int) -> String {
        switch level {
        case 0: return "No pain"
        case 1...3: return "Mild discomfort"
        case 4...6: return "Moderate pain"
        case 7...8: return "Severe pain"
        case 9...10: return "Extreme pain"
        default: return ""
        }
    }
}

// MARK: - Preview

struct InteractiveBodyMapView_Previews: PreviewProvider {
    static var previews: some View {
        InteractiveBodyMapView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
