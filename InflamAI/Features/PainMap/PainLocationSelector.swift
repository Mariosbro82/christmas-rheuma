//
//  PainLocationSelector.swift
//  InflamAI
//
//  Pixel-perfect pain location selector matching medical design spec
//

import SwiftUI
import CoreData

struct PainLocationSelector: View {
    @StateObject private var viewModel: PainMapViewModel
    @State private var selectedView: BodyView = .front

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: PainMapViewModel(context: context))
    }

    var body: some View {
        ZStack {
            // Background - very pale lavender-gray
            Color(hex: "f5f6fa")
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 0) {
                    // Header
                    Text("Select Pain Locations")
                        .font(.system(size: 20, weight: .bold, design: .default))
                        .foregroundColor(Color(hex: "1f1f1f"))
                        .frame(maxWidth: .infinity)
                        .padding(.top, 24)
                        .padding(.bottom, 20)

                    // Segmented Control
                    segmentedControl
                        .padding(.horizontal, 20)
                        .padding(.bottom, 24)

                    // Body diagram panel
                    bodyDiagramPanel
                        .padding(.horizontal, 20)
                        .padding(.bottom, 24)

                    // Save button
                    saveButton
                        .padding(.horizontal, 20)
                        .padding(.bottom, 40)
                }
            }
        }
    }

    // MARK: - Segmented Control

    private var segmentedControl: some View {
        HStack(spacing: 0) {
            // Front segment
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    selectedView = .front
                }
            } label: {
                Text("Front")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(selectedView == .front ? Color(hex: "1f1f1f") : Color.gray)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(
                        selectedView == .front ?
                        Color.white :
                        Color(hex: "e6e7eb")
                    )
                    .cornerRadius(12, corners: [.topLeft, .bottomLeft])
            }

            // Back segment
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    selectedView = .back
                }
            } label: {
                Text("Back")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(selectedView == .back ? Color(hex: "1f1f1f") : Color.gray)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(
                        selectedView == .back ?
                        Color.white :
                        Color(hex: "e6e7eb")
                    )
                    .cornerRadius(12, corners: [.topRight, .bottomRight])
            }
        }
        .background(Color(hex: "e6e7eb"))
        .cornerRadius(12)
        .shadow(color: selectedView == .front ? Color.black.opacity(0.08) : Color.clear, radius: 4, x: -2, y: 0)
        .shadow(color: selectedView == .back ? Color.black.opacity(0.08) : Color.clear, radius: 4, x: 2, y: 0)
    }

    // MARK: - Body Diagram Panel

    private var bodyDiagramPanel: some View {
        GeometryReader { geometry in
            ZStack {
                // Panel background
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(hex: "f0f1f5"))
                    .shadow(color: Color.black.opacity(0.06), radius: 8, x: 0, y: 2)

                // 29 circles arranged as body
                if selectedView == .front {
                    frontBodyCircles(in: geometry.size)
                } else {
                    backBodyCircles(in: geometry.size)
                }
            }
        }
        .frame(height: UIScreen.main.bounds.height * 0.75)
    }

    // MARK: - Front Body Circles (29 points)

    private func frontBodyCircles(in size: CGSize) -> some View {
        let centerX = size.width / 2
        let circleSpacing: CGFloat = 18
        let horizontalSpacing: CGFloat = 20
        let circleSize: CGFloat = 26

        return ZStack {
            // HEAD (1)
            painCircle(id: 1, position: CGPoint(x: centerX, y: 40))

            // NECK & UPPER CHEST (3)
            painCircle(id: 2, position: CGPoint(x: centerX, y: 40 + circleSpacing * 1.5))
            painCircle(id: 3, position: CGPoint(x: centerX, y: 40 + circleSpacing * 2.5))
            painCircle(id: 4, position: CGPoint(x: centerX, y: 40 + circleSpacing * 3.5))

            // SHOULDERS (4 - 2 on each side)
            painCircle(id: 5, position: CGPoint(x: centerX - horizontalSpacing * 2, y: 40 + circleSpacing * 2.5))
            painCircle(id: 6, position: CGPoint(x: centerX + horizontalSpacing * 2, y: 40 + circleSpacing * 2.5))
            painCircle(id: 7, position: CGPoint(x: centerX - horizontalSpacing * 2.5, y: 40 + circleSpacing * 3.5))
            painCircle(id: 8, position: CGPoint(x: centerX + horizontalSpacing * 2.5, y: 40 + circleSpacing * 3.5))

            // ARMS (8 - 4 on each side, descending)
            // Left arm
            painCircle(id: 9, position: CGPoint(x: centerX - horizontalSpacing * 2.8, y: 40 + circleSpacing * 4.5))
            painCircle(id: 10, position: CGPoint(x: centerX - horizontalSpacing * 3, y: 40 + circleSpacing * 5.5))
            painCircle(id: 11, position: CGPoint(x: centerX - horizontalSpacing * 3, y: 40 + circleSpacing * 6.5))
            painCircle(id: 12, position: CGPoint(x: centerX - horizontalSpacing * 3, y: 40 + circleSpacing * 7.5))

            // Right arm
            painCircle(id: 13, position: CGPoint(x: centerX + horizontalSpacing * 2.8, y: 40 + circleSpacing * 4.5))
            painCircle(id: 14, position: CGPoint(x: centerX + horizontalSpacing * 3, y: 40 + circleSpacing * 5.5))
            painCircle(id: 15, position: CGPoint(x: centerX + horizontalSpacing * 3, y: 40 + circleSpacing * 6.5))
            painCircle(id: 16, position: CGPoint(x: centerX + horizontalSpacing * 3, y: 40 + circleSpacing * 7.5))

            // CENTRAL COLUMN - Sternum to pelvis (6)
            painCircle(id: 17, position: CGPoint(x: centerX, y: 40 + circleSpacing * 4.5))
            painCircle(id: 18, position: CGPoint(x: centerX, y: 40 + circleSpacing * 5.5))
            painCircle(id: 19, position: CGPoint(x: centerX, y: 40 + circleSpacing * 6.5))
            painCircle(id: 20, position: CGPoint(x: centerX, y: 40 + circleSpacing * 7.5))
            painCircle(id: 21, position: CGPoint(x: centerX, y: 40 + circleSpacing * 8.5))
            painCircle(id: 22, position: CGPoint(x: centerX, y: 40 + circleSpacing * 9.5))

            // LEGS (8 - 4 on each side)
            // Left leg
            painCircle(id: 23, position: CGPoint(x: centerX - horizontalSpacing, y: 40 + circleSpacing * 10.5))
            painCircle(id: 24, position: CGPoint(x: centerX - horizontalSpacing, y: 40 + circleSpacing * 11.5))
            painCircle(id: 25, position: CGPoint(x: centerX - horizontalSpacing, y: 40 + circleSpacing * 12.5))
            painCircle(id: 26, position: CGPoint(x: centerX - horizontalSpacing, y: 40 + circleSpacing * 13.5))

            // Right leg
            painCircle(id: 27, position: CGPoint(x: centerX + horizontalSpacing, y: 40 + circleSpacing * 10.5))
            painCircle(id: 28, position: CGPoint(x: centerX + horizontalSpacing, y: 40 + circleSpacing * 11.5))
            painCircle(id: 29, position: CGPoint(x: centerX + horizontalSpacing, y: 40 + circleSpacing * 12.5))
            painCircle(id: 30, position: CGPoint(x: centerX + horizontalSpacing, y: 40 + circleSpacing * 13.5))
        }
    }

    // MARK: - Back Body Circles (29 points)

    private func backBodyCircles(in size: CGSize) -> some View {
        let centerX = size.width / 2
        let circleSpacing: CGFloat = 18
        let horizontalSpacing: CGFloat = 20

        return ZStack {
            // HEAD (1)
            painCircle(id: 31, position: CGPoint(x: centerX, y: 40))

            // NECK & UPPER BACK (3)
            painCircle(id: 32, position: CGPoint(x: centerX, y: 40 + circleSpacing * 1.5))
            painCircle(id: 33, position: CGPoint(x: centerX, y: 40 + circleSpacing * 2.5))
            painCircle(id: 34, position: CGPoint(x: centerX, y: 40 + circleSpacing * 3.5))

            // SHOULDERS (4)
            painCircle(id: 35, position: CGPoint(x: centerX - horizontalSpacing * 2, y: 40 + circleSpacing * 2.5))
            painCircle(id: 36, position: CGPoint(x: centerX + horizontalSpacing * 2, y: 40 + circleSpacing * 2.5))
            painCircle(id: 37, position: CGPoint(x: centerX - horizontalSpacing * 2.5, y: 40 + circleSpacing * 3.5))
            painCircle(id: 38, position: CGPoint(x: centerX + horizontalSpacing * 2.5, y: 40 + circleSpacing * 3.5))

            // ARMS (8)
            painCircle(id: 39, position: CGPoint(x: centerX - horizontalSpacing * 2.8, y: 40 + circleSpacing * 4.5))
            painCircle(id: 40, position: CGPoint(x: centerX - horizontalSpacing * 3, y: 40 + circleSpacing * 5.5))
            painCircle(id: 41, position: CGPoint(x: centerX - horizontalSpacing * 3, y: 40 + circleSpacing * 6.5))
            painCircle(id: 42, position: CGPoint(x: centerX - horizontalSpacing * 3, y: 40 + circleSpacing * 7.5))

            painCircle(id: 43, position: CGPoint(x: centerX + horizontalSpacing * 2.8, y: 40 + circleSpacing * 4.5))
            painCircle(id: 44, position: CGPoint(x: centerX + horizontalSpacing * 3, y: 40 + circleSpacing * 5.5))
            painCircle(id: 45, position: CGPoint(x: centerX + horizontalSpacing * 3, y: 40 + circleSpacing * 6.5))
            painCircle(id: 46, position: CGPoint(x: centerX + horizontalSpacing * 3, y: 40 + circleSpacing * 7.5))

            // SPINE - Thoracic to Sacral (6)
            painCircle(id: 47, position: CGPoint(x: centerX, y: 40 + circleSpacing * 4.5))
            painCircle(id: 48, position: CGPoint(x: centerX, y: 40 + circleSpacing * 5.5))
            painCircle(id: 49, position: CGPoint(x: centerX, y: 40 + circleSpacing * 6.5))
            painCircle(id: 50, position: CGPoint(x: centerX, y: 40 + circleSpacing * 7.5))
            painCircle(id: 51, position: CGPoint(x: centerX, y: 40 + circleSpacing * 8.5))
            painCircle(id: 52, position: CGPoint(x: centerX, y: 40 + circleSpacing * 9.5))

            // LEGS (8)
            painCircle(id: 53, position: CGPoint(x: centerX - horizontalSpacing, y: 40 + circleSpacing * 10.5))
            painCircle(id: 54, position: CGPoint(x: centerX - horizontalSpacing, y: 40 + circleSpacing * 11.5))
            painCircle(id: 55, position: CGPoint(x: centerX - horizontalSpacing, y: 40 + circleSpacing * 12.5))
            painCircle(id: 56, position: CGPoint(x: centerX - horizontalSpacing, y: 40 + circleSpacing * 13.5))

            painCircle(id: 57, position: CGPoint(x: centerX + horizontalSpacing, y: 40 + circleSpacing * 10.5))
            painCircle(id: 58, position: CGPoint(x: centerX + horizontalSpacing, y: 40 + circleSpacing * 11.5))
            painCircle(id: 59, position: CGPoint(x: centerX + horizontalSpacing, y: 40 + circleSpacing * 12.5))
            painCircle(id: 60, position: CGPoint(x: centerX + horizontalSpacing, y: 40 + circleSpacing * 13.5))
        }
    }

    // MARK: - Pain Circle

    private func painCircle(id: Int, position: CGPoint) -> some View {
        let isSelected = viewModel.isLocationSelected(id)
        let intensity = viewModel.getIntensity(id)

        return Circle()
            .strokeBorder(
                isSelected ? fillColorForIntensity(intensity) : Color(hex: "4A90E2"),
                lineWidth: 2
            )
            .background(
                Circle()
                    .fill(isSelected ? fillColorForIntensity(intensity) : Color.clear)
            )
            .frame(width: 26, height: 26)
            .position(position)
            .onTapGesture {
                viewModel.toggleLocation(id)
            }
    }

    private func fillColorForIntensity(_ intensity: Int) -> Color {
        switch intensity {
        case 1...3: return Color.yellow.opacity(0.6)
        case 4...6: return Color.orange.opacity(0.7)
        case 7...10: return Color.red.opacity(0.8)
        default: return Color.blue.opacity(0.3)
        }
    }

    // MARK: - Save Button

    private var saveButton: some View {
        Button {
            Task {
                await viewModel.savePainMap()
            }
        } label: {
            HStack {
                Image(systemName: "checkmark.circle.fill")
                Text("Save Pain Map")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(viewModel.hasSelections ? Color.blue : Color.gray.opacity(0.3))
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(!viewModel.hasSelections)
    }
}

// MARK: - Body View Enum

enum BodyView {
    case front
    case back
}

// MARK: - Helper Extensions
// NOTE: Color(hex:) moved to InflamAIDesignSystem.swift

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

// MARK: - Preview

struct PainLocationSelector_Previews: PreviewProvider {
    static var previews: some View {
        PainLocationSelector()
    }
}
