//
//  AnkylosaurusMascot.swift
//  InflamAI-Swift
//
//  Cute, friendly Ankylosaurus mascot with 3D appearance
//

import SwiftUI

/// A cute Ankylosaurus mascot with 3D-style rendering for use throughout the app
struct AnkylosaurusMascot: View {
    enum Expression {
        case happy
        case waving
        case excited
        case encouraging
    }

    var expression: Expression = .happy
    var size: CGFloat = 200
    @State private var isAnimating = false

    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let scale = min(size.width, size.height) / 200

            // Draw body (main rounded shape)
            drawBody(in: context, at: center, scale: scale)

            // Draw tail with club
            drawTail(in: context, at: center, scale: scale)

            // Draw legs
            drawLegs(in: context, at: center, scale: scale)

            // Draw head
            drawHead(in: context, at: center, scale: scale)

            // Draw armor plates
            drawArmorPlates(in: context, at: center, scale: scale)

            // Draw face based on expression
            drawFace(in: context, at: center, scale: scale, expression: expression)
        }
        .frame(width: size, height: size)
        .rotation3DEffect(
            .degrees(isAnimating ? 5 : -5),
            axis: (x: 0, y: 1, z: 0)
        )
        .animation(
            .easeInOut(duration: 2)
            .repeatForever(autoreverses: true),
            value: isAnimating
        )
        .onAppear {
            isAnimating = true
        }
    }

    private func drawBody(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        let bodyPath = Path { path in
            let bodyCenter = CGPoint(x: center.x, y: center.y + 10 * scale)
            path.addEllipse(in: CGRect(
                x: bodyCenter.x - 50 * scale,
                y: bodyCenter.y - 30 * scale,
                width: 100 * scale,
                height: 60 * scale
            ))
        }

        // Main body color with gradient for 3D effect
        let gradient = Gradient(colors: [
            Color(red: 0.4, green: 0.7, blue: 0.6),  // Soft teal
            Color(red: 0.3, green: 0.6, blue: 0.5)   // Darker teal
        ])

        context.fill(
            bodyPath,
            with: .linearGradient(
                gradient,
                startPoint: CGPoint(x: center.x - 30 * scale, y: center.y - 20 * scale),
                endPoint: CGPoint(x: center.x + 30 * scale, y: center.y + 40 * scale)
            )
        )

        // Add shadow for depth
        var shadowContext = context
        shadowContext.opacity = 0.3
        shadowContext.fill(
            bodyPath.offsetBy(dx: 2 * scale, dy: 2 * scale),
            with: .color(.black)
        )
    }

    private func drawTail(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        // Tail body
        let tailPath = Path { path in
            let start = CGPoint(x: center.x + 40 * scale, y: center.y + 15 * scale)
            let control1 = CGPoint(x: center.x + 60 * scale, y: center.y + 20 * scale)
            let control2 = CGPoint(x: center.x + 70 * scale, y: center.y)
            let end = CGPoint(x: center.x + 75 * scale, y: center.y - 5 * scale)

            path.move(to: start)
            path.addCurve(to: end, control1: control1, control2: control2)
            path.addLine(to: CGPoint(x: end.x, y: end.y + 8 * scale))
            path.addCurve(
                to: CGPoint(x: start.x, y: start.y + 8 * scale),
                control1: CGPoint(x: control2.x, y: control2.y + 8 * scale),
                control2: CGPoint(x: control1.x, y: control1.y + 8 * scale)
            )
            path.closeSubpath()
        }

        context.fill(
            tailPath,
            with: .linearGradient(
                Gradient(colors: [
                    Color(red: 0.4, green: 0.7, blue: 0.6),
                    Color(red: 0.35, green: 0.65, blue: 0.55)
                ]),
                startPoint: center,
                endPoint: CGPoint(x: center.x + 75 * scale, y: center.y)
            )
        )

        // Tail club (the iconic ankylosaurus feature!)
        let clubCenter = CGPoint(x: center.x + 78 * scale, y: center.y)
        let clubPath = Path { path in
            path.addEllipse(in: CGRect(
                x: clubCenter.x - 8 * scale,
                y: clubCenter.y - 10 * scale,
                width: 16 * scale,
                height: 20 * scale
            ))
        }

        // Club gradient for 3D look
        context.fill(
            clubPath,
            with: .radialGradient(
                Gradient(colors: [
                    Color(red: 0.5, green: 0.75, blue: 0.65),
                    Color(red: 0.3, green: 0.55, blue: 0.45)
                ]),
                center: CGPoint(x: clubCenter.x - 2 * scale, y: clubCenter.y - 3 * scale),
                startRadius: 0,
                endRadius: 12 * scale
            )
        )

        // Club spikes for detail
        let spikePositions = [
            CGPoint(x: clubCenter.x - 6 * scale, y: clubCenter.y - 8 * scale),
            CGPoint(x: clubCenter.x + 6 * scale, y: clubCenter.y - 6 * scale),
            CGPoint(x: clubCenter.x + 4 * scale, y: clubCenter.y + 8 * scale)
        ]

        for spikePos in spikePositions {
            let spikePath = Path { path in
                path.move(to: spikePos)
                path.addLine(to: CGPoint(x: spikePos.x - 3 * scale, y: spikePos.y + 2 * scale))
                path.addLine(to: CGPoint(x: spikePos.x + 2 * scale, y: spikePos.y + 3 * scale))
                path.closeSubpath()
            }
            context.fill(spikePath, with: .color(Color(red: 0.25, green: 0.5, blue: 0.4)))
        }
    }

    private func drawLegs(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        let legPositions = [
            CGPoint(x: center.x - 30 * scale, y: center.y + 30 * scale), // Front left
            CGPoint(x: center.x - 10 * scale, y: center.y + 30 * scale), // Front right
            CGPoint(x: center.x + 15 * scale, y: center.y + 30 * scale), // Back left
            CGPoint(x: center.x + 35 * scale, y: center.y + 30 * scale)  // Back right
        ]

        for legPos in legPositions {
            let legPath = Path { path in
                path.addRoundedRect(
                    in: CGRect(
                        x: legPos.x - 6 * scale,
                        y: legPos.y,
                        width: 12 * scale,
                        height: 20 * scale
                    ),
                    cornerSize: CGSize(width: 4 * scale, height: 4 * scale)
                )
            }

            // Leg gradient
            context.fill(
                legPath,
                with: .linearGradient(
                    Gradient(colors: [
                        Color(red: 0.35, green: 0.65, blue: 0.55),
                        Color(red: 0.3, green: 0.55, blue: 0.45)
                    ]),
                    startPoint: CGPoint(x: legPos.x, y: legPos.y),
                    endPoint: CGPoint(x: legPos.x, y: legPos.y + 20 * scale)
                )
            )

            // Foot
            let footPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: legPos.x - 8 * scale,
                    y: legPos.y + 18 * scale,
                    width: 16 * scale,
                    height: 8 * scale
                ))
            }
            context.fill(footPath, with: .color(Color(red: 0.3, green: 0.55, blue: 0.45)))
        }
    }

    private func drawHead(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        let headCenter = CGPoint(x: center.x - 45 * scale, y: center.y - 10 * scale)

        // Head shape
        let headPath = Path { path in
            path.addEllipse(in: CGRect(
                x: headCenter.x - 20 * scale,
                y: headCenter.y - 18 * scale,
                width: 40 * scale,
                height: 36 * scale
            ))
        }

        // Head gradient for 3D effect
        context.fill(
            headPath,
            with: .radialGradient(
                Gradient(colors: [
                    Color(red: 0.45, green: 0.75, blue: 0.65),
                    Color(red: 0.35, green: 0.65, blue: 0.55)
                ]),
                center: CGPoint(x: headCenter.x - 5 * scale, y: headCenter.y - 8 * scale),
                startRadius: 0,
                endRadius: 25 * scale
            )
        )

        // Snout
        let snoutPath = Path { path in
            path.addEllipse(in: CGRect(
                x: headCenter.x - 35 * scale,
                y: headCenter.y - 8 * scale,
                width: 20 * scale,
                height: 16 * scale
            ))
        }
        context.fill(
            snoutPath,
            with: .linearGradient(
                Gradient(colors: [
                    Color(red: 0.4, green: 0.7, blue: 0.6),
                    Color(red: 0.35, green: 0.6, blue: 0.5)
                ]),
                startPoint: CGPoint(x: headCenter.x - 35 * scale, y: headCenter.y),
                endPoint: CGPoint(x: headCenter.x - 15 * scale, y: headCenter.y)
            )
        )
    }

    private func drawArmorPlates(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        // Armor plates on back for that authentic Ankylosaurus look
        let platePositions = [
            CGPoint(x: center.x - 20 * scale, y: center.y - 15 * scale),
            CGPoint(x: center.x, y: center.y - 18 * scale),
            CGPoint(x: center.x + 20 * scale, y: center.y - 15 * scale),
            CGPoint(x: center.x + 35 * scale, y: center.y - 10 * scale)
        ]

        for platePos in platePositions {
            let platePath = Path { path in
                path.move(to: platePos)
                path.addLine(to: CGPoint(x: platePos.x - 6 * scale, y: platePos.y + 8 * scale))
                path.addLine(to: CGPoint(x: platePos.x + 6 * scale, y: platePos.y + 8 * scale))
                path.closeSubpath()
            }

            // Plate with gradient
            context.fill(
                platePath,
                with: .linearGradient(
                    Gradient(colors: [
                        Color(red: 0.5, green: 0.8, blue: 0.7),
                        Color(red: 0.35, green: 0.6, blue: 0.5)
                    ]),
                    startPoint: platePos,
                    endPoint: CGPoint(x: platePos.x, y: platePos.y + 8 * scale)
                )
            )
        }
    }

    private func drawFace(in context: GraphicsContext, at center: CGPoint, scale: CGFloat, expression: Expression) {
        let headCenter = CGPoint(x: center.x - 45 * scale, y: center.y - 10 * scale)

        // Eyes (always friendly and big!)
        let leftEyeCenter = CGPoint(x: headCenter.x - 10 * scale, y: headCenter.y - 5 * scale)
        let rightEyeCenter = CGPoint(x: headCenter.x + 5 * scale, y: headCenter.y - 5 * scale)

        for eyeCenter in [leftEyeCenter, rightEyeCenter] {
            // Eye white
            let eyeWhitePath = Path { path in
                path.addEllipse(in: CGRect(
                    x: eyeCenter.x - 6 * scale,
                    y: eyeCenter.y - 7 * scale,
                    width: 12 * scale,
                    height: 14 * scale
                ))
            }
            context.fill(eyeWhitePath, with: .color(.white))

            // Pupil
            let pupilPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: eyeCenter.x - 3 * scale,
                    y: eyeCenter.y - 3 * scale,
                    width: 6 * scale,
                    height: 6 * scale
                ))
            }
            context.fill(pupilPath, with: .color(Color(red: 0.2, green: 0.3, blue: 0.4)))

            // Highlight for sparkle
            let highlightPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: eyeCenter.x - 1 * scale,
                    y: eyeCenter.y - 2 * scale,
                    width: 2 * scale,
                    height: 2 * scale
                ))
            }
            context.fill(highlightPath, with: .color(.white))
        }

        // Mouth based on expression
        switch expression {
        case .happy, .encouraging:
            // Happy smile
            let smilePath = Path { path in
                path.move(to: CGPoint(x: headCenter.x - 15 * scale, y: headCenter.y + 5 * scale))
                path.addQuadCurve(
                    to: CGPoint(x: headCenter.x + 5 * scale, y: headCenter.y + 5 * scale),
                    control: CGPoint(x: headCenter.x - 5 * scale, y: headCenter.y + 10 * scale)
                )
            }
            context.stroke(
                smilePath,
                with: .color(Color(red: 0.2, green: 0.4, blue: 0.35)),
                lineWidth: 2 * scale
            )

        case .excited:
            // Big excited smile
            let excitedPath = Path { path in
                path.move(to: CGPoint(x: headCenter.x - 15 * scale, y: headCenter.y + 5 * scale))
                path.addQuadCurve(
                    to: CGPoint(x: headCenter.x + 5 * scale, y: headCenter.y + 5 * scale),
                    control: CGPoint(x: headCenter.x - 5 * scale, y: headCenter.y + 12 * scale)
                )
            }
            context.stroke(
                excitedPath,
                with: .color(Color(red: 0.2, green: 0.4, blue: 0.35)),
                lineWidth: 2.5 * scale
            )

        case .waving:
            // Friendly smile
            let wavePath = Path { path in
                path.move(to: CGPoint(x: headCenter.x - 12 * scale, y: headCenter.y + 5 * scale))
                path.addQuadCurve(
                    to: CGPoint(x: headCenter.x + 3 * scale, y: headCenter.y + 5 * scale),
                    control: CGPoint(x: headCenter.x - 5 * scale, y: headCenter.y + 9 * scale)
                )
            }
            context.stroke(
                wavePath,
                with: .color(Color(red: 0.2, green: 0.4, blue: 0.35)),
                lineWidth: 2 * scale
            )
        }

        // Rosy cheeks for extra cuteness
        let leftCheek = CGPoint(x: headCenter.x - 18 * scale, y: headCenter.y + 3 * scale)
        let rightCheek = CGPoint(x: headCenter.x + 8 * scale, y: headCenter.y + 3 * scale)

        for cheekCenter in [leftCheek, rightCheek] {
            let cheekPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: cheekCenter.x - 4 * scale,
                    y: cheekCenter.y - 3 * scale,
                    width: 8 * scale,
                    height: 6 * scale
                ))
            }
            var cheekContext = context
            cheekContext.opacity = 0.3
            cheekContext.fill(cheekPath, with: .color(Color(red: 1.0, green: 0.6, blue: 0.7)))
        }
    }
}

// MARK: - Animation Modifiers

extension AnkylosaurusMascot {
    /// Makes the mascot bounce up and down
    func bouncing() -> some View {
        modifier(BouncingModifier())
    }

    /// Makes the mascot wave
    func waving() -> some View {
        modifier(WavingModifier())
    }
}

private struct BouncingModifier: ViewModifier {
    @State private var bounce = false

    func body(content: Content) -> some View {
        content
            .offset(y: bounce ? -10 : 0)
            .animation(
                .easeInOut(duration: 0.6)
                .repeatForever(autoreverses: true),
                value: bounce
            )
            .onAppear {
                bounce = true
            }
    }
}

private struct WavingModifier: ViewModifier {
    @State private var wave = false

    func body(content: Content) -> some View {
        content
            .rotationEffect(.degrees(wave ? -10 : 10))
            .animation(
                .easeInOut(duration: 0.5)
                .repeatForever(autoreverses: true),
                value: wave
            )
            .onAppear {
                wave = true
            }
    }
}

// MARK: - Preview

#Preview("Happy") {
    VStack {
        AnkylosaurusMascot(expression: .happy, size: 250)
        Text("Happy Anky")
            .font(.headline)
    }
    .padding()
}

#Preview("All Expressions") {
    ScrollView {
        VStack(spacing: 30) {
            VStack {
                AnkylosaurusMascot(expression: .happy, size: 200)
                Text("Happy")
                    .font(.headline)
            }

            VStack {
                AnkylosaurusMascot(expression: .waving, size: 200)
                    .bouncing()
                Text("Waving")
                    .font(.headline)
            }

            VStack {
                AnkylosaurusMascot(expression: .excited, size: 200)
                Text("Excited")
                    .font(.headline)
            }

            VStack {
                AnkylosaurusMascot(expression: .encouraging, size: 200)
                    .waving()
                Text("Encouraging")
                    .font(.headline)
            }
        }
        .padding()
    }
}
