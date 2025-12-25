//
//  AnkyAnimatedMascot.swift
//  InflamAI
//
//  Premium 2.5D Animated Mascot with State Machine
//  Duolingo-style interactive character animation
//
//  Architecture:
//  - State machine for reactive expressions
//  - Parallax depth layers for 2.5D effect
//  - Smooth expression blending
//  - Eye tracking and look-at system
//

import SwiftUI
import Combine

// MARK: - Mascot State Machine

/// All possible states for Anky the Ankylosaurus
enum AnkyState: String, CaseIterable {
    case sleeping
    case waking
    case idle
    case attentive
    case happy
    case celebrating
    case encouraging
    case concerned
    case proud
    case sympathetic
    case waving
    case explaining
    case curious
    case thinking

    var idleAnimationEnabled: Bool {
        switch self {
        case .sleeping: return false
        case .waking: return false
        default: return true
        }
    }
}

/// Expression configuration for face rendering
struct AnkyExpression {
    var eyeOpenness: CGFloat = 1.0        // 0 = closed, 1 = open
    var eyeSparkle: Bool = false
    var eyeSize: CGFloat = 1.0
    var pupilOffset: CGPoint = .zero       // Look direction
    var browRaise: CGFloat = 0             // -1 = worried, 0 = neutral, 1 = raised
    var browTilt: CGFloat = 0              // Asymmetric brow
    var mouthCurve: CGFloat = 0.5          // 0 = sad, 0.5 = neutral, 1 = happy
    var mouthOpen: CGFloat = 0             // 0 = closed, 1 = open
    var cheekBlush: CGFloat = 0            // 0 = none, 1 = full blush

    static let sleeping = AnkyExpression(
        eyeOpenness: 0,
        eyeSparkle: false,
        browRaise: -0.3,
        mouthCurve: 0.5
    )

    static let waking = AnkyExpression(
        eyeOpenness: 0.3,
        eyeSparkle: false,
        browRaise: 0,
        mouthCurve: 0.4,
        mouthOpen: 0.3
    )

    static let idle = AnkyExpression(
        eyeOpenness: 1.0,
        eyeSparkle: false,
        browRaise: 0,
        mouthCurve: 0.6
    )

    static let happy = AnkyExpression(
        eyeOpenness: 0.9,
        eyeSparkle: true,
        eyeSize: 1.1,
        browRaise: 0.3,
        mouthCurve: 1.0,
        cheekBlush: 0.5
    )

    static let celebrating = AnkyExpression(
        eyeOpenness: 1.0,
        eyeSparkle: true,
        eyeSize: 1.2,
        browRaise: 0.5,
        mouthCurve: 1.0,
        mouthOpen: 0.4,
        cheekBlush: 0.8
    )

    static let encouraging = AnkyExpression(
        eyeOpenness: 1.0,
        eyeSparkle: false,
        eyeSize: 1.0,
        browRaise: 0.2,
        mouthCurve: 0.7,
        cheekBlush: 0.2
    )

    static let concerned = AnkyExpression(
        eyeOpenness: 1.1,
        eyeSparkle: false,
        eyeSize: 1.05,
        browRaise: -0.4,
        browTilt: 0.3,
        mouthCurve: 0.3
    )

    static let proud = AnkyExpression(
        eyeOpenness: 0.85,
        eyeSparkle: true,
        eyeSize: 1.0,
        browRaise: 0.4,
        mouthCurve: 0.9,
        cheekBlush: 0.6
    )

    static let waving = AnkyExpression(
        eyeOpenness: 1.0,
        eyeSparkle: true,
        eyeSize: 1.1,
        browRaise: 0.3,
        mouthCurve: 0.85,
        cheekBlush: 0.3
    )

    static let curious = AnkyExpression(
        eyeOpenness: 1.1,
        eyeSparkle: false,
        eyeSize: 1.15,
        browRaise: 0.2,
        browTilt: 0.4,
        mouthCurve: 0.5
    )

    static let thinking = AnkyExpression(
        eyeOpenness: 0.9,
        eyeSparkle: false,
        pupilOffset: CGPoint(x: 0.3, y: -0.2),
        browRaise: 0.1,
        mouthCurve: 0.45
    )

    static func expression(for state: AnkyState) -> AnkyExpression {
        switch state {
        case .sleeping: return .sleeping
        case .waking: return .waking
        case .idle, .attentive: return .idle
        case .happy: return .happy
        case .celebrating: return .celebrating
        case .encouraging: return .encouraging
        case .concerned, .sympathetic: return .concerned
        case .proud: return .proud
        case .waving: return .waving
        case .curious, .explaining: return .curious
        case .thinking: return .thinking
        }
    }
}

// MARK: - Anky View Model

@MainActor
class AnkyViewModel: ObservableObject {
    // MARK: - Published State
    @Published var currentState: AnkyState = .idle
    @Published var expression: AnkyExpression = .idle
    @Published var lookAtPoint: CGPoint = .zero
    @Published var isLookingAtUser: Bool = true

    // Animation States
    @Published var bounceOffset: CGFloat = 0
    @Published var tailWagAngle: CGFloat = 0
    @Published var bodyRotation: CGFloat = 0
    @Published var armWaveAngle: CGFloat = 0
    @Published var breathScale: CGFloat = 1.0
    @Published var celebrationJump: CGFloat = 0
    @Published var celebrationSpin: CGFloat = 0

    // Blink System
    @Published var blinkProgress: CGFloat = 0
    private var blinkTimer: Timer?
    private var lastBlinkTime: Date = Date()

    // MARK: - Private
    private var animationTimers: [Timer] = []
    private var cancellables = Set<AnyCancellable>()

    init() {
        setupAnimations()
    }

    deinit {
        // Use Task to call main actor method from deinit
        let timers = animationTimers
        Task { @MainActor in
            timers.forEach { $0.invalidate() }
        }
    }

    // MARK: - State Transitions

    func transitionTo(_ newState: AnkyState, animated: Bool = true) {
        guard newState != currentState else { return }

        let oldState = currentState
        currentState = newState

        // Blend expression
        let targetExpression = AnkyExpression.expression(for: newState)

        if animated {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                expression = targetExpression
            }
        } else {
            expression = targetExpression
        }

        // Trigger state-specific animations
        switch newState {
        case .waving:
            triggerWaveAnimation()
        case .celebrating:
            triggerCelebrationAnimation()
        case .sleeping:
            stopIdleAnimations()
            startBreathingAnimation()
        case .waking:
            triggerWakeAnimation()
        default:
            if oldState == .sleeping || oldState == .waking {
                startIdleAnimations()
            }
        }
    }

    // MARK: - Look At System

    func lookAt(_ point: CGPoint, in size: CGSize) {
        let normalizedX = (point.x / size.width - 0.5) * 2  // -1 to 1
        let normalizedY = (point.y / size.height - 0.5) * 2 // -1 to 1

        withAnimation(.easeOut(duration: 0.2)) {
            expression.pupilOffset = CGPoint(
                x: normalizedX * 0.3,
                y: normalizedY * 0.3
            )
        }
    }

    func resetLookAt() {
        withAnimation(.easeOut(duration: 0.3)) {
            expression.pupilOffset = .zero
        }
    }

    // MARK: - Animation Setup

    private func setupAnimations() {
        startIdleAnimations()
        startBlinkSystem()
    }

    private func startIdleAnimations() {
        // Bounce animation
        Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            Task { @MainActor in
                if self.currentState.idleAnimationEnabled {
                    let time = Date().timeIntervalSinceReferenceDate
                    self.bounceOffset = sin(time * 2.5) * 6
                }
            }
        }

        // Tail wag animation
        Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            Task { @MainActor in
                if self.currentState.idleAnimationEnabled {
                    let time = Date().timeIntervalSinceReferenceDate
                    let baseSpeed = self.currentState == .happy || self.currentState == .celebrating ? 4.0 : 2.0
                    let baseAmplitude = self.currentState == .happy || self.currentState == .celebrating ? 12.0 : 6.0
                    self.tailWagAngle = sin(time * baseSpeed) * baseAmplitude
                }
            }
        }

        // Subtle body sway
        Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            Task { @MainActor in
                if self.currentState.idleAnimationEnabled {
                    let time = Date().timeIntervalSinceReferenceDate
                    self.bodyRotation = sin(time * 1.2) * 1.5
                }
            }
        }
    }

    private func startBreathingAnimation() {
        Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            Task { @MainActor in
                if self.currentState == .sleeping {
                    let time = Date().timeIntervalSinceReferenceDate
                    self.breathScale = 1.0 + sin(time * 1.0) * 0.03
                }
            }
        }
    }

    private func stopIdleAnimations() {
        bounceOffset = 0
        tailWagAngle = 0
        bodyRotation = 0
    }

    private func startBlinkSystem() {
        blinkTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                self.checkForBlink()
            }
        }
    }

    private func checkForBlink() {
        guard currentState != .sleeping else { return }

        let timeSinceLastBlink = Date().timeIntervalSince(lastBlinkTime)
        let blinkInterval = Double.random(in: 2.5...4.5)

        if timeSinceLastBlink > blinkInterval {
            triggerBlink()
            lastBlinkTime = Date()
        }
    }

    private func triggerBlink() {
        withAnimation(.easeIn(duration: 0.08)) {
            blinkProgress = 1.0
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            withAnimation(.easeOut(duration: 0.1)) {
                self.blinkProgress = 0
            }
        }
    }

    // MARK: - Special Animations

    private func triggerWaveAnimation() {
        // Wave arm up
        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
            armWaveAngle = -30
        }

        // Wave back and forth
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            withAnimation(.easeInOut(duration: 0.4).repeatCount(3, autoreverses: true)) {
                self.armWaveAngle = 30
            }
        }

        // Return to rest
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                self.armWaveAngle = 0
            }
        }
    }

    func triggerCelebrationAnimation() {
        // Jump up
        withAnimation(.spring(response: 0.25, dampingFraction: 0.5)) {
            celebrationJump = -40
            celebrationSpin = 360
        }

        // Land
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
                self.celebrationJump = 0
            }
        }

        // Reset spin
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            self.celebrationSpin = 0
        }
    }

    private func triggerWakeAnimation() {
        // Gradual eye opening
        withAnimation(.easeInOut(duration: 0.8)) {
            expression = .waking
        }

        // Stretch
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.6)) {
                self.breathScale = 1.08
            }
        }

        // Settle to idle
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                self.breathScale = 1.0
                self.expression = .idle
            }
            self.currentState = .idle
            self.startIdleAnimations()
        }
    }

    private func stopAllAnimations() {
        animationTimers.forEach { $0.invalidate() }
        animationTimers.removeAll()
        blinkTimer?.invalidate()
    }
}

// MARK: - Anky Animated Mascot View

struct AnkyAnimatedMascot: View {
    @StateObject private var viewModel = AnkyViewModel()

    var size: CGFloat = 200
    var state: AnkyState = .idle
    var showShadow: Bool = true
    var onTap: (() -> Void)?

    var body: some View {
        ZStack {
            // Shadow layer
            if showShadow {
                Ellipse()
                    .fill(
                        RadialGradient(
                            colors: [Color.black.opacity(0.15), Color.clear],
                            center: .center,
                            startRadius: 0,
                            endRadius: size * 0.4
                        )
                    )
                    .frame(width: size * 0.6, height: size * 0.15)
                    .offset(y: size * 0.42 - viewModel.celebrationJump * 0.3)
                    .scaleEffect(x: 1 - viewModel.celebrationJump * 0.005)
            }

            // Main character
            AnkyCharacterView(
                viewModel: viewModel,
                size: size
            )
            .offset(y: viewModel.bounceOffset + viewModel.celebrationJump)
            .rotationEffect(.degrees(viewModel.bodyRotation))
            .rotation3DEffect(
                .degrees(viewModel.celebrationSpin),
                axis: (x: 0, y: 1, z: 0)
            )
            .scaleEffect(viewModel.breathScale)
        }
        .frame(width: size, height: size)
        .contentShape(Rectangle())
        .onTapGesture {
            HapticFeedback.light()
            viewModel.transitionTo(.happy)
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                viewModel.transitionTo(.idle)
            }
            onTap?()
        }
        .onChange(of: state) { newState in
            viewModel.transitionTo(newState)
        }
        .onAppear {
            viewModel.transitionTo(state, animated: false)
        }
    }
}

// MARK: - Character Rendering View
// REDESIGNED: Cute upright dinosaur matching Duolingo/reference style

private struct AnkyCharacterView: View {
    @ObservedObject var viewModel: AnkyViewModel
    let size: CGFloat

    // Vibrant teal color palette (matching reference dinos exactly)
    private let mainTeal = Color(red: 0.15, green: 0.78, blue: 0.70)        // Bright teal
    private let darkTeal = Color(red: 0.08, green: 0.52, blue: 0.48)        // Shadow
    private let lightTeal = Color(red: 0.45, green: 0.92, blue: 0.85)       // Highlight
    private let bellyColor = Color(red: 0.85, green: 0.98, blue: 0.95)      // Light cream belly
    private let spikeColor = Color(red: 0.12, green: 0.65, blue: 0.58)      // Spike color

    var body: some View {
        Canvas { context, canvasSize in
            let center = CGPoint(x: canvasSize.width / 2, y: canvasSize.height / 2 + 8)
            let scale = min(canvasSize.width, canvasSize.height) / 200

            // Draw in back-to-front order
            drawTail(in: context, at: center, scale: scale)
            drawSpikes(in: context, at: center, scale: scale)
            drawLegs(in: context, at: center, scale: scale)
            drawBody(in: context, at: center, scale: scale)
            drawArms(in: context, at: center, scale: scale)
            drawFace(in: context, at: center, scale: scale)
        }
        .frame(width: size, height: size)
    }

    // MARK: - Body (Compact, round like reference)

    private func drawBody(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        // Main body - compact oval, very round
        let bodyPath = Path { path in
            path.addEllipse(in: CGRect(
                x: center.x - 55 * scale,
                y: center.y - 50 * scale,
                width: 110 * scale,
                height: 100 * scale
            ))
        }

        // Rich 3D gradient
        context.fill(
            bodyPath,
            with: .radialGradient(
                Gradient(stops: [
                    .init(color: lightTeal, location: 0.0),
                    .init(color: mainTeal, location: 0.4),
                    .init(color: darkTeal, location: 1.0)
                ]),
                center: CGPoint(x: center.x - 20 * scale, y: center.y - 30 * scale),
                startRadius: 0,
                endRadius: 75 * scale
            )
        )

        // Big highlight blob (top-left shine)
        let shinePath = Path { path in
            path.addEllipse(in: CGRect(
                x: center.x - 40 * scale,
                y: center.y - 42 * scale,
                width: 35 * scale,
                height: 28 * scale
            ))
        }
        var shineCtx = context
        shineCtx.opacity = 0.4
        shineCtx.fill(shinePath, with: .color(.white))

        // Belly (cream colored, front-facing)
        let bellyPath = Path { path in
            path.addEllipse(in: CGRect(
                x: center.x - 38 * scale,
                y: center.y - 20 * scale,
                width: 76 * scale,
                height: 65 * scale
            ))
        }
        context.fill(
            bellyPath,
            with: .radialGradient(
                Gradient(colors: [bellyColor, bellyColor.opacity(0.6), mainTeal.opacity(0.2)]),
                center: CGPoint(x: center.x, y: center.y),
                startRadius: 0,
                endRadius: 45 * scale
            )
        )
    }

    // MARK: - Spikes (Softer, rounder like reference)

    private func drawSpikes(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        // 3 rounded spikes on top (like Duo's crown or the reference dinos)
        let spikeData: [(x: CGFloat, y: CGFloat, w: CGFloat, h: CGFloat, rot: CGFloat)] = [
            (-22, -62, 20, 28, -12),   // Left spike
            (0, -72, 24, 35, 0),        // Center spike (biggest)
            (22, -62, 20, 28, 12),      // Right spike
        ]

        for spike in spikeData {
            var ctx = context
            let spikePos = CGPoint(x: center.x + spike.x * scale, y: center.y + spike.y * scale)

            ctx.translateBy(x: spikePos.x, y: spikePos.y)
            ctx.rotate(by: .degrees(spike.rot))
            ctx.translateBy(x: -spikePos.x, y: -spikePos.y)

            // Rounded spike (more like a petal/leaf shape)
            let spikePath = Path { path in
                path.addEllipse(in: CGRect(
                    x: spikePos.x - spike.w / 2 * scale,
                    y: spikePos.y - spike.h * scale,
                    width: spike.w * scale,
                    height: spike.h * scale
                ))
            }

            ctx.fill(
                spikePath,
                with: .linearGradient(
                    Gradient(colors: [lightTeal, spikeColor]),
                    startPoint: CGPoint(x: spikePos.x, y: spikePos.y - spike.h * scale),
                    endPoint: CGPoint(x: spikePos.x, y: spikePos.y)
                )
            )

            // Highlight on spike
            let hlPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: spikePos.x - 4 * scale,
                    y: spikePos.y - spike.h * 0.7 * scale,
                    width: 6 * scale,
                    height: 8 * scale
                ))
            }
            var hlCtx = ctx
            hlCtx.opacity = 0.5
            hlCtx.fill(hlPath, with: .color(.white))
        }
    }

    // MARK: - Tail (Simple, cute)

    private func drawTail(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        var ctx = context
        let tailBase = CGPoint(x: center.x + 40 * scale, y: center.y + 20 * scale)

        ctx.translateBy(x: tailBase.x, y: tailBase.y)
        ctx.rotate(by: .degrees(viewModel.tailWagAngle * 0.4))
        ctx.translateBy(x: -tailBase.x, y: -tailBase.y)

        // Short cute tail
        let tailPath = Path { path in
            path.move(to: tailBase)
            path.addQuadCurve(
                to: CGPoint(x: center.x + 72 * scale, y: center.y + 10 * scale),
                control: CGPoint(x: center.x + 60 * scale, y: center.y + 30 * scale)
            )
            path.addQuadCurve(
                to: tailBase,
                control: CGPoint(x: center.x + 55 * scale, y: center.y + 5 * scale)
            )
            path.closeSubpath()
        }

        ctx.fill(tailPath, with: .linearGradient(
            Gradient(colors: [mainTeal, darkTeal]),
            startPoint: tailBase,
            endPoint: CGPoint(x: center.x + 72 * scale, y: center.y + 10 * scale)
        ))

        // Tail tip spike
        let tipPath = Path { path in
            path.addEllipse(in: CGRect(
                x: center.x + 62 * scale,
                y: center.y + 2 * scale,
                width: 16 * scale,
                height: 20 * scale
            ))
        }
        ctx.fill(tipPath, with: .color(spikeColor))
    }

    // MARK: - Legs (Short, stubby, cute)

    private func drawLegs(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        let legPositions = [
            CGPoint(x: center.x - 28 * scale, y: center.y + 38 * scale),
            CGPoint(x: center.x + 28 * scale, y: center.y + 38 * scale)
        ]

        for legPos in legPositions {
            // Stubby leg
            let legPath = Path { path in
                path.addRoundedRect(
                    in: CGRect(
                        x: legPos.x - 16 * scale,
                        y: legPos.y,
                        width: 32 * scale,
                        height: 38 * scale
                    ),
                    cornerSize: CGSize(width: 16 * scale, height: 16 * scale)
                )
            }
            context.fill(legPath, with: .linearGradient(
                Gradient(colors: [mainTeal, darkTeal]),
                startPoint: CGPoint(x: legPos.x, y: legPos.y),
                endPoint: CGPoint(x: legPos.x, y: legPos.y + 38 * scale)
            ))

            // Round foot
            let footPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: legPos.x - 18 * scale,
                    y: legPos.y + 30 * scale,
                    width: 36 * scale,
                    height: 18 * scale
                ))
            }
            context.fill(footPath, with: .color(darkTeal))
        }
    }

    // MARK: - Arms (Tiny T-Rex arms, cute)

    private func drawArms(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        // Left arm
        var leftCtx = context
        let leftPivot = CGPoint(x: center.x - 48 * scale, y: center.y - 15 * scale)

        if viewModel.currentState == .waving {
            leftCtx.translateBy(x: leftPivot.x, y: leftPivot.y)
            leftCtx.rotate(by: .degrees(-viewModel.armWaveAngle - 30))
            leftCtx.translateBy(x: -leftPivot.x, y: -leftPivot.y)
        }

        // Tiny arm
        let leftArmPath = Path { path in
            path.addEllipse(in: CGRect(
                x: leftPivot.x - 18 * scale,
                y: leftPivot.y - 8 * scale,
                width: 22 * scale,
                height: 16 * scale
            ))
        }
        leftCtx.fill(leftArmPath, with: .color(mainTeal))

        // Right arm
        let rightPivot = CGPoint(x: center.x + 48 * scale, y: center.y - 15 * scale)
        let rightArmPath = Path { path in
            path.addEllipse(in: CGRect(
                x: rightPivot.x - 4 * scale,
                y: rightPivot.y - 8 * scale,
                width: 22 * scale,
                height: 16 * scale
            ))
        }
        context.fill(rightArmPath, with: .color(mainTeal))
    }

    // MARK: - Face (HUGE eyes like reference - this is key!)

    private func drawFace(in context: GraphicsContext, at center: CGPoint, scale: CGFloat) {
        let expression = viewModel.expression
        let blinkFactor = 1 - viewModel.blinkProgress
        let faceY = center.y - 20 * scale

        // ========== HUGE EYES (like reference!) ==========
        let eyeSpacing: CGFloat = 38 * scale
        let baseEyeWidth: CGFloat = 32 * scale
        let baseEyeHeight: CGFloat = 38 * scale

        let eyePositions = [
            CGPoint(x: center.x - eyeSpacing / 2, y: faceY),
            CGPoint(x: center.x + eyeSpacing / 2, y: faceY)
        ]

        for (index, eyeCenter) in eyePositions.enumerated() {
            let eyeW = baseEyeWidth * expression.eyeSize
            let eyeH = baseEyeHeight * expression.eyeSize * expression.eyeOpenness * blinkFactor

            // Eye white (BIG!)
            let eyePath = Path { path in
                path.addEllipse(in: CGRect(
                    x: eyeCenter.x - eyeW / 2,
                    y: eyeCenter.y - eyeH / 2,
                    width: eyeW,
                    height: eyeH
                ))
            }
            context.fill(eyePath, with: .color(.white))

            // Subtle shadow under eye
            var shadowCtx = context
            shadowCtx.opacity = 0.1
            let shadowPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: eyeCenter.x - eyeW / 2 - 2 * scale,
                    y: eyeCenter.y - eyeH / 2 + 2 * scale,
                    width: eyeW + 4 * scale,
                    height: eyeH
                ))
            }
            shadowCtx.fill(shadowPath, with: .color(.black))

            // Big pupil
            if expression.eyeOpenness * blinkFactor > 0.3 {
                let pupilOffset = expression.pupilOffset
                let pupilX = eyeCenter.x + pupilOffset.x * 8 * scale
                let pupilY = eyeCenter.y + pupilOffset.y * 5 * scale + 3 * scale
                let pupilSize: CGFloat = 18 * scale

                // Pupil
                let pupilPath = Path { path in
                    path.addEllipse(in: CGRect(
                        x: pupilX - pupilSize / 2,
                        y: pupilY - pupilSize / 2,
                        width: pupilSize,
                        height: pupilSize
                    ))
                }
                context.fill(pupilPath, with: .color(Color(red: 0.08, green: 0.12, blue: 0.18)))

                // BIG white highlight (KEY for cute look!)
                let hl1 = Path { path in
                    path.addEllipse(in: CGRect(
                        x: pupilX + 3 * scale,
                        y: pupilY - 6 * scale,
                        width: 8 * scale,
                        height: 8 * scale
                    ))
                }
                context.fill(hl1, with: .color(.white))

                // Small secondary highlight
                let hl2 = Path { path in
                    path.addEllipse(in: CGRect(
                        x: pupilX - 5 * scale,
                        y: pupilY + 2 * scale,
                        width: 4 * scale,
                        height: 4 * scale
                    ))
                }
                context.fill(hl2, with: .color(.white.opacity(0.8)))
            }

            // Eyelid (for blink)
            if viewModel.blinkProgress > 0 {
                let lidPath = Path { path in
                    path.addEllipse(in: CGRect(
                        x: eyeCenter.x - eyeW / 2,
                        y: eyeCenter.y - eyeH / 2,
                        width: eyeW,
                        height: eyeH * viewModel.blinkProgress * 1.1
                    ))
                }
                context.fill(lidPath, with: .color(mainTeal))
            }

            // Eyebrow
            let isLeft = index == 0
            let browY = eyeCenter.y - 22 * scale - expression.browRaise * 6 * scale
            let browTilt = isLeft ? -expression.browTilt * 5 * scale : expression.browTilt * 5 * scale

            let browPath = Path { path in
                path.move(to: CGPoint(x: eyeCenter.x - 12 * scale, y: browY + browTilt))
                path.addQuadCurve(
                    to: CGPoint(x: eyeCenter.x + 12 * scale, y: browY - browTilt),
                    control: CGPoint(x: eyeCenter.x, y: browY - 5 * scale)
                )
            }
            context.stroke(browPath, with: .color(darkTeal), lineWidth: 3.5 * scale)
        }

        // ========== MOUTH ==========
        let mouthY = faceY + 30 * scale
        let mouthOpen = expression.mouthOpen

        if mouthOpen > 0.2 {
            // Open mouth
            let mouthPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: center.x - 18 * scale,
                    y: mouthY - 6 * scale,
                    width: 36 * scale,
                    height: 22 * scale * mouthOpen
                ))
            }
            context.fill(mouthPath, with: .color(Color(red: 0.12, green: 0.15, blue: 0.18)))

            // Tongue
            if mouthOpen > 0.35 {
                let tonguePath = Path { path in
                    path.addEllipse(in: CGRect(
                        x: center.x - 10 * scale,
                        y: mouthY + 4 * scale,
                        width: 20 * scale,
                        height: 12 * scale * mouthOpen
                    ))
                }
                context.fill(tonguePath, with: .color(Color(red: 0.95, green: 0.55, blue: 0.60)))
            }
        } else {
            // Smile
            let smilePath = Path { path in
                let curveY = (expression.mouthCurve - 0.5) * 24 * scale
                path.move(to: CGPoint(x: center.x - 22 * scale, y: mouthY))
                path.addQuadCurve(
                    to: CGPoint(x: center.x + 22 * scale, y: mouthY),
                    control: CGPoint(x: center.x, y: mouthY + curveY)
                )
            }
            context.stroke(smilePath, with: .color(darkTeal), lineWidth: 4 * scale)
        }

        // ========== CHEEKS (blush) ==========
        if expression.cheekBlush > 0 {
            for xMult in [-1.0, 1.0] {
                let cheekPath = Path { path in
                    path.addEllipse(in: CGRect(
                        x: center.x + CGFloat(xMult) * 40 * scale - 12 * scale,
                        y: faceY + 12 * scale,
                        width: 24 * scale,
                        height: 14 * scale
                    ))
                }
                var cheekCtx = context
                cheekCtx.opacity = Double(expression.cheekBlush) * 0.55
                cheekCtx.fill(cheekPath, with: .color(Color(red: 1.0, green: 0.45, blue: 0.55)))
            }
        }

        // ========== NOSTRILS ==========
        for xMult in [-1.0, 1.0] {
            let nostrilPath = Path { path in
                path.addEllipse(in: CGRect(
                    x: center.x + CGFloat(xMult) * 8 * scale - 3 * scale,
                    y: faceY + 18 * scale,
                    width: 6 * scale,
                    height: 5 * scale
                ))
            }
            context.fill(nostrilPath, with: .color(darkTeal))
        }
    }
}

// MARK: - Previews

#Preview("Anky States") {
    ScrollView {
        VStack(spacing: 30) {
            ForEach([AnkyState.idle, .happy, .waving, .celebrating, .concerned, .sleeping], id: \.self) { state in
                VStack {
                    AnkyAnimatedMascot(size: 180, state: state)
                    Text(state.rawValue.capitalized)
                        .font(.headline)
                }
            }
        }
        .padding()
    }
}

#Preview("Interactive Anky") {
    AnkyAnimatedMascot(size: 250, state: .idle)
        .padding(50)
}
