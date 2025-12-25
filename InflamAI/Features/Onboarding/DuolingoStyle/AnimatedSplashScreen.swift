//
//  AnimatedSplashScreen.swift
//  InflamAI
//
//  Premium Animated Splash Screen
//
//  The first 3 seconds set the emotional tone.
//  Anky wakes up - symbolizing the app coming to life WITH the user.
//
//  Psychology: The user isn't launching an app.
//  They're being greeted by a companion who was waiting for them.
//

import SwiftUI

struct AnimatedSplashScreen: View {
    @State private var phase: SplashPhase = .initial
    @State private var ankyState: AnkyState = .idle  // Start with idle (Rive-compatible)
    @State private var showSleepingImage: Bool = true  // Show static sleeping image initially
    @State private var logoOpacity: Double = 0
    @State private var taglineOpacity: Double = 0
    @State private var loadingProgress: CGFloat = 0
    @State private var backgroundGlow: CGFloat = 0
    @State private var sleepZs: [SleepZ] = []

    let onComplete: () -> Void

    enum SplashPhase {
        case initial      // Black/dark, nothing visible
        case sleeping     // Static sleeping image + Z's
        case waking       // Crossfade to Rive idle
        case greeting     // Rive wave, logo appears
        case ready        // Loading complete, ready to transition
    }

    var body: some View {
        ZStack {
            // Background gradient
            backgroundLayer

            // Ambient particles
            if phase != .initial {
                AmbientParticlesView()
                    .opacity(phase == .sleeping ? 0.3 : 0.6)
            }

            // Main content
            VStack(spacing: 0) {
                Spacer()

                // Anky character
                ZStack {
                    // Glow behind Anky
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    Color(hex: "#14B8A6").opacity(0.4 * backgroundGlow),
                                    Color.clear
                                ],
                                center: .center,
                                startRadius: 0,
                                endRadius: 150
                            )
                        )
                        .frame(width: 300, height: 300)

                    // Sleep Z's (only during sleeping phase)
                    ForEach(sleepZs) { z in
                        Text("z")
                            .font(.system(size: z.size, weight: .bold, design: .rounded))
                            .foregroundColor(.white.opacity(z.opacity))
                            .offset(x: z.offset.x, y: z.offset.y)
                    }

                    // Sleeping phase: Static image (no Optimus Prime!)
                    if showSleepingImage {
                        Image("dino-sleeping 1")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 200, height: 200)
                            .opacity(phase == .sleeping ? 1 : 0)
                    }

                    // Waking/Greeting phase: Rive Anky (no-loop version)
                    if !showSleepingImage {
                        AnkyRiveView(
                            size: 280,
                            state: ankyState,
                            showShadow: true,
                            allowRive: true  // Testing without state machine
                        )
                        .transition(.opacity)
                    }
                }

                Spacer().frame(height: Spacing.xxl)

                // Logo
                VStack(spacing: Spacing.sm) {
                    Text("InflamAI")
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .tracking(2)
                        .opacity(logoOpacity)

                    Text("Your AS Companion")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.white.opacity(0.8))
                        .opacity(taglineOpacity)
                }

                Spacer()

                // Loading indicator
                if phase == .greeting || phase == .ready {
                    LoadingPillView(progress: loadingProgress)
                        .padding(.horizontal, Spacing.xxl * 2)
                        .padding(.bottom, Spacing.xxl)
                        .transition(.opacity)
                }
            }
        }
        .ignoresSafeArea()
        .onAppear {
            startAnimationSequence()
        }
    }

    // MARK: - Background Layer

    private var backgroundLayer: some View {
        ZStack {
            // Base gradient
            LinearGradient(
                colors: [
                    Color(hex: "#0C4A6E"), // Dark blue
                    Color(hex: "#155E75"), // Teal-blue
                    Color(hex: "#134E4A")  // Dark teal
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            // Animated glow overlay
            RadialGradient(
                colors: [
                    Color(hex: "#14B8A6").opacity(0.3 * backgroundGlow),
                    Color.clear
                ],
                center: .center,
                startRadius: 50,
                endRadius: 400
            )
        }
    }

    // MARK: - Animation Sequence

    private func startAnimationSequence() {
        // Phase 0: Initial (already set)
        // Brief darkness to let eyes adjust

        // Phase 1: Sleeping (0.3s) - Show static sleeping image
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            withAnimation(.easeIn(duration: 0.5)) {
                phase = .sleeping
                backgroundGlow = 0.3
            }
            startSleepingZAnimation()
        }

        // Phase 2: Waking (1.5s) - Crossfade from image to Rive idle
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            stopSleepingZAnimation()

            withAnimation(.easeInOut(duration: 0.6)) {
                phase = .waking
                showSleepingImage = false  // Switch to Rive
                ankyState = .idle  // Rive idle animation
                backgroundGlow = 0.6
            }
        }

        // Phase 3: Greeting (2.3s) - Rive wave animation
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.3) {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7)) {
                phase = .greeting
                ankyState = .waving  // Rive wave animation
                backgroundGlow = 1.0
            }

            // Logo fade in
            withAnimation(.easeOut(duration: 0.5)) {
                logoOpacity = 1.0
            }

            // Tagline fade in (staggered)
            withAnimation(.easeOut(duration: 0.4).delay(0.2)) {
                taglineOpacity = 1.0
            }

            // Start loading progress
            startLoadingAnimation()
        }

        // Phase 4: Ready and complete (4.0s) - Stay on idle (Rive-compatible)
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.8) {
            withAnimation {
                phase = .ready
                ankyState = .idle  // Back to idle (Rive has this)
            }
        }

        // Complete and transition (4.2s)
        DispatchQueue.main.asyncAfter(deadline: .now() + 4.2) {
            HapticFeedback.success()
            onComplete()
        }
    }

    // MARK: - Sleep Z Animation

    private func startSleepingZAnimation() {
        // Spawn floating Z's periodically
        Timer.scheduledTimer(withTimeInterval: 0.6, repeats: true) { timer in
            if phase != .sleeping {
                timer.invalidate()
                return
            }

            let newZ = SleepZ(
                id: UUID(),
                offset: CGPoint(x: 80, y: -60),
                size: CGFloat.random(in: 14...22),
                opacity: 0
            )
            sleepZs.append(newZ)

            // Animate the Z floating up
            if let index = sleepZs.firstIndex(where: { $0.id == newZ.id }) {
                withAnimation(.easeOut(duration: 1.5)) {
                    sleepZs[index].offset = CGPoint(
                        x: sleepZs[index].offset.x + CGFloat.random(in: 20...40),
                        y: sleepZs[index].offset.y - CGFloat.random(in: 40...60)
                    )
                    sleepZs[index].opacity = 0.8
                }

                // Fade out
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    withAnimation(.easeIn(duration: 0.5)) {
                        if let idx = sleepZs.firstIndex(where: { $0.id == newZ.id }) {
                            sleepZs[idx].opacity = 0
                        }
                    }
                }

                // Remove
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    sleepZs.removeAll { $0.id == newZ.id }
                }
            }
        }
    }

    private func stopSleepingZAnimation() {
        withAnimation(.easeOut(duration: 0.3)) {
            for index in sleepZs.indices {
                sleepZs[index].opacity = 0
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            sleepZs.removeAll()
        }
    }

    // MARK: - Loading Animation

    private func startLoadingAnimation() {
        withAnimation(.easeInOut(duration: 1.5)) {
            loadingProgress = 1.0
        }
    }
}

// MARK: - Supporting Types

struct SleepZ: Identifiable {
    let id: UUID
    var offset: CGPoint
    var size: CGFloat
    var opacity: Double
}

// MARK: - Loading Pill View

struct LoadingPillView: View {
    let progress: CGFloat

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background track
                Capsule()
                    .fill(Color.white.opacity(0.2))
                    .frame(height: 6)

                // Progress fill
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(hex: "#14B8A6"),
                                Color(hex: "#06B6D4")
                            ],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: geometry.size.width * progress, height: 6)

                // Glow at progress edge
                if progress > 0 && progress < 1 {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 8, height: 8)
                        .blur(radius: 4)
                        .offset(x: geometry.size.width * progress - 4)
                }
            }
        }
        .frame(height: 6)
    }
}

// MARK: - Ambient Particles View

struct AmbientParticlesView: View {
    @State private var particles: [AmbientParticle] = []

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                let time = timeline.date.timeIntervalSinceReferenceDate

                for particle in particles {
                    let age = (time - particle.birthTime).truncatingRemainder(dividingBy: particle.duration)
                    let progress = age / particle.duration

                    // Slow float upward
                    let y = particle.startY - (particle.floatDistance * progress)
                    let x = particle.startX + sin(time * particle.wobbleSpeed + particle.wobbleOffset) * particle.wobbleAmount

                    // Fade in and out
                    let fadeIn = min(progress * 4, 1.0)
                    let fadeOut = max(1 - (progress - 0.7) * 3.33, 0)
                    let opacity = min(fadeIn, fadeOut) * particle.maxOpacity

                    var ctx = context
                    ctx.opacity = opacity

                    let rect = CGRect(
                        x: x - particle.size / 2,
                        y: y - particle.size / 2,
                        width: particle.size,
                        height: particle.size
                    )

                    ctx.fill(Circle().path(in: rect), with: .color(.white))
                }
            }
        }
        .onAppear {
            spawnParticles()
        }
    }

    private func spawnParticles() {
        for _ in 0..<20 {
            particles.append(AmbientParticle(
                birthTime: Date().timeIntervalSinceReferenceDate - Double.random(in: 0...8),
                duration: Double.random(in: 6...10),
                startX: CGFloat.random(in: 0...400),
                startY: CGFloat.random(in: 600...900),
                floatDistance: CGFloat.random(in: 300...500),
                wobbleSpeed: Double.random(in: 0.5...1.5),
                wobbleOffset: Double.random(in: 0...(.pi * 2)),
                wobbleAmount: CGFloat.random(in: 10...30),
                size: CGFloat.random(in: 2...5),
                maxOpacity: Double.random(in: 0.2...0.5)
            ))
        }
    }
}

struct AmbientParticle {
    let birthTime: TimeInterval
    let duration: TimeInterval
    let startX: CGFloat
    let startY: CGFloat
    let floatDistance: CGFloat
    let wobbleSpeed: Double
    let wobbleOffset: Double
    let wobbleAmount: CGFloat
    let size: CGFloat
    let maxOpacity: Double
}

// MARK: - Preview

#Preview {
    AnimatedSplashScreen {
        print("Splash complete!")
    }
}
