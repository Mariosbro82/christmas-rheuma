//
//  OnboardingAnimationExtras.swift
//  InflamAI
//
//  Premium Animation Extras for Onboarding
//
//  These are the micro-interactions and polish touches
//  that elevate the experience from good to exceptional.
//
//  Psychology: Every small detail signals "we care about you"
//

import SwiftUI

// MARK: - Typewriter Text Animation

/// Animates text appearing letter by letter, like Anky is speaking
struct TypewriterText: View {
    let fullText: String
    let typingSpeed: Double
    let onComplete: (() -> Void)?

    @State private var displayedText = ""
    @State private var currentIndex = 0

    init(_ text: String, speed: Double = 0.04, onComplete: (() -> Void)? = nil) {
        self.fullText = text
        self.typingSpeed = speed
        self.onComplete = onComplete
    }

    var body: some View {
        Text(displayedText)
            .onAppear {
                startTyping()
            }
    }

    private func startTyping() {
        displayedText = ""
        currentIndex = 0

        Timer.scheduledTimer(withTimeInterval: typingSpeed, repeats: true) { timer in
            if currentIndex < fullText.count {
                let index = fullText.index(fullText.startIndex, offsetBy: currentIndex)
                displayedText += String(fullText[index])
                currentIndex += 1

                // Subtle haptic on certain characters
                if ["!", "?", "."].contains(String(fullText[index])) {
                    HapticFeedback.light()
                }
            } else {
                timer.invalidate()
                onComplete?()
            }
        }
    }
}

// MARK: - Animated Counter

/// Animates a number counting up with spring physics
struct AnimatedCounter: View {
    let value: Int
    let font: Font
    let color: Color

    @State private var displayValue: Int = 0

    var body: some View {
        Text("\(displayValue)")
            .font(font)
            .foregroundColor(color)
            .contentTransition(.numericText())
            .onChange(of: value) { newValue in
                withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                    displayValue = newValue
                }
            }
            .onAppear {
                withAnimation(.spring(response: 0.6, dampingFraction: 0.7)) {
                    displayValue = value
                }
            }
    }
}

// MARK: - Pulsing Glow Effect

struct PulsingGlowModifier: ViewModifier {
    let color: Color
    let radius: CGFloat
    @State private var isPulsing = false

    func body(content: Content) -> some View {
        content
            .shadow(
                color: color.opacity(isPulsing ? 0.6 : 0.2),
                radius: isPulsing ? radius * 1.5 : radius,
                y: 0
            )
            .onAppear {
                withAnimation(
                    .easeInOut(duration: 1.5)
                    .repeatForever(autoreverses: true)
                ) {
                    isPulsing = true
                }
            }
    }
}

extension View {
    func pulsingGlow(color: Color, radius: CGFloat = 15) -> some View {
        modifier(PulsingGlowModifier(color: color, radius: radius))
    }
}

// MARK: - Floating Animation

struct FloatingModifier: ViewModifier {
    let amplitude: CGFloat
    let duration: Double
    @State private var isFloating = false

    func body(content: Content) -> some View {
        content
            .offset(y: isFloating ? -amplitude : amplitude)
            .onAppear {
                withAnimation(
                    .easeInOut(duration: duration)
                    .repeatForever(autoreverses: true)
                ) {
                    isFloating = true
                }
            }
    }
}

extension View {
    func floating(amplitude: CGFloat = 8, duration: Double = 2) -> some View {
        modifier(FloatingModifier(amplitude: amplitude, duration: duration))
    }
}

// MARK: - Shimmer Highlight

struct ShimmerHighlightModifier: ViewModifier {
    @State private var shimmerOffset: CGFloat = -1

    func body(content: Content) -> some View {
        content
            .overlay(
                GeometryReader { geometry in
                    LinearGradient(
                        colors: [
                            .clear,
                            .white.opacity(0.4),
                            .clear
                        ],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                    .frame(width: geometry.size.width * 0.5)
                    .offset(x: shimmerOffset * geometry.size.width)
                    .onAppear {
                        withAnimation(
                            .linear(duration: 2.5)
                            .repeatForever(autoreverses: false)
                        ) {
                            shimmerOffset = 2
                        }
                    }
                }
            )
            .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
    }
}

extension View {
    func shimmerHighlight() -> some View {
        modifier(ShimmerHighlightModifier())
    }
}

// MARK: - Success Checkmark Animation

struct AnimatedCheckmark: View {
    let isVisible: Bool
    let size: CGFloat
    let color: Color

    @State private var pathProgress: CGFloat = 0

    var body: some View {
        ZStack {
            // Background circle
            Circle()
                .fill(color.opacity(0.15))
                .frame(width: size, height: size)

            // Checkmark path
            Path { path in
                let rect = CGRect(x: 0, y: 0, width: size * 0.5, height: size * 0.5)
                let startPoint = CGPoint(x: rect.minX + rect.width * 0.15, y: rect.midY)
                let midPoint = CGPoint(x: rect.minX + rect.width * 0.4, y: rect.maxY - rect.height * 0.2)
                let endPoint = CGPoint(x: rect.maxX - rect.width * 0.1, y: rect.minY + rect.height * 0.2)

                path.move(to: startPoint)
                path.addLine(to: midPoint)
                path.addLine(to: endPoint)
            }
            .trim(from: 0, to: pathProgress)
            .stroke(color, style: StrokeStyle(lineWidth: size * 0.08, lineCap: .round, lineJoin: .round))
            .frame(width: size * 0.5, height: size * 0.5)
        }
        .onChange(of: isVisible) { visible in
            if visible {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.6).delay(0.1)) {
                    pathProgress = 1
                }
            } else {
                pathProgress = 0
            }
        }
    }
}

// MARK: - Ripple Effect on Tap

struct RippleEffect: ViewModifier {
    @State private var ripples: [Ripple] = []

    struct Ripple: Identifiable {
        let id = UUID()
        var scale: CGFloat = 0
        var opacity: Double = 0.5
        let position: CGPoint
    }

    func body(content: Content) -> some View {
        content
            .overlay(
                GeometryReader { geometry in
                    ZStack {
                        ForEach(ripples) { ripple in
                            Circle()
                                .fill(Color.white.opacity(ripple.opacity))
                                .scaleEffect(ripple.scale)
                                .position(ripple.position)
                        }
                    }
                }
                .allowsHitTesting(false)
            )
            .simultaneousGesture(
                DragGesture(minimumDistance: 0)
                    .onEnded { gesture in
                        addRipple(at: gesture.location)
                    }
            )
    }

    private func addRipple(at point: CGPoint) {
        var ripple = Ripple(position: point)
        ripples.append(ripple)

        let rippleID = ripple.id

        withAnimation(.easeOut(duration: 0.6)) {
            if let index = ripples.firstIndex(where: { $0.id == rippleID }) {
                ripples[index].scale = 2
                ripples[index].opacity = 0
            }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            ripples.removeAll { $0.id == rippleID }
        }
    }
}

extension View {
    func rippleEffect() -> some View {
        modifier(RippleEffect())
    }
}

// MARK: - Elastic Button Style

struct ElasticButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1.0)
            .animation(.spring(response: 0.25, dampingFraction: 0.5), value: configuration.isPressed)
    }
}

extension View {
    func elasticButton() -> some View {
        buttonStyle(ElasticButtonStyle())
    }
}

// MARK: - Parallax Scroll Effect

struct ParallaxScrollModifier: ViewModifier {
    let intensity: CGFloat
    @State private var offset: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .offset(y: offset * intensity)
            .background(
                GeometryReader { geometry in
                    Color.clear
                        .preference(
                            key: ScrollOffsetPreferenceKey.self,
                            value: geometry.frame(in: .global).minY
                        )
                }
            )
            .onPreferenceChange(ScrollOffsetPreferenceKey.self) { value in
                offset = value
            }
    }
}

struct ScrollOffsetPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

extension View {
    func parallaxScroll(intensity: CGFloat = 0.3) -> some View {
        modifier(ParallaxScrollModifier(intensity: intensity))
    }
}

// MARK: - Breathing Animation (for mascot)

struct BreathingModifier: ViewModifier {
    let minScale: CGFloat
    let maxScale: CGFloat
    let duration: Double
    @State private var isBreathingIn = false

    func body(content: Content) -> some View {
        content
            .scaleEffect(isBreathingIn ? maxScale : minScale)
            .onAppear {
                withAnimation(
                    .easeInOut(duration: duration)
                    .repeatForever(autoreverses: true)
                ) {
                    isBreathingIn = true
                }
            }
    }
}

extension View {
    func breathing(min: CGFloat = 0.98, max: CGFloat = 1.02, duration: Double = 2) -> some View {
        modifier(BreathingModifier(minScale: min, maxScale: max, duration: duration))
    }
}

// MARK: - Staggered List Animation

struct StaggeredListModifier: ViewModifier {
    let index: Int
    let totalItems: Int
    @State private var isVisible = false

    func body(content: Content) -> some View {
        content
            .opacity(isVisible ? 1 : 0)
            .offset(y: isVisible ? 0 : 30)
            .scaleEffect(isVisible ? 1 : 0.9)
            .onAppear {
                let delay = Double(index) * 0.08
                withAnimation(
                    .spring(response: 0.5, dampingFraction: 0.7)
                    .delay(delay)
                ) {
                    isVisible = true
                }
            }
    }
}

extension View {
    func staggeredListItem(index: Int, total: Int) -> some View {
        modifier(StaggeredListModifier(index: index, totalItems: total))
    }
}

// MARK: - Celebration Burst

struct CelebrationBurst: View {
    @State private var particles: [CelebrationParticle] = []
    @State private var isAnimating = false

    var body: some View {
        ZStack {
            ForEach(particles) { particle in
                Circle()
                    .fill(particle.color)
                    .frame(width: particle.size, height: particle.size)
                    .scaleEffect(isAnimating ? 0 : 1)
                    .offset(
                        x: isAnimating ? particle.endOffset.x : 0,
                        y: isAnimating ? particle.endOffset.y : 0
                    )
                    .opacity(isAnimating ? 0 : 1)
            }
        }
        .onAppear {
            generateParticles()
            withAnimation(.easeOut(duration: 1.0)) {
                isAnimating = true
            }
        }
    }

    private func generateParticles() {
        let colors: [Color] = [
            Colors.Accent.teal,
            Colors.Primary.p400,
            Color(hex: "#F59E0B"),
            Color(hex: "#EC4899"),
            .white
        ]

        for _ in 0..<16 {
            let angle = Double.random(in: 0...(2 * .pi))
            let distance = CGFloat.random(in: 60...150)

            particles.append(CelebrationParticle(
                color: colors.randomElement()!,
                size: CGFloat.random(in: 6...12),
                endOffset: CGPoint(
                    x: CGFloat(Darwin.cos(angle)) * distance,
                    y: CGFloat(Darwin.sin(angle)) * distance
                )
            ))
        }
    }
}

struct CelebrationParticle: Identifiable {
    let id = UUID()
    let color: Color
    let size: CGFloat
    let endOffset: CGPoint
}

// MARK: - Ring Progress Indicator

struct RingProgressView: View {
    let progress: CGFloat
    let lineWidth: CGFloat
    let color: Color
    let backgroundColor: Color

    @State private var animatedProgress: CGFloat = 0

    var body: some View {
        ZStack {
            // Background ring
            Circle()
                .stroke(backgroundColor, lineWidth: lineWidth)

            // Progress ring
            Circle()
                .trim(from: 0, to: animatedProgress)
                .stroke(
                    color,
                    style: StrokeStyle(
                        lineWidth: lineWidth,
                        lineCap: .round
                    )
                )
                .rotationEffect(.degrees(-90))
        }
        .onChange(of: progress) { newValue in
            withAnimation(.spring(response: 0.6, dampingFraction: 0.7)) {
                animatedProgress = newValue
            }
        }
        .onAppear {
            withAnimation(.spring(response: 0.8, dampingFraction: 0.7).delay(0.2)) {
                animatedProgress = progress
            }
        }
    }
}

// MARK: - Previews

#Preview("Typewriter Text") {
    VStack {
        TypewriterText("Hello! I'm Anky, your companion for managing AS together.", speed: 0.05)
            .font(.title2)
            .foregroundColor(Colors.Gray.g900)
    }
    .padding()
}

#Preview("Animated Counter") {
    AnimatedCounter(value: 42, font: .system(size: 64, weight: .bold, design: .rounded), color: Colors.Primary.p500)
}

#Preview("Celebration Burst") {
    CelebrationBurst()
        .frame(width: 200, height: 200)
}

#Preview("Ring Progress") {
    RingProgressView(
        progress: 0.75,
        lineWidth: 8,
        color: Colors.Accent.teal,
        backgroundColor: Colors.Gray.g200
    )
    .frame(width: 100, height: 100)
}
