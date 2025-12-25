//
//  WellnessJourneySystem.swift
//  InflamAI
//
//  Emotional Wellness Journey System
//
//  This is NOT gamification for its own sake.
//  This is psychology-driven emotional design:
//
//  - Tinder Psychology: Investment through action creates attachment
//  - Apple Psychology: Craftsmanship signals "we care about you"
//  - InflamAI Psychology: Anky is your COMPANION, not a game master
//
//  Key Principles:
//  1. Celebrate consistency, not performance
//  2. Acknowledge pain days with empathy, not guilt
//  3. Companion language: "We did this together"
//  4. Rituals over rewards
//

import SwiftUI
import Combine

// MARK: - Wellness Journey State

/// Tracks the user's emotional journey, not just metrics
@MainActor
class WellnessJourneyManager: ObservableObject {
    static let shared = WellnessJourneyManager()

    // MARK: - Published State

    /// Current wellness streak (consecutive days of check-ins)
    @Published var currentStreak: Int = 0

    /// Longest streak ever achieved
    @Published var longestStreak: Int = 0

    /// Days since starting the journey
    @Published var journeyDays: Int = 0

    /// Total check-ins completed
    @Published var totalCheckIns: Int = 0

    /// Milestones achieved
    @Published var milestones: [WellnessMilestone] = []

    /// Last check-in date
    @Published var lastCheckInDate: Date?

    /// User's wellness sentiment (based on recent entries)
    @Published var recentSentiment: WellnessSentiment = .neutral

    // MARK: - Persistence Keys

    private let streakKey = "wellness_current_streak"
    private let longestStreakKey = "wellness_longest_streak"
    private let journeyStartKey = "wellness_journey_start"
    private let totalCheckInsKey = "wellness_total_checkins"
    private let lastCheckInKey = "wellness_last_checkin"
    private let milestonesKey = "wellness_milestones"

    // MARK: - Init

    private init() {
        loadState()
    }

    // MARK: - Core Actions

    /// Called when user completes a daily check-in
    func recordCheckIn(painLevel: Int, mood: Int) {
        let today = Calendar.current.startOfDay(for: Date())

        // Check if this is a new day
        if let lastDate = lastCheckInDate {
            let lastDay = Calendar.current.startOfDay(for: lastDate)

            if today == lastDay {
                // Already checked in today - update but don't increment streak
                updateSentiment(painLevel: painLevel, mood: mood)
                return
            }

            let daysBetween = Calendar.current.dateComponents([.day], from: lastDay, to: today).day ?? 0

            if daysBetween == 1 {
                // Consecutive day - increment streak
                currentStreak += 1
            } else {
                // Streak broken - but with empathy
                currentStreak = 1
            }
        } else {
            // First check-in ever
            currentStreak = 1
            UserDefaults.standard.set(Date(), forKey: journeyStartKey)
        }

        // Update records
        lastCheckInDate = Date()
        totalCheckIns += 1

        if currentStreak > longestStreak {
            longestStreak = currentStreak
        }

        // Calculate journey days
        if let startDate = UserDefaults.standard.object(forKey: journeyStartKey) as? Date {
            journeyDays = Calendar.current.dateComponents([.day], from: startDate, to: Date()).day ?? 0 + 1
        }

        // Update sentiment
        updateSentiment(painLevel: painLevel, mood: mood)

        // Check for new milestones
        checkMilestones()

        // Persist
        saveState()
    }

    /// Called when user completes an assessment (BASDAI, etc.)
    func recordAssessmentCompletion(type: AssessmentType, score: Double) {
        // Assessments are significant moments - always acknowledge
        let milestone = WellnessMilestone(
            type: .assessmentCompleted,
            title: "\(type.displayName) Recorded",
            subtitle: "Score: \(String(format: "%.1f", score))",
            date: Date(),
            icon: type.icon
        )

        milestones.append(milestone)
        saveState()
    }

    // MARK: - Sentiment Analysis

    private func updateSentiment(painLevel: Int, mood: Int) {
        // Simple sentiment based on recent data
        // In a real app, this would analyze trends
        let combinedScore = (10 - painLevel) + mood // Higher = better
        let normalizedScore = Double(combinedScore) / 20.0

        if normalizedScore > 0.7 {
            recentSentiment = .thriving
        } else if normalizedScore > 0.5 {
            recentSentiment = .steady
        } else if normalizedScore > 0.3 {
            recentSentiment = .struggling
        } else {
            recentSentiment = .needsSupport
        }
    }

    // MARK: - Milestone Checking

    private func checkMilestones() {
        var newMilestones: [WellnessMilestone] = []

        // Streak milestones
        let streakMilestones = [3, 7, 14, 30, 60, 90, 180, 365]
        for days in streakMilestones {
            if currentStreak == days && !hasMilestone(.streak(days)) {
                newMilestones.append(WellnessMilestone(
                    type: .streak(days),
                    title: "\(days) Day Streak",
                    subtitle: streakSubtitle(for: days),
                    date: Date(),
                    icon: streakIcon(for: days)
                ))
            }
        }

        // Journey milestones
        let journeyMilestones = [7, 30, 90, 180, 365]
        for days in journeyMilestones {
            if journeyDays == days && !hasMilestone(.journey(days)) {
                newMilestones.append(WellnessMilestone(
                    type: .journey(days),
                    title: "\(days) Days Together",
                    subtitle: journeySubtitle(for: days),
                    date: Date(),
                    icon: "heart.fill"
                ))
            }
        }

        // Check-in count milestones
        let checkInMilestones = [10, 50, 100, 500, 1000]
        for count in checkInMilestones {
            if totalCheckIns == count && !hasMilestone(.checkIns(count)) {
                newMilestones.append(WellnessMilestone(
                    type: .checkIns(count),
                    title: "\(count) Check-Ins",
                    subtitle: "Your commitment inspires us",
                    date: Date(),
                    icon: "checkmark.seal.fill"
                ))
            }
        }

        milestones.append(contentsOf: newMilestones)
    }

    private func hasMilestone(_ type: MilestoneType) -> Bool {
        milestones.contains { $0.type == type }
    }

    private func streakSubtitle(for days: Int) -> String {
        switch days {
        case 3: return "You're building a habit"
        case 7: return "One week of consistency"
        case 14: return "Two weeks strong"
        case 30: return "A month of dedication"
        case 60: return "Two months! Incredible"
        case 90: return "Quarter year champion"
        case 180: return "Half a year together"
        case 365: return "One year! You're amazing"
        default: return "Keep going!"
        }
    }

    private func streakIcon(for days: Int) -> String {
        switch days {
        case 3...6: return "flame"
        case 7...29: return "flame.fill"
        case 30...89: return "star.fill"
        case 90...179: return "trophy"
        case 180...364: return "trophy.fill"
        default: return "crown.fill"
        }
    }

    private func journeySubtitle(for days: Int) -> String {
        switch days {
        case 7: return "Our first week together"
        case 30: return "A month of partnership"
        case 90: return "Three months of progress"
        case 180: return "Half a year as companions"
        case 365: return "One year! What a journey"
        default: return "Thank you for trusting us"
        }
    }

    // MARK: - Persistence

    private func loadState() {
        currentStreak = UserDefaults.standard.integer(forKey: streakKey)
        longestStreak = UserDefaults.standard.integer(forKey: longestStreakKey)
        totalCheckIns = UserDefaults.standard.integer(forKey: totalCheckInsKey)
        lastCheckInDate = UserDefaults.standard.object(forKey: lastCheckInKey) as? Date

        if let startDate = UserDefaults.standard.object(forKey: journeyStartKey) as? Date {
            journeyDays = Calendar.current.dateComponents([.day], from: startDate, to: Date()).day ?? 0 + 1
        }

        // Load milestones from UserDefaults
        if let data = UserDefaults.standard.data(forKey: milestonesKey),
           let decoded = try? JSONDecoder().decode([WellnessMilestone].self, from: data) {
            milestones = decoded
        }
    }

    private func saveState() {
        UserDefaults.standard.set(currentStreak, forKey: streakKey)
        UserDefaults.standard.set(longestStreak, forKey: longestStreakKey)
        UserDefaults.standard.set(totalCheckIns, forKey: totalCheckInsKey)
        UserDefaults.standard.set(lastCheckInDate, forKey: lastCheckInKey)

        if let encoded = try? JSONEncoder().encode(milestones) {
            UserDefaults.standard.set(encoded, forKey: milestonesKey)
        }
    }
}

// MARK: - Supporting Types

enum WellnessSentiment {
    case thriving      // User is doing great
    case steady        // Maintaining well
    case neutral       // No clear trend
    case struggling    // Having a hard time
    case needsSupport  // Really difficult period

    var ankyState: AnkyState {
        switch self {
        case .thriving: return .happy
        case .steady: return .encouraging
        case .neutral: return .idle
        case .struggling: return .concerned
        case .needsSupport: return .sympathetic
        }
    }

    var companionMessage: String {
        switch self {
        case .thriving:
            return "You're doing amazing! I'm so proud of us."
        case .steady:
            return "Steady progress. We've got this together."
        case .neutral:
            return "I'm here with you, whatever today brings."
        case .struggling:
            return "Tough days happen. I'm right here with you."
        case .needsSupport:
            return "I know it's hard. Let's take this moment by moment."
        }
    }
}

enum AssessmentType: String, Codable {
    case basdai = "BASDAI"
    case asdas = "ASDAS"
    case quickCheck = "Quick Check"
    case painMap = "Pain Map"

    var displayName: String {
        rawValue
    }

    var icon: String {
        switch self {
        case .basdai: return "chart.bar.doc.horizontal"
        case .asdas: return "waveform.path.ecg"
        case .quickCheck: return "checkmark.circle"
        case .painMap: return "figure.stand"
        }
    }
}

enum MilestoneType: Codable, Equatable {
    case streak(Int)
    case journey(Int)
    case checkIns(Int)
    case assessmentCompleted
    case firstFlareLogged
    case medicationAdherence(Int) // percentage

    var category: String {
        switch self {
        case .streak: return "Consistency"
        case .journey: return "Journey"
        case .checkIns: return "Dedication"
        case .assessmentCompleted: return "Health"
        case .firstFlareLogged: return "Awareness"
        case .medicationAdherence: return "Medication"
        }
    }
}

struct WellnessMilestone: Codable, Identifiable {
    let id: UUID
    let type: MilestoneType
    let title: String
    let subtitle: String
    let date: Date
    let icon: String

    init(type: MilestoneType, title: String, subtitle: String, date: Date, icon: String) {
        self.id = UUID()
        self.type = type
        self.title = title
        self.subtitle = subtitle
        self.date = date
        self.icon = icon
    }
}

// MARK: - Celebration View Components

/// Full-screen celebration for major milestones
struct MilestoneCelebrationView: View {
    let milestone: WellnessMilestone
    let onDismiss: () -> Void

    @State private var showContent = false
    @State private var showConfetti = false
    @State private var ankyState: AnkyState = .idle

    var body: some View {
        ZStack {
            // Background
            LinearGradient(
                colors: [
                    Colors.Accent.teal.opacity(0.9),
                    Colors.Primary.p500.opacity(0.8)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            // Confetti
            if showConfetti {
                ConfettiView()
                    .allowsHitTesting(false)
            }

            VStack(spacing: Spacing.xl) {
                Spacer()

                // Anky celebrating
                AnkyAnimatedMascot(size: 200, state: ankyState)
                    .scaleEffect(showContent ? 1 : 0.5)
                    .opacity(showContent ? 1 : 0)

                // Milestone badge
                VStack(spacing: Spacing.md) {
                    Image(systemName: milestone.icon)
                        .font(.system(size: 48))
                        .foregroundStyle(.white)
                        .scaleEffect(showContent ? 1 : 0)

                    Text(milestone.title)
                        .font(.system(size: 32, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .opacity(showContent ? 1 : 0)

                    Text(milestone.subtitle)
                        .font(.system(size: 18, weight: .medium))
                        .foregroundColor(.white.opacity(0.9))
                        .multilineTextAlignment(.center)
                        .opacity(showContent ? 1 : 0)
                }
                .padding(.horizontal, Spacing.xl)

                Spacer()

                // Continue button
                Button(action: {
                    HapticFeedback.medium()
                    onDismiss()
                }) {
                    Text("Continue")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(Colors.Primary.p600)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, Spacing.md)
                        .background(Color.white)
                        .cornerRadius(Radii.xl)
                }
                .padding(.horizontal, Spacing.xl)
                .padding(.bottom, Spacing.xxl)
                .opacity(showContent ? 1 : 0)
            }
        }
        .onAppear {
            // Staggered animation sequence
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7)) {
                showContent = true
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                ankyState = .celebrating
                showConfetti = true
                HapticFeedback.success()
            }
        }
    }
}

/// Assessment completion celebration - the key differentiator
struct AssessmentCompletionCelebration: View {
    let assessmentType: AssessmentType
    let score: Double
    let interpretation: String
    let onContinue: () -> Void

    @State private var phase: CelebrationPhase = .initial
    @State private var ankyState: AnkyState = .idle
    @State private var showScore = false
    @State private var showInterpretation = false

    enum CelebrationPhase {
        case initial
        case ankyReacts
        case scoreReveals
        case complete
    }

    var body: some View {
        ZStack {
            // Soft gradient background
            LinearGradient(
                colors: [
                    Colors.Gray.g50,
                    Color.white
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: Spacing.lg) {
                Spacer()

                // Anky reacting to the score
                ZStack {
                    // Glow effect behind Anky
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    scoreColor.opacity(0.3),
                                    scoreColor.opacity(0)
                                ],
                                center: .center,
                                startRadius: 0,
                                endRadius: 120
                            )
                        )
                        .frame(width: 240, height: 240)
                        .scaleEffect(phase == .ankyReacts ? 1.2 : 0.8)
                        .animation(.easeOut(duration: 0.8), value: phase)

                    AnkyAnimatedMascot(size: 180, state: ankyState)
                }

                // Speech bubble from Anky
                if phase == .ankyReacts || phase == .scoreReveals || phase == .complete {
                    SpeechBubble(text: companionMessage)
                        .transition(.scale.combined(with: .opacity))
                }

                Spacer()

                // Score card
                VStack(spacing: Spacing.md) {
                    // Assessment type
                    Text(assessmentType.displayName)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(Colors.Gray.g500)
                        .textCase(.uppercase)
                        .tracking(1.5)

                    // Big score reveal
                    if showScore {
                        Text(String(format: "%.1f", score))
                            .font(.system(size: 64, weight: .bold, design: .rounded))
                            .foregroundColor(scoreColor)
                            .transition(.scale.combined(with: .opacity))
                    }

                    // Interpretation
                    if showInterpretation {
                        Text(interpretation)
                            .font(.system(size: 17, weight: .medium))
                            .foregroundColor(Colors.Gray.g700)
                            .multilineTextAlignment(.center)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
                .padding(Spacing.xl)
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: Radii.xxl)
                        .fill(Color.white)
                        .shadow(color: scoreColor.opacity(0.2), radius: 20, y: 10)
                )
                .padding(.horizontal, Spacing.lg)

                Spacer()

                // Continue button
                if phase == .complete {
                    Button(action: {
                        HapticFeedback.medium()
                        onContinue()
                    }) {
                        HStack {
                            Text("Continue")
                            Image(systemName: "arrow.right")
                        }
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, Spacing.md)
                        .background(Colors.Primary.p500)
                        .cornerRadius(Radii.xl)
                    }
                    .padding(.horizontal, Spacing.xl)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                Spacer().frame(height: Spacing.xl)
            }
        }
        .onAppear {
            runCelebrationSequence()
        }
    }

    // MARK: - Computed Properties

    private var scoreColor: Color {
        switch score {
        case 0..<2: return Colors.Semantic.success
        case 2..<4: return Color(hex: "#84CC16") // Lime
        case 4..<6: return Colors.Semantic.warning
        default: return Colors.Semantic.error
        }
    }

    private var companionMessage: String {
        switch score {
        case 0..<2:
            return "Amazing! Your symptoms are well controlled. We're doing great together!"
        case 2..<4:
            return "Good job tracking this. Let's keep working on it together."
        case 4..<6:
            return "Thank you for being honest. This helps us understand what you're going through."
        default:
            return "I know things are tough right now. I'm here with you, and this data helps us help you."
        }
    }

    // MARK: - Animation Sequence

    private func runCelebrationSequence() {
        // Phase 1: Anky reacts
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7)) {
                phase = .ankyReacts
                ankyState = ankyStateForScore
            }
            HapticFeedback.light()
        }

        // Phase 2: Score reveals
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                phase = .scoreReveals
                showScore = true
            }
            HapticFeedback.medium()
        }

        // Phase 3: Interpretation
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            withAnimation(.easeOut(duration: 0.4)) {
                showInterpretation = true
            }
        }

        // Phase 4: Complete
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.2) {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                phase = .complete
            }
        }
    }

    private var ankyStateForScore: AnkyState {
        switch score {
        case 0..<2: return .celebrating
        case 2..<4: return .happy
        case 4..<6: return .encouraging
        default: return .sympathetic
        }
    }
}

/// Speech bubble component
struct SpeechBubble: View {
    let text: String

    var body: some View {
        VStack(spacing: 0) {
            Text(text)
                .font(.system(size: 16, weight: .medium))
                .foregroundColor(Colors.Gray.g800)
                .multilineTextAlignment(.center)
                .padding(Spacing.md)
                .background(
                    RoundedRectangle(cornerRadius: Radii.lg)
                        .fill(Color.white)
                        .shadow(color: Color.black.opacity(0.08), radius: 8, y: 4)
                )

            // Bubble tail
            Triangle()
                .fill(Color.white)
                .frame(width: 16, height: 10)
                .rotationEffect(.degrees(180))
                .offset(y: -1)
                .shadow(color: Color.black.opacity(0.05), radius: 2, y: 2)
        }
        .padding(.horizontal, Spacing.xl)
    }
}

struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}

// MARK: - Confetti System

struct ConfettiView: View {
    @State private var particles: [ConfettiParticle] = []

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                let now = timeline.date.timeIntervalSinceReferenceDate

                for particle in particles {
                    let age = now - particle.birthTime
                    guard age < particle.lifetime else { continue }

                    let progress = age / particle.lifetime
                    let opacity = 1 - (progress * progress) // Ease out fade

                    var transform = CGAffineTransform.identity
                    let x = particle.startX + particle.velocityX * age
                    let y = particle.startY + particle.velocityY * age + 0.5 * 400 * age * age // Gravity
                    transform = transform.translatedBy(x: x, y: y)
                    transform = transform.rotated(by: particle.rotation + particle.rotationSpeed * age)

                    var particleContext = context
                    particleContext.opacity = opacity
                    particleContext.transform = transform

                    let rect = CGRect(x: -particle.size/2, y: -particle.size/2, width: particle.size, height: particle.size)

                    switch particle.shape {
                    case .circle:
                        particleContext.fill(Circle().path(in: rect), with: .color(particle.color))
                    case .square:
                        particleContext.fill(Rectangle().path(in: rect), with: .color(particle.color))
                    case .star:
                        particleContext.fill(Star(corners: 5, smoothness: 0.45).path(in: rect), with: .color(particle.color))
                    }
                }
            }
        }
        .onAppear {
            spawnParticles()
        }
    }

    private func spawnParticles() {
        let colors: [Color] = [
            Colors.Accent.teal,
            Colors.Primary.p400,
            Color(hex: "#F59E0B"), // Gold
            Color(hex: "#EC4899"), // Pink
            .white
        ]

        for _ in 0..<40 {
            let particle = ConfettiParticle(
                birthTime: Date().timeIntervalSinceReferenceDate + Double.random(in: 0...0.3),
                lifetime: Double.random(in: 2.0...3.5),
                startX: CGFloat.random(in: 50...350),
                startY: -20,
                velocityX: CGFloat.random(in: -80...80),
                velocityY: CGFloat.random(in: 50...150),
                rotation: CGFloat.random(in: 0...(.pi * 2)),
                rotationSpeed: CGFloat.random(in: -3...3),
                size: CGFloat.random(in: 6...14),
                color: colors.randomElement()!,
                shape: ConfettiShape.allCases.randomElement()!
            )
            particles.append(particle)
        }
    }
}

struct ConfettiParticle {
    let birthTime: TimeInterval
    let lifetime: TimeInterval
    let startX: CGFloat
    let startY: CGFloat
    let velocityX: CGFloat
    let velocityY: CGFloat
    let rotation: CGFloat
    let rotationSpeed: CGFloat
    let size: CGFloat
    let color: Color
    let shape: ConfettiShape
}

enum ConfettiShape: CaseIterable {
    case circle, square, star
}

struct Star: Shape {
    let corners: Int
    let smoothness: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) / 2
        let innerRadius = radius * smoothness

        for i in 0..<(corners * 2) {
            let angle = (CGFloat(i) * .pi / CGFloat(corners)) - .pi / 2
            let r = i.isMultiple(of: 2) ? radius : innerRadius
            let point = CGPoint(
                x: center.x + cos(angle) * r,
                y: center.y + sin(angle) * r
            )

            if i == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }
        path.closeSubpath()
        return path
    }
}

// MARK: - Streak Display Component

struct StreakDisplayView: View {
    @ObservedObject var journeyManager: WellnessJourneyManager

    var body: some View {
        HStack(spacing: Spacing.md) {
            // Streak flame
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(hex: "#F97316"),
                                Color(hex: "#DC2626")
                            ],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: 44, height: 44)

                Image(systemName: journeyManager.currentStreak > 0 ? "flame.fill" : "flame")
                    .font(.system(size: 22))
                    .foregroundColor(.white)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("\(journeyManager.currentStreak)")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                Text(journeyManager.currentStreak == 1 ? "day streak" : "day streak")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(Colors.Gray.g500)
            }

            Spacer()

            // Journey days
            VStack(alignment: .trailing, spacing: 2) {
                Text("\(journeyManager.journeyDays)")
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                    .foregroundColor(Colors.Gray.g700)

                Text("days together")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Colors.Gray.g400)
            }
        }
        .padding(Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Radii.xl)
                .fill(Color.white)
                .shadow(color: Color.black.opacity(0.06), radius: 10, y: 4)
        )
    }
}

// MARK: - Previews

#Preview("Assessment Celebration") {
    AssessmentCompletionCelebration(
        assessmentType: .basdai,
        score: 3.2,
        interpretation: "Low Disease Activity"
    ) {
        print("Continue tapped")
    }
}

#Preview("Milestone Celebration") {
    MilestoneCelebrationView(
        milestone: WellnessMilestone(
            type: .streak(7),
            title: "7 Day Streak",
            subtitle: "One week of consistency!",
            date: Date(),
            icon: "flame.fill"
        )
    ) {
        print("Dismissed")
    }
}

#Preview("Streak Display") {
    StreakDisplayView(journeyManager: .shared)
        .padding()
}
