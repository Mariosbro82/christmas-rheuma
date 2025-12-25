//
//  OnboardingFlow.swift
//  InflamAI
//
//  Premium 12-page onboarding experience with Ankylosaurus mascot
//  Educational, engaging, and comprehensive introduction to AS management
//

import SwiftUI
import UserNotifications
import HealthKit
import CoreData

struct OnboardingFlow: View {
    @StateObject private var viewModel = OnboardingViewModel()
    @StateObject private var profileViewModel = UserProfileEditViewModel()
    @Environment(\.dismiss) private var dismiss
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false

    private let totalPages = 8  // Added profile setup page

    /// Checks if user can proceed from current page (validates required fields on profile page)
    private var canProceedFromCurrentPage: Bool {
        // Only validate on profile setup page (page 3)
        if viewModel.currentPage == 3 {
            // Require at least Gender, Height, and Weight for ML features
            return profileViewModel.gender.lowercased() != "" &&
                   profileViewModel.gender.lowercased() != "unknown" &&
                   profileViewModel.heightCm > 0 &&
                   profileViewModel.weightKg > 0
        }
        // All other pages can proceed
        return true
    }

    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                colors: [Color.blue.opacity(0.1), Color.purple.opacity(0.1)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            TabView(selection: $viewModel.currentPage) {
                // Page 1: Welcome + Meet Ankylosaurus (Combined)
                WelcomeAndMascotPage()
                    .tag(0)

                // Page 2: Understanding AS + Why Track (Combined)
                UnderstandingAndBenefitsPage()
                    .tag(1)

                // Page 3: All Features Overview (Combined: Daily logging, Meds, Exercise, Flares, Trends)
                AllFeaturesPage()
                    .tag(2)

                // Page 4: Profile Setup (NEW - collects demographics for ML)
                ProfileSetupOnboardingPage(viewModel: profileViewModel)
                    .tag(3)

                // Page 5: Permissions - HealthKit
                HealthKitPermissionPage(viewModel: viewModel)
                    .tag(4)

                // Page 6: Permissions - Notifications
                NotificationPermissionPage(viewModel: viewModel)
                    .tag(5)

                // Page 7: Reminder Preferences
                ReminderSetupPage(viewModel: viewModel)
                    .tag(6)

                // Page 8: Ready to Go!
                CompletionPage(
                    viewModel: viewModel,
                    hasCompletedOnboarding: $hasCompletedOnboarding,
                    onComplete: {
                        // Save profile data before completing onboarding
                        profileViewModel.saveProfile()
                        dismiss()
                    }
                )
                    .tag(7)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .animation(.easeInOut, value: viewModel.currentPage)

            // Custom Page Indicator & Navigation
            // Only show navigation overlay if not on completion page
            if viewModel.currentPage < totalPages - 1 {
                VStack(spacing: 0) {
                    Spacer()

                    VStack(spacing: 16) {
                        // Custom Page Indicator
                        HStack(spacing: 8) {
                            ForEach(0..<totalPages, id: \.self) { index in
                                Circle()
                                    .fill(index == viewModel.currentPage ? Color.blue : Color.gray.opacity(0.3))
                                    .frame(width: index == viewModel.currentPage ? 10 : 8, height: index == viewModel.currentPage ? 10 : 8)
                                    .animation(.easeInOut, value: viewModel.currentPage)
                            }
                        }

                        // Navigation Buttons
                        HStack {
                            // Back Button
                            if viewModel.currentPage > 0 {
                                Button {
                                    withAnimation {
                                        viewModel.currentPage -= 1
                                    }
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "chevron.left")
                                        Text("Back")
                                    }
                                    .font(.subheadline)
                                    .foregroundColor(.blue)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 10)
                                }
                            } else {
                                Spacer()
                                    .frame(width: 80)
                            }

                            Spacer()

                            // Next/Skip Button
                            Button {
                                // CRITICAL FIX: Save profile when leaving page 3 (Profile Setup)
                                // This prevents data loss if user quits after entering profile data
                                if viewModel.currentPage == 3 {
                                    profileViewModel.saveProfile()
                                    print("✅ Profile saved at page transition (intermediate save)")
                                }
                                withAnimation {
                                    viewModel.currentPage += 1
                                }
                            } label: {
                                HStack(spacing: 4) {
                                    Text(viewModel.currentPage < 4 ? "Next" : "Continue")
                                    Image(systemName: "chevron.right")
                                }
                                .font(.subheadline.weight(.semibold))
                                .foregroundColor(.white)
                                .padding(.horizontal, 24)
                                .padding(.vertical, 10)
                                .background(canProceedFromCurrentPage ? Color.blue : Color.gray)
                                .cornerRadius(20)
                            }
                            .disabled(!canProceedFromCurrentPage)
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 16)
                    .background(
                        Color(.systemBackground)
                            .opacity(0.95)
                            .shadow(color: Color.black.opacity(0.1), radius: 10, y: -5)
                    )
                }
                .ignoresSafeArea(edges: .bottom)
            }
        }
        .onAppear {
            viewModel.currentPage = 0
        }
    }
}

// MARK: - Page 1: Welcome + Meet Ankylosaurus (Combined)

struct WelcomeAndMascotPage: View {
    @State private var isAnimating = false

    var body: some View {
        ScrollView {
            VStack(spacing: 32) {
                // Welcome section
                VStack(spacing: 20) {
                    // App Icon/Logo
                    ZStack {
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [Color.blue, Color.purple],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 100, height: 100)

                        Image(systemName: "heart.text.square.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.white)
                    }
                    .shadow(color: Color.blue.opacity(0.3), radius: 20)

                    VStack(spacing: 8) {
                        Text("Welcome to")
                            .font(.title3)
                            .foregroundColor(.secondary)

                        Text("InflamAI")
                            .font(.system(size: 44, weight: .bold, design: .rounded))
                            .foregroundColor(.blue)

                        Text("Your Personal AS Management Companion")
                            .font(.body)
                            .multilineTextAlignment(.center)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                    }
                }
                .padding(.top, 40)

                // Quick features
                VStack(spacing: 12) {
                    FeatureHighlight(icon: "chart.xyaxis.line", text: "Track symptoms scientifically")
                    FeatureHighlight(icon: "pills.fill", text: "Manage medications effortlessly")
                    FeatureHighlight(icon: "figure.flexibility", text: "Follow personalized exercises")
                    FeatureHighlight(icon: "brain.head.profile", text: "Discover your patterns")
                }
                .padding(.horizontal, 24)

                Divider()
                    .padding(.vertical, 8)

                // Meet mascot section
                VStack(spacing: 24) {
                    Image("our dino")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 180, height: 180)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 20)
                                .fill(Color(.systemBackground))
                                .shadow(color: Color.black.opacity(0.1), radius: 10)
                        )
                        .scaleEffect(isAnimating ? 1.05 : 1.0)
                        .animation(
                            Animation.easeInOut(duration: 1.5)
                                .repeatForever(autoreverses: true),
                            value: isAnimating
                        )
                        .onAppear { isAnimating = true }

                    VStack(spacing: 12) {
                        Text("Meet Ankylosaurus!")
                            .font(.system(size: 28, weight: .bold, design: .rounded))
                            .foregroundColor(.green)

                        Text("Your Friendly Guide")
                            .font(.body)
                            .foregroundColor(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 16) {
                        InfoBubble(
                            text: "Did you know? The Ankylosaurus had a fused spine and armored plates—just like AS can fuse your spine!",
                            color: .green
                        )

                        InfoBubble(
                            text: "But here's the difference: with proper management, YOU have the power to maintain flexibility and live your best life!",
                            color: .blue
                        )

                        InfoBubble(
                            text: "I'll be here throughout your journey, cheering you on! 🎉",
                            color: .purple
                        )
                    }
                }
                .padding(.horizontal, 24)

                Spacer(minLength: 120)
            }
        }
    }
}

// MARK: - Page 2: Understanding AS + Why Track (Combined)

struct UnderstandingAndBenefitsPage: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 32) {
                // Understanding AS Section
                VStack(spacing: 20) {
                    Text("🦴")
                        .font(.system(size: 60))

                    Text("Understanding AS")
                        .font(.system(size: 32, weight: .bold))

                    VStack(spacing: 16) {
                        ASFactCard(
                            icon: "info.circle.fill",
                            title: "What is Ankylosing Spondylitis?",
                            description: "AS is inflammatory arthritis affecting the spine. 'Ankylosing' means stiffening, 'Spondylitis' means spine inflammation.",
                            color: .blue
                        )

                        ASFactCard(
                            icon: "heart.fill",
                            title: "Good News!",
                            description: "With proper management—medication, exercise, and tracking—most people with AS live full, active lives!",
                            color: .green
                        )
                    }
                }
                .padding(.top, 40)

                Divider()

                // Why Track Section
                VStack(spacing: 20) {
                    Text("🎯")
                        .font(.system(size: 60))

                    Text("Why Track Your Health?")
                        .font(.system(size: 28, weight: .bold))
                        .multilineTextAlignment(.center)

                    VStack(spacing: 14) {
                        BenefitCardCompact(
                            number: "1",
                            title: "Discover Your Patterns",
                            description: "What triggers your flares? Weather? Stress? Tracking reveals YOUR unique patterns.",
                            color: .purple
                        )

                        BenefitCardCompact(
                            number: "2",
                            title: "Better Doctor Visits",
                            description: "Show real data. \"I felt bad\" becomes \"My BASDAI increased from 3.2 to 5.8.\"",
                            color: .blue
                        )

                        BenefitCardCompact(
                            number: "3",
                            title: "Track Progress",
                            description: "See improvements over time. Celebrate wins and maintain healthy habits!",
                            color: .green
                        )
                    }
                }

                Spacer(minLength: 120)
            }
            .padding(.horizontal, 24)
        }
    }
}

// MARK: - Page 3: All Features Overview (Combined)

struct AllFeaturesPage: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 28) {
                VStack(spacing: 12) {
                    Text("✨")
                        .font(.system(size: 60))

                    Text("Everything You Need")
                        .font(.system(size: 32, weight: .bold))

                    Text("Powerful features designed for AS management")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 40)

                // Features in compact cards
                VStack(spacing: 16) {
                    FeatureCardCompact(
                        icon: "slider.horizontal.3",
                        title: "BASDAI Tracking",
                        description: "6-question disease activity score—the gold standard for AS",
                        color: .blue
                    )

                    FeatureCardCompact(
                        icon: "pills.fill",
                        title: "Medication Management",
                        description: "Smart reminders, adherence tracking, and quick logging",
                        color: .green
                    )

                    FeatureCardCompact(
                        icon: "figure.flexibility",
                        title: "52 AS-Specific Exercises",
                        description: "Stretching, strengthening, mobility & posture routines",
                        color: .orange
                    )

                    FeatureCardCompact(
                        icon: "exclamationmark.triangle.fill",
                        title: "Flare Management",
                        description: "JointTap SOS for quick capture during tough times",
                        color: .red
                    )

                    FeatureCardCompact(
                        icon: "chart.xyaxis.line",
                        title: "Trends & Insights",
                        description: "Beautiful charts, weather correlation, PDF reports",
                        color: .purple
                    )

                    FeatureCardCompact(
                        icon: "book.fill",
                        title: "Health Journal",
                        description: "Track mood, energy, sleep quality daily",
                        color: .cyan
                    )
                }

                AnkylosaurusTip(text: "All these features work together to help you understand and manage your AS! 💪")

                Spacer(minLength: 120)
            }
            .padding(.horizontal, 24)
        }
    }
}

// MARK: - Page 4: Why Track?

struct WhyTrackPage: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                VStack(spacing: 12) {
                    Text("🎯")
                        .font(.system(size: 70))

                    Text("Why Track Your Health?")
                        .font(.system(size: 32, weight: .bold))
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 40)

                VStack(spacing: 16) {
                    BenefitCard(
                        number: "1",
                        title: "Discover Your Patterns",
                        description: "What triggers your flares? Weather? Stress? Sleep? Tracking reveals YOUR unique patterns.",
                        icon: "brain.head.profile",
                        color: .purple
                    )

                    BenefitCard(
                        number: "2",
                        title: "Better Doctor Visits",
                        description: "Show your rheumatologist real data. \"I felt bad\" becomes \"My BASDAI increased from 3.2 to 5.8 over 2 weeks.\"",
                        icon: "doc.text.fill",
                        color: .blue
                    )

                    BenefitCard(
                        number: "3",
                        title: "Medication Decisions",
                        description: "Is your current treatment working? Data helps you and your doctor make informed decisions.",
                        icon: "pills.fill",
                        color: .green
                    )

                    BenefitCard(
                        number: "4",
                        title: "Stay Motivated",
                        description: "See your progress over time. Celebrate improvements and maintain healthy habits!",
                        icon: "star.fill",
                        color: .orange
                    )
                }

                Spacer(minLength: 120)
            }
            .padding()
        }
    }
}

// MARK: - Page 5: Daily Logging

struct DailyLoggingPage: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("📝")
                    .font(.system(size: 70))

                Text("Daily Symptom Logging")
                    .font(.system(size: 28, weight: .bold))
                    .multilineTextAlignment(.center)

                Text("Quick, easy, and scientifically validated")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                VStack(spacing: 16) {
                    FeatureDetailCard(
                        icon: "slider.horizontal.3",
                        title: "BASDAI Tracking",
                        description: "Answer 6 quick questions to calculate your disease activity score—the gold standard used by rheumatologists worldwide.",
                        gradient: [Color.blue, Color.cyan]
                    )

                    FeatureDetailCard(
                        icon: "bolt.fill",
                        title: "Pain Intensity",
                        description: "Track pain levels throughout your spine—from neck to lower back to hips.",
                        gradient: [Color.red, Color.orange]
                    )

                    FeatureDetailCard(
                        icon: "clock.fill",
                        title: "Morning Stiffness",
                        description: "Log how long your morning stiffness lasts—a key indicator of inflammation.",
                        gradient: [Color.purple, Color.pink]
                    )

                    FeatureDetailCard(
                        icon: "battery.75",
                        title: "Energy & Sleep",
                        description: "Track fatigue levels and sleep quality to understand their impact on your symptoms.",
                        gradient: [Color.green, Color.mint]
                    )
                }

                AnkylosaurusTip(text: "Pro tip: Log at the same time daily—consistency reveals patterns! ⏰")

                Spacer(minLength: 120)
            }
            .padding()
        }
    }
}

// MARK: - Page 6: Medication Management

struct MedicationFeaturePage: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("💊")
                    .font(.system(size: 70))

                Text("Medication Management")
                    .font(.system(size: 28, weight: .bold))
                    .multilineTextAlignment(.center)

                Text("Never miss a dose, track adherence")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                VStack(spacing: 16) {
                    FeatureDetailCard(
                        icon: "bell.badge.fill",
                        title: "Smart Reminders",
                        description: "Set multiple daily reminders for each medication. Get notifications at the right time, every time.",
                        gradient: [Color.blue, Color.indigo]
                    )

                    FeatureDetailCard(
                        icon: "checkmark.circle.fill",
                        title: "Quick Logging",
                        description: "Tap 'Taken' or 'Skip' in seconds. Track biologics, DMARDs, NSAIDs, and more.",
                        gradient: [Color.green, Color.teal]
                    )

                    FeatureDetailCard(
                        icon: "chart.bar.fill",
                        title: "Adherence Analytics",
                        description: "See your weekly and monthly adherence rates. Low adherence? We'll gently remind you why it matters.",
                        gradient: [Color.orange, Color.yellow]
                    )

                    FeatureDetailCard(
                        icon: "calendar",
                        title: "30-Day Calendar",
                        description: "Visual calendar showing every dose taken or missed. Patterns become obvious!",
                        gradient: [Color.purple, Color.pink]
                    )
                }

                AnkylosaurusTip(text: "Medication adherence is KEY to managing AS. Let's make it easy! 💪")

                Spacer(minLength: 120)
            }
            .padding()
        }
    }
}

// MARK: - Page 7: Exercise Library

struct ExerciseFeaturePage: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("🏃")
                    .font(.system(size: 70))

                Text("52 AS-Specific Exercises")
                    .font(.system(size: 28, weight: .bold))
                    .multilineTextAlignment(.center)

                Text("Professionally designed for AS patients")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                VStack(spacing: 12) {
                    ExerciseCategoryCard(icon: "figure.flexibility", category: "Stretching", count: 12, color: .blue)
                    ExerciseCategoryCard(icon: "figure.strengthtraining.traditional", category: "Strengthening", count: 12, color: .red)
                    ExerciseCategoryCard(icon: "figure.walk", category: "Mobility", count: 10, color: .green)
                    ExerciseCategoryCard(icon: "wind", category: "Breathing", count: 6, color: .cyan)
                    ExerciseCategoryCard(icon: "figure.stand", category: "Posture", count: 6, color: .purple)
                    ExerciseCategoryCard(icon: "figure.mind.and.body", category: "Balance", count: 6, color: .orange)
                }

                VStack(spacing: 16) {
                    FeatureDetailCard(
                        icon: "sparkles",
                        title: "AI Exercise Coach",
                        description: "Answer a few questions and get a personalized routine matching YOUR goals, mobility level, and time available.",
                        gradient: [Color.purple, Color.pink]
                    )

                    FeatureDetailCard(
                        icon: "list.bullet.clipboard",
                        title: "Step-by-Step Instructions",
                        description: "Each exercise includes detailed instructions, benefits, safety tips, and target areas.",
                        gradient: [Color.green, Color.mint]
                    )
                }

                AnkylosaurusTip(text: "Movement is medicine for AS! Even 5-10 minutes daily makes a huge difference. 🌟")

                Spacer(minLength: 120)
            }
            .padding()
        }
    }
}

// MARK: - Page 8: Flare Management

struct FlareManagementPage: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("🔥")
                    .font(.system(size: 70))

                Text("Flare Management")
                    .font(.system(size: 28, weight: .bold))
                    .multilineTextAlignment(.center)

                Text("Quick capture during tough times")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                VStack(spacing: 16) {
                    FeatureDetailCard(
                        icon: "exclamationmark.triangle.fill",
                        title: "JointTap SOS",
                        description: "Flare-up? Use our emergency interface with BIG buttons. Tap affected areas on the body diagram. Log in 3 taps, even with stiff fingers.",
                        gradient: [Color.red, Color.orange]
                    )

                    FeatureDetailCard(
                        icon: "calendar.badge.clock",
                        title: "Flare Timeline",
                        description: "See all your flares on a timeline. How often do they happen? How long do they last? What triggers them?",
                        gradient: [Color.orange, Color.yellow]
                    )

                    FeatureDetailCard(
                        icon: "brain.head.profile",
                        title: "Pattern Detection",
                        description: "Our AI analyzes your flares to identify common triggers: stress, weather, missed medications, poor sleep, and more.",
                        gradient: [Color.purple, Color.blue]
                    )

                    FeatureDetailCard(
                        icon: "chart.bar.xaxis",
                        title: "Frequency Analysis",
                        description: "6-month chart showing flare frequency. Are they getting better? More frequent? Data doesn't lie.",
                        gradient: [Color.blue, Color.cyan]
                    )
                }

                AnkylosaurusTip(text: "Tracking flares helps you avoid future ones. Knowledge is power! 💡")

                Spacer(minLength: 120)
            }
            .padding()
        }
    }
}

// MARK: - Page 9: Trends & Insights

struct TrendsFeaturePage: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("📈")
                    .font(.system(size: 70))

                Text("Trends & Insights")
                    .font(.system(size: 28, weight: .bold))
                    .multilineTextAlignment(.center)

                Text("Beautiful charts reveal your patterns")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                VStack(spacing: 16) {
                    FeatureDetailCard(
                        icon: "chart.xyaxis.line",
                        title: "Multi-Metric Charts",
                        description: "See pain, stiffness, fatigue, and BASDAI over time. Compare weeks, months, or years.",
                        gradient: [Color.blue, Color.purple]
                    )

                    FeatureDetailCard(
                        icon: "cloud.rain.fill",
                        title: "Weather Correlation",
                        description: "Does rain increase your pain? Cold weather worsen stiffness? We'll show you the connection!",
                        gradient: [Color.cyan, Color.blue]
                    )

                    FeatureDetailCard(
                        icon: "pills.circle.fill",
                        title: "Medication Impact",
                        description: "Visualize how medication changes affect your symptoms. Is your new biologic working?",
                        gradient: [Color.green, Color.mint]
                    )

                    FeatureDetailCard(
                        icon: "doc.richtext",
                        title: "PDF Reports for Doctors",
                        description: "Generate professional 3-page reports with charts, BASDAI scores, and medication adherence. Your rheumatologist will LOVE this!",
                        gradient: [Color.orange, Color.red]
                    )
                }

                AnkylosaurusTip(text: "Bring your PDF report to every appointment. Be your own health advocate! 📄")

                Spacer(minLength: 120)
            }
            .padding()
        }
    }
}

// MARK: - Page 10: HealthKit Permission

struct HealthKitPermissionPage: View {
    @ObservedObject var viewModel: OnboardingViewModel

    var body: some View {
        VStack(spacing: 30) {
            Spacer()

            Image(systemName: "heart.text.square.fill")
                .font(.system(size: 80))
                .foregroundColor(.pink)

            Text("Connect with Health")
                .font(.system(size: 32, weight: .bold))

            Text("Optional but recommended")
                .font(.subheadline)
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 20) {
                PermissionBenefit(
                    icon: "figure.walk",
                    title: "Activity Data",
                    description: "See how steps and exercise affect your symptoms"
                )

                PermissionBenefit(
                    icon: "bed.double.fill",
                    title: "Sleep Tracking",
                    description: "Correlate sleep quality with morning stiffness"
                )

                PermissionBenefit(
                    icon: "heart.fill",
                    title: "Heart Rate",
                    description: "Monitor inflammation through resting heart rate"
                )
            }
            .padding(.horizontal)

            if viewModel.healthKitAuthorized {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("HealthKit Connected!")
                        .fontWeight(.semibold)
                }
                .padding()
                .background(Color.green.opacity(0.1))
                .cornerRadius(12)
            } else {
                Button {
                    viewModel.requestHealthKitAuthorization()
                } label: {
                    Label("Connect HealthKit", systemImage: "heart.fill")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.pink)
                        .cornerRadius(12)
                }

                Button {
                    withAnimation {
                        viewModel.currentPage += 1
                    }
                } label: {
                    Text("Skip for now")
                        .foregroundColor(.secondary)
                }
            }

            Spacer()
        }
        .padding()
    }
}

// MARK: - Page 11: Notification Permission

struct NotificationPermissionPage: View {
    @ObservedObject var viewModel: OnboardingViewModel

    var body: some View {
        VStack(spacing: 30) {
            Spacer()

            Image(systemName: "bell.badge.fill")
                .font(.system(size: 80))
                .foregroundColor(.blue)

            Text("Enable Notifications")
                .font(.system(size: 32, weight: .bold))

            Text("Stay on track with gentle reminders")
                .font(.subheadline)
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 20) {
                PermissionBenefit(
                    icon: "pills.fill",
                    title: "Medication Reminders",
                    description: "Never miss a dose with timely notifications"
                )

                PermissionBenefit(
                    icon: "pencil.circle.fill",
                    title: "Daily Log Reminders",
                    description: "Gentle nudges to log your symptoms"
                )

                PermissionBenefit(
                    icon: "figure.flexibility",
                    title: "Exercise Reminders",
                    description: "Stay consistent with your routine"
                )
            }
            .padding(.horizontal)

            if viewModel.notificationsAuthorized {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Notifications Enabled!")
                        .fontWeight(.semibold)
                }
                .padding()
                .background(Color.green.opacity(0.1))
                .cornerRadius(12)
            } else {
                Button {
                    viewModel.requestNotificationAuthorization()
                } label: {
                    Label("Enable Notifications", systemImage: "bell.fill")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(12)
                }

                Button {
                    withAnimation {
                        viewModel.currentPage += 1
                    }
                } label: {
                    Text("Skip for now")
                        .foregroundColor(.secondary)
                }
            }

            Spacer()
        }
        .padding()
    }
}

// MARK: - Page 12: Completion
struct ReminderSetupPage: View {
    @ObservedObject var viewModel: OnboardingViewModel
    private let timezoneChoices: [(identifier: String, displayName: String)]
    
    init(viewModel: OnboardingViewModel) {
        self.viewModel = viewModel
        var options = Self.buildTimeZoneChoices()
        if !options.contains(where: { $0.identifier == viewModel.selectedTimeZoneIdentifier }),
           let tz = TimeZone(identifier: viewModel.selectedTimeZoneIdentifier) {
            let isDevice = tz.identifier == TimeZone.current.identifier
            options.insert((tz.identifier, Self.formattedName(for: tz, isDevice: isDevice)), at: 0)
        }
        self.timezoneChoices = options
    }
    
    private var weekdaySymbols: [String] {
        Calendar(identifier: .gregorian).weekdaySymbols
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 28) {
                VStack(spacing: 12) {
                    Text("Tailor Your Reminders")
                        .font(.title2.weight(.bold))
                        .multilineTextAlignment(.center)
                    
                    Text("Pick the timezone and reminder times you prefer. You can tweak these later in Settings.")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 24)
                
                VStack(alignment: .leading, spacing: 16) {
                    Text("Preferred Timezone")
                        .font(.headline)
                    
                    Picker("Preferred Timezone", selection: $viewModel.selectedTimeZoneIdentifier) {
                        ForEach(timezoneChoices, id: \.identifier) { choice in
                            Text(choice.displayName)
                                .tag(choice.identifier)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("timezone_picker")
                }
                .padding()
                .background(.thinMaterial)
                .cornerRadius(20)
                
                VStack(alignment: .leading, spacing: 16) {
                    Text("Daily BASDAI Reminder")
                        .font(.headline)
                    
                    DatePicker(
                        "Daily BASDAI Reminder",
                        selection: $viewModel.dailyReminderTime,
                        displayedComponents: .hourAndMinute
                    )
                    .datePickerStyle(.wheel)
                    .labelsHidden()
                    .accessibilityIdentifier("daily_basdai_time_picker")
                    
                    Text("We'll nudge you once per day to complete the BASDAI symptoms check-in.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .background(.thinMaterial)
                .cornerRadius(20)
                
                VStack(alignment: .leading, spacing: 16) {
                    Text("Weekly BASFI & BAS-G Reminder")
                        .font(.headline)
                    
                    Picker("Weekly Reminder Day", selection: $viewModel.weeklyReminderWeekday) {
                        ForEach(1...7, id: \.self) { index in
                            Text(weekdaySymbols[index - 1])
                                .tag(index)
                        }
                    }
                    .pickerStyle(.segmented)
                    .accessibilityIdentifier("weekly_weekday_picker")
                    
                    DatePicker(
                        "Weekly Reminder Time",
                        selection: $viewModel.weeklyReminderTime,
                        displayedComponents: .hourAndMinute
                    )
                    .datePickerStyle(.wheel)
                    .labelsHidden()
                    .accessibilityIdentifier("weekly_reminder_time_picker")
                    
                    Text("We'll send a gentle reminder on your chosen day to capture BASFI (function) and BAS-G (well-being).")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .background(.thinMaterial)
                .cornerRadius(20)
                
                Text("You can revisit these preferences anytime in Settings → Notifications & Reminders.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.bottom, 32)
            }
            .padding(.horizontal, 24)
        }
    }
    
    private static func buildTimeZoneChoices() -> [(identifier: String, displayName: String)] {
        var identifiers = [
            "Europe/Berlin",
            "Europe/London",
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Singapore",
            "Australia/Sydney"
        ]
        
        let deviceIdentifier = TimeZone.current.identifier
        if !identifiers.contains(deviceIdentifier) {
            identifiers.insert(deviceIdentifier, at: 0)
        }
        
        var seen = Set<String>()
        return identifiers.compactMap { identifier in
            guard let timezone = TimeZone(identifier: identifier), seen.insert(identifier).inserted else {
                return nil
            }
            let isDevice = identifier == deviceIdentifier
            let display = formattedName(for: timezone, isDevice: isDevice)
            return (identifier, display)
        }
    }
    
    private static func formattedName(for timezone: TimeZone, isDevice: Bool) -> String {
        let totalMinutes = timezone.secondsFromGMT() / 60
        let hours = totalMinutes / 60
        let minutes = abs(totalMinutes % 60)
        let readableIdentifier = timezone.identifier.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "/", with: " / ")
        let offset = String(format: "UTC%+02d:%02d", hours, minutes)
        let deviceSuffix = isDevice ? " (Device default)" : ""
        return "\(readableIdentifier) — \(offset)\(deviceSuffix)"
    }
}

struct CompletionPage: View {
    @ObservedObject var viewModel: OnboardingViewModel
    @Binding var hasCompletedOnboarding: Bool
    var onComplete: (() -> Void)? = nil
    @State private var isAnimating = false
    @State private var showConfetti = false

    var body: some View {
        ZStack {
            // Animated gradient background
            LinearGradient(
                colors: [
                    Color.blue.opacity(0.15),
                    Color.purple.opacity(0.15),
                    Color.green.opacity(0.1)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 0) {
                    Spacer(minLength: 40)

                    // Hero section with mascot
                ZStack {
                    // Animated glow circles
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    Color.green.opacity(0.3),
                                    Color.blue.opacity(0.2),
                                    Color.clear
                                ],
                                center: .center,
                                startRadius: 50,
                                endRadius: 200
                            )
                        )
                        .frame(width: 400, height: 400)
                        .scaleEffect(isAnimating ? 1.2 : 1.0)
                        .opacity(isAnimating ? 0.6 : 0.3)
                        .animation(
                            Animation.easeInOut(duration: 2.5)
                                .repeatForever(autoreverses: true),
                            value: isAnimating
                        )

                    VStack(spacing: 16) {
                        // Mascot image with celebration emoji
                        ZStack(alignment: .bottomTrailing) {
                            Image("our dino")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 200, height: 200)
                                .padding(24)
                                .background(
                                    Circle()
                                        .fill(
                                            LinearGradient(
                                                colors: [
                                                    Color.cyan.opacity(0.3),
                                                    Color.blue.opacity(0.2)
                                                ],
                                                startPoint: .topLeading,
                                                endPoint: .bottomTrailing
                                            )
                                        )
                                )
                                .overlay(
                                    Circle()
                                        .stroke(
                                            LinearGradient(
                                                colors: [Color.white.opacity(0.5), Color.clear],
                                                startPoint: .topLeading,
                                                endPoint: .bottomTrailing
                                            ),
                                            lineWidth: 2
                                        )
                                )
                                .shadow(color: Color.blue.opacity(0.3), radius: 20, x: 0, y: 10)
                                .scaleEffect(isAnimating ? 1.05 : 1.0)
                                .animation(
                                    Animation.easeInOut(duration: 1.5)
                                        .repeatForever(autoreverses: true),
                                    value: isAnimating
                                )

                            // Party emoji
                            Text("🎉")
                                .font(.system(size: 60))
                                .offset(x: 20, y: 20)
                                .scaleEffect(showConfetti ? 1.2 : 0.8)
                                .animation(
                                    Animation.spring(response: 0.6, dampingFraction: 0.5)
                                        .repeatForever(autoreverses: true),
                                    value: showConfetti
                                )
                        }
                    }
                }
                .padding(.bottom, 32)

                // Title section
                VStack(spacing: 8) {
                    Text("You're All Set!")
                        .font(.system(size: 42, weight: .bold, design: .rounded))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [Color.green, Color.blue],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )

                    Text("Welcome to InflamAI")
                        .font(.title2)
                        .foregroundColor(.secondary)
                        .fontWeight(.medium)
                }
                .padding(.bottom, 32)

                // Checkmarks section
                VStack(spacing: 12) {
                    CompletionCheckmarkEnhanced(text: "Understanding AS and why tracking matters", delay: 0.1)
                    CompletionCheckmarkEnhanced(text: "All features explained", delay: 0.2)
                    CompletionCheckmarkEnhanced(text: "Your profile is set up for personalized insights", delay: 0.3)
                    CompletionCheckmarkEnhanced(text: "Permissions configured", delay: 0.4)
                    CompletionCheckmarkEnhanced(text: "Ready to start your health journey!", delay: 0.5)
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 32)

                // Motivational text
                VStack(spacing: 12) {
                    Text("Let's take control of your AS together!")
                        .font(.title3)
                        .fontWeight(.bold)
                        .multilineTextAlignment(.center)
                        .foregroundColor(.primary)

                    Text("Remember: I'm here to help every step of the way.\nYou've got this! 💪")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .lineSpacing(4)
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 40)

                // CTA Button
                Button {
                    hasCompletedOnboarding = true
                    onComplete?()
                } label: {
                    HStack(spacing: 12) {
                        Text("Start Tracking")
                            .font(.title3)
                            .fontWeight(.bold)

                        Image(systemName: "arrow.right.circle.fill")
                            .font(.title2)
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 18)
                    .background(
                        LinearGradient(
                            colors: [Color.blue, Color.purple],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(16)
                    .shadow(color: Color.blue.opacity(0.4), radius: 20, x: 0, y: 10)
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 24)
                .scaleEffect(isAnimating ? 1.02 : 1.0)
                .animation(
                    Animation.easeInOut(duration: 1.0)
                        .repeatForever(autoreverses: true),
                    value: isAnimating
                )

                    Spacer(minLength: 40)
                }
            }
        }
        .onAppear {
            isAnimating = true
            withAnimation(.spring(response: 0.8, dampingFraction: 0.6).delay(0.3)) {
                showConfetti = true
            }
        }
    }
}

// Enhanced completion checkmark with animation
struct CompletionCheckmarkEnhanced: View {
    let text: String
    let delay: Double
    @State private var isVisible = false

    var body: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(Color.green.opacity(0.2))
                    .frame(width: 32, height: 32)

                Image(systemName: "checkmark.circle.fill")
                    .font(.title3)
                    .foregroundColor(.green)
            }
            .scaleEffect(isVisible ? 1.0 : 0.5)
            .opacity(isVisible ? 1.0 : 0.0)

            Text(text)
                .font(.body)
                .fontWeight(.medium)
                .foregroundColor(.primary)
                .opacity(isVisible ? 1.0 : 0.0)

            Spacer()
        }
        .padding(.vertical, 8)
        .onAppear {
            withAnimation(.spring(response: 0.6, dampingFraction: 0.7).delay(delay)) {
                isVisible = true
            }
        }
    }
}

// MARK: - Supporting Components

struct FeatureHighlight: View {
    let icon: String
    let text: String

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 30)

            Text(text)
                .font(.body)

            Spacer()
        }
    }
}

struct InfoBubble: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "bubble.left.fill")
                .foregroundColor(color)

            Text(text)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

struct ASFactCard: View {
    let icon: String
    let title: String
    let description: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title2)

                Text(title)
                    .font(.headline)
            }

            Text(description)
                .font(.body)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }
}

struct BenefitCard: View {
    let number: String
    let title: String
    let description: String
    let icon: String
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            ZStack {
                Circle()
                    .fill(color)
                    .frame(width: 50, height: 50)

                Text(number)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: icon)
                        .foregroundColor(color)
                    Text(title)
                        .font(.headline)
                }

                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }
}

struct FeatureDetailCard: View {
    let icon: String
    let title: String
    let description: String
    let gradient: [Color]

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            // Icon with gradient background
            ZStack {
                LinearGradient(
                    colors: gradient,
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .frame(width: 50, height: 50)
                .cornerRadius(12)

                Image(systemName: icon)
                    .foregroundColor(.white)
                    .font(.title3)
            }

            // Text content
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary)

                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.08), radius: 8, x: 0, y: 2)
        )
    }
}

struct AnkylosaurusTip: View {
    let text: String

    var body: some View {
        HStack(spacing: 12) {
            Text("🦕")
                .font(.system(size: 40))

            Text(text)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.green)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.green.opacity(0.1))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.green, lineWidth: 2)
        )
    }
}

struct ExerciseCategoryCard: View {
    let icon: String
    let category: String
    let count: Int
    let color: Color

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.title3)
                .frame(width: 40)

            Text(category)
                .font(.body)
                .fontWeight(.semibold)

            Spacer()

            Text("\(count) exercises")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(10)
    }
}

struct PermissionBenefit: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .font(.title3)
                .frame(width: 30)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)

                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct CompletionCheckmark: View {
    let text: String

    var body: some View {
        HStack {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)

            Text(text)
                .font(.body)

            Spacer()
        }
    }
}

struct BenefitCardCompact: View {
    let number: String
    let title: String
    let description: String
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle()
                    .fill(color)
                    .frame(width: 36, height: 36)

                Text(number)
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(color: Color.black.opacity(0.05), radius: 3)
    }
}

struct FeatureCardCompact: View {
    let icon: String
    let title: String
    let description: String
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 44, height: 44)

                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title3)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding(12)
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(color: Color.black.opacity(0.05), radius: 3)
    }
}

// MARK: - Profile Setup Page for Onboarding

struct ProfileSetupOnboardingPage: View {
    @ObservedObject var viewModel: UserProfileEditViewModel

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 12) {
                    Image(systemName: "person.crop.circle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.blue)

                    Text("Tell Us About You")
                        .font(.system(size: 28, weight: .bold))

                    Text("This helps personalize your experience and improve predictions")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 40)

                // Essential Fields
                VStack(spacing: 16) {
                    // Gender
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Gender")
                            .font(.headline)

                        Picker("Gender", selection: $viewModel.gender) {
                            Text("Select...").tag("")
                            Text("Male").tag("male")
                            Text("Female").tag("female")
                            Text("Other").tag("other")
                        }
                        .pickerStyle(.segmented)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Height & Weight
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Height")
                                .font(.headline)
                            HStack {
                                TextField("170", value: $viewModel.heightCm, format: .number)
                                    .keyboardType(.decimalPad)
                                    .textFieldStyle(.roundedBorder)
                                Text("cm")
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Weight")
                                .font(.headline)
                            HStack {
                                TextField("70", value: $viewModel.weightKg, format: .number)
                                    .keyboardType(.decimalPad)
                                    .textFieldStyle(.roundedBorder)
                                Text("kg")
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }

                    // BMI Display
                    if viewModel.calculatedBMI > 0 {
                        HStack {
                            Text("Your BMI:")
                                .foregroundColor(.secondary)
                            Text(String(format: "%.1f", viewModel.calculatedBMI))
                                .fontWeight(.bold)
                                .foregroundColor(viewModel.bmiColor)
                            Text("(\(viewModel.bmiCategory))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(viewModel.bmiColor.opacity(0.1))
                        .cornerRadius(12)
                    }

                    // Date of Birth
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Date of Birth")
                            .font(.headline)
                        DatePicker("", selection: $viewModel.dateOfBirth, in: ...Date(), displayedComponents: .date)
                            .datePickerStyle(.compact)
                            .labelsHidden()
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Smoking Status
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Smoking Status")
                            .font(.headline)
                        Picker("Smoking", selection: $viewModel.smokingStatus) {
                            Text("Never").tag("never")
                            Text("Former").tag("former")
                            Text("Current").tag("current")
                        }
                        .pickerStyle(.segmented)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Diagnosis Date
                    VStack(alignment: .leading, spacing: 8) {
                        Text("AS Diagnosis Date")
                            .font(.headline)
                        DatePicker("", selection: $viewModel.diagnosisDate, in: ...Date(), displayedComponents: .date)
                            .datePickerStyle(.compact)
                            .labelsHidden()
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // HLA-B27 Status
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("HLA-B27 Positive?")
                                .font(.headline)
                            Text("Common genetic marker for AS")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Toggle("", isOn: $viewModel.hlaB27Positive)
                            .labelsHidden()
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }

                // Required Fields Indicator
                if !viewModel.isProfileComplete {
                    VStack(spacing: 8) {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.orange)
                            Text("Please complete required fields:")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.orange)
                        }

                        HStack(spacing: 12) {
                            RequiredFieldBadge(label: "Gender", isComplete: !viewModel.gender.isEmpty && viewModel.gender != "")
                            RequiredFieldBadge(label: "Height", isComplete: viewModel.heightCm > 0)
                            RequiredFieldBadge(label: "Weight", isComplete: viewModel.weightKg > 0)
                        }

                        Text("These are essential for personalized predictions")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(12)
                }

                // Completion Badge
                if viewModel.isProfileComplete {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("Profile ready for personalized predictions!")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(.green)
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(12)
                }

                // Privacy Note
                HStack(spacing: 8) {
                    Image(systemName: "lock.shield.fill")
                        .foregroundColor(.green)
                    Text("All data stays on your device. Never shared.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()

                Spacer(minLength: 120)
            }
            .padding(.horizontal, 24)
        }
    }
}

// MARK: - User Profile Edit ViewModel (Consolidated)
// This ViewModel is used across OnboardingFlow, SettingsView, and UserProfileEditView
// to ensure consistent user profile management

@MainActor
class UserProfileEditViewModel: ObservableObject {
    // Personal Info
    @Published var name: String = ""
    @Published var dateOfBirth: Date = Calendar.current.date(byAdding: .year, value: -30, to: Date()) ?? Date()
    @Published var gender: String = ""

    // Body Measurements
    @Published var heightCm: Float = 0
    @Published var weightKg: Float = 0

    // AS Info
    @Published var diagnosisDate: Date = Date()
    @Published var hlaB27Positive: Bool = false
    @Published var biologicExperienced: Bool = false

    // Lifestyle
    @Published var smokingStatus: String = "never"

    // Healthcare
    @Published var primaryPhysicianName: String = ""
    @Published var rheumatologistName: String = ""

    private let persistenceController = InflamAIPersistenceController.shared
    private var existingProfile: UserProfile?

    var calculatedBMI: Float {
        guard heightCm > 0 && weightKg > 0 else { return 0 }
        let heightM = heightCm / 100.0
        return weightKg / (heightM * heightM)
    }

    var bmiColor: Color {
        let bmi = calculatedBMI
        if bmi < 18.5 { return .orange }
        if bmi < 25 { return .green }
        if bmi < 30 { return .orange }
        return .red
    }

    var bmiCategory: String {
        let bmi = calculatedBMI
        if bmi < 18.5 { return "Underweight" }
        if bmi < 25 { return "Normal" }
        if bmi < 30 { return "Overweight" }
        return "Obese"
    }

    var completenessPercentage: Int {
        var complete = 0
        let total = 6  // Key fields for ML

        if !gender.isEmpty && gender != "" { complete += 1 }
        if heightCm > 0 { complete += 1 }
        if weightKg > 0 { complete += 1 }
        if !smokingStatus.isEmpty { complete += 1 }
        // Date of birth is always set
        complete += 1
        // Diagnosis date counts if not today
        if !Calendar.current.isDateInToday(diagnosisDate) { complete += 1 }

        return Int((Double(complete) / Double(total)) * 100)
    }

    var missingFields: [String] {
        var missing: [String] = []
        if gender.isEmpty || gender == "" { missing.append("Gender") }
        if heightCm == 0 { missing.append("Height") }
        if weightKg == 0 { missing.append("Weight") }
        if Calendar.current.isDateInToday(diagnosisDate) { missing.append("Diagnosis Date") }
        return missing
    }

    /// True if minimum required fields for ML are complete
    var isProfileComplete: Bool {
        return !gender.isEmpty &&
               gender != "" &&
               gender.lowercased() != "unknown" &&
               heightCm > 0 &&
               weightKg > 0
    }

    init() {
        loadExistingProfile()
    }

    private func loadExistingProfile() {
        let context = persistenceController.container.viewContext
        let request: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        request.fetchLimit = 1

        if let profile = try? context.fetch(request).first {
            existingProfile = profile

            // Load values
            name = profile.name ?? ""
            if let dob = profile.dateOfBirth { dateOfBirth = dob }

            // FIXED: Normalize gender on load - catch invalid values like "yes", "true", etc.
            let loadedGender = profile.gender ?? ""
            let validGenders = ["male", "female", "other", "unknown", ""]
            if validGenders.contains(loadedGender.lowercased()) {
                gender = loadedGender.lowercased()
            } else {
                // Invalid value stored - reset to empty and log
                #if DEBUG
                print("⚠️ [Profile] Invalid gender value '\(loadedGender)' found - resetting to empty")
                #endif
                gender = ""
            }
            heightCm = profile.heightCm
            weightKg = profile.weightKg
            if let diagnosis = profile.diagnosisDate { diagnosisDate = diagnosis }
            hlaB27Positive = profile.hlaB27Positive
            biologicExperienced = profile.biologicExperienced
            smokingStatus = profile.smokingStatus ?? "never"
            primaryPhysicianName = profile.primaryPhysicianName ?? ""
            rheumatologistName = profile.rheumatologistName ?? ""
        }
    }

    /// Validates profile data before saving
    /// Returns array of validation warnings (empty = valid)
    func validateProfile() -> [String] {
        var warnings: [String] = []

        // Critical validations
        if heightCm <= 0 || heightCm > 300 {
            warnings.append("Height not set or invalid (\(heightCm)cm)")
        }
        if weightKg <= 0 || weightKg > 500 {
            warnings.append("Weight not set or invalid (\(weightKg)kg)")
        }
        if name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            warnings.append("Name is empty")
        }

        // BMI range check (only if height/weight valid)
        if heightCm > 0 && weightKg > 0 {
            let bmi = calculatedBMI
            if bmi < 10 || bmi > 60 {
                warnings.append("BMI seems unusual (\(String(format: "%.1f", bmi)))")
            }
        }

        return warnings
    }

    func saveProfile() {
        let context = persistenceController.container.viewContext

        // FIXED: Validate before saving - warn but don't block (users may want partial profiles)
        let warnings = validateProfile()
        if !warnings.isEmpty {
            #if DEBUG
            print("⚠️ [Profile] Validation warnings: \(warnings.joined(separator: ", "))")
            #endif
        }

        let profile: UserProfile
        if let existing = existingProfile {
            profile = existing
        } else {
            profile = UserProfile(context: context)
            profile.id = UUID()
            profile.createdAt = Date()
        }

        // Save all fields
        profile.name = name.isEmpty ? nil : name
        profile.dateOfBirth = dateOfBirth

        // FIXED: Validate gender before saving - only allow known values
        let validGenders = ["male", "female", "other", "unknown"]
        let normalizedGender = gender.lowercased()
        if gender.isEmpty {
            profile.gender = nil
        } else if validGenders.contains(normalizedGender) {
            profile.gender = normalizedGender
        } else {
            #if DEBUG
            print("⚠️ [Profile] Rejecting invalid gender value '\(gender)' - not saving")
            #endif
            profile.gender = nil
        }
        profile.heightCm = heightCm
        profile.weightKg = weightKg
        profile.bmi = calculatedBMI
        profile.diagnosisDate = diagnosisDate
        profile.hlaB27Positive = hlaB27Positive
        profile.biologicExperienced = biologicExperienced
        profile.smokingStatus = smokingStatus
        profile.primaryPhysicianName = primaryPhysicianName.isEmpty ? nil : primaryPhysicianName
        profile.rheumatologistName = rheumatologistName.isEmpty ? nil : rheumatologistName
        profile.lastModified = Date()

        // FIXED: Proper error handling with validation context
        do {
            try context.save()
            let status = warnings.isEmpty ? "✅ valid" : "⚠️ with warnings"
            #if DEBUG
            print("\(status) Profile saved:")
            print("   Name: \(name.isEmpty ? "(empty)" : name)")
            print("   Gender: \(profile.gender ?? "(not set)")")
            print("   Height: \(heightCm)cm, Weight: \(weightKg)kg, BMI: \(String(format: "%.1f", calculatedBMI))")
            print("   HLA-B27: \(hlaB27Positive), Biologic: \(biologicExperienced)")
            #endif
            existingProfile = profile  // Update reference for future saves
        } catch {
            print("❌ CRITICAL: Failed to save profile: \(error)")
        }
    }
}

// MARK: - Onboarding View Model

@MainActor
class OnboardingViewModel: ObservableObject {
    @Published var currentPage = 0
    @Published var healthKitAuthorized = false
    @Published var notificationsAuthorized = false
    @Published var selectedTimeZoneIdentifier: String = TimeZone(identifier: "Europe/Berlin")?.identifier ?? TimeZone.current.identifier
    @Published var dailyReminderTime: Date = OnboardingViewModel.defaultReminderDate(hour: 20, minute: 0)
    @Published var weeklyReminderTime: Date = OnboardingViewModel.defaultReminderDate(hour: 18, minute: 0)
    @Published var weeklyReminderWeekday: Int = 1 // Sunday default

    // NOTE: Using HealthKitService.shared instead of local healthStore
    // This ensures unified authorization across the app

    init() {
        // Note: Questionnaire preferences loading is commented out to avoid type resolution issues
        // In production, this would load saved preferences from QuestionnairePreferences
    }

    /// Request HealthKit authorization using unified HealthKitService
    /// This requests ALL 28 data types needed for ML predictions
    func requestHealthKitAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("❌ [Onboarding] HealthKit not available on this device")
            return
        }

        Task { @MainActor in
            do {
                // Use HealthKitService for unified authorization (28 types)
                try await HealthKitService.shared.requestAuthorization()
                self.healthKitAuthorized = true
                print("✅ [Onboarding] HealthKit authorized for all 28 data types")

                withAnimation {
                    self.currentPage += 1
                }
            } catch {
                print("❌ [Onboarding] HealthKit authorization failed: \(error.localizedDescription)")
                // Don't block onboarding - user can skip
                self.healthKitAuthorized = false
            }
        }
    }
    
    func requestNotificationAuthorization() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            DispatchQueue.main.async {
                self.notificationsAuthorized = granted
                if granted {
                    withAnimation {
                        self.currentPage += 1
                    }
                }
            }
        }
    }
    
    // Questionnaire preference persistence - commented out due to type resolution issues
    // This method is not currently called in the onboarding flow
    /*
    func persistQuestionnairePreferences() {
        let timezoneID = selectedTimeZoneIdentifier
        let timezone = TimeZone(identifier: timezoneID) ?? TimeZone(identifier: "Europe/Berlin") ?? .current
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timezone

        let dailyComponents = calendar.dateComponents([.hour, .minute], from: dailyReminderTime)
        let weeklyComponents = calendar.dateComponents([.hour, .minute], from: weeklyReminderTime)

        // Preference persistence would happen here with QuestionnaireSchedule objects
    }
    */
    
    private static func defaultReminderDate(hour: Int, minute: Int) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .current
        let now = Date()
        return calendar.date(bySettingHour: hour, minute: minute, second: 0, of: now) ?? now
    }
    
    // Commented out - references undefined QuestionnaireSchedule type
    /*
    private static func date(from frequency: QuestionnaireSchedule.Frequency, fallbackHour: Int, fallbackMinute: Int, timezoneIdentifier: String) -> Date {
        let components: DateComponents
        switch frequency {
        case .daily(let time):
            components = time
        case .weekly(_, let time):
            components = time
        }

        let hour = components.hour ?? fallbackHour
        let minute = components.minute ?? fallbackMinute

        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: timezoneIdentifier) ?? .current
        let now = Date()
        return calendar.date(bySettingHour: hour, minute: minute, second: 0, of: now)
            ?? defaultReminderDate(hour: hour, minute: minute)
    }
    */
}

// MARK: - Required Field Badge

/// Small badge showing completion status for a required field
struct RequiredFieldBadge: View {
    let label: String
    let isComplete: Bool

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: isComplete ? "checkmark.circle.fill" : "circle")
                .font(.caption)
                .foregroundColor(isComplete ? .green : .gray)
            Text(label)
                .font(.caption)
                .foregroundColor(isComplete ? .primary : .secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isComplete ? Color.green.opacity(0.1) : Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

// MARK: - Preview

struct OnboardingFlow_Previews: PreviewProvider {
    static var previews: some View {
        OnboardingFlow()
    }
}
