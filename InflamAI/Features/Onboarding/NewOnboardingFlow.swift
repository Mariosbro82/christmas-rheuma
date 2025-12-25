//
//  NewOnboardingFlow.swift
//  InflamAI
//
//  Premium 12-page onboarding with hand-drawn dino illustrations
//  Inspired by modern app onboarding design patterns
//

import SwiftUI
import HealthKit
import UserNotifications
import CoreData
#if canImport(Lottie)
import Lottie
#endif

// MARK: - Design System Constants (Mapped to Global Design System)

struct OnboardingDesign {
    // Colors - Using Global Design System Tokens
    static let primaryColor = Colors.Primary.p500
    static let accentColor = Colors.Accent.teal
    static let backgroundColor = Colors.Gray.g50
    static let cardBackground = Color.white
    static let textPrimary = Colors.Gray.g900
    static let textSecondary = Colors.Gray.g500
    static let toggleOn = Colors.Semantic.success

    // Typography - Using Global Design System Tokens
    static let titleFont = Font.system(size: Typography.xxl, weight: .bold, design: .rounded)
    static let subtitleFont = Font.system(size: Typography.md, weight: .regular)
    static let labelFont = Font.system(size: Typography.md, weight: .semibold)
    static let bodyFont = Font.system(size: Typography.base, weight: .regular)
    static let captionFont = Font.system(size: Typography.sm, weight: .regular)

    // Spacing - Using Global Design System Tokens
    static let horizontalPadding: CGFloat = Spacing.lg
    static let cardCornerRadius: CGFloat = Radii.xl
    static let buttonCornerRadius: CGFloat = Radii.full
}

// MARK: - Main Onboarding Flow

struct NewOnboardingFlow: View {
    @StateObject private var viewModel = NewOnboardingViewModel()
    @Environment(\.dismiss) private var dismiss
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false

    var body: some View {
        ZStack {
            // Background
            OnboardingDesign.backgroundColor
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // Progress Indicator
                OnboardingProgressIndicator(
                    currentPage: viewModel.currentPage,
                    totalPages: viewModel.totalPages
                )
                .padding(.top, 16)
                .padding(.horizontal, OnboardingDesign.horizontalPadding)

                // Page Content
                TabView(selection: $viewModel.currentPage) {
                    // Page 1: Welcome
                    WelcomePage()
                        .tag(0)

                    // Page 2: Understanding AS
                    UnderstandingASPage()
                        .tag(1)

                    // Page 3: Daily Check-In
                    DailyCheckInPage()
                        .tag(2)

                    // Page 4: Body Map Tour
                    BodyMapTourPage()
                        .tag(3)

                    // Page 5: Medication Setup
                    MedicationSetupPage(viewModel: viewModel)
                        .tag(4)

                    // Page 6: Exercise Discovery
                    ExerciseDiscoveryPage()
                        .tag(5)

                    // Page 7: Flare Tracking
                    FlareTrackingPage()
                        .tag(6)

                    // Page 8: AI & Weather Intelligence
                    AIWeatherPage()
                        .tag(7)

                    // Page 9: Trends & Reports
                    TrendsReportsPage()
                        .tag(8)

                    // Page 10: Privacy & Security
                    PrivacySecurityPage()
                        .tag(9)

                    // Page 11: Profile Setup
                    ProfileSetupPage(viewModel: viewModel)
                        .tag(10)

                    // Page 12: Completion
                    OnboardingCompletionPage(
                        viewModel: viewModel,
                        onComplete: {
                            hasCompletedOnboarding = true
                            dismiss()
                        }
                    )
                        .tag(11)
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                .animation(.easeInOut(duration: 0.3), value: viewModel.currentPage)

                // Navigation Bar
                if viewModel.currentPage < viewModel.totalPages - 1 {
                    OnboardingNavigationBar(
                        currentPage: $viewModel.currentPage,
                        canGoBack: viewModel.currentPage > 0,
                        nextButtonText: viewModel.nextButtonText
                    )
                }
            }
        }
    }
}

// MARK: - Progress Indicator

struct OnboardingProgressIndicator: View {
    let currentPage: Int
    let totalPages: Int

    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<totalPages, id: \.self) { index in
                if index == currentPage {
                    // Current page - elongated pill
                    Capsule()
                        .fill(OnboardingDesign.primaryColor)
                        .frame(width: 24, height: 8)
                } else {
                    // Other pages - dots
                    Circle()
                        .fill(
                            index < currentPage
                                ? OnboardingDesign.primaryColor.opacity(0.5)
                                : OnboardingDesign.primaryColor.opacity(0.2)
                        )
                        .frame(width: 8, height: 8)
                }
            }
        }
        .animation(.easeInOut(duration: 0.2), value: currentPage)
    }
}

// MARK: - Navigation Bar

struct OnboardingNavigationBar: View {
    @Binding var currentPage: Int
    let canGoBack: Bool
    let nextButtonText: String

    var body: some View {
        HStack {
            // Back Button
            if canGoBack {
                Button {
                    withAnimation { currentPage -= 1 }
                } label: {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 56, height: 56)
                        .shadow(color: .black.opacity(0.08), radius: 8, y: 2)
                        .overlay(
                            Image(systemName: "arrow.left")
                                .font(.system(size: 18, weight: .medium))
                                .foregroundColor(OnboardingDesign.textPrimary)
                        )
                }
            } else {
                Spacer().frame(width: 56)
            }

            Spacer()

            // Next Button
            Button {
                withAnimation { currentPage += 1 }
            } label: {
                HStack(spacing: 8) {
                    Text(nextButtonText)
                        .font(.system(size: 17, weight: .semibold))
                    Image(systemName: "arrow.right")
                        .font(.system(size: 15, weight: .semibold))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 32)
                .padding(.vertical, 18)
                .background(
                    Capsule()
                        .fill(OnboardingDesign.primaryColor)
                )
                .shadow(color: OnboardingDesign.primaryColor.opacity(0.3), radius: 12, y: 4)
            }
        }
        .padding(.horizontal, OnboardingDesign.horizontalPadding)
        .padding(.vertical, 16)
        .background(
            OnboardingDesign.backgroundColor
                .shadow(color: .black.opacity(0.05), radius: 10, y: -5)
        )
    }
}

// MARK: - Reusable Page Layout

struct OnboardingPageLayout<Content: View>: View {
    let dinoImage: String
    let title: String
    let subtitle: String?
    let content: Content

    init(
        dinoImage: String,
        title: String,
        subtitle: String? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.dinoImage = dinoImage
        self.title = title
        self.subtitle = subtitle
        self.content = content()
    }

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 24) {
                // Dino Illustration
                Image(dinoImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 200)
                    .padding(.top, 20)

                // Title & Subtitle
                VStack(spacing: 8) {
                    Text(title)
                        .font(OnboardingDesign.titleFont)
                        .foregroundColor(OnboardingDesign.textPrimary)
                        .multilineTextAlignment(.center)

                    if let subtitle = subtitle {
                        Text(subtitle)
                            .font(OnboardingDesign.subtitleFont)
                            .foregroundColor(OnboardingDesign.textSecondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 8)
                    }
                }
                .padding(.horizontal, OnboardingDesign.horizontalPadding)

                // Custom Content
                content
                    .padding(.horizontal, OnboardingDesign.horizontalPadding)

                Spacer(minLength: 100)
            }
        }
    }
}

// MARK: - Selection Card (Single Select)

struct OnboardingSelectionCard: View {
    let icon: String
    let title: String
    let subtitle: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                // Icon
                Image(systemName: icon)
                    .font(.system(size: 22))
                    .foregroundColor(isSelected ? OnboardingDesign.primaryColor : OnboardingDesign.textSecondary)
                    .frame(width: 32)

                // Text
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(OnboardingDesign.labelFont)
                        .foregroundColor(OnboardingDesign.textPrimary)

                    Text(subtitle)
                        .font(OnboardingDesign.captionFont)
                        .foregroundColor(OnboardingDesign.textSecondary)
                }

                Spacer()

                // Checkmark
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 24))
                        .foregroundColor(OnboardingDesign.primaryColor)
                }
            }
            .padding(16)
            .background(OnboardingDesign.cardBackground)
            .cornerRadius(OnboardingDesign.cardCornerRadius)
            .overlay(
                RoundedRectangle(cornerRadius: OnboardingDesign.cardCornerRadius)
                    .stroke(
                        isSelected ? OnboardingDesign.primaryColor : Color.clear,
                        lineWidth: 2
                    )
            )
            .shadow(color: .black.opacity(0.04), radius: 8, y: 2)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Toggle Row Card

struct OnboardingToggleCard: View {
    let icon: String
    let iconColor: Color
    let title: String
    let subtitle: String?
    @Binding var isOn: Bool

    var body: some View {
        HStack(spacing: 14) {
            // Icon
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundColor(iconColor)
                .frame(width: 28)

            // Text
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(OnboardingDesign.labelFont)
                    .foregroundColor(OnboardingDesign.textPrimary)

                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(OnboardingDesign.captionFont)
                        .foregroundColor(OnboardingDesign.textSecondary)
                }
            }

            Spacer()

            // Toggle
            Toggle("", isOn: $isOn)
                .labelsHidden()
                .tint(OnboardingDesign.toggleOn)
        }
        .padding(16)
    }
}

// MARK: - Toggle List Card

struct OnboardingToggleListCard<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        VStack(spacing: 0) {
            content
        }
        .background(OnboardingDesign.cardBackground)
        .cornerRadius(OnboardingDesign.cardCornerRadius)
        .shadow(color: .black.opacity(0.04), radius: 8, y: 2)
    }
}

// MARK: - Input Field Card

struct OnboardingInputCard: View {
    let label: String
    let value: String
    let icon: String
    let action: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(OnboardingDesign.captionFont)
                .foregroundColor(OnboardingDesign.textSecondary)

            Button(action: action) {
                HStack {
                    Text(value)
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(OnboardingDesign.primaryColor)

                    Spacer()

                    Image(systemName: icon)
                        .font(.system(size: 16))
                        .foregroundColor(OnboardingDesign.textSecondary.opacity(0.5))
                }
                .padding(16)
                .background(OnboardingDesign.cardBackground)
                .cornerRadius(OnboardingDesign.cardCornerRadius)
                .shadow(color: .black.opacity(0.04), radius: 8, y: 2)
            }
            .buttonStyle(.plain)
        }
    }
}

// MARK: - Large Value Display Card

struct OnboardingLargeValueCard: View {
    let value: String
    let hint: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 12) {
                Text(value)
                    .font(.system(size: 48, weight: .bold, design: .rounded))
                    .foregroundColor(OnboardingDesign.primaryColor)

                HStack(spacing: 4) {
                    Text(hint)
                        .font(OnboardingDesign.captionFont)
                        .foregroundColor(OnboardingDesign.textSecondary.opacity(0.6))

                    Image(systemName: "pencil")
                        .font(.system(size: 12))
                        .foregroundColor(OnboardingDesign.textSecondary.opacity(0.4))
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 32)
            .background(OnboardingDesign.cardBackground)
            .cornerRadius(OnboardingDesign.cardCornerRadius)
            .shadow(color: .black.opacity(0.04), radius: 8, y: 2)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Privacy Badge

struct OnboardingPrivacyBadge: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "lock.shield.fill")
                .font(.system(size: 14))
                .foregroundColor(OnboardingDesign.primaryColor)

            Text(text)
                .font(OnboardingDesign.captionFont)
                .foregroundColor(OnboardingDesign.textSecondary)
        }
    }
}

// MARK: - Feature Highlight Card

struct OnboardingFeatureCard: View {
    let icon: String
    let iconColor: Color
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            // Icon Circle
            Circle()
                .fill(iconColor.opacity(0.15))
                .frame(width: 44, height: 44)
                .overlay(
                    Image(systemName: icon)
                        .font(.system(size: 20))
                        .foregroundColor(iconColor)
                )

            // Text
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(OnboardingDesign.labelFont)
                    .foregroundColor(OnboardingDesign.textPrimary)

                Text(description)
                    .font(OnboardingDesign.bodyFont)
                    .foregroundColor(OnboardingDesign.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(16)
        .background(OnboardingDesign.cardBackground)
        .cornerRadius(OnboardingDesign.cardCornerRadius)
        .shadow(color: .black.opacity(0.04), radius: 8, y: 2)
    }
}

// MARK: - Segmented Control

struct OnboardingSegmentedControl: View {
    let options: [String]
    @Binding var selectedIndex: Int

    var body: some View {
        HStack(spacing: 0) {
            ForEach(options.indices, id: \.self) { index in
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        selectedIndex = index
                    }
                } label: {
                    Text(options[index])
                        .font(.system(size: 15, weight: selectedIndex == index ? .semibold : .regular))
                        .foregroundColor(selectedIndex == index ? OnboardingDesign.textPrimary : OnboardingDesign.textSecondary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(
                            selectedIndex == index
                                ? OnboardingDesign.cardBackground
                                : Color.clear
                        )
                        .cornerRadius(10)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(4)
        .background(OnboardingDesign.backgroundColor)
        .cornerRadius(12)
    }
}

// MARK: - ViewModel

class NewOnboardingViewModel: ObservableObject {
    @Published var currentPage = 0
    let totalPages = 12

    // Profile Data
    @Published var selectedGender: Int = 0 // 0: Male, 1: Female, 2: Other
    @Published var heightCm: Double = 170
    @Published var weightKg: Double = 70
    @Published var birthDate = Calendar.current.date(byAdding: .year, value: -30, to: Date()) ?? Date()
    @Published var unitSystem: Int = 1 // 0: Imperial, 1: Metric

    // Permissions
    @Published var healthKitEnabled = true
    @Published var notificationsEnabled = true
    @Published var syncSteps = true
    @Published var syncSleep = true
    @Published var syncHRV = true
    @Published var syncHeartRate = true

    // Medications
    @Published var hasMedications = false

    var nextButtonText: String {
        switch currentPage {
        case 0: return "Get Started"
        case 10: return "Almost Done"
        default: return "Next"
        }
    }

    var age: Int {
        Calendar.current.dateComponents([.year], from: birthDate, to: Date()).year ?? 0
    }

    var heightDisplay: String {
        if unitSystem == 0 {
            let totalInches = heightCm / 2.54
            let feet = Int(totalInches / 12)
            let inches = Int(totalInches.truncatingRemainder(dividingBy: 12))
            return "\(feet)' \(inches)\""
        } else {
            return "\(Int(heightCm)) cm"
        }
    }

    var weightDisplay: String {
        if unitSystem == 0 {
            let lbs = weightKg * 2.205
            return "\(Int(lbs)) lbs"
        } else {
            return "\(Int(weightKg)) kg"
        }
    }

    func saveProfile() {
        // Save to UserDefaults or Core Data
        UserDefaults.standard.set(selectedGender, forKey: "userGender")
        UserDefaults.standard.set(heightCm, forKey: "userHeight")
        UserDefaults.standard.set(weightKg, forKey: "userWeight")
        UserDefaults.standard.set(birthDate, forKey: "userBirthDate")
    }

    func requestHealthKitPermission() {
        guard HKHealthStore.isHealthDataAvailable() else { return }

        let healthStore = HKHealthStore()
        let types: Set<HKSampleType> = [
            HKQuantityType(.stepCount),
            HKQuantityType(.heartRateVariabilitySDNN),
            HKQuantityType(.restingHeartRate),
            HKCategoryType(.sleepAnalysis)
        ]

        healthStore.requestAuthorization(toShare: nil, read: types) { _, _ in }
    }

    func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { _, _ in }
    }
}

// MARK: - Page 1: Welcome

struct WelcomePage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-happy 1",
            title: "Welcome to InflamAI",
            subtitle: "Your personal companion for documenting your Ankylosing Spondylitis journey."
        ) {
            VStack(spacing: 16) {
                OnboardingFeatureCard(
                    icon: "chart.line.uptrend.xyaxis",
                    iconColor: .blue,
                    title: "Document Your Data",
                    description: "BASDAI scores, symptoms, and daily factors"
                )

                OnboardingFeatureCard(
                    icon: "brain.head.profile",
                    iconColor: .purple,
                    title: "Discover Patterns",
                    description: "View correlations in your logged data"
                )

                OnboardingFeatureCard(
                    icon: "heart.text.square",
                    iconColor: .red,
                    title: "Support Your Wellbeing",
                    description: "Mobility exercises and wellness suggestions"
                )

                // Medical Disclaimer - Compliance Requirement
                HStack(spacing: 8) {
                    Image(systemName: "info.circle.fill")
                        .foregroundColor(Colors.Gray.g400)
                        .font(.caption)
                    Text("InflamAI is not a medical device and does not replace professional medical advice, diagnosis, or treatment.")
                        .font(.caption2)
                        .foregroundColor(Colors.Gray.g500)
                        .multilineTextAlignment(.leading)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Colors.Gray.g100)
                .cornerRadius(8)
                .padding(.top, 8)
            }
        }
    }
}

// MARK: - Page 2: Understanding AS

struct UnderstandingASPage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-spine-showing",
            title: "Understanding AS",
            subtitle: "Ankylosing Spondylitis affects your spine, but knowledge is your superpower."
        ) {
            VStack(spacing: 16) {
                OnboardingFeatureCard(
                    icon: "info.circle.fill",
                    iconColor: .blue,
                    title: "What is AS?",
                    description: "Inflammatory arthritis primarily affecting the spine and sacroiliac joints"
                )

                OnboardingFeatureCard(
                    icon: "arrow.triangle.2.circlepath",
                    iconColor: .orange,
                    title: "Flares & Remission",
                    description: "AS cycles between active periods and calmer times"
                )

                OnboardingFeatureCard(
                    icon: "heart.fill",
                    iconColor: .green,
                    title: "Great News",
                    description: "With proper tracking and management, most people live full, active lives!"
                )
            }
        }
    }
}

// MARK: - Page 3: Daily Check-In

struct DailyCheckInPage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-showing-whiteboard",
            title: "Daily Check-In",
            subtitle: "Just 60 seconds a day to understand your health better."
        ) {
            VStack(spacing: 16) {
                // BASDAI explanation card
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 24))
                            .foregroundColor(OnboardingDesign.primaryColor)

                        Text("BASDAI Score")
                            .font(OnboardingDesign.labelFont)
                            .foregroundColor(OnboardingDesign.textPrimary)
                    }

                    Text("Answer 6 quick questions to log how you feel - a standardized questionnaire also used in clinical settings.")
                        .font(OnboardingDesign.bodyFont)
                        .foregroundColor(OnboardingDesign.textSecondary)

                    // Score scale
                    HStack(spacing: 8) {
                        ForEach(["Low", "Moderate", "High"], id: \.self) { level in
                            Text(level)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.white)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(
                                    level == "Low" ? Color.green :
                                    level == "Moderate" ? Color.orange : Color.red
                                )
                                .cornerRadius(8)
                        }
                    }
                }
                .padding(16)
                .background(OnboardingDesign.cardBackground)
                .cornerRadius(OnboardingDesign.cardCornerRadius)
                .shadow(color: .black.opacity(0.04), radius: 8, y: 2)

                OnboardingFeatureCard(
                    icon: "clock.fill",
                    iconColor: .blue,
                    title: "Takes < 1 Minute",
                    description: "Quick, simple sliders - no typing required"
                )
            }
        }
    }
}

// MARK: - Page 4: Body Map Tour

struct BodyMapTourPage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-showing-whiteboard",
            title: "Interactive Body Map",
            subtitle: "Tap any of 47 body regions to log exactly where you feel pain."
        ) {
            VStack(spacing: 16) {
                OnboardingFeatureCard(
                    icon: "hand.tap.fill",
                    iconColor: .purple,
                    title: "Tap to Log",
                    description: "Simply tap spine regions, joints, or muscle groups"
                )

                OnboardingFeatureCard(
                    icon: "paintpalette.fill",
                    iconColor: .orange,
                    title: "Visual Heatmap",
                    description: "See your pain patterns over 7, 30, or 90 days"
                )

                OnboardingFeatureCard(
                    icon: "accessibility",
                    iconColor: .blue,
                    title: "47 Regions",
                    description: "Cervical to lumbar spine, SI joints, and all major joints"
                )
            }
        }
    }
}

// MARK: - Page 5: Medication Setup

struct MedicationSetupPage: View {
    @ObservedObject var viewModel: NewOnboardingViewModel

    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-medications",
            title: "Medication Tracking",
            subtitle: "Never miss a dose. Track adherence and document your symptom journey."
        ) {
            VStack(spacing: 16) {
                OnboardingFeatureCard(
                    icon: "bell.fill",
                    iconColor: .orange,
                    title: "Smart Reminders",
                    description: "Customizable alerts for each medication"
                )

                OnboardingFeatureCard(
                    icon: "chart.bar.fill",
                    iconColor: .green,
                    title: "Adherence Tracking",
                    description: "See your medication consistency over time"
                )

                OnboardingFeatureCard(
                    icon: "arrow.triangle.merge",
                    iconColor: .purple,
                    title: "View Patterns",
                    description: "See your medication logs alongside your BASDAI entries"
                )

                // Optional: Add medications now
                VStack(alignment: .leading, spacing: 8) {
                    Text("Do you take medications for AS?")
                        .font(OnboardingDesign.bodyFont)
                        .foregroundColor(OnboardingDesign.textSecondary)

                    HStack(spacing: 12) {
                        OnboardingPillButton(
                            title: "Yes, I do",
                            isSelected: viewModel.hasMedications
                        ) {
                            viewModel.hasMedications = true
                        }

                        OnboardingPillButton(
                            title: "Not yet",
                            isSelected: !viewModel.hasMedications
                        ) {
                            viewModel.hasMedications = false
                        }
                    }
                }
                .padding(.top, 8)
            }
        }
    }
}

struct OnboardingPillButton: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 15, weight: isSelected ? .semibold : .regular))
                .foregroundColor(isSelected ? .white : OnboardingDesign.textPrimary)
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(
                    isSelected
                        ? OnboardingDesign.primaryColor
                        : OnboardingDesign.cardBackground
                )
                .cornerRadius(20)
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(
                            isSelected ? Color.clear : OnboardingDesign.textSecondary.opacity(0.2),
                            lineWidth: 1
                        )
                )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Page 6: Exercise Discovery

struct ExerciseDiscoveryPage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-walking-fast",
            title: "52 AS-Specific Exercises",
            subtitle: "Stretching, strengthening, and mobility routines designed for AS management."
        ) {
            VStack(spacing: 16) {
                // Exercise categories
                HStack(spacing: 12) {
                    NewOnboardingExerciseCategoryCard(icon: "figure.flexibility", title: "Stretch", color: .blue)
                    NewOnboardingExerciseCategoryCard(icon: "dumbbell.fill", title: "Strength", color: .orange)
                    NewOnboardingExerciseCategoryCard(icon: "figure.walk", title: "Mobility", color: .green)
                    NewOnboardingExerciseCategoryCard(icon: "figure.stand", title: "Posture", color: .purple)
                }

                OnboardingFeatureCard(
                    icon: "wand.and.stars",
                    iconColor: .purple,
                    title: "Movement Suggestions",
                    description: "Routine suggestions based on your preferences and goals"
                )

                OnboardingFeatureCard(
                    icon: "play.rectangle.fill",
                    iconColor: .red,
                    title: "Video Guidance",
                    description: "Follow along with clear video demonstrations"
                )
            }
        }
    }
}

private struct NewOnboardingExerciseCategoryCard: View {
    let icon: String
    let title: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Circle()
                .fill(color.opacity(0.15))
                .frame(width: 50, height: 50)
                .overlay(
                    Image(systemName: icon)
                        .font(.system(size: 22))
                        .foregroundColor(color)
                )

            Text(title)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(OnboardingDesign.textPrimary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(OnboardingDesign.cardBackground)
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.04), radius: 6, y: 2)
    }
}

// MARK: - Page 7: Flare Tracking

struct FlareTrackingPage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-sad",
            title: "Flare Support",
            subtitle: "When flares hit, we're here to help you log quickly and find patterns."
        ) {
            VStack(spacing: 16) {
                // SOS Button preview
                VStack(spacing: 12) {
                    Circle()
                        .fill(Color.red.opacity(0.15))
                        .frame(width: 80, height: 80)
                        .overlay(
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 36))
                                .foregroundColor(.red)
                        )

                    Text("JointTap SOS")
                        .font(OnboardingDesign.labelFont)
                        .foregroundColor(OnboardingDesign.textPrimary)

                    Text("3 taps to log during a flare - no typing needed")
                        .font(OnboardingDesign.captionFont)
                        .foregroundColor(OnboardingDesign.textSecondary)
                        .multilineTextAlignment(.center)
                }
                .padding(20)
                .frame(maxWidth: .infinity)
                .background(OnboardingDesign.cardBackground)
                .cornerRadius(OnboardingDesign.cardCornerRadius)
                .shadow(color: .black.opacity(0.04), radius: 8, y: 2)

                OnboardingFeatureCard(
                    icon: "chart.line.downtrend.xyaxis",
                    iconColor: .orange,
                    title: "Flare Timeline",
                    description: "Track duration, intensity, and recovery patterns"
                )

                OnboardingFeatureCard(
                    icon: "magnifyingglass",
                    iconColor: .purple,
                    title: "Review Your Logs",
                    description: "Look back at what you logged before flare events"
                )
            }
        }
    }
}

// MARK: - Page 8: AI & Weather Intelligence

struct AIWeatherPage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-stading-normal",
            title: "Data Visualization",
            subtitle: "View patterns and correlations in your logged data."
        ) {
            VStack(spacing: 16) {
                OnboardingFeatureCard(
                    icon: "brain",
                    iconColor: .purple,
                    title: "Pattern Visualization",
                    description: "See correlations across your logged data points"
                )

                OnboardingFeatureCard(
                    icon: "cloud.sun.fill",
                    iconColor: .blue,
                    title: "Weather Logging",
                    description: "Track weather conditions alongside your symptoms"
                )

                OnboardingFeatureCard(
                    icon: "bell.badge.fill",
                    iconColor: .orange,
                    title: "Daily Reminders",
                    description: "Stay consistent with logging reminders"
                )

                OnboardingPrivacyBadge(text: "All data stays on your device")

                // Compliance disclaimer
                HStack(spacing: 6) {
                    Image(systemName: "info.circle")
                        .foregroundColor(Colors.Gray.g400)
                        .font(.caption2)
                    Text("Pattern visualization is not medical advice. Discuss observations with your doctor.")
                        .font(.caption2)
                        .foregroundColor(Colors.Gray.g500)
                }
                .padding(.top, 4)
            }
        }
    }
}

// MARK: - Page 9: Trends & Reports

struct TrendsReportsPage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-showing-whiteboard",
            title: "Trends & Reports",
            subtitle: "Beautiful charts and exportable PDF reports for your doctor visits."
        ) {
            VStack(spacing: 16) {
                OnboardingFeatureCard(
                    icon: "chart.xyaxis.line",
                    iconColor: .blue,
                    title: "Visual Trends",
                    description: "See your BASDAI, pain, and symptoms over time"
                )

                OnboardingFeatureCard(
                    icon: "doc.richtext",
                    iconColor: .green,
                    title: "PDF Reports",
                    description: "One-tap export for rheumatologist appointments"
                )

                OnboardingFeatureCard(
                    icon: "calendar",
                    iconColor: .orange,
                    title: "Custom Date Ranges",
                    description: "View 7 days, 30 days, 90 days, or custom periods"
                )

                // Sample chart preview
                VStack(spacing: 8) {
                    HStack {
                        Text("Sample BASDAI Trend")
                            .font(OnboardingDesign.captionFont)
                            .foregroundColor(OnboardingDesign.textSecondary)
                        Spacer()
                    }

                    // Simple fake chart
                    HStack(alignment: .bottom, spacing: 4) {
                        ForEach([3.2, 4.1, 3.8, 2.9, 3.5, 2.8, 2.4], id: \.self) { value in
                            RoundedRectangle(cornerRadius: 4)
                                .fill(OnboardingDesign.primaryColor.opacity(0.7))
                                .frame(height: CGFloat(value * 15))
                        }
                    }
                    .frame(height: 70)
                }
                .padding(16)
                .background(OnboardingDesign.cardBackground)
                .cornerRadius(OnboardingDesign.cardCornerRadius)
                .shadow(color: .black.opacity(0.04), radius: 8, y: 2)
            }
        }
    }
}

// MARK: - Page 10: Privacy & Security

struct PrivacySecurityPage: View {
    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-stading-normal",
            title: "Your Privacy Matters",
            subtitle: "Your health data stays on YOUR device. We never sell or share it."
        ) {
            VStack(spacing: 16) {
                OnboardingFeatureCard(
                    icon: "iphone",
                    iconColor: .blue,
                    title: "100% On-Device",
                    description: "All data processing happens locally on your iPhone"
                )

                OnboardingFeatureCard(
                    icon: "xmark.shield.fill",
                    iconColor: .red,
                    title: "Zero Third-Party SDKs",
                    description: "No Facebook, Google Analytics, or tracking libraries"
                )

                OnboardingFeatureCard(
                    icon: "faceid",
                    iconColor: .green,
                    title: "Biometric Lock",
                    description: "Face ID or Touch ID protects your sensitive data"
                )

                OnboardingFeatureCard(
                    icon: "icloud.fill",
                    iconColor: .cyan,
                    title: "Optional iCloud Sync",
                    description: "Your choice - sync across devices or keep local only"
                )

                OnboardingPrivacyBadge(text: "GDPR compliant - delete anytime")
            }
        }
    }
}

// MARK: - Page 11: Profile Setup

struct ProfileSetupPage: View {
    @ObservedObject var viewModel: NewOnboardingViewModel
    @State private var showDatePicker = false
    @State private var showHeightPicker = false
    @State private var showWeightPicker = false

    var body: some View {
        OnboardingPageLayout(
            dinoImage: "dino-walking 1",
            title: "About You",
            subtitle: "This helps personalize your experience and improve predictions."
        ) {
            VStack(spacing: 20) {
                // Unit System Toggle
                OnboardingSegmentedControl(
                    options: ["Imperial", "Metric"],
                    selectedIndex: $viewModel.unitSystem
                )

                // Gender Selection
                VStack(alignment: .leading, spacing: 8) {
                    Text("Gender")
                        .font(OnboardingDesign.captionFont)
                        .foregroundColor(OnboardingDesign.textSecondary)

                    HStack(spacing: 12) {
                        ForEach(["Male", "Female", "Other"].indices, id: \.self) { index in
                            OnboardingPillButton(
                                title: ["Male", "Female", "Other"][index],
                                isSelected: viewModel.selectedGender == index
                            ) {
                                viewModel.selectedGender = index
                            }
                        }
                    }
                }

                // Birthday
                OnboardingInputCard(
                    label: "Birthday",
                    value: formatDate(viewModel.birthDate) + " (\(viewModel.age) years)",
                    icon: "calendar"
                ) {
                    showDatePicker = true
                }

                // Height
                OnboardingInputCard(
                    label: "Height",
                    value: viewModel.heightDisplay,
                    icon: "pencil"
                ) {
                    showHeightPicker = true
                }

                // Weight
                OnboardingInputCard(
                    label: "Weight",
                    value: viewModel.weightDisplay,
                    icon: "pencil"
                ) {
                    showWeightPicker = true
                }

                OnboardingPrivacyBadge(text: "Used for health calculations only")
            }
        }
        .sheet(isPresented: $showDatePicker) {
            DatePickerSheet(date: $viewModel.birthDate)
        }
        .sheet(isPresented: $showHeightPicker) {
            HeightPickerSheet(heightCm: $viewModel.heightCm, isMetric: viewModel.unitSystem == 1)
        }
        .sheet(isPresented: $showWeightPicker) {
            WeightPickerSheet(weightKg: $viewModel.weightKg, isMetric: viewModel.unitSystem == 1)
        }
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
}

// MARK: - Picker Sheets

struct DatePickerSheet: View {
    @Binding var date: Date
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            DatePicker("Birthday", selection: $date, displayedComponents: .date)
                .datePickerStyle(.wheel)
                .labelsHidden()
                .padding()
                .navigationTitle("Select Birthday")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
        .presentationDetents([.medium])
    }
}

struct HeightPickerSheet: View {
    @Binding var heightCm: Double
    let isMetric: Bool
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            VStack {
                if isMetric {
                    Picker("Height", selection: $heightCm) {
                        ForEach(120...220, id: \.self) { cm in
                            Text("\(cm) cm").tag(Double(cm))
                        }
                    }
                    .pickerStyle(.wheel)
                } else {
                    // Imperial picker would need feet + inches
                    Picker("Height", selection: $heightCm) {
                        ForEach(120...220, id: \.self) { cm in
                            let totalInches = Double(cm) / 2.54
                            let feet = Int(totalInches / 12)
                            let inches = Int(totalInches.truncatingRemainder(dividingBy: 12))
                            Text("\(feet)' \(inches)\"").tag(Double(cm))
                        }
                    }
                    .pickerStyle(.wheel)
                }
            }
            .navigationTitle("Select Height")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium])
    }
}

struct WeightPickerSheet: View {
    @Binding var weightKg: Double
    let isMetric: Bool
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            Picker("Weight", selection: $weightKg) {
                if isMetric {
                    ForEach(30...200, id: \.self) { kg in
                        Text("\(kg) kg").tag(Double(kg))
                    }
                } else {
                    ForEach(66...440, id: \.self) { lbs in
                        Text("\(lbs) lbs").tag(Double(lbs) / 2.205)
                    }
                }
            }
            .pickerStyle(.wheel)
            .navigationTitle("Select Weight")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium])
    }
}

// MARK: - Page 12: Completion

struct OnboardingCompletionPage: View {
    @ObservedObject var viewModel: NewOnboardingViewModel
    let onComplete: () -> Void

    @State private var showConfetti = false

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            // Completion Icon (Sleeping Dino animation)
            #if os(iOS)
            LottieView.loop("sleeping-dino", speed: 0.8)
                .frame(height: 240)
                .scaleEffect(showConfetti ? 1.0 : 0.9)
                .opacity(showConfetti ? 1.0 : 0.8)
                .animation(
                    .easeInOut(duration: 1.0),
                    value: showConfetti
                )
            #else
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 120))
                .foregroundColor(OnboardingDesign.accentColor)
                .frame(height: 240)
                .scaleEffect(showConfetti ? 1.0 : 0.9)
                .opacity(showConfetti ? 1.0 : 0.8)
                .animation(
                    .easeInOut(duration: 1.0),
                    value: showConfetti
                )
            #endif

            VStack(spacing: 12) {
                Text("You're All Set!")
                    .font(.system(size: 32, weight: .bold, design: .rounded))
                    .foregroundColor(OnboardingDesign.textPrimary)

                Text("Rest well, track smart. Your journey to better AS management starts now.")
                    .font(OnboardingDesign.subtitleFont)
                    .foregroundColor(OnboardingDesign.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }

            // Quick stats
            HStack(spacing: 20) {
                CompletionStatBadge(icon: "chart.xyaxis.line", value: "BASDAI", label: "Tracking")
                CompletionStatBadge(icon: "figure.flexibility", value: "52", label: "Exercises")
                CompletionStatBadge(icon: "brain", value: "AI", label: "Insights")
            }
            .padding(.vertical, 20)

            Spacer()

            // Complete Button
            Button(action: {
                viewModel.saveProfile()
                onComplete()
            }) {
                HStack(spacing: 8) {
                    Text("Start Using InflamAI")
                        .font(.system(size: 18, weight: .semibold))
                    Image(systemName: "arrow.right")
                        .font(.system(size: 16, weight: .semibold))
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 18)
                .background(
                    LinearGradient(
                        colors: [OnboardingDesign.primaryColor, OnboardingDesign.accentColor],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .cornerRadius(OnboardingDesign.buttonCornerRadius)
                .shadow(color: OnboardingDesign.primaryColor.opacity(0.3), radius: 12, y: 4)
            }
            .padding(.horizontal, OnboardingDesign.horizontalPadding)
            .padding(.bottom, 32)
        }
        .background(OnboardingDesign.backgroundColor)
        .onAppear {
            showConfetti = true
        }
    }
}

struct CompletionStatBadge: View {
    let icon: String
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 8) {
            Circle()
                .fill(OnboardingDesign.primaryColor.opacity(0.15))
                .frame(width: 56, height: 56)
                .overlay(
                    Image(systemName: icon)
                        .font(.system(size: 24))
                        .foregroundColor(OnboardingDesign.primaryColor)
                )

            Text(value)
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(OnboardingDesign.textPrimary)

            Text(label)
                .font(.system(size: 12))
                .foregroundColor(OnboardingDesign.textSecondary)
        }
    }
}

// MARK: - Preview

#Preview("Full Onboarding") {
    NewOnboardingFlow()
}

#Preview("Welcome Page") {
    WelcomePage()
        .background(OnboardingDesign.backgroundColor)
}

#Preview("Profile Setup") {
    ProfileSetupPage(viewModel: NewOnboardingViewModel())
        .background(OnboardingDesign.backgroundColor)
}
