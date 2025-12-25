//
//  MLFeaturesView.swift
//  InflamAI
//
//  Displays all 92 ML features used by the Neural Engine
//  Organized by category with descriptions
//

import SwiftUI

struct MLFeaturesView: View {
    @State private var selectedCategory: FeatureCategory = .all

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 16) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 60))
                        .foregroundColor(.purple)

                    Text("92 ML Features")
                        .font(.title)
                        .fontWeight(.bold)

                    Text("Neural Engine Data Sources")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    // Feature count badge
                    HStack(spacing: 16) {
                        FeatureCountBadge(
                            count: 92,
                            label: "Total Features",
                            color: .purple
                        )

                        FeatureCountBadge(
                            count: 9,
                            label: "Categories",
                            color: .blue
                        )
                    }
                }
                .padding(.top, 24)

                // Category Filter
                categoryFilter

                // Feature Categories
                VStack(spacing: 16) {
                    if selectedCategory == .all || selectedCategory == .demographics {
                        FeatureCategoryCard(
                            title: "Demographics",
                            icon: "person.fill",
                            color: .blue,
                            count: 6,
                            features: [
                                "Age", "Gender", "HLA-B27 Status",
                                "Disease Duration", "BMI", "Smoking Status"
                            ]
                        )
                    }

                    if selectedCategory == .all || selectedCategory == .clinical {
                        FeatureCategoryCard(
                            title: "Clinical Assessment",
                            icon: "stethoscope",
                            color: .red,
                            count: 15,
                            features: [
                                "BASDAI Score", "ASDAS-CRP", "BASFI", "BASMI",
                                "Patient Global Assessment", "Physician Global",
                                "Tender Joint Count", "Swollen Joint Count",
                                "Enthesitis Sites", "Dactylitis Presence",
                                "Spinal Mobility", "Disease Activity Composite",
                                "ESR Level", "CRP Level", "ASAS Response"
                            ]
                        )
                    }

                    if selectedCategory == .all || selectedCategory == .pain {
                        FeatureCategoryCard(
                            title: "Pain Metrics",
                            icon: "bandage.fill",
                            color: .orange,
                            count: 12,
                            features: [
                                "Current Pain Level", "24h Average Pain", "24h Peak Pain",
                                "Nocturnal Pain", "Morning Stiffness Duration",
                                "Morning Stiffness Severity", "Pain Location Count",
                                "Burning Pain", "Aching Pain", "Sharp Pain",
                                "Sleep Interference", "Activity Interference"
                            ]
                        )
                    }

                    if selectedCategory == .all || selectedCategory == .biometrics {
                        FeatureCategoryCard(
                            title: "Biometrics & Activity",
                            icon: "heart.fill",
                            color: .pink,
                            count: 20,
                            features: [
                                "Blood Oxygen", "Cardio Fitness (VO2 Max)",
                                "Respiratory Rate", "6-Minute Walk Distance",
                                "Resting Energy", "Heart Rate Variability (HRV)",
                                "Resting Heart Rate", "Walking Heart Rate",
                                "Cardio Recovery", "Daily Steps",
                                "Distance (km)", "Stairs Climbed Up", "Stairs Down",
                                "Stand Hours", "Stand Minutes", "Training Minutes",
                                "Active Minutes", "Active Energy", "Training Sessions",
                                "Walking Tempo", "Step Length", "Gait Asymmetry",
                                "Bipedal Support Time"
                            ]
                        )
                    }

                    if selectedCategory == .all || selectedCategory == .sleep {
                        FeatureCategoryCard(
                            title: "Sleep Quality",
                            icon: "bed.double.fill",
                            color: .indigo,
                            count: 9,
                            features: [
                                "Total Sleep Hours", "REM Sleep Duration",
                                "Deep Sleep Duration", "Core Sleep Duration",
                                "Awake Duration", "Sleep Score",
                                "Sleep Consistency", "Calories Burned",
                                "Exertion Level"
                            ]
                        )
                    }

                    if selectedCategory == .all || selectedCategory == .mental {
                        FeatureCategoryCard(
                            title: "Mental Health",
                            icon: "brain.head.profile",
                            color: .teal,
                            count: 11,
                            features: [
                                "Current Mood", "Mood Valence",
                                "Mood Stability", "Anxiety Level",
                                "Stress Level", "Stress Resilience",
                                "Mental Fatigue", "Cognitive Function",
                                "Emotional Regulation", "Social Engagement",
                                "Mental Wellbeing Score", "Depression Risk"
                            ]
                        )
                    }

                    if selectedCategory == .all || selectedCategory == .environmental {
                        FeatureCategoryCard(
                            title: "Environmental",
                            icon: "cloud.sun.fill",
                            color: .cyan,
                            count: 8,
                            features: [
                                "Daylight Exposure Time", "Ambient Temperature",
                                "Humidity Level", "Barometric Pressure",
                                "12h Pressure Change", "Air Quality Index",
                                "Weather Change Score", "Ambient Noise Level",
                                "Season"
                            ]
                        )
                    }

                    if selectedCategory == .all || selectedCategory == .adherence {
                        FeatureCategoryCard(
                            title: "Treatment Adherence",
                            icon: "pills.fill",
                            color: .green,
                            count: 5,
                            features: [
                                "Medication Adherence Rate",
                                "Physiotherapy Adherence",
                                "Physiotherapy Effectiveness",
                                "Journal Mood Entry",
                                "Quick Log Count"
                            ]
                        )
                    }

                    if selectedCategory == .all || selectedCategory == .assessments {
                        FeatureCategoryCard(
                            title: "Assessments",
                            icon: "doc.text.fill",
                            color: .yellow,
                            count: 3,
                            features: [
                                "Universal Assessment Score",
                                "Time-Weighted Assessment",
                                "Patient-Reported Outcomes"
                            ]
                        )
                    }
                }

                // How It Works
                howItWorksSection

                // Privacy Notice
                privacyNotice
            }
            .padding()
        }
        .navigationTitle("ML Features")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Category Filter

    private var categoryFilter: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(FeatureCategory.allCases, id: \.self) { category in
                    CategoryFilterChip(
                        category: category,
                        isSelected: selectedCategory == category
                    ) {
                        withAnimation {
                            selectedCategory = category
                        }
                    }
                }
            }
            .padding(.horizontal)
        }
    }

    // MARK: - How It Works

    private var howItWorksSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "info.circle.fill")
                    .foregroundColor(.blue)
                Text("How These Features Work")
                    .font(.headline)
                Spacer()
            }

            VStack(alignment: .leading, spacing: 12) {
                MLInfoRow(
                    number: "1",
                    title: "Data Collection",
                    description: "Features are extracted from your symptom logs, HealthKit data, and environmental sensors."
                )

                MLInfoRow(
                    number: "2",
                    title: "30-Day Sequences",
                    description: "The Neural Engine analyzes 30 consecutive days of each feature to identify patterns."
                )

                MLInfoRow(
                    number: "3",
                    title: "Personalization",
                    description: "Over 28 days, the model learns YOUR unique patterns and transitions from synthetic baseline to personalized predictions."
                )

                MLInfoRow(
                    number: "4",
                    title: "Continuous Learning",
                    description: "Weekly automatic updates refine predictions based on your actual outcomes."
                )
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    // MARK: - Privacy Notice

    private var privacyNotice: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "lock.shield.fill")
                    .foregroundColor(.green)
                Text("100% On-Device Processing")
                    .font(.headline)
                Spacer()
            }

            Text("All 92 features are processed locally on your iPhone. No data ever leaves your device. The Neural Engine runs entirely on-device using Apple's Core ML framework.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            Text("⚠️ Not Medical Advice")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.orange)

            Text("These features power statistical pattern analysis. Always consult your rheumatologist for medical decisions.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color(.tertiarySystemBackground))
        .cornerRadius(12)
    }
}

// MARK: - Supporting Views

struct FeatureCountBadge: View {
    let count: Int
    let label: String
    let color: Color

    var body: some View {
        VStack(spacing: 4) {
            Text("\(count)")
                .font(.system(size: 32, weight: .bold))
                .foregroundColor(color)

            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

struct CategoryFilterChip: View {
    let category: FeatureCategory
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(category.displayName)
                .font(.subheadline)
                .fontWeight(isSelected ? .semibold : .regular)
                .foregroundColor(isSelected ? .white : .primary)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(isSelected ? Color.purple : Color(.secondarySystemBackground))
                .cornerRadius(20)
        }
    }
}

struct FeatureCategoryCard: View {
    let title: String
    let icon: String
    let color: Color
    let count: Int
    let features: [String]

    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            Button {
                withAnimation {
                    isExpanded.toggle()
                }
            } label: {
                HStack {
                    HStack(spacing: 12) {
                        ZStack {
                            Circle()
                                .fill(color.opacity(0.2))
                                .frame(width: 44, height: 44)

                            Image(systemName: icon)
                                .foregroundColor(color)
                                .font(.title3)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text(title)
                                .font(.headline)
                                .foregroundColor(.primary)

                            Text("\(count) features")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    Spacer()

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                }
            }

            // Feature List
            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(features, id: \.self) { feature in
                        HStack(spacing: 8) {
                            Circle()
                                .fill(color)
                                .frame(width: 6, height: 6)

                            Text(feature)
                                .font(.subheadline)
                                .foregroundColor(.primary)

                            Spacer()
                        }
                    }
                }
                .padding(.leading, 56)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }
}

private struct MLInfoRow: View {
    let number: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.blue.opacity(0.2))
                    .frame(width: 32, height: 32)

                Text(number)
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundColor(.blue)
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
    }
}

// MARK: - Feature Category Enum

enum FeatureCategory: String, CaseIterable {
    case all = "All"
    case demographics = "Demographics"
    case clinical = "Clinical"
    case pain = "Pain"
    case biometrics = "Biometrics"
    case sleep = "Sleep"
    case mental = "Mental Health"
    case environmental = "Environmental"
    case adherence = "Adherence"
    case assessments = "Assessments"

    var displayName: String {
        rawValue
    }
}

// MARK: - Preview

#Preview {
    NavigationView {
        MLFeaturesView()
    }
}
