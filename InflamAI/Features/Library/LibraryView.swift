//
//  LibraryView.swift
//  InflamAI
//
//  Educational content about circadian rhythms and AS symptom patterns
//

import SwiftUI
#if canImport(Lottie)
import Lottie
#endif

struct LibraryView: View {
    @StateObject private var viewModel = LibraryViewModel()
    @State private var selectedSection: LibrarySection = .sleep

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is now presented via NavigationLink from MoreView,
        // which is already wrapped in NavigationView in MainTabView.
        ScrollView {
            VStack(spacing: 24) {
                    // Header
                    VStack(spacing: 8) {
                        Image(systemName: "book.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.blue)

                        Text("AS Knowledge Library")
                            .font(.title)
                            .fontWeight(.bold)

                        Text("Understanding how your body's rhythm affects AS symptoms")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding(.top, 20)

                    // Section Picker
                    Picker("Time Period", selection: $selectedSection) {
                        ForEach(LibrarySection.allCases) { section in
                            Label(section.title, systemImage: section.icon)
                                .tag(section)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)

                    // Content Card
                    VStack(alignment: .leading, spacing: 16) {
                        // Section-specific content
                        switch selectedSection {
                        case .sleep:
                            SleepSectionView()
                        case .morning:
                            MorningSectionView()
                        case .afternoon:
                            AfternoonSectionView()
                        case .evening:
                            EveningNightSectionView()
                        }
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color(.systemBackground))
                            .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 4)
                    )
                    .padding(.horizontal)

                // Medical Disclaimer
                DisclaimerView()
                    .padding(.horizontal)
                    .padding(.bottom, 20)
            }
        }
        .navigationTitle("Library")
        .navigationBarTitleDisplayMode(.large)
    }
}

// CRIT-001: NavigationView removed - this view is now accessed via NavigationLink

// MARK: - Sleep Section with Lottie Animation

struct SleepSectionView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header with Icon
            HStack {
                Image(systemName: "moon.stars.fill")
                    .font(.title2)
                    .foregroundColor(.indigo)
                Text("Sleep & AS")
                    .font(.title2)
                    .fontWeight(.bold)
            }

            // Sleep Animation
            #if os(iOS)
            LottieView.loop("sleeping-dino")
                .frame(height: 200)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.indigo.opacity(0.1))
                )
            #endif

            // Key Facts
            InfoCard(
                title: "Sleep Disorders in AS",
                icon: "exclamationmark.triangle.fill",
                color: .orange,
                content: "50-64.5% of AS patients experience sleep disorders due to inflammatory back pain, especially in the second half of the night."
            )

            InfoCard(
                title: "Why Night Pain Occurs",
                icon: "clock.fill",
                color: .purple,
                content: "Cortisol levels drop during the evening and night, leading to increased inflammation. This causes awakening due to back pain, typically during the second half of sleep."
            )

            InfoCard(
                title: "Morning Stiffness Connection",
                icon: "figure.walk",
                color: .blue,
                content: "Morning stiffness reflects the impact of lack of movement during sleep. This rigidity compromises deep sleep quality, which is essential for physical and mental recovery."
            )

            // Actionable Tips
            VStack(alignment: .leading, spacing: 12) {
                Label("Tips for Better Sleep", systemImage: "lightbulb.fill")
                    .font(.headline)
                    .foregroundColor(.yellow)

                TipRow(tip: "Maintain consistent sleep/wake times - even on weekends")
                TipRow(tip: "Consider a supportive mattress and pillows")
                TipRow(tip: "Gentle stretching before bed may reduce stiffness")
                TipRow(tip: "Discuss chronotherapy options with your rheumatologist")
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.yellow.opacity(0.1))
            )

            // Sources
            SourcesView(sources: [
                "Sleep disturbance prevalence: 50-64.5%",
                "Circadian symptoms are diagnostic criteria for AS",
                "Melatonin levels correlate with BASDAI scores"
            ])
        }
    }
}

// MARK: - Morning Section

struct MorningSectionView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "sunrise.fill")
                    .font(.title2)
                    .foregroundColor(.orange)
                Text("Morning Patterns")
                    .font(.title2)
                    .fontWeight(.bold)
            }

            InfoCard(
                title: "Peak Inflammation Time",
                icon: "chart.line.uptrend.xyaxis",
                color: .red,
                content: "Pain and stiffness are 2-3 times higher between 6:00-9:00 AM compared to noon. This is a diagnostic criterion for inflammatory back pain."
            )

            InfoCard(
                title: "TNF-alpha & IL-6 Surge",
                icon: "waveform.path.ecg",
                color: .pink,
                content: "Pro-inflammatory cytokines TNF-alpha and IL-6 peak in early morning hours. These are the same molecules targeted by biologic medications."
            )

            InfoCard(
                title: "The 30-Minute Rule",
                icon: "timer",
                color: .orange,
                content: "Morning stiffness lasting >30 minutes is one of the four classification criteria for inflammatory back pain and suggests active AS."
            )

            VStack(alignment: .leading, spacing: 12) {
                Label("Morning Management", systemImage: "sun.max.fill")
                    .font(.headline)
                    .foregroundColor(.orange)

                TipRow(tip: "Hot shower or bath to ease stiffness")
                TipRow(tip: "Gentle morning stretches (5-10 minutes)")
                TipRow(tip: "NSAIDs at night may reduce morning symptoms")
                TipRow(tip: "Track BASDAI scores - includes morning stiffness duration")
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.orange.opacity(0.1))
            )

            SourcesView(sources: [
                "Morning stiffness >30 min = diagnostic criterion",
                "TNF-alpha peaks in early morning",
                "Pain intensity 2-3x higher before 9 AM"
            ])
        }
    }
}

// MARK: - Afternoon Section

struct AfternoonSectionView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "sun.max.fill")
                    .font(.title2)
                    .foregroundColor(.yellow)
                Text("Afternoon Relief")
                    .font(.title2)
                    .fontWeight(.bold)
            }

            InfoCard(
                title: "Natural Symptom Relief",
                icon: "arrow.down.circle.fill",
                color: .green,
                content: "Symptoms typically improve between noon and 3:00 PM. Pain and stiffness are at their lowest during this window."
            )

            InfoCard(
                title: "Cortisol's Protective Role",
                icon: "shield.fill",
                color: .blue,
                content: "Cortisol levels are higher during the day, providing natural anti-inflammatory effects. This is why symptoms often feel better in the afternoon."
            )

            InfoCard(
                title: "Optimal Activity Window",
                icon: "figure.walk",
                color: .teal,
                content: "Afternoon is often the best time for exercise and physical activity. Your body is more flexible and inflammation is naturally lower."
            )

            VStack(alignment: .leading, spacing: 12) {
                Label("Afternoon Strategies", systemImage: "figure.run")
                    .font(.headline)
                    .foregroundColor(.green)

                TipRow(tip: "Schedule exercise during this low-symptom window")
                TipRow(tip: "Maintain movement to prevent evening stiffness")
                TipRow(tip: "Stay hydrated - supports inflammation regulation")
                TipRow(tip: "Consider light stretching at work/home")
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.green.opacity(0.1))
            )

            SourcesView(sources: [
                "Lowest symptom intensity: 12:00-15:00",
                "TNF-alpha and IL-6 at very low levels after noon",
                "Natural cortisol rhythm provides afternoon relief"
            ])
        }
    }
}

// MARK: - Evening/Night Section

struct EveningNightSectionView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "moon.fill")
                    .font(.title2)
                    .foregroundColor(.indigo)
                Text("Evening & Night")
                    .font(.title2)
                    .fontWeight(.bold)
            }

            InfoCard(
                title: "Secondary Peak",
                icon: "waveform",
                color: .purple,
                content: "A second, less prominent symptom peak occurs between 7:00-9:00 PM. This is when cortisol begins dropping for the night."
            )

            InfoCard(
                title: "Second-Half Night Pain",
                icon: "bed.double.fill",
                color: .indigo,
                content: "Awakening due to back pain during the second half of the night is a diagnostic criterion for inflammatory back pain - it's that specific to AS."
            )

            InfoCard(
                title: "Preparing for Morning",
                icon: "sunrise",
                color: .orange,
                content: "As cortisol drops overnight, inflammation builds. This creates the morning stiffness cycle that defines AS."
            )

            VStack(alignment: .leading, spacing: 12) {
                Label("Evening Best Practices", systemImage: "moon.stars.fill")
                    .font(.headline)
                    .foregroundColor(.indigo)

                TipRow(tip: "Take NSAIDs in evening for morning relief")
                TipRow(tip: "Gentle evening stretches before bed")
                TipRow(tip: "Avoid late-night heavy meals")
                TipRow(tip: "Consider chronotherapy with your doctor")
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.indigo.opacity(0.1))
            )

            SourcesView(sources: [
                "Second symptom peak: 19:00-21:00",
                "Second-half night awakening = diagnostic criterion",
                "Cortisol downregulation → increased inflammation"
            ])
        }
    }
}

// MARK: - Supporting Components

struct InfoCard: View {
    let title: String
    let icon: String
    let color: Color
    let content: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
                .frame(width: 30)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary)

                Text(content)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(color.opacity(0.1))
        )
    }
}

struct TipRow: View {
    let tip: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
                .font(.caption)
            Text(tip)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

struct SourcesView: View {
    let sources: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("📚 Key Research Findings")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)

            ForEach(sources, id: \.self) { source in
                HStack(alignment: .top, spacing: 6) {
                    Text("•")
                        .foregroundColor(.secondary)
                    Text(source)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray6))
        )
    }
}

struct DisclaimerView: View {
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                Text("Medical Disclaimer")
                    .font(.caption)
                    .fontWeight(.semibold)
            }

            Text("This content is educational and based on medical research. Always consult your rheumatologist before making changes to your treatment plan. Individual experiences may vary.")
                .font(.caption2)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.orange.opacity(0.1))
        )
    }
}

// MARK: - Library Section Enum

enum LibrarySection: String, CaseIterable, Identifiable {
    case sleep = "Sleep"
    case morning = "Morning"
    case afternoon = "Afternoon"
    case evening = "Evening"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .sleep: return "Sleep"
        case .morning: return "Morning"
        case .afternoon: return "Afternoon"
        case .evening: return "Evening"
        }
    }

    var icon: String {
        switch self {
        case .sleep: return "moon.stars.fill"
        case .morning: return "sunrise.fill"
        case .afternoon: return "sun.max.fill"
        case .evening: return "moon.fill"
        }
    }
}

// MARK: - Preview

#Preview {
    LibraryView()
}
