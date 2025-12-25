# Meditation Feature Implementation Plan

## Executive Summary

After thorough exploration of the codebase, I've discovered that **InflamAI already has extensive meditation functionality** implemented in three files:
- `InflamAI/Views/MeditationView.swift` (1,969 lines)
- `InflamAI/Modules/MeditationModule.swift` (1,359 lines)
- `InflamAI/MeditationMindfulnessModule.swift` (1,795 lines)

However, this meditation feature appears **disconnected from the main app architecture** and **not integrated with Core Data persistence**.

## Critical Discovery: Existing Implementation

### What Already Exists

The existing meditation code includes:

**✅ Comprehensive UI (MeditationView.swift)**
- 5-tab interface: Discover, Library, Player, Progress, Profile
- Session browsing with categories, search, and filters
- Active player with visualization and controls
- Progress tracking with streaks and achievements
- Download management for offline access
- Favorites and playlists

**✅ Rich Data Models (MeditationModule.swift)**
- 15+ meditation types (mindfulness, breathing, body scanning, pain relief, etc.)
- Session tracking with before/after metrics (mood, stress, pain, energy)
- Heart rate monitoring via HealthKit
- Breathing pattern guidance
- Achievement system
- Insights generation

**✅ Session Management**
- Audio playback with AVFoundation
- Background sounds
- Playback controls (pause, resume, seek, rate adjustment)
- Progress tracking
- Automatic HealthKit logging

### What's Missing

**❌ Core Data Integration**
- No `MeditationSession` entity in Core Data model
- Sessions stored in UserDefaults (not scalable)
- No relationship to `SymptomLog` for correlation analysis

**❌ Integration with Main App**
- Not referenced in navigation or main app flow
- Isolated from symptom tracking system
- No correlation with pain/flare data
- Missing from Features folder structure

**❌ AS-Specific Features**
- No pain relief meditation categories optimized for AS
- No integration with body map (47-region anatomy)
- No correlation with BASDAI scores
- Missing correlation with weather/biometric triggers

## Proposed Implementation Strategy

### Option A: Integrate Existing Implementation (RECOMMENDED)

**Pros:**
- 4,000+ lines of production-ready code
- Comprehensive feature set already built
- Save ~2-3 weeks of development time
- Professional UI/UX already designed

**Cons:**
- Need to refactor for Core Data
- Need to align with app architecture
- Some code duplication between the 3 files needs consolidation

**Steps:**
1. Consolidate the 3 meditation files into unified implementation
2. Add Core Data entities for persistence
3. Migrate to Features/Meditation folder (MVVM pattern)
4. Integrate with symptom correlation engine
5. Add AS-specific meditation content

### Option B: Build from Scratch

**Pros:**
- Perfect alignment with app architecture from start
- No technical debt from existing code
- Clean MVVM implementation

**Cons:**
- Duplicate 4,000+ lines of existing code
- 2-3 weeks additional development time
- Risk of missing features

## Recommended Implementation Plan (Option A)

### Phase 1: Core Data Integration (Week 1)

**1.1 Add Meditation Entities to Core Data Model**

```xml
<!-- MEDITATION SESSION - Completed meditation sessions -->
<entity name="MeditationSession" representedClassName="MeditationSession" syncable="YES" codeGenerationType="class">
    <attribute name="id" attributeType="UUID" usesScalarValueType="NO"/>
    <attribute name="timestamp" attributeType="Date" indexed="YES" usesScalarValueType="NO"/>
    <attribute name="sessionType" attributeType="String"/> <!-- mindfulness, breathing, body_scan, pain_relief, etc. -->
    <attribute name="title" attributeType="String"/>
    <attribute name="durationSeconds" attributeType="Integer 32" usesScalarValueType="YES"/>
    <attribute name="completedDuration" attributeType="Integer 32" usesScalarValueType="YES"/>
    <attribute name="isCompleted" attributeType="Boolean" defaultValueString="YES" usesScalarValueType="YES"/>

    <!-- Before/After Metrics -->
    <attribute name="stressLevelBefore" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>
    <attribute name="stressLevelAfter" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>
    <attribute name="painLevelBefore" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>
    <attribute name="painLevelAfter" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>
    <attribute name="moodBefore" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>
    <attribute name="moodAfter" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>
    <attribute name="energyBefore" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>
    <attribute name="energyAfter" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>

    <!-- Session Details -->
    <attribute name="breathingPattern" optional="YES" attributeType="String"/>
    <attribute name="heartRateData" optional="YES" attributeType="Binary"/>
    <attribute name="avgHeartRate" optional="YES" attributeType="Double" usesScalarValueType="YES"/>
    <attribute name="hrvValue" optional="YES" attributeType="Double" usesScalarValueType="YES"/>
    <attribute name="notes" optional="YES" attributeType="String"/>

    <!-- Correlation with Symptoms -->
    <relationship name="symptomLog" optional="YES" maxCount="1" deletionRule="Nullify" destinationEntity="SymptomLog"/>

    <fetchIndex name="byTimestamp">
        <fetchIndexElement property="timestamp" type="Binary" order="descending"/>
    </fetchIndex>
    <fetchIndex name="bySessionType">
        <fetchIndexElement property="sessionType" type="Binary" order="ascending"/>
        <fetchIndexElement property="timestamp" type="Binary" order="descending"/>
    </fetchIndex>
</entity>

<!-- MEDITATION STREAK - User's meditation consistency tracking -->
<entity name="MeditationStreak" representedClassName="MeditationStreak" syncable="YES" codeGenerationType="class">
    <attribute name="id" attributeType="UUID" usesScalarValueType="NO"/>
    <attribute name="currentStreak" attributeType="Integer 16" defaultValueString="0" usesScalarValueType="YES"/>
    <attribute name="longestStreak" attributeType="Integer 16" defaultValueString="0" usesScalarValueType="YES"/>
    <attribute name="totalSessions" attributeType="Integer 32" defaultValueString="0" usesScalarValueType="YES"/>
    <attribute name="totalMinutes" attributeType="Double" defaultValueString="0.0" usesScalarValueType="YES"/>
    <attribute name="lastSessionDate" optional="YES" attributeType="Date" usesScalarValueType="NO"/>
    <attribute name="weeklyGoal" attributeType="Integer 16" defaultValueString="7" usesScalarValueType="YES"/>
    <attribute name="monthlyGoal" attributeType="Integer 16" defaultValueString="30" usesScalarValueType="YES"/>
</entity>
```

**1.2 Create Core Data Helper for Meditation**

Location: `InflamAI/Core/Persistence/MeditationPersistenceHelper.swift`

```swift
@MainActor
class MeditationPersistenceHelper {
    let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    func saveMeditationSession(_ session: MeditationSessionData) throws {
        let entity = MeditationSession(context: context)
        entity.id = session.id
        entity.timestamp = session.timestamp
        entity.sessionType = session.type.rawValue
        // ... map all properties

        try context.save()
    }

    func fetchRecentSessions(days: Int = 30) throws -> [MeditationSession] {
        let request = MeditationSession.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp >= %@", Calendar.current.date(byAdding: .day, value: -days, to: Date())! as CVarArg)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \MeditationSession.timestamp, ascending: false)]

        return try context.fetch(request)
    }

    func getOrCreateStreak() throws -> MeditationStreak {
        let request = MeditationStreak.fetchRequest()
        request.fetchLimit = 1

        if let streak = try context.fetch(request).first {
            return streak
        }

        let newStreak = MeditationStreak(context: context)
        newStreak.id = UUID()
        newStreak.currentStreak = 0
        newStreak.longestStreak = 0
        newStreak.totalSessions = 0
        newStreak.totalMinutes = 0
        newStreak.weeklyGoal = 7
        newStreak.monthlyGoal = 30

        try context.save()
        return newStreak
    }
}
```

### Phase 2: Architecture Migration (Week 1-2)

**2.1 Consolidate Meditation Files**

Consolidate the 3 existing meditation files into:

```
InflamAI/Features/Meditation/
├── Views/
│   ├── MeditationHomeView.swift          # Main entry (Discover tab)
│   ├── MeditationLibraryView.swift       # Library tab
│   ├── MeditationPlayerView.swift        # Active session player
│   ├── MeditationProgressView.swift      # Progress/Stats tab
│   ├── MeditationProfileView.swift       # Settings/Profile tab
│   ├── SessionDetailView.swift           # Session details modal
│   ├── BreathingGuideView.swift          # Breathing exercise overlay
│   └── Components/
│       ├── SessionCard.swift
│       ├── StreakCard.swift
│       ├── AchievementBadge.swift
│       └── BreathingCircle.swift
├── ViewModels/
│   ├── MeditationViewModel.swift         # Main ViewModel
│   ├── SessionPlayerViewModel.swift      # Player controls
│   └── ProgressViewModel.swift           # Stats & insights
├── Models/
│   ├── MeditationSessionModel.swift      # Session metadata
│   ├── BreathingPattern.swift
│   ├── MeditationCategory.swift
│   └── MeditationInsight.swift
└── Services/
    ├── MeditationAudioManager.swift      # AVFoundation audio
    ├── MeditationDownloadManager.swift   # Offline downloads
    └── MeditationAnalyticsEngine.swift   # Insights generation
```

**2.2 Create Main ViewModel**

Location: `InflamAI/Features/Meditation/ViewModels/MeditationViewModel.swift`

```swift
@MainActor
class MeditationViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var availableSessions: [MeditationSessionModel] = []
    @Published var currentSession: MeditationSessionModel?
    @Published var isPlaying = false
    @Published var isPaused = false
    @Published var currentTime: TimeInterval = 0
    @Published var streak: MeditationStreak?
    @Published var recentSessions: [MeditationSession] = []
    @Published var insights: [MeditationInsight] = []

    // MARK: - Dependencies
    private let persistenceController: InflamAIPersistenceController
    private let audioManager: MeditationAudioManager
    private let analyticsEngine: MeditationAnalyticsEngine
    private let healthKitService: HealthKitService

    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
        self.audioManager = MeditationAudioManager()
        self.analyticsEngine = MeditationAnalyticsEngine()
        self.healthKitService = HealthKitService.shared

        loadData()
    }

    func loadData() {
        let helper = MeditationPersistenceHelper(context: persistenceController.container.viewContext)

        do {
            recentSessions = try helper.fetchRecentSessions(days: 30)
            streak = try helper.getOrCreateStreak()
        } catch {
            print("Failed to load meditation data: \(error)")
        }

        availableSessions = createDefaultSessions()
    }

    func startSession(_ session: MeditationSessionModel) async throws {
        guard !isPlaying else { return }

        currentSession = session
        currentTime = 0
        isPlaying = true

        // Start audio playback
        try await audioManager.play(session)

        // Start heart rate monitoring
        if let hrv = try? await healthKitService.fetchLatestHRV() {
            // Track heart rate during session
        }
    }

    func completeSession(stressAfter: Int?, painAfter: Int?, moodAfter: Int?) async throws {
        guard let session = currentSession else { return }

        let helper = MeditationPersistenceHelper(context: persistenceController.container.viewContext)

        let sessionData = MeditationSessionData(
            id: UUID(),
            timestamp: Date(),
            type: session.type,
            title: session.title,
            durationSeconds: Int(session.duration),
            completedDuration: Int(currentTime),
            isCompleted: currentTime >= session.duration * 0.8,
            stressLevelBefore: nil,
            stressLevelAfter: stressAfter,
            painLevelBefore: nil,
            painLevelAfter: painAfter,
            moodBefore: nil,
            moodAfter: moodAfter
        )

        try helper.saveMeditationSession(sessionData)

        // Update streak
        if let streak = streak {
            updateStreak(streak)
        }

        // Reload data
        loadData()

        // Reset state
        currentSession = nil
        isPlaying = false
        currentTime = 0
    }
}
```

### Phase 3: AS-Specific Integration (Week 2)

**3.1 Add Pain Relief Sessions**

Create 15+ AS-specific meditation sessions:

```swift
static let asSpecificSessions: [MeditationSessionModel] = [
    MeditationSessionModel(
        id: UUID(),
        title: "Spinal Pain Relief",
        description: "Guided meditation targeting chronic spinal pain and inflammation",
        type: .painRelief,
        category: .painManagement,
        duration: 900, // 15 minutes
        difficulty: .beginner,
        targetSymptoms: [.jointPain, .spinalPain, .inflammation],
        recommendedTime: [.evening, .night],
        benefits: [
            "Reduces pain perception through mindful awareness",
            "Calms nervous system response to inflammation",
            "Promotes muscle relaxation in affected areas"
        ]
    ),

    MeditationSessionModel(
        id: UUID(),
        title: "Morning Stiffness Relief",
        description: "Gentle breathing and body scan to ease morning stiffness",
        type: .bodyScan,
        category: .morningRoutine,
        duration: 600, // 10 minutes
        difficulty: .beginner,
        targetSymptoms: [.stiffness, .morningPain],
        recommendedTime: [.earlyMorning, .morning],
        hasBreathingGuide: true
    ),

    MeditationSessionModel(
        id: UUID(),
        title: "Flare Management",
        description: "Crisis meditation for managing acute flare episodes",
        type: .painRelief,
        category: .emergency,
        duration: 480, // 8 minutes
        difficulty: .beginner,
        targetSymptoms: [.flare, .severeInflammation, .anxietyDuringFlare]
    ),

    MeditationSessionModel(
        id: UUID(),
        title: "Sleep Preparation",
        description: "Evening meditation to improve sleep quality despite pain",
        type: .sleep,
        category: .sleepImprovement,
        duration: 1200, // 20 minutes
        difficulty: .beginner,
        targetSymptoms: [.insomnia, .nocturnalPain, .sleepDisturbance]
    )

    // ... 11 more AS-specific sessions
]
```

**3.2 Integrate with Body Map**

Location: `InflamAI/Features/Meditation/Services/BodyMapMeditationIntegration.swift`

```swift
class BodyMapMeditationIntegration {
    /// Suggests meditation based on current body region pain
    func suggestSessionForPainRegions(_ regions: [BodyRegionLog]) -> [MeditationSessionModel] {
        let painfulRegions = regions.filter { $0.painLevel >= 5 }

        if painfulRegions.contains(where: { $0.regionID?.hasPrefix("C") == true }) {
            // Cervical spine pain - suggest neck/upper body meditations
            return MeditationSessionModel.asSpecificSessions.filter {
                $0.targetSymptoms.contains(.neckPain) ||
                $0.targetSymptoms.contains(.upperSpinalPain)
            }
        }

        if painfulRegions.contains(where: { $0.regionID?.hasPrefix("L") == true }) {
            // Lumbar pain - suggest lower back meditations
            return MeditationSessionModel.asSpecificSessions.filter {
                $0.targetSymptoms.contains(.lowerBackPain)
            }
        }

        // General spinal pain
        return MeditationSessionModel.asSpecificSessions.filter {
            $0.type == .painRelief || $0.type == .bodyScan
        }
    }
}
```

**3.3 Correlation with Symptom Tracking**

Location: `InflamAI/Core/Analytics/MeditationCorrelationEngine.swift`

```swift
class MeditationCorrelationEngine {
    let persistenceController: InflamAIPersistenceController

    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
    }

    /// Analyzes correlation between meditation practice and symptom reduction
    func analyzeMeditationImpact(days: Int = 30) throws -> MeditationImpactAnalysis {
        let context = persistenceController.container.viewContext

        // Fetch meditation sessions
        let meditationRequest = MeditationSession.fetchRequest()
        meditationRequest.predicate = NSPredicate(format: "timestamp >= %@", Calendar.current.date(byAdding: .day, value: -days, to: Date())! as CVarArg)
        let meditationSessions = try context.fetch(meditationRequest)

        // Fetch symptom logs
        let symptomRequest = SymptomLog.fetchRequest()
        symptomRequest.predicate = NSPredicate(format: "timestamp >= %@", Calendar.current.date(byAdding: .day, value: -days, to: Date())! as CVarArg)
        let symptomLogs = try context.fetch(symptomRequest)

        // Group by day
        var daysWithMeditation: Set<Date> = []
        var daysWithoutMeditation: Set<Date> = []

        for session in meditationSessions {
            if let timestamp = session.timestamp {
                daysWithMeditation.insert(Calendar.current.startOfDay(for: timestamp))
            }
        }

        for log in symptomLogs {
            if let timestamp = log.timestamp {
                let day = Calendar.current.startOfDay(for: timestamp)
                if !daysWithMeditation.contains(day) {
                    daysWithoutMeditation.insert(day)
                }
            }
        }

        // Calculate average pain on meditation days vs non-meditation days
        var meditationDayPain: [Double] = []
        var nonMeditationDayPain: [Double] = []

        for log in symptomLogs {
            guard let timestamp = log.timestamp else { continue }
            let day = Calendar.current.startOfDay(for: timestamp)

            let painLevel = log.painAverage24h

            if daysWithMeditation.contains(day) {
                meditationDayPain.append(Double(painLevel))
            } else {
                nonMeditationDayPain.append(Double(painLevel))
            }
        }

        let avgPainWithMeditation = meditationDayPain.isEmpty ? 0 : meditationDayPain.reduce(0, +) / Double(meditationDayPain.count)
        let avgPainWithoutMeditation = nonMeditationDayPain.isEmpty ? 0 : nonMeditationDayPain.reduce(0, +) / Double(nonMeditationDayPain.count)

        let painReduction = avgPainWithoutMeditation - avgPainWithMeditation
        let percentageImprovement = avgPainWithoutMeditation > 0 ? (painReduction / avgPainWithoutMeditation) * 100 : 0

        return MeditationImpactAnalysis(
            daysAnalyzed: days,
            daysWithMeditation: daysWithMeditation.count,
            daysWithoutMeditation: daysWithoutMeditation.count,
            avgPainWithMeditation: avgPainWithMeditation,
            avgPainWithoutMeditation: avgPainWithoutMeditation,
            painReduction: painReduction,
            percentageImprovement: percentageImprovement,
            statisticalSignificance: calculateSignificance(meditationDayPain, nonMeditationDayPain)
        )
    }

    private func calculateSignificance(_ group1: [Double], _ group2: [Double]) -> Double {
        // Simple t-test implementation
        // In production, use proper statistical library
        return 0.05 // Placeholder
    }
}

struct MeditationImpactAnalysis {
    let daysAnalyzed: Int
    let daysWithMeditation: Int
    let daysWithoutMeditation: Int
    let avgPainWithMeditation: Double
    let avgPainWithoutMeditation: Double
    let painReduction: Double
    let percentageImprovement: Double
    let statisticalSignificance: Double
}
```

### Phase 4: UI Integration (Week 2-3)

**4.1 Add to Main Navigation**

Location: `InflamAI/InflamAIApp.swift`

```swift
enum Tab {
    case home
    case bodyMap
    case meditation  // NEW
    case trends
    case settings
}

TabView(selection: $selectedTab) {
    HomeView()
        .tabItem {
            Label("Home", systemImage: "house.fill")
        }
        .tag(Tab.home)

    BodyMapView()
        .tabItem {
            Label("Body Map", systemImage: "figure.walk")
        }
        .tag(Tab.bodyMap)

    MeditationHomeView()  // NEW
        .tabItem {
            Label("Meditation", systemImage: "brain.head.profile")
        }
        .tag(Tab.meditation)

    TrendsView()
        .tabItem {
            Label("Trends", systemImage: "chart.line.uptrend.xyaxis")
        }
        .tag(Tab.trends)

    SettingsView()
        .tabItem {
            Label("Settings", systemImage: "gearshape.fill")
        }
        .tag(Tab.settings)
}
```

**4.2 Add Quick Access from Home Dashboard**

Location: `InflamAI/Features/Home/HomeView.swift`

```swift
// Add to HomeView dashboard
VStack(spacing: 16) {
    // Existing widgets...

    // NEW: Meditation Quick Access
    MeditationQuickAccessCard()
}

struct MeditationQuickAccessCard: View {
    @StateObject private var viewModel = MeditationViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(.purple)

                Text("Meditation")
                    .font(.headline)

                Spacer()

                NavigationLink(destination: MeditationHomeView()) {
                    Text("See All")
                        .font(.subheadline)
                        .foregroundColor(.blue)
                }
            }

            if let streak = viewModel.streak, streak.currentStreak > 0 {
                HStack {
                    Image(systemName: "flame.fill")
                        .foregroundColor(.orange)
                    Text("\(streak.currentStreak) day streak")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }

            // Recommended session based on current symptoms
            if let recommended = viewModel.recommendedSession {
                Button {
                    // Quick start session
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Recommended")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text(recommended.title)
                                .font(.subheadline)
                                .fontWeight(.medium)

                            Text("\(Int(recommended.duration / 60)) min")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Image(systemName: "play.circle.fill")
                            .font(.title)
                            .foregroundColor(.purple)
                    }
                    .padding()
                    .background(Color.purple.opacity(0.1))
                    .cornerRadius(12)
                }
                .buttonStyle(.plain)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.05), radius: 2)
    }
}
```

### Phase 5: Enhanced Features (Week 3-4)

**5.1 AI-Powered Recommendations**

```swift
class MeditationRecommendationEngine {
    func generateRecommendations(
        recentSymptoms: [SymptomLog],
        recentSessions: [MeditationSession],
        currentTime: Date
    ) -> [MeditationSessionModel] {
        var recommendations: [MeditationSessionModel] = []

        // 1. Based on current symptoms
        if let latestSymptom = recentSymptoms.first {
            if latestSymptom.painAverage24h >= 6 {
                // High pain - recommend pain relief
                recommendations.append(contentsOf: getSessions(type: .painRelief))
            }

            if latestSymptom.stressLevel >= 7 {
                // High stress - recommend stress reduction
                recommendations.append(contentsOf: getSessions(type: .stressReduction))
            }

            if latestSymptom.sleepQuality <= 3 {
                // Poor sleep - recommend sleep meditation
                recommendations.append(contentsOf: getSessions(type: .sleep))
            }
        }

        // 2. Based on time of day
        let hour = Calendar.current.component(.hour, from: currentTime)
        if hour < 10 {
            recommendations.append(contentsOf: getSessions(recommendedTime: .morning))
        } else if hour > 20 {
            recommendations.append(contentsOf: getSessions(recommendedTime: .evening))
        }

        // 3. Based on meditation history
        if recentSessions.isEmpty {
            // Beginner - suggest beginner sessions
            recommendations.append(contentsOf: getSessions(difficulty: .beginner))
        }

        // 4. Trending up - continue streak
        if let lastSession = recentSessions.first,
           let timestamp = lastSession.timestamp,
           Calendar.current.isDateInToday(timestamp) {
            // Already meditated today - suggest variety
            let completedTypes = Set(recentSessions.compactMap { $0.sessionType })
            recommendations.append(contentsOf: getSessions(excluding: completedTypes))
        }

        return Array(Set(recommendations)).prefix(5).map { $0 }
    }
}
```

**5.2 Insights Dashboard**

Create meditation insights display:

```swift
struct MeditationInsightsView: View {
    @StateObject private var viewModel = MeditationInsightsViewModel()

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Impact Analysis
                if let impact = viewModel.impactAnalysis {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Meditation Impact")
                            .font(.headline)

                        HStack {
                            VStack(alignment: .leading) {
                                Text("Pain on meditation days")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.1f/10", impact.avgPainWithMeditation))
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.green)
                            }

                            Spacer()

                            VStack(alignment: .trailing) {
                                Text("Pain without meditation")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.1f/10", impact.avgPainWithoutMeditation))
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.orange)
                            }
                        }

                        Text("Meditation reduces pain by \(String(format: "%.1f%%", impact.percentageImprovement))")
                            .font(.subheadline)
                            .padding()
                            .background(Color.green.opacity(0.1))
                            .cornerRadius(8)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }

                // More insights...
            }
            .padding()
        }
        .navigationTitle("Insights")
    }
}
```

## Implementation Timeline

### Week 1: Foundation
- ✅ Add Core Data entities (2 days)
- ✅ Create persistence helpers (1 day)
- ✅ Consolidate existing meditation files (2 days)

### Week 2: Integration
- ✅ Migrate to Features/Meditation structure (2 days)
- ✅ Integrate with main navigation (1 day)
- ✅ Add AS-specific sessions (2 days)

### Week 3: Analytics
- ✅ Implement correlation engine (2 days)
- ✅ Add insights dashboard (2 days)
- ✅ Integrate with body map (1 day)

### Week 4: Polish
- ✅ Testing and bug fixes (2 days)
- ✅ Accessibility improvements (1 day)
- ✅ Documentation (1 day)
- ✅ User testing (1 day)

## Technical Considerations

### Privacy & Security
- ✅ All meditation data stored locally in Core Data
- ✅ Optional CloudKit sync (user-controlled)
- ✅ HealthKit integration for mindful minutes logging
- ✅ No third-party analytics or tracking

### Accessibility (WCAG AA)
- ✅ VoiceOver support for all meditation UI
- ✅ Audio descriptions for breathing exercises
- ✅ Dynamic Type support
- ✅ Reduce Motion respect for breathing animations
- ✅ Haptic feedback for breathing guidance

### Performance
- ✅ Background audio playback during meditation
- ✅ Offline-first architecture with downloads
- ✅ Lazy loading of meditation content
- ✅ Efficient Core Data queries with indexes

## Medical Disclaimers

All meditation content must include:

> ⚠️ **Medical Disclaimer**: Meditation is a complementary practice and not a substitute for professional medical treatment. Always consult your rheumatologist regarding pain management strategies.

## Success Metrics

After 4 weeks of implementation:
- ✅ 80%+ of existing meditation code reused
- ✅ Full Core Data integration
- ✅ 15+ AS-specific meditation sessions
- ✅ Correlation analysis with symptom tracking
- ✅ Main app navigation integration
- ✅ WCAG AA accessibility compliance

## Questions for User

Before proceeding with implementation, please clarify:

1. **Do you want to use the existing meditation code?**
   - Option A: Integrate existing implementation (saves 2-3 weeks)
   - Option B: Build from scratch (perfect architecture alignment)

2. **What's the priority for meditation content?**
   - Pre-recorded guided sessions (requires audio recording)
   - Timer-based unguided sessions (simpler, faster to implement)
   - Breathing exercises only (minimal content needs)

3. **HealthKit Integration Scope?**
   - Just log mindful minutes (simple)
   - Track heart rate during meditation (complex)
   - Analyze HRV trends (most complex)

4. **Download/Offline Features?**
   - Essential for v1.0?
   - Can be added later?

5. **Integration Priority?**
   - Body map integration (high value for AS patients)
   - Correlation analysis (requires 30+ days of data)
   - Quick access from Home (high visibility)

---

**Recommendation**: I strongly recommend **Option A** (integrate existing implementation) as it provides:
- 4,000+ lines of production-ready code
- Comprehensive feature set
- 2-3 weeks time savings
- Professional UI/UX

The main work would be:
1. Core Data integration (3-4 days)
2. Architecture migration to Features folder (2-3 days)
3. AS-specific content creation (3-4 days)
4. Symptom correlation (2-3 days)

**Total: ~2-3 weeks vs 5-6 weeks from scratch**

Please confirm your preferred approach, and I'll proceed with detailed implementation.
