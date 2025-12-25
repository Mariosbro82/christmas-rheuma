# InflamAI Widget Implementation Plan

**Status**: Planning Phase
**Target iOS**: iOS 17.0+ (WidgetKit), iOS 18+ (Control Widgets)
**Target watchOS**: watchOS 10+ (WidgetKit)

---

## Overview

This plan outlines the implementation of widgets across all Apple platforms for InflamAI, providing at-a-glance health insights and quick actions.

---

## Widget Categories

### 1. Home Screen Widgets (WidgetKit)

#### Small Widget (2x2) - Single Metric Focus

| Widget Name | Data Source | Description |
|------------|-------------|-------------|
| **Flare Risk** | `UnifiedNeuralEngine.currentPrediction` | Circular gauge showing 0-100% flare risk with color coding |
| **BASDAI Score** | `SymptomLog.basdaiScore` | Current BASDAI score with severity color |
| **Logging Streak** | Calculated from `SymptomLog` | Flame icon + streak count + "days" label |
| **Next Medication** | `Medication.reminderTimes` | Pill icon + medication name + time |

#### Medium Widget (4x2) - Multi-Metric Dashboard

| Widget Name | Content |
|------------|---------|
| **Today's Summary** | Pain entries count, BASDAI score, medication status, streak |
| **Flare Risk + Factors** | Risk percentage + top 2 contributing factors |
| **Weekly Trends** | Mini sparkline charts for pain/stiffness/fatigue |
| **Medication Schedule** | Next 3 upcoming medications with times |

#### Large Widget (4x4) - Detailed View

| Widget Name | Content |
|------------|---------|
| **Daily Health Dashboard** | Full day summary: BASDAI, flare risk, trends, medications, active flare alert |
| **7-Day Overview** | Weekly calendar view with daily BASDAI scores colored by severity |

---

### 2. Lock Screen Widgets (iOS 16+)

#### Circular Widgets (WidgetFamily.accessoryCircular)

| Widget Name | Display |
|------------|---------|
| **Flare Risk Gauge** | Circular progress showing risk % with icon |
| **BASDAI Circle** | Score number with severity ring color |
| **Streak Fire** | Flame icon with streak number |

#### Rectangular Widgets (WidgetFamily.accessoryRectangular)

| Widget Name | Display |
|------------|---------|
| **Flare Status** | "Risk: 45% - Moderate" with icon |
| **Next Med** | Pill icon + "Humira in 2h" |
| **Today Status** | "BASDAI: 4.2 | Streak: 12" |

#### Inline Widgets (WidgetFamily.accessoryInline)

| Widget Name | Display |
|------------|---------|
| **Flare Risk Text** | "Flare Risk: 45% Moderate" |
| **BASDAI Text** | "BASDAI: 4.2 Moderate" |

---

### 3. Lock Screen Control Buttons (iOS 18+)

**Note**: These are the bottom-left/right action buttons (like Camera & Flashlight)

#### Control Widgets (ControlWidget)

| Control Name | Action | Icon |
|-------------|--------|------|
| **Quick Log** | Opens app to QuickLogView | `pencil.circle.fill` |
| **SOS Flare** | Opens app to SOSFlareView | `flame.fill` |
| **Log Medication** | Opens medication logging | `pills.fill` |
| **Start Exercise** | Opens exercise routine | `figure.walk` |

---

### 4. Apple Watch Smart Stack Widgets (watchOS 10+)

#### Rectangular Watch Widgets (WidgetFamily.accessoryRectangular)

| Widget Name | Content |
|------------|---------|
| **Flare Risk** | Risk % with circular gauge + "Low/Moderate/High" |
| **Today's BASDAI** | Score with severity color + mini trend arrow |
| **Quick Stats** | Steps, HRV, and pain level in compact view |
| **Next Medication** | Medication name + countdown |

#### Circular Watch Widgets (WidgetFamily.accessoryCircular)

| Widget Name | Content |
|------------|---------|
| **Flare Gauge** | Circular progress with risk % |
| **BASDAI Ring** | Score with colored ring |
| **Streak** | Fire icon + number |

#### Corner Complications (WidgetFamily.accessoryCorner)

| Widget Name | Content |
|------------|---------|
| **Flare Risk** | Arc gauge + % text |
| **BASDAI** | Score with arc indicator |

---

## Technical Architecture

### File Structure

```
InflamAI/
├── InflamAIWidgetExtension/           # NEW: Widget Extension target
│   ├── InflamAIWidgetExtension.swift  # @main entry point
│   ├── Widgets/
│   │   ├── FlareRiskWidget.swift
│   │   ├── BASDAIWidget.swift
│   │   ├── StreakWidget.swift
│   │   ├── MedicationWidget.swift
│   │   ├── DailyDashboardWidget.swift
│   │   └── WeeklyOverviewWidget.swift
│   ├── Views/
│   │   ├── FlareRiskGaugeView.swift
│   │   ├── BASDAISeverityView.swift
│   │   ├── MedicationRowView.swift
│   │   └── TrendSparklineView.swift
│   ├── Providers/
│   │   ├── FlareRiskProvider.swift
│   │   ├── BASDAIProvider.swift
│   │   ├── MedicationProvider.swift
│   │   └── HealthDataProvider.swift
│   ├── Models/
│   │   └── WidgetDataModels.swift
│   ├── Intents/
│   │   └── WidgetIntents.swift
│   └── Assets.xcassets/
│
├── InflamAIControlExtension/          # NEW: Control Widget Extension (iOS 18)
│   ├── InflamAIControlExtension.swift
│   └── Controls/
│       ├── QuickLogControl.swift
│       ├── SOSFlareControl.swift
│       └── MedicationControl.swift
│
├── InflamAIWatch Extension/           # NEW: watchOS Widget Extension
│   ├── InflamAIWatchWidgets.swift
│   └── WatchWidgets/
│       ├── WatchFlareWidget.swift
│       ├── WatchBASDAIWidget.swift
│       └── WatchMedicationWidget.swift
│
└── Shared/                               # EXISTING: Shared code
    └── WidgetShared/                     # NEW: Code shared between app and widgets
        ├── WidgetDataProvider.swift      # Shared data fetching
        └── AppGroup+Widget.swift         # App Group configuration
```

### Data Sharing Strategy

#### App Groups (Required)

```swift
// App Group identifier for sharing data between app and widgets
let appGroupIdentifier = "group.com.spinalytics.shared"

// Shared UserDefaults
let sharedDefaults = UserDefaults(suiteName: appGroupIdentifier)

// Shared Core Data container
let sharedContainer = NSPersistentContainer(name: "InflamAI")
sharedContainer.persistentStoreDescriptions.first?.url =
    FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroupIdentifier)?
    .appendingPathComponent("InflamAI.sqlite")
```

#### Timeline Provider Pattern

```swift
struct FlareRiskProvider: TimelineProvider {
    typealias Entry = FlareRiskEntry

    func placeholder(in context: Context) -> FlareRiskEntry {
        FlareRiskEntry(date: Date(), riskPercentage: 35, riskLevel: .moderate)
    }

    func getSnapshot(in context: Context, completion: @escaping (FlareRiskEntry) -> Void) {
        // Return current data for widget gallery preview
        let entry = fetchCurrentFlareRisk()
        completion(entry)
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<FlareRiskEntry>) -> Void) {
        // Fetch data and create timeline
        let currentEntry = fetchCurrentFlareRisk()

        // Update every 15 minutes (widgets refresh on schedule)
        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: Date())!
        let timeline = Timeline(entries: [currentEntry], policy: .after(nextUpdate))
        completion(timeline)
    }
}
```

### Widget Data Models

```swift
// Shared data models for widgets
struct WidgetFlareData: Codable {
    let riskPercentage: Int
    let riskLevel: RiskLevel
    let topFactors: [String]
    let lastUpdated: Date

    enum RiskLevel: String, Codable {
        case low, moderate, high, veryHigh
    }
}

struct WidgetBASDAIData: Codable {
    let score: Double
    let category: String
    let trend: TrendDirection
    let lastAssessed: Date

    enum TrendDirection: String, Codable {
        case improving, stable, worsening
    }
}

struct WidgetMedicationData: Codable {
    let medications: [MedicationReminder]

    struct MedicationReminder: Codable {
        let name: String
        let time: Date
        let dosage: String
    }
}
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Tasks:**
1. Create Widget Extension target in Xcode
2. Configure App Groups for data sharing
3. Implement shared Core Data access for widgets
4. Create `WidgetDataProvider` for centralized data fetching
5. Set up widget entry models

**Files to Create:**
- `InflamAIWidgetExtension.swift`
- `Shared/WidgetShared/AppGroup+Widget.swift`
- `Shared/WidgetShared/WidgetDataProvider.swift`
- `Models/WidgetDataModels.swift`

### Phase 2: Home Screen Widgets (Week 2)

**Tasks:**
1. Implement Small widgets (Flare Risk, BASDAI, Streak)
2. Implement Medium widgets (Today's Summary, Medication Schedule)
3. Implement Large widgets (Daily Dashboard)
4. Add widget configuration intents
5. Design widget preview assets

**Files to Create:**
- `Widgets/FlareRiskWidget.swift`
- `Widgets/BASDAIWidget.swift`
- `Widgets/StreakWidget.swift`
- `Widgets/MedicationWidget.swift`
- `Widgets/DailyDashboardWidget.swift`
- `Views/*.swift` (supporting views)

### Phase 3: Lock Screen Widgets (Week 3)

**Tasks:**
1. Implement accessoryCircular widgets
2. Implement accessoryRectangular widgets
3. Implement accessoryInline widgets
4. Handle different context sizes
5. Test on various device sizes

**Updates to:**
- `Widgets/FlareRiskWidget.swift` (add lock screen families)
- `Widgets/BASDAIWidget.swift` (add lock screen families)
- `Widgets/MedicationWidget.swift` (add lock screen families)

### Phase 4: iOS 18 Control Widgets (Week 4)

**Tasks:**
1. Create Control Extension target
2. Implement Quick Log control
3. Implement SOS Flare control
4. Implement Medication control
5. Configure deep links for actions

**Files to Create:**
- `InflamAIControlExtension/InflamAIControlExtension.swift`
- `Controls/QuickLogControl.swift`
- `Controls/SOSFlareControl.swift`
- `Controls/MedicationControl.swift`

### Phase 5: Apple Watch Widgets (Week 5)

**Tasks:**
1. Create Watch Widget Extension target
2. Implement watch-specific widget designs
3. Add corner complications
4. Configure Smart Stack relevance
5. Implement HealthKit background delivery for watch

**Files to Create:**
- `InflamAIWatch Extension/InflamAIWatchWidgets.swift`
- `WatchWidgets/WatchFlareWidget.swift`
- `WatchWidgets/WatchBASDAIWidget.swift`
- `WatchWidgets/WatchMedicationWidget.swift`

### Phase 6: Polish & Optimization (Week 6)

**Tasks:**
1. Optimize widget refresh intervals
2. Add widget configuration options
3. Implement widget analytics
4. Accessibility review (VoiceOver, Dynamic Type)
5. Performance testing and optimization
6. Documentation

---

## Widget Designs

### Color Scheme (Severity-Based)

```swift
extension Color {
    static let widgetRemission = Color.green       // BASDAI 0-2
    static let widgetLow = Color(red: 0.6, green: 0.8, blue: 0.2)  // BASDAI 2-4
    static let widgetModerate = Color.orange      // BASDAI 4-6
    static let widgetHigh = Color(red: 0.9, green: 0.3, blue: 0.1) // BASDAI 6-8
    static let widgetVeryHigh = Color.red         // BASDAI 8+
}
```

### Flare Risk Gauge Design

```
┌────────────────┐
│    ╭──────╮    │   Small: Circular gauge
│   │   42%  │   │   - Percentage in center
│    ╰──────╯    │   - Arc colored by risk level
│   Low Risk     │   - Label below
└────────────────┘

┌────────────────────────────────┐
│  ⚠️ Flare Risk: 67%           │   Medium: Full info
│  ━━━━━━━━━━━░░░░░              │   - Progress bar
│  High Risk                     │   - Risk factors
│  📊 Weather │ 😴 Sleep         │
└────────────────────────────────┘
```

### BASDAI Widget Design

```
┌────────────────┐
│      4.2       │   Small: Score focus
│    ●●●●○○      │   - Large number
│   Moderate     │   - Dots indicator
└────────────────┘

┌────────────────────────────────┐
│  BASDAI Score           ↗️     │   Medium: With trend
│  ┌─────┐                       │
│  │ 4.2 │  Moderate Activity    │
│  └─────┘                       │
│  Last: 2h ago • Trend: +0.3    │
└────────────────────────────────┘
```

---

## Deep Linking

```swift
// URL scheme for widget actions
enum WidgetDeepLink: String {
    case quickLog = "spinalytics://widget/quicklog"
    case sosFlare = "spinalytics://widget/sosflare"
    case medication = "spinalytics://widget/medication"
    case exercise = "spinalytics://widget/exercise"
    case trends = "spinalytics://widget/trends"
    case flareDetails = "spinalytics://widget/flare"
}

// Handle in InflamAIApp.swift
.onOpenURL { url in
    handleWidgetDeepLink(url)
}
```

---

## Privacy Considerations

1. **Minimal Data in Widgets**: Only display aggregate scores, not detailed health records
2. **Redaction in Lock Screen**: Use `.privacySensitive()` for sensitive data when device is locked
3. **No PHI in Widget Gallery**: Placeholder data should be generic
4. **App Group Security**: Encrypted Core Data store in shared container

```swift
// Example: Privacy-sensitive widget content
Text(String(format: "%.1f", basdaiScore))
    .privacySensitive()  // Redacts on lock screen if needed
```

---

## Testing Checklist

### Widget Functionality
- [ ] All widget sizes render correctly
- [ ] Timeline updates work properly
- [ ] Deep links open correct views
- [ ] Widget configuration saves/loads
- [ ] Placeholder shows generic data

### Data Accuracy
- [ ] Flare risk matches main app
- [ ] BASDAI score is current
- [ ] Medication times are correct
- [ ] Streak calculation is accurate

### Edge Cases
- [ ] No data available (first launch)
- [ ] Offline mode
- [ ] Insufficient data for predictions
- [ ] Active flare state changes

### Accessibility
- [ ] VoiceOver labels are descriptive
- [ ] Dynamic Type scaling
- [ ] Sufficient color contrast
- [ ] Reduced motion respected

### Performance
- [ ] Widget loads in <100ms
- [ ] Background refresh is efficient
- [ ] Memory usage is minimal
- [ ] Battery impact is negligible

---

## Dependencies

### Required Frameworks
- `WidgetKit` - Core widget functionality
- `SwiftUI` - Widget UI
- `CoreData` - Shared data access
- `Intents` - Widget configuration
- `AppIntents` - iOS 17+ intents
- `HealthKit` - Watch health data

### Required Capabilities
- App Groups (enabled in both main app and extensions)
- WidgetKit Extension
- Background Modes (for timeline updates)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Core Data sync issues | Widget shows stale data | Implement robust sync with timestamps |
| Widget gallery crashes | Poor user experience | Use safe placeholder data |
| iOS 18 Control API changes | Broken controls | Conditional compilation for iOS versions |
| Watch connectivity issues | Missing data | Cache last known values locally |

---

## Success Metrics

1. **Widget Adoption**: >60% of users add at least one widget
2. **Daily Active Widget Users**: >40% of DAU
3. **Widget-Initiated Sessions**: Track deep link opens
4. **Refresh Performance**: <100ms average load time
5. **Crash-Free Rate**: >99.9% for widget extensions

---

## Appendix: Quick Reference Code

### Basic Widget Structure

```swift
import WidgetKit
import SwiftUI

struct FlareRiskWidget: Widget {
    let kind: String = "FlareRiskWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: FlareRiskProvider()) { entry in
            FlareRiskWidgetView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("Flare Risk")
        .description("Monitor your current flare risk")
        .supportedFamilies([.systemSmall, .systemMedium,
                           .accessoryCircular, .accessoryRectangular])
    }
}
```

### Control Widget Structure (iOS 18+)

```swift
import WidgetKit
import AppIntents

struct QuickLogControl: ControlWidget {
    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: "QuickLogControl") {
            ControlWidgetButton(action: OpenQuickLogIntent()) {
                Label("Quick Log", systemImage: "pencil.circle.fill")
            }
        }
        .displayName("Quick Log")
        .description("Quickly log symptoms")
    }
}

struct OpenQuickLogIntent: AppIntent {
    static var title: LocalizedStringResource = "Open Quick Log"
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult {
        return .result()
    }
}
```

---

*Plan created: November 26, 2025*
*Author: Claude Code Assistant*
