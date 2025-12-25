# InflamAI Widget Setup Instructions

This guide walks you through adding the widget targets to your Xcode project.

## Prerequisites

- Xcode 15.0+
- iOS 17.0+ deployment target
- Apple Developer account (for App Groups capability)

---

## Step 1: Add Widget Extension Target

1. **Open your project in Xcode**
   ```bash
   open InflamAI.xcodeproj
   ```

2. **Add Widget Extension target**
   - File > New > Target
   - Search for "Widget Extension"
   - Click Next
   - Product Name: `InflamAIWidgetExtension`
   - Team: Select your team
   - Bundle Identifier: `com.spinalytics.InflamAIWidgetExtension`
   - **Uncheck** "Include Configuration App Intent" (we have our own)
   - **Uncheck** "Include Live Activity"
   - Click Finish

3. **Delete the auto-generated files** from the new target (we have custom ones):
   - Delete `InflamAIWidgetExtension.swift` (the auto-generated one)
   - Delete any auto-generated widget files

4. **Add our widget files to the target**
   - Select all files in `InflamAI/InflamAIWidgetExtension/`
   - In File Inspector, check "InflamAIWidgetExtension" under Target Membership
   - Also add `Shared/WidgetShared/` files to both main app AND widget extension targets

---

## Step 2: Configure App Groups

1. **Main App Target**
   - Select your main app target
   - Go to Signing & Capabilities
   - Click "+ Capability"
   - Add "App Groups"
   - Click "+" and add: `group.com.spinalytics.shared`

2. **Widget Extension Target**
   - Select `InflamAIWidgetExtension` target
   - Go to Signing & Capabilities
   - Click "+ Capability"
   - Add "App Groups"
   - Add the same group: `group.com.spinalytics.shared`

---

## Step 3: Add iOS 18 Control Widget Extension (Optional)

> **Note**: Control Widgets require iOS 18.0+

1. **Add Control Widget Extension target**
   - File > New > Target
   - Search for "Widget Extension"
   - Product Name: `InflamAIControlExtension`
   - Bundle Identifier: `com.spinalytics.InflamAIControlExtension`
   - Deployment Target: iOS 18.0

2. **Add files to target**
   - Add all files from `InflamAI/InflamAIControlExtension/`
   - Add to Target Membership: InflamAIControlExtension

3. **Configure App Groups** (same as Step 2)

---

## Step 4: Add Apple Watch Widget Extension (Optional)

> **Note**: Requires watchOS 10.0+ and a watchOS app target

1. **Add Watch App target** (if not already present)
   - File > New > Target
   - Search for "watchOS App"
   - Product Name: `InflamAIWatch`

2. **Add Watch Widget Extension**
   - File > New > Target
   - Search for "Widget Extension"
   - Product Name: `InflamAIWatchWidgets`
   - Platform: watchOS
   - Bundle Identifier: `com.spinalytics.watch.widgets`

3. **Add files to target**
   - Add all files from `InflamAI/InflamAIWatchWidgets/`
   - Add to Target Membership: InflamAIWatchWidgets

4. **Configure App Groups** for watch (same identifier)

---

## Step 5: Configure Info.plist

### Widget Extension Info.plist

Add to `InflamAIWidgetExtension/Info.plist`:

```xml
<key>NSExtension</key>
<dict>
    <key>NSExtensionPointIdentifier</key>
    <string>com.apple.widgetkit-extension</string>
</dict>
```

### Main App Info.plist

Add URL scheme for deep linking:

```xml
<key>CFBundleURLTypes</key>
<array>
    <dict>
        <key>CFBundleURLSchemes</key>
        <array>
            <string>spinalytics</string>
        </array>
        <key>CFBundleURLName</key>
        <string>com.spinalytics</string>
    </dict>
</array>
```

---

## Step 6: Update Main App for Deep Links

Add deep link handling to your main app entry point:

```swift
// In InflamAIApp.swift
import SwiftUI

@main
struct InflamAIApp: App {
    @StateObject private var widgetNavigation = WidgetNavigationState.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .handleWidgetDeepLinks() // Add this modifier
                .environmentObject(widgetNavigation)
        }
    }
}
```

Handle navigation in your ContentView:

```swift
struct ContentView: View {
    @EnvironmentObject var widgetNavigation: WidgetNavigationState

    var body: some View {
        TabView {
            // Your tabs...
        }
        .onChange(of: widgetNavigation.destination) { destination in
            if let destination = destination {
                navigateTo(destination)
            }
        }
        .sheet(isPresented: $widgetNavigation.showQuickLogSheet) {
            QuickLogView()
        }
        .sheet(isPresented: $widgetNavigation.showSOSFlareSheet) {
            SOSFlareView()
        }
        .sheet(isPresented: $widgetNavigation.showMedicationSheet) {
            MedicationLogView()
        }
        .sheet(isPresented: $widgetNavigation.showExerciseSheet) {
            ExerciseView()
        }
    }

    private func navigateTo(_ destination: WidgetDeepLink) {
        // Handle tab/view navigation based on destination
        switch destination {
        case .trends:
            // Navigate to trends tab
            break
        case .flare:
            // Navigate to flare details
            break
        case .basdai:
            // Navigate to BASDAI view
            break
        case .dashboard:
            // Navigate to home/dashboard
            break
        default:
            break
        }
    }
}
```

---

## Step 7: Update Data for Widgets

Call `WidgetDataWriter` methods whenever relevant data changes in your app:

```swift
// Example: After BASDAI assessment
let score = BASDAICalculator.calculate(answers: answers)
let interpretation = BASDAICalculator.interpretation(score: score)

WidgetDataWriter.shared.updateBASDAI(
    score: score,
    category: interpretation.category,
    trend: calculateTrend() // "improving", "stable", or "worsening"
)

// Example: After flare prediction updates
let prediction = UnifiedNeuralEngine.shared.currentPrediction
WidgetDataWriter.shared.updateFlareRisk(
    percentage: Int(prediction.riskPercentage * 100),
    level: prediction.riskLevel.rawValue,
    factors: prediction.topFactors.map { $0.name }
)

// Example: After logging symptoms
WidgetDataWriter.shared.updateStreak(days: calculateStreak())
WidgetDataWriter.shared.updateTodaySummary(
    painEntries: todaysPainEntries.count,
    assessments: todaysAssessments.count,
    hasLogged: true,
    hasActiveFlare: hasActiveFlare()
)
```

---

## Step 8: Build and Test

1. **Select Widget Extension scheme**
   - Product > Scheme > InflamAIWidgetExtension

2. **Build and Run**
   - Select iPhone simulator
   - Run the widget extension
   - Add widget from home screen widget gallery

3. **Test all sizes**
   - Small, Medium, Large widgets
   - Lock screen widgets (circular, rectangular, inline)

4. **Test deep links**
   - Tap on widgets to verify they open the correct app screens

---

## File Structure Summary

After setup, your project should have:

```
InflamAI/
├── InflamAI/                    # Main app
│   └── Shared/
│       └── WidgetShared/               # Shared code (both targets)
│           ├── AppGroupConfig.swift
│           ├── WidgetDataProvider.swift
│           └── WidgetDeepLinkHandler.swift
│
├── InflamAIWidgetExtension/         # iOS Widget Extension
│   ├── InflamAIWidgetBundle.swift
│   ├── Widgets/
│   ├── Views/
│   ├── Providers/
│   ├── Models/
│   └── Intents/
│
├── InflamAIControlExtension/        # iOS 18 Control Widgets
│   ├── InflamAIControlBundle.swift
│   └── Controls/
│
└── InflamAIWatchWidgets/            # watchOS Widgets
    ├── InflamAIWatchWidgetBundle.swift
    ├── WatchWidgets/
    └── WatchDataProvider.swift
```

---

## Troubleshooting

### Widget Not Appearing in Gallery
- Ensure widget extension is signed with same team as main app
- Check that the extension's Info.plist has correct NSExtensionPointIdentifier
- Clean build folder (Cmd + Shift + K) and rebuild

### Data Not Syncing to Widget
- Verify both targets have same App Group capability
- Check that `group.com.spinalytics.shared` matches exactly
- Ensure `WidgetDataWriter` is called after data changes

### Deep Links Not Working
- Verify URL scheme is registered in main app's Info.plist
- Check that `handleWidgetDeepLinks()` modifier is applied
- Test URL manually: `open "spinalytics://widget/quicklog"`

### Widget Shows Placeholder Data
- Ensure main app has written data to shared UserDefaults
- Check that widget provider is reading from correct keys
- Verify App Group container is accessible

---

## Summary

You've now created:

| Widget Type | Platform | Sizes |
|------------|----------|-------|
| Flare Risk | iOS | Small, Medium, Lock Screen |
| BASDAI Score | iOS | Small, Medium, Lock Screen |
| Logging Streak | iOS | Small, Lock Screen |
| Medications | iOS | Small, Medium, Lock Screen |
| Daily Dashboard | iOS | Medium, Large |
| Quick Log Control | iOS 18+ | Lock Screen Button |
| SOS Flare Control | iOS 18+ | Lock Screen Button |
| Medication Control | iOS 18+ | Lock Screen Button |
| Exercise Control | iOS 18+ | Lock Screen Button |
| Watch Flare Risk | watchOS | Circular, Rectangular, Corner |
| Watch BASDAI | watchOS | Circular, Rectangular, Corner |
| Watch Medications | watchOS | Circular, Rectangular |
| Watch Streak | watchOS | Circular, Rectangular, Corner |
| Watch Quick Stats | watchOS | Rectangular |

**Total: 14 widgets across 3 platforms!**
