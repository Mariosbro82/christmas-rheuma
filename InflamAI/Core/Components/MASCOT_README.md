# Anky the Ankylosaurus - Official Mascot

Anky is the friendly Ankylosaurus mascot for InflamAI, helping users manage ankylosing spondylitis with encouragement, tips, and support.

## Files

### AnkylosaurusMascot.swift
The core mascot view with 3D-style rendering using SwiftUI Canvas.

**Features:**
- 3D-rendered Ankylosaurus with body, tail club, armor plates, and legs
- Multiple expressions: `.happy`, `.waving`, `.excited`, `.encouraging`
- Built-in animations: `.bouncing()`, `.waving()`
- Customizable size (default 200 points)
- Subtle 3D rotation effect

**Usage:**
```swift
AnkylosaurusMascot(expression: .happy, size: 200)
    .bouncing()
```

**Expressions:**
- **happy** - Standard friendly smile
- **waving** - Welcoming smile for greetings
- **excited** - Big excited smile for celebrations
- **encouraging** - Supportive smile for difficult moments

---

### MascotHelper.swift
Contextual helper components that integrate Anky throughout the app.

## Components

### 1. MascotHelper
Full mascot card with speech bubble for contextual tips and encouragement.

**Contexts:**
- `welcome` - First-time greeting
- `firstPainLog` - Encouragement after first symptom entry
- `flareDetected` - Support during flare-ups
- `improvement` - Celebration of progress
- `medicationReminder` - Medication adherence prompts
- `exerciseTime` - Exercise encouragement
- `dataInsight` - Announcing pattern discoveries
- `celebration` - Major milestone achievements
- `sympathy` - Empathy during difficult times
- `encouragement` - General positive reinforcement

**Example:**
```swift
MascotHelper(
    context: .flareDetected,
    size: 120,
    showDismissButton: true,
    onDismiss: { /* dismiss logic */ }
)
```

---

### 2. MascotTipCard
Compact inline tip card with icon and message.

**Example:**
```swift
MascotTipCard(
    icon: AssetsManager.Symbols.lightbulb,
    title: "Daily Tracking Tip",
    message: "Log symptoms at the same time each day for more accurate patterns!",
    color: AssetsManager.Colors.primary
)
```

**Use Cases:**
- Settings screen tips
- Feature explanations
- Quick guidance cards
- Onboarding hints

---

### 3. MascotBanner
Top banner for important messages and announcements.

**Types:**
- `.info` - General information (blue)
- `.success` - Positive achievements (green)
- `.warning` - Important alerts (orange)
- `.tip` - Helpful suggestions (primary color)

**Example:**
```swift
MascotBanner(
    type: .success,
    message: "7-day streak! Keep logging symptoms daily.",
    onDismiss: { /* dismiss */ }
)
```

**Use Cases:**
- Streaks and achievements
- High BASDAI warnings
- Data quality tips
- Feature announcements

---

### 4. MascotEmptyState
Empty state view with encouragement to take action.

**Example:**
```swift
MascotEmptyState(
    icon: AssetsManager.Symbols.chart,
    title: "No Data Yet",
    message: "Start logging your symptoms to see trends and insights!",
    actionTitle: "Log Symptoms",
    action: { /* navigate to log */ }
)
```

**Use Cases:**
- No trends data yet
- No medications added
- No exercise logs
- No flare history

---

### 5. MascotCelebration
Full-screen celebration overlay for major achievements.

**Example:**
```swift
MascotCelebration(
    title: "First Week Complete!",
    message: "You've logged symptoms for 7 days straight. This data will help identify patterns!",
    onDismiss: { /* dismiss */ }
)
```

**Use Cases:**
- First week milestone
- 30-day streak
- First complete BASDAI
- First PDF export
- Pattern discovered

---

## Integration Guide

### 1. Onboarding Flow
Already integrated! Anky appears throughout OnboardingFlow.swift with contextual messages.

**Recommendation:** Use `MascotHelper` with `.welcome` context on first page.

---

### 2. Home Screen
Add a daily tip or encouragement card:

```swift
// In HomeView.swift
MascotTipCard(
    icon: AssetsManager.Symbols.calendar,
    title: "Morning Check-In",
    message: "How are you feeling today?",
    color: AssetsManager.Colors.primary
)
.padding()
```

---

### 3. First Symptom Log
Show encouragement after first log:

```swift
// In QuickSymptomLogView.swift or similar
@State private var showFirstLogCelebration = false

// After saving first log:
if isFirstLog {
    showFirstLogCelebration = true
}

.sheet(isPresented: $showFirstLogCelebration) {
    MascotHelper(
        context: .firstPainLog,
        onDismiss: {
            showFirstLogCelebration = false
        }
    )
}
```

---

### 4. Flare Detection
Show sympathy and guidance:

```swift
// In AIInsightsView.swift or FlareDetection logic
if riskScore > 75 {
    MascotBanner(
        type: .warning,
        message: "High flare risk detected. Consider rest and contact your doctor if symptoms worsen.",
        onDismiss: { /* dismiss */ }
    )
}
```

---

### 5. Empty States
Replace "Text('No data')" placeholders:

```swift
// Instead of:
if logs.isEmpty {
    Text("No symptom logs yet")
}

// Use:
if logs.isEmpty {
    MascotEmptyState(
        icon: AssetsManager.Symbols.chart,
        title: "Start Your Journey",
        message: "Log your first symptoms to begin tracking patterns!",
        actionTitle: "Log Now",
        action: { showQuickLog = true }
    )
}
```

---

### 6. Achievements
Celebrate milestones:

```swift
// After completing 7-day streak:
MascotCelebration(
    title: "Week Warrior!",
    message: "7 days of consistent tracking. You're building valuable data!",
    onDismiss: { /* dismiss */ }
)
```

---

## Design Guidelines

### Colors
Anky uses a soft teal color palette:
- Body: `Color(red: 0.4, green: 0.7, blue: 0.6)`
- Darker areas: `Color(red: 0.3, green: 0.6, blue: 0.5)`
- Highlights: `Color(red: 0.5, green: 0.8, blue: 0.7)`

### Tone
- **Friendly & Supportive** - Never clinical or cold
- **Empathetic** - Acknowledges difficulty of AS
- **Encouraging** - Focuses on progress, not perfection
- **Informative** - Provides value, not just decoration

### When to Use

**✅ DO USE:**
- Empty states
- First-time experiences
- Achievements and milestones
- Difficult moments (flares, high pain)
- Educational tips
- Data insights

**❌ DON'T USE:**
- Critical errors (use standard alerts)
- Medical decisions (Anky is supportive, not medical)
- Every single screen (avoid mascot fatigue)

---

## Animation Performance

- Mascot uses SwiftUI Canvas for high performance
- Bouncing animation: `.easeInOut(duration: 0.6)` repeat forever
- 3D rotation: `.easeInOut(duration: 2)` subtle effect
- All animations are optimized for 60fps

---

## Accessibility

- All mascot components support VoiceOver
- Speech bubble text is screen-reader friendly
- Action buttons have clear labels
- Color is never the only indicator (icons + text)

---

## Future Enhancements

**Potential additions:**
- Seasonal variations (winter coat, summer hat)
- More expressions (confused, sleeping, thinking)
- Interactive animations (tap to wave)
- Voice-over integration (spoken encouragement)
- Customizable accessories
- Different poses (standing, walking, stretching)

---

## Medical Disclaimer

Anky is a supportive companion, **NOT a medical advisor**. All medical decisions should be made in consultation with healthcare professionals. The mascot provides encouragement and tips, but should never replace clinical judgment.
