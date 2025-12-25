# Rive Animation Integration Guide

## Overview

This guide explains how to integrate the Anky mascot Rive animations into InflamAI.

**Current Status:**
- Rive file: `InflamAI/Resources/Rive/anky_mascot.riv` (72KB)
- Available Rive animations: `idle`, `wave` (2 animations)
- Static image fallback: Asset catalog dino images for other states

## Strategy

**Rive animations** are used for interactive, high-engagement moments:
- `idle` - Default breathing animation (onboarding welcome, main dashboard)
- `wave` - Greeting animation (app launch, onboarding start)

**Static images** from `Assets.xcassets` are used elsewhere:
- Feature screens, tips, empty states, notifications, etc.
- Lighter weight, faster loading, no runtime overhead

## Setup Instructions

### 1. Add RiveRuntime Swift Package

In Xcode:
1. File → Add Package Dependencies
2. Enter URL: `https://github.com/rive-app/rive-ios`
3. Select version: **6.0.0** or later
4. Add to target: **InflamAI**

### 2. Add Rive File to Project

The file is already copied to `InflamAI/Resources/Rive/anky_mascot.riv`.

In Xcode:
1. Right-click on `InflamAI/Resources` folder
2. Add Files to "InflamAI"
3. Select the `Rive` folder
4. Ensure "Copy items if needed" is checked
5. Target: InflamAI

### 3. Enable Rive in Code

Edit `AnkyRiveView.swift`:

```swift
// Uncomment this import at the top:
import RiveRuntime
```

Then replace the placeholder `RiveMascotView` with the actual implementation (see the commented reference code in the file).

## File Structure

```
InflamAI/
├── Resources/
│   └── Rive/
│       └── anky_mascot.riv          # Rive animation file
├── Features/
│   └── Onboarding/
│       └── DuolingoStyle/
│           ├── AnkyRiveView.swift    # Rive wrapper (auto-fallback)
│           ├── AnkyAnimatedMascot.swift  # Canvas fallback
│           └── OnboardingRedesignFlow.swift  # Uses AnkyRiveView
```

## Available Assets

### Rive Animations (Interactive)

| State | Animation Name | Usage |
|-------|----------------|-------|
| idle | `idle` | Onboarding welcome, dashboard mascot |
| waving | `wave` | App launch greeting, onboarding start |

### Static Images (Asset Catalog)

| Asset Name | Use Case | Maps to State |
|------------|----------|---------------|
| `dino-happy` | Positive feedback, achievements | happy, celebrating |
| `dino-sad` | Empathy moments, flare logging | concerned, sympathetic |
| `dino-tired` | Rest reminders, fatigue tracking | sleeping, tired |
| `dino-sleeping` | Sleep tracking, night mode | sleeping |
| `dino-meditating` | Mindfulness, breathing exercises | thinking, calm |
| `dino-strong-mussel` | Exercise completion, strength | encouraging, proud |
| `dino-walking` | Activity tracking, movement | active |
| `dino-walking-fast` | Exercise in progress | exercising |
| `dino-medications` | Medication reminders | medication |
| `dino-hearth` | HealthKit, heart rate | health |
| `dino-spine-showing` | Body map, pain tracking | explaining |
| `dino-showing-whiteboard` | Tips, education | explaining, teaching |
| `dino-stop-hand` | Warnings, limits reached | concerned, stop |
| `dino-sweat-little-scared` | Anxiety, worry | anxious, nervous |
| `dino-stading-normal` | Default/neutral state | idle (static) |
| `dino-privacy` | Privacy settings, data | privacy |

## Usage in Code

### Rive Animation (for idle/waving only)

```swift
// Use in onboarding and high-engagement screens
AnkyRiveView(size: 200, state: .idle, showShadow: true)
AnkyRiveView(size: 200, state: .waving, showShadow: true)
```

### Static Image (for all other contexts)

```swift
// Simple image display
Image("dino-happy")
    .resizable()
    .aspectRatio(contentMode: .fit)
    .frame(width: 120, height: 120)

// With optional animation
Image("dino-celebrating")
    .resizable()
    .aspectRatio(contentMode: .fit)
    .frame(width: 100)
    .transition(.scale.combined(with: .opacity))
```

## Fallback Strategy

`AnkyRiveView` handles fallback automatically:
1. **Rive available + state is idle/waving** → Plays Rive animation
2. **Rive unavailable OR other states** → Falls back to Canvas animation (`AnkyAnimatedMascot`)

For non-interactive contexts, prefer static images directly for better performance.

## Troubleshooting

**"Failed to load Rive file"**
- Ensure `anky_mascot.riv` is added to the InflamAI target in Xcode
- Check file name matches exactly: `anky_mascot` (without .riv extension)
- Verify file is in `InflamAI/Resources/Rive/`

**Animation not playing**
- Check animation names: `idle`, `wave`
- View debug console for `🎬` prefixed messages

**Fallback showing instead of Rive**
- This is expected for states other than `idle` and `waving`
- For those 2 states, check Rive file is bundled correctly

---

## Asset Naming Reference

```
Assets.xcassets/
├── dino-happy 1.imageset/
├── dino-sad.imageset/
├── dino-tired 1.imageset/
├── dino-sleeping 1.imageset/
├── dino-meditating.imageset/
├── dino-strong-mussel.imageset/
├── dino-walking 1.imageset/
├── dino-walking-fast.imageset/
├── dino-medications.imageset/
├── dino-hearth 1.imageset/
├── dino-spine-showing.imageset/
├── dino-showing-whiteboard.imageset/
├── dino-stop-hand.imageset/
├── dino-sweat-little-scared.imageset/
├── dino-stading-normal.imageset/
└── dino-privacy.imageset/
```

---

*Last updated: December 2024*
