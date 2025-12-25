# Watch Target Restoration Plan

## Current State Analysis

### What Exists Now
| Component | Status | Location |
|-----------|--------|----------|
| **iOS App Target** | ✅ Working | `InflamAI` target |
| **Watch App Files** | ✅ Files exist | `InflamAI-AppleWatch Watch App/` |
| **Widget Files** | ✅ Files exist | `InflamAI/InflamAIWatchWidgets/` |
| **Watch Target** | ❌ Missing | Not in project.pbxproj |
| **Widget Target** | ❌ Missing | Not in project.pbxproj |
| **Embed Watch Phase** | ❌ Missing | Not in iOS target |

### Watch App Files Present (12 Swift files)
```
InflamAI-AppleWatch Watch App/
├── InflamAIWatchApp.swift          # @main entry point
├── Info.plist                        # ✅ Properly configured
├── InflamAI.entitlements
├── Assets.xcassets/
├── Services/
│   └── WatchConnectivityManager.swift
├── ViewModels/
│   ├── WatchFlareViewModel.swift
│   ├── WatchHealthViewModel.swift
│   ├── WatchMedicationViewModel.swift
│   └── WatchSymptomViewModel.swift
└── Views/
    ├── ContentView.swift
    ├── BiometricsView.swift
    ├── ExercisesWatchView.swift
    ├── FlareLogView.swift
    ├── MedicationTrackerView.swift
    └── QuickLogView.swift
```

### Widget Files Present (7 Swift files)
```
InflamAI/InflamAIWatchWidgets/
├── InflamAIWatchWidgetBundle.swift  # @main (watchOS only via #if)
├── WatchDataProvider.swift
├── Info.plist
├── InflamAIWatchWidget.entitlements
└── WatchWidgets/
    ├── WatchBASDAIWidget.swift
    ├── WatchFlareWidget.swift
    ├── WatchMedicationWidget.swift
    ├── WatchQuickStatsWidget.swift
    └── WatchStreakWidget.swift
```

---

## Risk Assessment

| Action | Risk Level | Impact if Failed |
|--------|------------|------------------|
| Add Watch target in Xcode | 🟢 LOW | Isolated, easy to undo |
| Add Widget target in Xcode | 🟢 LOW | Isolated, easy to undo |
| Add Embed Watch phase | 🟡 MEDIUM | iOS build may fail, easy fix |
| Edit project.pbxproj directly | 🔴 HIGH | Could break entire project |

**Recommendation**: Use Xcode UI for all target additions (safest approach)

---

## Phase 1: Preparation (No Risk)

### Step 1.1: Create Git Safety Checkpoint
```bash
git stash push -m "Pre-watch-target-restoration checkpoint"
# OR create a backup branch
git checkout -b watch-restoration-backup
git checkout might-work
```

### Step 1.2: Verify Files Are Complete
All required files are already in place:
- [x] Watch app entry point: `InflamAI-AppleWatch Watch App/InflamAIWatchApp.swift`
- [x] Watch Info.plist with correct `WKCompanionAppBundleIdentifier`
- [x] Widget bundle: `InflamAIWatchWidgetBundle.swift` with `#if os(watchOS)` guard
- [x] All ViewModels and Views present

**No file copying needed.**

---

## Phase 2: Add Watch App Target (Low Risk)

### Step 2.1: Open Xcode Project
```bash
open InflamAI.xcodeproj
```

### Step 2.2: Add Watch App Target via Xcode UI

1. **File → New → Target...**
2. Select **watchOS** tab
3. Choose **App** (NOT "App for Existing iOS App" - we have files already)
4. Configure:
   - **Product Name**: `InflamAI-AppleWatch`
   - **Bundle Identifier**: `com.inflamai.InflamAI.watchkitapp`
   - **Language**: Swift
   - **User Interface**: SwiftUI
   - **Include Tests**: No (uncheck)
   - **Embed in Companion App**: `InflamAI`
5. Click **Finish**

### Step 2.3: Delete Auto-Generated Files
Xcode will create template files. **DELETE THEM** (they conflict with our existing files):
- Delete the new `InflamAI-AppleWatch/` folder Xcode created
- We will point to our existing `InflamAI-AppleWatch Watch App/` folder

### Step 2.4: Add Existing Watch Files to Target

1. In Project Navigator, right-click `InflamAI-AppleWatch` target
2. **Add Files to "InflamAI"...**
3. Navigate to and select entire `InflamAI-AppleWatch Watch App/` folder
4. **CRITICAL Settings**:
   - [x] Copy items if needed: **UNCHECKED** (files already in place)
   - [x] Create folder references: **UNCHECKED**
   - [x] Add to targets: Select **ONLY** `InflamAI-AppleWatch` (NOT iOS target)
5. Click **Add**

### Step 2.5: Configure Watch Target Build Settings

In Xcode, select `InflamAI-AppleWatch` target → Build Settings:

| Setting | Value |
|---------|-------|
| `INFOPLIST_FILE` | `InflamAI-AppleWatch Watch App/Info.plist` |
| `CODE_SIGN_ENTITLEMENTS` | `InflamAI-AppleWatch Watch App/InflamAI.entitlements` |
| `PRODUCT_BUNDLE_IDENTIFIER` | `com.inflamai.InflamAI.watchkitapp` |
| `SDKROOT` | `watchos` |
| `TARGETED_DEVICE_FAMILY` | `4` (Watch) |
| `WATCHOS_DEPLOYMENT_TARGET` | `10.0` |
| `SWIFT_VERSION` | `5.0` |
| `ASSETCATALOG_COMPILER_APPICON_NAME` | `AppIcon` |

### Step 2.6: Verify Watch Target Builds

1. Select `InflamAI-AppleWatch` scheme
2. Select Apple Watch simulator
3. **Cmd + B** to build
4. Fix any errors before proceeding

---

## Phase 3: Add Widget Extension Target (Low Risk)

### Step 3.1: Add Widget Target via Xcode UI

1. **File → New → Target...**
2. Select **watchOS** tab
3. Choose **Widget Extension**
4. Configure:
   - **Product Name**: `InflamAIWatchWidgets`
   - **Bundle Identifier**: `com.inflamai.InflamAI.watchkitapp.widgets`
   - **Include Configuration App Intent**: No
   - **Embed in Watch App**: `InflamAI-AppleWatch`
5. Click **Finish**

### Step 3.2: Delete Auto-Generated Widget Files
Delete the template widget files Xcode created.

### Step 3.3: Add Existing Widget Files to Target

1. Right-click on project
2. **Add Files to "InflamAI"...**
3. Select entire `InflamAI/InflamAIWatchWidgets/` folder
4. **CRITICAL Settings**:
   - [x] Add to targets: Select **ONLY** `InflamAIWatchWidgets`
5. Click **Add**

### Step 3.4: Configure Widget Target Build Settings

| Setting | Value |
|---------|-------|
| `INFOPLIST_FILE` | `InflamAI/InflamAIWatchWidgets/Info.plist` |
| `CODE_SIGN_ENTITLEMENTS` | `InflamAI/InflamAIWatchWidgets/InflamAIWatchWidget.entitlements` |
| `PRODUCT_BUNDLE_IDENTIFIER` | `com.inflamai.InflamAI.watchkitapp.widgets` |
| `SDKROOT` | `watchos` |

### Step 3.5: Verify Widget Target Builds
1. Build the widget target independently
2. Fix any errors

---

## Phase 4: Configure iOS Target Integration (Medium Risk)

### Step 4.1: Add Watch Dependency to iOS Target

1. Select `InflamAI` target
2. Go to **General** tab
3. Scroll to **Frameworks, Libraries, and Embedded Content**
4. Ensure Watch app is listed (may happen automatically)

### Step 4.2: Verify Embed Watch Content Build Phase

1. Select `InflamAI` target
2. Go to **Build Phases** tab
3. Look for **"Embed Watch Content"** phase
4. If missing, add it:
   - Click **+** → **New Copy Files Phase**
   - Rename to "Embed Watch Content"
   - Set **Destination**: `Watch`
   - Add `InflamAI-AppleWatch.app`

### Step 4.3: Do NOT Modify iOS Source Files
The iOS app already has Watch connectivity code:
- `Core/Services/WatchSyncService.swift` (19KB)
- `Core/AppleWatchIntegration.swift` (22KB)
- `Core/AppleWatchIntegrationEngine.swift` (37KB)

**Leave these untouched.** They will work once the Watch target exists.

---

## Phase 5: Verification & Testing

### Step 5.1: Build All Targets
```
Scheme                    | Destination           | Expected Result
--------------------------|----------------------|----------------
InflamAI          | iPhone 15 Pro        | ✅ Success
InflamAI-AppleWatch     | Apple Watch Series 9 | ✅ Success
InflamAIWatchWidgets  | Apple Watch Series 9 | ✅ Success
```

### Step 5.2: Test on Simulators
1. Run iOS app on iPhone simulator
2. Run Watch app on paired Watch simulator
3. Verify they can communicate (check console for WatchConnectivity logs)

### Step 5.3: Verify Widget Appears
1. On Watch simulator, add widget to Smart Stack
2. Verify widgets render correctly

---

## Rollback Plan (If Something Goes Wrong)

### Option A: Git Reset
```bash
git checkout -- InflamAI.xcodeproj/project.pbxproj
```

### Option B: Restore from Backup
```bash
cp InflamAI.xcodeproj/project-backups/project.pbxproj InflamAI.xcodeproj/project.pbxproj
```

### Option C: Full Git Reset
```bash
git stash pop  # Restore pre-restoration state
```

---

## Reference: Nov 16 Working Configuration

The commit `8f867f5` had these key elements in project.pbxproj:

### Watch Target Definition
```
69BF6E8C2EC301540039E7C2 /* InflamAI.watch Watch App */ = {
    isa = PBXNativeTarget;
    buildPhases = (Sources, Frameworks, Resources);
    productType = "com.apple.product-type.application";
};
```

### Embed Watch Content Build Phase
```
69BF6E982EC301560039E7C2 /* Embed Watch Content */ = {
    isa = PBXCopyFilesBuildPhase;
    dstPath = Watch;
    dstSubfolderSpec = 16;
    files = (InflamAI.watch Watch App.app);
};
```

### Target Dependency
```
69BF6E962EC301560039E7C2 /* PBXTargetDependency */
```

Full reference available at: `recovered-watch-nov16/xcode-config/project.pbxproj.nov16-full`

---

## Summary: What This Plan Does NOT Do

| Action | Reason |
|--------|--------|
| ❌ Delete any existing files | Preserves all current work |
| ❌ Modify iOS app source code | iOS connectivity code already exists |
| ❌ Directly edit project.pbxproj | Too risky, use Xcode UI instead |
| ❌ Change Widget Swift files | Already properly guarded with `#if os(watchOS)` |
| ❌ Touch recovered-watch-nov16/ | Reference only, not for deployment |

---

## Estimated Time: 30-45 minutes

| Phase | Time |
|-------|------|
| Phase 1: Preparation | 5 min |
| Phase 2: Watch Target | 15 min |
| Phase 3: Widget Target | 10 min |
| Phase 4: iOS Integration | 5 min |
| Phase 5: Verification | 10 min |

---

## Ready to Execute?

This plan uses Xcode UI exclusively to minimize risk. All existing files remain untouched. The Watch and Widget files already exist in the correct locations - we're just telling Xcode about them.

**Next Step**: Open Xcode and follow Phase 2.
