# CRITICAL ISSUE: BASDAI Shows 0.0 in ML Model

## Problem
The ML model is receiving **BASDAI = 0.0** even though you entered BASDAI of 10!

## Evidence from Console Logs
```
Sample Day 1 features (raw → normalized):
  [0] age: 0.0 → 0.0
  [6] basdai: 0.0 → 0.0          ← SHOULD BE 10.0!
  [37] hrv: 42.31614 → 0.4136722  ← HealthKit works
  [38] resting_hr: 84.0 → 0.77377826
  [41] steps: 7785.0 → 0.67081034
  [55] sleep_hours: 7.9760942 → 0.68862987

🧠 [UnifiedNeuralEngine] Model Raw Output:
  probabilities[1] (flare): 0.49316406  ← ~49% because no symptom data!
```

## Why the Model Predicts ~47-50%
The model is **only seeing HealthKit biometric data**, not your symptom scores:
- ✅ Has: HRV, heart rate, steps, sleep
- ❌ Missing: BASDAI, BASFI, pain levels, morning stiffness, body map data
- ❌ Missing: Age, demographics

Without symptom severity data, the model defaults to **uncertain** (~50% chance).

## Root Cause Possibilities

### 1. SymptomLog Data Not Saved
**Most Likely**: When you enter BASDAI of 10, it's not being saved to Core Data properly.

Check in these files:
- `Features/CheckIn/DailyCheckInViewModel.swift:169` - where BASDAI is saved
- `Features/CheckIn/QuickLogViewModel.swift:106` - quick log BASDAI
- `Features/CheckIn/QuickSymptomLogView.swift:236` - quick symptom BASDAI

Possible issues:
- Context.save() not being called
- Data being saved to wrong entity
- Timestamp mismatch (saved for different day than being queried)

### 2. Feature Extraction Date Mismatch
**Possible**: The feature extractor is looking for the wrong date.

In `FeatureExtractor.swift:451`:
```swift
let symptomLog = fetchSymptomLog(for: date, context: context)
```

If `date` doesn't match when you entered the data, it will return `nil` and all values will be 0.0.

### 3. UserProfile Not Set
**Also Likely**: Age shows 0.0, which means:
```swift
let userProfile = fetchUserProfile(context: context)
```
Returns `nil` or has no birthDate set.

## Diagnostic Steps

### Step 1: Add Core Data Logging
Add to `FeatureExtractor.swift` after line 451:

```swift
let symptomLog = fetchSymptomLog(for: date, context: context)

#if DEBUG
if let log = symptomLog {
    print("   ✅ [SymptomLog Found] Date: \(date)")
    print("      BASDAI: \(log.basdaiScore)")
    print("      Pain: \(log.painAverage24h)")
    print("      Stiffness: \(log.morningStiffnessMinutes) mins")
} else {
    print("   ❌ [NO SymptomLog] for date: \(date)")
    print("      Querying: \(Calendar.current.startOfDay(for: date))")
}
#endif
```

### Step 2: Check fetchSymptomLog Implementation
Find the `fetchSymptomLog` function and verify:
1. Date comparison logic
2. Predicate correctness
3. Sort descriptors

### Step 3: Verify Data Entry
After entering BASDAI of 10:
1. Immediately check Core Data in Xcode (Window → Organizer → App Data)
2. Or add logging in DailyCheckInViewModel:
```swift
log.basdaiScore = score
print("💾 Saving BASDAI: \(score) for date: \(log.timestamp ?? Date())")
try context.save()
print("✅ Saved successfully")
```

## Expected vs Actual

### Expected (with BASDAI 10, severe symptoms):
```
[6] basdai: 10.0 → 0.9  (normalized)
Model output: ~0.85 (85% flare risk)
```

### Actual (current):
```
[6] basdai: 0.0 → 0.0
Model output: 0.49 (49% flare risk - uncertain)
```

## Solution Path

1. **Immediate**: Add the diagnostic logging above
2. **Run app**: Enter BASDAI of 10 again
3. **Check console**: Look for "SymptomLog Found" or "NO SymptomLog"
4. **If "NO SymptomLog"**: Problem is in data saving (DailyCheckInViewModel)
5. **If "SymptomLog Found but basdai=0.0"**: Problem is in how BASDAI is stored

## Files to Investigate

Priority order:
1. `FeatureExtractor.swift:451-615` - Feature extraction logic
2. `FeatureExtractor.swift` - `fetchSymptomLog()` function
3. `DailyCheckInViewModel.swift:169` - Where BASDAI is saved
4. `SymptomLog+CoreDataProperties.swift` - Entity definition

## Next Steps

With the diagnostic logging I added earlier, you should now see:
```
📋 [Clinical Assessment] Core Data Extraction:
   ⚠️ NO SymptomLog found for this date
```
or:
```
📋 [Clinical Assessment] Core Data Extraction:
   ❌ BASDAI: 0.0/10 | Source: CORE_DATA (SymptomLog.basdaiScore)
```

This will tell us if:
- Data isn't being saved at all (NO SymptomLog)
- Data is saved but BASDAI field is 0.0 (wrong field or data corruption)

---

**Critical**: The ML model CANNOT predict flares without symptom data. It's currently blind to your actual condition!
