# Anki Mascot - Rive Implementation Specification

> **Implementation Status (December 2024)**
>
> | Animation | Status | File |
> |-----------|--------|------|
> | `idle` | ✅ Implemented | `anky_mascot.riv` |
> | `wave` | ✅ Implemented | `anky_mascot.riv` |
> | Other states | 🖼️ Static images | `Assets.xcassets/dino-*.imageset` |
>
> The Rive file contains 2 core animations. Other mascot states use static images from the asset catalog. This document serves as reference for the implemented animations and future expansion.

---

## Character Overview
**Name:** Anki
**Type:** Friendly dinosaur mascot for health tracking app (Ankylosing Spondylitis)
**Personality:** Warm, supportive, playful, empathetic
**Style:** Soft, rounded, approachable (similar to Duolingo Duo)

---

## ARTBOARD SETUP

```
Artboard: "Anki"
Size: 512 x 512 px (square for universal scaling)
Background: Transparent
Origin: Center-bottom (Anki stands on ground)
```

---

## BONE HIERARCHY (Skeleton Setup)

```
root (center of body, hip level)
├── body_bone (torso)
│   ├── head_bone
│   │   ├── left_eye_bone
│   │   ├── right_eye_bone
│   │   ├── left_eyebrow_bone (optional)
│   │   ├── right_eyebrow_bone (optional)
│   │   └── mouth_bone
│   ├── spine_ridge_1_bone
│   ├── spine_ridge_2_bone
│   ├── spine_ridge_3_bone
│   ├── left_arm_upper_bone
│   │   └── left_arm_lower_bone (forearm/hand)
│   ├── right_arm_upper_bone
│   │   └── right_arm_lower_bone (forearm/hand)
│   └── tail_bone
│       └── tail_tip_bone (optional)
├── left_leg_bone
│   └── left_foot_bone
└── right_leg_bone
    └── right_foot_bone
```

**Binding Instructions:**
- Each shape group must be bound to its corresponding bone
- Use mesh deformation for body and head (smoother squash/stretch)
- Arms and legs can use simple bone transforms

---

## STATE MACHINE: "AnkiController"

### Inputs (from Swift/App)

| Input Name | Type | Purpose |
|------------|------|---------|
| `isHappy` | Boolean | Sustained happy mood state |
| `isSad` | Boolean | Sustained empathy/support state |
| `isThinking` | Boolean | Loading/processing state |
| `triggerWaveHello` | Trigger | Fire once: greeting animation |
| `triggerWaveGoodbye` | Trigger | Fire once: farewell animation |
| `triggerCelebrate` | Trigger | Fire once: achievement/milestone |
| `triggerEncourage` | Trigger | Fire once: gentle nudge |
| `triggerNod` | Trigger | Fire once: confirmation |
| `triggerBounce` | Trigger | Fire once: micro-interaction feedback |
| `painLevel` | Number (0-10) | Affects posture/expression subtly |

### States

```
Entry → idle

idle (default)
├── on triggerWaveHello → wave_hello → idle
├── on triggerWaveGoodbye → wave_goodbye → idle
├── on triggerCelebrate → celebrate → idle
├── on triggerEncourage → encourage → idle
├── on triggerNod → nod → idle
├── on triggerBounce → bounce → idle
├── on isHappy == true → happy_idle (loop)
├── on isSad == true → sad_idle (loop)
├── on isThinking == true → thinking (loop)
└── stays in idle (loops forever)

happy_idle
├── on isHappy == false → idle
├── on triggerCelebrate → celebrate → happy_idle
└── loops with extra energy

sad_idle
├── on isSad == false → idle
├── on triggerEncourage → encourage → sad_idle
└── loops with slower, softer movement

thinking
├── on isThinking == false → idle
└── loops with curiosity animation
```

### Transitions

| From | To | Duration | Easing |
|------|----|----------|--------|
| idle → wave_hello | 0.1s | ease-out |
| wave_hello → idle | 0.2s | ease-in-out |
| idle → celebrate | 0.1s | ease-out |
| celebrate → idle | 0.3s | ease-in-out |
| idle → sad_idle | 0.5s | ease-in-out (slow, gentle) |
| sad_idle → idle | 0.4s | ease-in-out |
| idle → happy_idle | 0.2s | ease-out |
| any → any (default) | 0.15s | ease-in-out |

---

## ANIMATION SPECIFICATIONS

### 1. idle (Priority: CRITICAL)
**Duration:** 3000ms (3 seconds)
**Loop:** Yes (seamless)
**Purpose:** Default breathing state - character feels ALIVE

```
Timeline Keyframes:

body_bone.y:
  0ms: 0
  1500ms: -3px (slight rise - inhale)
  3000ms: 0 (return - exhale)
  Easing: cubic-bezier(0.4, 0, 0.6, 1) (smooth sine)

head_bone.rotation:
  0ms: 0°
  750ms: 1° (micro tilt right)
  2250ms: -0.5° (micro tilt left)
  3000ms: 0°
  Easing: ease-in-out

spine_ridge_1_bone.rotation (top ridge):
  0ms: 0°
  1000ms: 2°
  2000ms: -1°
  3000ms: 0°

spine_ridge_2_bone.rotation (offset timing):
  200ms: 0°
  1200ms: 2°
  2200ms: -1°
  3200ms: 0°

spine_ridge_3_bone.rotation (offset timing):
  400ms: 0°
  1400ms: 2°
  2400ms: -1°
  3400ms: 0°

left_arm_upper_bone.rotation:
  0ms: 0°
  1500ms: 2°
  3000ms: 0°

right_arm_upper_bone.rotation:
  0ms: 0°
  1500ms: -2°
  3000ms: 0°

tail_bone.rotation:
  0ms: 0°
  1000ms: 5°
  2000ms: -3°
  3000ms: 0°
  Easing: ease-in-out

BLINK (nested timeline or blend):
  Every ~4000ms (randomized 3000-5000ms):
  left_eye + right_eye scale.y:
    0ms: 1
    50ms: 0.1 (closed)
    150ms: 1 (open)
```

**Quality Checklist:**
- [ ] Movement is subtle (2-5px, 1-3°)
- [ ] No sudden stops (all easing applied)
- [ ] Spine ridges have wave delay (creates organic flow)
- [ ] Blink adds life without distraction
- [ ] Loop point is seamless (end = start)

---

### 2. wave_hello
**Duration:** 1500ms
**Loop:** No
**Purpose:** Friendly greeting when app opens

```
Timeline Keyframes:

right_arm_upper_bone.rotation:
  0ms: 0° (arm at rest)
  200ms: -45° (arm raises)
  Easing: ease-out

right_arm_lower_bone.rotation:
  200ms: 0°
  300ms: -30° (forearm up, hand visible)

right_arm_lower_bone.rotation (wave motion):
  400ms: -30°
  550ms: -50° (wave out)
  700ms: -30° (wave in)
  850ms: -50° (wave out)
  1000ms: -30° (wave in)
  Easing: ease-in-out (bouncy feel)

right_arm_upper_bone.rotation (return):
  1200ms: -45°
  1500ms: 0°
  Easing: ease-in

body_bone.rotation:
  0ms: 0°
  300ms: -3° (slight lean toward wave)
  1200ms: -3°
  1500ms: 0°

head_bone.rotation:
  0ms: 0°
  300ms: 5° (look toward hand)
  1200ms: 5°
  1500ms: 0°

mouth_bone (or mouth shape):
  0ms: neutral
  200ms: smile (scale or morph)
  1300ms: smile
  1500ms: neutral

left_eye + right_eye:
  200ms: normal
  300ms: happy squint (scale.y: 0.7)
  1200ms: happy squint
  1400ms: normal
```

---

### 3. wave_goodbye
**Duration:** 1800ms
**Loop:** No
**Purpose:** Farewell when app closes/backgrounds

```
Similar to wave_hello but:
- Wave is slower, more deliberate
- Expression is slightly sadder (we'll miss you)
- Body leans back slightly at end
- Optional: small droop at finish

right_arm wave sequence: same pattern
Body at end:
  1500ms: body_bone.y = 0
  1800ms: body_bone.y = 2px (slight settle/droop)

head at end:
  1500ms: head_bone.rotation = 0°
  1800ms: head_bone.rotation = -2° (slight down look)
```

---

### 4. celebrate
**Duration:** 2000ms
**Loop:** No
**Purpose:** Achievement unlocked, milestone reached, check-in complete

```
Timeline Keyframes:

JUMP ANTICIPATION (squash):
  0ms: body_bone.y = 0, body_bone.scaleY = 1
  150ms: body_bone.y = 5px, body_bone.scaleY = 0.9 (squash down)
  Easing: ease-in

JUMP UP:
  150ms: values above
  400ms: body_bone.y = -30px, body_bone.scaleY = 1.1 (stretch up)
  Easing: ease-out (fast takeoff)

ARMS UP:
  150ms: left/right_arm_upper_bone.rotation = 0°
  400ms: both = -120° (arms up in V)
  Easing: ease-out

PEAK (hang time):
  400ms: body_bone.y = -30px
  500ms: body_bone.y = -28px (tiny float)
  Easing: linear

LAND (squash):
  500ms: body_bone.y = -28px, scaleY = 1.1
  650ms: body_bone.y = 5px, scaleY = 0.85 (impact squash)
  Easing: ease-in

RECOVER:
  650ms: squash position
  900ms: body_bone.y = -5px, scaleY = 1.05 (overshoot)
  1100ms: body_bone.y = 0, scaleY = 1
  Easing: ease-out

ARMS DOWN:
  650ms: arms up
  1000ms: arms at 0°
  Easing: ease-in-out

SPINE RIDGES (excited wiggle):
  400ms-1200ms: oscillate ±5° rapidly (100ms per cycle)

EXPRESSION:
  150ms: eyes wide (scale 1.2)
  400ms: eyes happy squint, mouth big smile
  1000ms: maintain happy
  1500ms: return to neutral

OPTIONAL - Particles:
  400ms: emit confetti/sparkles at peak
  1500ms: particles fade
```

---

### 5. sad_idle (Empathy Loop)
**Duration:** 3500ms
**Loop:** Yes
**Purpose:** User reported high pain/flare - Anki shows support

```
Timeline Keyframes:

POSTURE (slumped):
  All keyframes maintain slightly drooped posture

body_bone:
  Constant: y = 3px (lower stance)
  0ms: rotation = 2° (slight forward lean)
  1750ms: rotation = 3°
  3500ms: rotation = 2°

head_bone:
  Constant: rotation = -5° (looking down slightly)
  Subtle nod: 0ms → 1750ms → 3500ms: -5° → -7° → -5°

left_arm + right_arm:
  Held closer to body
  rotation offset: +10° inward
  Minimal sway (1° instead of 2°)

tail_bone:
  Slower movement (half speed of normal idle)
  rotation: 0° → 2° → 0° (less energetic)

EXPRESSION:
  Eyes: slightly drooped (rotation -3°)
  Mouth: gentle, closed (not frowning - supportive, not sad)
  Eyebrows (if available): inner raise (concerned look)

BREATHING:
  Same as idle but 30% slower
  Feels like a gentle, calming presence
```

---

### 6. encourage
**Duration:** 2000ms
**Loop:** No
**Purpose:** Gentle nudge - missed check-in, motivational moment

```
Timeline Keyframes:

REACH OUT:
  0ms: right_arm rest
  300ms: right_arm_upper = -30° (reaching toward viewer)
  300ms: right_arm_lower = -20°

HEAD TILT:
  0ms: head = 0°
  400ms: head = 8° (curious/caring tilt)

BODY LEAN:
  0ms: body = 0°
  400ms: body = -3° (lean toward viewer)

HOLD:
  400ms-1200ms: maintain caring pose

EXPRESSION:
  200ms: eyes soften (slight scale down to 0.95)
  200ms: eyebrows raise slightly
  300ms: gentle smile

RETURN:
  1200ms: start returning to neutral
  2000ms: all at rest

SPINE RIDGES:
  Gentle wave during hold (400-1200ms)
  Slower, calmer than normal
```

---

### 7. thinking
**Duration:** 2500ms
**Loop:** Yes
**Purpose:** Loading state, processing data

```
Timeline Keyframes:

HEAD:
  0ms: rotation = 0°
  625ms: rotation = 10° (tilt right)
  1250ms: rotation = -5° (tilt left)
  1875ms: rotation = 8°
  2500ms: rotation = 0°

EYES:
  Looking up-right (thinking pose)
  0ms: normal position
  300ms: y offset = -3px, x offset = 2px
  maintain during loop

  Optional: eyes drift slowly in circle

ONE ARM (optional chin touch):
  right_arm_upper: -20° (arm raised)
  right_arm_lower: -60° (hand near face)

BODY:
  Subtle breathing only
  Slower than normal (contemplative)

EXPRESSION:
  Mouth: slight purse or neutral
  No smile (focused, not happy or sad)
```

---

### 8. nod
**Duration:** 800ms
**Loop:** No
**Purpose:** Quick confirmation, acknowledgment

```
head_bone.rotation (up-down nod):
  0ms: 0°
  150ms: 8° (down)
  300ms: -3° (up overshoot)
  500ms: 5° (down again, smaller)
  800ms: 0°
  Easing: ease-out

Optional eye close on down nod:
  150ms: eyes scale.y = 0.7
  300ms: eyes scale.y = 1
```

---

### 9. bounce (Micro-interaction)
**Duration:** 500ms
**Loop:** No
**Purpose:** Button tap feedback, playful response

```
body_bone:
  0ms: y = 0, scaleY = 1
  100ms: y = 2px, scaleY = 0.95 (tiny squash)
  250ms: y = -8px, scaleY = 1.05 (small hop)
  400ms: y = 1px, scaleY = 0.98
  500ms: y = 0, scaleY = 1
  Easing: ease-out
```

---

### 10. happy_idle
**Duration:** 2500ms
**Loop:** Yes
**Purpose:** Sustained happy mood - more energetic idle

```
Same as idle but:
- All movements 20% larger amplitude
- Timing 15% faster
- Tail wags more (±8° instead of ±3°)
- Occasional extra bounce added
- Eyes have slight happy squint (scale.y = 0.9)
- Mouth in subtle smile
```

---

## BLENDING & LAYERS

### Additive Layers (blend on top of current state)

| Layer | Purpose | Blend Mode |
|-------|---------|------------|
| blink | Random blinks | Additive |
| breath_variation | Slight randomness to breathing | Additive |
| pain_posture | Subtle slump based on painLevel input | Additive (0-100% based on painLevel) |

### Pain Level Influence
```
When painLevel input changes (0-10):

painLevel 0-3:
  No modification

painLevel 4-6:
  body_bone.y += 2px (slight lower)
  head_bone.rotation += -2° (slight droop)
  movement_speed *= 0.9 (slightly slower)

painLevel 7-10:
  body_bone.y += 4px
  head_bone.rotation += -4°
  movement_speed *= 0.8
  Blend toward sad_idle colors/expression
```

---

## EVENTS (Fire to Swift)

| Event Name | Fire When | Swift Use Case |
|------------|-----------|----------------|
| `waveComplete` | wave_hello ends | Trigger next UI action |
| `celebrateComplete` | celebrate ends | Resume normal flow |
| `ankiReady` | Initial load complete | Show UI elements |

---

## OPTIMIZATION CHECKLIST

- [ ] Total file size < 50KB
- [ ] No more than 20 bones
- [ ] Meshes have < 50 vertices each
- [ ] State machine has < 15 states
- [ ] Transitions use consistent durations
- [ ] All loops are seamless (end keyframe = start keyframe)
- [ ] No orphan animations (all connected to state machine)
- [ ] Inputs named consistently (camelCase)
- [ ] Events fire at correct moments

---

## SWIFT INTEGRATION REFERENCE

```swift
// Input bindings
riveView.setInput("triggerWaveHello", value: true)
riveView.setInput("isSad", value: true)
riveView.setInput("painLevel", value: 7.0)

// Event listening
riveView.onEvent { event in
    switch event.name {
    case "waveComplete": handleWaveComplete()
    case "celebrateComplete": resumeFlow()
    default: break
    }
}
```

---

## FILE NAMING

```
anki_mascot.riv          (main file)
├── Artboard: "Anki"
├── State Machine: "AnkiController"
└── Animations:
    ├── idle
    ├── wave_hello
    ├── wave_goodbye
    ├── celebrate
    ├── sad_idle
    ├── encourage
    ├── thinking
    ├── nod
    ├── bounce
    └── happy_idle
```

---

## QUICK REFERENCE CARD

### Implemented Rive Animations

| Animation | Duration | Loop | Status |
|-----------|----------|------|--------|
| idle | 3000ms | ✅ | ✅ Implemented |
| wave | 1500ms | ❌ | ✅ Implemented |

### Static Image Fallbacks (Asset Catalog)

| State | Asset Name | Use Case |
|-------|------------|----------|
| happy | `dino-happy 1` | Achievements, positive feedback |
| celebrating | `dino-happy 1` | Milestones completed |
| sad/concerned | `dino-sad` | Empathy, flare logging |
| thinking | `dino-meditating` | Loading, processing |
| sleeping | `dino-sleeping 1` | Rest reminders |
| encouraging | `dino-strong-mussel` | Motivation |
| explaining | `dino-showing-whiteboard` | Tips, education |

### Future Animations (Reference Only)

| Animation | Duration | Loop | Priority |
|-----------|----------|------|----------|
| wave_goodbye | 1800ms | ❌ | Low |
| celebrate | 2000ms | ❌ | Medium |
| sad_idle | 3500ms | ✅ | Low |
| thinking | 2500ms | ✅ | Low |

---

*Last updated: December 2024*
