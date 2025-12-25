# InflamAI Onboarding Redesign Proposal
## Duolingo-Inspired 2.5D Animation Experience

**Version:** 1.0
**Date:** December 2024
**Status:** Design Proposal (No Code)

---

## Executive Summary

This proposal reimagines InflamAI's onboarding experience using Duolingo's proven engagement patterns:
- **Interactive 2.5D mascot** ("Anky" the Ankylosaurus) that reacts in real-time
- **Gamified progress system** with immediate rewards
- **Splash screen** with personality-driven animation
- **Reduced friction** through "try before signup" approach
- **State machine-driven animations** using Rive

The goal: Transform chronic disease management from a clinical chore into an engaging daily companion experience.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Duolingo Patterns We're Adopting](#duolingo-patterns-were-adopting)
3. [Splash Screen Design](#splash-screen-design)
4. [Mascot 2.5D Animation System](#mascot-25d-animation-system)
5. [Redesigned Onboarding Flow](#redesigned-onboarding-flow)
6. [Animation Specifications](#animation-specifications)
7. [Technical Implementation Notes](#technical-implementation-notes)
8. [Placeholder Asset Requirements](#placeholder-asset-requirements)

---

## 1. Current State Analysis

### What We Have (12-Page Flow)
- Static PNG mascot images (18 dino variants)
- TabView-based horizontal swiping
- Simple bounce/wave modifiers
- Permission requests mid-flow
- Profile data collection on page 11

### Current Pain Points
| Issue | Impact |
|-------|--------|
| Static mascot images | No emotional connection |
| Long 12-page flow | Completion drop-off |
| Mid-flow permissions | Interrupts engagement momentum |
| No gamification | Missing dopamine hits |
| No splash personality | Missed first impression |

---

## 2. Duolingo Patterns We're Adopting

### Core Psychological Principles

**1. Commitment Before Signup**
> Duolingo asks "What's your goal?" before any account creation. This leverages completion bias—users who start a journey want to finish it.

**2. Interactive Mascot as Companion**
> Duo the owl isn't decoration—he's a relationship. He celebrates, encourages, and even guilt-trips (playfully). This emotional bond drives 80% higher retention.

**3. Progress is Always Visible**
> Every interaction shows forward momentum. Progress bars, checkmarks, celebrations—constant positive reinforcement.

**4. Gamification Vocabulary**
- **Streaks**: "You've logged 3 days in a row!"
- **XP Points**: Earned for daily check-ins
- **Achievements**: Unlockable badges
- **Levels**: Health mastery progression

### Duolingo's Animation Tech Stack
- **Rive** for interactive character animation (state machines)
- **Lottie** for simpler UI animations
- **State Machines** for reactive mascot behavior
- **Additive blending** for smooth emotion transitions

---

## 3. Splash Screen Design

### Concept: "Anky Wakes Up"

```
┌─────────────────────────────────────┐
│                                     │
│                                     │
│          ╭─────────────╮           │
│         ╱   🦕 ANKY    ╲           │
│        │   (sleeping)   │           │
│        │    z z z       │           │
│         ╲              ╱            │
│          ╰─────────────╯            │
│                                     │
│         [subtle glow pulse]         │
│                                     │
│                                     │
│           I N F L A M A I          │
│                                     │
│         ════════════════            │
│         [loading bar]               │
│                                     │
└─────────────────────────────────────┘
```

### Animation Sequence (3 seconds total)

| Time | Animation | Mascot State | UI Elements |
|------|-----------|--------------|-------------|
| 0.0s | Fade in from black | Anky sleeping, gentle breathing | Logo invisible |
| 0.3s | Breathing cycle | Chest rises/falls, subtle "z z z" | — |
| 1.0s | Wake trigger | Eyes open, stretch animation | Logo fades in |
| 1.5s | Happy greeting | Tail wag, smile, wave at user | Tagline appears |
| 2.2s | Ready pose | Settles into idle bounce | Loading complete |
| 2.8s | Exit transition | Slides up eagerly | Screen transitions |

### Splash Screen States

```
STATE: Sleeping
├── Eyes: Closed with gentle movement
├── Body: Subtle breathing (scale 1.0 → 1.02 → 1.0)
├── Tail: Resting, occasional twitch
└── Ambient: Soft glow behind character

STATE: Waking
├── Eyes: Blink sequence (3 blinks)
├── Body: Stretch animation (arms up, yawn)
├── Expression: Transition sleepy → alert → happy
└── Audio: Optional gentle chime

STATE: Greeting
├── Eyes: Wide, sparkle effect
├── Arm: Wave animation (left arm)
├── Tail: Enthusiastic wag
├── Body: Slight bounce anticipation
└── Expression: Big smile, excited

STATE: Ready/Idle
├── Eyes: Soft blink every 3-4 seconds
├── Body: Gentle bounce (offset -8px, 600ms loop)
├── Tail: Slow, content wag
└── Expression: Warm smile, inviting
```

### Visual Design Specifications

**Background:**
- Gradient: `#0EA5E9` (sky-400) → `#0369A1` (sky-700)
- Subtle animated clouds/particles (very slow drift)
- Radial glow behind mascot: `#67E8F9` opacity 30%

**Logo Treatment:**
```
I N F L A M A I
───────────────
Your AS Companion
```
- Font: SF Pro Rounded, Bold, 32pt
- Letter-spacing: 4pt
- Tagline: SF Pro, Regular, 14pt, opacity 70%

**Loading Indicator:**
- Pill-shaped progress bar
- Fills with gradient matching brand colors
- Subtle pulse glow at fill edge
- 2.5 second duration for demo data load

---

## 4. Mascot 2.5D Animation System

### Why "2.5D"?

Duolingo's Duo owl uses a technique where:
- Characters are **2D artwork** (flat illustrations)
- Rigged with **skeletal animation** (bones/joints)
- Animated with **depth parallax** (layers move at different speeds)
- Rendered in **real-time** (not pre-recorded video)

This creates the illusion of 3D depth while maintaining the charm of 2D illustration.

### Anky Character Rig Specification

```
ANKY RIG HIERARCHY
==================

Root (Full Character)
├── Body_Group
│   ├── Body_Main (teal ellipse, primary shape)
│   ├── Armor_Plates[] (7 layered ovals along spine)
│   │   └── Each plate: individual transform for ripple effects
│   └── Underbelly (lighter gradient overlay)
│
├── Head_Group (parented to Body, offset front)
│   ├── Head_Shape (radial gradient, snout)
│   ├── Eyes_Group
│   │   ├── Eye_Left (socket + pupil + highlight)
│   │   ├── Eye_Right (socket + pupil + highlight)
│   │   └── Brows (for expressions)
│   ├── Mouth_Group
│   │   ├── Mouth_Neutral
│   │   ├── Mouth_Happy
│   │   ├── Mouth_Sad
│   │   ├── Mouth_Surprised
│   │   ├── Mouth_Encouraging
│   │   └── Mouth_Speaking (for lip-sync)
│   └── Cheeks (blush overlay, opacity animated)
│
├── Tail_Group (parented to Body, offset back)
│   ├── Tail_Base (bezier curve segment 1)
│   ├── Tail_Mid (bezier curve segment 2)
│   ├── Tail_Tip (bezier curve segment 3)
│   └── Tail_Club (iconic ankylosaurus tail club)
│
├── Legs_Group
│   ├── Leg_FrontLeft
│   ├── Leg_FrontRight
│   ├── Leg_BackLeft
│   └── Leg_BackRight
│   └── Each: Upper, Lower, Foot segments
│
└── Props_Group (optional accessories)
    ├── Prop_Clipboard (for check-in scenes)
    ├── Prop_Pill (for medication reminders)
    ├── Prop_Heart (for celebration)
    └── Prop_Confetti (particle emitter)
```

### Animation States (State Machine)

```
┌─────────────────────────────────────────────────────────────┐
│                    ANKY STATE MACHINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌─────────┐     user_active      ┌───────────┐           │
│    │ IDLE    │ ──────────────────▶ │ ATTENTIVE │           │
│    │(bounce) │ ◀────────────────── │ (looking) │           │
│    └────┬────┘      3s_timeout      └─────┬─────┘           │
│         │                                  │                 │
│         │ user_completes_action           │ user_succeeds   │
│         ▼                                  ▼                 │
│    ┌─────────┐                      ┌───────────┐           │
│    │ WAITING │                      │ CELEBRATE │           │
│    │(patient)│                      │ (confetti)│           │
│    └────┬────┘                      └─────┬─────┘           │
│         │                                  │                 │
│         │ user_struggles                   │ 2s_timeout      │
│         ▼                                  ▼                 │
│    ┌─────────────┐                  ┌───────────┐           │
│    │ ENCOURAGING │ ────────────────▶│   IDLE    │           │
│    │   (gentle)  │   user_continues │ (bounce)  │           │
│    └─────────────┘                  └───────────┘           │
│                                                              │
│    SPECIAL STATES (triggered by specific events):           │
│    • SLEEPING (splash screen only)                          │
│    • WAVING (first greeting)                                │
│    • EXPLAINING (with prop: whiteboard)                     │
│    • CONCERNED (pain level high)                            │
│    • PROUD (streak achieved)                                │
│    • SYMPATHETIC (flare detected)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Expression Library

| Expression | Eyes | Mouth | Brows | Cheeks | Use Case |
|------------|------|-------|-------|--------|----------|
| **Neutral** | Soft, slight smile curves | Gentle smile | Relaxed | None | Default idle |
| **Happy** | Wide, sparkle | Big grin | Raised | Pink blush | Success, celebration |
| **Encouraging** | Warm, gentle | Small smile | Slightly raised | Subtle | User struggling |
| **Concerned** | Wide, focused | Slight frown | Knitted | None | High pain reported |
| **Proud** | Beaming | Huge smile | High | Full blush | Streak achieved |
| **Sympathetic** | Soft, caring | Gentle | Tilted | None | Flare reported |
| **Sleepy** | Half-closed | Yawn shape | Low | None | Bedtime reminder |
| **Excited** | Huge, stars | Open smile | Way up | Full blush | Major achievement |
| **Curious** | One raised | Closed | One up, one neutral | None | Exploring feature |
| **Waving** | Bright | Smile | Raised | Pink | Greeting user |

### Parallax Depth Layers

For 2.5D effect during head turns or body movements:

```
LAYER DEPTH (front to back)
===========================
z: 1.0  → Props (clipboard, pill, etc.)
z: 0.9  → Eyes (move most during look-around)
z: 0.8  → Mouth, Brows
z: 0.7  → Head
z: 0.5  → Front Legs
z: 0.3  → Body
z: 0.2  → Armor Plates
z: 0.1  → Back Legs
z: 0.0  → Tail

PARALLAX FORMULA:
offset_x = base_offset * (1 - layer_z) * parallax_intensity
```

When Anky "looks around," deeper layers move less, creating depth illusion.

---

## 5. Redesigned Onboarding Flow

### Philosophy Shift

**Old Approach:** 12 pages of information → permission requests → profile setup
**New Approach:** Immediate value → emotional connection → commitment → configuration

### New Flow Structure (7 Interactive Stages)

```
┌─────────────────────────────────────────────────────────────┐
│                    ONBOARDING JOURNEY                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [SPLASH] → [STAGE 1] → [STAGE 2] → [STAGE 3]              │
│     │          │           │           │                     │
│     │          │           │           │                     │
│     ▼          ▼           ▼           ▼                     │
│   Wake      Meet         Your        First                  │
│   Anky      Anky        "Why"       Experience              │
│             + Goal       + Goal       (try it!)              │
│                                                              │
│                                                              │
│  → [STAGE 4] → [STAGE 5] → [STAGE 6] → [STAGE 7]           │
│        │           │           │           │                 │
│        ▼           ▼           ▼           ▼                 │
│    Celebrate    Power-Up    Profile    Ready to              │
│    First Win    Features    (quick)    Begin!               │
│   (XP earned!)  (features)                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

TOTAL TIME: ~3 minutes (vs. current ~5 minutes)
SCREENS: 7-9 (vs. current 12)
INTERACTIONS: 15+ (vs. current ~8)
```

---

### Stage 1: Meet Anky (Emotional Hook)

**Duration:** 30 seconds
**Mascot State:** Waving → Happy → Attentive

```
┌─────────────────────────────────────────┐
│                                         │
│           ╭───────────────╮            │
│          ╱                 ╲           │
│         │   🦕 ANKY        │           │
│         │   [waving]       │           │
│         │   "Hi there!"    │           │
│          ╲                 ╱            │
│           ╰───────────────╯             │
│                                         │
│                                         │
│      "I'm Anky, your companion         │
│       for managing AS together."       │
│                                         │
│      ┌─────────────────────────┐       │
│      │                         │       │
│      │   [Anky waves more]    │       │
│      │                         │       │
│      └─────────────────────────┘       │
│                                         │
│                                         │
│       ┌─────────────────────┐          │
│       │   Let's get started  │ ────▶   │
│       └─────────────────────┘          │
│                                         │
│        ○ ○ ○ ○ ○ ○ ○                   │
│        ●                                │
└─────────────────────────────────────────┘
```

**Animation Details:**
- Anky enters from bottom with bounce
- Waves enthusiastically for 2 seconds
- Speech bubble types out letter-by-letter (typewriter effect)
- Settles into idle bounce while waiting
- Eyes follow user's touch (if detected)

---

### Stage 2: Your "Why" (Commitment)

**Duration:** 45 seconds
**Mascot State:** Curious → Attentive (reacts to selection)

**Inspired by Duolingo's "Why are you learning?"**

```
┌─────────────────────────────────────────┐
│                                         │
│        ╭───────────╮                   │
│        │ 🦕 Anky   │                   │
│        │ [curious] │                   │
│        ╰───────────╯                   │
│                                         │
│     "What matters most to you?"        │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │  🎯  Track my daily symptoms    │  │
│   │      and see patterns           │◀─┤ Selected
│   └─────────────────────────────────┘  │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │  💊  Never miss a medication   │  │
│   └─────────────────────────────────┘  │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │  📊  Share reports with my     │  │
│   │      rheumatologist            │  │
│   └─────────────────────────────────┘  │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │  🔮  Predict and prevent flares │  │
│   └─────────────────────────────────┘  │
│                                         │
│       ┌─────────────────────┐          │
│       │      Continue        │ ────▶   │
│       └─────────────────────┘          │
│        ● ● ○ ○ ○ ○ ○                   │
└─────────────────────────────────────────┘
```

**Animation Details:**
- Cards slide in staggered (100ms delay each)
- On selection: Card pulses, checkmark animates in
- Anky reacts: Nods approvingly, tail wags faster
- Selected card elevates with shadow
- Multi-select allowed (up to 2)

---

### Stage 3: First Experience (Try Before Commit)

**Duration:** 60-90 seconds
**Mascot State:** Explaining → Encouraging → Celebrating

**Key Innovation:** Let users experience the core value BEFORE any account/permissions.

```
┌─────────────────────────────────────────┐
│                                         │
│    ╭───────╮                           │
│    │🦕[📋]│  "Let's do a quick         │
│    ╰───────╯   check-in together!"     │
│                                         │
│  ╔═══════════════════════════════════╗ │
│  ║                                   ║ │
│  ║   How's your fatigue today?      ║ │
│  ║                                   ║ │
│  ║    😴 ─────●───────────── 💪      ║ │
│  ║         5 / 10                    ║ │
│  ║                                   ║ │
│  ║   [Anky nods encouragingly]       ║ │
│  ║                                   ║ │
│  ╚═══════════════════════════════════╝ │
│                                         │
│         1 of 3 questions               │
│         ━━━━━━━━░░░░░░░░░              │
│                                         │
│       ┌─────────────────────┐          │
│       │        Next          │ ────▶   │
│       └─────────────────────┘          │
│        ● ● ● ○ ○ ○ ○                   │
└─────────────────────────────────────────┘
```

**Sub-screens:**
1. **Fatigue slider** (0-10, with emoji scale)
2. **Morning stiffness** ("How long until you loosen up?")
3. **Quick body tap** ("Tap where it hurts most today")

**After 3 questions → Immediate Reward:**

```
┌─────────────────────────────────────────┐
│                                         │
│         ✨ ✨ ✨ ✨ ✨ ✨                │
│                                         │
│           ╭───────────────╮            │
│           │  🦕 ANKY      │            │
│           │  [celebrating]│            │
│           │  🎉 confetti  │            │
│           ╰───────────────╯            │
│                                         │
│        "You just earned your           │
│         first Health Points!"          │
│                                         │
│           ╭─────────────╮              │
│           │   +50 XP    │              │
│           │   ⭐ First  │              │
│           │   Check-In  │              │
│           ╰─────────────╯              │
│                                         │
│      Your BASDAI estimate: 4.2         │
│      (We'll track this over time)      │
│                                         │
│       ┌─────────────────────┐          │
│       │     Keep going!      │ ────▶   │
│       └─────────────────────┘          │
│        ● ● ● ● ○ ○ ○                   │
└─────────────────────────────────────────┘
```

**Animation Details:**
- Confetti particle burst (12-16 pieces)
- XP counter animates up (+50 with bounce)
- Badge slides in from bottom with spring
- Anky does celebratory dance (arms up, spin)
- Achievement "ding" sound (optional)
- Screen shake effect (subtle, 100ms)

---

### Stage 4: Power-Up Features (Feature Discovery)

**Duration:** 45 seconds
**Mascot State:** Explaining → Excited

**Swipeable feature cards with Anky as guide:**

```
┌─────────────────────────────────────────┐
│                                         │
│   ╭─────╮                              │
│   │ 🦕  │  "Here's what I can          │
│   ╰─────╯   help you with..."          │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │                                   │ │
│  │      ┌─────────────────────┐     │ │
│  │      │    🗺️ Body Map      │     │ │
│  │      │                     │     │ │
│  │      │  [Mini animation:   │     │ │
│  │      │   body outline with │     │ │
│  │      │   tappable points]  │     │ │
│  │      │                     │     │ │
│  │      │  Track 47 specific  │     │ │
│  │      │  pain points        │     │ │
│  │      └─────────────────────┘     │ │
│  │                                   │ │
│  │        ◀ swipe ●○○○ ▶           │ │
│  │                                   │ │
│  └───────────────────────────────────┘ │
│                                         │
│       ┌─────────────────────┐          │
│       │   Unlock Features    │ ────▶   │
│       └─────────────────────┘          │
│        ● ● ● ● ● ○ ○                   │
└─────────────────────────────────────────┘
```

**Feature Cards (4 total, swipeable):**

| Card | Icon | Title | Mini-Animation |
|------|------|-------|----------------|
| 1 | 🗺️ | Body Map | Tappable body silhouette |
| 2 | 📊 | Smart Trends | Animated chart drawing |
| 3 | 🌤️ | Weather Alerts | Weather icons with prediction |
| 4 | 📄 | Doctor Reports | PDF export preview |

**Animation Details:**
- Cards have 3D rotation on swipe (perspective)
- Mini-animations loop within each card
- Anky looks at currently visible card
- Page indicators pulse on swipe

---

### Stage 5: Power-Ups (Optional Permissions)

**Duration:** 30 seconds
**Mascot State:** Helpful → Encouraging

**Frame permissions as "power-ups" that enhance experience:**

```
┌─────────────────────────────────────────┐
│                                         │
│   ╭─────╮  "These power-ups make       │
│   │ 🦕  │   me even more helpful!"     │
│   ╰─────╯                              │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │  ❤️  HealthKit Sync              │  │
│   │      See how sleep affects you   │  │
│   │                         ┌────┐  │  │
│   │                         │ ON │  │  │
│   │                         └────┘  │  │
│   └─────────────────────────────────┘  │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │  🔔  Smart Reminders            │  │
│   │      Never miss a check-in      │  │
│   │                         ┌────┐  │  │
│   │                         │ ON │  │  │
│   │                         └────┘  │  │
│   └─────────────────────────────────┘  │
│                                         │
│       "You can change these anytime    │
│        in Settings"                    │
│                                         │
│       ┌─────────────────────┐          │
│       │      Continue        │ ────▶   │
│       └─────────────────────┘          │
│        ● ● ● ● ● ● ○                   │
└─────────────────────────────────────────┘
```

**Key Design Choice:**
Permissions are **ON by default** with toggle—not a modal interrupt. Users who want to skip can proceed without friction.

**Animation Details:**
- Toggles slide in staggered
- ON state: Toggle animates with satisfying click
- Anky gives thumbs-up when enabled
- If user disables: Anky shrugs gently (no guilt)

---

### Stage 6: Quick Profile (Essential Only)

**Duration:** 30 seconds
**Mascot State:** Patient → Encouraging

**Minimal data collection—just enough for personalization:**

```
┌─────────────────────────────────────────┐
│                                         │
│   ╭─────╮  "Just a few quick details   │
│   │ 🦕  │   to personalize things!"    │
│   ╰─────╯                              │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │  How should I address you?      │  │
│   │                                  │  │
│   │   ┌─────────────────────────┐   │  │
│   │   │  Your name (optional)    │   │  │
│   │   └─────────────────────────┘   │  │
│   └─────────────────────────────────┘  │
│                                         │
│   ┌─────────────────────────────────┐  │
│   │  When's your birthday? 🎂        │  │
│   │                                  │  │
│   │   ┌───┐ ┌───┐ ┌────┐            │  │
│   │   │Jan│ │ 1 │ │1990│            │  │
│   │   └───┘ └───┘ └────┘            │  │
│   └─────────────────────────────────┘  │
│                                         │
│         ⚡ 2 more fields...            │
│                                         │
│       ┌─────────────────────┐          │
│       │    Almost done!      │ ────▶   │
│       └─────────────────────┘          │
│        ● ● ● ● ● ● ●                   │
└─────────────────────────────────────────┘
```

**Fields (maximum 4):**
1. Name (optional, for personalized greetings)
2. Birth date (for age-appropriate insights)
3. Height (for BMI calculations)
4. Weight (optional, can skip)

**Animation Details:**
- Keyboard-aware layout (content shifts up)
- Anky peeks from side during input
- Skip option clearly visible
- Progress fills as fields complete

---

### Stage 7: Ready to Begin! (Celebration)

**Duration:** 15 seconds
**Mascot State:** MAXIMUM CELEBRATION

```
┌─────────────────────────────────────────┐
│                                         │
│     ✨ 🎉 ✨ 🎊 ✨ 🎉 ✨                │
│                                         │
│           ╭───────────────╮            │
│           │   🦕 ANKY     │            │
│           │               │            │
│           │  [DANCING!]   │            │
│           │   arms up,    │            │
│           │   spinning,   │            │
│           │   confetti    │            │
│           │   everywhere  │            │
│           ╰───────────────╯            │
│                                         │
│        "You're all set, [Name]!        │
│         Let's manage AS together!"     │
│                                         │
│         ╭─────────────────────╮        │
│         │   🏆 Day 1 Started  │        │
│         │   📊 +50 XP Earned  │        │
│         │   🔥 Streak: 1      │        │
│         ╰─────────────────────╯        │
│                                         │
│       ┌─────────────────────┐          │
│       │   Start My Journey   │ ────▶   │
│       └─────────────────────┘          │
│        ● ● ● ● ● ● ●                   │
└─────────────────────────────────────────┘
```

**Animation Details:**
- Full-screen confetti explosion
- Anky does victory dance (2-3 second loop)
- Achievement cards slide in with bounce
- Streak counter starts at 1
- Haptic: Success pattern
- Sound: Celebration fanfare (optional)
- Background pulses with brand colors

**Transition to Home:**
- Anky waves goodbye
- Screen morphs into home dashboard
- Anky appears as mini helper in corner

---

## 6. Animation Specifications

### Timing Standards

```swift
// MOTION TIMING TOKENS
struct AnkyMotion {
    // Durations
    static let instant   = 0.1   // Micro-interactions
    static let fast      = 0.2   // Button presses
    static let normal    = 0.3   // Page transitions
    static let smooth    = 0.4   // Mascot expressions
    static let slow      = 0.6   // Emphasis animations
    static let dramatic  = 1.0   // Celebrations

    // Springs
    static let snappy    = Animation.spring(response: 0.3, dampingFraction: 0.7)
    static let bouncy    = Animation.spring(response: 0.5, dampingFraction: 0.5)
    static let gentle    = Animation.spring(response: 0.6, dampingFraction: 0.8)

    // Easing
    static let easeOut   = Animation.easeOut(duration: normal)
    static let easeInOut = Animation.easeInOut(duration: smooth)
}
```

### Anky Base Animations

| Animation | Duration | Easing | Loop | Description |
|-----------|----------|--------|------|-------------|
| Idle Bounce | 600ms | easeInOut | forever | Y offset: 0 → -8 → 0 |
| Tail Wag | 800ms | easeInOut | forever | Rotation: -5° → 5° |
| Blink | 200ms | easeIn | every 3-4s | Eyelids close/open |
| Wave | 500ms | spring | once | Arm rotation 0° → 25° → 0° |
| Celebrate | 1200ms | bouncy | once | Jump + spin + arms up |
| Nod | 400ms | easeInOut | once | Y rotation: -5° → 5° → 0° |
| Head Tilt | 300ms | spring | once | Z rotation: 0° → 15° |
| Look At | 200ms | easeOut | once | Eye offset toward point |

### Expression Transitions

```
EXPRESSION BLEND TIMING
=======================

neutral → happy:     300ms ease-out
neutral → concerned: 400ms ease-in-out
neutral → excited:   200ms spring (bouncy)
any → any:           350ms ease-in-out (default)

LAYER PRIORITIES:
1. Eyes change first (100ms lead)
2. Mouth follows (50ms delay)
3. Brows last (100ms delay)
4. Cheeks overlay (parallel with mouth)
```

### Parallax Motion

```swift
// When Anky looks around or user scrolls
func parallaxOffset(for layer: CGFloat, scrollOffset: CGFloat) -> CGFloat {
    let parallaxIntensity: CGFloat = 0.15
    return scrollOffset * (1 - layer) * parallaxIntensity
}

// Example: Eye layer (z: 0.9) moves 1.5x more than body (z: 0.3)
```

### Celebration Particles

```
CONFETTI SYSTEM
===============

Particle Count: 24
Colors: [brand-teal, brand-blue, gold, pink, white]
Shapes: [square, circle, star]
Spawn: Top of screen, random X
Physics:
  - Initial velocity: random(200-400) downward
  - Gravity: 300 pt/s²
  - Rotation: random spin
  - Fade: starts at 80% of lifetime
Lifetime: 2.5 seconds
```

---

## 7. Technical Implementation Notes

### Recommended Animation Stack

**Option A: Rive (Recommended)**
- Matches Duolingo's exact approach
- State machine support for reactive animations
- Tiny file sizes (~50-100KB per character)
- Native iOS SDK: `rive-app/rive-ios`
- Real-time interactivity

**Option B: Lottie + SwiftUI**
- Larger ecosystem of pre-made animations
- Easier designer handoff
- Good for simpler animations
- Native iOS SDK: `airbnb/lottie-ios`

**Option C: SwiftUI Native + Canvas**
- No external dependencies
- Full control
- More engineering effort
- Already have `AnkylosaurusMascot.swift` as base

### Rive Integration (If Chosen)

```swift
// Example integration structure
import RiveRuntime

struct AnkyView: View {
    @StateObject private var anky = RiveViewModel(
        fileName: "anky_character",
        stateMachineName: "Main"
    )

    var body: some View {
        anky.view()
            .onAppear {
                anky.setInput("expression", value: "happy")
            }
    }

    func celebrate() {
        anky.triggerInput("celebrate")
    }

    func lookAt(point: CGPoint) {
        anky.setInput("lookX", value: Float(point.x))
        anky.setInput("lookY", value: Float(point.y))
    }
}
```

### State Machine Inputs

```
RIVE STATE MACHINE INPUTS
=========================

Boolean Inputs:
- isActive (user interacting)
- isHappy (positive feedback)
- isConcerned (high pain levels)

Trigger Inputs:
- celebrate (one-shot celebration)
- wave (greeting gesture)
- nod (approval)
- encourage (supportive gesture)

Number Inputs:
- lookX (-1 to 1, horizontal gaze)
- lookY (-1 to 1, vertical gaze)
- expressionBlend (0-1, between states)
```

### Asset File Structure

```
Assets/
├── Rive/
│   ├── anky_character.riv       (main character file)
│   ├── anky_splash.riv          (splash screen version)
│   └── anky_mini.riv            (compact helper version)
├── Lottie/
│   ├── confetti_burst.json      (celebration particles)
│   ├── progress_fill.json       (XP bar animation)
│   ├── checkmark_success.json   (completion animation)
│   └── sparkle_loop.json        (ambient sparkles)
└── PNG/
    └── anky_fallback/           (static fallbacks)
        ├── anky_idle.png
        ├── anky_happy.png
        ├── anky_wave.png
        └── anky_celebrate.png
```

---

## 8. Placeholder Asset Requirements

### For Designer/Animator Brief

#### Anky Character Rive File

**Deliverable:** `anky_character.riv`

**Artboards Required:**
1. `Character_Full` - Complete character for onboarding
2. `Character_Mini` - Head + shoulders only for in-app helper
3. `Character_Splash` - Sleeping pose for splash screen

**State Machine:** `Main`

**Required States:**
| State Name | Description | Transitions |
|------------|-------------|-------------|
| `idle` | Gentle bounce, soft blink | Entry state |
| `attentive` | Eyes wide, leaning forward | from idle on user_active |
| `happy` | Big smile, tail wag fast | from any on success |
| `celebrating` | Jump, spin, arms up | triggered celebration |
| `encouraging` | Warm expression, slight nod | from attentive on struggle |
| `concerned` | Worried look, head tilt | from any on high_pain |
| `waving` | Arm wave animation | triggered greeting |
| `explaining` | Holding clipboard prop | triggered for tutorials |
| `sleeping` | Eyes closed, breathing | splash screen only |

**Required Inputs:**
```
Boolean: isActive, isHappy, isConcerned
Trigger: celebrate, wave, nod, encourage
Number:  lookX, lookY, expressionBlend
```

**File Size Target:** < 150KB

---

#### Lottie Animations

**1. Confetti Burst** - `confetti_burst.json`
- Duration: 2.5 seconds
- One-shot (no loop)
- 24 particles, 5 colors
- File size: < 30KB

**2. XP Counter** - `xp_counter.json`
- Duration: 1 second
- One-shot
- Numbers count up with bounce
- File size: < 15KB

**3. Progress Fill** - `progress_fill.json`
- Duration: 0.8 seconds
- One-shot
- Pill bar fills with glow
- File size: < 10KB

**4. Success Checkmark** - `checkmark_success.json`
- Duration: 0.6 seconds
- One-shot
- Check draws in with bounce
- File size: < 8KB

**5. Sparkle Loop** - `sparkle_loop.json`
- Duration: 2 seconds
- Looping
- Ambient sparkles for celebrations
- File size: < 12KB

---

#### Static Fallbacks (PNG)

For devices/situations where Rive can't run:

| Asset | Size | States |
|-------|------|--------|
| `anky_idle.png` | 512×512 @3x | Neutral standing |
| `anky_happy.png` | 512×512 @3x | Big smile |
| `anky_wave.png` | 512×512 @3x | Waving gesture |
| `anky_celebrate.png` | 512×512 @3x | Arms up, confetti |
| `anky_concerned.png` | 512×512 @3x | Worried expression |
| `anky_sleeping.png` | 512×512 @3x | Eyes closed, Zzzz |
| `anky_explaining.png` | 512×512 @3x | With clipboard |

---

## Appendix: Reference Resources

### Duolingo Research Sources
- [Duolingo User Onboarding Breakdown](https://goodux.appcues.com/blog/duolingo-user-onboarding)
- [How Duolingo Uses Rive](https://dev.to/uianimation/how-duolingo-uses-rive-for-their-character-animation-and-how-you-can-build-a-similar-rive-mascot-5d19)
- [Duolingo Lottie Case Study](https://lottiefiles.com/case-studies/duolingo)
- [Building Character at Duolingo](https://blog.duolingo.com/building-character/)

### Rive Integration
- [Rive iOS Guide](https://help.rive.app/runtimes/overview/ios)
- [Rive State Machines](https://help.rive.app/runtimes/state-machines)
- [SwiftUI + Rive Course](https://designcode.io/swiftui-rive)
- [rive-app/rive-ios GitHub](https://github.com/rive-app/rive-ios)

### Health Gamification
- [Gamification in Healthcare 2024](https://agentestudio.com/blog/healthcare-app-gamification)
- [Healthcare Onboarding Best Practices](https://nozomihealth.com/an-overview-of-user-onboarding-practices-in-digital-health/)

---

## Summary: Key Differentiators

| Current Flow | Redesigned Flow |
|--------------|-----------------|
| 12 static pages | 7 interactive stages |
| PNG mascot images | Real-time 2.5D animated mascot |
| Information dump | "Try before commit" approach |
| Mid-flow permission interrupts | Frictionless toggle power-ups |
| No gamification | XP, streaks, achievements |
| Generic welcome | Personalized emotional connection |
| 5+ minute completion | ~3 minute completion |
| 30% estimated completion | 70%+ target completion |

**The Goal:** Transform onboarding from "medical app setup" into "meeting a helpful companion who makes AS management feel achievable."

---

*Document Version 1.0 | December 2024 | Ready for Design Review*
