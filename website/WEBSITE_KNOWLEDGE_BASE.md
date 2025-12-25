# InflamAI Website Knowledge Base

> Comprehensive understanding for creating the InflamAI marketing website

---

## 1. BRAND IDENTITY

### App Name
- **Primary:** InflamAI (stylized as "InflamAi")
- **Alternative:** Spinalytics (used internally)

### Tagline Options
- **German:** "Verstehen, vorhersagen, vorbeugen" (Understand, predict, prevent)
- **English:** "Intelligent Prediction of Rheumatism Flares through AI-Powered Wearable Technology"
- **Short:** "Take control of your AS before it controls you"

### Logo
- Blue medical cross with heart symbol
- Friendly green dinosaur mascot ("Our Dino")
- Clean, medical-grade aesthetic

### Brand Colors
| Color | Hex | Usage |
|-------|-----|-------|
| Primary Blue | #007AFF | Brand, CTAs, Headers |
| Alert Red | #FF3B30 | Flare alerts, Critical pain |
| Success Green | #34C759 | Improvements, Low activity |
| Warning Orange | #FF9500 | Moderate severity |
| Dark Background | #0A1929 | Hero sections (from pitch deck) |
| Accent Cyan | #00D9FF | Highlights, Feature cards |

---

## 2. TARGET AUDIENCE

### Primary
- **Ankylosing Spondylitis (AS) patients** in DACH region
- 1.5M in Germany alone, 3M+ in DACH
- Age range: 25-55 (typical AS onset)

### Secondary
- Other inflammatory rheumatic diseases (Psoriatic Arthritis, RA)
- Rheumatologists seeking patient data tools
- Healthcare systems looking for DiGA-certified solutions

### Pain Points
1. **Uncertainty** - Never knowing when the next flare will hit
2. **Missing triggers** - Can't identify what causes flare-ups
3. **Reactive treatment** - Only treating after symptoms appear
4. **Doctor communication** - Hard to convey symptom history

---

## 3. VALUE PROPOSITION

### Core Promise
Enable rheumatism patients to **understand**, **predict**, and **prevent** flare-ups through AI-powered continuous monitoring.

### Key Differentiators
1. **Predictive Analytics** - Detect flares 3-7 days before onset
2. **Personal Trigger Discovery** - ML identifies YOUR individual triggers
3. **100% Privacy First** - No server, no cloud, all on-device
4. **90+ Data Points** - Comprehensive tracking (vitals, weather, lifestyle)
5. **Clinical Integration** - Auto-generated medical reports for doctors

---

## 4. FEATURE SET

### A. ML Module (Core Innovation)

#### Flare Prediction
- 3-7 day advance warning
- Machine Learning on wearable + environmental data
- Based on ActConnect and WearablesPro studies

#### Trigger Pattern Analysis
- Pearson correlation engine (p < 0.05, |r| > 0.4)
- Identifies personal triggers with statistical significance
- Weather, sleep, activity, stress correlations
- Lag analysis (0h, 12h, 24h offsets)

#### Digital Twin (Future)
- Test new medications on a digital copy
- Predict medication effectiveness based on similar patients
- Similar to pharma AI drug discovery

### B. Classic Tracking Functions

#### Body Map (47 Regions)
- Cervical spine: C1-C7
- Thoracic spine: T1-T12
- Lumbar spine: L1-L5
- SI joints
- Peripheral joints (bilateral): shoulders, elbows, wrists, hands, hips, knees, ankles, feet
- Real-time heatmap visualization
- Pain levels 0-10, stiffness duration, swelling/warmth

#### Clinical Assessments
- **BASDAI** (Bath Ankylosing Spondylitis Disease Activity Index)
- **ASDAS-CRP** (Ankylosing Spondylitis Disease Activity Score)
- **30+ additional assessments** available
- Validated against medical literature

#### Medication Tracking
- NSAIDs, DMARDs, Biologics, Corticosteroids
- Multiple daily reminders
- 30-day adherence calendar
- Biologic indicators

#### Journaling
- Daily symptom logs
- Trigger notes
- Pain type recording
- Context capturing

#### Quick Flare Capture (JointTap SOS)
- 3-tap emergency logging
- Large buttons for impaired mobility
- 4 severity levels
- Haptic feedback

#### Doctor Reports (PDF Export)
- 3-page clinical reports
- BASDAI scores, symptom trends, medication adherence
- HIPAA-compliant formatting
- Save doctor and patient time

#### Knowledge Quiz
- Test understanding of your disease
- Educational content delivery

### C. Wellness Module

#### Exercise Library
- **52 AS-specific exercises**
- 6 categories: Stretching, Strengthening, Mobility, Breathing, Posture, Balance
- Difficulty levels, duration, step-by-step instructions

#### AI Exercise Coach
- Personalized routine generation
- Goal-based (Flexibility, Strength, Pain Management)
- Symptom-aware recommendations
- Time-based options (5-30 minutes)

#### Breathing Exercises
- Guided breathing for pain management
- Stress reduction techniques

#### Meditation
- Mindfulness for chronic pain
- Relaxation techniques

### D. Platform Vision (Future)

#### Appointment Management
- Like Doclib but for all rheumatology services
- Physio, rheumatologist, radiologist, sports scheduling
- Full healthcare journey management

---

## 5. TECHNOLOGY

### Stack
- **Platform:** iOS 17.0+, Swift 5.9+
- **Architecture:** MVVM + Feature-based modular
- **Persistence:** Core Data (8 entities)
- **Health Data:** HealthKit (read-only)
- **Weather:** WeatherKit
- **Wearables:** Apple Watch (watchOS 10+)
- **Charts:** Swift Charts
- **Export:** PDF generation

### Data Points Collected (90+)
- Vital signs (HRV, heart rate, sleep)
- Movement patterns (steps, activity)
- Weather (barometric pressure, humidity, temperature)
- Lifestyle (medication, exercise, stress)
- Symptoms (pain, stiffness, fatigue, mood)

### Privacy Architecture
- **Zero third-party SDKs** (no Firebase, Analytics, Facebook)
- **100% on-device processing**
- **Optional CloudKit sync** (user-controlled)
- **Biometric lock** (Face ID/Touch ID)
- **GDPR compliant** data deletion
- **No tracking whatsoever**

---

## 6. SCIENTIFIC EVIDENCE

### ActConnect Study
- Validates flare prediction through wearable data + ML
- Rheumatic diseases focus
- Published research backing

### WearablesPro Study
- Automatic pain detection without manual input
- AI-powered analysis of movement + vital parameters
- Validates passive monitoring approach

### Clinical Validation
- BASDAI calculator tested against medical literature
- ASDAS-CRP formula implementation verified
- Correlation engine validated for statistical accuracy

---

## 7. MARKET DATA

### Market Size
- **1.5M** rheumatism patients in Germany
- **3M+** in DACH region
- **19% CAGR** Digital Health market growth to 2030
- **27.8% CAGR** Digital Therapeutics (DiGA)
- **38.6% CAGR** AI in Healthcare

### Cost Savings
- **€2,500** per patient per year through early intervention
- Reduced emergency visits
- Better medication adherence
- Proactive vs reactive treatment

### Competitive Landscape
| Competitor | Limitation |
|------------|------------|
| Manage My Pain | No AS-specific features |
| Bearable | No predictive capabilities |
| ArthritisPower | Limited body map |
| Generic trackers | No clinical validation |

---

## 8. TEAM

### Current
- **Fabian Harnisch** - Geschäftsführer (CEO)

### Open Positions
- **Medical Lead** - Rheumatologe (Rheumatologist)
- **Tech Lead** - Lead Developer (ML/iOS)
- **Research Partner** - University partnership sought

---

## 9. ROADMAP

### Q4 2025 (Current)
- Team expansion
- Optimized ML model
- Wellness feature expansion
- Fundraising & investor acquisition

### Q1 2026
- AI model improvement (via public data)
- UI/UX optimization
- Regulatory development (DiGA preparation)

### Q2 2026
- Key partnerships
- Pilot phase preparation
- Relevant data collection

### Q3 2026
- Clinical study
- User feedback implementation
- ML model refinement

---

## 10. WEBSITE STRUCTURE

### Homepage Sections

1. **Hero**
   - Tagline: "Intelligente Vorhersage von Rheuma-Schüben"
   - Subtext: Problem statement (1.5M patients, unpredictable flares)
   - CTA: Download / Coming Soon
   - App mockup

2. **The Problem**
   - Uncertainty
   - Missing triggers
   - Reactive treatment
   - Visual cards (like pitch deck)

3. **Our Vision**
   - "Verstehen, vorhersagen, vorbeugen"
   - Empower patients to take control

4. **The Solution (Features)**
   - Predictive Analytics
   - Trigger Tracking
   - Therapeutic Tools (50+ exercises)
   - Doctor Integration

5. **Technology**
   - App screenshot
   - Simple UI
   - 90+ data points
   - 100% Privacy First (highlighted)

6. **Scientific Evidence**
   - ActConnect Study
   - WearablesPro Study
   - Trust badges

7. **Market/Impact**
   - 1.5M patients
   - 19% market growth
   - €2.5K savings

8. **Team**
   - Fabian Harnisch
   - Open positions (hiring)

9. **Roadmap**
   - Timeline visualization
   - Q4 → Q1 → Q2 → Q3

10. **CTA Footer**
    - Download / Waitlist signup
    - Contact information

### Subpages
- `/features` - Detailed feature breakdown
- `/privacy` - Privacy policy (emphasize on-device)
- `/science` - Scientific background
- `/team` - Team and careers
- `/contact` - Contact form
- `/press` - Press kit

---

## 11. TONE & MESSAGING

### Voice
- **Empathetic** - We understand the struggle
- **Empowering** - You can take control
- **Medical but accessible** - Clinical accuracy, consumer-friendly
- **Privacy-focused** - Your data stays yours

### Key Messages
1. "Know before the flare" - Predictive power
2. "Your triggers, discovered" - Personalization
3. "Hospital-grade privacy" - Trust
4. "Built by patients, for patients" - Authenticity
5. "Clinical accuracy meets beautiful design" - Quality

### Avoid
- Overpromising cure
- Medical claims without disclaimers
- Technical jargon without explanation
- Fear-based marketing

---

## 12. VISUAL GUIDELINES

### From Pitch Deck
- Dark gradient backgrounds (deep blue to black)
- Cyan/teal accent highlights
- Card-based layouts with rounded corners
- Clean sans-serif typography
- App screenshots as hero elements
- Subtle glow effects on feature cards

### Photography
- Real patients (diverse, active)
- Healthcare contexts
- Apple devices (iPhone, Watch)
- Nature/outdoor activities

### Icons
- Medical cross (primary)
- Heart rate
- Brain/AI symbols
- Shield (privacy)
- Graph/chart (analytics)

---

## 13. TECHNICAL REQUIREMENTS

### Website Stack Recommendations
- Static site (fast, secure)
- Tailwind CSS (modern, responsive)
- Dark mode support
- German + English
- Mobile-first design
- App Store badges
- GDPR-compliant cookie consent
- Contact form with Formspree/Netlify Forms

### Performance
- < 3s load time
- 90+ Lighthouse score
- Optimized images (WebP)
- Lazy loading

### SEO
- Schema markup for medical app
- Meta descriptions in DE/EN
- Alt tags for all images
- Sitemap.xml

---

## 14. CONTENT INVENTORY

### Available Assets
- [ ] App icon (high-res)
- [ ] App screenshots (various screens)
- [ ] Pitch deck visuals
- [ ] Team photo (Fabian)
- [ ] Logo variations
- [ ] Color palette

### Needed
- [ ] Hero video/animation
- [ ] Feature demo GIFs
- [ ] Testimonials
- [ ] Press logos
- [ ] Certification badges

---

## 15. CALL TO ACTIONS

### Primary
- "Jetzt herunterladen" (Download now)
- "Auf die Warteliste" (Join waitlist)
- "Mehr erfahren" (Learn more)

### Secondary
- "Kontakt" (Contact)
- "Für Ärzte" (For doctors)
- "Unser Team" (Our team)
- "Investor Relations"

---

*Last Updated: 2025-12-06*
*Source: InflamAi-2 4.pdf, mindmap-inflamai 2.pdf, Xcode Project Analysis*
