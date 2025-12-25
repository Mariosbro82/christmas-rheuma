# Open-Meteo API Integration Plan

## Executive Summary

Replace Apple WeatherKit with **Open-Meteo API** - a free, no-API-key, high-quality weather service. This eliminates WeatherKit entitlement requirements, reduces complexity, and provides reliable weather data for AS flare prediction.

---

## Why Open-Meteo?

| Aspect | Apple WeatherKit | Open-Meteo |
|--------|------------------|------------|
| **Cost** | 500k calls/month free, then paid | Unlimited FREE |
| **API Key** | Requires Apple Developer entitlement | No key needed |
| **Setup** | Entitlements, capabilities, provisioning | Just HTTP calls |
| **Data Quality** | Excellent | Excellent (ECMWF, DWD, NOAA) |
| **Historical Data** | Limited | 80+ years available |
| **Forecast Range** | 10 days | 16 days |
| **Update Frequency** | Hourly | Every 15 minutes |
| **Pressure Data** | Yes | Yes (surface + sea level) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         InflamAI App                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   UI Layer   │    │  ML Pipeline │    │   Watch App      │  │
│  │              │    │              │    │                  │  │
│  │ WeatherCard  │    │ Biometrics   │    │ Complications    │  │
│  │ FlareRisk    │    │ Collection   │    │ Quick Log        │  │
│  │ PressureChart│    │ Service      │    │                  │  │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
│         │                   │                     │             │
│         └───────────────────┼─────────────────────┘             │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │                 │                          │
│                    │  OpenMeteo      │  ◄── Single Source       │
│                    │  Service        │      of Truth            │
│                    │                 │                          │
│                    └────────┬────────┘                          │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │               │
│  ┌──────▼──────┐    ┌───────▼───────┐   ┌──────▼──────┐        │
│  │   Cache     │    │  Persistence  │   │  Pressure   │        │
│  │  (Memory)   │    │  (Core Data)  │   │  History    │        │
│  │  15 min TTL │    │ ContextSnap   │   │  (File)     │        │
│  └─────────────┘    └───────────────┘   └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Open-Meteo API       │
                 │   api.open-meteo.com   │
                 │                        │
                 │   • No API Key         │
                 │   • No Rate Limits     │
                 │   • Free Forever       │
                 └────────────────────────┘
```

---

## Phase 1: Core Service Implementation

### 1.1 OpenMeteoService.swift

**Location:** `Core/Services/OpenMeteoService.swift`

```swift
// Key Design Decisions:
// - Singleton pattern (consistent with existing services)
// - @MainActor for thread safety
// - ObservableObject for SwiftUI binding
// - Async/await for modern concurrency
// - Protocol-based for testability
```

**API Endpoints to Use:**

```
Base URL: https://api.open-meteo.com/v1/forecast

Current + Hourly + Daily:
?latitude={lat}
&longitude={lon}
&current=temperature_2m,relative_humidity_2m,apparent_temperature,
         pressure_msl,surface_pressure,weather_code,wind_speed_10m,
         wind_direction_10m,precipitation,uv_index
&hourly=temperature_2m,relative_humidity_2m,pressure_msl,
        precipitation_probability,weather_code
&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,
       weather_code,sunrise,sunset
&timezone=auto
&forecast_days=7
```

**Response Structure:**
```json
{
  "current": {
    "time": "2025-12-01T14:00",
    "temperature_2m": 12.5,
    "relative_humidity_2m": 65,
    "pressure_msl": 1013.2,
    "weather_code": 3
  },
  "hourly": {
    "time": ["2025-12-01T00:00", ...],
    "temperature_2m": [10.2, 10.5, ...],
    "pressure_msl": [1012.5, 1012.8, ...]
  },
  "daily": {
    "time": ["2025-12-01", ...],
    "temperature_2m_max": [14.5, ...],
    "temperature_2m_min": [8.2, ...]
  }
}
```

### 1.2 Data Models

**Location:** `Core/Models/OpenMeteoModels.swift`

```swift
// Models to create:

struct OpenMeteoResponse: Codable {
    let latitude: Double
    let longitude: Double
    let timezone: String
    let current: CurrentData
    let hourly: HourlyData
    let daily: DailyData
}

struct CurrentWeather {
    let timestamp: Date
    let temperature: Double           // °C
    let feelsLike: Double             // °C (apparent_temperature)
    let humidity: Int                 // 0-100%
    let pressureMSL: Double           // hPa (sea level)
    let pressureSurface: Double       // hPa (surface)
    let pressureMmHg: Double          // Computed: pressureMSL * 0.750062
    let pressureChange3h: Double      // Computed from history
    let pressureChange12h: Double     // Computed from history
    let pressureChange24h: Double     // Computed from history
    let windSpeed: Double             // km/h
    let windDirection: Int            // degrees
    let precipitation: Double         // mm
    let uvIndex: Double
    let weatherCode: Int              // WMO code
    let condition: WeatherCondition   // Mapped from code
}

struct HourlyForecast: Identifiable {
    let id: UUID
    let date: Date
    let temperature: Double
    let humidity: Int
    let pressureMmHg: Double
    let precipitationChance: Int
    let condition: WeatherCondition
}

struct DailyForecast: Identifiable {
    let id: UUID
    let date: Date
    let temperatureHigh: Double
    let temperatureLow: Double
    let precipitationChance: Int
    let condition: WeatherCondition
    let sunrise: Date
    let sunset: Date
}
```

### 1.3 WMO Weather Code Mapping

```swift
// WMO Weather interpretation codes (WW)
// https://open-meteo.com/en/docs

enum WMOWeatherCode: Int {
    case clearSky = 0
    case mainlyClear = 1
    case partlyCloudy = 2
    case overcast = 3
    case fog = 45
    case depositingRimeFog = 48
    case drizzleLight = 51
    case drizzleModerate = 53
    case drizzleDense = 55
    case freezingDrizzleLight = 56
    case freezingDrizzleDense = 57
    case rainSlight = 61
    case rainModerate = 63
    case rainHeavy = 65
    case freezingRainLight = 66
    case freezingRainHeavy = 67
    case snowFallSlight = 71
    case snowFallModerate = 73
    case snowFallHeavy = 75
    case snowGrains = 77
    case rainShowersSlight = 80
    case rainShowersModerate = 81
    case rainShowersViolent = 82
    case snowShowersSlight = 85
    case snowShowersHeavy = 86
    case thunderstorm = 95
    case thunderstormSlightHail = 96
    case thunderstormHeavyHail = 99

    var condition: WeatherCondition { ... }
    var hasPrecipitation: Bool { ... }
    var isStormy: Bool { ... }
}
```

---

## Phase 2: Pressure History & Delta Calculation

### 2.1 PressureHistoryManager.swift

**Location:** `Core/Services/PressureHistoryManager.swift`

**Critical for AS Flare Prediction:**
- Rapid pressure drops (>5 mmHg in 12h) correlate with increased inflammation
- Need to track pressure over 7+ days for accurate deltas
- Must persist across app restarts

```swift
// Design:
// - File-based storage (JSON in Documents directory)
// - Atomic writes to prevent corruption
// - Automatic cleanup of data older than 7 days
// - Batched saves (every 5 readings) to reduce I/O
// - Thread-safe with actor isolation

struct PressureReading: Codable {
    let timestamp: Date
    let pressureMmHg: Double
    let source: PressureSource  // .api, .manual, .watch
}

actor PressureHistoryManager {
    // Store readings
    func record(_ pressure: Double, source: PressureSource) async

    // Calculate deltas
    func change(over hours: Int) async -> Double?  // nil if insufficient data
    func change3h() async -> Double?
    func change6h() async -> Double?
    func change12h() async -> Double?
    func change24h() async -> Double?

    // Trend analysis
    func trend() async -> PressureTrend  // .rising, .stable, .falling, .rapidDrop
    func isAcceleratingDrop() async -> Bool  // Critical flare indicator!

    // Data access
    func readings(last hours: Int) async -> [PressureReading]
    func readings(from: Date, to: Date) async -> [PressureReading]
}
```

### 2.2 Pressure Trend Analysis

```swift
struct PressureTrend {
    let direction: TrendDirection      // .rising, .stable, .falling
    let rate3h: Double                 // mmHg per hour (last 3h)
    let rate6h: Double                 // mmHg per hour (last 6h)
    let rate12h: Double                // mmHg per hour (last 12h)
    let isAccelerating: Bool           // Drop rate increasing?
    let flareRiskContribution: Double  // 0.0 - 0.5

    enum TrendDirection {
        case rising
        case stable
        case falling
        case rapidDrop    // >5 mmHg in 12h
        case extremeDrop  // >10 mmHg in 12h
    }
}
```

---

## Phase 3: Caching Strategy

### 3.1 In-Memory Cache

```swift
// Cache current weather for 15 minutes
// Cache hourly forecast for 30 minutes
// Cache daily forecast for 1 hour

actor WeatherCache {
    struct CachedData<T> {
        let data: T
        let timestamp: Date
        let ttl: TimeInterval

        var isValid: Bool {
            Date().timeIntervalSince(timestamp) < ttl
        }
    }

    private var currentWeather: CachedData<CurrentWeather>?
    private var hourlyForecast: CachedData<[HourlyForecast]>?
    private var dailyForecast: CachedData<[DailyForecast]>?

    // TTL values
    static let currentTTL: TimeInterval = 900      // 15 min
    static let hourlyTTL: TimeInterval = 1800      // 30 min
    static let dailyTTL: TimeInterval = 3600       // 1 hour
}
```

### 3.2 Offline Fallback

```swift
// When network unavailable:
// 1. Return cached data if still valid
// 2. Return stale cached data with warning flag
// 3. Return last known good data from Core Data
// 4. Return nil with appropriate error

struct WeatherResult {
    let data: CurrentWeather?
    let source: DataSource
    let staleness: TimeInterval?  // How old is this data?
    let warning: String?

    enum DataSource {
        case live           // Fresh from API
        case cached         // From memory cache
        case stale          // Expired cache
        case persisted      // From Core Data
        case unavailable    // No data available
    }
}
```

---

## Phase 4: Background Updates

### 4.1 Automatic Refresh Strategy

```swift
// Update triggers:
// 1. Timer: Every 15 minutes when app is active
// 2. App foreground: When returning from background
// 3. Significant location change: If moved >1km
// 4. Manual: User pull-to-refresh
// 5. Watch request: When Watch requests sync

class WeatherUpdateScheduler {
    private var updateTimer: Timer?

    func startAutomaticUpdates() {
        // Timer every 15 minutes
        updateTimer = Timer.scheduledTimer(withTimeInterval: 900, repeats: true) { _ in
            Task { await OpenMeteoService.shared.fetchCurrentWeather() }
        }

        // App lifecycle observers
        NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            ...
        )
    }
}
```

### 4.2 Background App Refresh

```swift
// Use BGTaskScheduler for periodic background updates
// Critical for pressure history continuity

// In AppDelegate/SceneDelegate:
BGTaskScheduler.shared.register(
    forTaskWithIdentifier: "com.inflamai.weather.refresh",
    using: nil
) { task in
    self.handleWeatherRefresh(task: task as! BGAppRefreshTask)
}

// Schedule next refresh
func scheduleWeatherRefresh() {
    let request = BGAppRefreshTaskRequest(identifier: "com.inflamai.weather.refresh")
    request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)  // 15 min
    try? BGTaskScheduler.shared.submit(request)
}
```

---

## Phase 5: Core Data Integration

### 5.1 ContextSnapshot Entity Updates

**Existing Entity:** `ContextSnapshot`

```swift
// Fields already in ContextSnapshot:
// - timestamp: Date
// - barometricPressure: Double
// - pressureChange12h: Double
// - humidity: Int16
// - temperature: Double
// - precipitation: Bool

// Additional fields to add (if not present):
// - pressureChange3h: Double
// - pressureChange24h: Double
// - weatherCode: Int16
// - uvIndex: Double
// - windSpeed: Double
// - dataSource: String  // "open_meteo", "manual", "watch"
```

### 5.2 Weather Sync to Core Data

```swift
// After each weather fetch, store in Core Data for:
// 1. ML pipeline consumption
// 2. Historical correlation analysis
// 3. Offline access
// 4. Chart visualization

extension OpenMeteoService {
    func syncToContextSnapshot(_ weather: CurrentWeather) async {
        let context = PersistenceController.shared.container.viewContext

        await context.perform {
            let snapshot = ContextSnapshot(context: context)
            snapshot.timestamp = weather.timestamp
            snapshot.barometricPressure = weather.pressureMmHg
            snapshot.pressureChange12h = weather.pressureChange12h
            snapshot.humidity = Int16(weather.humidity)
            snapshot.temperature = weather.temperature
            snapshot.precipitation = weather.precipitation > 0

            try? context.save()
        }
    }
}
```

---

## Phase 6: Flare Risk Assessment

### 6.1 Enhanced Flare Risk Algorithm

```swift
struct FlareRiskAssessment {
    let level: FlareRiskLevel          // .low, .moderate, .high, .critical
    let score: Double                  // 0.0 - 1.0
    let factors: [RiskFactor]
    let recommendation: String
    let confidence: Double             // Based on data quality

    struct RiskFactor {
        let name: String
        let contribution: Double       // How much this adds to score
        let severity: Severity
        let icon: String
    }
}

// Risk calculation:
func assessFlareRisk(weather: CurrentWeather, trend: PressureTrend) -> FlareRiskAssessment {
    var score = 0.0
    var factors: [RiskFactor] = []

    // 1. PRESSURE DROP (Most important - 40% weight)
    if trend.isAccelerating && trend.direction == .rapidDrop {
        score += 0.40
        factors.append(.init(name: "Accelerating pressure drop", contribution: 0.40, severity: .critical))
    } else if weather.pressureChange12h < -5 {
        score += 0.30
        factors.append(.init(name: "Rapid 12h drop (\(weather.pressureChange12h) mmHg)", contribution: 0.30, severity: .high))
    } else if weather.pressureChange12h < -3 {
        score += 0.15
        factors.append(.init(name: "Moderate pressure drop", contribution: 0.15, severity: .moderate))
    }

    // 2. HUMIDITY (20% weight)
    if weather.humidity > 80 {
        score += 0.20
        factors.append(.init(name: "Very high humidity (\(weather.humidity)%)", contribution: 0.20, severity: .high))
    } else if weather.humidity > 70 {
        score += 0.10
        factors.append(.init(name: "High humidity (\(weather.humidity)%)", contribution: 0.10, severity: .moderate))
    }

    // 3. TEMPERATURE (15% weight)
    if weather.temperature < 5 {
        score += 0.15
        factors.append(.init(name: "Cold temperature (\(weather.temperature)°C)", contribution: 0.15, severity: .high))
    } else if weather.temperature < 10 {
        score += 0.10
        factors.append(.init(name: "Cool temperature (\(weather.temperature)°C)", contribution: 0.10, severity: .moderate))
    }

    // 4. PRECIPITATION (10% weight)
    if weather.condition.hasPrecipitation {
        score += 0.10
        factors.append(.init(name: weather.condition.displayName, contribution: 0.10, severity: .low))
    }

    // 5. LOW ABSOLUTE PRESSURE (10% weight)
    if weather.pressureMmHg < 745 {  // Storm system
        score += 0.10
        factors.append(.init(name: "Low pressure system", contribution: 0.10, severity: .moderate))
    }

    // 6. STORM CONDITIONS (5% weight)
    if weather.condition.isStormy {
        score += 0.05
        factors.append(.init(name: "Storm conditions", contribution: 0.05, severity: .moderate))
    }

    // Determine level
    let level: FlareRiskLevel = switch score {
        case 0.7...: .critical
        case 0.5..<0.7: .high
        case 0.3..<0.5: .moderate
        default: .low
    }

    return FlareRiskAssessment(level: level, score: score, factors: factors, ...)
}
```

---

## Phase 7: Watch App Integration

### 7.1 Weather Data for Watch

```swift
// Watch needs lightweight weather data for:
// 1. Complications (pressure, temp, risk level)
// 2. Quick log context (auto-attach weather)
// 3. Flare risk display

struct WatchWeatherData: Codable {
    let timestamp: Date
    let temperature: Double
    let pressureMmHg: Double
    let pressureChange12h: Double
    let humidity: Int
    let flareRiskLevel: String       // "low", "moderate", "high"
    let flareRiskScore: Double
    let conditionIcon: String        // SF Symbol name
}

// Sync via WatchConnectivity
extension WatchConnectivityService {
    func sendWeatherToWatch(_ weather: CurrentWeather, risk: FlareRiskAssessment) {
        let watchData = WatchWeatherData(
            timestamp: weather.timestamp,
            temperature: weather.temperature,
            pressureMmHg: weather.pressureMmHg,
            pressureChange12h: weather.pressureChange12h,
            humidity: weather.humidity,
            flareRiskLevel: risk.level.rawValue,
            flareRiskScore: risk.score,
            conditionIcon: weather.condition.iconName
        )

        // Use transferUserInfo for background delivery
        session.transferUserInfo(["weather": watchData.encoded()])
    }
}
```

### 7.2 Watch Complications

```swift
// Pressure complication data
struct PressureComplicationData {
    let current: Double              // mmHg
    let trend: TrendDirection        // arrow indicator
    let riskColor: Color             // green/yellow/orange/red
}
```

---

## Phase 8: ML Pipeline Integration

### 8.1 BiometricsCollectionService Update

```swift
// Environmental metrics from OpenMeteoService
private func collectEnvironmentalMetrics() async -> [String: BiometricStream] {
    var streams: [String: BiometricStream] = [:]

    // Fetch from OpenMeteoService
    if let weather = await OpenMeteoService.shared.currentWeather {
        streams["ambient_temperature"] = BiometricStream(
            value: weather.temperature,
            isAvailable: true,
            source: .openMeteo,
            timestamp: weather.timestamp,
            isImputed: false
        )

        streams["ambient_humidity"] = BiometricStream(
            value: Double(weather.humidity),
            isAvailable: true,
            source: .openMeteo,
            timestamp: weather.timestamp,
            isImputed: false
        )

        streams["barometric_pressure"] = BiometricStream(
            value: weather.pressureMmHg,
            isAvailable: true,
            source: .openMeteo,
            timestamp: weather.timestamp,
            isImputed: false
        )

        streams["pressure_change_3h"] = BiometricStream(
            value: weather.pressureChange3h,
            isAvailable: weather.pressureChange3h != 0,
            source: .calculated,
            timestamp: weather.timestamp,
            isImputed: weather.pressureChange3h == 0
        )

        streams["pressure_change_12h"] = BiometricStream(
            value: weather.pressureChange12h,
            isAvailable: weather.pressureChange12h != 0,
            source: .calculated,
            timestamp: weather.timestamp,
            isImputed: weather.pressureChange12h == 0
        )

        // ... additional streams
    }

    return streams
}
```

---

## Phase 9: Error Handling

### 9.1 Comprehensive Error Types

```swift
enum OpenMeteoError: LocalizedError {
    // Network errors
    case networkUnavailable
    case requestTimeout
    case serverError(statusCode: Int)
    case rateLimited

    // Location errors
    case locationNotAuthorized
    case locationUnavailable
    case locationTimeout

    // Data errors
    case invalidResponse
    case decodingFailed(Error)
    case missingRequiredData(field: String)

    // Cache errors
    case cacheExpired
    case noCachedData

    var errorDescription: String? { ... }
    var recoverySuggestion: String? { ... }
    var isRecoverable: Bool { ... }
}
```

### 9.2 Retry Strategy

```swift
// Exponential backoff with jitter
struct RetryPolicy {
    let maxAttempts: Int = 3
    let baseDelay: TimeInterval = 1.0
    let maxDelay: TimeInterval = 30.0

    func delay(for attempt: Int) -> TimeInterval {
        let exponentialDelay = baseDelay * pow(2.0, Double(attempt - 1))
        let jitter = Double.random(in: 0...0.5)
        return min(exponentialDelay + jitter, maxDelay)
    }
}

func fetchWithRetry<T>(_ operation: () async throws -> T) async throws -> T {
    var lastError: Error?

    for attempt in 1...RetryPolicy().maxAttempts {
        do {
            return try await operation()
        } catch {
            lastError = error
            if !error.isRecoverable { throw error }
            try await Task.sleep(nanoseconds: UInt64(RetryPolicy().delay(for: attempt) * 1_000_000_000))
        }
    }

    throw lastError!
}
```

---

## Phase 10: Testing Strategy

### 10.1 Unit Tests

```swift
// Tests to implement:

// OpenMeteoServiceTests
- testFetchCurrentWeather_Success
- testFetchCurrentWeather_NetworkError
- testFetchCurrentWeather_InvalidResponse
- testFetchCurrentWeather_UsesCache
- testFetchCurrentWeather_CacheExpired

// PressureHistoryManagerTests
- testRecordPressure_SavesReading
- testChange12h_CalculatesCorrectly
- testChange12h_InsufficientData_ReturnsNil
- testTrend_DetectsRapidDrop
- testCleanup_RemovesOldReadings

// FlareRiskAssessmentTests
- testAssessRisk_LowConditions_ReturnsLow
- testAssessRisk_HighPressureDrop_ReturnsHigh
- testAssessRisk_MultipleFactors_CombinesCorrectly
- testAssessRisk_AcceleratingDrop_ReturnsCritical

// WeatherCacheTests
- testCache_StoresData
- testCache_ReturnsValid
- testCache_ExpiresCorrectly
- testCache_ClearsOnMemoryWarning
```

### 10.2 Mock Service for Testing

```swift
class MockOpenMeteoService: OpenMeteoServiceProtocol {
    var mockCurrentWeather: CurrentWeather?
    var mockHourlyForecast: [HourlyForecast] = []
    var mockError: Error?
    var fetchCallCount = 0

    func fetchCurrentWeather() async throws -> CurrentWeather {
        fetchCallCount += 1
        if let error = mockError { throw error }
        return mockCurrentWeather ?? .mock()
    }
}

extension CurrentWeather {
    static func mock(
        temperature: Double = 15.0,
        humidity: Int = 60,
        pressureMmHg: Double = 760.0,
        pressureChange12h: Double = 0.0
    ) -> CurrentWeather { ... }
}
```

---

## Phase 11: Migration Plan

### 11.1 Step-by-Step Migration

```
Week 1: Core Implementation
├── Day 1-2: OpenMeteoService.swift + Models
├── Day 3: PressureHistoryManager.swift
├── Day 4: WeatherCache.swift
└── Day 5: Unit tests for core components

Week 2: Integration
├── Day 1: Replace WeatherKitService calls with OpenMeteoService
├── Day 2: Update BiometricsCollectionService
├── Day 3: Update UI components (WeatherCardView, etc.)
├── Day 4: Watch app integration
└── Day 5: Integration testing

Week 3: Polish & Cleanup
├── Day 1-2: Background refresh implementation
├── Day 3: Error handling & offline mode
├── Day 4: Remove WeatherKit entitlements & old code
└── Day 5: Final testing & documentation
```

### 11.2 Feature Flags

```swift
// Gradual rollout with feature flags
struct FeatureFlags {
    static var useOpenMeteo: Bool {
        // Start with 0%, gradually increase
        UserDefaults.standard.bool(forKey: "feature.openmeteo.enabled")
    }
}

// In WeatherService factory:
static func createWeatherService() -> WeatherServiceProtocol {
    if FeatureFlags.useOpenMeteo {
        return OpenMeteoService.shared
    } else {
        return WeatherKitService.shared
    }
}
```

### 11.3 Rollback Plan

```
If issues arise:
1. Disable feature flag → Immediate rollback to WeatherKit
2. WeatherKit code remains in deprecated/ folder for 2 releases
3. Core Data schema remains backward compatible
4. Pressure history file format unchanged
```

---

## Phase 12: Files to Create/Modify

### New Files

```
Core/Services/
├── OpenMeteoService.swift           # Main service
├── PressureHistoryManager.swift     # Pressure tracking
├── WeatherCache.swift               # Caching layer
└── WeatherUpdateScheduler.swift     # Background updates

Core/Models/
├── OpenMeteoModels.swift            # API response models
├── OpenMeteoWeatherTypes.swift      # Weather condition types
└── FlareRiskAssessment.swift        # Risk calculation models

Core/Protocols/
└── WeatherServiceProtocol.swift     # Abstraction for testing
```

### Files to Modify

```
Core/Services/
├── BiometricsCollectionService.swift  # Use OpenMeteoService
└── ComprehensiveDataSyncService.swift # Use OpenMeteoService

Features/Weather/
├── WeatherFlareRiskViewModel.swift    # Use OpenMeteoService
└── WeatherFlareRiskCard.swift         # Update bindings

Watch App/
└── WatchConnectivityManager.swift     # Weather sync
```

### Files to Deprecate

```
Move to deprecated/:
├── WeatherKitService.swift
└── (any WeatherKit-specific code)

Remove entitlements:
├── WeatherKit capability
└── com.apple.weatherkit entitlement
```

---

## API Reference

### Open-Meteo Endpoints

```
Base: https://api.open-meteo.com/v1

Current + Forecast:
GET /forecast?latitude=52.52&longitude=13.41
    &current=temperature_2m,relative_humidity_2m,pressure_msl,...
    &hourly=temperature_2m,pressure_msl,...
    &daily=temperature_2m_max,temperature_2m_min,...
    &timezone=auto

Historical (if needed):
GET /archive?latitude=52.52&longitude=13.41
    &start_date=2024-01-01
    &end_date=2024-01-31
    &hourly=pressure_msl

Air Quality (optional enhancement):
GET /air-quality?latitude=52.52&longitude=13.41
    &current=pm10,pm2_5
```

### Rate Limits

```
- No API key required
- No hard rate limits for non-commercial use
- Recommended: Max 10,000 requests/day
- Cache aggressively to minimize requests
```

---

## Success Metrics

```
1. Reliability
   - 99.9% uptime (vs WeatherKit dependency)
   - <2s average response time
   - 100% offline capability with cache

2. Data Quality
   - Pressure accuracy: ±0.5 hPa
   - 12h delta calculation accuracy: ±1 mmHg
   - Flare risk correlation: Track against user-reported flares

3. Performance
   - Memory footprint: <5MB for weather data
   - Battery impact: Negligible (<1% daily)
   - Network usage: <100KB/day

4. Developer Experience
   - No entitlement setup required
   - Works in simulator
   - Easy to test with mocks
```

---

## Appendix: WMO Weather Codes

```
Code | Condition
-----|------------------
0    | Clear sky
1    | Mainly clear
2    | Partly cloudy
3    | Overcast
45   | Fog
48   | Depositing rime fog
51   | Light drizzle
53   | Moderate drizzle
55   | Dense drizzle
56   | Light freezing drizzle
57   | Dense freezing drizzle
61   | Slight rain
63   | Moderate rain
65   | Heavy rain
66   | Light freezing rain
67   | Heavy freezing rain
71   | Slight snow
73   | Moderate snow
75   | Heavy snow
77   | Snow grains
80   | Slight rain showers
81   | Moderate rain showers
82   | Violent rain showers
85   | Slight snow showers
86   | Heavy snow showers
95   | Thunderstorm
96   | Thunderstorm with slight hail
99   | Thunderstorm with heavy hail
```

---

**Document Version:** 1.0
**Created:** 2025-12-01
**Author:** Claude Code
**Status:** Ready for Implementation
