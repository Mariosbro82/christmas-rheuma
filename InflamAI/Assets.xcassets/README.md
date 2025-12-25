# Assets Catalog

This catalog contains all app assets managed through `AssetsManager.swift`.

## Structure

```
Assets.xcassets/
├── Colors/           # Color sets (Primary, Secondary, etc.)
├── AppIcon.appiconset/  # App icon (all sizes)
├── Images/           # Illustrations and graphics
├── BodyDiagrams/     # Body map images
└── Exercise/         # Exercise thumbnails
```

## Color Sets (Planned)

### Brand Colors
- `Primary` - Main brand blue
- `Secondary` - Brand purple
- `Accent` - Accent green

### Pain Levels
- `PainLow` - Green (0-2)
- `PainMild` - Yellow (3-4)
- `PainModerate` - Orange (5-6)
- `PainSevere` - Red-orange (7-8)
- `PainCritical` - Dark red (9-10)

### BASDAI Scores
- `BASDAILow` - Green (0-2)
- `BASDAIMild` - Yellow (2-4)
- `BASDAIModerate` - Orange (4-6)
- `BASDAIHigh` - Red (6+)

### UI Colors
- `Background`
- `SecondaryBackground`
- `CardBackground`
- `PrimaryText`
- `SecondaryText`

### Semantic
- `Success`
- `Warning`
- `Error`
- `Info`

## Usage

### In SwiftUI Views

```swift
import SwiftUI

struct MyView: View {
    var body: some View {
        VStack {
            Text("Hello")
                .foregroundColor(AssetsManager.Colors.primaryText)
                .padding()
                .background(AssetsManager.Colors.cardBackground)
                .cardStyle()  // Helper extension

            Button("Action") {
                // action
            }
            .primaryButtonStyle()  // Helper extension
        }
        .background(AssetsManager.Gradients.background)
    }
}
```

### Pain Color Helpers

```swift
let painLevel = 7.5
let color = AssetsManager.Colors.forPainLevel(painLevel)  // Returns painSevere

let basdai = 3.2
let basdaiColor = AssetsManager.Colors.forBASDAI(basdai)  // Returns basdaiMild
```

### SF Symbols

```swift
Image(systemName: AssetsManager.Symbols.medication)
Image(systemName: AssetsManager.Symbols.chart)
Image(systemName: AssetsManager.Symbols.ai)
```

### Spacing & Layout

```swift
VStack(spacing: AssetsManager.Spacing.md) {
    // content
}
.padding(AssetsManager.Spacing.lg)
.cornerRadius(AssetsManager.CornerRadius.md)
```

## Adding New Assets

### 1. Add Color Set to Assets.xcassets
1. Open Assets.xcassets in Xcode
2. Right-click → New Color Set
3. Name it (e.g., "NewFeatureColor")
4. Set Light/Dark mode variants

### 2. Reference in AssetsManager.swift

```swift
enum Colors {
    static let newFeature = Color("NewFeatureColor", bundle: nil) ?? Color.blue
}
```

### 3. Use in Views

```swift
.foregroundColor(AssetsManager.Colors.newFeature)
```

## Benefits

✅ **Type-safe** - Compile-time checking of asset names
✅ **Centralized** - All assets in one place
✅ **Dark mode** - Automatic support via Color Sets
✅ **Fallbacks** - Default colors if asset missing
✅ **Consistent** - Enforces design system
✅ **Helper functions** - Pain/BASDAI color mapping
✅ **Extensions** - Reusable button/card styles

## Current Status

**AssetsManager.swift**: ✅ Created (430 lines)
**Color Sets**: ⏳ To be added to Assets.xcassets
**Images**: ⏳ To be added as needed
**App Icon**: ⏳ To be designed

All assets currently use fallback colors/SF Symbols until actual assets are added to the catalog.
