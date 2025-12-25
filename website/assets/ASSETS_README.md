# InflamAI Website Assets

This folder contains all visual assets for the InflamAI marketing website.

## Required Assets

### Icons (`/icons/`)
- `favicon.png` - 32x32px favicon
- `favicon-16.png` - 16x16px small favicon
- `apple-touch-icon.png` - 180x180px Apple touch icon
- `icon-192.png` - 192x192px PWA icon
- `icon-512.png` - 512x512px PWA icon

### Images (`/images/`)

#### App Screenshots (Required)
These should be actual screenshots from the InflamAI iOS app:

1. `app-dashboard.png` - Home/Dashboard screen
2. `app-bodymap.png` - Body map with pain regions
3. `app-checkin.png` - BASDAI check-in screen
4. `app-trends.png` - Trends/analytics view
5. `app-exercises.png` - Exercise library
6. `app-medications.png` - Medication tracking
7. `app-sos.png` - JointTap SOS flare capture
8. `app-reports.png` - PDF export preview

**Recommended dimensions:** 390x844px (iPhone 14 Pro resolution)
**Format:** PNG with transparent background or device frame

#### Team Photos
- `team-fabian.jpg` - Fabian Harnisch headshot
  - Recommended: 400x400px, professional quality
  - Will be displayed in a circular frame

#### Marketing Graphics
- `og-image.png` - Open Graph social sharing image (1200x630px)
- `hero-mockup.png` - Optional custom hero device mockup
- `feature-icons/` - Custom feature icons if not using SVGs

### Logo Files
- `logo-full.svg` - Full logo with text
- `logo-icon.svg` - Icon only (medical cross)
- `logo-white.svg` - White version for dark backgrounds
- `logo-dark.svg` - Dark version for light backgrounds

## Color Reference

```
Primary Blue:    #007AFF
Accent Cyan:     #00D9FF
Accent Teal:     #00B8A9
Success Green:   #34C759
Alert Red:       #FF3B30
Warning Orange:  #FF9500
```

## Export Guidelines

### Screenshots
1. Use iPhone 15 Pro simulator in Xcode
2. Enable dark mode in app if available
3. Use "Capture Screenshot" (Cmd+S in Simulator)
4. Crop to remove status bar if needed
5. Optimize with ImageOptim or similar

### Photos
- JPEG format, 80-90% quality
- Max 200KB per image
- sRGB color space

### Icons/Graphics
- SVG preferred for scalability
- PNG fallback at 2x resolution
- Optimize with SVGO for SVGs

## Notes

- All images should be optimized for web (compressed)
- Consider providing 2x versions for retina displays
- Update og-image.png when branding changes
- Keep file names lowercase with hyphens
