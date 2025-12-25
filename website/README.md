# InflamAI Marketing Website

A modern, responsive marketing website for InflamAI - the AI-powered rheumatism flare prediction app.

## Quick Start

### Local Development

Open the website directly in a browser:
```bash
cd website
open index.html
```

Or use a local server (recommended for full functionality):
```bash
# Using Python
python3 -m http.server 8000

# Using Node.js (npx)
npx serve

# Using PHP
php -S localhost:8000
```

Then visit: `http://localhost:8000`

## Structure

```
website/
├── index.html              # Main HTML file
├── css/
│   └── styles.css          # All styles (dark theme)
├── js/
│   └── main.js             # JavaScript interactions
├── assets/
│   ├── images/             # Screenshots, photos
│   ├── icons/              # Favicon, app icons
│   └── ASSETS_README.md    # Asset guidelines
├── WEBSITE_KNOWLEDGE_BASE.md  # Content reference
└── README.md               # This file
```

## Features

- **Dark theme** matching the pitch deck aesthetic
- **Fully responsive** - works on all devices
- **Smooth animations** on scroll
- **Mobile navigation** with hamburger menu
- **Waitlist form** with localStorage demo
- **Accessibility** - semantic HTML, keyboard nav

## Sections

1. **Hero** - Main value proposition with app mockup
2. **Problem** - 3 pain points for patients
3. **Vision** - "Verstehen, vorhersagen, vorbeugen"
4. **Solution** - 4 key feature pillars
5. **Technology** - Privacy, data points, UI
6. **Features** - 8 feature cards
7. **Science** - Research backing (ActConnect, WearablesPro)
8. **Market** - Growth statistics
9. **Team** - Current team + open positions
10. **Roadmap** - Q4 2025 - Q3 2026
11. **CTA** - Waitlist signup
12. **Footer** - Links, legal, social

## Customization

### Colors
Edit CSS variables in `css/styles.css`:
```css
:root {
    --color-accent-cyan: #00D9FF;
    --color-accent-blue: #007AFF;
    /* ... */
}
```

### Content
All content is in German. To change:
1. Edit text directly in `index.html`
2. Reference `WEBSITE_KNOWLEDGE_BASE.md` for messaging

### Images
1. Add app screenshots to `assets/images/`
2. Update image references in HTML
3. See `assets/ASSETS_README.md` for specs

## Deployment

### Static Hosting (Recommended)
- **Netlify**: Drag & drop the `website` folder
- **Vercel**: `vercel deploy`
- **GitHub Pages**: Push to `gh-pages` branch

### Custom Domain
1. Configure DNS (A record or CNAME)
2. Update meta tags for SEO
3. Add SSL certificate (usually automatic)

## Before Launch Checklist

- [ ] Add real app screenshots
- [ ] Add team photo for Fabian
- [ ] Configure actual email collection (e.g., Mailchimp, ConvertKit)
- [ ] Set up analytics (Plausible, Fathom - privacy-friendly)
- [ ] Create privacy policy page (`/datenschutz`)
- [ ] Create imprint page (`/impressum`)
- [ ] Test on multiple devices
- [ ] Run Lighthouse audit (aim for 90+)
- [ ] Submit to search engines

## Tech Stack

- **HTML5** - Semantic markup
- **CSS3** - Custom properties, Grid, Flexbox
- **Vanilla JS** - No dependencies
- **Google Fonts** - Inter font family

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

Copyright 2025 InflamAI. All rights reserved.
