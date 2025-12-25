/**
 * InflamAI Website - Main JavaScript
 * Handles navigation, animations, and form interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initScrollAnimations();
    initWaitlistForm();
    initSmoothScroll();
    initNavbarScroll();
});

/**
 * Mobile Navigation Toggle
 */
function initNavigation() {
    const navToggle = document.getElementById('nav-toggle');
    const navLinks = document.getElementById('nav-links');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            navToggle.classList.toggle('active');
        });

        // Close menu when clicking a link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                navToggle.classList.remove('active');
            });
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
                navLinks.classList.remove('active');
                navToggle.classList.remove('active');
            }
        });
    }
}

/**
 * Navbar scroll effect
 */
function initNavbarScroll() {
    const navbar = document.getElementById('navbar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        lastScroll = currentScroll;
    });
}

/**
 * Intersection Observer for scroll animations
 */
function initScrollAnimations() {
    const animatedElements = document.querySelectorAll(
        '.problem-card, .solution-card, .tech-feature, .feature-card, ' +
        '.science-card, .market-stat, .team-card, .timeline-item'
    );

    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Add stagger delay based on index within its container
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, index * 100);
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    animatedElements.forEach((el, index) => {
        el.setAttribute('data-animate', '');
        el.style.transitionDelay = `${(index % 4) * 0.1}s`;
        observer.observe(el);
    });

    // Section headers
    const sectionHeaders = document.querySelectorAll('.section-header, .vision-content');
    sectionHeaders.forEach(header => {
        header.setAttribute('data-animate', '');
        observer.observe(header);
    });
}

/**
 * Waitlist Form Handler
 */
function initWaitlistForm() {
    const form = document.getElementById('waitlist-form');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const emailInput = form.querySelector('input[type="email"]');
            const submitBtn = form.querySelector('button[type="submit"]');
            const email = emailInput.value.trim();

            if (!email || !isValidEmail(email)) {
                showFormMessage(form, 'Bitte geben Sie eine gültige E-Mail-Adresse ein.', 'error');
                return;
            }

            // Disable form while submitting
            emailInput.disabled = true;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span>Wird gesendet...</span>';

            try {
                // Simulate API call (replace with actual endpoint)
                await simulateApiCall(email);

                showFormMessage(form, 'Vielen Dank! Sie sind jetzt auf der Warteliste.', 'success');
                emailInput.value = '';

                // Track conversion (if analytics available)
                if (typeof gtag === 'function') {
                    gtag('event', 'waitlist_signup', {
                        event_category: 'engagement',
                        event_label: 'waitlist'
                    });
                }
            } catch (error) {
                showFormMessage(form, 'Ein Fehler ist aufgetreten. Bitte versuchen Sie es später erneut.', 'error');
            } finally {
                emailInput.disabled = false;
                submitBtn.disabled = false;
                submitBtn.innerHTML = `
                    Auf die Warteliste
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <path d="M4 10H16M16 10L11 5M16 10L11 15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                `;
            }
        });
    }
}

/**
 * Email validation helper
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Show form message
 */
function showFormMessage(form, message, type) {
    // Remove existing message
    const existingMessage = form.querySelector('.form-message');
    if (existingMessage) {
        existingMessage.remove();
    }

    // Create new message
    const messageEl = document.createElement('div');
    messageEl.className = `form-message form-message--${type}`;
    messageEl.textContent = message;
    messageEl.style.cssText = `
        padding: 12px 16px;
        margin-top: 12px;
        border-radius: 8px;
        font-size: 14px;
        text-align: center;
        background: ${type === 'success' ? 'rgba(52, 199, 89, 0.2)' : 'rgba(255, 59, 48, 0.2)'};
        color: ${type === 'success' ? '#34C759' : '#FF3B30'};
        border: 1px solid ${type === 'success' ? 'rgba(52, 199, 89, 0.3)' : 'rgba(255, 59, 48, 0.3)'};
    `;

    form.appendChild(messageEl);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        messageEl.remove();
    }, 5000);
}

/**
 * Simulate API call (replace with actual implementation)
 */
function simulateApiCall(email) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            // Store locally for demo purposes
            const waitlist = JSON.parse(localStorage.getItem('inflamai_waitlist') || '[]');
            if (!waitlist.includes(email)) {
                waitlist.push(email);
                localStorage.setItem('inflamai_waitlist', JSON.stringify(waitlist));
            }
            resolve();
        }, 1000);
    });
}

/**
 * Smooth scroll for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');

            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                const navbarHeight = document.getElementById('navbar').offsetHeight;
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navbarHeight;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Parallax effect for hero section (subtle)
 */
function initParallax() {
    const heroVisual = document.querySelector('.hero-visual');

    if (heroVisual && window.innerWidth > 768) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * 0.3;

            if (scrolled < window.innerHeight) {
                heroVisual.style.transform = `translateY(${rate}px)`;
            }
        });
    }
}

/**
 * Counter animation for stats
 */
function animateCounters() {
    const counters = document.querySelectorAll('.stat-value');

    counters.forEach(counter => {
        const target = counter.textContent;
        const isPercent = target.includes('%');
        const isK = target.includes('K');
        const isM = target.includes('M');

        let numericValue = parseFloat(target.replace(/[^0-9.]/g, ''));

        if (isNaN(numericValue)) return;

        let suffix = '';
        if (isPercent) suffix = '%';
        if (isK) suffix = 'K';
        if (isM) suffix = 'M';

        let current = 0;
        const increment = numericValue / 50;
        const duration = 1500;
        const stepTime = duration / 50;

        const timer = setInterval(() => {
            current += increment;
            if (current >= numericValue) {
                current = numericValue;
                clearInterval(timer);
            }

            if (Number.isInteger(numericValue)) {
                counter.textContent = Math.floor(current) + suffix;
            } else {
                counter.textContent = current.toFixed(1) + suffix;
            }
        }, stepTime);
    });
}

/**
 * Lazy load images
 */
function initLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');

    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                imageObserver.unobserve(img);
            }
        });
    });

    images.forEach(img => imageObserver.observe(img));
}

/**
 * Theme switcher (future feature)
 */
function initThemeSwitcher() {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');

    // The site is dark by default, but this could be extended
    // to support a light mode in the future
}

/**
 * Performance monitoring
 */
function logPerformance() {
    if ('performance' in window) {
        window.addEventListener('load', () => {
            setTimeout(() => {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page Load Time:', Math.round(perfData.loadEventEnd - perfData.fetchStart), 'ms');
            }, 0);
        });
    }
}

// Initialize performance logging in development
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    logPerformance();
}
