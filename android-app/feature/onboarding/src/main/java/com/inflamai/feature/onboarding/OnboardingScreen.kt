package com.inflamai.feature.onboarding

import androidx.annotation.DrawableRes
import androidx.compose.animation.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.inflamai.core.ui.component.*
import com.inflamai.core.ui.theme.*
import com.inflamai.feature.onboarding.R
import kotlinx.coroutines.launch

/**
 * Onboarding page data
 * Based on Frame Analysis: 11 onboarding pages with mascot, title, features
 */
data class OnboardingPage(
    val title: String,
    val subtitle: String,
    val description: String,
    val icon: ImageVector? = null,
    val iconBackgroundColor: Color = IconCircleBlue,
    @DrawableRes val imageRes: Int? = null,
    val features: List<String> = emptyList()
)

/**
 * All onboarding pages based on Frame Analysis
 * Uses mascot images from iOS app where available
 */
val onboardingPages = listOf(
    // Page 1: Welcome - Dinosaur mascot with confetti
    OnboardingPage(
        title = "Welcome to InflamAI",
        subtitle = "Your Personal AS Companion",
        description = "Track symptoms, discover patterns, and take control of your ankylosing spondylitis journey.",
        imageRes = R.drawable.onboarding_welcome,
        features = listOf(
            "Track Scientifically",
            "Discover Patterns",
            "Live Better"
        )
    ),
    // Page 2: Understanding AS - Education about AS
    OnboardingPage(
        title = "Understanding AS",
        subtitle = "Knowledge is your superpower",
        description = "Ankylosing Spondylitis affects your spine, but knowledge helps you manage it effectively.",
        imageRes = R.drawable.onboarding_understanding_as,
        features = listOf(
            "What is AS?",
            "Flares & Remission",
            "Great News"
        )
    ),
    // Page 3: Daily Check-In - Quick logging
    OnboardingPage(
        title = "Daily Check-In",
        subtitle = "Just 60 seconds a day",
        description = "Quick daily check-ins help you understand your health better over time.",
        imageRes = R.drawable.onboarding_daily_checkin,
        features = listOf(
            "BASDAI Score",
            "Takes < 1 Minute",
            "No typing required"
        )
    ),
    // Page 4: Medication Tracking
    OnboardingPage(
        title = "Medication Tracking",
        subtitle = "Never miss a dose",
        description = "Track adherence and see how medications affect your symptoms.",
        imageRes = R.drawable.onboarding_medication,
        features = listOf(
            "Smart Reminders",
            "Adherence Tracking",
            "Correlation Analysis"
        )
    ),
    // Page 5: Flare Support
    OnboardingPage(
        title = "Flare Support",
        subtitle = "We're here to help",
        description = "When flares hit, we help you log quickly and find patterns.",
        imageRes = R.drawable.onboarding_flare_support,
        features = listOf(
            "JointTap SOS",
            "Flare Timeline",
            "Trigger Detection"
        )
    ),
    // Page 6: Trends & Reports
    OnboardingPage(
        title = "Trends & Reports",
        subtitle = "Beautiful charts for your doctor",
        description = "Export beautiful PDF reports and see your BASDAI, pain, and symptoms over time.",
        imageRes = R.drawable.onboarding_trends,
        features = listOf(
            "Visual Trends",
            "PDF Reports",
            "Custom Date Ranges"
        )
    ),
    // Page 7: Body Map - Uses icon (no dedicated image)
    OnboardingPage(
        title = "Interactive Body Map",
        subtitle = "47 anatomical regions",
        description = "Pinpoint exactly where you're experiencing pain with our detailed body map.",
        icon = Icons.Default.Accessibility,
        iconBackgroundColor = IconCircleTeal,
        features = listOf(
            "Spine: C1-L5 vertebrae",
            "SI joints tracking",
            "Peripheral joints",
            "Heat map visualization"
        )
    ),
    // Page 8: Exercise Library - Uses icon
    OnboardingPage(
        title = "Exercise Library",
        subtitle = "52 AS-specific exercises",
        description = "Stay mobile with exercises designed specifically for ankylosing spondylitis.",
        icon = Icons.Default.FitnessCenter,
        iconBackgroundColor = IconCircleGreen,
        features = listOf(
            "Stretching routines",
            "Strengthening exercises",
            "Breathing techniques",
            "Step-by-step guidance"
        )
    ),
    // Page 9: Privacy
    OnboardingPage(
        title = "Your Privacy Matters",
        subtitle = "100% on-device processing",
        description = "Your health data never leaves your device. No third-party analytics. No cloud storage.",
        icon = Icons.Default.Security,
        iconBackgroundColor = IconCircleBlue,
        features = listOf(
            "Zero third-party SDKs",
            "Biometric lock",
            "Optional backup",
            "GDPR compliant"
        )
    ),
    // Page 10: Health Connect
    OnboardingPage(
        title = "Health Connect",
        subtitle = "Integrate your health data",
        description = "Connect with Health Connect to automatically import sleep, heart rate, and activity data.",
        icon = Icons.Default.MonitorHeart,
        iconBackgroundColor = IconCircleGreen,
        features = listOf(
            "Sleep tracking",
            "Heart rate variability",
            "Step count",
            "Exercise sessions"
        )
    ),
    // Page 11: Get Started
    OnboardingPage(
        title = "Ready to Begin?",
        subtitle = "Your journey starts now",
        description = "Set up your profile and start tracking. Remember: consistency is key!",
        icon = Icons.Default.RocketLaunch,
        iconBackgroundColor = IconCircleOrange
    )
)

/**
 * Main Onboarding Screen
 * Based on Frame Analysis: Pager with mascot, title, description, features, dots, CTA
 */
@Composable
fun OnboardingScreen(
    viewModel: OnboardingViewModel = hiltViewModel(),
    onComplete: () -> Unit = {},
    onSkip: () -> Unit = {}
) {
    val uiState by viewModel.uiState.collectAsState()
    val pagerState = rememberPagerState(pageCount = { onboardingPages.size })
    val coroutineScope = rememberCoroutineScope()
    val isLastPage = pagerState.currentPage == onboardingPages.size - 1

    // Sync pager with viewModel
    LaunchedEffect(pagerState.currentPage) {
        viewModel.setStep(pagerState.currentPage)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Background)
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            // Top bar with skip
            TransparentTopBar(
                onSkipClick = if (!isLastPage) {
                    {
                        viewModel.completeOnboarding()
                        onSkip()
                    }
                } else null
            )

            // Pager
            HorizontalPager(
                state = pagerState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) { page ->
                OnboardingPageContent(
                    page = onboardingPages[page],
                    modifier = Modifier.fillMaxSize()
                )
            }

            // Bottom section
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Page indicator
                PageIndicator(
                    pageCount = onboardingPages.size,
                    currentPage = pagerState.currentPage
                )

                Spacer(modifier = Modifier.height(32.dp))

                // Navigation buttons
                if (isLastPage) {
                    PrimaryButton(
                        text = "Get Started",
                        onClick = {
                            viewModel.completeOnboarding()
                            onComplete()
                        }
                    )
                } else {
                    NavigationButtonRow(
                        onPrevious = if (pagerState.currentPage > 0) {
                            {
                                coroutineScope.launch {
                                    pagerState.animateScrollToPage(pagerState.currentPage - 1)
                                }
                            }
                        } else null,
                        onNext = {
                            coroutineScope.launch {
                                pagerState.animateScrollToPage(pagerState.currentPage + 1)
                            }
                        },
                        previousText = "Back",
                        nextText = "Next"
                    )
                }
            }
        }
    }
}

/**
 * Single onboarding page content
 * Supports both drawable images (mascot) and vector icons
 */
@Composable
fun OnboardingPageContent(
    page: OnboardingPage,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Mascot Image or Icon
        if (page.imageRes != null) {
            // Full mascot illustration from iOS app
            Image(
                painter = painterResource(id = page.imageRes),
                contentDescription = page.title,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp)
                    .clip(RoundedCornerShape(16.dp)),
                contentScale = ContentScale.Fit
            )
        } else if (page.icon != null) {
            // Fallback to icon in circular background
            Box(
                modifier = Modifier
                    .size(120.dp)
                    .clip(RoundedCornerShape(60.dp))
                    .background(page.iconBackgroundColor),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = page.icon,
                    contentDescription = null,
                    tint = when (page.iconBackgroundColor) {
                        IconCircleBlue -> PrimaryBlue
                        IconCircleGreen -> Success
                        IconCirclePink -> AccentPink
                        IconCirclePurple -> AccentPurple
                        IconCircleOrange -> AccentOrange
                        IconCircleTeal -> AccentTeal
                        IconCircleCream -> Warning
                        else -> TextPrimary
                    },
                    modifier = Modifier.size(56.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Title
        Text(
            text = page.title,
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Subtitle
        Text(
            text = page.subtitle,
            fontSize = 16.sp,
            color = PrimaryBlue,
            fontWeight = FontWeight.Medium,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Description
        Text(
            text = page.description,
            fontSize = 15.sp,
            color = TextSecondary,
            textAlign = TextAlign.Center,
            lineHeight = 22.sp
        )

        // Features list (if any)
        if (page.features.isNotEmpty()) {
            Spacer(modifier = Modifier.height(20.dp))

            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                page.features.forEach { feature ->
                    FeatureRow(text = feature)
                }
            }
        }
    }
}

/**
 * Feature row with checkmark
 */
@Composable
private fun FeatureRow(text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
    ) {
        Box(
            modifier = Modifier
                .size(24.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(SuccessLight),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.Check,
                contentDescription = null,
                tint = Success,
                modifier = Modifier.size(16.dp)
            )
        }

        Spacer(modifier = Modifier.width(12.dp))

        Text(
            text = text,
            fontSize = 15.sp,
            color = TextPrimary
        )
    }
}
