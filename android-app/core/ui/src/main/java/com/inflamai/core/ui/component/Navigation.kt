package com.inflamai.core.ui.component

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.inflamai.core.ui.theme.*

/**
 * Bottom navigation item data
 */
data class BottomNavItem(
    val route: String,
    val label: String,
    val icon: ImageVector,
    val selectedIcon: ImageVector = icon,
    val badgeCount: Int = 0
)

/**
 * Main bottom navigation bar
 * Based on Frame Analysis: 5 tabs - Home, Trends, Log, Meds, More
 */
@Composable
fun InflamAIBottomNavBar(
    items: List<BottomNavItem>,
    selectedRoute: String,
    onItemSelected: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    NavigationBar(
        modifier = modifier,
        containerColor = Surface,
        contentColor = TextPrimary,
        tonalElevation = 8.dp
    ) {
        items.forEach { item ->
            val isSelected = selectedRoute == item.route

            NavigationBarItem(
                selected = isSelected,
                onClick = { onItemSelected(item.route) },
                icon = {
                    BadgedBox(
                        badge = {
                            if (item.badgeCount > 0) {
                                Badge(
                                    containerColor = Error,
                                    contentColor = Color.White
                                ) {
                                    Text(
                                        text = if (item.badgeCount > 9) "9+" else item.badgeCount.toString(),
                                        fontSize = 10.sp
                                    )
                                }
                            }
                        }
                    ) {
                        Icon(
                            imageVector = if (isSelected) item.selectedIcon else item.icon,
                            contentDescription = item.label,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                },
                label = {
                    Text(
                        text = item.label,
                        fontSize = 11.sp,
                        fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal
                    )
                },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = PrimaryBlue,
                    selectedTextColor = PrimaryBlue,
                    unselectedIconColor = TextTertiary,
                    unselectedTextColor = TextTertiary,
                    indicatorColor = PrimaryBlueLight
                )
            )
        }
    }
}

/**
 * Default navigation items for InflamAI
 */
object InflamAINavItems {
    val home = BottomNavItem(
        route = "home",
        label = "Home",
        icon = Icons.Default.Home
    )

    val trends = BottomNavItem(
        route = "trends",
        label = "Trends",
        icon = Icons.Default.TrendingUp
    )

    val log = BottomNavItem(
        route = "log",
        label = "Log",
        icon = Icons.Default.Add
    )

    val meds = BottomNavItem(
        route = "medication",
        label = "Meds",
        icon = Icons.Default.Medication
    )

    val more = BottomNavItem(
        route = "more",
        label = "More",
        icon = Icons.Default.MoreHoriz
    )

    val defaultItems = listOf(home, trends, log, meds, more)
}

/**
 * Top app bar with back navigation
 * Based on Frame Analysis: Minimal top bar with back arrow
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InflamAITopBar(
    title: String,
    onBackClick: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
    actions: @Composable RowScope.() -> Unit = {}
) {
    TopAppBar(
        title = {
            Text(
                text = title,
                fontSize = 18.sp,
                fontWeight = FontWeight.SemiBold,
                color = TextPrimary
            )
        },
        navigationIcon = {
            if (onBackClick != null) {
                IconButton(onClick = onBackClick) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = TextPrimary
                    )
                }
            }
        },
        actions = actions,
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = Background,
            titleContentColor = TextPrimary
        ),
        modifier = modifier
    )
}

/**
 * Transparent top bar for onboarding/splash screens
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TransparentTopBar(
    onBackClick: (() -> Unit)? = null,
    onSkipClick: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    TopAppBar(
        title = { },
        navigationIcon = {
            if (onBackClick != null) {
                IconButton(onClick = onBackClick) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = TextPrimary
                    )
                }
            }
        },
        actions = {
            if (onSkipClick != null) {
                TextButton(onClick = onSkipClick) {
                    Text(
                        text = "Skip",
                        color = TextSecondary,
                        fontSize = 16.sp
                    )
                }
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = Color.Transparent,
            titleContentColor = TextPrimary
        ),
        modifier = modifier
    )
}

/**
 * Large title top bar (like iOS navigation bar with large title)
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LargeTitleTopBar(
    title: String,
    subtitle: String? = null,
    onBackClick: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
    actions: @Composable RowScope.() -> Unit = {}
) {
    LargeTopAppBar(
        title = {
            Column {
                Text(
                    text = title,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                if (subtitle != null) {
                    Text(
                        text = subtitle,
                        fontSize = 14.sp,
                        color = TextSecondary
                    )
                }
            }
        },
        navigationIcon = {
            if (onBackClick != null) {
                IconButton(onClick = onBackClick) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = TextPrimary
                    )
                }
            }
        },
        actions = actions,
        colors = TopAppBarDefaults.largeTopAppBarColors(
            containerColor = Background,
            titleContentColor = TextPrimary
        ),
        modifier = modifier
    )
}

/**
 * Tab row for switching views
 * Based on Frame Analysis: Segment control style tabs
 */
@Composable
fun InflamAITabRow(
    tabs: List<String>,
    selectedIndex: Int,
    onTabSelected: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    TabRow(
        selectedTabIndex = selectedIndex,
        modifier = modifier,
        containerColor = BackgroundSecondary,
        contentColor = TextPrimary,
        indicator = { tabPositions ->
            if (selectedIndex < tabPositions.size) {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .wrapContentSize(Alignment.BottomStart)
                        .offset(x = tabPositions[selectedIndex].left)
                        .width(tabPositions[selectedIndex].width)
                        .height(3.dp)
                        .background(PrimaryBlue, RoundedCornerShape(topStart = 3.dp, topEnd = 3.dp))
                )
            }
        },
        divider = {}
    ) {
        tabs.forEachIndexed { index, title ->
            Tab(
                selected = selectedIndex == index,
                onClick = { onTabSelected(index) },
                text = {
                    Text(
                        text = title,
                        fontWeight = if (selectedIndex == index) FontWeight.SemiBold else FontWeight.Normal,
                        fontSize = 14.sp
                    )
                },
                selectedContentColor = PrimaryBlue,
                unselectedContentColor = TextSecondary
            )
        }
    }
}

/**
 * Quick action floating button (for Quick Log)
 * Based on Frame Analysis: Center FAB on bottom nav
 */
@Composable
fun QuickLogFAB(
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    FloatingActionButton(
        onClick = onClick,
        modifier = modifier.size(56.dp),
        containerColor = PrimaryBlue,
        contentColor = Color.White,
        shape = CircleShape
    ) {
        Icon(
            imageVector = Icons.Default.Add,
            contentDescription = "Quick Log",
            modifier = Modifier.size(28.dp)
        )
    }
}

/**
 * Settings row item
 * Based on Frame Analysis: Icon, label, optional value, chevron
 */
@Composable
fun SettingsRow(
    icon: ImageVector,
    title: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    iconBackgroundColor: Color = IconCircleBlue,
    subtitle: String? = null,
    value: String? = null,
    showChevron: Boolean = true
) {
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        color = Surface
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(iconBackgroundColor),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = when (iconBackgroundColor) {
                        IconCircleBlue -> PrimaryBlue
                        IconCircleGreen -> Success
                        IconCirclePink -> AccentPink
                        IconCirclePurple -> AccentPurple
                        IconCircleOrange -> AccentOrange
                        IconCircleRed -> Error
                        else -> TextPrimary
                    },
                    modifier = Modifier.size(20.dp)
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    fontSize = 16.sp,
                    color = TextPrimary
                )
                if (subtitle != null) {
                    Text(
                        text = subtitle,
                        fontSize = 13.sp,
                        color = TextSecondary
                    )
                }
            }

            if (value != null) {
                Text(
                    text = value,
                    fontSize = 15.sp,
                    color = TextSecondary
                )
                Spacer(modifier = Modifier.width(8.dp))
            }

            if (showChevron) {
                Icon(
                    imageVector = Icons.Default.ChevronRight,
                    contentDescription = null,
                    tint = TextTertiary,
                    modifier = Modifier.size(20.dp)
                )
            }
        }
    }
}

/**
 * Settings toggle row
 */
@Composable
fun SettingsToggleRow(
    icon: ImageVector,
    title: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    iconBackgroundColor: Color = IconCircleBlue,
    subtitle: String? = null
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        color = Surface
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(iconBackgroundColor),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = when (iconBackgroundColor) {
                        IconCircleBlue -> PrimaryBlue
                        IconCircleGreen -> Success
                        IconCirclePink -> AccentPink
                        IconCirclePurple -> AccentPurple
                        IconCircleOrange -> AccentOrange
                        else -> TextPrimary
                    },
                    modifier = Modifier.size(20.dp)
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    fontSize = 16.sp,
                    color = TextPrimary
                )
                if (subtitle != null) {
                    Text(
                        text = subtitle,
                        fontSize = 13.sp,
                        color = TextSecondary
                    )
                }
            }

            Switch(
                checked = checked,
                onCheckedChange = onCheckedChange,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Color.White,
                    checkedTrackColor = PrimaryBlue,
                    uncheckedThumbColor = Color.White,
                    uncheckedTrackColor = BackgroundTertiary
                )
            )
        }
    }
}

/**
 * More menu section header
 */
@Composable
fun MenuSectionHeader(
    title: String,
    modifier: Modifier = Modifier
) {
    Text(
        text = title.uppercase(),
        fontSize = 12.sp,
        fontWeight = FontWeight.SemiBold,
        color = TextTertiary,
        letterSpacing = 1.sp,
        modifier = modifier.padding(horizontal = 16.dp, vertical = 8.dp)
    )
}
