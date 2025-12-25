package com.inflamai.core.ui.component

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.inflamai.core.ui.theme.*

/**
 * Body region data class
 * Based on Frame Analysis: 47 anatomical regions for pain tracking
 */
data class BodyRegion(
    val id: String,
    val name: String,
    val shortName: String,
    val category: RegionCategory,
    val side: RegionSide,
    // Relative position (0-1 range) on the body silhouette
    val centerX: Float,
    val centerY: Float,
    val hitRadius: Float = 0.04f, // Hit target radius relative to body width
    val painLevel: Int = 0 // 0-10
)

enum class RegionCategory {
    CERVICAL_SPINE,    // C1-C7
    THORACIC_SPINE,    // T1-T12
    LUMBAR_SPINE,      // L1-L5
    SACROILIAC,        // SI joints
    SHOULDER,
    ELBOW,
    WRIST,
    HAND,
    HIP,
    KNEE,
    ANKLE,
    FOOT
}

enum class RegionSide {
    CENTER,
    LEFT,
    RIGHT
}

/**
 * All 47 body regions with anatomical positions
 */
object BodyRegions {
    // Cervical Spine (7 vertebrae)
    val c1 = BodyRegion("c1", "C1 (Atlas)", "C1", RegionCategory.CERVICAL_SPINE, RegionSide.CENTER, 0.5f, 0.08f)
    val c2 = BodyRegion("c2", "C2 (Axis)", "C2", RegionCategory.CERVICAL_SPINE, RegionSide.CENTER, 0.5f, 0.09f)
    val c3 = BodyRegion("c3", "C3", "C3", RegionCategory.CERVICAL_SPINE, RegionSide.CENTER, 0.5f, 0.10f)
    val c4 = BodyRegion("c4", "C4", "C4", RegionCategory.CERVICAL_SPINE, RegionSide.CENTER, 0.5f, 0.11f)
    val c5 = BodyRegion("c5", "C5", "C5", RegionCategory.CERVICAL_SPINE, RegionSide.CENTER, 0.5f, 0.12f)
    val c6 = BodyRegion("c6", "C6", "C6", RegionCategory.CERVICAL_SPINE, RegionSide.CENTER, 0.5f, 0.13f)
    val c7 = BodyRegion("c7", "C7", "C7", RegionCategory.CERVICAL_SPINE, RegionSide.CENTER, 0.5f, 0.14f)

    // Thoracic Spine (12 vertebrae)
    val t1 = BodyRegion("t1", "T1", "T1", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.16f)
    val t2 = BodyRegion("t2", "T2", "T2", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.18f)
    val t3 = BodyRegion("t3", "T3", "T3", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.20f)
    val t4 = BodyRegion("t4", "T4", "T4", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.22f)
    val t5 = BodyRegion("t5", "T5", "T5", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.24f)
    val t6 = BodyRegion("t6", "T6", "T6", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.26f)
    val t7 = BodyRegion("t7", "T7", "T7", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.28f)
    val t8 = BodyRegion("t8", "T8", "T8", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.30f)
    val t9 = BodyRegion("t9", "T9", "T9", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.32f)
    val t10 = BodyRegion("t10", "T10", "T10", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.34f)
    val t11 = BodyRegion("t11", "T11", "T11", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.36f)
    val t12 = BodyRegion("t12", "T12", "T12", RegionCategory.THORACIC_SPINE, RegionSide.CENTER, 0.5f, 0.38f)

    // Lumbar Spine (5 vertebrae)
    val l1 = BodyRegion("l1", "L1", "L1", RegionCategory.LUMBAR_SPINE, RegionSide.CENTER, 0.5f, 0.40f)
    val l2 = BodyRegion("l2", "L2", "L2", RegionCategory.LUMBAR_SPINE, RegionSide.CENTER, 0.5f, 0.42f)
    val l3 = BodyRegion("l3", "L3", "L3", RegionCategory.LUMBAR_SPINE, RegionSide.CENTER, 0.5f, 0.44f)
    val l4 = BodyRegion("l4", "L4", "L4", RegionCategory.LUMBAR_SPINE, RegionSide.CENTER, 0.5f, 0.46f)
    val l5 = BodyRegion("l5", "L5", "L5", RegionCategory.LUMBAR_SPINE, RegionSide.CENTER, 0.5f, 0.48f)

    // SI Joints (bilateral)
    val siLeft = BodyRegion("si_left", "Left SI Joint", "SI-L", RegionCategory.SACROILIAC, RegionSide.LEFT, 0.42f, 0.50f)
    val siRight = BodyRegion("si_right", "Right SI Joint", "SI-R", RegionCategory.SACROILIAC, RegionSide.RIGHT, 0.58f, 0.50f)

    // Shoulders
    val shoulderLeft = BodyRegion("shoulder_left", "Left Shoulder", "Shoulder-L", RegionCategory.SHOULDER, RegionSide.LEFT, 0.25f, 0.18f, 0.06f)
    val shoulderRight = BodyRegion("shoulder_right", "Right Shoulder", "Shoulder-R", RegionCategory.SHOULDER, RegionSide.RIGHT, 0.75f, 0.18f, 0.06f)

    // Elbows
    val elbowLeft = BodyRegion("elbow_left", "Left Elbow", "Elbow-L", RegionCategory.ELBOW, RegionSide.LEFT, 0.18f, 0.34f, 0.05f)
    val elbowRight = BodyRegion("elbow_right", "Right Elbow", "Elbow-R", RegionCategory.ELBOW, RegionSide.RIGHT, 0.82f, 0.34f, 0.05f)

    // Wrists
    val wristLeft = BodyRegion("wrist_left", "Left Wrist", "Wrist-L", RegionCategory.WRIST, RegionSide.LEFT, 0.14f, 0.46f, 0.04f)
    val wristRight = BodyRegion("wrist_right", "Right Wrist", "Wrist-R", RegionCategory.WRIST, RegionSide.RIGHT, 0.86f, 0.46f, 0.04f)

    // Hands
    val handLeft = BodyRegion("hand_left", "Left Hand", "Hand-L", RegionCategory.HAND, RegionSide.LEFT, 0.12f, 0.52f, 0.05f)
    val handRight = BodyRegion("hand_right", "Right Hand", "Hand-R", RegionCategory.HAND, RegionSide.RIGHT, 0.88f, 0.52f, 0.05f)

    // Hips
    val hipLeft = BodyRegion("hip_left", "Left Hip", "Hip-L", RegionCategory.HIP, RegionSide.LEFT, 0.35f, 0.54f, 0.06f)
    val hipRight = BodyRegion("hip_right", "Right Hip", "Hip-R", RegionCategory.HIP, RegionSide.RIGHT, 0.65f, 0.54f, 0.06f)

    // Knees
    val kneeLeft = BodyRegion("knee_left", "Left Knee", "Knee-L", RegionCategory.KNEE, RegionSide.LEFT, 0.38f, 0.72f, 0.05f)
    val kneeRight = BodyRegion("knee_right", "Right Knee", "Knee-R", RegionCategory.KNEE, RegionSide.RIGHT, 0.62f, 0.72f, 0.05f)

    // Ankles
    val ankleLeft = BodyRegion("ankle_left", "Left Ankle", "Ankle-L", RegionCategory.ANKLE, RegionSide.LEFT, 0.38f, 0.90f, 0.04f)
    val ankleRight = BodyRegion("ankle_right", "Right Ankle", "Ankle-R", RegionCategory.ANKLE, RegionSide.RIGHT, 0.62f, 0.90f, 0.04f)

    // Feet
    val footLeft = BodyRegion("foot_left", "Left Foot", "Foot-L", RegionCategory.FOOT, RegionSide.LEFT, 0.36f, 0.96f, 0.05f)
    val footRight = BodyRegion("foot_right", "Right Foot", "Foot-R", RegionCategory.FOOT, RegionSide.RIGHT, 0.64f, 0.96f, 0.05f)

    val allRegions = listOf(
        // Spine
        c1, c2, c3, c4, c5, c6, c7,
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12,
        l1, l2, l3, l4, l5,
        siLeft, siRight,
        // Upper body
        shoulderLeft, shoulderRight,
        elbowLeft, elbowRight,
        wristLeft, wristRight,
        handLeft, handRight,
        // Lower body
        hipLeft, hipRight,
        kneeLeft, kneeRight,
        ankleLeft, ankleRight,
        footLeft, footRight
    )

    val spineRegions = listOf(c1, c2, c3, c4, c5, c6, c7, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, l1, l2, l3, l4, l5, siLeft, siRight)
    val peripheralRegions = allRegions - spineRegions.toSet()
}

/**
 * Interactive Body Map for pain tracking
 * Based on Frame Analysis: 47 tappable regions with color-coded pain levels
 */
@Composable
fun BodyMapView(
    regions: List<BodyRegion>,
    onRegionTapped: (BodyRegion) -> Unit,
    modifier: Modifier = Modifier,
    selectedRegionId: String? = null,
    showLabels: Boolean = false,
    viewMode: BodyMapViewMode = BodyMapViewMode.FRONT
) {
    var canvasSize by remember { mutableStateOf(Size.Zero) }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .aspectRatio(0.5f) // Tall body silhouette
            .background(Color.Transparent)
    ) {
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(regions) {
                    detectTapGestures { offset ->
                        val width = size.width.toFloat()
                        val height = size.height.toFloat()

                        // Find tapped region
                        regions.forEach { region ->
                            val regionX = region.centerX * width
                            val regionY = region.centerY * height
                            val hitRadius = region.hitRadius * width

                            val distance = kotlin.math.sqrt(
                                (offset.x - regionX) * (offset.x - regionX) +
                                        (offset.y - regionY) * (offset.y - regionY)
                            )

                            if (distance <= hitRadius) {
                                onRegionTapped(region)
                                return@detectTapGestures
                            }
                        }
                    }
                }
        ) {
            canvasSize = size
            val width = size.width
            val height = size.height

            // Draw body silhouette
            drawBodySilhouette(width, height, viewMode)

            // Draw pain regions
            regions.forEach { region ->
                val centerX = region.centerX * width
                val centerY = region.centerY * height
                val radius = region.hitRadius * width

                val isSelected = region.id == selectedRegionId
                val painColor = getPainColor(region.painLevel)

                // Draw region circle
                if (region.painLevel > 0 || isSelected) {
                    drawCircle(
                        color = painColor.copy(alpha = 0.7f),
                        radius = radius,
                        center = Offset(centerX, centerY)
                    )
                }

                // Draw selection ring
                if (isSelected) {
                    drawCircle(
                        color = PrimaryBlue,
                        radius = radius + 4.dp.toPx(),
                        center = Offset(centerX, centerY),
                        style = Stroke(width = 3.dp.toPx())
                    )
                }

                // Draw touch target outline (subtle)
                drawCircle(
                    color = TextTertiary.copy(alpha = 0.3f),
                    radius = radius,
                    center = Offset(centerX, centerY),
                    style = Stroke(width = 1.dp.toPx())
                )
            }
        }

        // Labels overlay
        if (showLabels) {
            regions.filter { it.painLevel > 0 }.forEach { region ->
                val offsetX = with(LocalDensity.current) {
                    (region.centerX * canvasSize.width).toDp() - 20.dp
                }
                val offsetY = with(LocalDensity.current) {
                    (region.centerY * canvasSize.height).toDp() - 10.dp
                }

                Text(
                    text = region.shortName,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary,
                    modifier = Modifier.offset(x = offsetX, y = offsetY)
                )
            }
        }
    }
}

enum class BodyMapViewMode {
    FRONT,
    BACK
}

/**
 * Get color for pain level (0-10)
 */
fun getPainColor(painLevel: Int): Color {
    return when {
        painLevel == 0 -> Color.Transparent
        painLevel <= 2 -> BodyMapColors.heatLow
        painLevel <= 4 -> BodyMapColors.heatMedium
        painLevel <= 6 -> BodyMapColors.heatHigh
        painLevel <= 8 -> BodyMapColors.heatExtreme
        else -> BodyMapColors.heatExtreme
    }
}

/**
 * Draw simplified body silhouette
 */
private fun DrawScope.drawBodySilhouette(
    width: Float,
    height: Float,
    viewMode: BodyMapViewMode
) {
    val silhouetteColor = Color(0xFFE0E0E0)

    // Head
    val headCenterX = width * 0.5f
    val headCenterY = height * 0.05f
    val headRadius = width * 0.08f

    drawCircle(
        color = silhouetteColor,
        radius = headRadius,
        center = Offset(headCenterX, headCenterY)
    )

    // Neck
    val neckPath = Path().apply {
        moveTo(width * 0.45f, headCenterY + headRadius * 0.8f)
        lineTo(width * 0.55f, headCenterY + headRadius * 0.8f)
        lineTo(width * 0.55f, height * 0.12f)
        lineTo(width * 0.45f, height * 0.12f)
        close()
    }
    drawPath(neckPath, silhouetteColor)

    // Torso
    val torsoPath = Path().apply {
        moveTo(width * 0.25f, height * 0.15f) // Left shoulder
        lineTo(width * 0.75f, height * 0.15f) // Right shoulder
        lineTo(width * 0.70f, height * 0.52f) // Right hip
        lineTo(width * 0.30f, height * 0.52f) // Left hip
        close()
    }
    drawPath(torsoPath, silhouetteColor)

    // Left arm
    val leftArmPath = Path().apply {
        moveTo(width * 0.25f, height * 0.15f)
        lineTo(width * 0.18f, height * 0.18f)
        lineTo(width * 0.12f, height * 0.35f)
        lineTo(width * 0.08f, height * 0.50f)
        lineTo(width * 0.14f, height * 0.52f)
        lineTo(width * 0.18f, height * 0.36f)
        lineTo(width * 0.22f, height * 0.20f)
        close()
    }
    drawPath(leftArmPath, silhouetteColor)

    // Right arm
    val rightArmPath = Path().apply {
        moveTo(width * 0.75f, height * 0.15f)
        lineTo(width * 0.82f, height * 0.18f)
        lineTo(width * 0.88f, height * 0.35f)
        lineTo(width * 0.92f, height * 0.50f)
        lineTo(width * 0.86f, height * 0.52f)
        lineTo(width * 0.82f, height * 0.36f)
        lineTo(width * 0.78f, height * 0.20f)
        close()
    }
    drawPath(rightArmPath, silhouetteColor)

    // Left leg
    val leftLegPath = Path().apply {
        moveTo(width * 0.30f, height * 0.52f)
        lineTo(width * 0.42f, height * 0.52f)
        lineTo(width * 0.40f, height * 0.75f)
        lineTo(width * 0.38f, height * 0.92f)
        lineTo(width * 0.32f, height * 0.98f)
        lineTo(width * 0.28f, height * 0.92f)
        lineTo(width * 0.30f, height * 0.75f)
        close()
    }
    drawPath(leftLegPath, silhouetteColor)

    // Right leg
    val rightLegPath = Path().apply {
        moveTo(width * 0.58f, height * 0.52f)
        lineTo(width * 0.70f, height * 0.52f)
        lineTo(width * 0.70f, height * 0.75f)
        lineTo(width * 0.72f, height * 0.92f)
        lineTo(width * 0.68f, height * 0.98f)
        lineTo(width * 0.62f, height * 0.92f)
        lineTo(width * 0.60f, height * 0.75f)
        close()
    }
    drawPath(rightLegPath, silhouetteColor)

    // Spine line (if back view)
    if (viewMode == BodyMapViewMode.BACK) {
        drawLine(
            color = TextTertiary,
            start = Offset(width * 0.5f, height * 0.08f),
            end = Offset(width * 0.5f, height * 0.50f),
            strokeWidth = 3.dp.toPx()
        )
    }
}

/**
 * Pain level selector bottom sheet content
 * Based on Frame Analysis: Slider with region name
 */
@Composable
fun PainLevelSelector(
    region: BodyRegion,
    currentLevel: Int,
    onLevelChanged: (Int) -> Unit,
    onConfirm: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Region name
        Text(
            text = region.name,
            fontSize = 20.sp,
            fontWeight = FontWeight.SemiBold,
            color = TextPrimary
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Category badge
        Surface(
            shape = RoundedCornerShape(8.dp),
            color = IconCircleBlue
        ) {
            Text(
                text = region.category.name.replace("_", " "),
                fontSize = 12.sp,
                color = PrimaryBlue,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Pain level display
        Text(
            text = "$currentLevel/10",
            fontSize = 48.sp,
            fontWeight = FontWeight.Bold,
            color = getPainColor(currentLevel).takeIf { currentLevel > 0 } ?: TextPrimary
        )

        Text(
            text = getPainLabel(currentLevel),
            fontSize = 16.sp,
            color = TextSecondary
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Slider
        Slider(
            value = currentLevel.toFloat(),
            onValueChange = { onLevelChanged(it.toInt()) },
            valueRange = 0f..10f,
            steps = 9,
            modifier = Modifier.fillMaxWidth(),
            colors = SliderDefaults.colors(
                thumbColor = getPainColor(currentLevel).takeIf { currentLevel > 0 } ?: PrimaryBlue,
                activeTrackColor = getPainColor(currentLevel).takeIf { currentLevel > 0 } ?: PrimaryBlue,
                inactiveTrackColor = BackgroundSecondary
            )
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = "No Pain", fontSize = 12.sp, color = TextTertiary)
            Text(text = "Severe", fontSize = 12.sp, color = TextTertiary)
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Confirm button
        PrimaryButton(
            text = "Save",
            onClick = onConfirm
        )
    }
}

/**
 * Get descriptive label for pain level
 */
fun getPainLabel(level: Int): String {
    return when {
        level == 0 -> "No Pain"
        level <= 2 -> "Mild"
        level <= 4 -> "Moderate"
        level <= 6 -> "Significant"
        level <= 8 -> "Severe"
        else -> "Extreme"
    }
}

/**
 * Body Map Legend
 */
@Composable
fun BodyMapLegend(
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        LegendItem(color = BodyMapColors.heatLow, label = "Mild")
        LegendItem(color = BodyMapColors.heatMedium, label = "Moderate")
        LegendItem(color = BodyMapColors.heatHigh, label = "High")
        LegendItem(color = BodyMapColors.heatExtreme, label = "Severe")
    }
}

@Composable
private fun LegendItem(
    color: Color,
    label: String
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Box(
            modifier = Modifier
                .size(12.dp)
                .background(color, RoundedCornerShape(2.dp))
        )
        Text(
            text = label,
            fontSize = 11.sp,
            color = TextSecondary
        )
    }
}
