package com.inflamai.feature.bodymap.component

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import com.inflamai.core.data.database.entity.BodyRegion
import com.inflamai.core.data.database.entity.BodyRegionCategory
import com.inflamai.core.ui.theme.InflamAIColors
import com.inflamai.core.ui.theme.PrimaryBlue
import com.inflamai.core.ui.theme.TextTertiary
import com.inflamai.feature.bodymap.viewmodel.BodyMapViewMode
import com.inflamai.feature.bodymap.viewmodel.RegionPainData
import com.inflamai.feature.bodymap.viewmodel.TimeRange
import kotlin.math.sqrt

/**
 * Body Map Canvas
 *
 * Interactive 47-region body map with pain visualization.
 * Supports front, back, and spine views.
 */
@Composable
fun BodyMapCanvas(
    viewMode: BodyMapViewMode,
    regionPainData: Map<String, RegionPainData>,
    selectedRegionId: String?,
    timeRange: TimeRange,
    onRegionTap: (BodyRegion) -> Unit,
    modifier: Modifier = Modifier
) {
    // Get regions to display based on view mode
    val visibleRegions = remember(viewMode) {
        when (viewMode) {
            BodyMapViewMode.FRONT -> BodyRegion.entries.filter {
                it.category != BodyRegionCategory.SPINE ||
                it == BodyRegion.CHEST ||
                it.id.startsWith("ribs")
            }
            BodyMapViewMode.BACK -> BodyRegion.entries.filter {
                it.category == BodyRegionCategory.SPINE ||
                it.category == BodyRegionCategory.UPPER_EXTREMITY ||
                it.category == BodyRegionCategory.LOWER_EXTREMITY
            }
            BodyMapViewMode.SPINE -> BodyRegion.spineRegions()
        }
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .aspectRatio(0.5f)
            .background(Color.Transparent)
    ) {
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(visibleRegions) {
                    detectTapGestures { offset ->
                        val width = size.width.toFloat()
                        val height = size.height.toFloat()

                        // Find tapped region
                        visibleRegions.forEach { region ->
                            val pos = getRegionPosition(region, viewMode)
                            val regionX = pos.first * width
                            val regionY = pos.second * height
                            val hitRadius = getHitRadius(region) * width

                            val distance = sqrt(
                                (offset.x - regionX) * (offset.x - regionX) +
                                (offset.y - regionY) * (offset.y - regionY)
                            )

                            if (distance <= hitRadius) {
                                onRegionTap(region)
                                return@detectTapGestures
                            }
                        }
                    }
                }
        ) {
            val width = size.width
            val height = size.height

            // Draw body silhouette
            drawBodySilhouette(width, height, viewMode)

            // Draw pain regions
            visibleRegions.forEach { region ->
                val pos = getRegionPosition(region, viewMode)
                val centerX = pos.first * width
                val centerY = pos.second * height
                val radius = getHitRadius(region) * width

                val isSelected = region.id == selectedRegionId
                val painData = regionPainData[region.id]
                val painLevel = getPainLevelForTimeRange(painData, timeRange)
                val painColor = getPainColor(painLevel)

                // Draw region circle if has pain or is selected
                if (painLevel > 0 || isSelected) {
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
    }
}

/**
 * Get region position based on view mode
 */
private fun getRegionPosition(region: BodyRegion, viewMode: BodyMapViewMode): Pair<Float, Float> {
    return when (region) {
        // Cervical Spine
        BodyRegion.CERVICAL_C1 -> 0.5f to 0.08f
        BodyRegion.CERVICAL_C2 -> 0.5f to 0.09f
        BodyRegion.CERVICAL_C3 -> 0.5f to 0.10f
        BodyRegion.CERVICAL_C4 -> 0.5f to 0.11f
        BodyRegion.CERVICAL_C5 -> 0.5f to 0.12f
        BodyRegion.CERVICAL_C6 -> 0.5f to 0.13f
        BodyRegion.CERVICAL_C7 -> 0.5f to 0.14f

        // Thoracic Spine
        BodyRegion.THORACIC_T1 -> 0.5f to 0.16f
        BodyRegion.THORACIC_T2 -> 0.5f to 0.18f
        BodyRegion.THORACIC_T3 -> 0.5f to 0.20f
        BodyRegion.THORACIC_T4 -> 0.5f to 0.22f
        BodyRegion.THORACIC_T5 -> 0.5f to 0.24f
        BodyRegion.THORACIC_T6 -> 0.5f to 0.26f
        BodyRegion.THORACIC_T7 -> 0.5f to 0.28f
        BodyRegion.THORACIC_T8 -> 0.5f to 0.30f
        BodyRegion.THORACIC_T9 -> 0.5f to 0.32f
        BodyRegion.THORACIC_T10 -> 0.5f to 0.34f
        BodyRegion.THORACIC_T11 -> 0.5f to 0.36f
        BodyRegion.THORACIC_T12 -> 0.5f to 0.38f

        // Lumbar Spine
        BodyRegion.LUMBAR_L1 -> 0.5f to 0.40f
        BodyRegion.LUMBAR_L2 -> 0.5f to 0.42f
        BodyRegion.LUMBAR_L3 -> 0.5f to 0.44f
        BodyRegion.LUMBAR_L4 -> 0.5f to 0.46f
        BodyRegion.LUMBAR_L5 -> 0.5f to 0.48f

        // Sacrum & SI Joints
        BodyRegion.SACRUM -> 0.5f to 0.50f
        BodyRegion.SI_JOINT_LEFT -> 0.42f to 0.50f
        BodyRegion.SI_JOINT_RIGHT -> 0.58f to 0.50f

        // Upper Extremities
        BodyRegion.SHOULDER_LEFT -> 0.25f to 0.18f
        BodyRegion.SHOULDER_RIGHT -> 0.75f to 0.18f
        BodyRegion.ELBOW_LEFT -> 0.18f to 0.34f
        BodyRegion.ELBOW_RIGHT -> 0.82f to 0.34f
        BodyRegion.WRIST_LEFT -> 0.14f to 0.46f
        BodyRegion.WRIST_RIGHT -> 0.86f to 0.46f
        BodyRegion.HAND_LEFT -> 0.12f to 0.52f
        BodyRegion.HAND_RIGHT -> 0.88f to 0.52f

        // Lower Extremities
        BodyRegion.HIP_LEFT -> 0.35f to 0.54f
        BodyRegion.HIP_RIGHT -> 0.65f to 0.54f
        BodyRegion.KNEE_LEFT -> 0.38f to 0.72f
        BodyRegion.KNEE_RIGHT -> 0.62f to 0.72f
        BodyRegion.ANKLE_LEFT -> 0.38f to 0.90f
        BodyRegion.ANKLE_RIGHT -> 0.62f to 0.90f
        BodyRegion.FOOT_LEFT -> 0.36f to 0.96f
        BodyRegion.FOOT_RIGHT -> 0.64f to 0.96f

        // Thorax
        BodyRegion.CHEST -> 0.5f to 0.25f
        BodyRegion.RIBS_LEFT -> 0.35f to 0.28f
        BodyRegion.RIBS_RIGHT -> 0.65f to 0.28f
    }
}

/**
 * Get hit radius for region
 */
private fun getHitRadius(region: BodyRegion): Float {
    return when (region.category) {
        BodyRegionCategory.SPINE -> 0.03f
        BodyRegionCategory.UPPER_EXTREMITY -> when {
            region.id.contains("shoulder") -> 0.06f
            region.id.contains("elbow") -> 0.05f
            region.id.contains("wrist") -> 0.04f
            region.id.contains("hand") -> 0.05f
            else -> 0.04f
        }
        BodyRegionCategory.LOWER_EXTREMITY -> when {
            region.id.contains("hip") -> 0.06f
            region.id.contains("knee") -> 0.05f
            region.id.contains("ankle") -> 0.04f
            region.id.contains("foot") -> 0.05f
            else -> 0.04f
        }
        BodyRegionCategory.THORAX -> 0.06f
    }
}

/**
 * Get pain level based on selected time range
 */
private fun getPainLevelForTimeRange(painData: RegionPainData?, timeRange: TimeRange): Int {
    if (painData == null) return 0

    // If there's a current pain level set, use that
    painData.currentPainLevel?.let { return it }

    // Otherwise use the average for the time range
    val average = when (timeRange) {
        TimeRange.DAYS_7 -> painData.averagePain7Day
        TimeRange.DAYS_30 -> painData.averagePain30Day
        TimeRange.DAYS_90 -> painData.averagePain90Day
    }

    return average.toInt()
}

/**
 * Get color for pain level (0-10)
 */
private fun getPainColor(painLevel: Int): Color {
    return when {
        painLevel == 0 -> Color.Transparent
        painLevel <= 2 -> InflamAIColors.PainMild
        painLevel <= 4 -> InflamAIColors.PainModerate
        painLevel <= 6 -> InflamAIColors.PainSevere
        else -> InflamAIColors.PainExtreme
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
        moveTo(width * 0.25f, height * 0.15f)
        lineTo(width * 0.75f, height * 0.15f)
        lineTo(width * 0.70f, height * 0.52f)
        lineTo(width * 0.30f, height * 0.52f)
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

    // Spine line (if back or spine view)
    if (viewMode == BodyMapViewMode.BACK || viewMode == BodyMapViewMode.SPINE) {
        drawLine(
            color = TextTertiary,
            start = Offset(width * 0.5f, height * 0.08f),
            end = Offset(width * 0.5f, height * 0.50f),
            strokeWidth = 3.dp.toPx()
        )
    }
}
