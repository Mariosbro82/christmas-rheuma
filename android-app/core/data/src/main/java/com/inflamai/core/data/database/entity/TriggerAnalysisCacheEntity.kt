package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Cached trigger analysis results from pattern analysis
 */
@Entity(tableName = "trigger_analysis_cache")
data class TriggerAnalysisCacheEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),

    // Trigger identification
    val triggerType: String,
    val triggerName: String,

    // Correlation statistics
    val correlationCoefficient: Double,
    val pValue: Double,
    val confidenceLevel: ConfidenceLevel,
    val sampleSize: Int,

    // Lag analysis (JSON: {0: r, 12: r, 24: r, 48: r, 72: r})
    val lagAnalysisJson: String = "{}",
    val strongestLagHours: Int = 0,

    // Effect description
    val effectDirection: EffectDirection,
    val averageEffectSize: Double,
    val effectDescription: String,
    val recommendation: String,

    // Analysis metadata
    val analysisDate: Long = System.currentTimeMillis(),
    val dataRangeStartDate: Long,
    val dataRangeEndDate: Long,
    val dataPointsAnalyzed: Int,

    // Validity
    val isValid: Boolean = true,
    val expiresAt: Long,

    val createdAt: Long = System.currentTimeMillis()
)

enum class ConfidenceLevel {
    WEAK,      // |r| < 0.3
    MODERATE,  // 0.3 <= |r| < 0.5
    STRONG,    // 0.5 <= |r| < 0.7
    VERY_STRONG // |r| >= 0.7
}

enum class EffectDirection {
    INCREASES_SYMPTOMS,
    DECREASES_SYMPTOMS,
    NO_EFFECT
}
