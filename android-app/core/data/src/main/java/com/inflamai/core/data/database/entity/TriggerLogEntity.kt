package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.UUID

/**
 * Trigger event logging for pattern analysis
 * Used by CorrelationEngine for statistical analysis
 */
@Entity(
    tableName = "trigger_logs",
    foreignKeys = [
        ForeignKey(
            entity = SymptomLogEntity::class,
            parentColumns = ["id"],
            childColumns = ["symptomLogId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index("symptomLogId"),
        Index("triggerType"),
        Index("timestamp")
    ]
)
data class TriggerLogEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val symptomLogId: String? = null,
    val timestamp: Long = System.currentTimeMillis(),

    // Trigger Details
    val triggerType: String,             // Category/type
    val triggerName: String,             // Specific trigger
    val triggerValue: Double? = null,    // Numeric value if applicable
    val triggerDescription: String? = null,

    // Impact
    val impactLevel: Int? = null,        // 1-10 perceived impact
    val confidenceLevel: Int? = null,    // 1-10 confidence it's a trigger

    // Timing
    val hoursBeforeSymptoms: Int? = null,
    val lagCategory: LagCategory = LagCategory.SAME_DAY,

    val lastModified: Long = System.currentTimeMillis()
)

enum class LagCategory {
    IMMEDIATE,      // 0-2 hours
    SAME_DAY,       // 2-12 hours
    NEXT_DAY,       // 12-24 hours
    DELAYED         // 24+ hours
}

/**
 * Cached trigger analysis results
 * Stores computed correlations to avoid recalculating
 */
@Entity(
    tableName = "trigger_analysis_cache",
    indices = [Index("analysisDate")]
)
data class TriggerAnalysisCacheEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val analysisDate: Long = System.currentTimeMillis(),

    // Analysis Period
    val startDate: Long,
    val endDate: Long,
    val dataPointsAnalyzed: Int,

    // Results (JSON)
    val correlationsJson: String,        // Array of CorrelationResult
    val topTriggersJson: String,         // Top N triggers
    val confidenceLevel: Double,         // Overall analysis confidence

    // Metadata
    val analysisVersion: String = "1.0",
    val isValid: Boolean = true,

    val lastModified: Long = System.currentTimeMillis()
)

/**
 * Data class representing a correlation result
 * Used in JSON serialization for cache
 */
data class CorrelationResult(
    val factor: String,
    val factorCategory: String,
    val correlationCoefficient: Double,  // Pearson r (-1 to 1)
    val pValue: Double,
    val isSignificant: Boolean,          // p < 0.05
    val sampleSize: Int,
    val lag: Int,                         // Hours
    val direction: CorrelationDirection
)

enum class CorrelationDirection {
    POSITIVE,    // Higher value = more symptoms
    NEGATIVE,    // Higher value = fewer symptoms
    NONE
}
