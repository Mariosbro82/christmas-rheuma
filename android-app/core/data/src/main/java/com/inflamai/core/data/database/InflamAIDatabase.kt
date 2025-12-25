package com.inflamai.core.data.database

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.inflamai.core.data.database.converter.Converters
import com.inflamai.core.data.database.dao.*
import com.inflamai.core.data.database.entity.*

/**
 * Main Room database for InflamAI
 * Equivalent to iOS InflamAI.xcdatamodeld Core Data model
 *
 * Contains all 18 entities for comprehensive AS management:
 * - Symptom tracking (SymptomLog, BodyRegionLog, ContextSnapshot)
 * - Medication management (Medication, DoseLog)
 * - Exercise tracking (ExerciseSession, Exercise)
 * - Flare management (FlareEvent)
 * - Pattern analysis (TriggerLog, TriggerAnalysisCache)
 * - User data (UserProfile)
 * - Meditation (MeditationSession, MeditationStreak)
 */
@Database(
    entities = [
        SymptomLogEntity::class,
        BodyRegionLogEntity::class,
        ContextSnapshotEntity::class,
        MedicationEntity::class,
        DoseLogEntity::class,
        FlareEventEntity::class,
        ExerciseSessionEntity::class,
        ExerciseEntity::class,
        UserProfileEntity::class,
        MeditationSessionEntity::class,
        MeditationStreakEntity::class,
        TriggerLogEntity::class,
        TriggerAnalysisCacheEntity::class
    ],
    version = 1,
    exportSchema = true
)
@TypeConverters(Converters::class)
abstract class InflamAIDatabase : RoomDatabase() {

    // DAOs
    abstract fun symptomLogDao(): SymptomLogDao
    abstract fun bodyRegionLogDao(): BodyRegionLogDao
    abstract fun contextSnapshotDao(): ContextSnapshotDao
    abstract fun medicationDao(): MedicationDao
    abstract fun doseLogDao(): DoseLogDao
    abstract fun flareEventDao(): FlareEventDao
    abstract fun exerciseSessionDao(): ExerciseSessionDao
    abstract fun exerciseDao(): ExerciseDao
    abstract fun userProfileDao(): UserProfileDao
    abstract fun meditationSessionDao(): MeditationSessionDao
    abstract fun meditationStreakDao(): MeditationStreakDao
    abstract fun triggerLogDao(): TriggerLogDao
    abstract fun triggerAnalysisCacheDao(): TriggerAnalysisCacheDao

    companion object {
        const val DATABASE_NAME = "inflamai_database"
    }
}
