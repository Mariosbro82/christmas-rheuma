package com.inflamai.core.data.di

import com.inflamai.core.data.database.dao.*
import com.inflamai.core.data.repository.*
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt module for providing repository instances
 */
@Module
@InstallIn(SingletonComponent::class)
object RepositoryModule {

    @Provides
    @Singleton
    fun provideSymptomRepository(
        symptomLogDao: SymptomLogDao,
        bodyRegionLogDao: BodyRegionLogDao,
        contextSnapshotDao: ContextSnapshotDao
    ): SymptomRepository = SymptomRepository(symptomLogDao, bodyRegionLogDao, contextSnapshotDao)

    @Provides
    @Singleton
    fun provideMedicationRepository(
        medicationDao: MedicationDao,
        doseLogDao: DoseLogDao
    ): MedicationRepository = MedicationRepository(medicationDao, doseLogDao)

    @Provides
    @Singleton
    fun provideFlareRepository(
        flareEventDao: FlareEventDao
    ): FlareRepository = FlareRepository(flareEventDao)

    @Provides
    @Singleton
    fun provideExerciseRepository(
        exerciseDao: ExerciseDao,
        exerciseSessionDao: ExerciseSessionDao
    ): ExerciseRepository = ExerciseRepository(exerciseDao, exerciseSessionDao)

    @Provides
    @Singleton
    fun provideMeditationRepository(
        meditationSessionDao: MeditationSessionDao,
        meditationStreakDao: MeditationStreakDao
    ): MeditationRepository = MeditationRepository(meditationSessionDao, meditationStreakDao)

    @Provides
    @Singleton
    fun provideUserRepository(
        userProfileDao: UserProfileDao
    ): UserRepository = UserRepository(userProfileDao)

    @Provides
    @Singleton
    fun provideTriggerRepository(
        triggerLogDao: TriggerLogDao,
        triggerAnalysisCacheDao: TriggerAnalysisCacheDao
    ): TriggerRepository = TriggerRepository(triggerLogDao, triggerAnalysisCacheDao)
}
