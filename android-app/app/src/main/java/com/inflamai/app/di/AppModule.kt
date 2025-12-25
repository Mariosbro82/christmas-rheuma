package com.inflamai.app.di

import android.content.Context
import androidx.room.Room
import com.inflamai.core.data.database.InflamAIDatabase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Main Hilt module for application-level dependencies
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideDatabase(
        @ApplicationContext context: Context
    ): InflamAIDatabase {
        return Room.databaseBuilder(
            context,
            InflamAIDatabase::class.java,
            InflamAIDatabase.DATABASE_NAME
        )
            .fallbackToDestructiveMigration()
            .build()
    }

    // DAOs
    @Provides
    @Singleton
    fun provideSymptomLogDao(database: InflamAIDatabase) = database.symptomLogDao()

    @Provides
    @Singleton
    fun provideBodyRegionLogDao(database: InflamAIDatabase) = database.bodyRegionLogDao()

    @Provides
    @Singleton
    fun provideContextSnapshotDao(database: InflamAIDatabase) = database.contextSnapshotDao()

    @Provides
    @Singleton
    fun provideMedicationDao(database: InflamAIDatabase) = database.medicationDao()

    @Provides
    @Singleton
    fun provideDoseLogDao(database: InflamAIDatabase) = database.doseLogDao()

    @Provides
    @Singleton
    fun provideFlareEventDao(database: InflamAIDatabase) = database.flareEventDao()

    @Provides
    @Singleton
    fun provideExerciseSessionDao(database: InflamAIDatabase) = database.exerciseSessionDao()

    @Provides
    @Singleton
    fun provideExerciseDao(database: InflamAIDatabase) = database.exerciseDao()

    @Provides
    @Singleton
    fun provideUserProfileDao(database: InflamAIDatabase) = database.userProfileDao()

    @Provides
    @Singleton
    fun provideMeditationSessionDao(database: InflamAIDatabase) = database.meditationSessionDao()

    @Provides
    @Singleton
    fun provideMeditationStreakDao(database: InflamAIDatabase) = database.meditationStreakDao()

    @Provides
    @Singleton
    fun provideTriggerLogDao(database: InflamAIDatabase) = database.triggerLogDao()

    @Provides
    @Singleton
    fun provideTriggerAnalysisCacheDao(database: InflamAIDatabase) = database.triggerAnalysisCacheDao()
}
