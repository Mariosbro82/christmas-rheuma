package com.inflamai.core.data.database.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * User profile and settings (singleton pattern)
 * Equivalent to iOS UserProfile Core Data entity
 */
@Entity(tableName = "user_profile")
data class UserProfileEntity(
    @PrimaryKey
    val id: String = "user_profile",  // Singleton - always same ID

    // Personal Information
    val name: String? = null,
    val dateOfBirth: Long? = null,
    val gender: Gender? = null,
    val heightCm: Float? = null,
    val weightKg: Float? = null,
    val bmi: Float? = null,

    // Diagnosis Information
    val diagnosisDate: Long? = null,
    val diagnosisType: DiagnosisType = DiagnosisType.ANKYLOSING_SPONDYLITIS,
    val hlaB27Positive: Boolean? = null,
    val yearsWithCondition: Int? = null,

    // Treatment History
    val biologicExperienced: Boolean = false,
    val currentBiologic: String? = null,
    val previousBiologicsJson: String = "[]",

    // Healthcare Providers
    val primaryPhysicianName: String? = null,
    val primaryPhysicianPhone: String? = null,
    val rheumatologistName: String? = null,
    val rheumatologistPhone: String? = null,
    val physicalTherapistName: String? = null,
    val physicalTherapistPhone: String? = null,

    // Health Data Integrations
    val healthConnectEnabled: Boolean = false,
    val weatherEnabled: Boolean = false,
    val locationEnabled: Boolean = false,

    // Sync Settings
    val cloudSyncEnabled: Boolean = false,
    val lastSyncDate: Long? = null,

    // Security
    val biometricLockEnabled: Boolean = true,
    val autoLockTimeout: Int = 0,  // 0 = immediate, seconds

    // Notifications
    val notificationsEnabled: Boolean = true,
    val dailyCheckInReminderEnabled: Boolean = true,
    val dailyCheckInReminderTime: String = "09:00", // HH:mm format
    val medicationRemindersEnabled: Boolean = true,
    val flareAlertsEnabled: Boolean = true,
    val exerciseRemindersEnabled: Boolean = false,

    // Preferences
    val measurementSystem: MeasurementSystem = MeasurementSystem.METRIC,
    val temperatureUnit: TemperatureUnit = TemperatureUnit.CELSIUS,
    val themeMode: ThemeMode = ThemeMode.SYSTEM,
    val useDynamicColors: Boolean = true,
    val hapticFeedbackEnabled: Boolean = true,
    val reducedMotion: Boolean = false,
    val largeText: Boolean = false,

    // Onboarding
    val hasCompletedOnboarding: Boolean = false,
    val onboardingCompletedDate: Long? = null,
    val termsAcceptedDate: Long? = null,
    val privacyAcceptedDate: Long? = null,

    // App Usage
    val firstLaunchDate: Long = System.currentTimeMillis(),
    val lastOpenDate: Long = System.currentTimeMillis(),
    val totalCheckIns: Int = 0,
    val streakDays: Int = 0,
    val longestStreak: Int = 0,

    val createdAt: Long = System.currentTimeMillis(),
    val lastModified: Long = System.currentTimeMillis()
)

enum class Gender {
    MALE,
    FEMALE,
    NON_BINARY,
    PREFER_NOT_TO_SAY
}

enum class DiagnosisType {
    ANKYLOSING_SPONDYLITIS,
    NON_RADIOGRAPHIC_AXIAL_SPA,
    PSORIATIC_ARTHRITIS,
    REACTIVE_ARTHRITIS,
    ENTEROPATHIC_ARTHRITIS,
    UNDIFFERENTIATED_SPA
}

enum class MeasurementSystem {
    METRIC,
    IMPERIAL
}

enum class TemperatureUnit {
    CELSIUS,
    FAHRENHEIT
}

enum class ThemeMode {
    LIGHT,
    DARK,
    SYSTEM
}
