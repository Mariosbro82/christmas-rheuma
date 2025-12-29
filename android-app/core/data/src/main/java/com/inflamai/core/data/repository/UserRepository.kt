package com.inflamai.core.data.repository

import com.inflamai.core.data.database.dao.UserProfileDao
import com.inflamai.core.data.database.entity.ThemeMode
import com.inflamai.core.data.database.entity.UserProfileEntity
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for user profile data
 * Abstracts UserProfileDao
 */
@Singleton
class UserRepository @Inject constructor(
    private val userProfileDao: UserProfileDao
) {
    fun observeUserProfile(): Flow<UserProfileEntity?> = userProfileDao.observe()

    suspend fun getUserProfile(): UserProfileEntity? = userProfileDao.get()

    suspend fun createOrUpdateProfile(userProfile: UserProfileEntity): Long =
        userProfileDao.insert(userProfile)

    suspend fun updateProfile(userProfile: UserProfileEntity) =
        userProfileDao.update(userProfile)

    suspend fun deleteProfile() = userProfileDao.delete()

    suspend fun incrementCheckIn(newStreakDays: Int) =
        userProfileDao.incrementCheckIn(newStreakDays)

    // Create default profile if none exists
    suspend fun ensureProfileExists(): UserProfileEntity {
        var profile = userProfileDao.get()
        if (profile == null) {
            profile = UserProfileEntity(
                id = "user_profile",
                diagnosisDate = null,
                hlaB27Positive = null,
                currentBiologic = null,
                gender = null,
                dateOfBirth = null,
                notificationsEnabled = true,
                dailyCheckInReminderTime = "09:00",
                biometricLockEnabled = false,
                themeMode = ThemeMode.SYSTEM,
                hasCompletedOnboarding = false,
                streakDays = 0,
                longestStreak = 0,
                totalCheckIns = 0
            )
            userProfileDao.insert(profile)
        }
        return profile
    }

    // Update specific settings
    suspend fun updateNotificationsEnabled(enabled: Boolean) {
        val profile = getUserProfile() ?: return
        updateProfile(profile.copy(notificationsEnabled = enabled))
    }

    suspend fun updateBiometricEnabled(enabled: Boolean) {
        val profile = getUserProfile() ?: return
        updateProfile(profile.copy(biometricLockEnabled = enabled))
    }

    suspend fun updateDarkMode(enabled: Boolean) {
        val profile = getUserProfile() ?: return
        updateProfile(profile.copy(themeMode = if (enabled) ThemeMode.DARK else ThemeMode.LIGHT))
    }

    suspend fun setOnboardingComplete() {
        val profile = getUserProfile() ?: return
        updateProfile(profile.copy(hasCompletedOnboarding = true))
    }

    suspend fun updateDailyCheckInReminder(time: String) {
        val profile = getUserProfile() ?: return
        updateProfile(profile.copy(dailyCheckInReminderTime = time))
    }
}
