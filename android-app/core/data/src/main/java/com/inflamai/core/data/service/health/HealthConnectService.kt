package com.inflamai.core.data.service.health

import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.activity.result.contract.ActivityResultContract
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.PermissionController
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.*
import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.time.Instant
import java.time.temporal.ChronoUnit
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.reflect.KClass

/**
 * Health Connect Service
 *
 * Android equivalent of iOS HealthKitService.
 * Provides read-only access to health data for pattern analysis.
 *
 * Data Types:
 * - Heart Rate & HRV (critical for AS inflammation markers)
 * - Sleep duration & efficiency
 * - Step count & distance
 * - Active calories
 * - Oxygen saturation
 * - Respiratory rate
 *
 * Privacy:
 * - Read-only access (no writes)
 * - All data stays on-device
 * - User-controlled permissions
 */
@Singleton
class HealthConnectService @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private var healthConnectClient: HealthConnectClient? = null

    companion object {
        // Required permissions for pattern analysis
        val REQUIRED_PERMISSIONS = setOf(
            HealthPermission.getReadPermission(HeartRateRecord::class),
            HealthPermission.getReadPermission(HeartRateVariabilityRmssdRecord::class),
            HealthPermission.getReadPermission(RestingHeartRateRecord::class),
            HealthPermission.getReadPermission(StepsRecord::class),
            HealthPermission.getReadPermission(DistanceRecord::class),
            HealthPermission.getReadPermission(SleepSessionRecord::class),
            HealthPermission.getReadPermission(ActiveCaloriesBurnedRecord::class),
            HealthPermission.getReadPermission(OxygenSaturationRecord::class),
            HealthPermission.getReadPermission(RespiratoryRateRecord::class),
            HealthPermission.getReadPermission(ExerciseSessionRecord::class)
        )
    }

    /**
     * Check if Health Connect is available on this device
     */
    fun isHealthConnectAvailable(): HealthConnectAvailability {
        return when (HealthConnectClient.getSdkStatus(context)) {
            HealthConnectClient.SDK_AVAILABLE -> {
                healthConnectClient = HealthConnectClient.getOrCreate(context)
                HealthConnectAvailability.AVAILABLE
            }
            HealthConnectClient.SDK_UNAVAILABLE -> HealthConnectAvailability.NOT_SUPPORTED
            HealthConnectClient.SDK_UNAVAILABLE_PROVIDER_UPDATE_REQUIRED -> HealthConnectAvailability.UPDATE_REQUIRED
            else -> HealthConnectAvailability.NOT_INSTALLED
        }
    }

    /**
     * Get the intent to open Health Connect app
     */
    fun getHealthConnectSettingsIntent(): Intent? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            Intent(HealthConnectClient.ACTION_HEALTH_CONNECT_SETTINGS)
        } else {
            context.packageManager.getLaunchIntentForPackage("com.google.android.apps.healthdata")
        }
    }

    /**
     * Check if all required permissions are granted
     */
    suspend fun hasAllPermissions(): Boolean {
        val client = healthConnectClient ?: return false
        val granted = client.permissionController.getGrantedPermissions()
        return REQUIRED_PERMISSIONS.all { it in granted }
    }

    /**
     * Create permission request contract
     */
    fun createPermissionRequestContract(): ActivityResultContract<Set<String>, Set<String>> {
        return PermissionController.createRequestPermissionResultContract()
    }

    /**
     * Read heart rate data for a time range
     */
    suspend fun readHeartRate(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): List<HeartRateData> {
        val client = healthConnectClient ?: return emptyList()

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = HeartRateRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.flatMap { record ->
                record.samples.map { sample ->
                    HeartRateData(
                        timestamp = sample.time.toEpochMilli(),
                        beatsPerMinute = sample.beatsPerMinute.toInt()
                    )
                }
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Read HRV data (critical for AS inflammation correlation)
     */
    suspend fun readHeartRateVariability(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): List<HrvData> {
        val client = healthConnectClient ?: return emptyList()

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = HeartRateVariabilityRmssdRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.map { record ->
                HrvData(
                    timestamp = record.time.toEpochMilli(),
                    rmssdMs = record.heartRateVariabilityMillis
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Read resting heart rate
     */
    suspend fun readRestingHeartRate(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): List<RestingHeartRateData> {
        val client = healthConnectClient ?: return emptyList()

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = RestingHeartRateRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.map { record ->
                RestingHeartRateData(
                    timestamp = record.time.toEpochMilli(),
                    beatsPerMinute = record.beatsPerMinute.toInt()
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Read step count
     */
    suspend fun readSteps(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): Long {
        val client = healthConnectClient ?: return 0

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = StepsRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.sumOf { it.count }
        } catch (e: Exception) {
            0
        }
    }

    /**
     * Read sleep data
     */
    suspend fun readSleep(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): List<SleepData> {
        val client = healthConnectClient ?: return emptyList()

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = SleepSessionRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.map { record ->
                val durationMinutes = ChronoUnit.MINUTES.between(record.startTime, record.endTime)

                SleepData(
                    startTime = record.startTime.toEpochMilli(),
                    endTime = record.endTime.toEpochMilli(),
                    durationMinutes = durationMinutes.toInt(),
                    stages = record.stages.map { stage ->
                        SleepStage(
                            startTime = stage.startTime.toEpochMilli(),
                            endTime = stage.endTime.toEpochMilli(),
                            stage = mapSleepStage(stage.stage)
                        )
                    }
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Read distance walked/run
     */
    suspend fun readDistance(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): Double {
        val client = healthConnectClient ?: return 0.0

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = DistanceRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.sumOf { it.distance.inMeters }
        } catch (e: Exception) {
            0.0
        }
    }

    /**
     * Read active calories burned
     */
    suspend fun readActiveCalories(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): Double {
        val client = healthConnectClient ?: return 0.0

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = ActiveCaloriesBurnedRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.sumOf { it.energy.inKilocalories }
        } catch (e: Exception) {
            0.0
        }
    }

    /**
     * Read oxygen saturation
     */
    suspend fun readOxygenSaturation(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): List<OxygenSaturationData> {
        val client = healthConnectClient ?: return emptyList()

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = OxygenSaturationRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.map { record ->
                OxygenSaturationData(
                    timestamp = record.time.toEpochMilli(),
                    percentage = record.percentage.value
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Read respiratory rate
     */
    suspend fun readRespiratoryRate(
        startTime: Instant,
        endTime: Instant = Instant.now()
    ): List<RespiratoryRateData> {
        val client = healthConnectClient ?: return emptyList()

        return try {
            val response = client.readRecords(
                ReadRecordsRequest(
                    recordType = RespiratoryRateRecord::class,
                    timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
                )
            )

            response.records.map { record ->
                RespiratoryRateData(
                    timestamp = record.time.toEpochMilli(),
                    breathsPerMinute = record.rate
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Get comprehensive health snapshot for a day
     */
    suspend fun getDailyHealthSnapshot(date: Instant): DailyHealthSnapshot {
        val startOfDay = date.truncatedTo(ChronoUnit.DAYS)
        val endOfDay = startOfDay.plus(1, ChronoUnit.DAYS)

        val heartRates = readHeartRate(startOfDay, endOfDay)
        val hrvData = readHeartRateVariability(startOfDay, endOfDay)
        val restingHR = readRestingHeartRate(startOfDay, endOfDay)
        val steps = readSteps(startOfDay, endOfDay)
        val distance = readDistance(startOfDay, endOfDay)
        val calories = readActiveCalories(startOfDay, endOfDay)
        val sleepData = readSleep(startOfDay.minus(12, ChronoUnit.HOURS), endOfDay)

        return DailyHealthSnapshot(
            date = startOfDay.toEpochMilli(),
            averageHeartRate = heartRates.map { it.beatsPerMinute }.average().takeIf { !it.isNaN() }?.toInt(),
            latestHrv = hrvData.maxByOrNull { it.timestamp }?.rmssdMs,
            restingHeartRate = restingHR.minByOrNull { it.beatsPerMinute }?.beatsPerMinute,
            stepCount = steps.toInt(),
            distanceMeters = distance,
            activeCalories = calories,
            sleepDurationMinutes = sleepData.sumOf { it.durationMinutes },
            sleepEfficiency = calculateSleepEfficiency(sleepData)
        )
    }

    private fun calculateSleepEfficiency(sleepData: List<SleepData>): Double? {
        if (sleepData.isEmpty()) return null

        val totalSleep = sleepData.sumOf { it.durationMinutes }
        val deepSleep = sleepData.flatMap { it.stages }
            .filter { it.stage == SleepStageType.DEEP }
            .sumOf { ChronoUnit.MINUTES.between(
                Instant.ofEpochMilli(it.startTime),
                Instant.ofEpochMilli(it.endTime)
            ) }

        return if (totalSleep > 0) (deepSleep.toDouble() / totalSleep).coerceIn(0.0, 1.0) else null
    }

    private fun mapSleepStage(stage: Int): SleepStageType {
        return when (stage) {
            SleepSessionRecord.STAGE_TYPE_AWAKE -> SleepStageType.AWAKE
            SleepSessionRecord.STAGE_TYPE_LIGHT -> SleepStageType.LIGHT
            SleepSessionRecord.STAGE_TYPE_DEEP -> SleepStageType.DEEP
            SleepSessionRecord.STAGE_TYPE_REM -> SleepStageType.REM
            else -> SleepStageType.UNKNOWN
        }
    }
}

// Data classes for health data

data class HeartRateData(
    val timestamp: Long,
    val beatsPerMinute: Int
)

data class HrvData(
    val timestamp: Long,
    val rmssdMs: Double
)

data class RestingHeartRateData(
    val timestamp: Long,
    val beatsPerMinute: Int
)

data class SleepData(
    val startTime: Long,
    val endTime: Long,
    val durationMinutes: Int,
    val stages: List<SleepStage>
)

data class SleepStage(
    val startTime: Long,
    val endTime: Long,
    val stage: SleepStageType
)

enum class SleepStageType {
    AWAKE,
    LIGHT,
    DEEP,
    REM,
    UNKNOWN
}

data class OxygenSaturationData(
    val timestamp: Long,
    val percentage: Double
)

data class RespiratoryRateData(
    val timestamp: Long,
    val breathsPerMinute: Double
)

data class DailyHealthSnapshot(
    val date: Long,
    val averageHeartRate: Int?,
    val latestHrv: Double?,
    val restingHeartRate: Int?,
    val stepCount: Int,
    val distanceMeters: Double,
    val activeCalories: Double,
    val sleepDurationMinutes: Int,
    val sleepEfficiency: Double?
)

enum class HealthConnectAvailability {
    AVAILABLE,
    NOT_INSTALLED,
    NOT_SUPPORTED,
    UPDATE_REQUIRED
}
