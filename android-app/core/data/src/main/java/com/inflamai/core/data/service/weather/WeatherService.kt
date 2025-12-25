package com.inflamai.core.data.service.weather

import android.content.Context
import android.location.Location
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Weather Service using Open-Meteo API
 *
 * Android equivalent of iOS WeatherKitService.
 * Uses Open-Meteo (free, no API key required) for weather data.
 *
 * Critical for AS:
 * - Barometric pressure (rapid drops are key AS triggers)
 * - 12-hour pressure change tracking
 * - Humidity and temperature
 *
 * Privacy:
 * - Location used only for weather context
 * - Never stored or shared
 */
@Singleton
class WeatherService @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val fusedLocationClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val httpClient = OkHttpClient()
    private val json = Json { ignoreUnknownKeys = true }

    companion object {
        private const val BASE_URL = "https://api.open-meteo.com/v1/forecast"

        // AS trigger threshold
        const val PRESSURE_DROP_THRESHOLD_MMHG = 5.0  // >5 mmHg drop in 12h is significant
    }

    /**
     * Get current weather for user's location
     */
    suspend fun getCurrentWeather(): WeatherData? {
        val location = getCurrentLocation() ?: return null
        return getWeatherForLocation(location.latitude, location.longitude)
    }

    /**
     * Get weather for specific coordinates
     */
    suspend fun getWeatherForLocation(latitude: Double, longitude: Double): WeatherData? {
        return withContext(Dispatchers.IO) {
            try {
                val url = buildUrl(latitude, longitude)
                val request = Request.Builder().url(url).build()

                httpClient.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) return@withContext null

                    val body = response.body?.string() ?: return@withContext null
                    parseWeatherResponse(body, latitude, longitude)
                }
            } catch (e: Exception) {
                null
            }
        }
    }

    /**
     * Calculate flare risk based on weather conditions
     */
    fun calculateFlareRisk(weather: WeatherData): FlareRiskAssessment {
        val pressureRisk = calculatePressureRisk(weather)
        val humidityRisk = calculateHumidityRisk(weather)
        val temperatureRisk = calculateTemperatureRisk(weather)

        val overallRisk = (pressureRisk * 0.6 + humidityRisk * 0.2 + temperatureRisk * 0.2)
            .coerceIn(0.0, 1.0)

        val riskLevel = when {
            overallRisk >= 0.7 -> FlareRiskLevel.HIGH
            overallRisk >= 0.4 -> FlareRiskLevel.MODERATE
            else -> FlareRiskLevel.LOW
        }

        val factors = mutableListOf<String>()
        if (pressureRisk > 0.5) factors.add("Significant pressure drop detected")
        if (humidityRisk > 0.5) factors.add("High humidity levels")
        if (temperatureRisk > 0.5) factors.add("Temperature change")

        return FlareRiskAssessment(
            level = riskLevel,
            score = overallRisk,
            factors = factors,
            pressureChange12h = weather.pressureChange12h,
            recommendation = getRecommendation(riskLevel)
        )
    }

    private fun calculatePressureRisk(weather: WeatherData): Double {
        val change = weather.pressureChange12h ?: return 0.0

        // Rapid pressure drops are problematic for AS
        return when {
            change <= -PRESSURE_DROP_THRESHOLD_MMHG -> 1.0
            change <= -3.0 -> 0.7
            change <= -1.0 -> 0.3
            else -> 0.0
        }
    }

    private fun calculateHumidityRisk(weather: WeatherData): Double {
        val humidity = weather.humidity ?: return 0.0

        return when {
            humidity >= 80 -> 0.8
            humidity >= 70 -> 0.5
            humidity >= 60 -> 0.3
            else -> 0.0
        }
    }

    private fun calculateTemperatureRisk(weather: WeatherData): Double {
        val temp = weather.temperature ?: return 0.0

        // Cold temperatures can worsen AS symptoms
        return when {
            temp <= 0 -> 0.7
            temp <= 10 -> 0.4
            else -> 0.0
        }
    }

    private fun getRecommendation(level: FlareRiskLevel): String {
        return when (level) {
            FlareRiskLevel.HIGH -> "Weather conditions may trigger symptoms. Consider preventive measures and gentle stretching."
            FlareRiskLevel.MODERATE -> "Be mindful of potential weather-related symptoms. Stay active but listen to your body."
            FlareRiskLevel.LOW -> "Weather conditions are favorable. Good time for regular activities."
        }
    }

    private suspend fun getCurrentLocation(): Location? {
        return suspendCancellableCoroutine { continuation ->
            try {
                val cancellationToken = CancellationTokenSource()

                fusedLocationClient.getCurrentLocation(
                    Priority.PRIORITY_BALANCED_POWER_ACCURACY,
                    cancellationToken.token
                ).addOnSuccessListener { location ->
                    continuation.resume(location)
                }.addOnFailureListener { e ->
                    continuation.resume(null)
                }

                continuation.invokeOnCancellation {
                    cancellationToken.cancel()
                }
            } catch (e: SecurityException) {
                continuation.resume(null)
            }
        }
    }

    private fun buildUrl(latitude: Double, longitude: Double): String {
        return "$BASE_URL?" +
            "latitude=$latitude&" +
            "longitude=$longitude&" +
            "current=temperature_2m,relative_humidity_2m,pressure_msl,weather_code,cloud_cover,wind_speed_10m&" +
            "hourly=pressure_msl&" +
            "timezone=auto&" +
            "past_hours=12&" +
            "forecast_hours=24"
    }

    private fun parseWeatherResponse(jsonString: String, lat: Double, lon: Double): WeatherData? {
        return try {
            val response = json.decodeFromString<OpenMeteoResponse>(jsonString)

            val current = response.current ?: return null
            val hourlyPressure = response.hourly?.pressure_msl ?: emptyList()

            // Calculate 12-hour pressure change
            val pressureChange12h = if (hourlyPressure.size >= 13) {
                val now = current.pressure_msl ?: hourlyPressure.lastOrNull() ?: 0.0
                val hours12Ago = hourlyPressure.firstOrNull() ?: now
                // Convert hPa to mmHg (1 hPa = 0.750062 mmHg)
                (now - hours12Ago) * 0.750062
            } else null

            WeatherData(
                timestamp = System.currentTimeMillis(),
                latitude = lat,
                longitude = lon,
                temperature = current.temperature_2m,
                humidity = current.relative_humidity_2m,
                barometricPressure = current.pressure_msl?.let { it * 0.750062 }, // Convert to mmHg
                pressureChange12h = pressureChange12h,
                cloudCover = current.cloud_cover,
                windSpeed = current.wind_speed_10m,
                weatherCode = current.weather_code,
                weatherCondition = mapWeatherCode(current.weather_code)
            )
        } catch (e: Exception) {
            null
        }
    }

    private fun mapWeatherCode(code: Int?): String {
        return when (code) {
            0 -> "Clear"
            1, 2, 3 -> "Partly Cloudy"
            45, 48 -> "Foggy"
            51, 53, 55 -> "Drizzle"
            61, 63, 65 -> "Rain"
            71, 73, 75 -> "Snow"
            77 -> "Snow Grains"
            80, 81, 82 -> "Rain Showers"
            85, 86 -> "Snow Showers"
            95 -> "Thunderstorm"
            96, 99 -> "Thunderstorm with Hail"
            else -> "Unknown"
        }
    }
}

// Response models for Open-Meteo API
@Serializable
data class OpenMeteoResponse(
    val current: CurrentWeather? = null,
    val hourly: HourlyData? = null
)

@Serializable
data class CurrentWeather(
    val temperature_2m: Double? = null,
    val relative_humidity_2m: Int? = null,
    val pressure_msl: Double? = null,
    val weather_code: Int? = null,
    val cloud_cover: Int? = null,
    val wind_speed_10m: Double? = null
)

@Serializable
data class HourlyData(
    val pressure_msl: List<Double>? = null
)

// Domain models

data class WeatherData(
    val timestamp: Long,
    val latitude: Double,
    val longitude: Double,
    val temperature: Double?,           // Celsius
    val humidity: Int?,                  // Percentage
    val barometricPressure: Double?,    // mmHg
    val pressureChange12h: Double?,     // mmHg change in last 12 hours
    val cloudCover: Int?,               // Percentage
    val windSpeed: Double?,             // km/h
    val weatherCode: Int?,
    val weatherCondition: String
)

data class FlareRiskAssessment(
    val level: FlareRiskLevel,
    val score: Double,                  // 0.0 - 1.0
    val factors: List<String>,
    val pressureChange12h: Double?,
    val recommendation: String
)

enum class FlareRiskLevel {
    LOW,
    MODERATE,
    HIGH
}
