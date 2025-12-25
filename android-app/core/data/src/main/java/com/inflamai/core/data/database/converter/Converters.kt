package com.inflamai.core.data.database.converter

import androidx.room.TypeConverter
import com.inflamai.core.data.database.entity.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.util.Date

/**
 * Room TypeConverters for complex types
 */
class Converters {

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    // Date converters
    @TypeConverter
    fun fromTimestamp(value: Long?): Date? {
        return value?.let { Date(it) }
    }

    @TypeConverter
    fun dateToTimestamp(date: Date?): Long? {
        return date?.time
    }

    // List<String> converters
    @TypeConverter
    fun fromStringList(value: List<String>): String {
        return json.encodeToString(value)
    }

    @TypeConverter
    fun toStringList(value: String): List<String> {
        return try {
            json.decodeFromString(value)
        } catch (e: Exception) {
            emptyList()
        }
    }

    // Map<String, Int> converters (for joint pain maps)
    @TypeConverter
    fun fromIntMap(value: Map<String, Int>): String {
        return json.encodeToString(value)
    }

    @TypeConverter
    fun toIntMap(value: String): Map<String, Int> {
        return try {
            json.decodeFromString(value)
        } catch (e: Exception) {
            emptyMap()
        }
    }

    // Enum converters
    @TypeConverter
    fun fromMedicationCategory(value: MedicationCategory): String = value.name

    @TypeConverter
    fun toMedicationCategory(value: String): MedicationCategory =
        MedicationCategory.valueOf(value)

    @TypeConverter
    fun fromMedicationFrequency(value: MedicationFrequency): String = value.name

    @TypeConverter
    fun toMedicationFrequency(value: String): MedicationFrequency =
        MedicationFrequency.valueOf(value)

    @TypeConverter
    fun fromMedicationRoute(value: MedicationRoute): String = value.name

    @TypeConverter
    fun toMedicationRoute(value: String): MedicationRoute =
        MedicationRoute.valueOf(value)

    @TypeConverter
    fun fromSkipReason(value: SkipReason?): String? = value?.name

    @TypeConverter
    fun toSkipReason(value: String?): SkipReason? =
        value?.let { SkipReason.valueOf(it) }

    @TypeConverter
    fun fromFlareType(value: FlareType): String = value.name

    @TypeConverter
    fun toFlareType(value: String): FlareType =
        FlareType.valueOf(value)

    @TypeConverter
    fun fromExerciseRoutineType(value: ExerciseRoutineType): String = value.name

    @TypeConverter
    fun toExerciseRoutineType(value: String): ExerciseRoutineType =
        ExerciseRoutineType.valueOf(value)

    @TypeConverter
    fun fromExerciseCategory(value: ExerciseCategory): String = value.name

    @TypeConverter
    fun toExerciseCategory(value: String): ExerciseCategory =
        ExerciseCategory.valueOf(value)

    @TypeConverter
    fun fromMeditationType(value: MeditationType): String = value.name

    @TypeConverter
    fun toMeditationType(value: String): MeditationType =
        MeditationType.valueOf(value)

    @TypeConverter
    fun fromGender(value: Gender?): String? = value?.name

    @TypeConverter
    fun toGender(value: String?): Gender? =
        value?.let { Gender.valueOf(it) }

    @TypeConverter
    fun fromDiagnosisType(value: DiagnosisType): String = value.name

    @TypeConverter
    fun toDiagnosisType(value: String): DiagnosisType =
        DiagnosisType.valueOf(value)

    @TypeConverter
    fun fromMeasurementSystem(value: MeasurementSystem): String = value.name

    @TypeConverter
    fun toMeasurementSystem(value: String): MeasurementSystem =
        MeasurementSystem.valueOf(value)

    @TypeConverter
    fun fromTemperatureUnit(value: TemperatureUnit): String = value.name

    @TypeConverter
    fun toTemperatureUnit(value: String): TemperatureUnit =
        TemperatureUnit.valueOf(value)

    @TypeConverter
    fun fromThemeMode(value: ThemeMode): String = value.name

    @TypeConverter
    fun toThemeMode(value: String): ThemeMode =
        ThemeMode.valueOf(value)

    @TypeConverter
    fun fromLagCategory(value: LagCategory): String = value.name

    @TypeConverter
    fun toLagCategory(value: String): LagCategory =
        LagCategory.valueOf(value)
}
