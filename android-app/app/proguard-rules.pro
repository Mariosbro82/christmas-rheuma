# InflamAI ProGuard Rules
# Privacy-first app - no third-party analytics

# ===== Jetpack Compose =====
# Keep Compose runtime classes to prevent lock verification issues
-keep class androidx.compose.runtime.** { *; }
-keepclassmembers class androidx.compose.runtime.** { *; }

# Keep SnapshotStateList and related classes
-keep class androidx.compose.runtime.snapshots.** { *; }
-keepclassmembers class androidx.compose.runtime.snapshots.** {
    *;
}

# Keep Compose UI classes
-keep class androidx.compose.ui.** { *; }
-keep class androidx.compose.material3.** { *; }
-keep class androidx.compose.material.** { *; }
-keep class androidx.compose.foundation.** { *; }
-keep class androidx.compose.animation.** { *; }

# Prevent R8 from stripping Compose internals
-dontwarn androidx.compose.**

# Keep Room entities
-keep class com.inflamai.core.data.database.entity.** { *; }

# Keep Kotlin serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt

-keepclassmembers class kotlinx.serialization.json.** {
    *** Companion;
}
-keepclasseswithmembers class kotlinx.serialization.json.** {
    kotlinx.serialization.KSerializer serializer(...);
}

-keep,includedescriptorclasses class com.inflamai.**$$serializer { *; }
-keepclassmembers class com.inflamai.** {
    *** Companion;
}
-keepclasseswithmembers class com.inflamai.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# Keep calculators (medical accuracy critical)
-keep class com.inflamai.core.domain.calculator.** { *; }

# Keep enum values
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Health Connect
-keep class androidx.health.connect.client.** { *; }

# Retrofit
-keepattributes Signature, InnerClasses, EnclosingMethod
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations
-keepclassmembers,allowshrinking,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}
-dontwarn org.codehaus.mojo.animal_sniffer.IgnoreJRERequirement
-dontwarn javax.annotation.**
-dontwarn kotlin.Unit
-dontwarn retrofit2.KotlinExtensions
-dontwarn retrofit2.KotlinExtensions$*

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn javax.annotation.**
-keepnames class okhttp3.internal.publicsuffix.PublicSuffixDatabase

# Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}

# Hilt
-keep class dagger.hilt.** { *; }
-keep class javax.inject.** { *; }
-keep class * extends dagger.hilt.android.internal.managers.ComponentSupplier { *; }
-keep class * extends dagger.hilt.android.internal.managers.ViewComponentManager$FragmentContextWrapper { *; }

# Keep accessibility labels
-keepclassmembers class * {
    @androidx.compose.ui.semantics.* <methods>;
}
