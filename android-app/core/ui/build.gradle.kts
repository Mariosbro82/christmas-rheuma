plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.hilt)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.inflamai.core.ui"
    compileSdk = 34

    defaultConfig {
        minSdk = 28
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = libs.versions.compose.compiler.get()
    }
}

dependencies {
    implementation(project(":core:common"))

    implementation(libs.core.ktx)

    // Compose
    implementation(platform(libs.compose.bom))
    api(libs.bundles.compose)
    api(libs.activity.compose)
    debugImplementation(libs.bundles.compose.debug)

    // Lifecycle
    api(libs.bundles.lifecycle)

    // Charts
    api(libs.vico.compose)
    api(libs.vico.core)

    // Image Loading
    api(libs.coil.compose)

    // Media3 (ExoPlayer replacement)
    api(libs.media3.exoplayer)
    api(libs.media3.ui)
    api(libs.media3.common)

    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)

    testImplementation(libs.bundles.testing)
}
