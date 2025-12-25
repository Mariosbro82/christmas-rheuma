pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "InflamAI"

// Main app module
include(":app")

// Core modules
include(":core:common")
include(":core:data")
include(":core:domain")
include(":core:ui")

// Feature modules
include(":feature:home")
include(":feature:bodymap")
include(":feature:checkin")
include(":feature:medication")
include(":feature:trends")
include(":feature:flares")
include(":feature:exercise")
include(":feature:ai")
include(":feature:settings")
include(":feature:onboarding")
include(":feature:meditation")
include(":feature:quickcapture")

// Wear OS module
include(":wear")
