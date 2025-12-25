package com.inflamai.app

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * Main Application class for InflamAI
 *
 * Features:
 * - Hilt dependency injection setup
 * - Privacy-first architecture (no third-party analytics)
 * - On-device data processing only
 */
@HiltAndroidApp
class InflamAIApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        // Initialize any application-wide components here
        // Note: NO third-party analytics or tracking SDKs
        // All processing remains on-device

        if (BuildConfig.DEBUG) {
            // Run validation tests for medical calculators in debug builds
            validateMedicalCalculators()
        }
    }

    private fun validateMedicalCalculators() {
        try {
            val basidaiValid = com.inflamai.core.domain.calculator.BASDAICalculator.runValidationTests()
            val asdasValid = com.inflamai.core.domain.calculator.ASDACalculator.runValidationTests()
            val correlationValid = com.inflamai.core.domain.calculator.CorrelationEngine().runValidationTests()

            if (!basidaiValid || !asdasValid || !correlationValid) {
                android.util.Log.e("InflamAI", "Medical calculator validation failed!")
            } else {
                android.util.Log.d("InflamAI", "All medical calculators validated successfully")
            }
        } catch (e: Exception) {
            android.util.Log.e("InflamAI", "Error validating calculators", e)
        }
    }
}
