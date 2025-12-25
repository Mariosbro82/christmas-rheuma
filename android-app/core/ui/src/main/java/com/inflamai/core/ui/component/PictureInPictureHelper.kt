package com.inflamai.core.ui.component

import android.app.Activity
import android.app.PictureInPictureParams
import android.content.Context
import android.content.ContextWrapper
import android.content.pm.PackageManager
import android.os.Build
import android.util.Rational
import androidx.annotation.RequiresApi
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.platform.LocalContext
import androidx.core.app.PictureInPictureModeChangedInfo
import androidx.core.util.Consumer

/**
 * Helper class for managing Picture-in-Picture (PiP) mode.
 * Supports Android 8.0+ (API 26+)
 */
object PictureInPictureHelper {

    /**
     * Check if device supports Picture-in-Picture mode
     */
    fun isPictureInPictureSupported(context: Context): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            context.packageManager.hasSystemFeature(PackageManager.FEATURE_PICTURE_IN_PICTURE)
        } else {
            false
        }
    }

    /**
     * Enter Picture-in-Picture mode with video aspect ratio
     */
    @RequiresApi(Build.VERSION_CODES.O)
    fun enterPictureInPictureMode(
        activity: Activity,
        aspectRatio: Rational = Rational(16, 9)
    ): Boolean {
        return if (isPictureInPictureSupported(activity)) {
            val params = PictureInPictureParams.Builder()
                .setAspectRatio(aspectRatio)
                .build()
            activity.enterPictureInPictureMode(params)
        } else {
            false
        }
    }

    /**
     * Update PiP parameters while in PiP mode
     */
    @RequiresApi(Build.VERSION_CODES.O)
    fun updatePictureInPictureParams(
        activity: Activity,
        aspectRatio: Rational = Rational(16, 9)
    ) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val params = PictureInPictureParams.Builder()
                .setAspectRatio(aspectRatio)
                .build()
            activity.setPictureInPictureParams(params)
        }
    }
}

/**
 * Extension function to find Activity from Context
 */
fun Context.findActivity(): Activity? {
    var context = this
    while (context is ContextWrapper) {
        if (context is Activity) return context
        context = context.baseContext
    }
    return null
}

/**
 * Composable effect to handle PiP mode changes
 */
@Composable
fun PictureInPictureModeEffect(
    onPictureInPictureModeChanged: (isInPictureInPictureMode: Boolean) -> Unit
) {
    val context = LocalContext.current
    val activity = context.findActivity()

    DisposableEffect(activity) {
        if (activity != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val listener = Consumer<PictureInPictureModeChangedInfo> { info ->
                onPictureInPictureModeChanged(info.isInPictureInPictureMode)
            }
            activity.addOnPictureInPictureModeChangedListener(listener)

            onDispose {
                activity.removeOnPictureInPictureModeChangedListener(listener)
            }
        } else {
            onDispose { }
        }
    }
}
