package com.inflamai.wear

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.inflamai.wear.presentation.WearApp
import dagger.hilt.android.AndroidEntryPoint

/**
 * Main Activity for InflamAI Wear OS app
 */
@AndroidEntryPoint
class WearMainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            WearApp()
        }
    }
}
