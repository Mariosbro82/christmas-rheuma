package com.inflamai.wear.data

import android.content.Intent
import com.google.android.gms.wearable.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

/**
 * Wearable Data Layer Listener Service
 *
 * Handles data sync between phone and Wear OS app.
 * Receives BASDAI scores, medication reminders, and syncs logged symptoms.
 */
class WearDataListenerService : WearableListenerService() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    companion object {
        // Data paths
        const val PATH_BASDAI_SCORE = "/basdai/current"
        const val PATH_MEDICATION_REMINDER = "/medication/reminder"
        const val PATH_SYMPTOM_LOG = "/symptom/log"
        const val PATH_FLARE_LOG = "/flare/log"
        const val PATH_PAIN_LOG = "/pain/log"
        const val PATH_SYNC_REQUEST = "/sync/request"

        // Message paths
        const val MESSAGE_SYNC_COMPLETE = "/sync/complete"
        const val MESSAGE_REQUEST_UPDATE = "/update/request"
    }

    override fun onDataChanged(dataEvents: DataEventBuffer) {
        super.onDataChanged(dataEvents)

        dataEvents.forEach { event ->
            when (event.type) {
                DataEvent.TYPE_CHANGED -> handleDataChanged(event.dataItem)
                DataEvent.TYPE_DELETED -> handleDataDeleted(event.dataItem)
            }
        }
    }

    override fun onMessageReceived(messageEvent: MessageEvent) {
        super.onMessageReceived(messageEvent)

        when (messageEvent.path) {
            MESSAGE_REQUEST_UPDATE -> {
                // Phone is requesting latest data from watch
                scope.launch {
                    syncDataToPhone()
                }
            }
            MESSAGE_SYNC_COMPLETE -> {
                // Phone confirmed sync complete
                broadcastSyncStatus(true)
            }
        }
    }

    override fun onCapabilityChanged(capabilityInfo: CapabilityInfo) {
        super.onCapabilityChanged(capabilityInfo)

        // Handle phone connection status changes
        val connectedNodes = capabilityInfo.nodes
        if (connectedNodes.isNotEmpty()) {
            broadcastConnectionStatus(true)
        } else {
            broadcastConnectionStatus(false)
        }
    }

    private fun handleDataChanged(dataItem: DataItem) {
        val dataMap = DataMapItem.fromDataItem(dataItem).dataMap

        when (dataItem.uri.path) {
            PATH_BASDAI_SCORE -> {
                // Received updated BASDAI score from phone
                val basdaiScore = dataMap.getDouble("score")
                val timestamp = dataMap.getLong("timestamp")
                val interpretation = dataMap.getString("interpretation")

                broadcastBASDAIUpdate(basdaiScore, timestamp, interpretation ?: "")
            }

            PATH_MEDICATION_REMINDER -> {
                // Received medication reminder from phone
                val medicationName = dataMap.getString("name")
                val dosage = dataMap.getString("dosage")
                val scheduledTime = dataMap.getLong("scheduledTime")

                showMedicationReminder(medicationName ?: "", dosage ?: "", scheduledTime)
            }
        }
    }

    private fun handleDataDeleted(dataItem: DataItem) {
        // Handle data deletion if needed
    }

    private suspend fun syncDataToPhone() {
        try {
            val dataClient = Wearable.getDataClient(this)

            // Sync any pending symptom logs
            // In a full implementation, this would read from local watch storage
            // and sync to phone

            broadcastSyncStatus(true)
        } catch (e: Exception) {
            broadcastSyncStatus(false)
        }
    }

    private fun broadcastBASDAIUpdate(score: Double, timestamp: Long, interpretation: String) {
        val intent = Intent("com.inflamai.BASDAI_UPDATE").apply {
            putExtra("score", score)
            putExtra("timestamp", timestamp)
            putExtra("interpretation", interpretation)
        }
        sendBroadcast(intent)
    }

    private fun broadcastSyncStatus(success: Boolean) {
        val intent = Intent("com.inflamai.SYNC_STATUS").apply {
            putExtra("success", success)
        }
        sendBroadcast(intent)
    }

    private fun broadcastConnectionStatus(connected: Boolean) {
        val intent = Intent("com.inflamai.CONNECTION_STATUS").apply {
            putExtra("connected", connected)
        }
        sendBroadcast(intent)
    }

    private fun showMedicationReminder(name: String, dosage: String, scheduledTime: Long) {
        // Show notification or update UI
        val intent = Intent("com.inflamai.MEDICATION_REMINDER").apply {
            putExtra("name", name)
            putExtra("dosage", dosage)
            putExtra("scheduledTime", scheduledTime)
        }
        sendBroadcast(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}
