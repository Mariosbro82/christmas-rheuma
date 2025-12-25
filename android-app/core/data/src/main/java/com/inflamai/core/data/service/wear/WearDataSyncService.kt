package com.inflamai.core.data.service.wear

import android.content.Context
import com.google.android.gms.wearable.*
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await
import java.time.Instant
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Wear Data Sync Service
 *
 * Manages bidirectional data sync between phone and Wear OS app.
 * Syncs BASDAI scores, symptom logs, medication reminders, and flare events.
 */
@Singleton
class WearDataSyncService @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val dataClient: DataClient by lazy { Wearable.getDataClient(context) }
    private val messageClient: MessageClient by lazy { Wearable.getMessageClient(context) }
    private val capabilityClient: CapabilityClient by lazy { Wearable.getCapabilityClient(context) }
    private val nodeClient: NodeClient by lazy { Wearable.getNodeClient(context) }

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

        // Capability
        const val WEAR_CAPABILITY = "inflamai_wear"
    }

    /**
     * Check if Wear OS is available and connected
     */
    suspend fun isWearConnected(): Boolean {
        return try {
            val nodes = nodeClient.connectedNodes.await()
            nodes.isNotEmpty()
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Get connected Wear OS nodes
     */
    suspend fun getConnectedNodes(): List<Node> {
        return try {
            nodeClient.connectedNodes.await().toList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Observe Wear OS connection status
     */
    fun observeConnectionStatus(): Flow<Boolean> = callbackFlow {
        val listener = CapabilityClient.OnCapabilityChangedListener { info ->
            trySend(info.nodes.isNotEmpty())
        }

        capabilityClient.addListener(listener, WEAR_CAPABILITY)

        // Initial check
        trySend(isWearConnected())

        awaitClose {
            capabilityClient.removeListener(listener)
        }
    }

    /**
     * Sync BASDAI score to Wear OS
     */
    suspend fun syncBASDAIScore(
        score: Double,
        timestamp: Instant,
        interpretation: String
    ): Boolean {
        return try {
            val putDataReq = PutDataMapRequest.create(PATH_BASDAI_SCORE).apply {
                dataMap.putDouble("score", score)
                dataMap.putLong("timestamp", timestamp.toEpochMilli())
                dataMap.putString("interpretation", interpretation)
                dataMap.putLong("updateTime", System.currentTimeMillis())
            }.asPutDataRequest().setUrgent()

            dataClient.putDataItem(putDataReq).await()
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Send medication reminder to Wear OS
     */
    suspend fun sendMedicationReminder(
        medicationName: String,
        dosage: String,
        scheduledTime: Instant
    ): Boolean {
        return try {
            val putDataReq = PutDataMapRequest.create(PATH_MEDICATION_REMINDER).apply {
                dataMap.putString("name", medicationName)
                dataMap.putString("dosage", dosage)
                dataMap.putLong("scheduledTime", scheduledTime.toEpochMilli())
                dataMap.putLong("updateTime", System.currentTimeMillis())
            }.asPutDataRequest().setUrgent()

            dataClient.putDataItem(putDataReq).await()
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Request data sync from Wear OS
     */
    suspend fun requestSyncFromWear(): Boolean {
        return try {
            val nodes = getConnectedNodes()
            if (nodes.isEmpty()) return false

            nodes.forEach { node ->
                messageClient.sendMessage(
                    node.id,
                    MESSAGE_REQUEST_UPDATE,
                    ByteArray(0)
                ).await()
            }
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Observe incoming symptom logs from Wear OS
     */
    fun observeSymptomLogs(): Flow<WearSymptomLog> = callbackFlow {
        val listener = DataClient.OnDataChangedListener { dataEvents ->
            dataEvents.forEach { event ->
                if (event.type == DataEvent.TYPE_CHANGED) {
                    val dataMap = DataMapItem.fromDataItem(event.dataItem).dataMap
                    when (event.dataItem.uri.path) {
                        PATH_SYMPTOM_LOG -> {
                            val log = WearSymptomLog(
                                timestamp = Instant.ofEpochMilli(dataMap.getLong("timestamp")),
                                basdaiScore = dataMap.getDouble("basdaiScore"),
                                fatigueLevel = dataMap.getInt("fatigueLevel"),
                                painLevel = dataMap.getInt("painLevel"),
                                stiffnessMinutes = dataMap.getInt("stiffnessMinutes")
                            )
                            trySend(log)
                        }
                    }
                }
            }
        }

        dataClient.addListener(listener)

        awaitClose {
            dataClient.removeListener(listener)
        }
    }

    /**
     * Observe incoming flare logs from Wear OS
     */
    fun observeFlareLogs(): Flow<WearFlareLog> = callbackFlow {
        val listener = DataClient.OnDataChangedListener { dataEvents ->
            dataEvents.forEach { event ->
                if (event.type == DataEvent.TYPE_CHANGED) {
                    val dataMap = DataMapItem.fromDataItem(event.dataItem).dataMap
                    when (event.dataItem.uri.path) {
                        PATH_FLARE_LOG -> {
                            val log = WearFlareLog(
                                timestamp = Instant.ofEpochMilli(dataMap.getLong("timestamp")),
                                severity = dataMap.getInt("severity"),
                                notes = dataMap.getString("notes")
                            )
                            trySend(log)
                        }
                    }
                }
            }
        }

        dataClient.addListener(listener)

        awaitClose {
            dataClient.removeListener(listener)
        }
    }

    /**
     * Observe incoming pain logs from Wear OS
     */
    fun observePainLogs(): Flow<WearPainLog> = callbackFlow {
        val listener = DataClient.OnDataChangedListener { dataEvents ->
            dataEvents.forEach { event ->
                if (event.type == DataEvent.TYPE_CHANGED) {
                    val dataMap = DataMapItem.fromDataItem(event.dataItem).dataMap
                    when (event.dataItem.uri.path) {
                        PATH_PAIN_LOG -> {
                            val log = WearPainLog(
                                timestamp = Instant.ofEpochMilli(dataMap.getLong("timestamp")),
                                regionId = dataMap.getString("regionId") ?: "",
                                painLevel = dataMap.getInt("painLevel")
                            )
                            trySend(log)
                        }
                    }
                }
            }
        }

        dataClient.addListener(listener)

        awaitClose {
            dataClient.removeListener(listener)
        }
    }

    /**
     * Confirm sync completion to Wear OS
     */
    suspend fun confirmSyncComplete(): Boolean {
        return try {
            val nodes = getConnectedNodes()
            if (nodes.isEmpty()) return false

            nodes.forEach { node ->
                messageClient.sendMessage(
                    node.id,
                    MESSAGE_SYNC_COMPLETE,
                    ByteArray(0)
                ).await()
            }
            true
        } catch (e: Exception) {
            false
        }
    }
}

// Data classes for Wear OS logs
data class WearSymptomLog(
    val timestamp: Instant,
    val basdaiScore: Double,
    val fatigueLevel: Int,
    val painLevel: Int,
    val stiffnessMinutes: Int
)

data class WearFlareLog(
    val timestamp: Instant,
    val severity: Int,
    val notes: String?
)

data class WearPainLog(
    val timestamp: Instant,
    val regionId: String,
    val painLevel: Int
)
