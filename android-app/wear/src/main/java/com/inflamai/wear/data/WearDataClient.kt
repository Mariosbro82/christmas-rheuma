package com.inflamai.wear.data

import android.content.Context
import com.google.android.gms.wearable.*
import kotlinx.coroutines.tasks.await
import java.time.Instant

/**
 * Wear OS Data Client
 *
 * Sends logged data from Wear OS to phone app via Wearable Data Layer.
 */
class WearDataClient(
    private val context: Context
) {
    private val dataClient: DataClient by lazy { Wearable.getDataClient(context) }
    private val messageClient: MessageClient by lazy { Wearable.getMessageClient(context) }
    private val nodeClient: NodeClient by lazy { Wearable.getNodeClient(context) }

    companion object {
        const val PATH_SYMPTOM_LOG = "/symptom/log"
        const val PATH_FLARE_LOG = "/flare/log"
        const val PATH_PAIN_LOG = "/pain/log"
        const val PATH_QUICK_CHECKIN = "/checkin/quick"
    }

    /**
     * Check if phone is connected
     */
    suspend fun isPhoneConnected(): Boolean {
        return try {
            val nodes = nodeClient.connectedNodes.await()
            nodes.isNotEmpty()
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Send quick check-in data to phone
     */
    suspend fun sendQuickCheckIn(
        painLevel: Int,
        fatigueLevel: Int,
        stiffnessMinutes: Int
    ): Boolean {
        return try {
            val putDataReq = PutDataMapRequest.create(PATH_QUICK_CHECKIN).apply {
                dataMap.putInt("painLevel", painLevel)
                dataMap.putInt("fatigueLevel", fatigueLevel)
                dataMap.putInt("stiffnessMinutes", stiffnessMinutes)
                dataMap.putLong("timestamp", Instant.now().toEpochMilli())
                dataMap.putLong("updateTime", System.currentTimeMillis())
            }.asPutDataRequest().setUrgent()

            dataClient.putDataItem(putDataReq).await()
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Send flare log to phone
     */
    suspend fun sendFlareLog(
        severity: Int,
        notes: String? = null
    ): Boolean {
        return try {
            val putDataReq = PutDataMapRequest.create(PATH_FLARE_LOG).apply {
                dataMap.putInt("severity", severity)
                notes?.let { dataMap.putString("notes", it) }
                dataMap.putLong("timestamp", Instant.now().toEpochMilli())
                dataMap.putLong("updateTime", System.currentTimeMillis())
            }.asPutDataRequest().setUrgent()

            dataClient.putDataItem(putDataReq).await()
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Send pain log for specific body region to phone
     */
    suspend fun sendPainLog(
        regionId: String,
        painLevel: Int
    ): Boolean {
        return try {
            val putDataReq = PutDataMapRequest.create(PATH_PAIN_LOG).apply {
                dataMap.putString("regionId", regionId)
                dataMap.putInt("painLevel", painLevel)
                dataMap.putLong("timestamp", Instant.now().toEpochMilli())
                dataMap.putLong("updateTime", System.currentTimeMillis())
            }.asPutDataRequest().setUrgent()

            dataClient.putDataItem(putDataReq).await()
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Send full symptom log to phone
     */
    suspend fun sendSymptomLog(
        basdaiScore: Double,
        fatigueLevel: Int,
        painLevel: Int,
        stiffnessMinutes: Int,
        notes: String? = null
    ): Boolean {
        return try {
            val putDataReq = PutDataMapRequest.create(PATH_SYMPTOM_LOG).apply {
                dataMap.putDouble("basdaiScore", basdaiScore)
                dataMap.putInt("fatigueLevel", fatigueLevel)
                dataMap.putInt("painLevel", painLevel)
                dataMap.putInt("stiffnessMinutes", stiffnessMinutes)
                notes?.let { dataMap.putString("notes", it) }
                dataMap.putLong("timestamp", Instant.now().toEpochMilli())
                dataMap.putLong("updateTime", System.currentTimeMillis())
            }.asPutDataRequest().setUrgent()

            dataClient.putDataItem(putDataReq).await()
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Request full sync from phone
     */
    suspend fun requestPhoneSync(): Boolean {
        return try {
            val nodes = nodeClient.connectedNodes.await()
            if (nodes.isEmpty()) return false

            nodes.forEach { node ->
                messageClient.sendMessage(
                    node.id,
                    "/sync/request",
                    ByteArray(0)
                ).await()
            }
            true
        } catch (e: Exception) {
            false
        }
    }
}
