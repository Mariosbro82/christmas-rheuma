//
//  VoiceNoteManager.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation
import AVFoundation

final class VoiceNoteManager: NSObject, AVAudioRecorderDelegate {
    private var recorder: AVAudioRecorder?
    private let audioSession = AVAudioSession.sharedInstance()
    
    func requestPermissionIfNeeded() async -> Bool {
        do {
            try audioSession.setCategory(.playAndRecord, mode: .spokenAudio, options: [.duckOthers, .allowBluetooth])
            try audioSession.setActive(true)
        } catch {
            return false
        }
        
        return await withCheckedContinuation { continuation in
            audioSession.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }
    
    func startRecording() throws -> URL {
        let filename = UUID().uuidString.appending(".m4a")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        
        recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder?.delegate = self
        recorder?.record()
        return url
    }
    
    func stopRecording() {
        recorder?.stop()
        recorder = nil
    }
}
