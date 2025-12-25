//
//  QuickCheckInViewModel.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation

@MainActor
final class QuickCheckInViewModel: ObservableObject {
    @Published var mood: Double = 5
    @Published private(set) var autoTags: [String] = []
    @Published private(set) var recordingURL: URL?
    
    private weak var environment: TraeAppEnvironment?
    private let voiceManager: VoiceNoteManager
    
    init(environment: TraeAppEnvironment, voiceNoteManager: VoiceNoteManager = VoiceNoteManager()) {
        self.environment = environment
        self.voiceManager = voiceNoteManager
        refreshTags()
    }
    
    func updateMood(_ value: Double) {
        mood = value
        refreshTags()
    }
    
    func startRecording() async -> Bool {
        guard await voiceManager.requestPermissionIfNeeded() else { return false }
        do {
            recordingURL = try voiceManager.startRecording()
            return true
        } catch {
            return false
        }
    }
    
    func stopRecording() {
        voiceManager.stopRecording()
    }
    
    func save() {
        let entry = MotherQuickEntry(
            id: UUID(),
            timestamp: Date(),
            mood: mood,
            tags: autoTags,
            voiceNoteURL: recordingURL
        )
        environment?.logQuickEntry(entry)
    }
    
    private func refreshTags() {
        var tags: [String] = []
        let hour = Calendar.current.component(.hour, from: Date())
        if hour < 6 {
            tags.append("mother.quick.auto_tag.night")
        }
        if mood <= 3 {
            tags.append("mother.quick.auto_tag.energy_low")
        }
        if mood >= 8 {
            tags.append("mother.quick.auto_tag.feeling_strong")
        }
        autoTags = tags
    }
}
