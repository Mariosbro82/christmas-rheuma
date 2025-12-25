//
//  MotherModeViewModel.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation

@MainActor
final class MotherModeViewModel: ObservableObject {
    @Published var settings: MotherModeSettings
    
    private weak var environment: TraeAppEnvironment?
    
    init(environment: TraeAppEnvironment) {
        self.environment = environment
        self.settings = environment.motherModeSettings
    }
    
    func toggleMotherMode(_ value: Bool) {
        settings.isEnabled = value
        persist()
    }
    
    func updatePregnancyStatus(_ value: Bool) {
        settings.isPregnantOrPostpartum = value
        persist()
    }
    
    func toggleOneHandMode(_ value: Bool) {
        settings.oneHandMode = value
        persist()
    }
    
    func toggleVoiceNotes(_ value: Bool) {
        settings.voiceNotesAllowed = value
        persist()
    }
    
    func addNapWindow() {
        settings.napWindows.append(MotherQuietWindow(
            start: DateComponents(hour: 13, minute: 0),
            end: DateComponents(hour: 15, minute: 0)))
        persist()
    }
    
    func removeNapWindow(at offsets: IndexSet) {
        settings.napWindows.remove(atOffsets: offsets)
        persist()
    }
    
    func addFeedingWindow() {
        settings.feedingWindows.append(MotherQuietWindow(
            start: DateComponents(hour: 5, minute: 0),
            end: DateComponents(hour: 6, minute: 0)))
        persist()
    }
    
    func removeFeedingWindow(at offsets: IndexSet) {
        settings.feedingWindows.remove(atOffsets: offsets)
        persist()
    }
    
    private func persist() {
        environment?.motherModeSettings = settings
    }
}
