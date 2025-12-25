//
//  MotherModeModels.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation

struct MotherQuickEntry: Identifiable, Equatable {
    let id: UUID
    let timestamp: Date
    let mood: Double
    let tags: [String]
    let voiceNoteURL: URL?
}

struct MotherQuietWindow: Identifiable, Equatable, Hashable {
    let id: UUID
    var start: DateComponents
    var end: DateComponents
    
    init(id: UUID = UUID(), start: DateComponents, end: DateComponents) {
        self.id = id
        self.start = start
        self.end = end
    }
    
    func contains(_ components: DateComponents) -> Bool {
        guard
            let startHour = start.hour, let startMinute = start.minute,
            let endHour = end.hour, let endMinute = end.minute,
            let hour = components.hour, let minute = components.minute
        else { return false }
        
        let startTotal = startHour * 60 + startMinute
        let endTotal = endHour * 60 + endMinute
        let target = hour * 60 + minute
        
        if startTotal <= endTotal {
            return (target >= startTotal && target <= endTotal)
        } else {
            // Overnight range (e.g., 22:00 - 06:00)
            return target >= startTotal || target <= endTotal
        }
    }
}

struct MotherModeSettings: Equatable {
    var isEnabled: Bool
    var isPregnantOrPostpartum: Bool
    var oneHandMode: Bool
    var voiceNotesAllowed: Bool
    var napWindows: [MotherQuietWindow]
    var feedingWindows: [MotherQuietWindow]
    
    static let `default` = MotherModeSettings(
        isEnabled: false,
        isPregnantOrPostpartum: false,
        oneHandMode: true,
        voiceNotesAllowed: false,
        napWindows: [],
        feedingWindows: []
    )
}
