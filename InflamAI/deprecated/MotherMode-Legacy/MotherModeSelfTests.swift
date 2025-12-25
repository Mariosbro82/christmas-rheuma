//
//  MotherModeSelfTests.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

#if DEBUG
import Foundation

@MainActor
enum MotherModeSelfTests {
    static func run() {
        let environment = TraeAppEnvironment()
        let motherVM = MotherModeViewModel(environment: environment)
        assert(motherVM.settings.napWindows.isEmpty)
        motherVM.addNapWindow()
        assert(!motherVM.settings.napWindows.isEmpty)
        motherVM.toggleMotherMode(true)
        assert(environment.motherModeSettings.isEnabled)
        
        let quickVM = QuickCheckInViewModel(environment: environment)
        quickVM.updateMood(2)
        assert(quickVM.autoTags.contains("mother.quick.auto_tag.energy_low"))
        quickVM.save()
        assert(!environment.motherQuickEntries.isEmpty)
    }
}
#endif
