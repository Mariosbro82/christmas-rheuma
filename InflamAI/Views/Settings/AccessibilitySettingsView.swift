//
//  AccessibilitySettingsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import AVFoundation

struct AccessibilitySettingsView: View {
    @StateObject private var accessibilityManager = AccessibilityManager.shared
    @State private var showingLanguageSelector = false
    @State private var showingVoiceSettings = false
    @State private var showingColorBlindnessInfo = false
    @State private var showingAccessibilityAudit = false
    @State private var auditReport: AccessibilityAuditReport?
    
    var body: some View {
        NavigationView {
            List {
                // Language and Localization Section
                Section("Language & Localization") {
                    HStack {
                        Image(systemName: "globe")
                            .foregroundColor(.blue)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Language")
                                .font(.body)
                            Text(accessibilityManager.currentLanguage.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Text(accessibilityManager.currentLanguage.flag)
                            .font(.title2)
                        
                        Image(systemName: "chevron.right")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        showingLanguageSelector = true
                        accessibilityManager.provideHapticFeedback(type: .selection)
                    }
                    .accessibilityLabel("Language: \(accessibilityManager.currentLanguage.displayName)")
                    .accessibilityHint("Tap to change language")
                    
                    Toggle("Right-to-Left Layout", isOn: .constant(accessibilityManager.isRTLLanguage))
                        .disabled(true)
                        .accessibilityHint("Automatically set based on selected language")
                }
                
                // Visual Accessibility Section
                Section("Visual Accessibility") {
                    // Text Size
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "textformat.size")
                                .foregroundColor(.green)
                                .frame(width: 24)
                            Text("Text Size")
                            Spacer()
                            Text(accessibilityManager.fontSize.displayName)
                                .foregroundColor(.secondary)
                                .font(.caption)
                        }
                        
                        Picker("Font Size", selection: $accessibilityManager.fontSize) {
                            ForEach(FontSize.allCases, id: \.self) { size in
                                Text(size.displayName)
                                    .tag(size)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .onChange(of: accessibilityManager.fontSize) { newSize in
                            accessibilityManager.updateFontSize(newSize)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Text size: \(accessibilityManager.fontSize.displayName)")
                    
                    // Text Scale Factor
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "plus.magnifyingglass")
                                .foregroundColor(.orange)
                                .frame(width: 24)
                            Text("Text Scale")
                            Spacer()
                            Text("\(Int(accessibilityManager.textScaleFactor * 100))%")
                                .foregroundColor(.secondary)
                                .font(.caption)
                        }
                        
                        Slider(
                            value: $accessibilityManager.textScaleFactor,
                            in: 0.5...3.0,
                            step: 0.1
                        ) {
                            Text("Text Scale Factor")
                        } minimumValueLabel: {
                            Text("50%")
                                .font(.caption)
                        } maximumValueLabel: {
                            Text("300%")
                                .font(.caption)
                        }
                        .onChange(of: accessibilityManager.textScaleFactor) { newValue in
                            accessibilityManager.updateTextScaleFactor(newValue)
                        }
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Text scale: \(Int(accessibilityManager.textScaleFactor * 100)) percent")
                    
                    // Color Scheme
                    HStack {
                        Image(systemName: "paintbrush")
                            .foregroundColor(.purple)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Color Scheme")
                            Text(accessibilityManager.customColorScheme.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Picker("Color Scheme", selection: $accessibilityManager.customColorScheme) {
                            ForEach(AccessibilityColorScheme.allCases, id: \.self) { scheme in
                                Text(scheme.displayName)
                                    .tag(scheme)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .onChange(of: accessibilityManager.customColorScheme) { newScheme in
                            accessibilityManager.updateColorScheme(newScheme)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Color scheme: \(accessibilityManager.customColorScheme.displayName)")
                    
                    // Contrast Level
                    HStack {
                        Image(systemName: "circle.lefthalf.filled")
                            .foregroundColor(.gray)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Contrast Level")
                            Text(accessibilityManager.contrastLevel.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Picker("Contrast Level", selection: $accessibilityManager.contrastLevel) {
                            ForEach(ContrastLevel.allCases, id: \.self) { level in
                                Text(level.displayName)
                                    .tag(level)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .onChange(of: accessibilityManager.contrastLevel) { newLevel in
                            accessibilityManager.updateContrastLevel(newLevel)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Contrast level: \(accessibilityManager.contrastLevel.displayName)")
                    
                    // Color Blindness Support
                    HStack {
                        Image(systemName: "eye")
                            .foregroundColor(.red)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Color Blindness Support")
                            Text(accessibilityManager.colorBlindnessType.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Button("Info") {
                            showingColorBlindnessInfo = true
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        showingColorBlindnessInfo = true
                        accessibilityManager.provideHapticFeedback(type: .selection)
                    }
                    .accessibilityLabel("Color blindness support: \(accessibilityManager.colorBlindnessType.displayName)")
                    .accessibilityHint("Tap to configure color blindness settings")
                }
                
                // Audio and Speech Section
                Section("Audio & Speech") {
                    HStack {
                        Image(systemName: "speaker.wave.2")
                            .foregroundColor(.blue)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Voice Settings")
                            Text("Rate, Pitch, Volume")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Image(systemName: "chevron.right")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        showingVoiceSettings = true
                        accessibilityManager.provideHapticFeedback(type: .selection)
                    }
                    .accessibilityLabel("Voice settings")
                    .accessibilityHint("Tap to configure speech rate, pitch, and volume")
                    
                    Toggle("Audio Feedback", isOn: $accessibilityManager.isAudioFeedbackEnabled)
                        .onChange(of: accessibilityManager.isAudioFeedbackEnabled) { _ in
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                        .accessibilityHint("Enable audio feedback for actions")
                    
                    Toggle("Audio Descriptions", isOn: $accessibilityManager.isAudioDescriptionsEnabled)
                        .onChange(of: accessibilityManager.isAudioDescriptionsEnabled) { _ in
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                        .accessibilityHint("Enable detailed audio descriptions")
                }
                
                // Motor Accessibility Section
                Section("Motor Accessibility") {
                    // Haptic Feedback
                    HStack {
                        Image(systemName: "iphone.radiowaves.left.and.right")
                            .foregroundColor(.orange)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Haptic Feedback")
                            Text(accessibilityManager.hapticFeedbackLevel.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Picker("Haptic Feedback", selection: $accessibilityManager.hapticFeedbackLevel) {
                            ForEach(HapticFeedbackLevel.allCases, id: \.self) { level in
                                Text(level.displayName)
                                    .tag(level)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .onChange(of: accessibilityManager.hapticFeedbackLevel) { newLevel in
                            accessibilityManager.updateHapticFeedbackLevel(newLevel)
                            accessibilityManager.provideHapticFeedback(type: .impact(.medium))
                        }
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Haptic feedback: \(accessibilityManager.hapticFeedbackLevel.displayName)")
                    
                    // Touch Accommodations
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "hand.tap")
                                .foregroundColor(.green)
                                .frame(width: 24)
                            Text("Touch Hold Duration")
                            Spacer()
                            Text("\(accessibilityManager.touchAccommodations.holdDuration, specifier: "%.1f")s")
                                .foregroundColor(.secondary)
                                .font(.caption)
                        }
                        
                        Slider(
                            value: Binding(
                                get: { accessibilityManager.touchAccommodations.holdDuration },
                                set: { newValue in
                                    var accommodations = accessibilityManager.touchAccommodations
                                    accommodations.holdDuration = newValue
                                    accessibilityManager.updateTouchAccommodations(accommodations)
                                }
                            ),
                            in: 0.1...2.0,
                            step: 0.1
                        ) {
                            Text("Hold Duration")
                        } minimumValueLabel: {
                            Text("0.1s")
                                .font(.caption)
                        } maximumValueLabel: {
                            Text("2.0s")
                                .font(.caption)
                        }
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Touch hold duration: \(accessibilityManager.touchAccommodations.holdDuration, specifier: "%.1f") seconds")
                    
                    Toggle("Ignore Repeat Touches", isOn: Binding(
                        get: { accessibilityManager.touchAccommodations.ignoreRepeat },
                        set: { newValue in
                            var accommodations = accessibilityManager.touchAccommodations
                            accommodations.ignoreRepeat = newValue
                            accessibilityManager.updateTouchAccommodations(accommodations)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                    ))
                    .accessibilityHint("Prevent accidental repeated touches")
                    
                    Toggle("Touch Assistance", isOn: Binding(
                        get: { accessibilityManager.touchAccommodations.touchAssistance },
                        set: { newValue in
                            var accommodations = accessibilityManager.touchAccommodations
                            accommodations.touchAssistance = newValue
                            accessibilityManager.updateTouchAccommodations(accommodations)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                    ))
                    .accessibilityHint("Enable touch assistance features")
                }
                
                // Cognitive Accessibility Section
                Section("Cognitive Accessibility") {
                    Toggle("Simplified Interface", isOn: $accessibilityManager.isSimplifiedUIEnabled)
                        .onChange(of: accessibilityManager.isSimplifiedUIEnabled) { newValue in
                            accessibilityManager.enableSimplifiedUI(newValue)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                        .accessibilityHint("Use a simplified user interface")
                    
                    Toggle("Medication Reminders", isOn: Binding(
                        get: { accessibilityManager.reminderSettings.medicationReminders },
                        set: { newValue in
                            var settings = accessibilityManager.reminderSettings
                            settings.medicationReminders = newValue
                            accessibilityManager.updateReminderSettings(settings)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                    ))
                    .accessibilityHint("Enable medication reminder notifications")
                    
                    Toggle("Pain Tracking Reminders", isOn: Binding(
                        get: { accessibilityManager.reminderSettings.painTrackingReminders },
                        set: { newValue in
                            var settings = accessibilityManager.reminderSettings
                            settings.painTrackingReminders = newValue
                            accessibilityManager.updateReminderSettings(settings)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }
                    ))
                    .accessibilityHint("Enable pain tracking reminder notifications")
                }
                
                // System Accessibility Status Section
                Section("System Accessibility Status") {
                    AccessibilityStatusRow(
                        title: "VoiceOver",
                        isEnabled: accessibilityManager.isVoiceOverEnabled,
                        icon: "speaker.wave.3",
                        color: .blue
                    )
                    
                    AccessibilityStatusRow(
                        title: "High Contrast",
                        isEnabled: accessibilityManager.isHighContrastEnabled,
                        icon: "circle.lefthalf.filled",
                        color: .gray
                    )
                    
                    AccessibilityStatusRow(
                        title: "Reduce Motion",
                        isEnabled: accessibilityManager.isReduceMotionEnabled,
                        icon: "arrow.triangle.2.circlepath",
                        color: .orange
                    )
                    
                    AccessibilityStatusRow(
                        title: "Bold Text",
                        isEnabled: accessibilityManager.isBoldTextEnabled,
                        icon: "bold",
                        color: .purple
                    )
                    
                    AccessibilityStatusRow(
                        title: "Button Shapes",
                        isEnabled: accessibilityManager.isButtonShapesEnabled,
                        icon: "rectangle",
                        color: .green
                    )
                }
                
                // Accessibility Testing Section
                Section("Accessibility Testing") {
                    Button(action: {
                        auditReport = accessibilityManager.runAccessibilityAudit()
                        showingAccessibilityAudit = true
                        accessibilityManager.provideHapticFeedback(type: .selection)
                    }) {
                        HStack {
                            Image(systemName: "checkmark.shield")
                                .foregroundColor(.green)
                                .frame(width: 24)
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Run Accessibility Audit")
                                    .foregroundColor(.primary)
                                Text("Check app accessibility compliance")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            Image(systemName: "chevron.right")
                                .foregroundColor(.secondary)
                                .font(.caption)
                        }
                    }
                    .accessibilityLabel("Run accessibility audit")
                    .accessibilityHint("Check the app's accessibility compliance")
                    
                    Button(action: {
                        testAccessibilityFeatures()
                    }) {
                        HStack {
                            Image(systemName: "play.circle")
                                .foregroundColor(.blue)
                                .frame(width: 24)
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Test Accessibility Features")
                                    .foregroundColor(.primary)
                                Text("Test speech, haptics, and feedback")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                        }
                    }
                    .accessibilityLabel("Test accessibility features")
                    .accessibilityHint("Test speech synthesis, haptic feedback, and audio feedback")
                }
            }
            .navigationTitle("Accessibility")
            .navigationBarTitleDisplayMode(.large)
        }
        .sheet(isPresented: $showingLanguageSelector) {
            LanguageSelectorView()
        }
        .sheet(isPresented: $showingVoiceSettings) {
            VoiceSettingsView()
        }
        .sheet(isPresented: $showingColorBlindnessInfo) {
            ColorBlindnessSettingsView()
        }
        .sheet(isPresented: $showingAccessibilityAudit) {
            if let report = auditReport {
                AccessibilityAuditView(report: report)
            }
        }
    }
    
    private func testAccessibilityFeatures() {
        // Test speech synthesis
        accessibilityManager.speak("Testing speech synthesis. This is a sample message.")
        
        // Test haptic feedback
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            accessibilityManager.provideHapticFeedback(type: .impact(.medium))
        }
        
        // Test audio feedback
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            accessibilityManager.provideAudioFeedback(type: .success)
        }
        
        // Announce completion
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            accessibilityManager.announceForAccessibility("Accessibility features test completed")
        }
    }
}

struct AccessibilityStatusRow: View {
    let title: String
    let isEnabled: Bool
    let icon: String
    let color: Color
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 24)
            
            Text(title)
            
            Spacer()
            
            Image(systemName: isEnabled ? "checkmark.circle.fill" : "xmark.circle")
                .foregroundColor(isEnabled ? .green : .red)
        }
        .accessibilityLabel("\(title): \(isEnabled ? "enabled" : "disabled")")
        .accessibilityAddTraits(isEnabled ? [] : .notEnabled)
    }
}

struct LanguageSelectorView: View {
    @StateObject private var accessibilityManager = AccessibilityManager.shared
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                ForEach(SupportedLanguage.allCases, id: \.self) { language in
                    Button(action: {
                        accessibilityManager.changeLanguage(to: language)
                        accessibilityManager.provideHapticFeedback(type: .selection)
                        dismiss()
                    }) {
                        HStack {
                            Text(language.flag)
                                .font(.title2)
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text(language.displayName)
                                    .foregroundColor(.primary)
                                Text(language.code.uppercased())
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            if accessibilityManager.currentLanguage == language {
                                Image(systemName: "checkmark")
                                    .foregroundColor(.blue)
                            }
                        }
                    }
                    .accessibilityLabel("\(language.displayName) language")
                    .accessibilityHint(accessibilityManager.currentLanguage == language ? "Currently selected" : "Tap to select")
                }
            }
            .navigationTitle("Select Language")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct VoiceSettingsView: View {
    @StateObject private var accessibilityManager = AccessibilityManager.shared
    @Environment(\.dismiss) private var dismiss
    @State private var testText = "This is a test of the speech synthesis settings."
    
    var body: some View {
        NavigationView {
            List {
                Section("Speech Rate") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Rate")
                            Spacer()
                            Text("\(Int(accessibilityManager.speechRate * 100))%")
                                .foregroundColor(.secondary)
                        }
                        
                        Slider(
                            value: Binding(
                                get: { accessibilityManager.speechRate },
                                set: { newValue in
                                    accessibilityManager.updateSpeechSettings(
                                        rate: newValue,
                                        pitch: accessibilityManager.speechPitch,
                                        volume: accessibilityManager.speechVolume
                                    )
                                }
                            ),
                            in: 0.0...1.0,
                            step: 0.1
                        ) {
                            Text("Speech Rate")
                        } minimumValueLabel: {
                            Text("Slow")
                                .font(.caption)
                        } maximumValueLabel: {
                            Text("Fast")
                                .font(.caption)
                        }
                    }
                }
                
                Section("Speech Pitch") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Pitch")
                            Spacer()
                            Text("\(accessibilityManager.speechPitch, specifier: "%.1f")")
                                .foregroundColor(.secondary)
                        }
                        
                        Slider(
                            value: Binding(
                                get: { accessibilityManager.speechPitch },
                                set: { newValue in
                                    accessibilityManager.updateSpeechSettings(
                                        rate: accessibilityManager.speechRate,
                                        pitch: newValue,
                                        volume: accessibilityManager.speechVolume
                                    )
                                }
                            ),
                            in: 0.5...2.0,
                            step: 0.1
                        ) {
                            Text("Speech Pitch")
                        } minimumValueLabel: {
                            Text("Low")
                                .font(.caption)
                        } maximumValueLabel: {
                            Text("High")
                                .font(.caption)
                        }
                    }
                }
                
                Section("Speech Volume") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Volume")
                            Spacer()
                            Text("\(Int(accessibilityManager.speechVolume * 100))%")
                                .foregroundColor(.secondary)
                        }
                        
                        Slider(
                            value: Binding(
                                get: { accessibilityManager.speechVolume },
                                set: { newValue in
                                    accessibilityManager.updateSpeechSettings(
                                        rate: accessibilityManager.speechRate,
                                        pitch: accessibilityManager.speechPitch,
                                        volume: newValue
                                    )
                                }
                            ),
                            in: 0.0...1.0,
                            step: 0.1
                        ) {
                            Text("Speech Volume")
                        } minimumValueLabel: {
                            Text("Quiet")
                                .font(.caption)
                        } maximumValueLabel: {
                            Text("Loud")
                                .font(.caption)
                        }
                    }
                }
                
                Section("Test Speech") {
                    VStack(alignment: .leading, spacing: 12) {
                        TextField("Test Text", text: $testText, axis: .vertical)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .lineLimit(3...6)
                        
                        HStack {
                            Button("Test Speech") {
                                accessibilityManager.speak(testText)
                                accessibilityManager.provideHapticFeedback(type: .selection)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(Colors.Primary.p500)

                            Spacer()
                            
                            Button("Stop") {
                                accessibilityManager.stopSpeaking()
                                accessibilityManager.provideHapticFeedback(type: .selection)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
            }
            .navigationTitle("Voice Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct ColorBlindnessSettingsView: View {
    @StateObject private var accessibilityManager = AccessibilityManager.shared
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                Section("Color Blindness Type") {
                    ForEach(ColorBlindnessType.allCases, id: \.self) { type in
                        Button(action: {
                            accessibilityManager.updateColorBlindnessType(type)
                            accessibilityManager.provideHapticFeedback(type: .selection)
                        }) {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(type.displayName)
                                        .foregroundColor(.primary)
                                    Text(type.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                
                                Spacer()
                                
                                if accessibilityManager.colorBlindnessType == type {
                                    Image(systemName: "checkmark")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                        .accessibilityLabel(type.displayName)
                        .accessibilityHint(type.description)
                    }
                }
                
                Section("Color Test") {
                    VStack(spacing: 16) {
                        Text("Color Test Samples")
                            .font(.headline)
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 12) {
                            ForEach([Color.red, Color.green, Color.blue, Color.orange, Color.purple, Color.yellow, Color.pink, Color.cyan], id: \.self) { color in
                                Rectangle()
                                    .fill(accessibilityManager.getAccessibleColor(for: color))
                                    .frame(height: 40)
                                    .cornerRadius(8)
                            }
                        }
                        
                        Text("These colors are adjusted based on your selected color blindness type.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.vertical)
                }
            }
            .navigationTitle("Color Blindness")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct AccessibilityAuditView: View {
    let report: AccessibilityAuditReport
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                Section("Audit Summary") {
                    HStack {
                        Text("Overall Score")
                        Spacer()
                        Text("\(Int(report.overallScore))%")
                            .foregroundColor(scoreColor)
                            .fontWeight(.semibold)
                    }
                    
                    HStack {
                        Text("Issues Found")
                        Spacer()
                        Text("\(report.issues.count)")
                            .foregroundColor(report.issues.isEmpty ? .green : .orange)
                    }
                    
                    HStack {
                        Text("Audit Date")
                        Spacer()
                        Text(report.timestamp, style: .date)
                            .foregroundColor(.secondary)
                    }
                }
                
                if !report.issues.isEmpty {
                    Section("Issues") {
                        ForEach(Array(report.issues.enumerated()), id: \.offset) { index, issue in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(issue.description)
                                        .fontWeight(.medium)
                                    Spacer()
                                    Text(severityText(issue.severity))
                                        .font(.caption)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 2)
                                        .background(severityColor(issue.severity))
                                        .foregroundColor(.white)
                                        .cornerRadius(4)
                                }
                                
                                Text(issue.recommendation)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
                
                Section("Recommendations") {
                    ForEach(Array(report.recommendations.enumerated()), id: \.offset) { index, recommendation in
                        HStack(alignment: .top) {
                            Image(systemName: "lightbulb")
                                .foregroundColor(.yellow)
                                .frame(width: 20)
                            Text(recommendation)
                        }
                    }
                }
            }
            .navigationTitle("Accessibility Audit")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private var scoreColor: Color {
        switch report.overallScore {
        case 90...100: return .green
        case 70..<90: return .orange
        default: return .red
        }
    }
    
    private func severityText(_ severity: AccessibilityIssue.IssueSeverity) -> String {
        switch severity {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .critical: return "Critical"
        }
    }
    
    private func severityColor(_ severity: AccessibilityIssue.IssueSeverity) -> Color {
        switch severity {
        case .low: return .blue
        case .medium: return .orange
        case .high: return .red
        case .critical: return .purple
        }
    }
}

#Preview {
    AccessibilitySettingsView()
}