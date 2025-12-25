//
//  QuickCheckInView.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct QuickCheckInView: View {
    @ObservedObject var viewModel: QuickCheckInViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var isRecording = false
    @State private var showPermissionAlert = false
    
    var body: some View {
        VStack(spacing: 16) {
            Capsule()
                .frame(width: 40, height: 4)
                .foregroundStyle(Color.secondary)
                .padding(.top, 8)
            
            Text(String(localized: "mother.quick.title"))
                .font(.headline)
            
            Text(String(localized: "mother.quick.prompt"))
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            
            Slider(value: Binding(
                get: { viewModel.mood },
                set: { viewModel.updateMood($0) }
            ), in: 0...10, step: 1)
            .accessibilityLabel(String(localized: "mother.quick.prompt"))
            
            Text("\(Int(viewModel.mood))/10")
                .font(.system(size: 40, weight: .bold, design: .rounded))
                .accessibilityHidden(true)
            
            tagsView
            
            Button {
                Task { await toggleRecording() }
            } label: {
            Label(
                    isRecording ? String(localized: "common.stop") : String(localized: "mother.quick.voice_record"),
                    systemImage: isRecording ? "stop.circle.fill" : "mic.circle.fill"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .padding(.top, 4)
            .accessibilityHint("Records a short voice note that stays on device.")
            
            Button(String(localized: "checkin.save")) {
                viewModel.save()
                dismiss()
            }
            .buttonStyle(.borderedProminent)
            .frame(maxWidth: .infinity)
        }
        .padding(20)
        .alert(String(localized: "mother.quick.voice_consent"), isPresented: $showPermissionAlert) {
            Button("OK", role: .cancel) { }
        }
    }
    
    private var tagsView: some View {
        HStack(spacing: 8) {
            ForEach(viewModel.autoTags, id: \.self) { key in
                Text(LocalizedStringKey(key))
                    .font(.caption)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Capsule().fill(Color(.secondarySystemBackground)))
            }
        }
        .frame(maxWidth: .infinity)
    }
    
    private func toggleRecording() async {
        if isRecording {
            viewModel.stopRecording()
            isRecording = false
            return
        }
        let success = await viewModel.startRecording()
        if success {
            isRecording = true
        } else {
            showPermissionAlert = true
        }
    }
}
