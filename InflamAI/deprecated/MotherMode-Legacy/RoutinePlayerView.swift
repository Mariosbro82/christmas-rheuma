//
//  RoutinePlayerView.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct RoutinePlayerView: View {
    @ObservedObject var viewModel: RoutinePlayerViewModel
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(String(localized: "disclaimer.general_info"))
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.leading)
            
            Text(viewModel.routine.titleKey)
                .font(.title2.weight(.semibold))
            Text(viewModel.routine.subtitleKey)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            if let step = viewModel.currentStep {
                VStack(alignment: .leading, spacing: 8) {
                    Text(step.title)
                        .font(.headline)
                    Text(step.instruction)
                        .font(.body)
                    if let modification = step.modification {
                        Text("Modification: \(modification)")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Text(String(localized: "mother.micro.stop"))
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 18).fill(Color(.secondarySystemBackground)))
            }
            
            Spacer()
            
            Text(viewModel.timerDisplay)
                .font(.system(size: 50, weight: .bold, design: .rounded))
                .frame(maxWidth: .infinity, alignment: .center)
            
            HStack(spacing: 16) {
                Button {
                    viewModel.togglePlay()
                } label: {
                    Label(viewModel.isPlaying ? "Pause" : "Start", systemImage: viewModel.isPlaying ? "pause.fill" : "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                
                Button(String(localized: "mother.micro.stop")) {
                    viewModel.stop()
                    dismiss()
                }
                .frame(maxWidth: .infinity)
                .buttonStyle(.bordered)
            }
            
            Button(String(localized: "mother.micro.complete")) {
                viewModel.completeRoutine()
                dismiss()
            }
            .frame(maxWidth: .infinity)
            .buttonStyle(.bordered)
            .disabled(viewModel.remainingSeconds > 0)
        }
        .padding()
        .interactiveDismissDisabled()
    }
}
