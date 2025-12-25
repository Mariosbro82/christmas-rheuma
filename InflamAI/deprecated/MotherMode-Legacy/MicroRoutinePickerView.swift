//
//  MicroRoutinePickerView.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct MicroRoutinePickerView: View {
    let routines: [ExerciseRoutine]
    let isPregnantOrPostpartum: Bool
    var onSelect: (ExerciseRoutine) -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            Capsule()
                .frame(width: 40, height: 4)
                .foregroundStyle(Color.secondary)
                .padding(.top, 8)
            
            Text(String(localized: "mother.micro.title"))
                .font(.headline)
            
            if isPregnantOrPostpartum {
                Text(String(localized: "mother.pregnancy.caution"))
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            
            ScrollView {
                VStack(spacing: 12) {
                    ForEach(routines) { routine in
                        Button {
                            onSelect(routine)
                        } label: {
                            routineCard(for: routine)
                        }
                        .buttonStyle(.plain)
                        .accessibilityHint("Starts micro routine \(routine.titleKey)")
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 20)
            }
        }
        .padding(.bottom, 12)
    }
    
    private func routineCard(for routine: ExerciseRoutine) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(routine.titleKey)
                .font(.headline)
            Text(routine.subtitleKey)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            HStack {
                Label("\(routine.estimatedMinutes) min", systemImage: "clock")
                    .font(.caption)
                if routine.titleKey == "routine.morning_mobility.title" {
                    Label(String(localized: "mother.micro.stroller"), systemImage: "figure.roll")
                        .font(.caption)
                }
                if routine.titleKey == "routine.desk_unwind.title" {
                    Label(String(localized: "mother.micro.babywearing"), systemImage: "figure.wave.circle")
                        .font(.caption)
                }
            }
            Text(String(localized: "mother.micro.stop"))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 18).fill(Color(.secondarySystemBackground)))
    }
}
