//
//  RoutinePlayerViewModel.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation

@MainActor
final class RoutinePlayerViewModel: ObservableObject {
    @Published private(set) var routine: ExerciseRoutine
    @Published private(set) var currentStepIndex: Int = 0
    @Published private(set) var remainingSeconds: Int
    @Published private(set) var currentStepRemaining: Int
    @Published private(set) var isPlaying = false
    
    private weak var environment: TraeAppEnvironment?
    private var timer: Timer?
    
    init(routine: ExerciseRoutine, environment: TraeAppEnvironment?) {
        self.routine = routine
        self.environment = environment
        self.remainingSeconds = routine.steps.reduce(0) { $0 + $1.durationSeconds }
        self.currentStepRemaining = routine.steps.first?.durationSeconds ?? 0
    }
    
    var currentStep: ExerciseRoutine.Step? {
        guard currentStepIndex < routine.steps.count else { return nil }
        return routine.steps[currentStepIndex]
    }
    
    var timerDisplay: String {
        let minutes = remainingSeconds / 60
        let seconds = remainingSeconds % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
    
    func togglePlay() {
        isPlaying.toggle()
        if isPlaying {
            if currentStepRemaining == 0, let step = currentStep {
                currentStepRemaining = step.durationSeconds
            }
            startTimer()
        } else {
            stopTimer()
        }
    }
    
    func stop() {
        isPlaying = false
        stopTimer()
    }
    
    func completeRoutine() {
        stopTimer()
        remainingSeconds = 0
        currentStepIndex = routine.steps.count
        currentStepRemaining = 0
    }
    
    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { await self?.tick() }
        }
    }
    
    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
    
    private func tick() {
        guard remainingSeconds > 0 else {
            completeRoutine()
            return
        }
        remainingSeconds -= 1
        currentStepRemaining = max(currentStepRemaining - 1, 0)
        if currentStepRemaining == 0 {
            advanceStep()
        }
    }
    
    private func advanceStep() {
        guard currentStepIndex < routine.steps.count - 1 else {
            completeRoutine()
            return
        }
        currentStepIndex += 1
        currentStepRemaining = routine.steps[currentStepIndex].durationSeconds
    }
}
