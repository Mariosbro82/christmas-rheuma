//
//  SpeechRecognizer.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import Foundation
import Speech
import AVFoundation
import Combine

class SpeechRecognizer: NSObject, ObservableObject {
    @Published var transcript = ""
    @Published var isRecording = false
    @Published var isAvailable = false
    @Published var confidence: Float = 0.0
    @Published var partialResults: [String] = []
    @Published var finalResults: [String] = []
    
    private let speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    private var cancellables = Set<AnyCancellable>()
    
    // Configuration
    var locale: Locale = Locale(identifier: "en-US") {
        didSet {
            setupSpeechRecognizer()
        }
    }
    
    var requiresOnDeviceRecognition = true
    var shouldReportPartialResults = true
    var contextualStrings: [String] = []
    
    override init() {
        speechRecognizer = SFSpeechRecognizer(locale: locale)
        super.init()
        setupSpeechRecognizer()
        requestPermissions()
    }
    
    init(locale: Locale) {
        self.locale = locale
        speechRecognizer = SFSpeechRecognizer(locale: locale)
        super.init()
        setupSpeechRecognizer()
        requestPermissions()
    }
    
    private func setupSpeechRecognizer() {
        speechRecognizer?.delegate = self
        isAvailable = speechRecognizer?.isAvailable ?? false
    }
    
    private func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { [weak self] authStatus in
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized:
                    self?.isAvailable = self?.speechRecognizer?.isAvailable ?? false
                case .denied, .restricted, .notDetermined:
                    self?.isAvailable = false
                @unknown default:
                    self?.isAvailable = false
                }
            }
        }
        
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            DispatchQueue.main.async {
                if !granted {
                    self?.isAvailable = false
                }
            }
        }
    }
    
    func startRecording() {
        guard isAvailable && !isRecording else { return }
        
        // Cancel any ongoing recognition
        stopRecording()
        
        do {
            // Setup audio session
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            
            // Create recognition request
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else { return }
            
            recognitionRequest.shouldReportPartialResults = shouldReportPartialResults
            
            if #available(iOS 13.0, *) {
                recognitionRequest.requiresOnDeviceRecognition = requiresOnDeviceRecognition
            }
            
            // Add contextual strings for better recognition
            if !contextualStrings.isEmpty {
                recognitionRequest.contextualStrings = contextualStrings
            }
            
            // Setup audio input
            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                recognitionRequest.append(buffer)
            }
            
            // Start audio engine
            audioEngine.prepare()
            try audioEngine.start()
            
            // Start recognition task
            recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                self?.handleRecognitionResult(result: result, error: error)
            }
            
            DispatchQueue.main.async {
                self.isRecording = true
                self.transcript = ""
                self.partialResults.removeAll()
                self.finalResults.removeAll()
            }
            
        } catch {
            print("Failed to start recording: \(error)")
        }
    }
    
    func stopRecording() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        
        recognitionRequest = nil
        recognitionTask = nil
        
        DispatchQueue.main.async {
            self.isRecording = false
        }
        
        // Deactivate audio session
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to deactivate audio session: \(error)")
        }
    }
    
    private func handleRecognitionResult(result: SFSpeechRecognitionResult?, error: Error?) {
        if let error = error {
            print("Recognition error: \(error)")
            DispatchQueue.main.async {
                self.stopRecording()
            }
            return
        }
        
        guard let result = result else { return }
        
        DispatchQueue.main.async {
            let newTranscript = result.bestTranscription.formattedString
            self.transcript = newTranscript
            self.confidence = result.bestTranscription.averageConfidence
            
            if result.isFinal {
                self.finalResults.append(newTranscript)
                self.stopRecording()
            } else {
                // Update partial results
                if !self.partialResults.contains(newTranscript) {
                    self.partialResults.append(newTranscript)
                }
            }
        }
    }
    
    // MARK: - Utility Methods
    
    func reset() {
        transcript = ""
        confidence = 0.0
        partialResults.removeAll()
        finalResults.removeAll()
    }
    
    func addContextualStrings(_ strings: [String]) {
        contextualStrings.append(contentsOf: strings)
    }
    
    func clearContextualStrings() {
        contextualStrings.removeAll()
    }
    
    func setMedicalContextualStrings() {
        contextualStrings = [
            // Pain levels
            "pain level", "zero", "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten",
            
            // Body parts
            "head", "neck", "shoulder", "arm", "elbow", "wrist", "hand",
            "chest", "back", "spine", "hip", "thigh", "knee", "calf", "ankle", "foot",
            "cervical", "thoracic", "lumbar", "sacral",
            
            // Medical terms
            "arthritis", "rheumatoid", "osteoarthritis", "fibromyalgia",
            "inflammation", "swelling", "stiffness", "tenderness",
            
            // Medications
            "ibuprofen", "advil", "motrin", "tylenol", "acetaminophen",
            "aspirin", "naproxen", "aleve", "tramadol", "gabapentin",
            
            // Actions
            "take medication", "log pain", "save entry", "emergency",
            "analyze patterns", "pain summary", "remove pain", "clear pain"
        ]
    }
    
    // MARK: - Advanced Features
    
    func getAlternativeTranscriptions() -> [String] {
        guard let task = recognitionTask,
              let result = task.result else {
            return []
        }
        
        return result.transcriptions.map { $0.formattedString }
    }
    
    func getSegmentConfidences() -> [Float] {
        guard let task = recognitionTask,
              let result = task.result else {
            return []
        }
        
        return result.bestTranscription.segments.map { $0.confidence }
    }
    
    func getTimestamps() -> [(String, TimeInterval)] {
        guard let task = recognitionTask,
              let result = task.result else {
            return []
        }
        
        return result.bestTranscription.segments.map { segment in
            (segment.substring, segment.timestamp)
        }
    }
    
    // MARK: - Voice Activity Detection
    
    private var voiceActivityThreshold: Float = 0.01
    private var silenceTimeout: TimeInterval = 2.0
    private var silenceTimer: Timer?
    
    func enableVoiceActivityDetection() {
        // This would require additional audio processing
        // For now, we'll use a simple timer-based approach
        startSilenceTimer()
    }
    
    private func startSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceTimeout, repeats: false) { [weak self] _ in
            if self?.isRecording == true {
                self?.stopRecording()
            }
        }
    }
    
    private func resetSilenceTimer() {
        if isRecording {
            startSilenceTimer()
        }
    }
    
    deinit {
        silenceTimer?.invalidate()
        stopRecording()
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension SpeechRecognizer: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        DispatchQueue.main.async {
            self.isAvailable = available
            if !available && self.isRecording {
                self.stopRecording()
            }
        }
    }
}

// MARK: - Convenience Extensions

extension SpeechRecognizer {
    static func requestPermissions() async -> Bool {
        let speechAuth = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
        
        let audioAuth = await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
        
        return speechAuth && audioAuth
    }
    
    func transcribe(audioURL: URL) async throws -> String {
        guard let speechRecognizer = speechRecognizer else {
            throw SpeechRecognitionError.recognizerUnavailable
        }
        
        let request = SFSpeechURLRecognitionRequest(url: audioURL)
        request.requiresOnDeviceRecognition = requiresOnDeviceRecognition
        request.shouldReportPartialResults = false
        
        return try await withCheckedThrowingContinuation { continuation in
            speechRecognizer.recognitionTask(with: request) { result, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else if let result = result, result.isFinal {
                    continuation.resume(returning: result.bestTranscription.formattedString)
                }
            }
        }
    }
}

// MARK: - Error Types

enum SpeechRecognitionError: Error, LocalizedError {
    case recognizerUnavailable
    case audioEngineError
    case recognitionFailed
    case permissionDenied
    
    var errorDescription: String? {
        switch self {
        case .recognizerUnavailable:
            return "Speech recognizer is not available"
        case .audioEngineError:
            return "Audio engine error"
        case .recognitionFailed:
            return "Speech recognition failed"
        case .permissionDenied:
            return "Speech recognition permission denied"
        }
    }
}

// MARK: - Configuration

struct SpeechRecognitionConfig {
    var locale: Locale = Locale(identifier: "en-US")
    var requiresOnDeviceRecognition = true
    var shouldReportPartialResults = true
    var contextualStrings: [String] = []
    var voiceActivityDetection = false
    var silenceTimeout: TimeInterval = 2.0
    
    static let medical = SpeechRecognitionConfig(
        locale: Locale(identifier: "en-US"),
        requiresOnDeviceRecognition: true,
        shouldReportPartialResults: true,
        contextualStrings: [
            "pain level", "arthritis", "rheumatoid", "inflammation",
            "ibuprofen", "tylenol", "medication", "emergency"
        ],
        voiceActivityDetection: true,
        silenceTimeout: 3.0
    )
}