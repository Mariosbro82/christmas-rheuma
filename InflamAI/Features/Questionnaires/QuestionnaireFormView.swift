//
//  QuestionnaireFormView.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-06-09.
//

import SwiftUI
import CoreData

struct QuestionnaireFormView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) var reduceMotion

    let questionnaireID: QuestionnaireID
    let definition: QuestionnaireDefinition

    @State private var currentIndex: Int = 0
    @State private var sliderValues: [String: Double] = [:]
    @State private var touchedItems: Set<String> = []
    @State private var note: String = ""
    @State private var showSpeedAlert = false
    @State private var showValidationAlert = false
    @State private var validationMessage: String = ""
    @State private var saveMessage: String?
    @State private var showSaveConfirmation = false
    @State private var isSaving = false
    @State private var startTime = Date()
    @State private var showNotes = false
    @State private var showHistory = false

    private var context: NSManagedObjectContext {
        if viewContext.persistentStoreCoordinator != nil {
            return viewContext
        }
        return InflamAIPersistenceController.shared.container.viewContext
    }

    private var progress: Double {
        guard !definition.items.isEmpty else { return 0 }
        return Double(currentIndex + 1) / Double(definition.items.count)
    }

    private var isOnLastQuestion: Bool {
        currentIndex == definition.items.count - 1
    }

    init(questionnaireID: QuestionnaireID) {
        self.questionnaireID = questionnaireID
        if let def = QuestionnaireDefinition.definition(for: questionnaireID) {
            self.definition = def
        } else {
            self.definition = QuestionnaireDefinition(
                id: questionnaireID,
                version: "0.0.1-placeholder",
                items: [],
                periodDescriptionKey: "questionnaire.placeholder.period",
                notesAllowed: true
            )
        }
    }

    var body: some View {
        // CRIT-001 FIX: Removed NavigationStack wrapper.
        // This view is presented via NavigationLink from AssessmentsView,
        // which is already inside NavigationView from MoreView tab.
        VStack(spacing: 0) {
            // Fixed Progress Header
            progressHeader

            // Scrollable Question Content
            ScrollView {
                VStack(spacing: 24) {
                    questionPager
                }
                .padding(.vertical, 24)
            }
            .scrollIndicators(.hidden)

            // Fixed Navigation Footer
            navigationFooter
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle(NSLocalizedString(definition.id.titleKey, comment: ""))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // Cancel button removed - back arrow provides same functionality
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showHistory = true
                } label: {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                }
                .accessibilityLabel("View History")
            }
        }
        .sheet(isPresented: $showHistory) {
            QuestionnaireHistoryView(questionnaireID: questionnaireID)
        }
        .onAppear {
            startTime = Date()
        }
        .sheet(isPresented: $showNotes) {
            notesSheet
        }
        .alert("questionnaire.validation.title", isPresented: $showValidationAlert, actions: {
            Button("OK", role: .cancel) { }
        }, message: {
            Text(validationMessage)
        })
        .alert("questionnaire.speed_check.title", isPresented: $showSpeedAlert, actions: {
            Button("questionnaire.speed_check.confirm", role: .destructive) {
                dismiss()
            }
            Button("questionnaire.speed_check.review", role: .cancel) { }
        }, message: {
            Text("questionnaire.speed_check.message")
        })
        .alert("questionnaire.save_success.title", isPresented: $showSaveConfirmation, actions: {
            Button("OK") {
                if saveMessage == "questionnaire.save_success.body" {
                    dismiss()
                }
            }
        }, message: {
            Text(saveMessage ?? "")
        })
    }

    // MARK: - Progress Header

    private var progressHeader: some View {
        VStack(spacing: 12) {
            // Premium Progress Bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.systemGray5))
                        .frame(height: 8)

                    RoundedRectangle(cornerRadius: 4)
                        .fill(
                            LinearGradient(
                                colors: [.orange, .yellow],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * progress, height: 8)
                        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: progress)
                }
            }
            .frame(height: 8)

            HStack {
                // Question counter badge
                HStack(spacing: 6) {
                    Circle()
                        .fill(Color.orange)
                        .frame(width: 8, height: 8)

                    Text("\(currentIndex + 1) / \(definition.items.count)")
                        .font(.subheadline)
                        .fontWeight(.medium)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    Capsule()
                        .fill(Color.orange.opacity(0.12))
                )

                Spacer()

                Text(NSLocalizedString(definition.periodDescriptionKey, comment: ""))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)
        .padding(.bottom, 8)
        .background(Color(.systemBackground))
    }

    // MARK: - Question Pager

    private var questionPager: some View {
        let item = definition.items[currentIndex]
        let currentValue = sliderValues[item.id] ?? Double(item.minimum)
        let isTouched = touchedItems.contains(item.id)

        return VStack(spacing: 28) {
            // Question Text
            Text(NSLocalizedString(item.promptKey, comment: ""))
                .font(.title2)
                .fontWeight(.semibold)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
                .fixedSize(horizontal: false, vertical: true)

            // Value Display Card
            VStack(spacing: 16) {
                // Large Value
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(isTouched ? String(format: "%.0f", currentValue) : "--")
                        .font(.system(size: 72, weight: .bold, design: .rounded))
                        .foregroundColor(isTouched ? colorForValue(currentValue, item: item) : .secondary)
                        .contentTransition(.numericText())
                        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: currentValue)

                    Text("/ \(item.maximum)")
                        .font(.system(size: 24, weight: .medium, design: .rounded))
                        .foregroundColor(.secondary)
                }

                // Anchor labels
                if let minAnchor = item.anchors[item.minimum], let maxAnchor = item.anchors[item.maximum] {
                    HStack {
                        Text(NSLocalizedString(minAnchor, comment: ""))
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(NSLocalizedString(maxAnchor, comment: ""))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal, 8)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 24)
            .padding(.horizontal, 16)
            .background(
                RoundedRectangle(cornerRadius: 24)
                    .fill(Color(.systemBackground))
                    .shadow(color: (isTouched ? colorForValue(currentValue, item: item) : Color.gray).opacity(0.15), radius: 20, x: 0, y: 8)
            )
            .padding(.horizontal, 20)

            // Premium Slider
            VStack(spacing: 12) {
                // Visual track
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color(.systemGray5))
                            .frame(height: 12)

                        RoundedRectangle(cornerRadius: 6)
                            .fill(
                                LinearGradient(
                                    colors: gradientForValue(currentValue, item: item),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: geometry.size.width * (currentValue / Double(item.maximum)), height: 12)
                            .animation(.spring(response: 0.3, dampingFraction: 0.8), value: currentValue)
                    }
                }
                .frame(height: 12)
                .padding(.horizontal, 20)

                // Functional Slider
                Slider(
                    value: binding(for: item),
                    in: Double(item.minimum)...Double(item.maximum),
                    step: 1
                ) { editing in
                    if !editing {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    }
                }
                .tint(colorForValue(currentValue, item: item))
                .padding(.horizontal, 20)
                .accessibilityIdentifier("slider_\(item.id)")
                .onChange(of: sliderValues[item.id]) { _ in
                    touchedItems.insert(item.id)
                }

                // Value labels
                HStack {
                    Text("\(item.minimum)")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(item.maximum)")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 20)
            }
        }
        .id(currentIndex)
        .transition(.asymmetric(
            insertion: .move(edge: .trailing).combined(with: .opacity),
            removal: .move(edge: .leading).combined(with: .opacity)
        ))
    }

    // MARK: - Navigation Footer

    private var navigationFooter: some View {
        VStack(spacing: 0) {
            Divider()

            HStack(spacing: 12) {
                // Previous button
                if currentIndex > 0 {
                    Button {
                        withAnimation(reduceMotion ? .none : .spring(response: 0.35, dampingFraction: 0.8)) {
                            currentIndex -= 1
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 14, weight: .semibold))
                            Text("Previous")
                                .fontWeight(.medium)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .foregroundColor(.primary)
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }
                }

                // Next/Complete button
                Button {
                    if isOnLastQuestion {
                        submit()
                    } else {
                        withAnimation(reduceMotion ? .none : .spring(response: 0.35, dampingFraction: 0.8)) {
                            currentIndex += 1
                        }
                    }
                } label: {
                    HStack(spacing: 6) {
                        Text(isOnLastQuestion ? "Complete" : "Next")
                            .fontWeight(.semibold)
                        Image(systemName: isOnLastQuestion ? "checkmark" : "chevron.right")
                            .font(.system(size: 14, weight: .semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .foregroundColor(.white)
                    .background(
                        LinearGradient(
                            colors: isOnLastQuestion ? [.green, .mint] : [.orange, .yellow],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(12)
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .background(Color(.systemBackground))
        }
    }

    // MARK: - Notes Sheet

    private var notesSheet: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                Text("Add any additional notes about this assessment")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                TextEditor(text: $note)
                    .frame(minHeight: 200)
                    .padding(12)
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                Spacer()
            }
            .padding()
            .navigationTitle("Notes")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        showNotes = false
                    }
                }
            }
        }
        .presentationDetents([.medium])
    }

    // MARK: - Helpers

    private func colorForValue(_ value: Double, item: QuestionnaireItem) -> Color {
        let percentage = value / Double(item.maximum)
        switch percentage {
        case 0..<0.3: return .green
        case 0.3..<0.5: return .yellow
        case 0.5..<0.7: return .orange
        default: return .red
        }
    }

    private func gradientForValue(_ value: Double, item: QuestionnaireItem) -> [Color] {
        let percentage = value / Double(item.maximum)
        switch percentage {
        case 0..<0.3: return [.green, .mint]
        case 0.3..<0.5: return [.yellow, .orange]
        case 0.5..<0.7: return [.orange, .red]
        default: return [.red, .pink]
        }
    }
    
    private func progressLabel(for index: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .none
        return String(
            format: NSLocalizedString("questionnaire.progress_format", comment: ""),
            formatter.string(from: NSNumber(value: index + 1)) ?? "\(index + 1)",
            formatter.string(from: NSNumber(value: definition.items.count)) ?? "\(definition.items.count)"
        )
    }
    
    private func binding(for item: QuestionnaireItem) -> Binding<Double> {
        Binding(
            get: {
                sliderValues[item.id] ?? Double(item.minimum)
            },
            set: { newValue in
                sliderValues[item.id] = newValue
            }
        )
    }
    
    private func displayValue(for item: QuestionnaireItem) -> String {
        if touchedItems.contains(item.id), let value = sliderValues[item.id] {
            return String(format: "%.0f", value)
        }
        return "--"
    }
    
    private func anchorLabel(for item: QuestionnaireItem, value: Int) -> String {
        guard let key = item.anchors[value] else {
            return "\(value)"
        }
        return NSLocalizedString(key, comment: "")
    }
    
    private func currentAnswerSet() -> QuestionnaireAnswerSet {
        let answeredValues = sliderValues.filter { touchedItems.contains($0.key) }
        return QuestionnaireAnswerSet(values: answeredValues)
    }
    
    private func submit() {
        isSaving = true
        defer { isSaving = false }
        
        guard touchedItems.count == definition.items.count else {
            validationMessage = NSLocalizedString("questionnaire.validation.incomplete", comment: "")
            showValidationAlert = true
            return
        }
        
        let answerSet = currentAnswerSet()
        let duration = Date().timeIntervalSince(startTime)
        let manager = QuestionnaireManager(viewContext: context)
        
        do {
            let outcome = try manager.recordResponse(
                for: questionnaireID,
                answers: answerSet,
                note: note.isEmpty ? nil : note,
                duration: duration,
                appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev",
                deviceLocale: .current,
                isDraft: false,
                createdAt: Date()
            )
            note.removeAll()
            sliderValues.removeAll()
            touchedItems.removeAll()
            startTime = Date()
            if outcome.requiresSpeedConfirmation {
                showSpeedAlert = true
            } else {
                saveMessage = "questionnaire.save_success.body"
                showSaveConfirmation = true
            }
        } catch {
            validationMessage = error.localizedDescription
            showValidationAlert = true
        }
    }
    
    private func saveDraft() {
        let manager = QuestionnaireManager(viewContext: context)
        let answerSet = currentAnswerSet()
        do {
            _ = try manager.recordResponse(
                for: questionnaireID,
                answers: answerSet,
                note: note.isEmpty ? nil : note,
                duration: Date().timeIntervalSince(startTime),
                appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev",
                deviceLocale: .current,
                isDraft: true,
                createdAt: Date()
            )
            saveMessage = "questionnaire.draft_saved.body"
            showSaveConfirmation = true
        } catch {
            validationMessage = error.localizedDescription
            showValidationAlert = true
        }
    }
    
    private func resetForm() {
        sliderValues.removeAll()
        touchedItems.removeAll()
        note.removeAll()
        currentIndex = 0
        startTime = Date()
    }
}

struct QuestionnaireFormView_Previews: PreviewProvider {
    static var previews: some View {
        QuestionnaireFormView(questionnaireID: .basdai)
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}
