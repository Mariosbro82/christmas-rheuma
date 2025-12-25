//
//  QuestionnaireSettingsView.swift
//  InflamAI-Swift
//
//  Comprehensive questionnaire management interface
//  Allows users to browse, enable/disable, and configure all 40+ questionnaires
//

import SwiftUI

struct QuestionnaireSettingsView: View {
    @StateObject private var preferences = QuestionnaireUserPreferences.shared
    @State private var searchText = ""
    @State private var selectedCategory: DiseaseCategory?
    @State private var selectedQuestionnaire: QuestionnaireID?
    @State private var showingConfig = false

    var body: some View {
        List {
            // Summary Section
            Section {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Active Questionnaires")
                            .font(.headline)
                        Text("\(preferences.enabledCount) enabled")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Image(systemName: "checkmark.circle.fill")
                        .font(.title2)
                        .foregroundColor(.green)
                }
                .padding(.vertical, 4)
            }

            // Category Filter
            if !searchText.isEmpty {
                Section {
                    ForEach(filteredQuestionnaires, id: \.id) { questionnaire in
                        questionnaireRow(for: questionnaire)
                    }
                }
            } else {
                // Grouped by Category
                ForEach(groupedQuestionnaires, id: \.category) { group in
                    Section(header: categoryHeader(for: group.category)) {
                        ForEach(group.questionnaires, id: \.id) { questionnaire in
                            questionnaireRow(for: questionnaire)
                        }
                    }
                }
            }
        }
        .searchable(text: $searchText, prompt: "Search questionnaires")
        .navigationTitle("Questionnaires")
        .navigationBarTitleDisplayMode(.large)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Menu {
                    ForEach(DiseaseCategory.allCases, id: \.self) { category in
                        Button {
                            selectedCategory = selectedCategory == category ? nil : category
                        } label: {
                            Label(category.rawValue, systemImage: category.icon)
                            if selectedCategory == category {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                    Divider()
                    Button {
                        selectedCategory = nil
                    } label: {
                        Label("Show All", systemImage: "list.bullet")
                    }
                } label: {
                    Image(systemName: "line.3.horizontal.decrease.circle")
                }
            }
        }
        .sheet(item: $selectedQuestionnaire) { questionnaireID in
            NavigationView {
                QuestionnaireConfigView(questionnaireID: questionnaireID)
            }
        }
    }

    // MARK: - View Builders

    private func categoryHeader(for category: DiseaseCategory) -> some View {
        HStack {
            Image(systemName: category.icon)
                .foregroundColor(.blue)
            Text(category.rawValue)
        }
    }

    private func questionnaireRow(for questionnaireID: QuestionnaireID) -> some View {
        Button {
            selectedQuestionnaire = questionnaireID
        } label: {
            HStack(spacing: 12) {
                // Toggle
                Toggle("", isOn: Binding(
                    get: { preferences.isEnabled(questionnaireID) },
                    set: { isEnabled in
                        if isEnabled {
                            preferences.enableQuestionnaire(questionnaireID)
                        } else {
                            preferences.disableQuestionnaire(questionnaireID)
                        }
                    }
                ))
                .toggleStyle(SwitchToggleStyle(tint: .blue))
                .labelsHidden()

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(NSLocalizedString(questionnaireID.titleKey, comment: ""))
                            .font(.body)
                            .foregroundColor(.primary)

                        if questionnaireID.isDefault {
                            Text("DEFAULT")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.blue)
                                .cornerRadius(4)
                        }
                    }

                    Text(NSLocalizedString(questionnaireID.descriptionKey, comment: ""))
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)

                    // Schedule info
                    HStack(spacing: 4) {
                        Image(systemName: scheduleIcon(for: questionnaireID))
                            .font(.caption2)
                        Text(scheduleText(for: questionnaireID))
                            .font(.caption2)
                    }
                    .foregroundColor(.secondary)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(PlainButtonStyle())
    }

    // MARK: - Computed Properties

    private var groupedQuestionnaires: [(category: DiseaseCategory, questionnaires: [QuestionnaireID])] {
        var result: [DiseaseCategory: [QuestionnaireID]] = [:]

        let questionnairesToShow = selectedCategory != nil
            ? QuestionnaireID.allCases.filter { $0.category == selectedCategory }
            : Array(QuestionnaireID.allCases)

        for id in questionnairesToShow {
            result[id.category, default: []].append(id)
        }

        return result.sorted { $0.key.rawValue < $1.key.rawValue }.map { category, questionnaires in
            (category: category, questionnaires: questionnaires.sorted { $0.rawValue < $1.rawValue })
        }
    }

    private var filteredQuestionnaires: [QuestionnaireID] {
        if searchText.isEmpty {
            return []
        }

        return QuestionnaireID.allCases.filter { id in
            let title = NSLocalizedString(id.titleKey, comment: "").lowercased()
            let description = NSLocalizedString(id.descriptionKey, comment: "").lowercased()
            let search = searchText.lowercased()
            return title.contains(search) || description.contains(search) || id.rawValue.contains(search)
        }
    }

    // MARK: - Helper Methods

    private func scheduleIcon(for id: QuestionnaireID) -> String {
        let schedule = preferences.getSchedule(for: id)
        switch schedule.frequency {
        case .daily: return "calendar"
        case .weekly: return "calendar.badge.clock"
        case .monthly: return "calendar.circle"
        case .onDemand: return "hand.tap"
        }
    }

    private func scheduleText(for id: QuestionnaireID) -> String {
        let schedule = preferences.getSchedule(for: id)
        switch schedule.frequency {
        case .daily: return "Daily"
        case .weekly: return "Weekly"
        case .monthly: return "Monthly"
        case .onDemand: return "On-Demand"
        }
    }
}

// MARK: - Preview

struct QuestionnaireSettingsView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            QuestionnaireSettingsView()
        }
    }
}
