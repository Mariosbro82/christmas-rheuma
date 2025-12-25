//
//  TraeInsightsDashboardView.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct TraeInsightsDashboardView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: TraeSpacing.lg) {
                    symptomOverviewSection
                    mobilitySection
                    medicationSection
                    insightsSection
                }
                .padding()
            }
            .navigationTitle("Insights")
            .background(TraePalette.snow.ignoresSafeArea())
        }
    }
    
    private var symptomOverviewSection: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            Text(String(localized: "insights.symptom.title"))
                .font(TraeTypography.title2)
            Text(String(localized: "insights.symptom.subtitle"))
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            symptomStats
        }
    }
    
    private var mobilitySection: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            Text(String(localized: "insights.mobility.title"))
                .font(TraeTypography.title2)
            Text(String(localized: "insights.mobility.subtitle"))
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            mobilityStats
            HStack {
                Label(String(localized: "insights.mobility.hint"), systemImage: "figure.cooldown")
                    .foregroundStyle(TraePalette.forestGreen)
            }
            .font(TraeTypography.subheadline)
        }
    }
    
    private var medicationSection: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            Text(String(localized: "insights.medication.title"))
                .font(TraeTypography.title2)
            Text(String(localized: "insights.medication.subtitle"))
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            if environment.notifications.isEmpty {
                Text(String(localized: "insights.medication.empty"))
                    .font(TraeTypography.body)
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: TraeSpacing.sm) {
                    ForEach(environment.notifications) { note in
                        VStack(alignment: .leading) {
                            Text(note.title)
                                .font(TraeTypography.headline)
                            Text(note.subtitle)
                                .font(TraeTypography.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(RoundedRectangle(cornerRadius: 18).fill(Color(.secondarySystemBackground)))
                    }
                }
            }
        }
    }
    
    private var insightsSection: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            Text(String(localized: "insights.section.title"))
                .font(TraeTypography.title2)
            Text(String(localized: "insights.section.subtitle"))
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            ForEach(environment.analytics.insights) { insight in
                VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                    Text(insight.title)
                        .font(TraeTypography.headline)
                    Text(insight.detail)
                        .font(TraeTypography.body)
                        .foregroundStyle(.secondary)
                    Text(insight.action)
                        .font(TraeTypography.footnote)
                        .foregroundStyle(TraePalette.traeOrange)
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 20).fill(Color(.secondarySystemBackground)))
            }
        }
    }

    private var symptomStats: some View {
        let entries = environment.symptomEntries
        return VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            if entries.isEmpty {
                Text("Log your first check-in to see averages here.")
                    .font(TraeTypography.body)
                    .foregroundStyle(.secondary)
            } else {
                metricRow(label: "Average pain", value: entries.averagePainFormatted())
                metricRow(label: "Average stiffness", value: "\(entries.averageStiffnessFormatted()) min")
                metricRow(label: "Average fatigue", value: entries.averageFatigueFormatted())
            }
        }
    }
    
    private var mobilityStats: some View {
        let entries = environment.symptomEntries
        let completed = entries.filter { $0.mobilityCompleted }.count
        let rate = entries.isEmpty ? 0 : Int(Double(completed) / Double(entries.count) * 100.0)
        let completionValue = String(format: NSLocalizedString("insights.metric.mobility_completion.value", comment: ""), rate)
        let noteValue: String
        if let note = entries.first?.notes, !note.isEmpty {
            noteValue = note
        } else {
            noteValue = String(localized: "insights.metric.no_notes")
        }
        return VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            metricRow(label: String(localized: "insights.metric.mobility_completion"), value: completionValue)
            metricRow(label: String(localized: "insights.metric.recent_note"), value: noteValue)
        }
    }
    
    private func metricRow(label: String, value: String) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(label)
                    .font(TraeTypography.subheadline)
                Text(value)
                    .font(TraeTypography.headline)
            }
            Spacer()
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 18).fill(Color(.secondarySystemBackground)))
    }
}

private extension Array where Element == SymptomEntry {
    func averagePainFormatted() -> String {
        guard !isEmpty else { return "--" }
        let value = reduce(0) { $0 + $1.pain } / Double(count)
        return String(format: "%.1f", value)
    }
    
    func averageStiffnessFormatted() -> Int {
        guard !isEmpty else { return 0 }
        let value = Double(reduce(0) { $0 + $1.stiffnessMinutes }) / Double(count)
        return Int(value.rounded())
    }
    
    func averageFatigueFormatted() -> String {
        guard !isEmpty else { return "--" }
        let value = reduce(0) { $0 + $1.fatigue } / Double(count)
        return String(format: "%.1f", value)
    }
}
