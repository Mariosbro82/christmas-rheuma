//
//  AssessmentsView.swift
//  InflamAI-Swift
//
//  Complete assessments/questionnaires browser
//  Access all 40+ questionnaires directly without needing to enable them
//

import SwiftUI
import CoreData

// NOTE: String extensions (displayName, localizedWithFallback) moved to
// InflamAI/Extensions/StringExtensions.swift for project-wide use

struct AssessmentsView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @State private var searchText = ""
    @State private var selectedCategory: DiseaseCategory?

    var body: some View {
        listContent
            .searchable(text: $searchText, prompt: "Search assessments")
            .navigationTitle("Assessments")
            .navigationBarTitleDisplayMode(.large)
    }

    private var listContent: some View {
        List {
            ForEach(filteredCategories, id: \.self) { category in
                categorySection(for: category)
            }
        }
    }

    private func categorySection(for category: DiseaseCategory) -> some View {
        Section(header: Text(category.rawValue)) {
            ForEach(questionnairesForCategory(category), id: \.self) { questionnaireID in
                NavigationLink {
                    QuestionnaireFormView(questionnaireID: questionnaireID)
                        .environment(\.managedObjectContext, viewContext)
                } label: {
                    QuestionnaireRow(questionnaireID: questionnaireID)
                }
            }
        }
    }

    private var filteredCategories: [DiseaseCategory] {
        DiseaseCategory.allCases.filter { category in
            !questionnairesForCategory(category).isEmpty
        }
    }

    private func questionnairesForCategory(_ category: DiseaseCategory) -> [QuestionnaireID] {
        let filtered = QuestionnaireID.allCases.filter { $0.category == category }

        if searchText.isEmpty {
            return filtered
        } else {
            return filtered.filter { questionnaireID in
                // CRIT-002 FIX: Use localizedWithFallback for search
                let title = questionnaireID.titleKey.localizedWithFallback
                let description = questionnaireID.descriptionKey.localizedWithFallback
                return title.localizedCaseInsensitiveContains(searchText) ||
                       description.localizedCaseInsensitiveContains(searchText)
            }
        }
    }
}

struct QuestionnaireRow: View {
    let questionnaireID: QuestionnaireID
    @Environment(\.managedObjectContext) private var viewContext

    var body: some View {
        HStack(spacing: Spacing.md) {
            // Icon
            Circle()
                .fill(iconColor.opacity(0.15))
                .frame(width: 44, height: 44)
                .overlay(
                    Image(systemName: iconName)
                        .font(.system(size: Typography.lg, weight: .semibold))
                        .foregroundColor(iconColor)
                )

            // Title and description
            // CRIT-002 FIX: Use localizedWithFallback to prevent raw keys from showing
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(questionnaireID.titleKey.localizedWithFallback)
                    .font(.system(size: Typography.base, weight: .medium))
                    .foregroundColor(Colors.Gray.g900)

                Text(questionnaireID.descriptionKey.localizedWithFallback)
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)
                    .lineLimit(2)

                // Show last completed date if available
                if let lastDate = lastCompletedDate {
                    HStack(spacing: Spacing.xxs) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: Typography.xxs))
                            .foregroundColor(Colors.Semantic.success)
                        Text("Last: \(lastDate, style: .relative)")
                            .font(.system(size: Typography.xxs))
                            .foregroundColor(Colors.Gray.g500)
                    }
                    .padding(.top, Spacing.xxs)
                }
            }

            Spacer()
        }
        .padding(.vertical, Spacing.xxs)
    }

    private var iconColor: Color {
        switch questionnaireID.category {
        case .axialSpa: return Colors.Primary.p500
        case .rheumatoidArthritis: return Colors.Accent.purple
        case .psoriaticArthritis: return .pink
        case .lupus: return .orange
        case .sjogrens: return .cyan
        case .scleroderma: return .indigo
        case .vasculitis: return Colors.Semantic.error
        case .gout: return Colors.Semantic.warning
        case .osteoarthritis: return .brown
        case .fibromyalgia: return .mint
        case .myopathies: return Colors.Accent.teal
        case .pediatric: return Colors.Semantic.success
        case .generic: return Colors.Gray.g500
        }
    }

    private var iconName: String {
        switch questionnaireID.category {
        case .axialSpa: return "figure.walk"
        case .rheumatoidArthritis: return "hand.raised.fill"
        case .psoriaticArthritis: return "sparkles"
        case .lupus: return "sun.max.fill"
        case .sjogrens: return "drop.fill"
        case .scleroderma: return "hand.point.up.left.fill"
        case .vasculitis: return "heart.text.square.fill"
        case .gout: return "waveform.path.ecg"
        case .osteoarthritis: return "figure.stand"
        case .fibromyalgia: return "bolt.fill"
        case .myopathies: return "figure.strengthtraining.traditional"
        case .pediatric: return "figure.and.child.holdinghands"
        case .generic: return "list.clipboard.fill"
        }
    }

    private var lastCompletedDate: Date? {
        let request = QuestionnaireResponse.fetchRequest()
        request.predicate = NSPredicate(format: "questionnaireID == %@", questionnaireID.rawValue)
        request.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]
        request.fetchLimit = 1

        guard let results = try? viewContext.fetch(request),
              let latest = results.first else {
            return nil
        }

        return latest.createdAt
    }
}

// Preview
struct AssessmentsView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            AssessmentsView()
                .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
        }
    }
}
