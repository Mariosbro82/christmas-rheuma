//
//  TriggerSelectorView.swift
//  InflamAI
//
//  UI for selecting potential flare triggers
//

import SwiftUI

struct TriggerSelectorView: View {
    @Binding var selectedTriggers: Set<FlareTrigger>
    @State private var searchText = ""
    @State private var selectedCategory: FlareTriggerCategory? = nil
    @Environment(\.dismiss) private var dismiss

    var filteredTriggers: [FlareTrigger] {
        let triggers = selectedCategory?.triggers ?? FlareTrigger.allCases

        if searchText.isEmpty {
            return triggers
        } else {
            return triggers.filter { $0.displayName.localizedCaseInsensitiveContains(searchText) }
        }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    TextField("Search triggers...", text: $searchText)
                    if !searchText.isEmpty {
                        Button {
                            searchText = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(12)
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .padding()

                // Category filter
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        // All categories button
                        TriggerCategoryChip(
                            title: "All",
                            icon: "square.grid.2x2.fill",
                            color: .blue,
                            isSelected: selectedCategory == nil
                        ) {
                            selectedCategory = nil
                        }

                        ForEach(FlareTriggerCategory.allCases) { category in
                            TriggerCategoryChip(
                                title: category.rawValue,
                                icon: category.icon,
                                color: category.color,
                                isSelected: selectedCategory == category
                            ) {
                                if selectedCategory == category {
                                    selectedCategory = nil
                                } else {
                                    selectedCategory = category
                                }
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.bottom, 8)

                // Selected count
                if !selectedTriggers.isEmpty {
                    HStack {
                        Text("\(selectedTriggers.count) trigger\(selectedTriggers.count == 1 ? "" : "s") selected")
                            .font(.subheadline)
                            .foregroundColor(.secondary)

                        Spacer()

                        Button("Clear All") {
                            selectedTriggers.removeAll()
                        }
                        .font(.subheadline)
                        .foregroundColor(.red)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                }

                // Triggers list
                ScrollView {
                    LazyVStack(spacing: 0) {
                        if selectedCategory == nil && searchText.isEmpty {
                            // Show by category
                            ForEach(FlareTriggerCategory.allCases) { category in
                                if !category.triggers.isEmpty {
                                    Section {
                                        ForEach(category.triggers) { trigger in
                                            TriggerSelectorRow(
                                                trigger: trigger,
                                                isSelected: selectedTriggers.contains(trigger)
                                            ) {
                                                toggleTrigger(trigger)
                                            }
                                        }
                                    } header: {
                                        HStack {
                                            Image(systemName: category.icon)
                                                .foregroundColor(category.color)
                                                .frame(width: 24)

                                            Text(category.rawValue)
                                                .font(.headline)
                                                .foregroundColor(.primary)

                                            Spacer()
                                        }
                                        .padding(.horizontal)
                                        .padding(.vertical, 8)
                                        .background(Color(.systemGray6))
                                    }
                                }
                            }
                        } else {
                            // Show filtered results
                            ForEach(filteredTriggers) { trigger in
                                TriggerSelectorRow(
                                    trigger: trigger,
                                    isSelected: selectedTriggers.contains(trigger)
                                ) {
                                    toggleTrigger(trigger)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Select Triggers")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }

    private func toggleTrigger(_ trigger: FlareTrigger) {
        if selectedTriggers.contains(trigger) {
            selectedTriggers.remove(trigger)
        } else {
            selectedTriggers.insert(trigger)
        }
    }
}

// MARK: - Supporting Views

struct TriggerCategoryChip: View {
    let title: String
    let icon: String
    let color: Color
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.caption)
                Text(title)
                    .font(.subheadline)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(isSelected ? color : Color(.systemGray6))
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(20)
        }
    }
}

struct TriggerSelectorRow: View {
    let trigger: FlareTrigger
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                // Icon
                Image(systemName: trigger.icon)
                    .font(.title3)
                    .foregroundColor(trigger.category.color)
                    .frame(width: 32)

                // Name
                Text(trigger.displayName)
                    .font(.body)
                    .foregroundColor(.primary)

                Spacer()

                // Checkmark
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(.blue)
                } else {
                    Image(systemName: "circle")
                        .font(.title3)
                        .foregroundColor(.gray.opacity(0.3))
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 12)
            .background(isSelected ? Color.blue.opacity(0.05) : Color.clear)
        }
    }
}

// MARK: - Compact Selection View (for inline use)

struct CompactTriggerSelector: View {
    @Binding var selectedTriggers: Set<FlareTrigger>
    @State private var showingFullSelector = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Possible Triggers", systemImage: "exclamationmark.triangle.fill")
                    .font(.headline)

                Spacer()

                Button {
                    showingFullSelector = true
                } label: {
                    HStack(spacing: 4) {
                        Text(selectedTriggers.isEmpty ? "Add" : "Edit")
                        Image(systemName: "chevron.right")
                    }
                    .font(.subheadline)
                    .foregroundColor(.blue)
                }
            }

            if selectedTriggers.isEmpty {
                Text("Tap to select triggers that may have caused this flare")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                FlowLayout(spacing: 8) {
                    ForEach(Array(selectedTriggers), id: \.self) { trigger in
                        TriggerChip(trigger: trigger) {
                            selectedTriggers.remove(trigger)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .sheet(isPresented: $showingFullSelector) {
            TriggerSelectorView(selectedTriggers: $selectedTriggers)
        }
    }
}

struct TriggerChip: View {
    let trigger: FlareTrigger
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: trigger.icon)
                .font(.caption2)

            Text(trigger.displayName)
                .font(.caption)

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.caption)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(trigger.category.color.opacity(0.2))
        .foregroundColor(trigger.category.color)
        .cornerRadius(16)
    }
}

// MARK: - Flow Layout for wrapping chips

struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = FlowResult(
            in: proposal.replacingUnspecifiedDimensions().width,
            subviews: subviews,
            spacing: spacing
        )
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = FlowResult(
            in: bounds.width,
            subviews: subviews,
            spacing: spacing
        )
        for (index, subview) in subviews.enumerated() {
            subview.place(at: CGPoint(x: bounds.minX + result.positions[index].x, y: bounds.minY + result.positions[index].y), proposal: .unspecified)
        }
    }

    struct FlowResult {
        var size: CGSize
        var positions: [CGPoint]

        init(in maxWidth: CGFloat, subviews: Subviews, spacing: CGFloat) {
            var positions: [CGPoint] = []
            var size: CGSize = .zero
            var currentX: CGFloat = 0
            var currentY: CGFloat = 0
            var lineHeight: CGFloat = 0

            for subview in subviews {
                let subviewSize = subview.sizeThatFits(.unspecified)

                if currentX + subviewSize.width > maxWidth && currentX > 0 {
                    currentX = 0
                    currentY += lineHeight + spacing
                    lineHeight = 0
                }

                positions.append(CGPoint(x: currentX, y: currentY))
                lineHeight = max(lineHeight, subviewSize.height)
                currentX += subviewSize.width + spacing
                size.width = max(size.width, currentX - spacing)
            }

            size.height = currentY + lineHeight
            self.size = size
            self.positions = positions
        }
    }
}

// MARK: - Preview

#Preview {
    CompactTriggerSelector(selectedTriggers: .constant([.coldWeather, .poorSleep, .missedMedication]))
}
