//
//  TraePantryView.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct TraePantryView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    @State private var selectedItem: PantryItem?
    @State private var shoppingList: [ShoppingListEntry] = []
    @State private var showScanner = false
    
    var body: some View {
        NavigationSplitView {
            List(selection: $selectedItem) {
                Section("Pantry inventory") {
                    ForEach(environment.pantry.pantryItems) { item in
                        TraePantryRow(item: item)
                            .tag(item)
                    }
                }
                
                Section("Shopping list") {
                    ForEach(shoppingList) { entry in
                        TraeShoppingEntryRow(entry: entry)
                    }
                    .onDelete(perform: deleteShoppingItem)
                }
            }
            .navigationTitle("Pantry & Shopping")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showScanner.toggle()
                    } label: {
                        Image(systemName: "barcode.viewfinder")
                    }
                    .accessibilityLabel("Scan barcode")
                }
            }
            .sheet(isPresented: $showScanner) {
                TraeBarcodeScannerView { barcode in
                    handleScan(barcode: barcode)
                }
            }
        } detail: {
            if let item = selectedItem {
                TraePantryDetail(item: item) { entry in
                    shoppingList.append(entry)
                }
            } else {
                TraePantryEmptyState()
            }
        }
        .task {
            selectedItem = environment.pantry.pantryItems.first
            if shoppingList.isEmpty {
                shoppingList = environment.pantry.pantryItems
                    .filter { $0.stockStatus != .inStock }
                    .map { item in
                        ShoppingListEntry(pantryItem: item, isCompleted: false, householdAssignee: environment.mealPlans.plans.first?.collaborators.randomElement())
                    }
            }
        }
    }
    
    private func deleteShoppingItem(at offsets: IndexSet) {
        shoppingList.remove(atOffsets: offsets)
    }
    
    private func handleScan(barcode: String) {
        if let match = environment.pantry.pantryItems.first(where: { $0.barcode == barcode }) {
            selectedItem = match
        } else {
            let newItem = PantryItem(name: "New Ingredient", category: "Uncategorized", quantity: 1, unit: "unit", preferredBrand: nil, barcode: barcode, allergens: [], stockStatus: .inStock, updatedAt: Date())
            environment.pantry.update(item: newItem)
            selectedItem = newItem
        }
    }
}

struct TraePantryRow: View {
    let item: PantryItem
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                Text(item.name)
                    .font(TraeTypography.headline)
                Text("\(item.quantity.formatted()) \(item.unit) • \(item.category)")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            statusBadge
        }
        .padding(.vertical, TraeSpacing.xs)
    }
    
    private var statusBadge: some View {
        switch item.stockStatus {
        case .inStock:
            return Label("Ready", systemImage: "checkmark.circle.fill")
                .labelStyle(.iconOnly)
                .foregroundStyle(TraePalette.forestGreen)
        case .runningLow:
            return Label("Low", systemImage: "exclamationmark.circle")
                .labelStyle(.iconOnly)
                .foregroundStyle(TraePalette.saffron)
        case .outOfStock:
            return Label("Out", systemImage: "xmark.circle")
                .labelStyle(.iconOnly)
                .foregroundStyle(TraePalette.danger)
        case .expiringSoon:
            return Label("Expiring", systemImage: "timer")
                .labelStyle(.iconOnly)
                .foregroundStyle(TraePalette.warning)
        }
    }
}

struct TraeShoppingEntryRow: View {
    var entry: ShoppingListEntry
    
    var body: some View {
        HStack {
            Image(systemName: entry.isCompleted ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(entry.isCompleted ? TraePalette.forestGreen : Color.secondary)
            VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                Text(entry.pantryItem.name)
                    .font(TraeTypography.body)
                if let assignee = entry.householdAssignee {
                    Text("Assigned to \(assignee)")
                        .font(TraeTypography.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
    }
}

struct TraePantryDetail: View {
    let item: PantryItem
    var addToShopping: (ShoppingListEntry) -> Void
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: TraeSpacing.lg) {
                Text(item.name)
                    .font(TraeTypography.title)
                Text("Category: \(item.category)")
                    .font(TraeTypography.body)
                    .foregroundStyle(.secondary)
                
                statusSection
                allergensSection
                suggestedRecipes
                householdSync
            }
            .padding()
        }
    }
    
    private var statusSection: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            Text("Stock status")
                .font(TraeTypography.headline)
            HStack {
                switch item.stockStatus {
                case .inStock:
                    Label("Plenty on hand", systemImage: "checkmark.circle")
                        .foregroundStyle(TraePalette.forestGreen)
                case .runningLow:
                    Label("Running low", systemImage: "exclamationmark.circle")
                        .foregroundStyle(TraePalette.saffron)
                case .outOfStock:
                    Label("Out of stock", systemImage: "xmark.circle")
                        .foregroundStyle(TraePalette.danger)
                case .expiringSoon(let date):
                    Label("Use soon (\(date.formatted(.dateTime.month().day())))", systemImage: "hourglass")
                        .foregroundStyle(TraePalette.warning)
                }
                Spacer()
                Button("Add to list") {
                    let entry = ShoppingListEntry(pantryItem: item, isCompleted: false, householdAssignee: nil)
                    addToShopping(entry)
                    TraeHaptics.shared.performGentleProgress()
                }
                .buttonStyle(.bordered)
            }
        }
    }
    
    private var allergensSection: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            Text("Allergens & alternatives")
                .font(TraeTypography.headline)
            if item.allergens.isEmpty {
                Text("No allergens flagged. Adaptations still available in cook mode.")
                    .font(TraeTypography.body)
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                    ForEach(item.allergens, id: \.self) { allergen in
                        HStack {
                            Image(systemName: "exclamationmark.shield")
                                .foregroundStyle(TraePalette.danger)
                            Text(allergen.rawValue.capitalized)
                        }
                    }
                    Text("Tap recipe steps for secure swaps and allergen-friendly plating.")
                        .font(TraeTypography.footnote)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
    
    private var suggestedRecipes: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            Text("Suggested recipes")
                .font(TraeTypography.headline)
            Text("Deterministic recommendations consider inventory, goals, and waste patterns.")
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            ForEach(TraeFixtures.sampleRecipes.prefix(2)) { recipe in
                HStack {
                    VStack(alignment: .leading) {
                        Text(recipe.title)
                            .font(TraeTypography.subheadline)
                        Text(recipe.subtitle)
                            .font(TraeTypography.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Cook") {
                        TraeHaptics.shared.performStageChange()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(TraePalette.traeOrange)
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 20).fill(Color(.systemGroupedBackground)))
            }
        }
    }
    
    private var householdSync: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            Text("Household sync")
                .font(TraeTypography.headline)
            Text("Updates replay across devices within seconds, preserving offline adjustments.")
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            TraeSyncStatusView()
        }
    }
}

struct TraePantryEmptyState: View {
    var body: some View {
        VStack(spacing: TraeSpacing.md) {
            Image(systemName: "tray.full")
                .font(.system(size: 48))
                .foregroundStyle(TraePalette.saffron)
            Text("Select an ingredient to see stock levels, allergen-safe swaps, and collaborative shopping tools.")
                .font(TraeTypography.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}

struct TraeBarcodeScannerView: View {
    var completion: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        VStack(spacing: TraeSpacing.lg) {
            Text("Barcode scanner")
                .font(TraeTypography.title2)
            Text("Simulated scan. In production this view would use AVFoundation to capture codes.")
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            
            Button("Simulate scan") {
                completion("1234567890123")
                dismiss()
            }
            .buttonStyle(.borderedProminent)
            .tint(TraePalette.traeOrange)
            
            Spacer()
        }
        .padding()
    }
}

struct TraeSyncStatusView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            switch environment.syncEngine.state {
            case .idle:
                Text("Idle • Offline edits queued.")
                    .font(TraeTypography.footnote)
            case .syncing(let progress):
                ProgressView(value: progress) {
                    Text("Syncing \(Int(progress * 100))%")
                }
                .progressViewStyle(.linear)
            case .success(let date):
                Text("Last synced: \(date.formatted(date: .numeric, time: .shortened))")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(TraePalette.forestGreen)
            case .failed(let error):
                Text("Sync failed: \(error)")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(TraePalette.danger)
            }
            Button("Replay updates") {
                environment.syncEngine.replayPendingOperations()
            }
            .buttonStyle(.bordered)
        }
    }
}
