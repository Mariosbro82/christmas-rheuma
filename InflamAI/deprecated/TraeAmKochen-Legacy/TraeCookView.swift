//
//  TraeCookView.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct TraeCookView: View {
    enum Section: String, CaseIterable, Identifiable {
        case library = "Recipe Library"
        case cookMode = "Cook Mode"
        case portions = "Precision Portions"
        case media = "Media Gallery"
        
        var id: String { rawValue }
    }
    
    @EnvironmentObject private var environment: TraeAppEnvironment
    @State private var section = Section.library
    @State private var selectedRecipe: Recipe?
    
    var body: some View {
        NavigationStack {
            VStack {
                Picker("Section", selection: $section) {
                    ForEach(Section.allCases) { section in
                        Text(section.rawValue).tag(section)
                    }
                }
                .pickerStyle(.segmented)
                .padding()
                
                switch section {
                case .library:
                    TraeRecipeLibraryView(selectedRecipe: $selectedRecipe)
                case .cookMode:
                    TraeCookModeView(selectedRecipe: environment.recipeLibrary.recipes.first ?? TraeFixtures.sampleRecipes.first!)
                case .portions:
                    TraePrecisionPortionsView(recipe: environment.recipeLibrary.recipes.first ?? TraeFixtures.sampleRecipes.first!)
                case .media:
                    TraeMediaGalleryView()
                }
            }
            .navigationTitle("Cook smarter")
        }
    }
}

// MARK: - Recipe Library

struct TraeRecipeLibraryView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    @Binding var selectedRecipe: Recipe?
    
    var body: some View {
        VStack(spacing: 0) {
            filterBar
            ScrollViewReader { proxy in
                List {
                    ForEach(environment.recipeLibrary.filtered) { recipe in
                        TraeRecipeRow(recipe: recipe)
                            .onTapGesture {
                                selectedRecipe = recipe
                                TraeHaptics.shared.performGentleProgress()
                            }
                    }
                }
                .listStyle(.plain)
                .onChange(of: environment.recipeLibrary.filtered) { _ in
                    if let first = environment.recipeLibrary.filtered.first {
                        proxy.scrollTo(first.id, anchor: .top)
                    }
                }
            }
        }
        .sheet(item: $selectedRecipe) { recipe in
            TraeCookModeView(selectedRecipe: recipe)
        }
        .onAppear {
            environment.recipeLibrary.applyFilters()
        }
    }
    
    private var filterBar: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            HStack(spacing: TraeSpacing.sm) {
                Image(systemName: "magnifyingglass")
                TextField("Search or say “show summer grills”", text: Binding(
                    get: { environment.recipeLibrary.searchTerm },
                    set: {
                        environment.recipeLibrary.searchTerm = $0
                        environment.recipeLibrary.applyFilters()
                    }
                ))
                .textFieldStyle(.roundedBorder)
            }
            .padding(.horizontal)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: TraeSpacing.sm) {
                    ForEach(allCategories, id: \.self) { category in
                        let isSelected = environment.recipeLibrary.selectedCategories.contains(category)
                        Text(category)
                            .font(TraeTypography.subheadline)
                            .padding(.vertical, TraeSpacing.xs)
                            .padding(.horizontal, TraeSpacing.sm)
                            .background(isSelected ? TraePalette.forestGreen : Color(.systemGray6))
                            .foregroundStyle(isSelected ? Color.white : Color.primary)
                            .clipShape(Capsule())
                            .onTapGesture {
                                toggle(category)
                            }
                    }
                    Toggle("Pinned only", isOn: Binding(
                        get: { environment.recipeLibrary.showPinned },
                        set: {
                            environment.recipeLibrary.showPinned = $0
                            environment.recipeLibrary.applyFilters()
                        }
                    ))
                    .toggleStyle(.button)
                }
                .padding(.horizontal)
                .padding(.vertical, TraeSpacing.xs)
            }
        }
        .background(Color(.systemGroupedBackground))
    }
    
    private var allCategories: [String] {
        let categories = environment.recipeLibrary.recipes.flatMap { $0.categories }
        return Array(Set(categories)).sorted()
    }
    
    private func toggle(_ category: String) {
        if environment.recipeLibrary.selectedCategories.contains(category) {
            environment.recipeLibrary.selectedCategories.remove(category)
        } else {
            environment.recipeLibrary.selectedCategories.insert(category)
        }
        environment.recipeLibrary.applyFilters()
    }
}

struct TraeRecipeRow: View {
    let recipe: Recipe
    private let formatter = DateComponentsFormatter()
    
    init(recipe: Recipe) {
        self.recipe = recipe
        formatter.allowedUnits = [.minute]
        formatter.unitsStyle = .abbreviated
    }
    
    var body: some View {
        HStack(spacing: TraeSpacing.md) {
            RoundedRectangle(cornerRadius: 16)
                .fill(TraePalette.traeOrange.opacity(0.12))
                .frame(width: 64, height: 64)
                .overlay(
                    Image(systemName: recipe.isPinned ? "pin.fill" : "leaf")
                        .foregroundStyle(recipe.isPinned ? TraePalette.saffron : TraePalette.forestGreen)
                )
            
            VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                Text(recipe.title)
                    .font(TraeTypography.headline)
                Text(recipe.subtitle)
                    .font(TraeTypography.body)
                    .foregroundStyle(.secondary)
                
                HStack(spacing: TraeSpacing.sm) {
                    Label(recipe.difficulty.rawValue, systemImage: "dial.medium.fill")
                        .font(TraeTypography.footnote)
                    Label(formatter.string(from: recipe.duration) ?? "45m", systemImage: "clock")
                        .font(TraeTypography.footnote)
                    if recipe.isDownloaded {
                        Label("Offline", systemImage: "arrow.down.circle")
                            .font(TraeTypography.footnote)
                            .foregroundStyle(TraePalette.forestGreen)
                    }
                }
            }
            Spacer()
        }
        .padding(.vertical, TraeSpacing.sm)
        .id(recipe.id)
    }
}

// MARK: - Cook Mode

struct TraeCookModeView: View {
    let selectedRecipe: Recipe
    @State private var currentStepIndex = 0
    @State private var timerRemaining: TimeInterval?
    @State private var timerActive = false
    
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.lg) {
            Text(selectedRecipe.title)
                .font(TraeTypography.title)
                .padding(.top, TraeSpacing.md)
            Text(selectedRecipe.subtitle)
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            
            TabView(selection: $currentStepIndex) {
                ForEach(Array(selectedRecipe.steps.enumerated()), id: \.element.id) { index, step in
                    TraeCookStepCard(step: step, isActive: index == currentStepIndex)
                        .padding()
                        .tag(index)
                        .onAppear {
                            triggerHaptics(for: step)
                        }
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .always))
            .frame(height: 280)
            
            if let remaining = timerRemaining, timerActive {
                TraeCookTimerView(remaining: remaining)
            }
            
            HStack {
                Button(action: previousStep) {
                    Label("Back", systemImage: "chevron.left")
                }
                .disabled(currentStepIndex == 0)
                
                Spacer()
                
                Button(action: toggleTimer) {
                    Label(timerActive ? "Pause Timer" : "Start Timer", systemImage: timerActive ? "pause.circle" : "timer")
                }
                
                Spacer()
                
                Button(action: nextStep) {
                    Label(currentStepIndex == selectedRecipe.steps.count - 1 ? "Finish" : "Next", systemImage: "chevron.right")
                }
            }
            .font(TraeTypography.subheadline)
            
            Spacer()
        }
        .padding()
        .onAppear {
            prepareTimer()
        }
        .onReceive(timer) { _ in
            guard timerActive, let remaining = timerRemaining else { return }
            if remaining <= 1 {
                timerActive = false
                timerRemaining = nil
                TraeHaptics.shared.performAlert()
            } else {
                timerRemaining = remaining - 1
            }
        }
    }
    
    private func prepareTimer() {
        timerRemaining = selectedRecipe.steps[currentStepIndex].duration
    }
    
    private func toggleTimer() {
        if timerActive {
            timerActive = false
        } else {
            if timerRemaining == nil {
                timerRemaining = selectedRecipe.steps[currentStepIndex].duration
            }
            timerActive = true
        }
    }
    
    private func triggerHaptics(for step: Recipe.Step) {
        switch step.stage.hapticStyle {
        case .alert:
            TraeHaptics.shared.performAlert()
        case .plating:
            TraeHaptics.shared.performStageChange()
        case .simmer:
            TraeHaptics.shared.performGentleProgress()
        case .tactile:
            TraeHaptics.shared.performStageChange()
        }
    }
    
    private func previousStep() {
        guard currentStepIndex > 0 else { return }
        currentStepIndex -= 1
        prepareTimer()
    }
    
    private func nextStep() {
        guard currentStepIndex < selectedRecipe.steps.count - 1 else {
            TraeHaptics.shared.performAlert()
            return
        }
        currentStepIndex += 1
        prepareTimer()
    }
}

struct TraeCookStepCard: View {
    let step: Recipe.Step
    let isActive: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            Text(step.title)
                .font(TraeTypography.headline)
                .foregroundStyle(TraePalette.graphite)
            Text(step.subtitle)
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            Label(step.stage.motionNarrative, systemImage: "sparkles")
                .font(TraeTypography.footnote)
                .foregroundStyle(TraePalette.saffron)
            Spacer()
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 24)
                .fill(isActive ? TraePalette.traeOrange.opacity(0.14) : Color(.systemBackground))
        )
    }
}

struct TraeCookTimerView: View {
    let remaining: TimeInterval
    private let formatter: DateComponentsFormatter = {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.zeroFormattingBehavior = .pad
        return formatter
    }()
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Stage timer")
                    .font(TraeTypography.subheadline)
                Text(formatter.string(from: remaining) ?? "00:00")
                    .font(.system(size: 42, weight: .bold, design: .monospaced))
            }
            Spacer()
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 20).fill(TraePalette.forestGreen.opacity(0.12)))
    }
}

// MARK: - Precision Portions

struct TraePrecisionPortionsView: View {
    let recipe: Recipe
    @State private var servings: Int
    
    init(recipe: Recipe) {
        self.recipe = recipe
        _servings = State(initialValue: recipe.servings)
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.lg) {
            Text("Precision portions")
                .font(TraeTypography.title2)
            Text("Slide to adjust portion and Trae recalculates conversions, allergens, and macros instantly.")
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            
            VStack {
                Slider(value: Binding(get: { Double(servings) },
                                      set: { servings = Int($0.rounded()) }),
                       in: 1...Double(max(1, recipe.servings * 2)),
                       step: 1)
                HStack {
                    Text("Servings: \(servings)")
                    Spacer()
                    Text("Reference: \(recipe.servings)")
                }
                .font(TraeTypography.subheadline)
            }
            
            VStack(alignment: .leading, spacing: TraeSpacing.sm) {
                ForEach(recipe.ingredients) { ingredient in
                    let scaledQuantity = ingredient.quantity * Double(servings) / Double(max(recipe.servings, 1))
                    HStack {
                        Text(ingredient.name)
                            .font(TraeTypography.body)
                        Spacer()
                        Text("\(scaledQuantity.formatted(.number.precision(.fractionLength(0...1)))) \(ingredient.unit)")
                            .font(TraeTypography.subheadline)
                    }
                    if !ingredient.allergens.isEmpty {
                        Text("Allergens: \(ingredient.allergens.map { $0.rawValue.capitalized }.joined(separator: ", "))")
                            .font(TraeTypography.footnote)
                            .foregroundStyle(TraePalette.danger)
                    }
                }
            }
            
            TraeNutritionSummary(nutrition: recipe.nutrition, servings: servings, baseServings: recipe.servings)
            
            Spacer()
        }
        .padding()
    }
}

struct TraeNutritionSummary: View {
    let nutrition: Recipe.Nutrition
    let servings: Int
    let baseServings: Int
    
    private var scale: Double {
        Double(servings) / Double(max(baseServings, 1))
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            Text("Nutrition per serving")
                .font(TraeTypography.subheadline)
            HStack {
                nutritionColumn(title: "Calories", value: "\(Int(Double(nutrition.calories) * scale)) kcal")
                Spacer()
                nutritionColumn(title: "Protein", value: "\(scaled(nutrition.protein)) g")
                Spacer()
                nutritionColumn(title: "Carbs", value: "\(scaled(nutrition.carbs)) g")
                Spacer()
                nutritionColumn(title: "Fats", value: "\(scaled(nutrition.fats)) g")
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 20).fill(TraePalette.traeOrange.opacity(0.12)))
        }
    }
    
    private func scaled(_ value: Double) -> String {
        (value * scale).formatted(.number.precision(.fractionLength(0...1)))
    }
    
    private func nutritionColumn(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: TraeSpacing.xs) {
            Text(title)
                .font(TraeTypography.footnote)
                .foregroundStyle(.secondary)
            Text(value)
                .font(TraeTypography.subheadline)
        }
    }
}

// MARK: - Media Gallery

struct TraeMediaGalleryView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: TraeSpacing.lg) {
                Text("Media queue")
                    .font(TraeTypography.title2)
                Text("Download videos for offline step-by-step coaching. Jump directly into associated recipe stages.")
                    .font(TraeTypography.body)
                    .foregroundStyle(.secondary)
                
                LazyVStack(spacing: TraeSpacing.md) {
                    ForEach(environment.tutorialVideos) { video in
                        TraeMediaCard(video: video)
                    }
                }
            }
            .padding()
        }
    }
}

struct TraeMediaCard: View {
    let video: TutorialVideo
    
    private let formatter: DateComponentsFormatter = {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.unitsStyle = .abbreviated
        formatter.zeroFormattingBehavior = .pad
        return formatter
    }()
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            HStack {
                Text(video.title)
                    .font(TraeTypography.headline)
                Spacer()
                Text(formatter.string(from: video.duration) ?? "10m")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(.secondary)
            }
            
            if let recipe = video.associatedRecipe {
                Label(recipe.title, systemImage: "fork.knife.circle")
                    .font(TraeTypography.subheadline)
                    .foregroundStyle(TraePalette.saffron)
            }
            
            ProgressView(value: video.downloadProgress)
                .progressViewStyle(.linear)
            
            HStack {
                Label(video.isDownloaded ? "Available offline" : "Downloading...", systemImage: video.isDownloaded ? "checkmark.circle" : "arrow.down.circle")
                    .foregroundStyle(video.isDownloaded ? TraePalette.forestGreen : TraePalette.saffron)
                Spacer()
                Button("AirPlay") {}
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 22).fill(.background))
        .overlay(RoundedRectangle(cornerRadius: 22).stroke(Color(.systemGray5), lineWidth: 1))
    }
}
