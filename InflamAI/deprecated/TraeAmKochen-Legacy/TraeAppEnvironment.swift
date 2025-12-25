//
//  TraeAppEnvironment.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//

import Foundation
import Combine

/// Shared environment coordinating managers, offline-first state, and sample data.
final class TraeAppEnvironment: ObservableObject {
    @Published var profile: TraeProfile
    @Published var notifications: [TraeNotification]
    @Published var seasonalHighlights: [SeasonalHighlight]
    @Published var tutorialVideos: [TutorialVideo]
    @Published var articles: [Article]
    @Published var motherModeSettings: MotherModeSettings
    @Published var motherQuickEntries: [MotherQuickEntry]
    @Published var microRoutines: [ExerciseRoutine]
    @Published var symptomEntries: [SymptomEntry]
    
    let recipeLibrary: RecipeLibrary
    let pantry: PantryIntelligence
    let mealPlans: MealPlanCollaboration
    let analytics: TraeAnalyticsEngine
    let offlineCache: TraeOfflineCacheManager
    let securityCenter: TraeSecurityCenter
    let syncEngine: TraeSyncEngine
    
    init(dateProvider: @escaping () -> Date = { Date() }) {
        let sampleRecipes = TraeFixtures.sampleRecipes
        let samplePlan = TraeFixtures.sampleMealPlan(recipes: sampleRecipes)
        let samplePantry = TraeFixtures.samplePantry(dateProvider: dateProvider)
        let sampleProfile = TraeFixtures.sampleProfile
        
        self.recipeLibrary = RecipeLibrary(recipes: sampleRecipes)
        self.mealPlans = MealPlanCollaboration(plans: [samplePlan])
        self.pantry = PantryIntelligence(pantryItems: samplePantry)
        self.analytics = TraeAnalyticsEngine()
        self.offlineCache = TraeOfflineCacheManager()
        self.securityCenter = TraeSecurityCenter()
        self.syncEngine = TraeSyncEngine()
        
        self.profile = sampleProfile
        self.notifications = TraeFixtures.sampleNotifications
        self.seasonalHighlights = TraeFixtures.sampleSeasonalHighlights
        self.tutorialVideos = TraeFixtures.sampleVideos(recipes: sampleRecipes)
        self.articles = TraeFixtures.sampleArticles(recipes: sampleRecipes)
        self.motherModeSettings = .default
        self.motherQuickEntries = []
        self.microRoutines = TraeFixtures.sampleMicroRoutines
        self.symptomEntries = TraeFixtures.sampleSymptomEntries(dateProvider)
        
        analytics.updateTrending(with: sampleRecipes)
        analytics.refreshInsights(for: profile, pantry: samplePantry, plans: [samplePlan])
    }
    
    func cacheOfflineAssets() {
        offlineCache.cache(recipeLibrary.recipes, for: "recipes", type: .recipeLibrary)
        offlineCache.cache(tutorialVideos, for: "videos", type: .tutorialVideos)
        offlineCache.cache(TraeFixtures.timerPresets, for: "timers", type: .cookTimers)
        offlineCache.cache(TraeFixtures.conversionTable, for: "conversions", type: .conversions)
    }
    
    func logQuickEntry(_ entry: MotherQuickEntry) {
        motherQuickEntries.insert(entry, at: 0)
    }
    
    func logSymptomEntry(_ entry: SymptomEntry) {
        symptomEntries.insert(entry, at: 0)
    }
}

// MARK: - Fixtures

enum TraeFixtures {
    static let sampleProfile = TraeProfile(name: "Fabian",
                                           householdSize: 3,
                                           dietaryPreferences: [.omnivore],
                                           allergens: [.nuts],
                                           prefersMetric: true,
                                           lowMotion: false,
                                           notificationsEnabled: true,
                                           preferredLanguage: "en")
    
    static let timerPresets: [String: TimeInterval] = [
        "Simmer": 1800,
        "Proof Dough": 7200,
        "Rest Steak": 480
    ]
    
    static let conversionTable: [String: Double] = [
        "cupToMl": 240,
        "ounceToGram": 28.35,
        "teaspoonToMl": 5
    ]
    
    static let sampleSeasonalHighlights: [SeasonalHighlight] = []
    
    static let sampleNotifications: [TraeNotification] = [
        TraeNotification(title: "Morning stiffness check",
                         subtitle: "Log pain, stiffness, and fatigue in under a minute",
                         type: .pantry,
                         deliveryDate: Date(),
                         isRead: false),
        TraeNotification(title: "Posture reset",
                         subtitle: "Schedule a 2-minute thoracic opener between meetings",
                         type: .insights,
                         deliveryDate: Date().addingTimeInterval(-3600),
                         isRead: true),
        TraeNotification(title: "Biologic reminder",
                         subtitle: "Dose due tomorrow—review your plan",
                         type: .reminder,
                         deliveryDate: Date().addingTimeInterval(3600 * 12),
                         isRead: false)
    ]
    
    static func sampleMealPlan(recipes: [Recipe]) -> MealPlan {
        var breakfastRecipe = recipes.first!
        if let smoothie = recipes.first(where: { $0.categories.contains("Smoothie") }) {
            breakfastRecipe = smoothie
        }
        return MealPlan(title: "Weekend reset", owner: "Fabian", collaborators: ["Lina", "Noah"], entries: [
            MealPlan.Entry(date: Date(), mealType: .breakfast, recipe: breakfastRecipe),
            MealPlan.Entry(date: Date().addingTimeInterval(3600 * 6), mealType: .dinner, recipe: recipes.first!)
        ], comments: [
            MealPlanComment(user: "Lina",
                            message: "Can we swap dairy for oat crème?",
                            timestamp: Date().addingTimeInterval(-3600)),
            MealPlanComment(user: "Noah",
                            message: "I'll handle the shopping list.",
                            timestamp: Date().addingTimeInterval(-7200))
        ])
    }
    
    static func samplePantry(dateProvider: () -> Date) -> [PantryItem] {
        [
            PantryItem(name: "Olive Oil", category: "Essentials", quantity: 0.5, unit: "L",
                       preferredBrand: "Trae Reserve", barcode: "1234567890123",
                       allergens: [], stockStatus: .runningLow, updatedAt: dateProvider()),
            PantryItem(name: "Chickpeas", category: "Canned", quantity: 3, unit: "cans",
                       preferredBrand: nil, barcode: "2345678901234", allergens: [], stockStatus: .inStock, updatedAt: dateProvider()),
            PantryItem(name: "Almond Butter", category: "Spreads", quantity: 0.2, unit: "kg",
                       preferredBrand: "Nutty & Co", barcode: "3456789012345",
                       allergens: [.nuts], stockStatus: .expiringSoon(dateProvider().addingTimeInterval(86400 * 2)), updatedAt: dateProvider())
        ]
    }
    
    static var sampleRecipes: [Recipe] {
        let prepStage = Recipe.CookingStage(type: .prep,
                                            hapticStyle: .tactile,
                                            motionNarrative: "Guides gentle joint-friendly prep motions.")
        let simmerStage = Recipe.CookingStage(type: .simmer,
                                              hapticStyle: .simmer,
                                              motionNarrative: "Encourages slow stirring without wrist strain.")
        let plateStage = Recipe.CookingStage(type: .plate,
                                             hapticStyle: .plating,
                                             motionNarrative: "Focuses on ergonomic plating techniques.")
        
        let citrusGlazedSalmon = Recipe(
            title: "Citrus-Glazed Salmon",
            subtitle: "Omega-rich dinner with anti-inflammatory citrus notes",
            author: "Chef Trae Collective",
            heroVideo: "citrus-salmon-demo",
            heroImageName: "citrusSalmonHero",
            categories: ["Dinner", "Anti-Inflammatory", "Omega-3"],
            difficulty: .moderate,
            duration: TimeInterval(25 * 60),
            servings: 2,
            ingredients: [
                .init(name: "Salmon fillets", quantity: 2, unit: "pieces", preparation: "skin-on", allergens: [.fish]),
                .init(name: "Orange juice", quantity: 120, unit: "ml", preparation: "freshly squeezed", allergens: []),
                .init(name: "Honey", quantity: 2, unit: "tbsp", preparation: nil, allergens: []),
                .init(name: "Fresh ginger", quantity: 1, unit: "tbsp", preparation: "grated", allergens: []),
                .init(name: "Baby spinach", quantity: 120, unit: "g", preparation: "rinsed", allergens: [])
            ],
            steps: [
                .init(title: "Whisk glaze",
                      subtitle: "Combine citrus juice, honey, and ginger.",
                      duration: TimeInterval(5 * 60),
                      stage: prepStage),
                .init(title: "Sear salmon",
                      subtitle: "Sear skin-side down in a non-stick skillet.",
                      duration: TimeInterval(8 * 60),
                      stage: simmerStage),
                .init(title: "Finish and plate",
                      subtitle: "Glaze salmon and serve over spinach.",
                      duration: TimeInterval(6 * 60),
                      stage: plateStage)
            ],
            nutrition: .init(calories: 520, protein: 38, carbs: 22, fats: 28, fiber: 4, sugars: 16, sodium: 420),
            seasonalTag: .spring,
            isPinned: true,
            isDownloaded: true
        )
        
        let turmericLentilSoup = Recipe(
            title: "Golden Turmeric Lentil Soup",
            subtitle: "Comforting batch-friendly soup with pantry staples",
            author: "Trae Test Kitchen",
            heroVideo: nil,
            heroImageName: "turmericSoupHero",
            categories: ["Lunch", "Batch Cooking", "Vegetarian"],
            difficulty: .easy,
            duration: TimeInterval(35 * 60),
            servings: 4,
            ingredients: [
                .init(name: "Red lentils", quantity: 180, unit: "g", preparation: "rinsed", allergens: []),
                .init(name: "Vegetable broth", quantity: 1.2, unit: "L", preparation: nil, allergens: []),
                .init(name: "Turmeric powder", quantity: 2, unit: "tsp", preparation: nil, allergens: []),
                .init(name: "Coconut milk", quantity: 160, unit: "ml", preparation: "unsweetened", allergens: []),
                .init(name: "Baby kale", quantity: 90, unit: "g", preparation: "roughly chopped", allergens: [])
            ],
            steps: [
                .init(title: "Bloom aromatics",
                      subtitle: "Sauté turmeric with aromatics for deeper flavor.",
                      duration: TimeInterval(6 * 60),
                      stage: prepStage),
                .init(title: "Simmer lentils",
                      subtitle: "Add broth and lentils; simmer until tender.",
                      duration: TimeInterval(18 * 60),
                      stage: simmerStage),
                .init(title: "Finish with coconut milk",
                      subtitle: "Stir in coconut milk and kale before serving.",
                      duration: TimeInterval(4 * 60),
                      stage: plateStage)
            ],
            nutrition: .init(calories: 340, protein: 18, carbs: 38, fats: 12, fiber: 9, sugars: 7, sodium: 320),
            seasonalTag: .winter,
            isPinned: false,
            isDownloaded: false
        )
        
        let sunriseSmoothie = Recipe(
            title: "Sunrise Ginger Smoothie",
            subtitle: "Quick morning blend for gentle energy",
            author: "Mother Mode Daily",
            heroVideo: "sunrise-smoothie",
            heroImageName: "sunriseSmoothieHero",
            categories: ["Breakfast", "Smoothie", "Mother Mode"],
            difficulty: .easy,
            duration: TimeInterval(6 * 60),
            servings: 1,
            ingredients: [
                .init(name: "Frozen mango", quantity: 150, unit: "g", preparation: "chunks", allergens: []),
                .init(name: "Greek yogurt", quantity: 120, unit: "g", preparation: "plain", allergens: [.dairy]),
                .init(name: "Fresh ginger", quantity: 0.5, unit: "tsp", preparation: "grated", allergens: []),
                .init(name: "Ground flaxseed", quantity: 1, unit: "tbsp", preparation: nil, allergens: []),
                .init(name: "Chilled water", quantity: 120, unit: "ml", preparation: nil, allergens: [])
            ],
            steps: [
                .init(title: "Layer blender",
                      subtitle: "Add liquids first, then frozen fruit and boosters.",
                      duration: TimeInterval(2 * 60),
                      stage: prepStage),
                .init(title: "Blend until smooth",
                      subtitle: "Blend for 45 seconds, pausing if needed.",
                      duration: TimeInterval(2 * 60),
                      stage: simmerStage),
                .init(title: "Serve immediately",
                      subtitle: "Pour into a chilled glass and garnish.",
                      duration: TimeInterval(1 * 60),
                      stage: plateStage)
            ],
            nutrition: .init(calories: 290, protein: 14, carbs: 42, fats: 8, fiber: 5, sugars: 32, sodium: 120),
            seasonalTag: .summer,
            isPinned: false,
            isDownloaded: true
        )
        
        return [citrusGlazedSalmon, turmericLentilSoup, sunriseSmoothie]
    }
    
    static func sampleVideos(recipes: [Recipe]) -> [TutorialVideo] {
        recipes.enumerated().map { index, recipe in
            TutorialVideo(title: "\(recipe.title) Masterclass",
                          duration: TimeInterval(10 * 60 + index * 120),
                          isDownloaded: recipe.isDownloaded,
                          associatedRecipe: recipe,
                          downloadProgress: recipe.isDownloaded ? 1.0 : 0.35)
        }
    }
    
    static func sampleArticles(recipes: [Recipe]) -> [Article] {
        [
            Article(title: "Build a Fermentation Pantry",
                    author: "Trae Editorial",
                    publishDate: Date().addingTimeInterval(-86400 * 2),
                    readingTime: 6,
                    summary: "From koji to charred citrus, learn how to layer flavor and keep your pantry balanced.",
                    relatedRecipes: recipes),
            Article(title: "Low Waste Meal Planning",
                    author: "Chef Evelyn",
                    publishDate: Date().addingTimeInterval(-86400 * 7),
                    readingTime: 8,
                    summary: "Use adaptive shopping lists and dynamic servings to reduce waste by 30% weekly.",
                    relatedRecipes: recipes)
        ]
    }
    
    static var sampleMicroRoutines: [ExerciseRoutine] {
        [
            ExerciseRoutine.morningMobility(),
            ExerciseRoutine.deskUnwind()
        ]
    }
    
    static func sampleSymptomEntries(_ dateProvider: () -> Date) -> [SymptomEntry] {
        let now = dateProvider()
        return (0..<5).map { offset in
            SymptomEntry(
                id: UUID(),
                loggedAt: Calendar.current.date(byAdding: .day, value: -offset, to: now) ?? now,
                pain: Double.random(in: 3...6),
                stiffnessMinutes: Int.random(in: 20...45),
                fatigue: Double.random(in: 4...7),
                sleepQuality: Double.random(in: 5...8),
                mobilityCompleted: Bool.random(),
                notes: ""
            )
        }
    }
}
