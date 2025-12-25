//
//  TraeModels.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//

import Foundation
import SwiftUI

struct Recipe: Identifiable, Hashable, Codable {
    struct Step: Identifiable, Hashable, Codable {
        let id = UUID()
        let title: String
        let subtitle: String
        let duration: TimeInterval
        let stage: CookingStage
    }
    
    struct CookingStage: Hashable, Codable {
        enum StageType: String, Codable {
            case prep, simmer, bake, plate, rest
        }
        
        let type: StageType
        let hapticStyle: HapticStyle
        let motionNarrative: String
    }
    
    struct Nutrition: Hashable, Codable {
        let calories: Int
        let protein: Double
        let carbs: Double
        let fats: Double
        let fiber: Double
        let sugars: Double
        let sodium: Double
    }
    
    struct Ingredient: Identifiable, Hashable, Codable {
        enum Allergen: String, CaseIterable, Codable {
            case gluten, dairy, nuts, soy, shellfish, eggs, sesame, fish
        }
        
        let id = UUID()
        let name: String
        let quantity: Double
        let unit: String
        let preparation: String?
        let allergens: [Allergen]
    }
    
    enum Difficulty: String, CaseIterable, Codable {
        case easy = "Easy"
        case moderate = "Moderate"
        case advanced = "Advanced"
    }
    
    let id = UUID()
    let title: String
    let subtitle: String
    let author: String
    let heroVideo: String?
    let heroImageName: String
    let categories: [String]
    let difficulty: Difficulty
    let duration: TimeInterval
    let servings: Int
    let ingredients: [Ingredient]
    let steps: [Step]
    let nutrition: Nutrition
    let seasonalTag: SeasonalHighlight.Tag?
    let isPinned: Bool
    let isDownloaded: Bool
}

struct SeasonalHighlight: Identifiable, Hashable, Codable {
    enum Tag: String, CaseIterable, Codable {
        case spring, summer, autumn, winter, trending
    }
    
    let id = UUID()
    let title: String
    let description: String
    let tag: Tag
    let heroAssetName: String
    let callToAction: String
}

struct PantryItem: Identifiable, Hashable {
    enum StockStatus: Hashable {
        case inStock
        case runningLow
        case outOfStock
        case expiringSoon(Date)
    }
    
    let id = UUID()
    let name: String
    let category: String
    let quantity: Double
    let unit: String
    let preferredBrand: String?
    let barcode: String?
    let allergens: [Recipe.Ingredient.Allergen]
    let stockStatus: StockStatus
    let updatedAt: Date
}

struct ShoppingListEntry: Identifiable, Hashable {
    let id = UUID()
    let pantryItem: PantryItem
    var isCompleted: Bool
    let householdAssignee: String?
}

struct MealPlan: Identifiable, Hashable {
    struct Entry: Identifiable, Hashable {
        let id = UUID()
        let date: Date
        let mealType: MealType
        let recipe: Recipe
    }
    
    enum MealType: String, CaseIterable {
        case breakfast, lunch, dinner, snack
    }
    
    let id = UUID()
    let title: String
    let owner: String
    let collaborators: [String]
    var entries: [Entry]
    var comments: [MealPlanComment]
}

struct MealPlanComment: Identifiable, Hashable {
    let id = UUID()
    let user: String
    let message: String
    let timestamp: Date
}

struct PantryAlert: Identifiable, Hashable {
    enum AlertType: Hashable {
        case restock
        case spoilage
        case allergenConflict(String)
        case upcomingRecipe(String)
    }
    
    let id = UUID()
    let title: String
    let message: String
    let type: AlertType
    let created: Date
}

struct TraeNotification: Identifiable, Hashable {
    enum NotificationType {
        case onboarding
        case pantry
        case mealPlan
        case insights
        case reminder
    }
    
    let id = UUID()
    let title: String
    let subtitle: String
    let type: NotificationType
    let deliveryDate: Date
    let isRead: Bool
}

struct Article: Identifiable, Hashable {
    let id = UUID()
    let title: String
    let author: String
    let publishDate: Date
    let readingTime: Int
    let summary: String
    let relatedRecipes: [Recipe]
}

struct TutorialVideo: Identifiable, Hashable, Codable {
    let id = UUID()
    let title: String
    let duration: TimeInterval
    let isDownloaded: Bool
    let associatedRecipe: Recipe?
    let downloadProgress: Double
}

struct SupportTopic: Identifiable, Hashable {
    enum TopicType: String {
        case faq = "FAQ"
        case chat = "Live Chat"
        case community = "Community"
        case feedback = "Feedback"
    }
    
    let id = UUID()
    let title: String
    let description: String
    let type: TopicType
    let isModerated: Bool
}

struct TraeProfile: Hashable {
    enum DietaryPreference: String, CaseIterable {
        case omnivore, vegetarian, vegan, pescatarian, keto, paleo, lowFodmap
    }
    
    var name: String
    var householdSize: Int
    var dietaryPreferences: [DietaryPreference]
    var allergens: [Recipe.Ingredient.Allergen]
    var prefersMetric: Bool
    var lowMotion: Bool
    var notificationsEnabled: Bool
    var preferredLanguage: String
}

enum HapticStyle: String, Codable {
    case tactile
    case simmer
    case plating
    case alert
}
