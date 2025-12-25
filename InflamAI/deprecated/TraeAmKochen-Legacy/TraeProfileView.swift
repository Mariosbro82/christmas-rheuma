//
//  TraeProfileView.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct TraeProfileView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    
    var body: some View {
        NavigationStack {
            List {
                Section("Profile & Preferences") {
                    NavigationLink {
                        TraePreferencesView()
                    } label: {
                        Label("Dietary profile", systemImage: "person.crop.circle.badge.checkmark")
                    }
                    NavigationLink {
                        TraeNotificationSettingsView()
                    } label: {
                        Label("Notifications & localization", systemImage: "bell.badge")
                    }
                    NavigationLink {
                        TraeAccessibilitySettingsView()
                    } label: {
                        Label("Accessibility & motion", systemImage: "figure.run")
                    }
                }
                
                Section("Learning & media") {
                    NavigationLink {
                        TraeRichMediaReader()
                    } label: {
                        Label("Articles & guides", systemImage: "book.pages")
                    }
                    NavigationLink {
                        TraeAudioNarrationView()
                    } label: {
                        Label("Audio narration", systemImage: "ear")
                    }
                }
                
                Section("Support & community") {
                    NavigationLink {
                        TraeSupportCenterView()
                    } label: {
                        Label("Support hub", systemImage: "lifepreserver")
                    }
                    NavigationLink {
                        TraeCommunityModerationView()
                    } label: {
                        Label("Community comments", systemImage: "text.bubble")
                    }
                }
                
                Section("Privacy & security") {
                    NavigationLink {
                        TraeSecuritySettingsView()
                    } label: {
                        Label("Security controls", systemImage: "lock.shield")
                    }
                }
            }
            .navigationTitle("Profile")
            .listStyle(.insetGrouped)
        }
    }
}

// MARK: - Profile settings

struct TraePreferencesView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    
    var body: some View {
        Form {
            Section("Household") {
                Stepper("Household size \(environment.profile.householdSize)", value: binding(\.householdSize), in: 1...8)
                Toggle("Metric units", isOn: binding(\.prefersMetric))
            }
            
            Section("Dietary preferences") {
                ForEach(TraeProfile.DietaryPreference.allCases, id: \.self) { preference in
                    Toggle(preference.rawValue.capitalized, isOn: Binding(
                        get: { environment.profile.dietaryPreferences.contains(preference) },
                        set: { newValue in
                            if newValue {
                                environment.profile.dietaryPreferences.append(preference)
                            } else {
                                environment.profile.dietaryPreferences.removeAll { $0 == preference }
                            }
                        }
                    ))
                }
            }
            
            Section("Allergens") {
                ForEach(Recipe.Ingredient.Allergen.allCases, id: \.self) { allergen in
                    Toggle(allergen.rawValue.capitalized, isOn: Binding(
                        get: { environment.profile.allergens.contains(allergen) },
                        set: { newValue in
                            if newValue {
                                environment.profile.allergens.append(allergen)
                                environment.securityCenter.markAllergenFilterUpdate()
                            } else {
                                environment.profile.allergens.removeAll { $0 == allergen }
                            }
                        }
                    ))
                }
            }
        }
        .navigationTitle("Dietary profile")
    }
    
    private func binding<Value>(_ keyPath: WritableKeyPath<TraeProfile, Value>) -> Binding<Value> {
        Binding(
            get: { environment.profile[keyPath: keyPath] },
            set: { environment.profile[keyPath: keyPath] = $0 }
        )
    }
}

struct TraeNotificationSettingsView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    @State private var selectedLanguage: String = "en"
    
    var body: some View {
        Form {
            Section("Language") {
                Picker("Preferred language", selection: $selectedLanguage) {
                    ForEach(["en", "de", "fr", "es"], id: \.self) { language in
                        Text(language.uppercased()).tag(language)
                    }
                }
                .onChange(of: selectedLanguage) { newValue in
                    environment.profile.preferredLanguage = newValue
                }
            }
            
            Section("Notifications") {
                Toggle("Enable push notifications", isOn: Binding(
                    get: { environment.profile.notificationsEnabled },
                    set: { environment.profile.notificationsEnabled = $0 }
                ))
                
                ForEach(environment.notifications) { notification in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(notification.title)
                                .font(TraeTypography.body)
                            Text(notification.subtitle)
                                .font(TraeTypography.footnote)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(notification.deliveryDate, style: .time)
                            .font(TraeTypography.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle("Notifications")
    }
}

struct TraeAccessibilitySettingsView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    
    var body: some View {
        Form {
            Section("Motion") {
                Toggle("Reduce motion and animations", isOn: Binding(
                    get: { environment.profile.lowMotion },
                    set: { environment.profile.lowMotion = $0 }
                ))
            }
            Section("Audio descriptions") {
                Toggle("Enable narrated steps", isOn: .constant(true))
                Toggle("Auto captions for videos", isOn: .constant(true))
            }
        }
        .navigationTitle("Accessibility")
    }
}

// MARK: - Media & articles

struct TraeRichMediaReader: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: TraeSpacing.lg) {
                ForEach(environment.articles) { article in
                    TraeArticleCard(article: article)
                }
            }
            .padding()
        }
        .navigationTitle("Rich media reader")
    }
}

struct TraeArticleCard: View {
    let article: Article
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            Text(article.title)
                .font(TraeTypography.title2)
            Text("By \(article.author) • \(article.publishDate.formatted(date: .long, time: .omitted))")
                .font(TraeTypography.footnote)
                .foregroundStyle(.secondary)
            Text(article.summary)
                .font(TraeTypography.body)
            Divider()
            VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                Text("Related recipes")
                    .font(TraeTypography.subheadline)
                ForEach(article.relatedRecipes) { recipe in
                    Label(recipe.title, systemImage: "fork.knife")
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 24).fill(.background))
        .shadow(color: Color.black.opacity(0.06), radius: 12, x: 0, y: 4)
    }
}

struct TraeAudioNarrationView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.lg) {
            Text("Audio narration")
                .font(TraeTypography.title)
            Text("Narrated instructions adapt pacing to match cook mode. Download packs for offline use.")
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            
            List {
                Label("Seasonal playlist: Fermentation finesse", systemImage: "headphones")
                Label("Knife skills warmup", systemImage: "scissors")
                Label("Guided sensory tasting", systemImage: "sparkles")
            }
        }
        .padding()
        .navigationTitle("Audio narration")
    }
}

// MARK: - Support

struct TraeSupportCenterView: View {
    private let topics: [SupportTopic] = [
        SupportTopic(title: "Pantry sync delays", description: "Troubleshoot offline cache conflicts.", type: .faq, isModerated: true),
        SupportTopic(title: "Live chat with culinary coach", description: "Connect within 2 minutes.", type: .chat, isModerated: true),
        SupportTopic(title: "Community allergen swaps", description: "Share safe substitutions.", type: .community, isModerated: true),
        SupportTopic(title: "Submit feature feedback", description: "Help steer the roadmap.", type: .feedback, isModerated: true)
    ]
    
    var body: some View {
        List {
            ForEach(topics) { topic in
                VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                    HStack {
                        Text(topic.title)
                            .font(TraeTypography.headline)
                        Spacer()
                        Text(topic.type.rawValue)
                            .font(TraeTypography.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Text(topic.description)
                        .font(TraeTypography.body)
                        .foregroundStyle(.secondary)
                    if topic.isModerated {
                        Label("Moderated for safety", systemImage: "shield.checkerboard")
                            .font(TraeTypography.footnote)
                            .foregroundStyle(TraePalette.forestGreen)
                    }
                }
                .padding(.vertical, TraeSpacing.sm)
            }
        }
        .navigationTitle("Support center")
    }
}

struct TraeCommunityModerationView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    @State private var abuseDetectionEnabled = true
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.lg) {
            Toggle("Abuse detection", isOn: $abuseDetectionEnabled)
                .toggleStyle(SwitchToggleStyle(tint: TraePalette.traeOrange))
            
            Text("Community comments remain respectful. Machine learning flags spam, allergens, and harmful tips before publishing.")
                .font(TraeTypography.body)
                .foregroundStyle(.secondary)
            
            List {
                ForEach(environment.mealPlans.recentComments) { comment in
                    VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                        Text(comment.user)
                            .font(TraeTypography.subheadline)
                        Text(comment.message)
                            .font(TraeTypography.body)
                    }
                }
            }
        }
        .padding()
        .navigationTitle("Community comments")
    }
}

struct TraeSecuritySettingsView: View {
    @EnvironmentObject private var environment: TraeAppEnvironment
    @State private var telemetryOptIn = true
    
    var body: some View {
        Form {
            Section("Encrypted favorites") {
                Button("Encrypt current favorites") {
                    _ = environment.securityCenter.encryptFavorites(environment.recipeLibrary.recipes.filter { $0.isPinned })
                }
            }
            
            Section("Profile security") {
                Button("Enable profile lock") {
                    environment.securityCenter.recordProfileLock()
                }
            }
            
            Section("Telemetry & privacy") {
                Toggle("Allow anonymous analytics", isOn: $telemetryOptIn)
                    .onChange(of: telemetryOptIn) { newValue in
                        environment.securityCenter.updateTelemetryConsent(newValue)
                    }
            }
        }
        .navigationTitle("Security")
    }
}
