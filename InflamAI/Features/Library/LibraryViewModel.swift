//
//  LibraryViewModel.swift
//  InflamAI
//
//  ViewModel for Library feature - manages educational content state
//

import Foundation
import Combine

@MainActor
class LibraryViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var selectedSection: LibrarySection = .sleep
    @Published var favoriteTopics: Set<String> = []
    @Published var readProgress: [String: Bool] = [:]

    // MARK: - Private Properties

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init() {
        loadUserPreferences()
    }

    // MARK: - Public Methods

    /// Mark a section as read
    func markAsRead(_ section: LibrarySection) {
        readProgress[section.rawValue] = true
        saveUserPreferences()
    }

    /// Toggle favorite status for a topic
    func toggleFavorite(_ topic: String) {
        if favoriteTopics.contains(topic) {
            favoriteTopics.remove(topic)
        } else {
            favoriteTopics.insert(topic)
        }
        saveUserPreferences()
    }

    /// Check if section has been read
    func hasRead(_ section: LibrarySection) -> Bool {
        return readProgress[section.rawValue] ?? false
    }

    /// Get completion percentage
    func completionPercentage() -> Double {
        let totalSections = Double(LibrarySection.allCases.count)
        let readSections = Double(readProgress.values.filter { $0 }.count)
        return totalSections > 0 ? (readSections / totalSections) * 100 : 0
    }

    // MARK: - Private Methods

    private func loadUserPreferences() {
        // Load from UserDefaults
        if let savedFavorites = UserDefaults.standard.array(forKey: "LibraryFavorites") as? [String] {
            favoriteTopics = Set(savedFavorites)
        }

        if let savedProgress = UserDefaults.standard.dictionary(forKey: "LibraryReadProgress") as? [String: Bool] {
            readProgress = savedProgress
        }
    }

    private func saveUserPreferences() {
        UserDefaults.standard.set(Array(favoriteTopics), forKey: "LibraryFavorites")
        UserDefaults.standard.set(readProgress, forKey: "LibraryReadProgress")
    }
}

// MARK: - Educational Content Models

struct EducationalTopic: Identifiable {
    let id = UUID()
    let title: String
    let content: String
    let category: LibrarySection
    let sources: [String]
    let icon: String
    let color: String
}

// MARK: - Research Sources

extension LibraryViewModel {
    /// Get all research sources for citations
    static let researchSources = [
        ResearchSource(
            title: "Circadian Rhythms in Polymyalgia Rheumatica and Ankylosing Spondylitis",
            journal: "The Journal of Rheumatology",
            url: "https://www.jrheum.org/content/37/5/894"
        ),
        ResearchSource(
            title: "Sleep Disturbance in Ankylosing Spondylitis: Systematic Review and Meta-analysis",
            journal: "Advances in Rheumatology",
            url: "https://advancesinrheumatology.biomedcentral.com/articles/10.1186/s42358-023-00315-1"
        ),
        ResearchSource(
            title: "Effect of Sleep on Stiffness and Pain in Ankylosing Spondylitis",
            journal: "PubMed",
            url: "https://pubmed.ncbi.nlm.nih.gov/7774107/"
        ),
        ResearchSource(
            title: "Circadian Rhythms in Rheumatology - A Glucocorticoid Perspective",
            journal: "Arthritis Research & Therapy",
            url: "https://arthritis-research.biomedcentral.com/articles/10.1186/ar4687"
        )
    ]
}

struct ResearchSource: Identifiable {
    let id = UUID()
    let title: String
    let journal: String
    let url: String
}
