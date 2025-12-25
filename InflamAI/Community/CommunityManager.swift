//
//  CommunityManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import SwiftUI
import Combine
import CoreLocation
import CryptoKit
import os.log

// MARK: - Community Manager
class CommunityManager: NSObject, ObservableObject {
    
    static let shared = CommunityManager()
    
    private let logger = Logger(subsystem: "InflamAI", category: "Community")
    private let networkManager = NetworkManager.shared
    private let securityManager = SecurityManager.shared
    private let locationManager = CLLocationManager()
    
    // Published properties
    @Published var isConnected = false
    @Published var currentUser: CommunityUser?
    @Published var supportGroups: [SupportGroup] = []
    @Published var nearbyGroups: [SupportGroup] = []
    @Published var joinedGroups: [SupportGroup] = []
    @Published var discussions: [Discussion] = []
    @Published var experiences: [SharedExperience] = []
    @Published var mentorships: [Mentorship] = []
    @Published var events: [CommunityEvent] = []
    @Published var resources: [CommunityResource] = []
    @Published var notifications: [CommunityNotification] = []
    
    // Privacy and safety
    @Published var privacySettings = CommunityPrivacySettings()
    @Published var safetySettings = CommunitySafetySettings()
    @Published var moderationReports: [ModerationReport] = []
    @Published var blockedUsers: Set<String> = []
    @Published var mutedTopics: Set<String> = []
    
    // Search and discovery
    @Published var searchResults: [CommunitySearchResult] = []
    @Published var recommendedGroups: [SupportGroup] = []
    @Published var trendingTopics: [String] = []
    @Published var featuredContent: [FeaturedContent] = []
    
    // User engagement
    @Published var userStats = CommunityUserStats()
    @Published var achievements: [CommunityAchievement] = []
    @Published var badges: [CommunityBadge] = []
    @Published var reputation: Int = 0
    
    // Internal state
    private var cancellables = Set<AnyCancellable>()
    private var currentLocation: CLLocation?
    private var isLocationAuthorized = false
    private var anonymousIdentifier: String
    private var encryptionKey: SymmetricKey
    
    override init() {
        self.anonymousIdentifier = UUID().uuidString
        self.encryptionKey = SymmetricKey(size: .bits256)
        
        super.init()
        
        setupLocationManager()
        loadCommunityData()
        setupNetworkObservers()
        generateAnonymousProfile()
    }
    
    // MARK: - Public Methods
    
    func joinCommunity(with profile: CommunityProfile) async -> Bool {
        do {
            let anonymousProfile = anonymizeProfile(profile)
            let user = CommunityUser(
                id: anonymousIdentifier,
                profile: anonymousProfile,
                joinDate: Date(),
                isAnonymous: true,
                verificationLevel: .basic
            )
            
            let success = try await networkManager.joinCommunity(user: user)
            
            if success {
                currentUser = user
                isConnected = true
                saveCommunityData()
                
                logger.info("Successfully joined community")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to join community: \(error.localizedDescription)")
            return false
        }
    }
    
    func leaveCommunity() async -> Bool {
        guard let user = currentUser else { return false }
        
        do {
            let success = try await networkManager.leaveCommunity(userId: user.id)
            
            if success {
                currentUser = nil
                isConnected = false
                clearCommunityData()
                
                logger.info("Successfully left community")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to leave community: \(error.localizedDescription)")
            return false
        }
    }
    
    func searchSupportGroups(query: String, filters: GroupSearchFilters) async -> [SupportGroup] {
        do {
            let results = try await networkManager.searchSupportGroups(
                query: query,
                filters: filters,
                location: currentLocation
            )
            
            DispatchQueue.main.async {
                self.searchResults = results.map { .group($0) }
            }
            
            return results
        } catch {
            logger.error("Failed to search support groups: \(error.localizedDescription)")
            return []
        }
    }
    
    func findNearbyGroups(radius: Double = 50000) async -> [SupportGroup] {
        guard let location = currentLocation else {
            logger.warning("Location not available for nearby group search")
            return []
        }
        
        do {
            let groups = try await networkManager.findNearbyGroups(
                location: location,
                radius: radius
            )
            
            DispatchQueue.main.async {
                self.nearbyGroups = groups
            }
            
            return groups
        } catch {
            logger.error("Failed to find nearby groups: \(error.localizedDescription)")
            return []
        }
    }
    
    func joinSupportGroup(_ group: SupportGroup) async -> Bool {
        guard let user = currentUser else { return false }
        
        do {
            let success = try await networkManager.joinSupportGroup(
                groupId: group.id,
                userId: user.id
            )
            
            if success {
                DispatchQueue.main.async {
                    if !self.joinedGroups.contains(where: { $0.id == group.id }) {
                        self.joinedGroups.append(group)
                    }
                }
                
                saveCommunityData()
                logger.info("Successfully joined support group: \(group.name)")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to join support group: \(error.localizedDescription)")
            return false
        }
    }
    
    func leaveSupportGroup(_ group: SupportGroup) async -> Bool {
        guard let user = currentUser else { return false }
        
        do {
            let success = try await networkManager.leaveSupportGroup(
                groupId: group.id,
                userId: user.id
            )
            
            if success {
                DispatchQueue.main.async {
                    self.joinedGroups.removeAll { $0.id == group.id }
                }
                
                saveCommunityData()
                logger.info("Successfully left support group: \(group.name)")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to leave support group: \(error.localizedDescription)")
            return false
        }
    }
    
    func shareExperience(_ experience: SharedExperience) async -> Bool {
        guard let user = currentUser else { return false }
        
        do {
            let anonymizedExperience = anonymizeExperience(experience)
            let success = try await networkManager.shareExperience(
                experience: anonymizedExperience,
                userId: user.id
            )
            
            if success {
                DispatchQueue.main.async {
                    self.experiences.append(anonymizedExperience)
                    self.userStats.experiencesShared += 1
                }
                
                saveCommunityData()
                logger.info("Successfully shared experience")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to share experience: \(error.localizedDescription)")
            return false
        }
    }
    
    func startDiscussion(_ discussion: Discussion) async -> Bool {
        guard let user = currentUser else { return false }
        
        do {
            let anonymizedDiscussion = anonymizeDiscussion(discussion)
            let success = try await networkManager.startDiscussion(
                discussion: anonymizedDiscussion,
                userId: user.id
            )
            
            if success {
                DispatchQueue.main.async {
                    self.discussions.append(anonymizedDiscussion)
                    self.userStats.discussionsStarted += 1
                }
                
                saveCommunityData()
                logger.info("Successfully started discussion")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to start discussion: \(error.localizedDescription)")
            return false
        }
    }
    
    func replyToDiscussion(_ discussionId: String, reply: DiscussionReply) async -> Bool {
        guard let user = currentUser else { return false }
        
        do {
            let anonymizedReply = anonymizeReply(reply)
            let success = try await networkManager.replyToDiscussion(
                discussionId: discussionId,
                reply: anonymizedReply,
                userId: user.id
            )
            
            if success {
                DispatchQueue.main.async {
                    if let index = self.discussions.firstIndex(where: { $0.id == discussionId }) {
                        self.discussions[index].replies.append(anonymizedReply)
                        self.discussions[index].replyCount += 1
                    }
                    self.userStats.repliesPosted += 1
                }
                
                saveCommunityData()
                logger.info("Successfully replied to discussion")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to reply to discussion: \(error.localizedDescription)")
            return false
        }
    }
    
    func requestMentorship(in area: MentorshipArea) async -> Bool {
        guard let user = currentUser else { return false }
        
        do {
            let request = MentorshipRequest(
                id: UUID().uuidString,
                requesterId: user.id,
                area: area,
                description: "",
                createdAt: Date(),
                status: .pending
            )
            
            let success = try await networkManager.requestMentorship(request: request)
            
            if success {
                logger.info("Successfully requested mentorship in \(area)")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to request mentorship: \(error.localizedDescription)")
            return false
        }
    }
    
    func offerMentorship(in area: MentorshipArea, experience: String) async -> Bool {
        guard let user = currentUser else { return false }
        
        do {
            let offer = MentorshipOffer(
                id: UUID().uuidString,
                mentorId: user.id,
                area: area,
                experience: experience,
                availability: [],
                createdAt: Date(),
                isActive: true
            )
            
            let success = try await networkManager.offerMentorship(offer: offer)
            
            if success {
                logger.info("Successfully offered mentorship in \(area)")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to offer mentorship: \(error.localizedDescription)")
            return false
        }
    }
    
    func reportContent(_ report: ModerationReport) async -> Bool {
        do {
            let success = try await networkManager.reportContent(report: report)
            
            if success {
                DispatchQueue.main.async {
                    self.moderationReports.append(report)
                }
                
                logger.info("Successfully reported content")
                return true
            }
            
            return false
        } catch {
            logger.error("Failed to report content: \(error.localizedDescription)")
            return false
        }
    }
    
    func blockUser(_ userId: String) {
        blockedUsers.insert(userId)
        saveCommunityData()
        
        // Remove blocked user's content from current views
        discussions.removeAll { $0.authorId == userId }
        experiences.removeAll { $0.authorId == userId }
        
        logger.info("User blocked: \(userId)")
    }
    
    func unblockUser(_ userId: String) {
        blockedUsers.remove(userId)
        saveCommunityData()
        
        logger.info("User unblocked: \(userId)")
    }
    
    func muteTopics(_ topics: [String]) {
        mutedTopics.formUnion(topics)
        saveCommunityData()
        
        // Filter out muted topics from current content
        filterMutedContent()
        
        logger.info("Topics muted: \(topics.joined(separator: ", "))")
    }
    
    func unmuteTopics(_ topics: [String]) {
        topics.forEach { mutedTopics.remove($0) }
        saveCommunityData()
        
        logger.info("Topics unmuted: \(topics.joined(separator: ", "))")
    }
    
    func updatePrivacySettings(_ settings: CommunityPrivacySettings) {
        privacySettings = settings
        saveCommunityData()
        
        logger.info("Privacy settings updated")
    }
    
    func updateSafetySettings(_ settings: CommunitySafetySettings) {
        safetySettings = settings
        saveCommunityData()
        
        logger.info("Safety settings updated")
    }
    
    func loadDiscussions(for groupId: String) async -> [Discussion] {
        do {
            let discussions = try await networkManager.loadDiscussions(groupId: groupId)
            let filteredDiscussions = filterBlockedAndMutedContent(discussions)
            
            DispatchQueue.main.async {
                self.discussions = filteredDiscussions
            }
            
            return filteredDiscussions
        } catch {
            logger.error("Failed to load discussions: \(error.localizedDescription)")
            return []
        }
    }
    
    func loadExperiences(for groupId: String) async -> [SharedExperience] {
        do {
            let experiences = try await networkManager.loadExperiences(groupId: groupId)
            let filteredExperiences = filterBlockedAndMutedExperiences(experiences)
            
            DispatchQueue.main.async {
                self.experiences = filteredExperiences
            }
            
            return filteredExperiences
        } catch {
            logger.error("Failed to load experiences: \(error.localizedDescription)")
            return []
        }
    }
    
    func loadCommunityEvents() async -> [CommunityEvent] {
        do {
            let events = try await networkManager.loadCommunityEvents()
            
            DispatchQueue.main.async {
                self.events = events
            }
            
            return events
        } catch {
            logger.error("Failed to load community events: \(error.localizedDescription)")
            return []
        }
    }
    
    func loadCommunityResources() async -> [CommunityResource] {
        do {
            let resources = try await networkManager.loadCommunityResources()
            
            DispatchQueue.main.async {
                self.resources = resources
            }
            
            return resources
        } catch {
            logger.error("Failed to load community resources: \(error.localizedDescription)")
            return []
        }
    }
    
    func getRecommendations() async -> CommunityRecommendations {
        guard let user = currentUser else {
            return CommunityRecommendations()
        }
        
        do {
            let recommendations = try await networkManager.getRecommendations(
                userId: user.id,
                interests: user.profile.interests,
                location: currentLocation
            )
            
            DispatchQueue.main.async {
                self.recommendedGroups = recommendations.groups
                self.featuredContent = recommendations.content
                self.trendingTopics = recommendations.topics
            }
            
            return recommendations
        } catch {
            logger.error("Failed to get recommendations: \(error.localizedDescription)")
            return CommunityRecommendations()
        }
    }
    
    func exportCommunityData() -> Data? {
        do {
            let export = CommunityDataExport(
                timestamp: Date(),
                user: currentUser,
                joinedGroups: joinedGroups,
                experiences: experiences,
                discussions: discussions,
                userStats: userStats,
                achievements: achievements,
                badges: badges,
                reputation: reputation
            )
            
            return try JSONEncoder().encode(export)
        } catch {
            logger.error("Failed to export community data: \(error.localizedDescription)")
            return nil
        }
    }
    
    func importCommunityData(from data: Data) -> Bool {
        do {
            let export = try JSONDecoder().decode(CommunityDataExport.self, from: data)
            
            currentUser = export.user
            joinedGroups = export.joinedGroups
            experiences = export.experiences
            discussions = export.discussions
            userStats = export.userStats
            achievements = export.achievements
            badges = export.badges
            reputation = export.reputation
            
            saveCommunityData()
            
            logger.info("Community data imported successfully")
            return true
        } catch {
            logger.error("Failed to import community data: \(error.localizedDescription)")
            return false
        }
    }
    
    // MARK: - Private Methods
    
    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyKilometer
        
        if CLLocationManager.locationServicesEnabled() {
            locationManager.requestWhenInUseAuthorization()
        }
    }
    
    private func setupNetworkObservers() {
        networkManager.$isConnected
            .sink { [weak self] isConnected in
                self?.isConnected = isConnected
            }
            .store(in: &cancellables)
    }
    
    private func generateAnonymousProfile() {
        let profile = CommunityProfile(
            displayName: "Anonymous User",
            bio: "",
            interests: [],
            conditions: [],
            experienceLevel: .beginner,
            isPublic: false,
            showLocation: false,
            allowDirectMessages: false
        )
        
        currentUser = CommunityUser(
            id: anonymousIdentifier,
            profile: profile,
            joinDate: Date(),
            isAnonymous: true,
            verificationLevel: .basic
        )
    }
    
    private func anonymizeProfile(_ profile: CommunityProfile) -> CommunityProfile {
        var anonymized = profile
        
        if !privacySettings.showRealName {
            anonymized.displayName = generateAnonymousName()
        }
        
        if !privacySettings.showDetailedProfile {
            anonymized.bio = ""
        }
        
        if !privacySettings.showLocation {
            anonymized.showLocation = false
        }
        
        return anonymized
    }
    
    private func anonymizeExperience(_ experience: SharedExperience) -> SharedExperience {
        var anonymized = experience
        
        if privacySettings.anonymizeContent {
            anonymized.content = removePersonalInfo(from: experience.content)
        }
        
        anonymized.authorId = anonymousIdentifier
        
        return anonymized
    }
    
    private func anonymizeDiscussion(_ discussion: Discussion) -> Discussion {
        var anonymized = discussion
        
        if privacySettings.anonymizeContent {
            anonymized.title = removePersonalInfo(from: discussion.title)
            anonymized.content = removePersonalInfo(from: discussion.content)
        }
        
        anonymized.authorId = anonymousIdentifier
        
        return anonymized
    }
    
    private func anonymizeReply(_ reply: DiscussionReply) -> DiscussionReply {
        var anonymized = reply
        
        if privacySettings.anonymizeContent {
            anonymized.content = removePersonalInfo(from: reply.content)
        }
        
        anonymized.authorId = anonymousIdentifier
        
        return anonymized
    }
    
    private func generateAnonymousName() -> String {
        let adjectives = ["Brave", "Strong", "Hopeful", "Resilient", "Caring", "Wise", "Gentle", "Bold"]
        let nouns = ["Warrior", "Fighter", "Survivor", "Helper", "Friend", "Guide", "Supporter", "Champion"]
        
        let adjective = adjectives.randomElement() ?? "Anonymous"
        let noun = nouns.randomElement() ?? "User"
        let number = Int.random(in: 100...999)
        
        return "\(adjective)\(noun)\(number)"
    }
    
    private func removePersonalInfo(from text: String) -> String {
        // Simple implementation - in production, use more sophisticated NLP
        var cleaned = text
        
        // Remove potential names, addresses, phone numbers, etc.
        let patterns = [
            "\\b[A-Z][a-z]+ [A-Z][a-z]+\\b", // Names
            "\\b\\d{3}-\\d{3}-\\d{4}\\b", // Phone numbers
            "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b" // Emails
        ]
        
        for pattern in patterns {
            cleaned = cleaned.replacingOccurrences(
                of: pattern,
                with: "[REDACTED]",
                options: .regularExpression
            )
        }
        
        return cleaned
    }
    
    private func filterBlockedAndMutedContent(_ discussions: [Discussion]) -> [Discussion] {
        return discussions.filter { discussion in
            !blockedUsers.contains(discussion.authorId) &&
            !discussion.tags.contains { mutedTopics.contains($0) }
        }
    }
    
    private func filterBlockedAndMutedExperiences(_ experiences: [SharedExperience]) -> [SharedExperience] {
        return experiences.filter { experience in
            !blockedUsers.contains(experience.authorId) &&
            !experience.tags.contains { mutedTopics.contains($0) }
        }
    }
    
    private func filterMutedContent() {
        discussions = filterBlockedAndMutedContent(discussions)
        experiences = filterBlockedAndMutedExperiences(experiences)
    }
    
    private func saveCommunityData() {
        let data = CommunityData(
            currentUser: currentUser,
            joinedGroups: joinedGroups,
            privacySettings: privacySettings,
            safetySettings: safetySettings,
            blockedUsers: Array(blockedUsers),
            mutedTopics: Array(mutedTopics),
            userStats: userStats,
            achievements: achievements,
            badges: badges,
            reputation: reputation
        )
        
        do {
            let encoded = try JSONEncoder().encode(data)
            UserDefaults.standard.set(encoded, forKey: "CommunityData")
        } catch {
            logger.error("Failed to save community data: \(error.localizedDescription)")
        }
    }
    
    private func loadCommunityData() {
        guard let data = UserDefaults.standard.data(forKey: "CommunityData"),
              let communityData = try? JSONDecoder().decode(CommunityData.self, from: data) else {
            return
        }
        
        currentUser = communityData.currentUser
        joinedGroups = communityData.joinedGroups
        privacySettings = communityData.privacySettings
        safetySettings = communityData.safetySettings
        blockedUsers = Set(communityData.blockedUsers)
        mutedTopics = Set(communityData.mutedTopics)
        userStats = communityData.userStats
        achievements = communityData.achievements
        badges = communityData.badges
        reputation = communityData.reputation
        
        isConnected = currentUser != nil
    }
    
    private func clearCommunityData() {
        UserDefaults.standard.removeObject(forKey: "CommunityData")
        
        supportGroups.removeAll()
        nearbyGroups.removeAll()
        joinedGroups.removeAll()
        discussions.removeAll()
        experiences.removeAll()
        mentorships.removeAll()
        events.removeAll()
        resources.removeAll()
        notifications.removeAll()
        searchResults.removeAll()
        recommendedGroups.removeAll()
        trendingTopics.removeAll()
        featuredContent.removeAll()
        moderationReports.removeAll()
        blockedUsers.removeAll()
        mutedTopics.removeAll()
        achievements.removeAll()
        badges.removeAll()
        
        userStats = CommunityUserStats()
        reputation = 0
    }
}

// MARK: - CLLocationManagerDelegate

extension CommunityManager: CLLocationManagerDelegate {
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentLocation = locations.last
    }
    
    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        switch status {
        case .authorizedWhenInUse, .authorizedAlways:
            isLocationAuthorized = true
            locationManager.startUpdatingLocation()
            
        case .denied, .restricted:
            isLocationAuthorized = false
            currentLocation = nil
            
        case .notDetermined:
            locationManager.requestWhenInUseAuthorization()
            
        @unknown default:
            break
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        logger.error("Location manager failed: \(error.localizedDescription)")
    }
}

// MARK: - Supporting Types

struct CommunityUser: Codable, Identifiable {
    let id: String
    var profile: CommunityProfile
    let joinDate: Date
    let isAnonymous: Bool
    let verificationLevel: VerificationLevel
    var lastActive: Date = Date()
    var isOnline: Bool = false
}

struct CommunityProfile: Codable {
    var displayName: String
    var bio: String
    var interests: [String]
    var conditions: [String]
    var experienceLevel: ExperienceLevel
    var isPublic: Bool
    var showLocation: Bool
    var allowDirectMessages: Bool
    var profileImageUrl: String?
    var languages: [String] = []
    var timezone: String?
}

struct SupportGroup: Codable, Identifiable {
    let id: String
    let name: String
    let description: String
    let category: GroupCategory
    let memberCount: Int
    let isPrivate: Bool
    let location: GroupLocation?
    let tags: [String]
    let moderators: [String]
    let createdAt: Date
    let rules: [String]
    let meetingSchedule: MeetingSchedule?
    var isJoined: Bool = false
}

struct Discussion: Codable, Identifiable {
    let id: String
    var title: String
    var content: String
    let authorId: String
    let groupId: String
    let createdAt: Date
    var updatedAt: Date
    var replies: [DiscussionReply]
    var replyCount: Int
    var likeCount: Int
    var tags: [String]
    let category: DiscussionCategory
    var isPinned: Bool = false
    var isLocked: Bool = false
}

struct DiscussionReply: Codable, Identifiable {
    let id: String
    var content: String
    let authorId: String
    let discussionId: String
    let createdAt: Date
    var likeCount: Int
    let parentReplyId: String?
}

struct SharedExperience: Codable, Identifiable {
    let id: String
    var title: String
    var content: String
    let authorId: String
    let createdAt: Date
    var tags: [String]
    let category: ExperienceCategory
    var helpfulCount: Int
    var commentCount: Int
    let isAnonymous: Bool
    var attachments: [ExperienceAttachment]
}

struct Mentorship: Codable, Identifiable {
    let id: String
    let mentorId: String
    let menteeId: String
    let area: MentorshipArea
    let status: MentorshipStatus
    let createdAt: Date
    var sessions: [MentorshipSession]
    var notes: String
}

struct CommunityEvent: Codable, Identifiable {
    let id: String
    let title: String
    let description: String
    let organizer: String
    let startDate: Date
    let endDate: Date
    let location: EventLocation
    let category: EventCategory
    var attendeeCount: Int
    let maxAttendees: Int?
    var isRegistered: Bool = false
    let tags: [String]
}

struct CommunityResource: Codable, Identifiable {
    let id: String
    let title: String
    let description: String
    let type: ResourceType
    let url: String?
    let content: String?
    let author: String
    let createdAt: Date
    var rating: Double
    var reviewCount: Int
    let tags: [String]
    let category: ResourceCategory
}

struct CommunityNotification: Codable, Identifiable {
    let id: String
    let type: NotificationType
    let title: String
    let message: String
    let createdAt: Date
    var isRead: Bool = false
    let actionUrl: String?
    let relatedId: String?
}

struct CommunityPrivacySettings: Codable {
    var showRealName = false
    var showLocation = false
    var showDetailedProfile = false
    var allowDirectMessages = true
    var anonymizeContent = true
    var shareActivityStatus = false
    var allowProfileSearch = false
    var showOnlineStatus = false
}

struct CommunitySafetySettings: Codable {
    var enableContentFiltering = true
    var blockInappropriateContent = true
    var requireModerationApproval = false
    var enableAutomaticReporting = true
    var allowAnonymousReporting = true
    var hideReportedContent = true
    var enableSafeMode = false
}

struct ModerationReport: Codable, Identifiable {
    let id: String
    let reporterId: String
    let contentId: String
    let contentType: ContentType
    let reason: ReportReason
    let description: String
    let createdAt: Date
    var status: ReportStatus = .pending
}

struct CommunityUserStats: Codable {
    var experiencesShared = 0
    var discussionsStarted = 0
    var repliesPosted = 0
    var helpfulVotes = 0
    var mentorshipSessions = 0
    var eventsAttended = 0
    var resourcesShared = 0
    var daysActive = 0
}

struct CommunityAchievement: Codable, Identifiable {
    let id: String
    let name: String
    let description: String
    let iconName: String
    let unlockedAt: Date
    let category: AchievementCategory
    let points: Int
}

struct CommunityBadge: Codable, Identifiable {
    let id: String
    let name: String
    let description: String
    let iconName: String
    let earnedAt: Date
    let level: BadgeLevel
}

// MARK: - Enums

enum VerificationLevel: String, Codable, CaseIterable {
    case basic = "basic"
    case verified = "verified"
    case expert = "expert"
    case moderator = "moderator"
}

enum ExperienceLevel: String, Codable, CaseIterable {
    case beginner = "beginner"
    case intermediate = "intermediate"
    case experienced = "experienced"
    case expert = "expert"
}

enum GroupCategory: String, Codable, CaseIterable {
    case support = "support"
    case education = "education"
    case social = "social"
    case advocacy = "advocacy"
    case research = "research"
    case wellness = "wellness"
}

enum DiscussionCategory: String, Codable, CaseIterable {
    case general = "general"
    case symptoms = "symptoms"
    case treatment = "treatment"
    case lifestyle = "lifestyle"
    case emotional = "emotional"
    case practical = "practical"
}

enum ExperienceCategory: String, Codable, CaseIterable {
    case diagnosis = "diagnosis"
    case treatment = "treatment"
    case lifestyle = "lifestyle"
    case coping = "coping"
    case relationships = "relationships"
    case work = "work"
}

enum MentorshipArea: String, Codable, CaseIterable {
    case newDiagnosis = "new_diagnosis"
    case treatmentOptions = "treatment_options"
    case lifestyleChanges = "lifestyle_changes"
    case emotionalSupport = "emotional_support"
    case workAccommodations = "work_accommodations"
    case relationships = "relationships"
}

enum MentorshipStatus: String, Codable, CaseIterable {
    case pending = "pending"
    case active = "active"
    case completed = "completed"
    case cancelled = "cancelled"
}

enum EventCategory: String, Codable, CaseIterable {
    case support = "support"
    case education = "education"
    case social = "social"
    case fundraising = "fundraising"
    case awareness = "awareness"
    case research = "research"
}

enum ResourceType: String, Codable, CaseIterable {
    case article = "article"
    case video = "video"
    case podcast = "podcast"
    case document = "document"
    case tool = "tool"
    case website = "website"
}

enum ResourceCategory: String, Codable, CaseIterable {
    case medical = "medical"
    case lifestyle = "lifestyle"
    case emotional = "emotional"
    case practical = "practical"
    case legal = "legal"
    case financial = "financial"
}

enum NotificationType: String, Codable, CaseIterable {
    case newMessage = "new_message"
    case newReply = "new_reply"
    case mentorshipRequest = "mentorship_request"
    case eventReminder = "event_reminder"
    case groupInvitation = "group_invitation"
    case achievement = "achievement"
    case moderation = "moderation"
}

enum ContentType: String, Codable, CaseIterable {
    case discussion = "discussion"
    case reply = "reply"
    case experience = "experience"
    case profile = "profile"
    case message = "message"
}

enum ReportReason: String, Codable, CaseIterable {
    case inappropriate = "inappropriate"
    case spam = "spam"
    case harassment = "harassment"
    case misinformation = "misinformation"
    case privacy = "privacy"
    case other = "other"
}

enum ReportStatus: String, Codable, CaseIterable {
    case pending = "pending"
    case reviewing = "reviewing"
    case resolved = "resolved"
    case dismissed = "dismissed"
}

enum AchievementCategory: String, Codable, CaseIterable {
    case participation = "participation"
    case helpfulness = "helpfulness"
    case leadership = "leadership"
    case milestone = "milestone"
    case special = "special"
}

enum BadgeLevel: String, Codable, CaseIterable {
    case bronze = "bronze"
    case silver = "silver"
    case gold = "gold"
    case platinum = "platinum"
}

// MARK: - Additional Supporting Types

struct GroupLocation: Codable {
    let city: String
    let state: String
    let country: String
    let coordinate: CLLocationCoordinate2D?
}

struct MeetingSchedule: Codable {
    let frequency: MeetingFrequency
    let dayOfWeek: Int?
    let time: String
    let timezone: String
    let isVirtual: Bool
    let meetingUrl: String?
}

struct ExperienceAttachment: Codable, Identifiable {
    let id: String
    let type: AttachmentType
    let url: String
    let filename: String
    let size: Int
}

struct MentorshipSession: Codable, Identifiable {
    let id: String
    let scheduledAt: Date
    let duration: TimeInterval
    let notes: String
    let rating: Int?
    let status: SessionStatus
}

struct EventLocation: Codable {
    let name: String
    let address: String?
    let city: String
    let state: String
    let country: String
    let isVirtual: Bool
    let virtualUrl: String?
}

struct GroupSearchFilters: Codable {
    var category: GroupCategory?
    var location: String?
    var radius: Double?
    var memberCountRange: ClosedRange<Int>?
    var isPrivate: Bool?
    var hasScheduledMeetings: Bool?
    var languages: [String]?
}

struct CommunitySearchResult {
    enum ResultType {
        case group(SupportGroup)
        case discussion(Discussion)
        case experience(SharedExperience)
        case user(CommunityUser)
        case resource(CommunityResource)
        case event(CommunityEvent)
    }
    
    let type: ResultType
    let relevanceScore: Double
}

struct FeaturedContent: Codable, Identifiable {
    let id: String
    let title: String
    let description: String
    let contentType: ContentType
    let contentId: String
    let imageUrl: String?
    let priority: Int
    let expiresAt: Date?
}

struct CommunityRecommendations {
    var groups: [SupportGroup] = []
    var content: [FeaturedContent] = []
    var topics: [String] = []
    var events: [CommunityEvent] = []
    var resources: [CommunityResource] = []
    var mentors: [CommunityUser] = []
}

struct MentorshipRequest: Codable, Identifiable {
    let id: String
    let requesterId: String
    let area: MentorshipArea
    let description: String
    let createdAt: Date
    var status: MentorshipRequestStatus
}

struct MentorshipOffer: Codable, Identifiable {
    let id: String
    let mentorId: String
    let area: MentorshipArea
    let experience: String
    let availability: [TimeSlot]
    let createdAt: Date
    var isActive: Bool
}

struct TimeSlot: Codable {
    let dayOfWeek: Int
    let startTime: String
    let endTime: String
    let timezone: String
}

struct CommunityData: Codable {
    let currentUser: CommunityUser?
    let joinedGroups: [SupportGroup]
    let privacySettings: CommunityPrivacySettings
    let safetySettings: CommunitySafetySettings
    let blockedUsers: [String]
    let mutedTopics: [String]
    let userStats: CommunityUserStats
    let achievements: [CommunityAchievement]
    let badges: [CommunityBadge]
    let reputation: Int
}

struct CommunityDataExport: Codable {
    let timestamp: Date
    let user: CommunityUser?
    let joinedGroups: [SupportGroup]
    let experiences: [SharedExperience]
    let discussions: [Discussion]
    let userStats: CommunityUserStats
    let achievements: [CommunityAchievement]
    let badges: [CommunityBadge]
    let reputation: Int
}

enum MeetingFrequency: String, Codable, CaseIterable {
    case weekly = "weekly"
    case biweekly = "biweekly"
    case monthly = "monthly"
    case irregular = "irregular"
}

enum AttachmentType: String, Codable, CaseIterable {
    case image = "image"
    case document = "document"
    case audio = "audio"
    case video = "video"
}

enum SessionStatus: String, Codable, CaseIterable {
    case scheduled = "scheduled"
    case completed = "completed"
    case cancelled = "cancelled"
    case noShow = "no_show"
}

enum MentorshipRequestStatus: String, Codable, CaseIterable {
    case pending = "pending"
    case matched = "matched"
    case declined = "declined"
    case expired = "expired"
}

// MARK: - Network Manager Extension

class NetworkManager: ObservableObject {
    static let shared = NetworkManager()
    
    @Published var isConnected = false
    
    private init() {}
    
    // Community-related network methods
    func joinCommunity(user: CommunityUser) async throws -> Bool {
        // Implementation for joining community
        return true
    }
    
    func leaveCommunity(userId: String) async throws -> Bool {
        // Implementation for leaving community
        return true
    }
    
    func searchSupportGroups(query: String, filters: GroupSearchFilters, location: CLLocation?) async throws -> [SupportGroup] {
        // Implementation for searching support groups
        return []
    }
    
    func findNearbyGroups(location: CLLocation, radius: Double) async throws -> [SupportGroup] {
        // Implementation for finding nearby groups
        return []
    }
    
    func joinSupportGroup(groupId: String, userId: String) async throws -> Bool {
        // Implementation for joining support group
        return true
    }
    
    func leaveSupportGroup(groupId: String, userId: String) async throws -> Bool {
        // Implementation for leaving support group
        return true
    }
    
    func shareExperience(experience: SharedExperience, userId: String) async throws -> Bool {
        // Implementation for sharing experience
        return true
    }
    
    func startDiscussion(discussion: Discussion, userId: String) async throws -> Bool {
        // Implementation for starting discussion
        return true
    }
    
    func replyToDiscussion(discussionId: String, reply: DiscussionReply, userId: String) async throws -> Bool {
        // Implementation for replying to discussion
        return true
    }
    
    func requestMentorship(request: MentorshipRequest) async throws -> Bool {
        // Implementation for requesting mentorship
        return true
    }
    
    func offerMentorship(offer: MentorshipOffer) async throws -> Bool {
        // Implementation for offering mentorship
        return true
    }
    
    func reportContent(report: ModerationReport) async throws -> Bool {
        // Implementation for reporting content
        return true
    }
    
    func loadDiscussions(groupId: String) async throws -> [Discussion] {
        // Implementation for loading discussions
        return []
    }
    
    func loadExperiences(groupId: String) async throws -> [SharedExperience] {
        // Implementation for loading experiences
        return []
    }
    
    func loadCommunityEvents() async throws -> [CommunityEvent] {
        // Implementation for loading community events
        return []
    }
    
    func loadCommunityResources() async throws -> [CommunityResource] {
        // Implementation for loading community resources
        return []
    }
    
    func getRecommendations(userId: String, interests: [String], location: CLLocation?) async throws -> CommunityRecommendations {
        // Implementation for getting recommendations
        return CommunityRecommendations()
    }
}