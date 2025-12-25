//
//  SocialSupportModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import SwiftUI
import Combine
import CloudKit
import CryptoKit

// MARK: - Data Models

struct SupportGroup: Identifiable, Codable {
    let id = UUID()
    let name: String
    let description: String
    let category: GroupCategory
    let privacy: GroupPrivacy
    let memberCount: Int
    let createdDate: Date
    let creatorID: String
    let tags: [String]
    let rules: [String]
    let moderators: [String]
    let isVerified: Bool
    let activityLevel: ActivityLevel
    let location: GroupLocation?
    let meetingSchedule: MeetingSchedule?
    let resources: [GroupResource]
    
    var displayMemberCount: String {
        if memberCount < 1000 {
            return "\(memberCount)"
        } else {
            return "\(String(format: "%.1f", Double(memberCount) / 1000))K"
        }
    }
}

enum GroupCategory: String, CaseIterable, Codable {
    case rheumatoidArthritis = "rheumatoid_arthritis"
    case osteoarthritis = "osteoarthritis"
    case fibromyalgia = "fibromyalgia"
    case lupus = "lupus"
    case chronicPain = "chronic_pain"
    case mentalHealth = "mental_health"
    case nutrition = "nutrition"
    case exercise = "exercise"
    case medication = "medication"
    case caregivers = "caregivers"
    case newlyDiagnosed = "newly_diagnosed"
    case general = "general"
    
    var displayName: String {
        switch self {
        case .rheumatoidArthritis: return "Rheumatoid Arthritis"
        case .osteoarthritis: return "Osteoarthritis"
        case .fibromyalgia: return "Fibromyalgia"
        case .lupus: return "Lupus"
        case .chronicPain: return "Chronic Pain"
        case .mentalHealth: return "Mental Health"
        case .nutrition: return "Nutrition"
        case .exercise: return "Exercise"
        case .medication: return "Medication"
        case .caregivers: return "Caregivers"
        case .newlyDiagnosed: return "Newly Diagnosed"
        case .general: return "General Support"
        }
    }
    
    var systemImage: String {
        switch self {
        case .rheumatoidArthritis, .osteoarthritis: return "figure.walk"
        case .fibromyalgia: return "heart.text.square"
        case .lupus: return "cross.case"
        case .chronicPain: return "bandage"
        case .mentalHealth: return "brain.head.profile"
        case .nutrition: return "leaf"
        case .exercise: return "figure.run"
        case .medication: return "pills"
        case .caregivers: return "person.2"
        case .newlyDiagnosed: return "info.circle"
        case .general: return "heart.circle"
        }
    }
    
    var color: String {
        switch self {
        case .rheumatoidArthritis: return "blue"
        case .osteoarthritis: return "green"
        case .fibromyalgia: return "purple"
        case .lupus: return "red"
        case .chronicPain: return "orange"
        case .mentalHealth: return "indigo"
        case .nutrition: return "mint"
        case .exercise: return "cyan"
        case .medication: return "pink"
        case .caregivers: return "brown"
        case .newlyDiagnosed: return "yellow"
        case .general: return "gray"
        }
    }
}

enum GroupPrivacy: String, CaseIterable, Codable {
    case open = "open"
    case closed = "closed"
    case secret = "secret"
    
    var displayName: String {
        switch self {
        case .open: return "Open"
        case .closed: return "Closed"
        case .secret: return "Secret"
        }
    }
    
    var description: String {
        switch self {
        case .open: return "Anyone can see the group and join"
        case .closed: return "Anyone can see the group, but must request to join"
        case .secret: return "Only members can see the group"
        }
    }
}

enum ActivityLevel: String, CaseIterable, Codable {
    case veryActive = "very_active"
    case active = "active"
    case moderate = "moderate"
    case quiet = "quiet"
    
    var displayName: String {
        switch self {
        case .veryActive: return "Very Active"
        case .active: return "Active"
        case .moderate: return "Moderate"
        case .quiet: return "Quiet"
        }
    }
    
    var color: Color {
        switch self {
        case .veryActive: return .green
        case .active: return .blue
        case .moderate: return .orange
        case .quiet: return .gray
        }
    }
}

struct GroupLocation: Codable {
    let city: String
    let state: String
    let country: String
    let isVirtual: Bool
    
    var displayLocation: String {
        if isVirtual {
            return "Virtual"
        }
        return "\(city), \(state)"
    }
}

struct MeetingSchedule: Codable {
    let frequency: MeetingFrequency
    let dayOfWeek: Int? // 1-7, Sunday = 1
    let time: Date
    let timezone: String
    let duration: TimeInterval
    let isRecurring: Bool
    
    var displaySchedule: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        
        switch frequency {
        case .weekly:
            let dayName = Calendar.current.weekdaySymbols[dayOfWeek! - 1]
            return "Every \(dayName) at \(formatter.string(from: time))"
        case .biweekly:
            return "Every other week at \(formatter.string(from: time))"
        case .monthly:
            return "Monthly at \(formatter.string(from: time))"
        case .asNeeded:
            return "As needed"
        }
    }
}

enum MeetingFrequency: String, CaseIterable, Codable {
    case weekly = "weekly"
    case biweekly = "biweekly"
    case monthly = "monthly"
    case asNeeded = "as_needed"
    
    var displayName: String {
        switch self {
        case .weekly: return "Weekly"
        case .biweekly: return "Bi-weekly"
        case .monthly: return "Monthly"
        case .asNeeded: return "As Needed"
        }
    }
}

struct GroupResource: Identifiable, Codable {
    let id = UUID()
    let title: String
    let description: String
    let type: ResourceType
    let url: URL?
    let fileData: Data?
    let addedBy: String
    let addedDate: Date
    let tags: [String]
    let isVerified: Bool
}

enum ResourceType: String, CaseIterable, Codable {
    case article = "article"
    case video = "video"
    case document = "document"
    case website = "website"
    case app = "app"
    case book = "book"
    case research = "research"
    
    var displayName: String {
        switch self {
        case .article: return "Article"
        case .video: return "Video"
        case .document: return "Document"
        case .website: return "Website"
        case .app: return "App"
        case .book: return "Book"
        case .research: return "Research"
        }
    }
    
    var systemImage: String {
        switch self {
        case .article: return "doc.text"
        case .video: return "play.rectangle"
        case .document: return "doc"
        case .website: return "globe"
        case .app: return "app"
        case .book: return "book"
        case .research: return "microscope"
        }
    }
}

struct SupportPost: Identifiable, Codable {
    let id = UUID()
    let groupID: UUID
    let authorID: String
    let authorName: String
    let content: String
    let type: PostType
    let createdDate: Date
    let lastModified: Date
    let tags: [String]
    let attachments: [PostAttachment]
    let reactions: [PostReaction]
    let comments: [PostComment]
    let isAnonymous: Bool
    let isPinned: Bool
    let isModerated: Bool
    let mood: MoodLevel?
    let painLevel: Int?
    let location: String?
    
    var reactionCounts: [ReactionType: Int] {
        var counts: [ReactionType: Int] = [:]
        for reaction in reactions {
            counts[reaction.type, default: 0] += 1
        }
        return counts
    }
    
    var totalReactions: Int {
        reactions.count
    }
    
    var totalComments: Int {
        comments.count
    }
}

enum PostType: String, CaseIterable, Codable {
    case question = "question"
    case experience = "experience"
    case support = "support"
    case celebration = "celebration"
    case resource = "resource"
    case announcement = "announcement"
    case poll = "poll"
    
    var displayName: String {
        switch self {
        case .question: return "Question"
        case .experience: return "Experience"
        case .support: return "Support"
        case .celebration: return "Celebration"
        case .resource: return "Resource"
        case .announcement: return "Announcement"
        case .poll: return "Poll"
        }
    }
    
    var systemImage: String {
        switch self {
        case .question: return "questionmark.circle"
        case .experience: return "person.crop.circle.badge.checkmark"
        case .support: return "heart.circle"
        case .celebration: return "party.popper"
        case .resource: return "link.circle"
        case .announcement: return "megaphone"
        case .poll: return "chart.bar"
        }
    }
    
    var color: Color {
        switch self {
        case .question: return .blue
        case .experience: return .green
        case .support: return .red
        case .celebration: return .yellow
        case .resource: return .purple
        case .announcement: return .orange
        case .poll: return .cyan
        }
    }
}

struct PostAttachment: Identifiable, Codable {
    let id = UUID()
    let type: AttachmentType
    let url: URL?
    let data: Data?
    let filename: String
    let size: Int64
    let mimeType: String
}

enum AttachmentType: String, CaseIterable, Codable {
    case image = "image"
    case document = "document"
    case link = "link"
    
    var displayName: String {
        switch self {
        case .image: return "Image"
        case .document: return "Document"
        case .link: return "Link"
        }
    }
}

struct PostReaction: Identifiable, Codable {
    let id = UUID()
    let userID: String
    let type: ReactionType
    let createdDate: Date
}

enum ReactionType: String, CaseIterable, Codable {
    case like = "like"
    case love = "love"
    case support = "support"
    case helpful = "helpful"
    case celebrate = "celebrate"
    case empathy = "empathy"
    
    var displayName: String {
        switch self {
        case .like: return "Like"
        case .love: return "Love"
        case .support: return "Support"
        case .helpful: return "Helpful"
        case .celebrate: return "Celebrate"
        case .empathy: return "Empathy"
        }
    }
    
    var emoji: String {
        switch self {
        case .like: return "👍"
        case .love: return "❤️"
        case .support: return "🤗"
        case .helpful: return "💡"
        case .celebrate: return "🎉"
        case .empathy: return "🫂"
        }
    }
}

struct PostComment: Identifiable, Codable {
    let id = UUID()
    let postID: UUID
    let authorID: String
    let authorName: String
    let content: String
    let createdDate: Date
    let lastModified: Date
    let parentCommentID: UUID?
    let reactions: [PostReaction]
    let isAnonymous: Bool
    let isModerated: Bool
    
    var isReply: Bool {
        parentCommentID != nil
    }
}

struct UserProfile: Identifiable, Codable {
    let id = UUID()
    let userID: String
    let displayName: String
    let bio: String?
    let avatar: Data?
    let joinDate: Date
    let lastActive: Date
    let conditions: [String]
    let interests: [String]
    let location: String?
    let isVerified: Bool
    let privacySettings: PrivacySettings
    let supportStats: SupportStats
    let badges: [UserBadge]
    let preferences: SocialPreferences
}

struct PrivacySettings: Codable {
    var showRealName: Bool
    var showLocation: Bool
    var showConditions: Bool
    var allowDirectMessages: Bool
    var showOnlineStatus: Bool
    var shareProgress: Bool
    var allowMentions: Bool
}

struct SupportStats: Codable {
    let postsCreated: Int
    let commentsPosted: Int
    let reactionsGiven: Int
    let reactionsReceived: Int
    let helpfulVotes: Int
    let groupsJoined: Int
    let daysActive: Int
    let supportGiven: Int
    let supportReceived: Int
    
    var engagementScore: Double {
        let totalEngagement = Double(postsCreated * 3 + commentsPosted * 2 + reactionsGiven + helpfulVotes * 2)
        let daysFactor = max(1.0, Double(daysActive))
        return totalEngagement / daysFactor
    }
}

struct UserBadge: Identifiable, Codable {
    let id = UUID()
    let type: BadgeType
    let earnedDate: Date
    let level: Int
    let description: String
}

enum BadgeType: String, CaseIterable, Codable {
    case helper = "helper"
    case supporter = "supporter"
    case contributor = "contributor"
    case mentor = "mentor"
    case advocate = "advocate"
    case researcher = "researcher"
    case community = "community"
    
    var displayName: String {
        switch self {
        case .helper: return "Helper"
        case .supporter: return "Supporter"
        case .contributor: return "Contributor"
        case .mentor: return "Mentor"
        case .advocate: return "Advocate"
        case .researcher: return "Researcher"
        case .community: return "Community Builder"
        }
    }
    
    var systemImage: String {
        switch self {
        case .helper: return "hand.raised"
        case .supporter: return "heart.circle"
        case .contributor: return "plus.circle"
        case .mentor: return "person.crop.circle.badge.checkmark"
        case .advocate: return "megaphone"
        case .researcher: return "microscope"
        case .community: return "person.3"
        }
    }
    
    var color: Color {
        switch self {
        case .helper: return .blue
        case .supporter: return .red
        case .contributor: return .green
        case .mentor: return .purple
        case .advocate: return .orange
        case .researcher: return .cyan
        case .community: return .pink
        }
    }
}

struct SocialPreferences: Codable {
    var notificationSettings: NotificationSettings
    var contentFilters: ContentFilters
    var interactionPreferences: InteractionPreferences
}

struct NotificationSettings: Codable {
    var newPosts: Bool
    var comments: Bool
    var reactions: Bool
    var mentions: Bool
    var directMessages: Bool
    var groupInvites: Bool
    var weeklyDigest: Bool
    var emergencyAlerts: Bool
}

struct ContentFilters: Codable {
    var hideNegativeContent: Bool
    var filterProfanity: Bool
    var hideTriggerWords: [String]
    var showOnlyVerifiedContent: Bool
    var minimumModerationLevel: ModerationLevel
}

enum ModerationLevel: String, CaseIterable, Codable {
    case none = "none"
    case basic = "basic"
    case moderate = "moderate"
    case strict = "strict"
    
    var displayName: String {
        switch self {
        case .none: return "None"
        case .basic: return "Basic"
        case .moderate: return "Moderate"
        case .strict: return "Strict"
        }
    }
}

struct InteractionPreferences: Codable {
    var autoJoinRecommendedGroups: Bool
    var allowGroupInvites: Bool
    var showSimilarExperiences: Bool
    var enableMentorship: Bool
    var participateInResearch: Bool
}

struct DirectMessage: Identifiable, Codable {
    let id = UUID()
    let conversationID: UUID
    let senderID: String
    let recipientID: String
    let content: String
    let sentDate: Date
    let readDate: Date?
    let type: MessageType
    let attachments: [PostAttachment]
    let isEncrypted: Bool
    
    var isRead: Bool {
        readDate != nil
    }
}

enum MessageType: String, CaseIterable, Codable {
    case text = "text"
    case supportOffer = "support_offer"
    case resourceShare = "resource_share"
    case checkIn = "check_in"
    case encouragement = "encouragement"
    
    var displayName: String {
        switch self {
        case .text: return "Message"
        case .supportOffer: return "Support Offer"
        case .resourceShare: return "Resource Share"
        case .checkIn: return "Check-in"
        case .encouragement: return "Encouragement"
        }
    }
}

struct Conversation: Identifiable, Codable {
    let id = UUID()
    let participants: [String]
    let lastMessage: DirectMessage?
    let createdDate: Date
    let lastActivity: Date
    let isArchived: Bool
    let isMuted: Bool
    
    var unreadCount: Int {
        // This would be calculated based on messages
        0
    }
}

// MARK: - Error Types

enum SocialSupportError: LocalizedError {
    case networkError(String)
    case authenticationRequired
    case permissionDenied
    case contentViolation(String)
    case groupNotFound
    case userNotFound
    case messageEncryptionFailed
    case moderationRequired
    case rateLimitExceeded
    case invalidContent
    
    var errorDescription: String? {
        switch self {
        case .networkError(let message):
            return "Network error: \(message)"
        case .authenticationRequired:
            return "Authentication required to access social features"
        case .permissionDenied:
            return "Permission denied for this action"
        case .contentViolation(let reason):
            return "Content violation: \(reason)"
        case .groupNotFound:
            return "Support group not found"
        case .userNotFound:
            return "User not found"
        case .messageEncryptionFailed:
            return "Failed to encrypt message"
        case .moderationRequired:
            return "Content requires moderation before posting"
        case .rateLimitExceeded:
            return "Rate limit exceeded. Please try again later"
        case .invalidContent:
            return "Invalid content format"
        }
    }
}

// MARK: - Social Support Manager

@MainActor
class SocialSupportManager: ObservableObject {
    static let shared = SocialSupportManager()
    
    @Published var currentUser: UserProfile?
    @Published var joinedGroups: [SupportGroup] = []
    @Published var availableGroups: [SupportGroup] = []
    @Published var recentPosts: [SupportPost] = []
    @Published var conversations: [Conversation] = []
    @Published var notifications: [SocialNotification] = []
    @Published var isLoading = false
    @Published var error: SocialSupportError?
    
    private let cloudKitManager: CloudKitManager
    private let encryptionManager: EncryptionManager
    private let moderationEngine: ModerationEngine
    private let recommendationEngine: RecommendationEngine
    private let notificationManager: SocialNotificationManager
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        self.cloudKitManager = CloudKitManager()
        self.encryptionManager = EncryptionManager()
        self.moderationEngine = ModerationEngine()
        self.recommendationEngine = RecommendationEngine()
        self.notificationManager = SocialNotificationManager()
        
        setupObservers()
    }
    
    private func setupObservers() {
        // Setup real-time observers for CloudKit changes
        cloudKitManager.groupUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] groups in
                self?.availableGroups = groups
            }
            .store(in: &cancellables)
        
        cloudKitManager.postUpdates
            .receive(on: DispatchQueue.main)
            .sink { [weak self] posts in
                self?.recentPosts = posts
            }
            .store(in: &cancellables)
    }
    
    // MARK: - User Management
    
    func createUserProfile(_ profile: UserProfile) async throws {
        isLoading = true
        defer { isLoading = false }
        
        do {
            try await cloudKitManager.saveUserProfile(profile)
            currentUser = profile
        } catch {
            self.error = .networkError(error.localizedDescription)
            throw error
        }
    }
    
    func updateUserProfile(_ profile: UserProfile) async throws {
        isLoading = true
        defer { isLoading = false }
        
        do {
            try await cloudKitManager.updateUserProfile(profile)
            currentUser = profile
        } catch {
            self.error = .networkError(error.localizedDescription)
            throw error
        }
    }
    
    func loadUserProfile(userID: String) async throws -> UserProfile {
        do {
            return try await cloudKitManager.fetchUserProfile(userID: userID)
        } catch {
            self.error = .userNotFound
            throw error
        }
    }
    
    // MARK: - Group Management
    
    func loadAvailableGroups() async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            let groups = try await cloudKitManager.fetchAvailableGroups()
            availableGroups = groups
        } catch {
            self.error = .networkError(error.localizedDescription)
        }
    }
    
    func searchGroups(query: String, category: GroupCategory? = nil) async -> [SupportGroup] {
        do {
            return try await cloudKitManager.searchGroups(query: query, category: category)
        } catch {
            self.error = .networkError(error.localizedDescription)
            return []
        }
    }
    
    func joinGroup(_ group: SupportGroup) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        isLoading = true
        defer { isLoading = false }
        
        do {
            try await cloudKitManager.joinGroup(groupID: group.id, userID: currentUser.userID)
            
            if !joinedGroups.contains(where: { $0.id == group.id }) {
                joinedGroups.append(group)
            }
            
            // Send welcome notification
            await notificationManager.sendWelcomeNotification(to: currentUser.userID, group: group)
            
        } catch {
            self.error = .networkError(error.localizedDescription)
            throw error
        }
    }
    
    func leaveGroup(_ group: SupportGroup) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        isLoading = true
        defer { isLoading = false }
        
        do {
            try await cloudKitManager.leaveGroup(groupID: group.id, userID: currentUser.userID)
            joinedGroups.removeAll { $0.id == group.id }
        } catch {
            self.error = .networkError(error.localizedDescription)
            throw error
        }
    }
    
    func createGroup(_ group: SupportGroup) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        isLoading = true
        defer { isLoading = false }
        
        do {
            try await cloudKitManager.createGroup(group)
            joinedGroups.append(group)
        } catch {
            self.error = .networkError(error.localizedDescription)
            throw error
        }
    }
    
    // MARK: - Post Management
    
    func createPost(_ post: SupportPost) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        isLoading = true
        defer { isLoading = false }
        
        // Content moderation
        let moderationResult = await moderationEngine.moderateContent(post.content)
        if !moderationResult.isApproved {
            throw SocialSupportError.contentViolation(moderationResult.reason)
        }
        
        do {
            try await cloudKitManager.createPost(post)
            recentPosts.insert(post, at: 0)
            
            // Update user stats
            await updateUserStats(for: currentUser.userID, action: .postCreated)
            
        } catch {
            self.error = .networkError(error.localizedDescription)
            throw error
        }
    }
    
    func loadGroupPosts(groupID: UUID, limit: Int = 20) async -> [SupportPost] {
        do {
            return try await cloudKitManager.fetchGroupPosts(groupID: groupID, limit: limit)
        } catch {
            self.error = .networkError(error.localizedDescription)
            return []
        }
    }
    
    func addReaction(to post: SupportPost, reaction: ReactionType) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        do {
            let postReaction = PostReaction(
                userID: currentUser.userID,
                type: reaction,
                createdDate: Date()
            )
            
            try await cloudKitManager.addReaction(postReaction, to: post.id)
            
            // Update local post
            if let index = recentPosts.firstIndex(where: { $0.id == post.id }) {
                recentPosts[index].reactions.append(postReaction)
            }
            
            // Update user stats
            await updateUserStats(for: currentUser.userID, action: .reactionGiven)
            await updateUserStats(for: post.authorID, action: .reactionReceived)
            
        } catch {
            self.error = .networkError(error.localizedDescription)
            throw error
        }
    }
    
    func addComment(to post: SupportPost, content: String, parentCommentID: UUID? = nil) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        // Content moderation
        let moderationResult = await moderationEngine.moderateContent(content)
        if !moderationResult.isApproved {
            throw SocialSupportError.contentViolation(moderationResult.reason)
        }
        
        do {
            let comment = PostComment(
                postID: post.id,
                authorID: currentUser.userID,
                authorName: currentUser.displayName,
                content: content,
                createdDate: Date(),
                lastModified: Date(),
                parentCommentID: parentCommentID,
                reactions: [],
                isAnonymous: false,
                isModerated: moderationResult.requiresReview
            )
            
            try await cloudKitManager.addComment(comment)
            
            // Update local post
            if let index = recentPosts.firstIndex(where: { $0.id == post.id }) {
                recentPosts[index].comments.append(comment)
            }
            
            // Update user stats
            await updateUserStats(for: currentUser.userID, action: .commentPosted)
            
        } catch {
            self.error = .networkError(error.localizedDescription)
            throw error
        }
    }
    
    // MARK: - Direct Messaging
    
    func sendDirectMessage(to recipientID: String, content: String, type: MessageType = .text) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        // Content moderation
        let moderationResult = await moderationEngine.moderateContent(content)
        if !moderationResult.isApproved {
            throw SocialSupportError.contentViolation(moderationResult.reason)
        }
        
        do {
            // Encrypt message content
            let encryptedContent = try encryptionManager.encrypt(content)
            
            let message = DirectMessage(
                conversationID: UUID(), // This should be determined by existing conversation
                senderID: currentUser.userID,
                recipientID: recipientID,
                content: encryptedContent,
                sentDate: Date(),
                readDate: nil,
                type: type,
                attachments: [],
                isEncrypted: true
            )
            
            try await cloudKitManager.sendDirectMessage(message)
            
            // Send push notification
            await notificationManager.sendMessageNotification(to: recipientID, from: currentUser.displayName)
            
        } catch {
            if error is CryptoKitError {
                self.error = .messageEncryptionFailed
            } else {
                self.error = .networkError(error.localizedDescription)
            }
            throw error
        }
    }
    
    func loadConversations() async {
        guard let currentUser = currentUser else { return }
        
        do {
            conversations = try await cloudKitManager.fetchConversations(for: currentUser.userID)
        } catch {
            self.error = .networkError(error.localizedDescription)
        }
    }
    
    func loadMessages(for conversationID: UUID) async -> [DirectMessage] {
        do {
            let messages = try await cloudKitManager.fetchMessages(for: conversationID)
            
            // Decrypt messages
            return messages.compactMap { message in
                guard message.isEncrypted else { return message }
                
                do {
                    let decryptedContent = try encryptionManager.decrypt(message.content)
                    var decryptedMessage = message
                    decryptedMessage.content = decryptedContent
                    return decryptedMessage
                } catch {
                    return nil
                }
            }
        } catch {
            self.error = .networkError(error.localizedDescription)
            return []
        }
    }
    
    // MARK: - Recommendations
    
    func getRecommendedGroups() async -> [SupportGroup] {
        guard let currentUser = currentUser else { return [] }
        
        return await recommendationEngine.recommendGroups(for: currentUser, availableGroups: availableGroups)
    }
    
    func getRecommendedPosts() async -> [SupportPost] {
        guard let currentUser = currentUser else { return [] }
        
        return await recommendationEngine.recommendPosts(for: currentUser, allPosts: recentPosts)
    }
    
    func getSimilarExperiences(for post: SupportPost) async -> [SupportPost] {
        return await recommendationEngine.findSimilarExperiences(to: post, in: recentPosts)
    }
    
    // MARK: - Helper Methods
    
    private func updateUserStats(for userID: String, action: UserAction) async {
        // Update user statistics based on actions
        // This would update the user's SupportStats
    }
    
    func reportContent(contentID: UUID, reason: String) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        try await cloudKitManager.reportContent(contentID: contentID, reportedBy: currentUser.userID, reason: reason)
    }
    
    func blockUser(_ userID: String) async throws {
        guard let currentUser = currentUser else {
            throw SocialSupportError.authenticationRequired
        }
        
        try await cloudKitManager.blockUser(blockedUserID: userID, blockedBy: currentUser.userID)
    }
}

// MARK: - Supporting Classes

class CloudKitManager {
    let groupUpdates = PassthroughSubject<[SupportGroup], Never>()
    let postUpdates = PassthroughSubject<[SupportPost], Never>()
    
    func saveUserProfile(_ profile: UserProfile) async throws {
        // CloudKit implementation
    }
    
    func updateUserProfile(_ profile: UserProfile) async throws {
        // CloudKit implementation
    }
    
    func fetchUserProfile(userID: String) async throws -> UserProfile {
        // CloudKit implementation
        throw SocialSupportError.userNotFound
    }
    
    func fetchAvailableGroups() async throws -> [SupportGroup] {
        // CloudKit implementation
        return []
    }
    
    func searchGroups(query: String, category: GroupCategory?) async throws -> [SupportGroup] {
        // CloudKit implementation
        return []
    }
    
    func joinGroup(groupID: UUID, userID: String) async throws {
        // CloudKit implementation
    }
    
    func leaveGroup(groupID: UUID, userID: String) async throws {
        // CloudKit implementation
    }
    
    func createGroup(_ group: SupportGroup) async throws {
        // CloudKit implementation
    }
    
    func createPost(_ post: SupportPost) async throws {
        // CloudKit implementation
    }
    
    func fetchGroupPosts(groupID: UUID, limit: Int) async throws -> [SupportPost] {
        // CloudKit implementation
        return []
    }
    
    func addReaction(_ reaction: PostReaction, to postID: UUID) async throws {
        // CloudKit implementation
    }
    
    func addComment(_ comment: PostComment) async throws {
        // CloudKit implementation
    }
    
    func sendDirectMessage(_ message: DirectMessage) async throws {
        // CloudKit implementation
    }
    
    func fetchConversations(for userID: String) async throws -> [Conversation] {
        // CloudKit implementation
        return []
    }
    
    func fetchMessages(for conversationID: UUID) async throws -> [DirectMessage] {
        // CloudKit implementation
        return []
    }
    
    func reportContent(contentID: UUID, reportedBy: String, reason: String) async throws {
        // CloudKit implementation
    }
    
    func blockUser(blockedUserID: String, blockedBy: String) async throws {
        // CloudKit implementation
    }
}

class EncryptionManager {
    func encrypt(_ content: String) throws -> String {
        // End-to-end encryption implementation
        return content // Placeholder
    }
    
    func decrypt(_ encryptedContent: String) throws -> String {
        // Decryption implementation
        return encryptedContent // Placeholder
    }
}

struct ModerationResult {
    let isApproved: Bool
    let requiresReview: Bool
    let reason: String
    let confidence: Double
}

class ModerationEngine {
    func moderateContent(_ content: String) async -> ModerationResult {
        // AI-powered content moderation
        // Check for inappropriate content, spam, etc.
        return ModerationResult(
            isApproved: true,
            requiresReview: false,
            reason: "",
            confidence: 0.95
        )
    }
}

class RecommendationEngine {
    func recommendGroups(for user: UserProfile, availableGroups: [SupportGroup]) async -> [SupportGroup] {
        // ML-based group recommendations
        return Array(availableGroups.prefix(5))
    }
    
    func recommendPosts(for user: UserProfile, allPosts: [SupportPost]) async -> [SupportPost] {
        // ML-based post recommendations
        return Array(allPosts.prefix(10))
    }
    
    func findSimilarExperiences(to post: SupportPost, in posts: [SupportPost]) async -> [SupportPost] {
        // Find posts with similar content/experiences
        return Array(posts.filter { $0.id != post.id }.prefix(5))
    }
}

struct SocialNotification: Identifiable, Codable {
    let id = UUID()
    let type: NotificationType
    let title: String
    let message: String
    let userID: String
    let createdDate: Date
    let isRead: Bool
    let actionURL: URL?
}

enum NotificationType: String, CaseIterable, Codable {
    case newPost = "new_post"
    case newComment = "new_comment"
    case newReaction = "new_reaction"
    case newMessage = "new_message"
    case groupInvite = "group_invite"
    case mention = "mention"
    case welcome = "welcome"
    case achievement = "achievement"
    case reminder = "reminder"
}

class SocialNotificationManager {
    func sendWelcomeNotification(to userID: String, group: SupportGroup) async {
        // Send welcome notification
    }
    
    func sendMessageNotification(to userID: String, from senderName: String) async {
        // Send message notification
    }
}

enum UserAction {
    case postCreated
    case commentPosted
    case reactionGiven
    case reactionReceived
    case helpfulVote
}