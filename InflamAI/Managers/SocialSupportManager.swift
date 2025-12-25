//
//  SocialSupportManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-20.
//

import Foundation
import SwiftUI
import Combine

@MainActor
class SocialSupportManager: ObservableObject {
    static let shared = SocialSupportManager()
    
    // MARK: - Published Properties
    
    @Published var currentUserProfile: UserProfile?
    @Published var joinedGroups: [SupportGroup] = []
    @Published var availableGroups: [SupportGroup] = []
    @Published var suggestedGroups: [SupportGroup] = []
    @Published var featuredGroups: [SupportGroup] = []
    @Published var popularGroups: [SupportGroup] = []
    @Published var newGroups: [SupportGroup] = []
    
    @Published var conversations: [Conversation] = []
    @Published var recentPosts: [SupportPost] = []
    @Published var userRecentPosts: [SupportPost] = []
    
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    // MARK: - Private Properties
    
    private var cancellables = Set<AnyCancellable>()
    private let networkManager = NetworkManager.shared
    
    // MARK: - Initialization
    
    private init() {
        setupMockData()
        loadCurrentUserProfile()
    }
    
    // MARK: - User Profile Management
    
    func loadCurrentUserProfile() {
        // Mock current user profile
        currentUserProfile = UserProfile(
            id: "current-user-123",
            displayName: "Sarah Johnson",
            bio: "Living with RA for 5 years. Passionate about helping others on their journey.",
            joinDate: Calendar.current.date(byAdding: .year, value: -2, to: Date()) ?? Date(),
            supportStats: SupportStats(
                totalPosts: 47,
                helpfulReactions: 156,
                commentsGiven: 89,
                groupsJoined: 4
            ),
            badges: [
                UserBadge(
                    id: "helpful-member",
                    title: "Helpful Member",
                    description: "Received 100+ helpful reactions",
                    systemImage: "hand.thumbsup.circle.fill",
                    color: .blue,
                    earnedDate: Date()
                ),
                UserBadge(
                    id: "active-contributor",
                    title: "Active Contributor",
                    description: "Posted 25+ times",
                    systemImage: "pencil.circle.fill",
                    color: .green,
                    earnedDate: Date()
                ),
                UserBadge(
                    id: "community-builder",
                    title: "Community Builder",
                    description: "Joined 3+ groups",
                    systemImage: "person.3.fill",
                    color: .orange,
                    earnedDate: Date()
                )
            ],
            preferences: UserPreferences(
                allowDirectMessages: true,
                showOnlineStatus: true,
                emailNotifications: true,
                pushNotifications: true
            )
        )
    }
    
    func updateUserProfile(_ profile: UserProfile) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 1_000_000_000)
        
        currentUserProfile = profile
    }
    
    // MARK: - Groups Management
    
    func loadAvailableGroups() async {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 500_000_000)
        
        // Mock data is already loaded in setupMockData()
    }
    
    func loadSuggestedGroups() async {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 500_000_000)
        
        suggestedGroups = [
            availableGroups[1], // RA Warriors
            availableGroups[3]  // Mindful Living
        ]
    }
    
    func loadFeaturedGroups() async {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 500_000_000)
        
        featuredGroups = [
            availableGroups[0], // Daily Support Circle
            availableGroups[2]  // Exercise & Wellness
        ]
    }
    
    func loadPopularGroups() async {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 500_000_000)
        
        popularGroups = [
            availableGroups[1], // RA Warriors
            availableGroups[4]  // Medication Support
        ]
    }
    
    func loadNewGroups() async {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 500_000_000)
        
        newGroups = [
            availableGroups[5], // Young Adults with RA
            availableGroups[6]  // Working with Chronic Pain
        ]
    }
    
    func joinGroup(_ group: SupportGroup) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 1_000_000_000)
        
        if !joinedGroups.contains(where: { $0.id == group.id }) {
            joinedGroups.append(group)
        }
    }
    
    func leaveGroup(_ group: SupportGroup) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 1_000_000_000)
        
        joinedGroups.removeAll { $0.id == group.id }
    }
    
    func createGroup(_ group: SupportGroup) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 1_500_000_000)
        
        availableGroups.append(group)
        joinedGroups.append(group)
    }
    
    // MARK: - Posts Management
    
    func loadRecentPosts(for groupID: String? = nil) async {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 500_000_000)
        
        if let groupID = groupID {
            recentPosts = recentPosts.filter { $0.groupID == groupID }
        }
    }
    
    func loadUserRecentPosts() async {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 500_000_000)
        
        userRecentPosts = recentPosts.filter { $0.authorID == currentUserProfile?.id }
    }
    
    func createPost(_ post: SupportPost) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 1_000_000_000)
        
        recentPosts.insert(post, at: 0)
        
        if post.authorID == currentUserProfile?.id {
            userRecentPosts.insert(post, at: 0)
        }
    }
    
    func reactToPost(_ post: SupportPost, reaction: ReactionType) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 500_000_000)
        
        if let index = recentPosts.firstIndex(where: { $0.id == post.id }) {
            let userID = currentUserProfile?.id ?? ""
            var updatedPost = recentPosts[index]
            
            // Remove existing reaction from this user
            updatedPost.reactions.removeAll { $0.userID == userID }
            
            // Add new reaction
            let newReaction = PostReaction(
                id: UUID().uuidString,
                userID: userID,
                userName: currentUserProfile?.displayName ?? "Anonymous",
                type: reaction,
                timestamp: Date()
            )
            updatedPost.reactions.append(newReaction)
            
            recentPosts[index] = updatedPost
        }
    }
    
    func addComment(to post: SupportPost, content: String) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 1_000_000_000)
        
        if let index = recentPosts.firstIndex(where: { $0.id == post.id }) {
            var updatedPost = recentPosts[index]
            
            let comment = PostComment(
                id: UUID().uuidString,
                authorID: currentUserProfile?.id ?? "",
                authorName: currentUserProfile?.displayName ?? "Anonymous",
                content: content,
                timestamp: Date(),
                reactions: []
            )
            
            updatedPost.comments.append(comment)
            recentPosts[index] = updatedPost
        }
    }
    
    // MARK: - Conversations Management
    
    func loadConversations() async {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 500_000_000)
        
        // Mock data is already loaded in setupMockData()
    }
    
    func startConversation(with userID: String) async throws -> Conversation {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 1_000_000_000)
        
        let conversation = Conversation(
            id: UUID().uuidString,
            participants: [
                ConversationParticipant(
                    id: currentUserProfile?.id ?? "",
                    name: currentUserProfile?.displayName ?? "You",
                    isOnline: true
                ),
                ConversationParticipant(
                    id: userID,
                    name: "New Contact",
                    isOnline: false
                )
            ],
            lastMessage: nil,
            hasUnreadMessages: false,
            isGroup: false,
            createdDate: Date()
        )
        
        conversations.insert(conversation, at: 0)
        return conversation
    }
    
    func sendMessage(to conversationID: String, content: String) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try await Task.sleep(nanoseconds: 500_000_000)
        
        if let index = conversations.firstIndex(where: { $0.id == conversationID }) {
            var updatedConversation = conversations[index]
            
            let message = ConversationMessage(
                id: UUID().uuidString,
                senderID: currentUserProfile?.id ?? "",
                senderName: currentUserProfile?.displayName ?? "You",
                content: content,
                timestamp: Date(),
                isRead: false
            )
            
            updatedConversation.lastMessage = message
            conversations[index] = updatedConversation
            
            // Move to top of list
            conversations.remove(at: index)
            conversations.insert(updatedConversation, at: 0)
        }
    }
    
    // MARK: - Search and Discovery
    
    func searchGroups(query: String) async -> [SupportGroup] {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 300_000_000)
        
        return availableGroups.filter {
            $0.name.localizedCaseInsensitiveContains(query) ||
            $0.description.localizedCaseInsensitiveContains(query) ||
            $0.tags.contains { $0.localizedCaseInsensitiveContains(query) }
        }
    }
    
    func searchPosts(query: String) async -> [SupportPost] {
        isLoading = true
        defer { isLoading = false }
        
        // Simulate API call
        try? await Task.sleep(nanoseconds: 300_000_000)
        
        return recentPosts.filter {
            $0.content.localizedCaseInsensitiveContains(query) ||
            $0.tags.contains { $0.localizedCaseInsensitiveContains(query) }
        }
    }
    
    // MARK: - Mock Data Setup
    
    private func setupMockData() {
        setupMockGroups()
        setupMockPosts()
        setupMockConversations()
    }
    
    private func setupMockGroups() {
        availableGroups = [
            SupportGroup(
                id: "group-1",
                name: "Daily Support Circle",
                description: "A welcoming space for daily check-ins, sharing experiences, and mutual support among people living with rheumatoid arthritis.",
                category: .general,
                privacy: .public,
                memberCount: 1247,
                activityLevel: .high,
                tags: ["daily-support", "check-ins", "community"],
                guidelines: "Be respectful, supportive, and kind to all members.",
                createdDate: Calendar.current.date(byAdding: .month, value: -8, to: Date()) ?? Date(),
                location: nil,
                meetingSchedule: nil,
                resources: []
            ),
            SupportGroup(
                id: "group-2",
                name: "RA Warriors",
                description: "For those who refuse to let RA define them. Share your victories, challenges, and strategies for living your best life.",
                category: .motivation,
                privacy: .public,
                memberCount: 892,
                activityLevel: .high,
                tags: ["motivation", "warriors", "strength"],
                guidelines: "Focus on positivity and empowerment.",
                createdDate: Calendar.current.date(byAdding: .month, value: -6, to: Date()) ?? Date(),
                location: nil,
                meetingSchedule: nil,
                resources: []
            ),
            SupportGroup(
                id: "group-3",
                name: "Exercise & Wellness",
                description: "Sharing safe exercise routines, wellness tips, and physical therapy experiences for people with RA.",
                category: .exercise,
                privacy: .public,
                memberCount: 634,
                activityLevel: .medium,
                tags: ["exercise", "wellness", "physical-therapy"],
                guidelines: "Always consult healthcare providers before trying new exercises.",
                createdDate: Calendar.current.date(byAdding: .month, value: -4, to: Date()) ?? Date(),
                location: nil,
                meetingSchedule: nil,
                resources: []
            ),
            SupportGroup(
                id: "group-4",
                name: "Mindful Living",
                description: "Exploring mindfulness, meditation, and mental health strategies for managing chronic illness.",
                category: .mentalHealth,
                privacy: .public,
                memberCount: 445,
                activityLevel: .medium,
                tags: ["mindfulness", "meditation", "mental-health"],
                guidelines: "Create a safe space for mental health discussions.",
                createdDate: Calendar.current.date(byAdding: .month, value: -3, to: Date()) ?? Date(),
                location: nil,
                meetingSchedule: nil,
                resources: []
            ),
            SupportGroup(
                id: "group-5",
                name: "Medication Support",
                description: "Discussing medications, side effects, and treatment experiences. Share what works and what doesn't.",
                category: .medical,
                privacy: .public,
                memberCount: 756,
                activityLevel: .high,
                tags: ["medication", "treatment", "side-effects"],
                guidelines: "Share experiences, not medical advice.",
                createdDate: Calendar.current.date(byAdding: .month, value: -5, to: Date()) ?? Date(),
                location: nil,
                meetingSchedule: nil,
                resources: []
            ),
            SupportGroup(
                id: "group-6",
                name: "Young Adults with RA",
                description: "A community for young adults (18-35) navigating RA while building careers, relationships, and futures.",
                category: .ageSpecific,
                privacy: .public,
                memberCount: 289,
                activityLevel: .medium,
                tags: ["young-adults", "career", "relationships"],
                guidelines: "Age-appropriate discussions for young adults.",
                createdDate: Calendar.current.date(byAdding: .week, value: -2, to: Date()) ?? Date(),
                location: nil,
                meetingSchedule: nil,
                resources: []
            ),
            SupportGroup(
                id: "group-7",
                name: "Working with Chronic Pain",
                description: "Strategies for maintaining employment and productivity while managing chronic pain and fatigue.",
                category: .lifestyle,
                privacy: .public,
                memberCount: 412,
                activityLevel: .low,
                tags: ["work", "productivity", "chronic-pain"],
                guidelines: "Focus on practical workplace strategies.",
                createdDate: Calendar.current.date(byAdding: .day, value: -5, to: Date()) ?? Date(),
                location: nil,
                meetingSchedule: nil,
                resources: []
            )
        ]
        
        // Set some groups as joined
        joinedGroups = [
            availableGroups[0], // Daily Support Circle
            availableGroups[1], // RA Warriors
            availableGroups[2]  // Exercise & Wellness
        ]
    }
    
    private func setupMockPosts() {
        recentPosts = [
            SupportPost(
                id: "post-1",
                groupID: "group-1",
                authorID: "user-456",
                authorName: "Emma Chen",
                content: "Good morning everyone! Starting my day with some gentle stretching. The weather change has my joints feeling stiff, but I'm determined to stay positive. How is everyone doing today?",
                type: .general,
                isAnonymous: false,
                mood: .okay,
                painLevel: 4,
                tags: ["morning", "stretching", "weather"],
                attachments: [],
                reactions: [
                    PostReaction(id: "r1", userID: "user-123", userName: "Sarah J.", type: .heart, timestamp: Date()),
                    PostReaction(id: "r2", userID: "user-789", userName: "Mike R.", type: .thumbsUp, timestamp: Date())
                ],
                comments: [
                    PostComment(
                        id: "c1",
                        authorID: "user-123",
                        authorName: "Sarah J.",
                        content: "Hope your day gets better! Weather changes are tough.",
                        timestamp: Date(),
                        reactions: []
                    )
                ],
                createdDate: Calendar.current.date(byAdding: .hour, value: -2, to: Date()) ?? Date(),
                updatedDate: Calendar.current.date(byAdding: .hour, value: -2, to: Date()) ?? Date(),
                isPinned: false,
                location: nil
            ),
            SupportPost(
                id: "post-2",
                groupID: "group-2",
                authorID: "user-789",
                authorName: "Michael Rodriguez",
                content: "Celebrating a small victory today! I was able to walk for 30 minutes without significant pain. It's been months since I could do that. Sometimes the little wins mean everything. 💪",
                type: .celebration,
                isAnonymous: false,
                mood: .great,
                painLevel: 2,
                tags: ["victory", "walking", "progress"],
                attachments: [],
                reactions: [
                    PostReaction(id: "r3", userID: "user-456", userName: "Emma C.", type: .celebrate, timestamp: Date()),
                    PostReaction(id: "r4", userID: "user-123", userName: "Sarah J.", type: .heart, timestamp: Date())
                ],
                comments: [],
                createdDate: Calendar.current.date(byAdding: .hour, value: -4, to: Date()) ?? Date(),
                updatedDate: Calendar.current.date(byAdding: .hour, value: -4, to: Date()) ?? Date(),
                isPinned: false,
                location: nil
            ),
            SupportPost(
                id: "post-3",
                groupID: "group-3",
                authorID: "user-321",
                authorName: "Anonymous",
                content: "Has anyone tried water aerobics? My rheumatologist suggested it, but I'm nervous about starting something new. Would love to hear about your experiences.",
                type: .question,
                isAnonymous: true,
                mood: .anxious,
                painLevel: 6,
                tags: ["water-aerobics", "exercise", "advice"],
                attachments: [],
                reactions: [],
                comments: [
                    PostComment(
                        id: "c2",
                        authorID: "user-456",
                        authorName: "Emma C.",
                        content: "I love water aerobics! The buoyancy really helps with joint pain. Start slow and listen to your body.",
                        timestamp: Date(),
                        reactions: []
                    )
                ],
                createdDate: Calendar.current.date(byAdding: .hour, value: -6, to: Date()) ?? Date(),
                updatedDate: Calendar.current.date(byAdding: .hour, value: -6, to: Date()) ?? Date(),
                isPinned: false,
                location: nil
            )
        ]
        
        // Set user's recent posts
        userRecentPosts = [
            SupportPost(
                id: "user-post-1",
                groupID: "group-1",
                authorID: "current-user-123",
                authorName: "Sarah Johnson",
                content: "Thank you all for the warm welcome! This community already feels like home. Looking forward to sharing this journey with you all.",
                type: .general,
                isAnonymous: false,
                mood: .happy,
                painLevel: 3,
                tags: ["welcome", "community", "gratitude"],
                attachments: [],
                reactions: [
                    PostReaction(id: "r5", userID: "user-456", userName: "Emma C.", type: .heart, timestamp: Date())
                ],
                comments: [],
                createdDate: Calendar.current.date(byAdding: .day, value: -1, to: Date()) ?? Date(),
                updatedDate: Calendar.current.date(byAdding: .day, value: -1, to: Date()) ?? Date(),
                isPinned: false,
                location: nil
            )
        ]
    }
    
    private func setupMockConversations() {
        conversations = [
            Conversation(
                id: "conv-1",
                participants: [
                    ConversationParticipant(
                        id: "current-user-123",
                        name: "You",
                        isOnline: true
                    ),
                    ConversationParticipant(
                        id: "user-456",
                        name: "Emma Chen",
                        isOnline: true
                    )
                ],
                lastMessage: ConversationMessage(
                    id: "msg-1",
                    senderID: "user-456",
                    senderName: "Emma Chen",
                    content: "Thanks for the encouragement in the group today! It really helped.",
                    timestamp: Calendar.current.date(byAdding: .minute, value: -15, to: Date()) ?? Date(),
                    isRead: false
                ),
                hasUnreadMessages: true,
                isGroup: false,
                createdDate: Calendar.current.date(byAdding: .day, value: -3, to: Date()) ?? Date()
            ),
            Conversation(
                id: "conv-2",
                participants: [
                    ConversationParticipant(
                        id: "current-user-123",
                        name: "You",
                        isOnline: true
                    ),
                    ConversationParticipant(
                        id: "user-789",
                        name: "Michael Rodriguez",
                        isOnline: false
                    )
                ],
                lastMessage: ConversationMessage(
                    id: "msg-2",
                    senderID: "current-user-123",
                    senderName: "You",
                    content: "Congratulations on your walking milestone! That's amazing progress.",
                    timestamp: Calendar.current.date(byAdding: .hour, value: -2, to: Date()) ?? Date(),
                    isRead: true
                ),
                hasUnreadMessages: false,
                isGroup: false,
                createdDate: Calendar.current.date(byAdding: .day, value: -1, to: Date()) ?? Date()
            )
        ]
    }
}

// MARK: - Error Types

enum SocialSupportError: LocalizedError {
    case networkError
    case invalidData
    case unauthorized
    case groupNotFound
    case postNotFound
    case conversationNotFound
    
    var errorDescription: String? {
        switch self {
        case .networkError:
            return "Network connection error. Please check your internet connection."
        case .invalidData:
            return "Invalid data received from server."
        case .unauthorized:
            return "You are not authorized to perform this action."
        case .groupNotFound:
            return "The requested group could not be found."
        case .postNotFound:
            return "The requested post could not be found."
        case .conversationNotFound:
            return "The requested conversation could not be found."
        }
    }
}