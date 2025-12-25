//
//  CommunityManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import CoreLocation
import Combine
import os.log

// MARK: - Community Manager
class CommunityManager: ObservableObject {
    static let shared = CommunityManager()
    
    // MARK: - Properties
    @Published var currentUser: CommunityUser?
    @Published var supportGroups: [SupportGroup] = []
    @Published var nearbyGroups: [SupportGroup] = []
    @Published var posts: [CommunityPost] = []
    @Published var conversations: [Conversation] = []
    @Published var notifications: [CommunityNotification] = []
    @Published var isConnected = false
    @Published var connectionStatus: ConnectionStatus = .disconnected
    @Published var moderationQueue: [ModerationItem] = []
    @Published var reportedContent: [ReportedContent] = []
    
    private let networkManager = CommunityNetworkManager()
    private let locationManager = CLLocationManager()
    private let contentModerator = ContentModerator()
    private let privacyManager = CommunityPrivacyManager()
    private let analyticsManager = CommunityAnalyticsManager()
    private let logger = Logger(subsystem: "com.inflamai.community", category: "CommunityManager")
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    private init() {
        setupLocationManager()
        setupNetworkMonitoring()
        loadUserProfile()
        loadCommunityData()
    }
    
    // MARK: - Setup
    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyKilometer
    }
    
    private func setupNetworkMonitoring() {
        networkManager.connectionStatusPublisher
            .receive(on: DispatchQueue.main)
            .assign(to: \.$connectionStatus, on: self)
            .store(in: &cancellables)
        
        networkManager.isConnectedPublisher
            .receive(on: DispatchQueue.main)
            .assign(to: \.$isConnected, on: self)
            .store(in: &cancellables)
    }
    
    private func loadUserProfile() {
        // Load anonymous user profile
        if let savedProfile = UserDefaults.standard.data(forKey: "community_user_profile"),
           let profile = try? JSONDecoder().decode(CommunityUser.self, from: savedProfile) {
            self.currentUser = profile
        }
    }
    
    private func loadCommunityData() {
        Task {
            await loadSupportGroups()
            await loadRecentPosts()
            await loadConversations()
        }
    }
    
    // MARK: - User Management
    func createAnonymousProfile(demographics: UserDemographics, interests: [CommunityInterest]) async throws -> CommunityUser {
        let anonymousId = UUID().uuidString
        let hashedId = privacyManager.hashUserIdentifier(anonymousId)
        
        let user = CommunityUser(
            id: hashedId,
            anonymousId: anonymousId,
            demographics: demographics,
            interests: interests,
            joinDate: Date(),
            reputation: 0,
            isVerified: false,
            privacySettings: CommunityPrivacySettings()
        )
        
        try await networkManager.createUser(user)
        
        self.currentUser = user
        saveUserProfile(user)
        
        analyticsManager.trackUserCreation(demographics: demographics)
        logger.info("Anonymous user profile created")
        
        return user
    }
    
    func updateUserProfile(demographics: UserDemographics?, interests: [CommunityInterest]?) async throws {
        guard var user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        if let demographics = demographics {
            user.demographics = demographics
        }
        
        if let interests = interests {
            user.interests = interests
        }
        
        user.lastUpdated = Date()
        
        try await networkManager.updateUser(user)
        
        self.currentUser = user
        saveUserProfile(user)
        
        logger.info("User profile updated")
    }
    
    func updatePrivacySettings(_ settings: CommunityPrivacySettings) async throws {
        guard var user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        user.privacySettings = settings
        user.lastUpdated = Date()
        
        try await networkManager.updateUser(user)
        
        self.currentUser = user
        saveUserProfile(user)
        
        logger.info("Privacy settings updated")
    }
    
    // MARK: - Support Groups
    func findSupportGroups(location: CLLocation? = nil, radius: Double = 50000) async throws -> [SupportGroup] {
        let groups = try await networkManager.findSupportGroups(
            location: location,
            radius: radius,
            interests: currentUser?.interests ?? []
        )
        
        DispatchQueue.main.async {
            if location != nil {
                self.nearbyGroups = groups
            } else {
                self.supportGroups = groups
            }
        }
        
        analyticsManager.trackGroupSearch(resultCount: groups.count, hasLocation: location != nil)
        logger.info("Found \(groups.count) support groups")
        
        return groups
    }
    
    func createSupportGroup(_ group: SupportGroup) async throws -> SupportGroup {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        var newGroup = group
        newGroup.createdBy = user.id
        newGroup.createdDate = Date()
        newGroup.moderators = [user.id]
        newGroup.members = [user.id]
        newGroup.memberCount = 1
        
        let createdGroup = try await networkManager.createSupportGroup(newGroup)
        
        DispatchQueue.main.async {
            self.supportGroups.append(createdGroup)
        }
        
        analyticsManager.trackGroupCreation(type: group.type, isVirtual: group.isVirtual)
        logger.info("Support group created: \(group.name)")
        
        return createdGroup
    }
    
    func joinSupportGroup(_ groupId: String) async throws {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        try await networkManager.joinSupportGroup(groupId: groupId, userId: user.id)
        
        // Update local data
        if let index = supportGroups.firstIndex(where: { $0.id == groupId }) {
            DispatchQueue.main.async {
                self.supportGroups[index].members.append(user.id)
                self.supportGroups[index].memberCount += 1
            }
        }
        
        analyticsManager.trackGroupJoin(groupId: groupId)
        logger.info("Joined support group: \(groupId)")
    }
    
    func leaveSupportGroup(_ groupId: String) async throws {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        try await networkManager.leaveSupportGroup(groupId: groupId, userId: user.id)
        
        // Update local data
        if let index = supportGroups.firstIndex(where: { $0.id == groupId }) {
            DispatchQueue.main.async {
                self.supportGroups[index].members.removeAll { $0 == user.id }
                self.supportGroups[index].memberCount -= 1
            }
        }
        
        analyticsManager.trackGroupLeave(groupId: groupId)
        logger.info("Left support group: \(groupId)")
    }
    
    func requestLocationPermission() {
        locationManager.requestWhenInUseAuthorization()
    }
    
    // MARK: - Community Posts
    func createPost(_ content: String, type: PostType, groupId: String? = nil, tags: [String] = []) async throws -> CommunityPost {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        // Content moderation
        let moderationResult = await contentModerator.moderateContent(content)
        guard moderationResult.isApproved else {
            throw CommunityError.contentViolation(moderationResult.reason)
        }
        
        let post = CommunityPost(
            id: UUID().uuidString,
            authorId: user.id,
            content: content,
            type: type,
            groupId: groupId,
            tags: tags,
            createdDate: Date(),
            likes: 0,
            replies: 0,
            isAnonymous: user.privacySettings.postAnonymously
        )
        
        let createdPost = try await networkManager.createPost(post)
        
        DispatchQueue.main.async {
            self.posts.insert(createdPost, at: 0)
        }
        
        analyticsManager.trackPostCreation(type: type, hasGroup: groupId != nil)
        logger.info("Post created: \(type.rawValue)")
        
        return createdPost
    }
    
    func likePost(_ postId: String) async throws {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        try await networkManager.likePost(postId: postId, userId: user.id)
        
        // Update local data
        if let index = posts.firstIndex(where: { $0.id == postId }) {
            DispatchQueue.main.async {
                self.posts[index].likes += 1
                self.posts[index].likedByCurrentUser = true
            }
        }
        
        analyticsManager.trackPostInteraction(type: .like, postId: postId)
        logger.info("Post liked: \(postId)")
    }
    
    func replyToPost(_ postId: String, content: String) async throws -> CommunityReply {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        // Content moderation
        let moderationResult = await contentModerator.moderateContent(content)
        guard moderationResult.isApproved else {
            throw CommunityError.contentViolation(moderationResult.reason)
        }
        
        let reply = CommunityReply(
            id: UUID().uuidString,
            postId: postId,
            authorId: user.id,
            content: content,
            createdDate: Date(),
            likes: 0,
            isAnonymous: user.privacySettings.postAnonymously
        )
        
        let createdReply = try await networkManager.createReply(reply)
        
        // Update local data
        if let index = posts.firstIndex(where: { $0.id == postId }) {
            DispatchQueue.main.async {
                self.posts[index].replies += 1
            }
        }
        
        analyticsManager.trackPostInteraction(type: .reply, postId: postId)
        logger.info("Reply created for post: \(postId)")
        
        return createdReply
    }
    
    func reportPost(_ postId: String, reason: ReportReason, details: String? = nil) async throws {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        let report = ContentReport(
            id: UUID().uuidString,
            contentId: postId,
            contentType: .post,
            reportedBy: user.id,
            reason: reason,
            details: details,
            reportDate: Date(),
            status: .pending
        )
        
        try await networkManager.reportContent(report)
        
        analyticsManager.trackContentReport(type: .post, reason: reason)
        logger.info("Post reported: \(postId) for \(reason.rawValue)")
    }
    
    // MARK: - Peer-to-Peer Messaging
    func startConversation(with userId: String, initialMessage: String) async throws -> Conversation {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        // Content moderation
        let moderationResult = await contentModerator.moderateContent(initialMessage)
        guard moderationResult.isApproved else {
            throw CommunityError.contentViolation(moderationResult.reason)
        }
        
        let conversation = Conversation(
            id: UUID().uuidString,
            participants: [user.id, userId],
            createdDate: Date(),
            lastMessageDate: Date(),
            isActive: true
        )
        
        let message = Message(
            id: UUID().uuidString,
            conversationId: conversation.id,
            senderId: user.id,
            content: initialMessage,
            timestamp: Date(),
            isRead: false,
            messageType: .text
        )
        
        let createdConversation = try await networkManager.createConversation(conversation, initialMessage: message)
        
        DispatchQueue.main.async {
            self.conversations.insert(createdConversation, at: 0)
        }
        
        analyticsManager.trackConversationStart()
        logger.info("Conversation started with user: \(userId)")
        
        return createdConversation
    }
    
    func sendMessage(to conversationId: String, content: String, type: MessageType = .text) async throws -> Message {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        // Content moderation for text messages
        if type == .text {
            let moderationResult = await contentModerator.moderateContent(content)
            guard moderationResult.isApproved else {
                throw CommunityError.contentViolation(moderationResult.reason)
            }
        }
        
        let message = Message(
            id: UUID().uuidString,
            conversationId: conversationId,
            senderId: user.id,
            content: content,
            timestamp: Date(),
            isRead: false,
            messageType: type
        )
        
        let sentMessage = try await networkManager.sendMessage(message)
        
        // Update local conversation
        if let index = conversations.firstIndex(where: { $0.id == conversationId }) {
            DispatchQueue.main.async {
                self.conversations[index].lastMessageDate = Date()
                self.conversations[index].messages.append(sentMessage)
            }
        }
        
        analyticsManager.trackMessageSent(type: type)
        logger.info("Message sent to conversation: \(conversationId)")
        
        return sentMessage
    }
    
    func markMessageAsRead(_ messageId: String) async throws {
        try await networkManager.markMessageAsRead(messageId)
        
        // Update local data
        for conversationIndex in conversations.indices {
            if let messageIndex = conversations[conversationIndex].messages.firstIndex(where: { $0.id == messageId }) {
                DispatchQueue.main.async {
                    self.conversations[conversationIndex].messages[messageIndex].isRead = true
                }
                break
            }
        }
    }
    
    // MARK: - Healthcare Provider Portal
    func connectWithProvider(_ providerId: String, accessCode: String) async throws -> HealthcareConnection {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        let connection = try await networkManager.connectWithProvider(
            userId: user.id,
            providerId: providerId,
            accessCode: accessCode
        )
        
        analyticsManager.trackProviderConnection(providerId: providerId)
        logger.info("Connected with healthcare provider: \(providerId)")
        
        return connection
    }
    
    func shareDataWithProvider(_ providerId: String, dataTypes: [SharedDataType], duration: TimeInterval) async throws {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        let sharingAgreement = DataSharingAgreement(
            id: UUID().uuidString,
            userId: user.id,
            providerId: providerId,
            dataTypes: dataTypes,
            startDate: Date(),
            endDate: Date().addingTimeInterval(duration),
            isActive: true
        )
        
        try await networkManager.createDataSharingAgreement(sharingAgreement)
        
        analyticsManager.trackDataSharing(dataTypes: dataTypes, duration: duration)
        logger.info("Data sharing agreement created with provider: \(providerId)")
    }
    
    func revokeProviderAccess(_ providerId: String) async throws {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        try await networkManager.revokeProviderAccess(userId: user.id, providerId: providerId)
        
        analyticsManager.trackProviderAccessRevoked(providerId: providerId)
        logger.info("Provider access revoked: \(providerId)")
    }
    
    // MARK: - Content Moderation
    func moderateContent(_ content: String) async -> ModerationResult {
        return await contentModerator.moderateContent(content)
    }
    
    func reviewModerationQueue() async throws -> [ModerationItem] {
        guard let user = currentUser, user.isModerator else {
            throw CommunityError.insufficientPermissions
        }
        
        let items = try await networkManager.getModerationQueue()
        
        DispatchQueue.main.async {
            self.moderationQueue = items
        }
        
        return items
    }
    
    func approveModerationItem(_ itemId: String) async throws {
        guard let user = currentUser, user.isModerator else {
            throw CommunityError.insufficientPermissions
        }
        
        try await networkManager.approveModerationItem(itemId, moderatorId: user.id)
        
        // Remove from local queue
        DispatchQueue.main.async {
            self.moderationQueue.removeAll { $0.id == itemId }
        }
        
        logger.info("Moderation item approved: \(itemId)")
    }
    
    func rejectModerationItem(_ itemId: String, reason: String) async throws {
        guard let user = currentUser, user.isModerator else {
            throw CommunityError.insufficientPermissions
        }
        
        try await networkManager.rejectModerationItem(itemId, moderatorId: user.id, reason: reason)
        
        // Remove from local queue
        DispatchQueue.main.async {
            self.moderationQueue.removeAll { $0.id == itemId }
        }
        
        logger.info("Moderation item rejected: \(itemId)")
    }
    
    // MARK: - Analytics and Insights
    func getCommunityInsights() async throws -> CommunityInsights {
        let insights = try await networkManager.getCommunityInsights()
        
        analyticsManager.trackInsightsViewed()
        logger.info("Community insights retrieved")
        
        return insights
    }
    
    func getUserEngagementStats() async throws -> UserEngagementStats {
        guard let user = currentUser else {
            throw CommunityError.userNotFound
        }
        
        let stats = try await networkManager.getUserEngagementStats(userId: user.id)
        
        logger.info("User engagement stats retrieved")
        return stats
    }
    
    // MARK: - Private Methods
    private func loadSupportGroups() async {
        do {
            let groups = try await networkManager.getSupportGroups()
            DispatchQueue.main.async {
                self.supportGroups = groups
            }
        } catch {
            logger.error("Failed to load support groups: \(error.localizedDescription)")
        }
    }
    
    private func loadRecentPosts() async {
        do {
            let posts = try await networkManager.getRecentPosts(limit: 50)
            DispatchQueue.main.async {
                self.posts = posts
            }
        } catch {
            logger.error("Failed to load recent posts: \(error.localizedDescription)")
        }
    }
    
    private func loadConversations() async {
        guard let user = currentUser else { return }
        
        do {
            let conversations = try await networkManager.getUserConversations(userId: user.id)
            DispatchQueue.main.async {
                self.conversations = conversations
            }
        } catch {
            logger.error("Failed to load conversations: \(error.localizedDescription)")
        }
    }
    
    private func saveUserProfile(_ user: CommunityUser) {
        if let data = try? JSONEncoder().encode(user) {
            UserDefaults.standard.set(data, forKey: "community_user_profile")
        }
    }
}

// MARK: - CLLocationManagerDelegate
extension CommunityManager: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        
        Task {
            do {
                _ = try await findSupportGroups(location: location)
            } catch {
                logger.error("Failed to find nearby support groups: \(error.localizedDescription)")
            }
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        logger.error("Location manager failed: \(error.localizedDescription)")
    }
    
    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        switch status {
        case .authorizedWhenInUse, .authorizedAlways:
            locationManager.requestLocation()
        case .denied, .restricted:
            logger.info("Location access denied")
        case .notDetermined:
            break
        @unknown default:
            break
        }
    }
}

// MARK: - Supporting Classes
class CommunityNetworkManager {
    private let baseURL = "https://api.inflamai.com/community"
    private let session = URLSession.shared
    
    @Published var connectionStatus: ConnectionStatus = .disconnected
    @Published var isConnected = false
    
    var connectionStatusPublisher: Published<ConnectionStatus>.Publisher { $connectionStatus }
    var isConnectedPublisher: Published<Bool>.Publisher { $isConnected }
    
    // User management
    func createUser(_ user: CommunityUser) async throws {
        // Implementation for creating user
    }
    
    func updateUser(_ user: CommunityUser) async throws {
        // Implementation for updating user
    }
    
    // Support groups
    func findSupportGroups(location: CLLocation?, radius: Double, interests: [CommunityInterest]) async throws -> [SupportGroup] {
        // Implementation for finding support groups
        return []
    }
    
    func getSupportGroups() async throws -> [SupportGroup] {
        // Implementation for getting support groups
        return []
    }
    
    func createSupportGroup(_ group: SupportGroup) async throws -> SupportGroup {
        // Implementation for creating support group
        return group
    }
    
    func joinSupportGroup(groupId: String, userId: String) async throws {
        // Implementation for joining support group
    }
    
    func leaveSupportGroup(groupId: String, userId: String) async throws {
        // Implementation for leaving support group
    }
    
    // Posts and content
    func createPost(_ post: CommunityPost) async throws -> CommunityPost {
        // Implementation for creating post
        return post
    }
    
    func getRecentPosts(limit: Int) async throws -> [CommunityPost] {
        // Implementation for getting recent posts
        return []
    }
    
    func likePost(postId: String, userId: String) async throws {
        // Implementation for liking post
    }
    
    func createReply(_ reply: CommunityReply) async throws -> CommunityReply {
        // Implementation for creating reply
        return reply
    }
    
    func reportContent(_ report: ContentReport) async throws {
        // Implementation for reporting content
    }
    
    // Messaging
    func createConversation(_ conversation: Conversation, initialMessage: Message) async throws -> Conversation {
        // Implementation for creating conversation
        return conversation
    }
    
    func getUserConversations(userId: String) async throws -> [Conversation] {
        // Implementation for getting user conversations
        return []
    }
    
    func sendMessage(_ message: Message) async throws -> Message {
        // Implementation for sending message
        return message
    }
    
    func markMessageAsRead(_ messageId: String) async throws {
        // Implementation for marking message as read
    }
    
    // Healthcare provider portal
    func connectWithProvider(userId: String, providerId: String, accessCode: String) async throws -> HealthcareConnection {
        // Implementation for connecting with provider
        return HealthcareConnection(id: UUID().uuidString, userId: userId, providerId: providerId, connectedDate: Date(), isActive: true)
    }
    
    func createDataSharingAgreement(_ agreement: DataSharingAgreement) async throws {
        // Implementation for creating data sharing agreement
    }
    
    func revokeProviderAccess(userId: String, providerId: String) async throws {
        // Implementation for revoking provider access
    }
    
    // Moderation
    func getModerationQueue() async throws -> [ModerationItem] {
        // Implementation for getting moderation queue
        return []
    }
    
    func approveModerationItem(_ itemId: String, moderatorId: String) async throws {
        // Implementation for approving moderation item
    }
    
    func rejectModerationItem(_ itemId: String, moderatorId: String, reason: String) async throws {
        // Implementation for rejecting moderation item
    }
    
    // Analytics
    func getCommunityInsights() async throws -> CommunityInsights {
        // Implementation for getting community insights
        return CommunityInsights(totalUsers: 0, activeUsers: 0, totalPosts: 0, totalGroups: 0, engagementRate: 0.0)
    }
    
    func getUserEngagementStats(userId: String) async throws -> UserEngagementStats {
        // Implementation for getting user engagement stats
        return UserEngagementStats(postsCreated: 0, repliesCreated: 0, likesReceived: 0, groupsJoined: 0, messagesExchanged: 0)
    }
}

class ContentModerator {
    private let bannedWords = ["spam", "inappropriate", "offensive"] // Simplified list
    
    func moderateContent(_ content: String) async -> ModerationResult {
        let lowercaseContent = content.lowercased()
        
        // Check for banned words
        for word in bannedWords {
            if lowercaseContent.contains(word) {
                return ModerationResult(isApproved: false, reason: "Content contains inappropriate language")
            }
        }
        
        // Check content length
        if content.count > 5000 {
            return ModerationResult(isApproved: false, reason: "Content exceeds maximum length")
        }
        
        // Additional AI-based moderation would go here
        
        return ModerationResult(isApproved: true, reason: nil)
    }
}

class CommunityPrivacyManager {
    func hashUserIdentifier(_ identifier: String) -> String {
        // Simple hash implementation
        return identifier.hash.description
    }
    
    func anonymizeUserData(_ user: CommunityUser) -> CommunityUser {
        var anonymizedUser = user
        anonymizedUser.demographics.age = nil
        anonymizedUser.demographics.location = nil
        return anonymizedUser
    }
}

class CommunityAnalyticsManager {
    private let logger = Logger(subsystem: "com.inflamai.community", category: "Analytics")
    
    func trackUserCreation(demographics: UserDemographics) {
        logger.info("User created with demographics")
    }
    
    func trackGroupSearch(resultCount: Int, hasLocation: Bool) {
        logger.info("Group search performed: \(resultCount) results, location: \(hasLocation)")
    }
    
    func trackGroupCreation(type: SupportGroupType, isVirtual: Bool) {
        logger.info("Group created: \(type.rawValue), virtual: \(isVirtual)")
    }
    
    func trackGroupJoin(groupId: String) {
        logger.info("Group joined: \(groupId)")
    }
    
    func trackGroupLeave(groupId: String) {
        logger.info("Group left: \(groupId)")
    }
    
    func trackPostCreation(type: PostType, hasGroup: Bool) {
        logger.info("Post created: \(type.rawValue), in group: \(hasGroup)")
    }
    
    func trackPostInteraction(type: InteractionType, postId: String) {
        logger.info("Post interaction: \(type.rawValue) on \(postId)")
    }
    
    func trackContentReport(type: ContentType, reason: ReportReason) {
        logger.info("Content reported: \(type.rawValue) for \(reason.rawValue)")
    }
    
    func trackConversationStart() {
        logger.info("Conversation started")
    }
    
    func trackMessageSent(type: MessageType) {
        logger.info("Message sent: \(type.rawValue)")
    }
    
    func trackProviderConnection(providerId: String) {
        logger.info("Provider connected: \(providerId)")
    }
    
    func trackDataSharing(dataTypes: [SharedDataType], duration: TimeInterval) {
        logger.info("Data sharing: \(dataTypes.count) types for \(duration) seconds")
    }
    
    func trackProviderAccessRevoked(providerId: String) {
        logger.info("Provider access revoked: \(providerId)")
    }
    
    func trackInsightsViewed() {
        logger.info("Community insights viewed")
    }
}

// MARK: - Supporting Types
enum ConnectionStatus {
    case disconnected
    case connecting
    case connected
    case error(String)
}

enum CommunityInterest: String, CaseIterable, Codable {
    case rheumatoidArthritis = "Rheumatoid Arthritis"
    case osteoarthritis = "Osteoarthritis"
    case fibromyalgia = "Fibromyalgia"
    case lupus = "Lupus"
    case psoriasis = "Psoriasis"
    case ankylosingSpondylitis = "Ankylosing Spondylitis"
    case chronicPain = "Chronic Pain"
    case mentalHealth = "Mental Health"
    case nutrition = "Nutrition"
    case exercise = "Exercise"
    case medication = "Medication"
    case lifestyle = "Lifestyle"
}

enum SupportGroupType: String, CaseIterable, Codable {
    case general = "General Support"
    case diseaseSpecific = "Disease Specific"
    case ageSpecific = "Age Specific"
    case localMeetup = "Local Meetup"
    case onlineSupport = "Online Support"
    case professionalLed = "Professional Led"
}

enum PostType: String, CaseIterable, Codable {
    case question = "Question"
    case experience = "Experience"
    case tip = "Tip"
    case support = "Support"
    case celebration = "Celebration"
    case resource = "Resource"
}

enum MessageType: String, CaseIterable, Codable {
    case text = "Text"
    case image = "Image"
    case voice = "Voice"
    case document = "Document"
    case location = "Location"
}

enum ReportReason: String, CaseIterable, Codable {
    case spam = "Spam"
    case harassment = "Harassment"
    case inappropriateContent = "Inappropriate Content"
    case misinformation = "Misinformation"
    case privacy = "Privacy Violation"
    case other = "Other"
}

enum ContentType: String, CaseIterable, Codable {
    case post = "Post"
    case reply = "Reply"
    case message = "Message"
    case profile = "Profile"
}

enum InteractionType: String, CaseIterable, Codable {
    case like = "Like"
    case reply = "Reply"
    case share = "Share"
    case report = "Report"
}

enum SharedDataType: String, CaseIterable, Codable {
    case painLevels = "Pain Levels"
    case medications = "Medications"
    case symptoms = "Symptoms"
    case activities = "Activities"
    case mood = "Mood"
    case sleep = "Sleep"
    case vitals = "Vitals"
}

struct CommunityUser: Codable {
    let id: String
    let anonymousId: String
    var demographics: UserDemographics
    var interests: [CommunityInterest]
    let joinDate: Date
    var lastUpdated: Date?
    var reputation: Int
    var isVerified: Bool
    var isModerator: Bool = false
    var privacySettings: CommunityPrivacySettings
}

struct UserDemographics: Codable {
    var age: Int?
    var gender: String?
    var location: String?
    var diagnosisYear: Int?
    var primaryCondition: String?
    var secondaryConditions: [String] = []
}

struct CommunityPrivacySettings: Codable {
    var postAnonymously = true
    var shareLocation = false
    var allowDirectMessages = true
    var shareWithProviders = false
    var participateInResearch = false
    var showOnlineStatus = false
}

struct SupportGroup: Codable {
    let id: String
    var name: String
    var description: String
    var type: SupportGroupType
    var interests: [CommunityInterest]
    var location: String?
    var isVirtual: Bool
    var maxMembers: Int?
    var memberCount: Int
    var members: [String]
    var moderators: [String]
    var createdBy: String?
    var createdDate: Date?
    var isPrivate: Bool
    var meetingSchedule: String?
}

struct CommunityPost: Codable {
    let id: String
    let authorId: String
    var content: String
    let type: PostType
    let groupId: String?
    var tags: [String]
    let createdDate: Date
    var editedDate: Date?
    var likes: Int
    var replies: Int
    var isAnonymous: Bool
    var likedByCurrentUser: Bool = false
    var isPinned: Bool = false
}

struct CommunityReply: Codable {
    let id: String
    let postId: String
    let authorId: String
    var content: String
    let createdDate: Date
    var editedDate: Date?
    var likes: Int
    var isAnonymous: Bool
    var likedByCurrentUser: Bool = false
}

struct Conversation: Codable {
    let id: String
    var participants: [String]
    let createdDate: Date
    var lastMessageDate: Date
    var isActive: Bool
    var messages: [Message] = []
    var unreadCount: Int = 0
}

struct Message: Codable {
    let id: String
    let conversationId: String
    let senderId: String
    var content: String
    let timestamp: Date
    var isRead: Bool
    let messageType: MessageType
    var attachmentUrl: String?
}

struct CommunityNotification: Codable {
    let id: String
    let userId: String
    let type: NotificationType
    var title: String
    var message: String
    let createdDate: Date
    var isRead: Bool
    var actionUrl: String?
}

enum NotificationType: String, CaseIterable, Codable {
    case newMessage = "New Message"
    case postReply = "Post Reply"
    case postLike = "Post Like"
    case groupInvite = "Group Invite"
    case groupUpdate = "Group Update"
    case systemUpdate = "System Update"
}

struct HealthcareConnection: Codable {
    let id: String
    let userId: String
    let providerId: String
    let connectedDate: Date
    var isActive: Bool
    var permissions: [String] = []
}

struct DataSharingAgreement: Codable {
    let id: String
    let userId: String
    let providerId: String
    var dataTypes: [SharedDataType]
    let startDate: Date
    let endDate: Date
    var isActive: Bool
}

struct ModerationResult {
    let isApproved: Bool
    let reason: String?
}

struct ModerationItem: Codable {
    let id: String
    let contentId: String
    let contentType: ContentType
    var content: String
    let reportedBy: String
    let reportDate: Date
    var status: ModerationStatus
    var reviewedBy: String?
    var reviewDate: Date?
}

enum ModerationStatus: String, CaseIterable, Codable {
    case pending = "Pending"
    case approved = "Approved"
    case rejected = "Rejected"
    case escalated = "Escalated"
}

struct ContentReport: Codable {
    let id: String
    let contentId: String
    let contentType: ContentType
    let reportedBy: String
    let reason: ReportReason
    let details: String?
    let reportDate: Date
    var status: ReportStatus
}

enum ReportStatus: String, CaseIterable, Codable {
    case pending = "Pending"
    case reviewed = "Reviewed"
    case resolved = "Resolved"
    case dismissed = "Dismissed"
}

struct ReportedContent: Codable {
    let id: String
    let contentId: String
    let contentType: ContentType
    let reportCount: Int
    let lastReportDate: Date
    var status: ReportStatus
}

struct CommunityInsights: Codable {
    let totalUsers: Int
    let activeUsers: Int
    let totalPosts: Int
    let totalGroups: Int
    let engagementRate: Double
    let topInterests: [CommunityInterest] = []
    let popularGroups: [String] = []
}

struct UserEngagementStats: Codable {
    let postsCreated: Int
    let repliesCreated: Int
    let likesReceived: Int
    let groupsJoined: Int
    let messagesExchanged: Int
    let reputationScore: Int = 0
}

enum CommunityError: LocalizedError {
    case userNotFound
    case groupNotFound
    case postNotFound
    case conversationNotFound
    case contentViolation(String)
    case insufficientPermissions
    case networkError(String)
    case moderationFailed
    
    var errorDescription: String? {
        switch self {
        case .userNotFound:
            return "User not found"
        case .groupNotFound:
            return "Support group not found"
        case .postNotFound:
            return "Post not found"
        case .conversationNotFound:
            return "Conversation not found"
        case .contentViolation(let reason):
            return "Content violation: \(reason)"
        case .insufficientPermissions:
            return "Insufficient permissions"
        case .networkError(let message):
            return "Network error: \(message)"
        case .moderationFailed:
            return "Content moderation failed"
        }
    }
}