//
//  CommunityManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import Combine
import CoreLocation
import OSLog

// MARK: - Community Manager
@MainActor
class CommunityManager: ObservableObject {
    
    // MARK: - Singleton
    static let shared = CommunityManager()
    
    // MARK: - Published Properties
    @Published var isConnected = false
    @Published var currentUser: CommunityUser?
    @Published var supportGroups: [SupportGroup] = []
    @Published var nearbyGroups: [SupportGroup] = []
    @Published var posts: [CommunityPost] = []
    @Published var messages: [Message] = []
    @Published var notifications: [CommunityNotification] = []
    @Published var moderationReports: [ModerationReport] = []
    @Published var userReputation: UserReputation?
    @Published var privacySettings = CommunityPrivacySettings()
    
    // MARK: - Private Properties
    private let logger = Logger(subsystem: "com.inflamai", category: "Community")
    private let networkManager = NetworkManager()
    private let locationManager = CLLocationManager()
    private let securityManager = SecurityManager.shared
    private let contentModerator = ContentModerator()
    private let sentimentAnalyzer = SentimentAnalyzer()
    
    private var cancellables = Set<AnyCancellable>()
    private var currentLocation: CLLocation?
    
    // MARK: - Constants
    private let maxPostLength = 2000
    private let maxMessageLength = 500
    private let nearbyRadius: CLLocationDistance = 50000 // 50km
    private let anonymousUserPrefix = "Anonymous"
    
    // MARK: - Initialization
    private init() {
        setupLocationManager()
        loadCommunityData()
        setupNotifications()
    }
    
    // MARK: - User Management
    
    func createAnonymousProfile(_ profile: AnonymousProfile) async -> Result<CommunityUser, CommunityError> {
        do {
            // Generate anonymous ID
            let anonymousId = generateAnonymousId()
            
            // Create community user
            let user = CommunityUser(
                id: anonymousId,
                displayName: "\(anonymousUserPrefix)\(String(anonymousId.suffix(6)))",
                anonymousProfile: profile,
                joinedDate: Date(),
                reputation: UserReputation(),
                privacyLevel: .anonymous
            )
            
            // Validate profile
            guard validateAnonymousProfile(profile) else {
                return .failure(.invalidProfile)
            }
            
            // Store user data securely
            try await storeUserData(user)
            
            currentUser = user
            isConnected = true
            
            logger.info("Anonymous profile created successfully")
            return .success(user)
            
        } catch {
            logger.error("Failed to create anonymous profile: \(error.localizedDescription)")
            return .failure(.profileCreationFailed)
        }
    }
    
    func updateProfile(_ updates: ProfileUpdates) async -> Result<CommunityUser, CommunityError> {
        guard var user = currentUser else {
            return .failure(.userNotFound)
        }
        
        // Apply updates
        if let displayName = updates.displayName {
            user.displayName = displayName
        }
        
        if let bio = updates.bio {
            user.anonymousProfile?.bio = bio
        }
        
        if let interests = updates.interests {
            user.anonymousProfile?.interests = interests
        }
        
        if let privacyLevel = updates.privacyLevel {
            user.privacyLevel = privacyLevel
        }
        
        do {
            try await storeUserData(user)
            currentUser = user
            
            logger.info("Profile updated successfully")
            return .success(user)
            
        } catch {
            logger.error("Failed to update profile: \(error.localizedDescription)")
            return .failure(.profileUpdateFailed)
        }
    }
    
    func deleteProfile() async -> Result<Void, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            // Delete all user content
            try await deleteUserContent(user.id)
            
            // Delete user data
            try await deleteUserData(user.id)
            
            currentUser = nil
            isConnected = false
            
            logger.info("Profile deleted successfully")
            return .success(())
            
        } catch {
            logger.error("Failed to delete profile: \(error.localizedDescription)")
            return .failure(.profileDeletionFailed)
        }
    }
    
    // MARK: - Support Groups
    
    func findSupportGroups(criteria: GroupSearchCriteria) async -> Result<[SupportGroup], CommunityError> {
        do {
            var groups = supportGroups
            
            // Filter by condition
            if let condition = criteria.condition {
                groups = groups.filter { $0.condition == condition }
            }
            
            // Filter by location
            if let location = criteria.location, let currentLocation = currentLocation {
                groups = groups.filter { group in
                    guard let groupLocation = group.location else { return false }
                    let distance = currentLocation.distance(from: groupLocation)
                    return distance <= criteria.maxDistance
                }
            }
            
            // Filter by group type
            if let groupType = criteria.groupType {
                groups = groups.filter { $0.groupType == groupType }
            }
            
            // Filter by meeting type
            if let meetingType = criteria.meetingType {
                groups = groups.filter { $0.meetingType == meetingType }
            }
            
            // Sort by relevance
            groups = sortGroupsByRelevance(groups, criteria: criteria)
            
            logger.info("Found \(groups.count) support groups matching criteria")
            return .success(groups)
            
        } catch {
            logger.error("Failed to find support groups: \(error.localizedDescription)")
            return .failure(.searchFailed)
        }
    }
    
    func joinSupportGroup(_ groupId: String) async -> Result<SupportGroup, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        guard let group = supportGroups.first(where: { $0.id == groupId }) else {
            return .failure(.groupNotFound)
        }
        
        do {
            // Check if user is already a member
            if group.members.contains(where: { $0.userId == user.id }) {
                return .failure(.alreadyMember)
            }
            
            // Check group capacity
            if group.members.count >= group.maxMembers {
                return .failure(.groupFull)
            }
            
            // Create membership
            let membership = GroupMembership(
                userId: user.id,
                groupId: groupId,
                joinedDate: Date(),
                role: .member,
                isActive: true
            )
            
            // Add member to group
            var updatedGroup = group
            updatedGroup.members.append(membership)
            
            // Update local data
            if let index = supportGroups.firstIndex(where: { $0.id == groupId }) {
                supportGroups[index] = updatedGroup
            }
            
            // Store membership
            try await storeMembership(membership)
            
            // Send welcome message
            await sendWelcomeMessage(to: user, in: updatedGroup)
            
            logger.info("User joined support group: \(group.name)")
            return .success(updatedGroup)
            
        } catch {
            logger.error("Failed to join support group: \(error.localizedDescription)")
            return .failure(.joinGroupFailed)
        }
    }
    
    func leaveSupportGroup(_ groupId: String) async -> Result<Void, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        guard let groupIndex = supportGroups.firstIndex(where: { $0.id == groupId }) else {
            return .failure(.groupNotFound)
        }
        
        do {
            // Remove member from group
            supportGroups[groupIndex].members.removeAll { $0.userId == user.id }
            
            // Delete membership
            try await deleteMembership(userId: user.id, groupId: groupId)
            
            logger.info("User left support group")
            return .success(())
            
        } catch {
            logger.error("Failed to leave support group: \(error.localizedDescription)")
            return .failure(.leaveGroupFailed)
        }
    }
    
    func createSupportGroup(_ groupData: SupportGroupData) async -> Result<SupportGroup, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            // Validate group data
            guard validateGroupData(groupData) else {
                return .failure(.invalidGroupData)
            }
            
            // Create group
            let group = SupportGroup(
                id: UUID().uuidString,
                name: groupData.name,
                description: groupData.description,
                condition: groupData.condition,
                groupType: groupData.groupType,
                meetingType: groupData.meetingType,
                location: groupData.location,
                createdBy: user.id,
                createdDate: Date(),
                maxMembers: groupData.maxMembers,
                isPrivate: groupData.isPrivate,
                requiresApproval: groupData.requiresApproval,
                tags: groupData.tags,
                rules: groupData.rules,
                members: [GroupMembership(
                    userId: user.id,
                    groupId: "", // Will be set after group creation
                    joinedDate: Date(),
                    role: .admin,
                    isActive: true
                )]
            )
            
            // Update member group ID
            var updatedGroup = group
            updatedGroup.members[0].groupId = group.id
            
            // Store group
            try await storeGroup(updatedGroup)
            
            // Add to local data
            supportGroups.append(updatedGroup)
            
            logger.info("Support group created: \(group.name)")
            return .success(updatedGroup)
            
        } catch {
            logger.error("Failed to create support group: \(error.localizedDescription)")
            return .failure(.groupCreationFailed)
        }
    }
    
    // MARK: - Community Posts
    
    func createPost(_ postData: PostData) async -> Result<CommunityPost, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            // Validate post content
            guard validatePostContent(postData.content) else {
                return .failure(.invalidContent)
            }
            
            // Moderate content
            let moderationResult = await contentModerator.moderateContent(postData.content)
            guard moderationResult.isApproved else {
                return .failure(.contentRejected)
            }
            
            // Analyze sentiment
            let sentiment = sentimentAnalyzer.analyzeSentiment(postData.content)
            
            // Create post
            let post = CommunityPost(
                id: UUID().uuidString,
                authorId: user.id,
                authorDisplayName: user.displayName,
                content: postData.content,
                postType: postData.postType,
                category: postData.category,
                tags: postData.tags,
                groupId: postData.groupId,
                createdDate: Date(),
                sentiment: sentiment,
                isAnonymous: postData.isAnonymous,
                allowComments: postData.allowComments,
                moderationStatus: .approved
            )
            
            // Store post
            try await storePost(post)
            
            // Add to local data
            posts.insert(post, at: 0)
            
            // Update user reputation
            await updateUserReputation(user.id, action: .createPost)
            
            logger.info("Community post created")
            return .success(post)
            
        } catch {
            logger.error("Failed to create post: \(error.localizedDescription)")
            return .failure(.postCreationFailed)
        }
    }
    
    func getPosts(for groupId: String? = nil, category: PostCategory? = nil) async -> Result<[CommunityPost], CommunityError> {
        do {
            var filteredPosts = posts
            
            // Filter by group
            if let groupId = groupId {
                filteredPosts = filteredPosts.filter { $0.groupId == groupId }
            }
            
            // Filter by category
            if let category = category {
                filteredPosts = filteredPosts.filter { $0.category == category }
            }
            
            // Sort by date (newest first)
            filteredPosts.sort { $0.createdDate > $1.createdDate }
            
            return .success(filteredPosts)
            
        } catch {
            logger.error("Failed to get posts: \(error.localizedDescription)")
            return .failure(.fetchFailed)
        }
    }
    
    func reactToPost(_ postId: String, reaction: PostReaction) async -> Result<Void, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        guard let postIndex = posts.firstIndex(where: { $0.id == postId }) else {
            return .failure(.postNotFound)
        }
        
        do {
            // Remove existing reaction from same user
            posts[postIndex].reactions.removeAll { $0.userId == user.id }
            
            // Add new reaction
            let newReaction = PostReactionData(
                userId: user.id,
                reaction: reaction,
                timestamp: Date()
            )
            posts[postIndex].reactions.append(newReaction)
            
            // Store reaction
            try await storeReaction(newReaction, postId: postId)
            
            // Update author reputation
            if let authorId = posts[postIndex].authorId {
                await updateUserReputation(authorId, action: .receiveReaction(reaction))
            }
            
            logger.info("Reaction added to post")
            return .success(())
            
        } catch {
            logger.error("Failed to react to post: \(error.localizedDescription)")
            return .failure(.reactionFailed)
        }
    }
    
    func commentOnPost(_ postId: String, content: String) async -> Result<PostComment, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        guard let postIndex = posts.firstIndex(where: { $0.id == postId }) else {
            return .failure(.postNotFound)
        }
        
        guard posts[postIndex].allowComments else {
            return .failure(.commentsDisabled)
        }
        
        do {
            // Validate comment content
            guard validateCommentContent(content) else {
                return .failure(.invalidContent)
            }
            
            // Moderate content
            let moderationResult = await contentModerator.moderateContent(content)
            guard moderationResult.isApproved else {
                return .failure(.contentRejected)
            }
            
            // Create comment
            let comment = PostComment(
                id: UUID().uuidString,
                postId: postId,
                authorId: user.id,
                authorDisplayName: user.displayName,
                content: content,
                createdDate: Date(),
                isAnonymous: user.privacyLevel == .anonymous
            )
            
            // Add to post
            posts[postIndex].comments.append(comment)
            
            // Store comment
            try await storeComment(comment)
            
            // Update user reputation
            await updateUserReputation(user.id, action: .createComment)
            
            logger.info("Comment added to post")
            return .success(comment)
            
        } catch {
            logger.error("Failed to comment on post: \(error.localizedDescription)")
            return .failure(.commentFailed)
        }
    }
    
    // MARK: - Messaging
    
    func sendMessage(_ messageData: MessageData) async -> Result<Message, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            // Validate message content
            guard validateMessageContent(messageData.content) else {
                return .failure(.invalidContent)
            }
            
            // Moderate content
            let moderationResult = await contentModerator.moderateContent(messageData.content)
            guard moderationResult.isApproved else {
                return .failure(.contentRejected)
            }
            
            // Create message
            let message = Message(
                id: UUID().uuidString,
                senderId: user.id,
                senderDisplayName: user.displayName,
                recipientId: messageData.recipientId,
                groupId: messageData.groupId,
                content: messageData.content,
                messageType: messageData.messageType,
                sentDate: Date(),
                isAnonymous: messageData.isAnonymous,
                isEncrypted: true
            )
            
            // Encrypt message content
            let encryptedMessage = try await encryptMessage(message)
            
            // Store message
            try await storeMessage(encryptedMessage)
            
            // Add to local data
            messages.append(message)
            
            // Send notification to recipient
            await sendMessageNotification(message)
            
            logger.info("Message sent")
            return .success(message)
            
        } catch {
            logger.error("Failed to send message: \(error.localizedDescription)")
            return .failure(.messageFailed)
        }
    }
    
    func getMessages(for conversationId: String) async -> Result<[Message], CommunityError> {
        do {
            let conversationMessages = messages.filter { message in
                (message.recipientId == conversationId && message.senderId == currentUser?.id) ||
                (message.senderId == conversationId && message.recipientId == currentUser?.id)
            }
            
            // Decrypt messages
            let decryptedMessages = try await decryptMessages(conversationMessages)
            
            return .success(decryptedMessages.sorted { $0.sentDate < $1.sentDate })
            
        } catch {
            logger.error("Failed to get messages: \(error.localizedDescription)")
            return .failure(.fetchFailed)
        }
    }
    
    func markMessageAsRead(_ messageId: String) async -> Result<Void, CommunityError> {
        guard let messageIndex = messages.firstIndex(where: { $0.id == messageId }) else {
            return .failure(.messageNotFound)
        }
        
        do {
            messages[messageIndex].isRead = true
            messages[messageIndex].readDate = Date()
            
            // Update stored message
            try await updateMessage(messages[messageIndex])
            
            return .success(())
            
        } catch {
            logger.error("Failed to mark message as read: \(error.localizedDescription)")
            return .failure(.updateFailed)
        }
    }
    
    // MARK: - Healthcare Provider Portal
    
    func connectWithProvider(_ provider: HealthcareProvider) async -> Result<ProviderConnection, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            // Validate provider credentials
            guard validateProviderCredentials(provider) else {
                return .failure(.invalidProvider)
            }
            
            // Create connection request
            let connection = ProviderConnection(
                id: UUID().uuidString,
                userId: user.id,
                providerId: provider.id,
                providerName: provider.name,
                connectionDate: Date(),
                status: .pending,
                permissions: provider.requestedPermissions,
                isVerified: provider.isVerified
            )
            
            // Store connection
            try await storeProviderConnection(connection)
            
            // Send connection request to provider
            await sendProviderConnectionRequest(connection, provider: provider)
            
            logger.info("Provider connection request sent")
            return .success(connection)
            
        } catch {
            logger.error("Failed to connect with provider: \(error.localizedDescription)")
            return .failure(.providerConnectionFailed)
        }
    }
    
    func shareDataWithProvider(_ providerId: String, dataTypes: [HealthDataType]) async -> Result<Void, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            // Check if provider connection exists and is approved
            guard let connection = await getProviderConnection(providerId),
                  connection.status == .approved else {
                return .failure(.providerNotConnected)
            }
            
            // Check permissions
            for dataType in dataTypes {
                guard connection.permissions.contains(dataType.permission) else {
                    return .failure(.insufficientPermissions)
                }
            }
            
            // Create data sharing record
            let sharingRecord = DataSharingRecord(
                id: UUID().uuidString,
                userId: user.id,
                providerId: providerId,
                dataTypes: dataTypes,
                sharedDate: Date(),
                expirationDate: Date().addingTimeInterval(30 * 24 * 60 * 60), // 30 days
                isActive: true
            )
            
            // Store sharing record
            try await storeSharingRecord(sharingRecord)
            
            // Log data access
            securityManager.logDataAccess(DataAccessLog(
                userId: user.id,
                dataType: "HealthData",
                action: .share,
                timestamp: Date(),
                purpose: "Healthcare Provider Sharing"
            ))
            
            logger.info("Data shared with healthcare provider")
            return .success(())
            
        } catch {
            logger.error("Failed to share data with provider: \(error.localizedDescription)")
            return .failure(.dataSharingFailed)
        }
    }
    
    // MARK: - Moderation
    
    func reportContent(_ contentId: String, type: ContentType, reason: ReportReason, description: String?) async -> Result<Void, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            let report = ModerationReport(
                id: UUID().uuidString,
                reporterId: user.id,
                contentId: contentId,
                contentType: type,
                reason: reason,
                description: description,
                reportDate: Date(),
                status: .pending
            )
            
            // Store report
            try await storeReport(report)
            
            // Add to local data
            moderationReports.append(report)
            
            // Notify moderators
            await notifyModerators(report)
            
            logger.info("Content reported for moderation")
            return .success(())
            
        } catch {
            logger.error("Failed to report content: \(error.localizedDescription)")
            return .failure(.reportFailed)
        }
    }
    
    func blockUser(_ userId: String) async -> Result<Void, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            // Add to blocked users list
            var settings = privacySettings
            settings.blockedUsers.append(userId)
            privacySettings = settings
            
            // Store updated settings
            try await storePrivacySettings(settings)
            
            logger.info("User blocked")
            return .success(())
            
        } catch {
            logger.error("Failed to block user: \(error.localizedDescription)")
            return .failure(.blockFailed)
        }
    }
    
    func unblockUser(_ userId: String) async -> Result<Void, CommunityError> {
        do {
            // Remove from blocked users list
            var settings = privacySettings
            settings.blockedUsers.removeAll { $0 == userId }
            privacySettings = settings
            
            // Store updated settings
            try await storePrivacySettings(settings)
            
            logger.info("User unblocked")
            return .success(())
            
        } catch {
            logger.error("Failed to unblock user: \(error.localizedDescription)")
            return .failure(.unblockFailed)
        }
    }
    
    // MARK: - Notifications
    
    func getNotifications() async -> Result<[CommunityNotification], CommunityError> {
        return .success(notifications.sorted { $0.timestamp > $1.timestamp })
    }
    
    func markNotificationAsRead(_ notificationId: String) async -> Result<Void, CommunityError> {
        guard let index = notifications.firstIndex(where: { $0.id == notificationId }) else {
            return .failure(.notificationNotFound)
        }
        
        notifications[index].isRead = true
        return .success(())
    }
    
    func clearAllNotifications() async -> Result<Void, CommunityError> {
        notifications.removeAll()
        return .success(())
    }
    
    // MARK: - Privacy & Settings
    
    func updatePrivacySettings(_ settings: CommunityPrivacySettings) async -> Result<Void, CommunityError> {
        do {
            privacySettings = settings
            try await storePrivacySettings(settings)
            
            logger.info("Privacy settings updated")
            return .success(())
            
        } catch {
            logger.error("Failed to update privacy settings: \(error.localizedDescription)")
            return .failure(.settingsUpdateFailed)
        }
    }
    
    func exportUserData() async -> Result<CommunityDataExport, CommunityError> {
        guard let user = currentUser else {
            return .failure(.userNotFound)
        }
        
        do {
            let userPosts = posts.filter { $0.authorId == user.id }
            let userMessages = messages.filter { $0.senderId == user.id }
            let userGroups = supportGroups.filter { $0.members.contains { $0.userId == user.id } }
            
            let export = CommunityDataExport(
                user: user,
                posts: userPosts,
                messages: userMessages,
                groups: userGroups,
                exportDate: Date()
            )
            
            logger.info("User data exported")
            return .success(export)
            
        } catch {
            logger.error("Failed to export user data: \(error.localizedDescription)")
            return .failure(.exportFailed)
        }
    }
    
    // MARK: - Private Methods
    
    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyKilometer
    }
    
    private func loadCommunityData() {
        // Load cached community data
        loadSupportGroups()
        loadPosts()
        loadMessages()
        loadNotifications()
    }
    
    private func setupNotifications() {
        // Set up notification observers
    }
    
    private func generateAnonymousId() -> String {
        return UUID().uuidString
    }
    
    private func validateAnonymousProfile(_ profile: AnonymousProfile) -> Bool {
        return !profile.condition.isEmpty && profile.ageRange != nil
    }
    
    private func validateGroupData(_ data: SupportGroupData) -> Bool {
        return !data.name.isEmpty && !data.description.isEmpty && data.maxMembers > 0
    }
    
    private func validatePostContent(_ content: String) -> Bool {
        return !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
               content.count <= maxPostLength
    }
    
    private func validateCommentContent(_ content: String) -> Bool {
        return !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
               content.count <= maxMessageLength
    }
    
    private func validateMessageContent(_ content: String) -> Bool {
        return !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
               content.count <= maxMessageLength
    }
    
    private func validateProviderCredentials(_ provider: HealthcareProvider) -> Bool {
        return !provider.licenseNumber.isEmpty && provider.isVerified
    }
    
    private func sortGroupsByRelevance(_ groups: [SupportGroup], criteria: GroupSearchCriteria) -> [SupportGroup] {
        return groups.sorted { group1, group2 in
            var score1 = 0
            var score2 = 0
            
            // Score based on member count
            score1 += group1.members.count
            score2 += group2.members.count
            
            // Score based on activity (recent posts)
            // This would require additional data about group activity
            
            // Score based on location proximity
            if let location = criteria.location, let currentLocation = currentLocation {
                if let group1Location = group1.location {
                    let distance1 = currentLocation.distance(from: group1Location)
                    score1 += Int(max(0, 10000 - distance1 / 1000)) // Closer groups get higher scores
                }
                
                if let group2Location = group2.location {
                    let distance2 = currentLocation.distance(from: group2Location)
                    score2 += Int(max(0, 10000 - distance2 / 1000))
                }
            }
            
            return score1 > score2
        }
    }
    
    // MARK: - Storage Methods
    
    private func storeUserData(_ user: CommunityUser) async throws {
        let data = try JSONEncoder().encode(user)
        let encryptedData = try securityManager.encryptData(data)
        try securityManager.storeSecureData(encryptedData.data, for: "community_user_\(user.id)")
    }
    
    private func deleteUserData(_ userId: String) async throws {
        try securityManager.deleteSecureData(for: "community_user_\(userId)")
    }
    
    private func deleteUserContent(_ userId: String) async throws {
        // Delete user posts
        posts.removeAll { $0.authorId == userId }
        
        // Delete user messages
        messages.removeAll { $0.senderId == userId }
        
        // Remove user from groups
        for i in 0..<supportGroups.count {
            supportGroups[i].members.removeAll { $0.userId == userId }
        }
    }
    
    private func storeMembership(_ membership: GroupMembership) async throws {
        let data = try JSONEncoder().encode(membership)
        try securityManager.storeSecureData(data, for: "membership_\(membership.userId)_\(membership.groupId)")
    }
    
    private func deleteMembership(userId: String, groupId: String) async throws {
        try securityManager.deleteSecureData(for: "membership_\(userId)_\(groupId)")
    }
    
    private func storeGroup(_ group: SupportGroup) async throws {
        let data = try JSONEncoder().encode(group)
        try securityManager.storeSecureData(data, for: "group_\(group.id)")
    }
    
    private func storePost(_ post: CommunityPost) async throws {
        let data = try JSONEncoder().encode(post)
        try securityManager.storeSecureData(data, for: "post_\(post.id)")
    }
    
    private func storeReaction(_ reaction: PostReactionData, postId: String) async throws {
        let data = try JSONEncoder().encode(reaction)
        try securityManager.storeSecureData(data, for: "reaction_\(postId)_\(reaction.userId)")
    }
    
    private func storeComment(_ comment: PostComment) async throws {
        let data = try JSONEncoder().encode(comment)
        try securityManager.storeSecureData(data, for: "comment_\(comment.id)")
    }
    
    private func storeMessage(_ message: Message) async throws {
        let data = try JSONEncoder().encode(message)
        let encryptedData = try securityManager.encryptData(data)
        try securityManager.storeSecureData(encryptedData.data, for: "message_\(message.id)")
    }
    
    private func updateMessage(_ message: Message) async throws {
        try await storeMessage(message)
    }
    
    private func storeProviderConnection(_ connection: ProviderConnection) async throws {
        let data = try JSONEncoder().encode(connection)
        try securityManager.storeSecureData(data, for: "provider_connection_\(connection.id)")
    }
    
    private func storeSharingRecord(_ record: DataSharingRecord) async throws {
        let data = try JSONEncoder().encode(record)
        try securityManager.storeSecureData(data, for: "sharing_record_\(record.id)")
    }
    
    private func storeReport(_ report: ModerationReport) async throws {
        let data = try JSONEncoder().encode(report)
        try securityManager.storeSecureData(data, for: "report_\(report.id)")
    }
    
    private func storePrivacySettings(_ settings: CommunityPrivacySettings) async throws {
        let data = try JSONEncoder().encode(settings)
        try securityManager.storeSecureData(data, for: "community_privacy_settings")
    }
    
    // MARK: - Load Methods
    
    private func loadSupportGroups() {
        // Load support groups from storage
        // This would be implemented with actual data loading
    }
    
    private func loadPosts() {
        // Load posts from storage
    }
    
    private func loadMessages() {
        // Load messages from storage
    }
    
    private func loadNotifications() {
        // Load notifications from storage
    }
    
    // MARK: - Encryption Methods
    
    private func encryptMessage(_ message: Message) async throws -> Message {
        var encryptedMessage = message
        let encryptedContent = try securityManager.encryptString(message.content)
        encryptedMessage.content = encryptedContent.data.base64EncodedString()
        return encryptedMessage
    }
    
    private func decryptMessages(_ messages: [Message]) async throws -> [Message] {
        return try messages.map { message in
            var decryptedMessage = message
            if message.isEncrypted {
                guard let contentData = Data(base64Encoded: message.content) else {
                    throw CommunityError.decryptionFailed
                }
                let encryptedData = EncryptedData(data: contentData, iv: Data(), tag: Data())
                decryptedMessage.content = try securityManager.decryptString(encryptedData)
            }
            return decryptedMessage
        }
    }
    
    // MARK: - Notification Methods
    
    private func sendWelcomeMessage(to user: CommunityUser, in group: SupportGroup) async {
        let welcomeNotification = CommunityNotification(
            id: UUID().uuidString,
            userId: user.id,
            type: .groupWelcome,
            title: "Welcome to \(group.name)",
            message: "Welcome to the \(group.name) support group! Feel free to introduce yourself and connect with other members.",
            timestamp: Date(),
            isRead: false,
            relatedId: group.id
        )
        
        notifications.append(welcomeNotification)
    }
    
    private func sendMessageNotification(_ message: Message) async {
        guard let recipientId = message.recipientId else { return }
        
        let notification = CommunityNotification(
            id: UUID().uuidString,
            userId: recipientId,
            type: .newMessage,
            title: "New Message",
            message: "You have a new message from \(message.senderDisplayName)",
            timestamp: Date(),
            isRead: false,
            relatedId: message.id
        )
        
        notifications.append(notification)
    }
    
    private func sendProviderConnectionRequest(_ connection: ProviderConnection, provider: HealthcareProvider) async {
        // Send connection request to healthcare provider
        // This would involve API calls to the provider's system
    }
    
    private func notifyModerators(_ report: ModerationReport) async {
        // Notify moderators about new report
        // This would involve sending notifications to moderator accounts
    }
    
    // MARK: - Reputation Methods
    
    private func updateUserReputation(_ userId: String, action: ReputationAction) async {
        guard let user = currentUser, user.id == userId else { return }
        
        var reputation = user.reputation ?? UserReputation()
        
        switch action {
        case .createPost:
            reputation.points += 5
            reputation.postsCreated += 1
        case .createComment:
            reputation.points += 2
            reputation.commentsCreated += 1
        case .receiveReaction(let reaction):
            switch reaction {
            case .like, .helpful:
                reputation.points += 1
            case .love:
                reputation.points += 2
            case .dislike:
                reputation.points -= 1
            }
        case .helpfulPost:
            reputation.points += 10
            reputation.helpfulPosts += 1
        }
        
        // Update reputation level
        reputation.level = calculateReputationLevel(reputation.points)
        
        userReputation = reputation
    }
    
    private func calculateReputationLevel(_ points: Int) -> ReputationLevel {
        switch points {
        case 0..<50:
            return .newcomer
        case 50..<200:
            return .contributor
        case 200..<500:
            return .supporter
        case 500..<1000:
            return .mentor
        default:
            return .champion
        }
    }
    
    // MARK: - Helper Methods
    
    private func getProviderConnection(_ providerId: String) async -> ProviderConnection? {
        // Retrieve provider connection from storage
        // This would be implemented with actual data retrieval
        return nil
    }
}

// MARK: - CLLocationManagerDelegate

extension CommunityManager: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentLocation = locations.last
        
        // Update nearby groups
        Task {
            await updateNearbyGroups()
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        logger.error("Location manager failed with error: \(error.localizedDescription)")
    }
    
    private func updateNearbyGroups() async {
        guard let location = currentLocation else { return }
        
        nearbyGroups = supportGroups.filter { group in
            guard let groupLocation = group.location else { return false }
            let distance = location.distance(from: groupLocation)
            return distance <= nearbyRadius
        }
    }
}

// MARK: - Supporting Types

// Community User
struct CommunityUser: Codable, Identifiable {
    let id: String
    var displayName: String
    var anonymousProfile: AnonymousProfile?
    let joinedDate: Date
    var reputation: UserReputation?
    var privacyLevel: PrivacyLevel
    var isActive: Bool = true
    var lastActiveDate: Date = Date()
}

struct AnonymousProfile: Codable {
    var condition: String
    var ageRange: AgeRange?
    var bio: String?
    var interests: [String] = []
    var experienceLevel: ExperienceLevel?
    var preferredLanguage: String = "en"
}

enum AgeRange: String, Codable, CaseIterable {
    case under18 = "under18"
    case age18to25 = "18-25"
    case age26to35 = "26-35"
    case age36to45 = "36-45"
    case age46to55 = "46-55"
    case age56to65 = "56-65"
    case over65 = "over65"
    
    var displayName: String {
        switch self {
        case .under18: return "Under 18"
        case .age18to25: return "18-25"
        case .age26to35: return "26-35"
        case .age36to45: return "36-45"
        case .age46to55: return "46-55"
        case .age56to65: return "56-65"
        case .over65: return "Over 65"
        }
    }
}

enum ExperienceLevel: String, Codable, CaseIterable {
    case newlyDiagnosed = "newly_diagnosed"
    case someExperience = "some_experience"
    case experienced = "experienced"
    case veteran = "veteran"
    
    var displayName: String {
        switch self {
        case .newlyDiagnosed: return "Newly Diagnosed"
        case .someExperience: return "Some Experience"
        case .experienced: return "Experienced"
        case .veteran: return "Veteran"
        }
    }
}

enum PrivacyLevel: String, Codable {
    case anonymous = "anonymous"
    case pseudonymous = "pseudonymous"
    case identified = "identified"
}

// Support Groups
struct SupportGroup: Codable, Identifiable {
    let id: String
    let name: String
    let description: String
    let condition: String
    let groupType: GroupType
    let meetingType: MeetingType
    let location: CLLocation?
    let createdBy: String
    let createdDate: Date
    let maxMembers: Int
    let isPrivate: Bool
    let requiresApproval: Bool
    let tags: [String]
    let rules: [String]
    var members: [GroupMembership]
    var isActive: Bool = true
}

struct SupportGroupData {
    let name: String
    let description: String
    let condition: String
    let groupType: GroupType
    let meetingType: MeetingType
    let location: CLLocation?
    let maxMembers: Int
    let isPrivate: Bool
    let requiresApproval: Bool
    let tags: [String]
    let rules: [String]
}

enum GroupType: String, Codable, CaseIterable {
    case support = "support"
    case educational = "educational"
    case social = "social"
    case advocacy = "advocacy"
    case research = "research"
    
    var displayName: String {
        switch self {
        case .support: return "Support"
        case .educational: return "Educational"
        case .social: return "Social"
        case .advocacy: return "Advocacy"
        case .research: return "Research"
        }
    }
}

enum MeetingType: String, Codable, CaseIterable {
    case online = "online"
    case inPerson = "in_person"
    case hybrid = "hybrid"
    
    var displayName: String {
        switch self {
        case .online: return "Online"
        case .inPerson: return "In Person"
        case .hybrid: return "Hybrid"
        }
    }
}

struct GroupMembership: Codable {
    let userId: String
    var groupId: String
    let joinedDate: Date
    let role: MemberRole
    var isActive: Bool
}

enum MemberRole: String, Codable {
    case member = "member"
    case moderator = "moderator"
    case admin = "admin"
}

struct GroupSearchCriteria {
    let condition: String?
    let location: CLLocation?
    let maxDistance: CLLocationDistance
    let groupType: GroupType?
    let meetingType: MeetingType?
    let keywords: [String]?
}

// Community Posts
struct CommunityPost: Codable, Identifiable {
    let id: String
    let authorId: String?
    let authorDisplayName: String
    let content: String
    let postType: PostType
    let category: PostCategory
    let tags: [String]
    let groupId: String?
    let createdDate: Date
    let sentiment: SentimentScore?
    let isAnonymous: Bool
    let allowComments: Bool
    let moderationStatus: ModerationStatus
    var reactions: [PostReactionData] = []
    var comments: [PostComment] = []
    var viewCount: Int = 0
}

struct PostData {
    let content: String
    let postType: PostType
    let category: PostCategory
    let tags: [String]
    let groupId: String?
    let isAnonymous: Bool
    let allowComments: Bool
}

enum PostType: String, Codable, CaseIterable {
    case question = "question"
    case experience = "experience"
    case tip = "tip"
    case support = "support"
    case celebration = "celebration"
    case resource = "resource"
    
    var displayName: String {
        switch self {
        case .question: return "Question"
        case .experience: return "Experience"
        case .tip: return "Tip"
        case .support: return "Support"
        case .celebration: return "Celebration"
        case .resource: return "Resource"
        }
    }
}

enum PostCategory: String, Codable, CaseIterable {
    case symptoms = "symptoms"
    case medications = "medications"
    case lifestyle = "lifestyle"
    case mental_health = "mental_health"
    case relationships = "relationships"
    case work = "work"
    case exercise = "exercise"
    case diet = "diet"
    case sleep = "sleep"
    case general = "general"
    
    var displayName: String {
        switch self {
        case .symptoms: return "Symptoms"
        case .medications: return "Medications"
        case .lifestyle: return "Lifestyle"
        case .mental_health: return "Mental Health"
        case .relationships: return "Relationships"
        case .work: return "Work"
        case .exercise: return "Exercise"
        case .diet: return "Diet"
        case .sleep: return "Sleep"
        case .general: return "General"
        }
    }
}

enum PostReaction: String, Codable {
    case like = "like"
    case love = "love"
    case helpful = "helpful"
    case dislike = "dislike"
    
    var emoji: String {
        switch self {
        case .like: return "👍"
        case .love: return "❤️"
        case .helpful: return "💡"
        case .dislike: return "👎"
        }
    }
}

struct PostReactionData: Codable {
    let userId: String
    let reaction: PostReaction
    let timestamp: Date
}

struct PostComment: Codable, Identifiable {
    let id: String
    let postId: String
    let authorId: String?
    let authorDisplayName: String
    let content: String
    let createdDate: Date
    let isAnonymous: Bool
    var reactions: [PostReactionData] = []
}

// Messages
struct Message: Codable, Identifiable {
    let id: String
    let senderId: String
    let senderDisplayName: String
    let recipientId: String?
    let groupId: String?
    var content: String
    let messageType: MessageType
    let sentDate: Date
    let isAnonymous: Bool
    let isEncrypted: Bool
    var isRead: Bool = false
    var readDate: Date?
}

struct MessageData {
    let recipientId: String?
    let groupId: String?
    let content: String
    let messageType: MessageType
    let isAnonymous: Bool
}

enum MessageType: String, Codable {
    case text = "text"
    case image = "image"
    case voice = "voice"
    case system = "system"
}

// Healthcare Provider
struct HealthcareProvider: Codable, Identifiable {
    let id: String
    let name: String
    let specialty: String
    let licenseNumber: String
    let institution: String
    let email: String
    let phone: String?
    let isVerified: Bool
    let requestedPermissions: [ProviderPermission]
}

struct ProviderConnection: Codable, Identifiable {
    let id: String
    let userId: String
    let providerId: String
    let providerName: String
    let connectionDate: Date
    var status: ConnectionStatus
    let permissions: [ProviderPermission]
    let isVerified: Bool
}

enum ConnectionStatus: String, Codable {
    case pending = "pending"
    case approved = "approved"
    case rejected = "rejected"
    case revoked = "revoked"
}

enum ProviderPermission: String, Codable {
    case readHealthData = "read_health_data"
    case readMedications = "read_medications"
    case readSymptoms = "read_symptoms"
    case readMoodData = "read_mood_data"
    case readActivityData = "read_activity_data"
    case writeRecommendations = "write_recommendations"
}

struct DataSharingRecord: Codable, Identifiable {
    let id: String
    let userId: String
    let providerId: String
    let dataTypes: [HealthDataType]
    let sharedDate: Date
    let expirationDate: Date
    var isActive: Bool
}

enum HealthDataType {
    case symptoms
    case medications
    case mood
    case activity
    case vitals
    
    var permission: ProviderPermission {
        switch self {
        case .symptoms: return .readSymptoms
        case .medications: return .readMedications
        case .mood: return .readMoodData
        case .activity: return .readActivityData
        case .vitals: return .readHealthData
        }
    }
}

// Moderation
struct ModerationReport: Codable, Identifiable {
    let id: String
    let reporterId: String
    let contentId: String
    let contentType: ContentType
    let reason: ReportReason
    let description: String?
    let reportDate: Date
    var status: ReportStatus
}

enum ContentType: String, Codable {
    case post = "post"
    case comment = "comment"
    case message = "message"
    case profile = "profile"
}

enum ReportReason: String, Codable, CaseIterable {
    case spam = "spam"
    case harassment = "harassment"
    case inappropriate = "inappropriate"
    case misinformation = "misinformation"
    case privacy = "privacy"
    case other = "other"
    
    var displayName: String {
        switch self {
        case .spam: return "Spam"
        case .harassment: return "Harassment"
        case .inappropriate: return "Inappropriate Content"
        case .misinformation: return "Misinformation"
        case .privacy: return "Privacy Violation"
        case .other: return "Other"
        }
    }
}

enum ReportStatus: String, Codable {
    case pending = "pending"
    case reviewed = "reviewed"
    case resolved = "resolved"
    case dismissed = "dismissed"
}

enum ModerationStatus: String, Codable {
    case pending = "pending"
    case approved = "approved"
    case rejected = "rejected"
    case flagged = "flagged"
}

// Notifications
struct CommunityNotification: Codable, Identifiable {
    let id: String
    let userId: String
    let type: NotificationType
    let title: String
    let message: String
    let timestamp: Date
    var isRead: Bool
    let relatedId: String?
}

enum NotificationType: String, Codable {
    case newMessage = "new_message"
    case groupInvite = "group_invite"
    case groupWelcome = "group_welcome"
    case postReaction = "post_reaction"
    case postComment = "post_comment"
    case providerRequest = "provider_request"
    case moderationUpdate = "moderation_update"
    case systemUpdate = "system_update"
}

// Privacy Settings
struct CommunityPrivacySettings: Codable {
    var allowDirectMessages = true
    var allowGroupInvites = true
    var showOnlineStatus = false
    var shareLocationInGroups = false
    var allowProviderConnections = true
    var blockedUsers: [String] = []
    var mutedGroups: [String] = []
    var notificationSettings = NotificationSettings()
}

struct NotificationSettings: Codable {
    var enablePushNotifications = true
    var enableEmailNotifications = false
    var notifyOnNewMessages = true
    var notifyOnGroupActivity = true
    var notifyOnPostReactions = true
    var notifyOnComments = true
    var quietHoursEnabled = false
    var quietHoursStart = "22:00"
    var quietHoursEnd = "08:00"
}

// Reputation
struct UserReputation: Codable {
    var points: Int = 0
    var level: ReputationLevel = .newcomer
    var postsCreated: Int = 0
    var commentsCreated: Int = 0
    var helpfulPosts: Int = 0
    var groupsJoined: Int = 0
}

enum ReputationLevel: String, Codable {
    case newcomer = "newcomer"
    case contributor = "contributor"
    case supporter = "supporter"
    case mentor = "mentor"
    case champion = "champion"
    
    var displayName: String {
        switch self {
        case .newcomer: return "Newcomer"
        case .contributor: return "Contributor"
        case .supporter: return "Supporter"
        case .mentor: return "Mentor"
        case .champion: return "Champion"
        }
    }
    
    var badge: String {
        switch self {
        case .newcomer: return "🌱"
        case .contributor: return "🤝"
        case .supporter: return "💪"
        case .mentor: return "🎓"
        case .champion: return "👑"
        }
    }
}

enum ReputationAction {
    case createPost
    case createComment
    case receiveReaction(PostReaction)
    case helpfulPost
}

// Profile Updates
struct ProfileUpdates {
    let displayName: String?
    let bio: String?
    let interests: [String]?
    let privacyLevel: PrivacyLevel?
}

// Data Export
struct CommunityDataExport: Codable {
    let user: CommunityUser
    let posts: [CommunityPost]
    let messages: [Message]
    let groups: [SupportGroup]
    let exportDate: Date
}

// Errors
enum CommunityError: Error, LocalizedError {
    case userNotFound
    case invalidProfile
    case profileCreationFailed
    case profileUpdateFailed
    case profileDeletionFailed
    case groupNotFound
    case groupFull
    case alreadyMember
    case joinGroupFailed
    case leaveGroupFailed
    case invalidGroupData
    case groupCreationFailed
    case postNotFound
    case invalidContent
    case contentRejected
    case postCreationFailed
    case commentFailed
    case commentsDisabled
    case reactionFailed
    case messageNotFound
    case messageFailed
    case invalidProvider
    case providerNotConnected
    case insufficientPermissions
    case providerConnectionFailed
    case dataSharingFailed
    case reportFailed
    case blockFailed
    case unblockFailed
    case notificationNotFound
    case settingsUpdateFailed
    case exportFailed
    case searchFailed
    case fetchFailed
    case updateFailed
    case decryptionFailed
    
    var errorDescription: String? {
        switch self {
        case .userNotFound:
            return "User not found"
        case .invalidProfile:
            return "Invalid profile data"
        case .profileCreationFailed:
            return "Failed to create profile"
        case .profileUpdateFailed:
            return "Failed to update profile"
        case .profileDeletionFailed:
            return "Failed to delete profile"
        case .groupNotFound:
            return "Support group not found"
        case .groupFull:
            return "Support group is full"
        case .alreadyMember:
            return "Already a member of this group"
        case .joinGroupFailed:
            return "Failed to join support group"
        case .leaveGroupFailed:
            return "Failed to leave support group"
        case .invalidGroupData:
            return "Invalid group data"
        case .groupCreationFailed:
            return "Failed to create support group"
        case .postNotFound:
            return "Post not found"
        case .invalidContent:
            return "Invalid content"
        case .contentRejected:
            return "Content rejected by moderation"
        case .postCreationFailed:
            return "Failed to create post"
        case .commentFailed:
            return "Failed to add comment"
        case .commentsDisabled:
            return "Comments are disabled for this post"
        case .reactionFailed:
            return "Failed to add reaction"
        case .messageNotFound:
            return "Message not found"
        case .messageFailed:
            return "Failed to send message"
        case .invalidProvider:
            return "Invalid healthcare provider"
        case .providerNotConnected:
            return "Healthcare provider not connected"
        case .insufficientPermissions:
            return "Insufficient permissions"
        case .providerConnectionFailed:
            return "Failed to connect with healthcare provider"
        case .dataSharingFailed:
            return "Failed to share data"
        case .reportFailed:
            return "Failed to report content"
        case .blockFailed:
            return "Failed to block user"
        case .unblockFailed:
            return "Failed to unblock user"
        case .notificationNotFound:
            return "Notification not found"
        case .settingsUpdateFailed:
            return "Failed to update settings"
        case .exportFailed:
            return "Failed to export data"
        case .searchFailed:
            return "Search failed"
        case .fetchFailed:
            return "Failed to fetch data"
        case .updateFailed:
            return "Failed to update data"
        case .decryptionFailed:
            return "Failed to decrypt data"
        }
    }
}

// MARK: - Helper Classes

// Network Manager for API calls
class NetworkManager {
    func sendRequest<T: Codable>(_ request: URLRequest, responseType: T.Type) async throws -> T {
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(T.self, from: data)
    }
}

// Content Moderation
class ContentModerator {
    func moderateContent(_ content: String) async -> ModerationResult {
        // Implement content moderation logic
        // This could include:
        // - Profanity filtering
        // - Spam detection
        // - Inappropriate content detection
        // - Medical misinformation detection
        
        let bannedWords = ["spam", "scam", "fake"]
        let containsBannedWords = bannedWords.contains { content.lowercased().contains($0) }
        
        return ModerationResult(
            isApproved: !containsBannedWords,
            confidence: containsBannedWords ? 0.9 : 0.1,
            flags: containsBannedWords ? [.inappropriateContent] : [],
            suggestedAction: containsBannedWords ? .reject : .approve
        )
    }
}

struct ModerationResult {
    let isApproved: Bool
    let confidence: Double
    let flags: [ModerationFlag]
    let suggestedAction: ModerationAction
}

enum ModerationFlag {
    case inappropriateContent
    case spam
    case harassment
    case misinformation
}

enum ModerationAction {
    case approve
    case reject
    case flagForReview
}

// Sentiment Analysis
class SentimentAnalyzer {
    func analyzeSentiment(_ text: String) -> SentimentScore {
        // Implement sentiment analysis
        // This could use Core ML or external APIs
        
        let positiveWords = ["good", "great", "better", "improved", "happy", "thankful"]
        let negativeWords = ["bad", "worse", "terrible", "pain", "hurt", "sad"]
        
        let words = text.lowercased().components(separatedBy: .whitespacesAndPunctuation)
        
        let positiveCount = words.filter { positiveWords.contains($0) }.count
        let negativeCount = words.filter { negativeWords.contains($0) }.count
        
        let score: Double
        if positiveCount > negativeCount {
            score = 0.7
        } else if negativeCount > positiveCount {
            score = 0.3
        } else {
            score = 0.5
        }
        
        return SentimentScore(
            score: score,
            confidence: 0.8,
            emotion: score > 0.6 ? .positive : score < 0.4 ? .negative : .neutral
        )
    }
}

struct SentimentScore: Codable {
    let score: Double // 0.0 (very negative) to 1.0 (very positive)
    let confidence: Double
    let emotion: EmotionType
}

enum EmotionType: String, Codable {
    case positive = "positive"
    case negative = "negative"
    case neutral = "neutral"
    case mixed = "mixed"
}