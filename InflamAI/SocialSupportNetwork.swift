//
//  SocialSupportNetwork.swift
//  InflamAI-Swift
//
//  Social support network for community features and peer support
//

import Foundation
import Combine
import CoreData
import UserNotifications
import CloudKit
import CryptoKit
import Network

// MARK: - User Models

struct CommunityUser: Codable, Identifiable {
    let id: String
    let username: String
    let displayName: String
    let profileImageURL: String?
    let bio: String?
    let joinDate: Date
    let lastActiveDate: Date
    let isVerified: Bool
    let privacySettings: PrivacySettings
    let supportRole: SupportRole
    let badges: [UserBadge]
    let stats: UserStats
    let preferences: CommunityPreferences
    let location: UserLocation?
    let languages: [String]
    let timeZone: String
    let isOnline: Bool
    let isModerator: Bool
    let isBlocked: Bool
    let reportCount: Int
}

struct PrivacySettings: Codable {
    let profileVisibility: ProfileVisibility
    let allowDirectMessages: Bool
    let allowGroupInvites: Bool
    let shareHealthData: Bool
    let shareLocation: Bool
    let allowMentions: Bool
    let showOnlineStatus: Bool
    let allowFriendRequests: Bool
}

enum ProfileVisibility: String, Codable {
    case publicProfile = "public"
    case friendsOnly = "friends_only"
    case privateProfile = "private"
}

enum SupportRole: String, Codable, CaseIterable {
    case member = "member"
    case supporter = "supporter"
    case mentor = "mentor"
    case advocate = "advocate"
    case healthcareProfessional = "healthcare_professional"
    case researcher = "researcher"
    case moderator = "moderator"
    
    var displayName: String {
        switch self {
        case .member: return "Community Member"
        case .supporter: return "Peer Supporter"
        case .mentor: return "Mentor"
        case .advocate: return "Patient Advocate"
        case .healthcareProfessional: return "Healthcare Professional"
        case .researcher: return "Researcher"
        case .moderator: return "Moderator"
        }
    }
    
    var icon: String {
        switch self {
        case .member: return "person.circle"
        case .supporter: return "hands.sparkles"
        case .mentor: return "graduationcap"
        case .advocate: return "megaphone"
        case .healthcareProfessional: return "stethoscope"
        case .researcher: return "microscope"
        case .moderator: return "shield"
        }
    }
    
    var color: String {
        switch self {
        case .member: return "blue"
        case .supporter: return "green"
        case .mentor: return "purple"
        case .advocate: return "orange"
        case .healthcareProfessional: return "red"
        case .researcher: return "indigo"
        case .moderator: return "yellow"
        }
    }
}

struct UserBadge: Codable, Identifiable {
    let id: String
    let name: String
    let description: String
    let iconName: String
    let color: String
    let category: BadgeCategory
    let earnedDate: Date
    let isVisible: Bool
}

enum BadgeCategory: String, Codable {
    case participation = "participation"
    case helpfulness = "helpfulness"
    case milestone = "milestone"
    case expertise = "expertise"
    case leadership = "leadership"
    case special = "special"
}

struct UserStats: Codable {
    let postsCount: Int
    let commentsCount: Int
    let likesReceived: Int
    let helpfulVotes: Int
    let friendsCount: Int
    let groupsCount: Int
    let supportGiven: Int
    let supportReceived: Int
    let streakDays: Int
    let totalPoints: Int
}

struct CommunityPreferences: Codable {
    let notificationSettings: NotificationSettings
    let contentFilters: ContentFilters
    let interactionPreferences: InteractionPreferences
    let accessibilitySettings: AccessibilitySettings
}

struct NotificationSettings: Codable {
    let newMessages: Bool
    let newPosts: Bool
    let newComments: Bool
    let friendRequests: Bool
    let groupInvites: Bool
    let mentions: Bool
    let supportRequests: Bool
    let emergencyAlerts: Bool
    let weeklyDigest: Bool
    let pushNotifications: Bool
    let emailNotifications: Bool
    let quietHours: QuietHours?
}

struct QuietHours: Codable {
    let startTime: String // HH:mm format
    let endTime: String
    let isEnabled: Bool
    let allowEmergency: Bool
}

struct ContentFilters: Codable {
    let hideNegativeContent: Bool
    let hideTriggerContent: Bool
    let filterProfanity: Bool
    let hideUnverifiedAdvice: Bool
    let blockedKeywords: [String]
    let blockedUsers: [String]
    let blockedTopics: [String]
}

struct InteractionPreferences: Codable {
    let allowAnonymousPosts: Bool
    let requireModeration: Bool
    let autoAcceptFriends: Bool
    let showReadReceipts: Bool
    let allowVoiceMessages: Bool
    let allowVideoMessages: Bool
    let preferredLanguage: String
}

struct AccessibilitySettings: Codable {
    let fontSize: FontSize
    let highContrast: Bool
    let reduceMotion: Bool
    let voiceOverEnabled: Bool
    let screenReaderOptimized: Bool
}

enum FontSize: String, Codable {
    case small = "small"
    case medium = "medium"
    case large = "large"
    case extraLarge = "extra_large"
}

struct UserLocation: Codable {
    let city: String?
    let state: String?
    let country: String
    let isVisible: Bool
}

// MARK: - Community Models

struct CommunityPost: Codable, Identifiable {
    let id: String
    let authorId: String
    let authorUsername: String
    let authorDisplayName: String
    let authorProfileImageURL: String?
    let authorRole: SupportRole
    let content: String
    let contentType: PostContentType
    let category: PostCategory
    let tags: [String]
    let attachments: [PostAttachment]
    let createdAt: Date
    let updatedAt: Date?
    let isEdited: Bool
    let isAnonymous: Bool
    let isPinned: Bool
    let isLocked: Bool
    let isReported: Bool
    let reportCount: Int
    let moderationStatus: ModerationStatus
    let visibility: PostVisibility
    let groupId: String?
    let parentPostId: String? // For replies
    let threadId: String?
    let engagement: PostEngagement
    let location: String?
    let mood: MoodLevel?
    let painLevel: Int?
    let triggerWarning: Bool
    let supportType: SupportType?
}

enum PostContentType: String, Codable {
    case text = "text"
    case image = "image"
    case video = "video"
    case audio = "audio"
    case poll = "poll"
    case article = "article"
    case question = "question"
    case story = "story"
    case tip = "tip"
    case resource = "resource"
}

enum PostCategory: String, Codable, CaseIterable {
    case general = "general"
    case support = "support"
    case advice = "advice"
    case celebration = "celebration"
    case venting = "venting"
    case question = "question"
    case resource = "resource"
    case research = "research"
    case treatment = "treatment"
    case lifestyle = "lifestyle"
    case mental_health = "mental_health"
    case relationships = "relationships"
    case work = "work"
    case exercise = "exercise"
    case nutrition = "nutrition"
    case medication = "medication"
    case symptoms = "symptoms"
    case diagnosis = "diagnosis"
    case insurance = "insurance"
    case accessibility = "accessibility"
    
    var displayName: String {
        switch self {
        case .general: return "General Discussion"
        case .support: return "Support & Encouragement"
        case .advice: return "Advice & Tips"
        case .celebration: return "Celebrations & Wins"
        case .venting: return "Venting & Frustrations"
        case .question: return "Questions & Help"
        case .resource: return "Resources & Tools"
        case .research: return "Research & Studies"
        case .treatment: return "Treatments & Therapies"
        case .lifestyle: return "Lifestyle & Daily Living"
        case .mental_health: return "Mental Health"
        case .relationships: return "Relationships & Family"
        case .work: return "Work & Career"
        case .exercise: return "Exercise & Movement"
        case .nutrition: return "Nutrition & Diet"
        case .medication: return "Medications"
        case .symptoms: return "Symptoms & Flares"
        case .diagnosis: return "Diagnosis Journey"
        case .insurance: return "Insurance & Healthcare"
        case .accessibility: return "Accessibility"
        }
    }
    
    var icon: String {
        switch self {
        case .general: return "bubble.left.and.bubble.right"
        case .support: return "heart.circle"
        case .advice: return "lightbulb"
        case .celebration: return "party.popper"
        case .venting: return "cloud.rain"
        case .question: return "questionmark.circle"
        case .resource: return "book.circle"
        case .research: return "microscope"
        case .treatment: return "cross.circle"
        case .lifestyle: return "house.circle"
        case .mental_health: return "brain.head.profile"
        case .relationships: return "person.2.circle"
        case .work: return "briefcase.circle"
        case .exercise: return "figure.walk.circle"
        case .nutrition: return "leaf.circle"
        case .medication: return "pills.circle"
        case .symptoms: return "thermometer"
        case .diagnosis: return "stethoscope.circle"
        case .insurance: return "doc.text.circle"
        case .accessibility: return "accessibility.circle"
        }
    }
    
    var color: String {
        switch self {
        case .general: return "blue"
        case .support: return "green"
        case .advice: return "yellow"
        case .celebration: return "purple"
        case .venting: return "gray"
        case .question: return "orange"
        case .resource: return "indigo"
        case .research: return "teal"
        case .treatment: return "red"
        case .lifestyle: return "brown"
        case .mental_health: return "pink"
        case .relationships: return "cyan"
        case .work: return "mint"
        case .exercise: return "green"
        case .nutrition: return "orange"
        case .medication: return "blue"
        case .symptoms: return "red"
        case .diagnosis: return "purple"
        case .insurance: return "gray"
        case .accessibility: return "blue"
        }
    }
}

enum ModerationStatus: String, Codable {
    case approved = "approved"
    case pending = "pending"
    case flagged = "flagged"
    case removed = "removed"
    case hidden = "hidden"
}

enum PostVisibility: String, Codable {
    case publicPost = "public"
    case friendsOnly = "friends_only"
    case groupOnly = "group_only"
    case privatePost = "private"
}

enum SupportType: String, Codable {
    case emotional = "emotional"
    case practical = "practical"
    case informational = "informational"
    case crisis = "crisis"
}

struct PostAttachment: Codable, Identifiable {
    let id: String
    let type: AttachmentType
    let url: String
    let thumbnailURL: String?
    let filename: String?
    let fileSize: Int64?
    let mimeType: String?
    let duration: TimeInterval? // For audio/video
    let dimensions: AttachmentDimensions? // For images/video
    let altText: String? // For accessibility
}

enum AttachmentType: String, Codable {
    case image = "image"
    case video = "video"
    case audio = "audio"
    case document = "document"
    case link = "link"
}

struct AttachmentDimensions: Codable {
    let width: Int
    let height: Int
}

struct PostEngagement: Codable {
    let likesCount: Int
    let commentsCount: Int
    let sharesCount: Int
    let viewsCount: Int
    let helpfulVotes: Int
    let reportCount: Int
    let isLikedByCurrentUser: Bool
    let isBookmarkedByCurrentUser: Bool
    let isFollowedByCurrentUser: Bool
}

struct Comment: Codable, Identifiable {
    let id: String
    let postId: String
    let authorId: String
    let authorUsername: String
    let authorDisplayName: String
    let authorProfileImageURL: String?
    let authorRole: SupportRole
    let content: String
    let attachments: [PostAttachment]
    let createdAt: Date
    let updatedAt: Date?
    let isEdited: Bool
    let isAnonymous: Bool
    let parentCommentId: String? // For nested comments
    let likesCount: Int
    let repliesCount: Int
    let isLikedByCurrentUser: Bool
    let moderationStatus: ModerationStatus
    let reportCount: Int
    let isHelpful: Bool
    let helpfulVotes: Int
}

// MARK: - Group Models

struct CommunityGroup: Codable, Identifiable {
    let id: String
    let name: String
    let description: String
    let category: GroupCategory
    let type: GroupType
    let privacy: GroupPrivacy
    let coverImageURL: String?
    let iconURL: String?
    let createdAt: Date
    let creatorId: String
    let moderatorIds: [String]
    let memberCount: Int
    let maxMembers: Int?
    let isVerified: Bool
    let isActive: Bool
    let rules: [GroupRule]
    let tags: [String]
    let location: String?
    let language: String
    let ageRestriction: AgeRestriction?
    let requiresApproval: Bool
    let allowsInvites: Bool
    let stats: GroupStats
}

enum GroupCategory: String, Codable, CaseIterable {
    case support = "support"
    case condition = "condition"
    case treatment = "treatment"
    case lifestyle = "lifestyle"
    case age = "age"
    case location = "location"
    case professional = "professional"
    case research = "research"
    case advocacy = "advocacy"
    case social = "social"
    
    var displayName: String {
        switch self {
        case .support: return "Support Groups"
        case .condition: return "Condition-Specific"
        case .treatment: return "Treatment & Therapy"
        case .lifestyle: return "Lifestyle & Wellness"
        case .age: return "Age-Based"
        case .location: return "Location-Based"
        case .professional: return "Professional"
        case .research: return "Research & Studies"
        case .advocacy: return "Advocacy & Awareness"
        case .social: return "Social & Recreation"
        }
    }
}

enum GroupType: String, Codable {
    case open = "open"
    case closed = "closed"
    case secret = "secret"
    case professional = "professional"
}

enum GroupPrivacy: String, Codable {
    case publicGroup = "public"
    case privateGroup = "private"
    case inviteOnly = "invite_only"
}

enum AgeRestriction: String, Codable {
    case teens = "13-17"
    case youngAdults = "18-25"
    case adults = "26-64"
    case seniors = "65+"
    case adultsOnly = "18+"
}

struct GroupRule: Codable, Identifiable {
    let id: String
    let title: String
    let description: String
    let isRequired: Bool
    let violationConsequence: String
}

struct GroupStats: Codable {
    let totalPosts: Int
    let totalComments: Int
    let activeMembers: Int
    let weeklyActivity: Int
    let averageResponseTime: TimeInterval
    let satisfactionRating: Double
}

struct GroupMembership: Codable, Identifiable {
    let id: String
    let groupId: String
    let userId: String
    let role: GroupRole
    let joinedAt: Date
    let isActive: Bool
    let isMuted: Bool
    let notificationSettings: GroupNotificationSettings
    let contributionScore: Int
    let lastActiveAt: Date
}

enum GroupRole: String, Codable {
    case member = "member"
    case moderator = "moderator"
    case admin = "admin"
    case creator = "creator"
}

struct GroupNotificationSettings: Codable {
    let newPosts: Bool
    let newComments: Bool
    let mentions: Bool
    let announcements: Bool
    let events: Bool
}

// MARK: - Messaging Models

struct DirectMessage: Codable, Identifiable {
    let id: String
    let conversationId: String
    let senderId: String
    let recipientId: String
    let content: String
    let messageType: MessageType
    let attachments: [MessageAttachment]
    let sentAt: Date
    let deliveredAt: Date?
    let readAt: Date?
    let isEdited: Bool
    let editedAt: Date?
    let isDeleted: Bool
    let deletedAt: Date?
    let replyToMessageId: String?
    let isEncrypted: Bool
    let metadata: MessageMetadata?
}

enum MessageType: String, Codable {
    case text = "text"
    case image = "image"
    case video = "video"
    case audio = "audio"
    case file = "file"
    case location = "location"
    case contact = "contact"
    case sticker = "sticker"
    case reaction = "reaction"
    case system = "system"
}

struct MessageAttachment: Codable, Identifiable {
    let id: String
    let type: AttachmentType
    let url: String
    let thumbnailURL: String?
    let filename: String?
    let fileSize: Int64?
    let mimeType: String?
    let duration: TimeInterval?
    let isEncrypted: Bool
}

struct MessageMetadata: Codable {
    let location: MessageLocation?
    let mood: MoodLevel?
    let urgency: MessageUrgency?
    let supportRequest: Bool
    let crisis: Bool
}

struct MessageLocation: Codable {
    let latitude: Double
    let longitude: Double
    let address: String?
}

enum MessageUrgency: String, Codable {
    case low = "low"
    case normal = "normal"
    case high = "high"
    case urgent = "urgent"
}

struct Conversation: Codable, Identifiable {
    let id: String
    let participants: [String]
    let type: ConversationType
    let title: String?
    let lastMessage: DirectMessage?
    let lastActivity: Date
    let unreadCount: Int
    let isMuted: Bool
    let isArchived: Bool
    let isBlocked: Bool
    let createdAt: Date
    let metadata: ConversationMetadata?
}

enum ConversationType: String, Codable {
    case direct = "direct"
    case group = "group"
    case support = "support"
    case crisis = "crisis"
}

struct ConversationMetadata: Codable {
    let isSupport: Bool
    let supportType: SupportType?
    let priority: MessageUrgency?
    let tags: [String]
    let assignedModerator: String?
}

// MARK: - Support Models

struct SupportRequest: Codable, Identifiable {
    let id: String
    let requesterId: String
    let title: String
    let description: String
    let category: SupportCategory
    let urgency: SupportUrgency
    let type: SupportType
    let status: SupportStatus
    let createdAt: Date
    let updatedAt: Date
    let assignedSupporterId: String?
    let assignedAt: Date?
    let resolvedAt: Date?
    let tags: [String]
    let location: String?
    let isAnonymous: Bool
    let attachments: [PostAttachment]
    let responses: [SupportResponse]
    let rating: Int?
    let feedback: String?
}

enum SupportCategory: String, Codable, CaseIterable {
    case emotional = "emotional"
    case practical = "practical"
    case medical = "medical"
    case financial = "financial"
    case legal = "legal"
    case technical = "technical"
    case crisis = "crisis"
    case advocacy = "advocacy"
    
    var displayName: String {
        switch self {
        case .emotional: return "Emotional Support"
        case .practical: return "Practical Help"
        case .medical: return "Medical Questions"
        case .financial: return "Financial Assistance"
        case .legal: return "Legal Advice"
        case .technical: return "Technical Help"
        case .crisis: return "Crisis Support"
        case .advocacy: return "Advocacy"
        }
    }
}

enum SupportUrgency: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case crisis = "crisis"
    
    var displayName: String {
        switch self {
        case .low: return "Low Priority"
        case .medium: return "Medium Priority"
        case .high: return "High Priority"
        case .crisis: return "Crisis - Immediate Help Needed"
        }
    }
    
    var color: String {
        switch self {
        case .low: return "green"
        case .medium: return "yellow"
        case .high: return "orange"
        case .crisis: return "red"
        }
    }
}

enum SupportStatus: String, Codable {
    case open = "open"
    case assigned = "assigned"
    case inProgress = "in_progress"
    case resolved = "resolved"
    case closed = "closed"
    case escalated = "escalated"
}

struct SupportResponse: Codable, Identifiable {
    let id: String
    let supportRequestId: String
    let responderId: String
    let responderRole: SupportRole
    let content: String
    let attachments: [PostAttachment]
    let createdAt: Date
    let isHelpful: Bool
    let helpfulVotes: Int
    let isVerified: Bool
    let verifiedBy: String?
}

// MARK: - Event Models

struct CommunityEvent: Codable, Identifiable {
    let id: String
    let title: String
    let description: String
    let category: EventCategory
    let type: EventType
    let startDate: Date
    let endDate: Date
    let timeZone: String
    let location: EventLocation?
    let organizerId: String
    let groupId: String?
    let maxAttendees: Int?
    let currentAttendees: Int
    let isPublic: Bool
    let requiresApproval: Bool
    let cost: EventCost?
    let tags: [String]
    let imageURL: String?
    let agenda: [EventAgendaItem]
    let resources: [EventResource]
    let attendees: [EventAttendee]
    let createdAt: Date
    let updatedAt: Date
    let status: EventStatus
}

enum EventCategory: String, Codable, CaseIterable {
    case support = "support"
    case educational = "educational"
    case social = "social"
    case exercise = "exercise"
    case meditation = "meditation"
    case advocacy = "advocacy"
    case fundraising = "fundraising"
    case awareness = "awareness"
    case research = "research"
    case professional = "professional"
    
    var displayName: String {
        switch self {
        case .support: return "Support Group"
        case .educational: return "Educational"
        case .social: return "Social Gathering"
        case .exercise: return "Exercise & Movement"
        case .meditation: return "Meditation & Mindfulness"
        case .advocacy: return "Advocacy"
        case .fundraising: return "Fundraising"
        case .awareness: return "Awareness Campaign"
        case .research: return "Research Study"
        case .professional: return "Professional Development"
        }
    }
}

enum EventType: String, Codable {
    case inPerson = "in_person"
    case virtual = "virtual"
    case hybrid = "hybrid"
}

struct EventLocation: Codable {
    let name: String?
    let address: String?
    let city: String?
    let state: String?
    let country: String?
    let latitude: Double?
    let longitude: Double?
    let virtualLink: String?
    let accessibilityInfo: String?
}

struct EventCost: Codable {
    let amount: Double
    let currency: String
    let isFree: Bool
    let scholarshipAvailable: Bool
}

struct EventAgendaItem: Codable, Identifiable {
    let id: String
    let title: String
    let description: String?
    let startTime: Date
    let endTime: Date
    let speakerId: String?
    let speakerName: String?
    let location: String?
}

struct EventResource: Codable, Identifiable {
    let id: String
    let title: String
    let description: String?
    let type: ResourceType
    let url: String
    let isRequired: Bool
}

enum ResourceType: String, Codable {
    case document = "document"
    case video = "video"
    case audio = "audio"
    case link = "link"
    case presentation = "presentation"
}

struct EventAttendee: Codable, Identifiable {
    let id: String
    let userId: String
    let eventId: String
    let status: AttendeeStatus
    let registeredAt: Date
    let checkedInAt: Date?
    let feedback: EventFeedback?
}

enum AttendeeStatus: String, Codable {
    case registered = "registered"
    case confirmed = "confirmed"
    case attended = "attended"
    case noShow = "no_show"
    case cancelled = "cancelled"
}

struct EventFeedback: Codable {
    let rating: Int
    let comment: String?
    let wouldRecommend: Bool
    let submittedAt: Date
}

enum EventStatus: String, Codable {
    case draft = "draft"
    case published = "published"
    case cancelled = "cancelled"
    case completed = "completed"
}

// MARK: - Social Support Network Manager

class SocialSupportNetworkManager: NSObject, ObservableObject {
    // Core Data
    private let context: NSManagedObjectContext
    
    // CloudKit
    private let cloudKitContainer = CKContainer.default()
    
    // Network
    private let networkMonitor = NWPathMonitor()
    private let networkQueue = DispatchQueue(label: "NetworkMonitor")
    
    // Published Properties
    @Published var currentUser: CommunityUser?
    @Published var communityPosts: [CommunityPost] = []
    @Published var userGroups: [CommunityGroup] = []
    @Published var conversations: [Conversation] = []
    @Published var supportRequests: [SupportRequest] = []
    @Published var upcomingEvents: [CommunityEvent] = []
    @Published var friends: [CommunityUser] = []
    @Published var notifications: [CommunityNotification] = []
    
    // Connection Status
    @Published var isConnected = false
    @Published var connectionQuality: ConnectionQuality = .unknown
    
    // Real-time Updates
    @Published var unreadMessagesCount = 0
    @Published var unreadNotificationsCount = 0
    @Published var activeSupportRequests = 0
    
    // Filters and Search
    @Published var selectedCategory: PostCategory?
    @Published var searchQuery = ""
    @Published var sortOption: SortOption = .recent
    
    // Moderation
    @Published var reportedContent: [ReportedContent] = []
    @Published var moderationQueue: [ModerationItem] = []
    
    // Internal State
    private var cancellables = Set<AnyCancellable>()
    private var realTimeSubscription: CKQuerySubscription?
    private var messageEncryption = MessageEncryption()
    
    override init() {
        self.context = InflamAIPersistenceController.shared.container.viewContext
        super.init()
        
        setupNetworkMonitoring()
        setupCloudKitSubscriptions()
        loadCurrentUser()
        loadInitialData()
        setupNotifications()
    }
    
    // MARK: - Setup
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isConnected = path.status == .satisfied
                self?.connectionQuality = self?.determineConnectionQuality(path) ?? .unknown
            }
        }
        networkMonitor.start(queue: networkQueue)
    }
    
    private func determineConnectionQuality(_ path: NWPath) -> ConnectionQuality {
        if path.isExpensive {
            return .cellular
        } else if path.usesInterfaceType(.wifi) {
            return .wifi
        } else if path.usesInterfaceType(.wiredEthernet) {
            return .ethernet
        } else {
            return .unknown
        }
    }
    
    private func setupCloudKitSubscriptions() {
        // Setup real-time subscriptions for posts, messages, etc.
    }
    
    private func loadCurrentUser() {
        Task {
            do {
                let user = try await fetchCurrentUser()
                DispatchQueue.main.async {
                    self.currentUser = user
                }
            } catch {
                print("Error loading current user: \(error)")
            }
        }
    }
    
    private func loadInitialData() {
        Task {
            await loadCommunityPosts()
            await loadUserGroups()
            await loadConversations()
            await loadSupportRequests()
            await loadUpcomingEvents()
            await loadFriends()
            await loadNotifications()
        }
    }
    
    private func setupNotifications() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error = error {
                print("Notification authorization error: \(error)")
            }
        }
    }
    
    // MARK: - Community Posts
    
    func createPost(_ content: String, category: PostCategory, isAnonymous: Bool = false, attachments: [PostAttachment] = [], tags: [String] = [], mood: MoodLevel? = nil, painLevel: Int? = nil, triggerWarning: Bool = false, supportType: SupportType? = nil) async throws {
        guard let currentUser = currentUser else {
            throw SocialNetworkError.userNotAuthenticated
        }
        
        let post = CommunityPost(
            id: UUID().uuidString,
            authorId: currentUser.id,
            authorUsername: isAnonymous ? "Anonymous" : currentUser.username,
            authorDisplayName: isAnonymous ? "Anonymous User" : currentUser.displayName,
            authorProfileImageURL: isAnonymous ? nil : currentUser.profileImageURL,
            authorRole: currentUser.supportRole,
            content: content,
            contentType: determineContentType(content, attachments),
            category: category,
            tags: tags,
            attachments: attachments,
            createdAt: Date(),
            updatedAt: nil,
            isEdited: false,
            isAnonymous: isAnonymous,
            isPinned: false,
            isLocked: false,
            isReported: false,
            reportCount: 0,
            moderationStatus: .pending,
            visibility: .publicPost,
            groupId: nil,
            parentPostId: nil,
            threadId: nil,
            engagement: PostEngagement(
                likesCount: 0,
                commentsCount: 0,
                sharesCount: 0,
                viewsCount: 0,
                helpfulVotes: 0,
                reportCount: 0,
                isLikedByCurrentUser: false,
                isBookmarkedByCurrentUser: false,
                isFollowedByCurrentUser: false
            ),
            location: nil,
            mood: mood,
            painLevel: painLevel,
            triggerWarning: triggerWarning,
            supportType: supportType
        )
        
        try await savePost(post)
        
        DispatchQueue.main.async {
            self.communityPosts.insert(post, at: 0)
        }
        
        // Send notifications to relevant users
        await notifyRelevantUsers(for: post)
    }
    
    func likePost(_ postId: String) async throws {
        guard let currentUser = currentUser else {
            throw SocialNetworkError.userNotAuthenticated
        }
        
        try await togglePostLike(postId, userId: currentUser.id)
        
        DispatchQueue.main.async {
            if let index = self.communityPosts.firstIndex(where: { $0.id == postId }) {
                var post = self.communityPosts[index]
                var engagement = post.engagement
                
                if engagement.isLikedByCurrentUser {
                    engagement = PostEngagement(
                        likesCount: max(engagement.likesCount - 1, 0),
                        commentsCount: engagement.commentsCount,
                        sharesCount: engagement.sharesCount,
                        viewsCount: engagement.viewsCount,
                        helpfulVotes: engagement.helpfulVotes,
                        reportCount: engagement.reportCount,
                        isLikedByCurrentUser: false,
                        isBookmarkedByCurrentUser: engagement.isBookmarkedByCurrentUser,
                        isFollowedByCurrentUser: engagement.isFollowedByCurrentUser
                    )
                } else {
                    engagement = PostEngagement(
                        likesCount: engagement.likesCount + 1,
                        commentsCount: engagement.commentsCount,
                        sharesCount: engagement.sharesCount,
                        viewsCount: engagement.viewsCount,
                        helpfulVotes: engagement.helpfulVotes,
                        reportCount: engagement.reportCount,
                        isLikedByCurrentUser: true,
                        isBookmarkedByCurrentUser: engagement.isBookmarkedByCurrentUser,
                        isFollowedByCurrentUser: engagement.isFollowedByCurrentUser
                    )
                }
                
                post = CommunityPost(
                    id: post.id,
                    authorId: post.authorId,
                    authorUsername: post.authorUsername,
                    authorDisplayName: post.authorDisplayName,
                    authorProfileImageURL: post.authorProfileImageURL,
                    authorRole: post.authorRole,
                    content: post.content,
                    contentType: post.contentType,
                    category: post.category,
                    tags: post.tags,
                    attachments: post.attachments,
                    createdAt: post.createdAt,
                    updatedAt: post.updatedAt,
                    isEdited: post.isEdited,
                    isAnonymous: post.isAnonymous,
                    isPinned: post.isPinned,
                    isLocked: post.isLocked,
                    isReported: post.isReported,
                    reportCount: post.reportCount,
                    moderationStatus: post.moderationStatus,
                    visibility: post.visibility,
                    groupId: post.groupId,
                    parentPostId: post.parentPostId,
                    threadId: post.threadId,
                    engagement: engagement,
                    location: post.location,
                    mood: post.mood,
                    painLevel: post.painLevel,
                    triggerWarning: post.triggerWarning,
                    supportType: post.supportType
                )
                
                self.communityPosts[index] = post
            }
        }
    }
    
    func addComment(to postId: String, content: String, isAnonymous: Bool = false, attachments: [PostAttachment] = []) async throws {
        guard let currentUser = currentUser else {
            throw SocialNetworkError.userNotAuthenticated
        }
        
        let comment = Comment(
            id: UUID().uuidString,
            postId: postId,
            authorId: currentUser.id,
            authorUsername: isAnonymous ? "Anonymous" : currentUser.username,
            authorDisplayName: isAnonymous ? "Anonymous User" : currentUser.displayName,
            authorProfileImageURL: isAnonymous ? nil : currentUser.profileImageURL,
            authorRole: currentUser.supportRole,
            content: content,
            attachments: attachments,
            createdAt: Date(),
            updatedAt: nil,
            isEdited: false,
            isAnonymous: isAnonymous,
            parentCommentId: nil,
            likesCount: 0,
            repliesCount: 0,
            isLikedByCurrentUser: false,
            moderationStatus: .pending,
            reportCount: 0,
            isHelpful: false,
            helpfulVotes: 0
        )
        
        try await saveComment(comment)
        
        // Update post comment count
        DispatchQueue.main.async {
            if let index = self.communityPosts.firstIndex(where: { $0.id == postId }) {
                var post = self.communityPosts[index]
                var engagement = post.engagement
                engagement = PostEngagement(
                    likesCount: engagement.likesCount,
                    commentsCount: engagement.commentsCount + 1,
                    sharesCount: engagement.sharesCount,
                    viewsCount: engagement.viewsCount,
                    helpfulVotes: engagement.helpfulVotes,
                    reportCount: engagement.reportCount,
                    isLikedByCurrentUser: engagement.isLikedByCurrentUser,
                    isBookmarkedByCurrentUser: engagement.isBookmarkedByCurrentUser,
                    isFollowedByCurrentUser: engagement.isFollowedByCurrentUser
                )
                
                post = CommunityPost(
                    id: post.id,
                    authorId: post.authorId,
                    authorUsername: post.authorUsername,
                    authorDisplayName: post.authorDisplayName,
                    authorProfileImageURL: post.authorProfileImageURL,
                    authorRole: post.authorRole,
                    content: post.content,
                    contentType: post.contentType,
                    category: post.category,
                    tags: post.tags,
                    attachments: post.attachments,
                    createdAt: post.createdAt,
                    updatedAt: post.updatedAt,
                    isEdited: post.isEdited,
                    isAnonymous: post.isAnonymous,
                    isPinned: post.isPinned,
                    isLocked: post.isLocked,
                    isReported: post.isReported,
                    reportCount: post.reportCount,
                    moderationStatus: post.moderationStatus,
                    visibility: post.visibility,
                    groupId: post.groupId,
                    parentPostId: post.parentPostId,
                    threadId: post.threadId,
                    engagement: engagement,
                    location: post.location,
                    mood: post.mood,
                    painLevel: post.painLevel,
                    triggerWarning: post.triggerWarning,
                    supportType: post.supportType
                )
                
                self.communityPosts[index] = post
            }
        }
    }
    
    // MARK: - Support Requests
    
    func createSupportRequest(_ title: String, description: String, category: SupportCategory, urgency: SupportUrgency, type: SupportType, isAnonymous: Bool = false, attachments: [PostAttachment] = []) async throws {
        guard let currentUser = currentUser else {
            throw SocialNetworkError.userNotAuthenticated
        }
        
        let request = SupportRequest(
            id: UUID().uuidString,
            requesterId: currentUser.id,
            title: title,
            description: description,
            category: category,
            urgency: urgency,
            type: type,
            status: .open,
            createdAt: Date(),
            updatedAt: Date(),
            assignedSupporterId: nil,
            assignedAt: nil,
            resolvedAt: nil,
            tags: [],
            location: nil,
            isAnonymous: isAnonymous,
            attachments: attachments,
            responses: [],
            rating: nil,
            feedback: nil
        )
        
        try await saveSupportRequest(request)
        
        DispatchQueue.main.async {
            self.supportRequests.insert(request, at: 0)
            self.activeSupportRequests += 1
        }
        
        // Notify potential supporters
        await notifyPotentialSupporters(for: request)
        
        // If crisis, trigger emergency protocols
        if urgency == .crisis {
            await handleCrisisRequest(request)
        }
    }
    
    // MARK: - Messaging
    
    func sendDirectMessage(to recipientId: String, content: String, type: MessageType = .text, attachments: [MessageAttachment] = [], urgency: MessageUrgency = .normal, supportRequest: Bool = false) async throws {
        guard let currentUser = currentUser else {
            throw SocialNetworkError.userNotAuthenticated
        }
        
        let conversationId = generateConversationId(currentUser.id, recipientId)
        
        let message = DirectMessage(
            id: UUID().uuidString,
            conversationId: conversationId,
            senderId: currentUser.id,
            recipientId: recipientId,
            content: content,
            messageType: type,
            attachments: attachments,
            sentAt: Date(),
            deliveredAt: nil,
            readAt: nil,
            isEdited: false,
            editedAt: nil,
            isDeleted: false,
            deletedAt: nil,
            replyToMessageId: nil,
            isEncrypted: true,
            metadata: MessageMetadata(
                location: nil,
                mood: nil,
                urgency: urgency,
                supportRequest: supportRequest,
                crisis: urgency == .urgent
            )
        )
        
        // Encrypt message
        let encryptedMessage = try messageEncryption.encrypt(message)
        
        try await saveMessage(encryptedMessage)
        
        // Update conversation
        await updateConversation(conversationId, with: message)
        
        // Send push notification
        await sendMessageNotification(to: recipientId, from: currentUser, message: message)
    }
    
    // MARK: - Groups
    
    func joinGroup(_ groupId: String) async throws {
        guard let currentUser = currentUser else {
            throw SocialNetworkError.userNotAuthenticated
        }
        
        let membership = GroupMembership(
            id: UUID().uuidString,
            groupId: groupId,
            userId: currentUser.id,
            role: .member,
            joinedAt: Date(),
            isActive: true,
            isMuted: false,
            notificationSettings: GroupNotificationSettings(
                newPosts: true,
                newComments: true,
                mentions: true,
                announcements: true,
                events: true
            ),
            contributionScore: 0,
            lastActiveAt: Date()
        )
        
        try await saveGroupMembership(membership)
        
        // Update local state
        await loadUserGroups()
    }
    
    // MARK: - Events
    
    func registerForEvent(_ eventId: String) async throws {
        guard let currentUser = currentUser else {
            throw SocialNetworkError.userNotAuthenticated
        }
        
        let attendee = EventAttendee(
            id: UUID().uuidString,
            userId: currentUser.id,
            eventId: eventId,
            status: .registered,
            registeredAt: Date(),
            checkedInAt: nil,
            feedback: nil
        )
        
        try await saveEventAttendee(attendee)
        
        // Update event attendee count
        await loadUpcomingEvents()
    }
    
    // MARK: - Moderation
    
    func reportContent(_ contentId: String, type: ContentType, reason: ReportReason, description: String?) async throws {
        guard let currentUser = currentUser else {
            throw SocialNetworkError.userNotAuthenticated
        }
        
        let report = ContentReport(
            id: UUID().uuidString,
            contentId: contentId,
            contentType: type,
            reporterId: currentUser.id,
            reason: reason,
            description: description,
            createdAt: Date(),
            status: .pending,
            reviewedBy: nil,
            reviewedAt: nil,
            action: nil
        )
        
        try await saveContentReport(report)
        
        // Notify moderators
        await notifyModerators(of: report)
    }
    
    // MARK: - Search and Filtering
    
    func searchPosts(query: String, category: PostCategory? = nil, sortBy: SortOption = .recent) async -> [CommunityPost] {
        // Implement search functionality
        return []
    }
    
    func filterPosts(by category: PostCategory) {
        selectedCategory = category
        Task {
            await loadCommunityPosts()
        }
    }
    
    // MARK: - Private Methods
    
    private func loadCommunityPosts() async {
        // Load posts from API/CloudKit
    }
    
    private func loadUserGroups() async {
        // Load user's groups
    }
    
    private func loadConversations() async {
        // Load user's conversations
    }
    
    private func loadSupportRequests() async {
        // Load support requests
    }
    
    private func loadUpcomingEvents() async {
        // Load upcoming events
    }
    
    private func loadFriends() async {
        // Load user's friends
    }
    
    private func loadNotifications() async {
        // Load notifications
    }
    
    private func fetchCurrentUser() async throws -> CommunityUser {
        // Fetch current user from API
        throw SocialNetworkError.userNotFound
    }
    
    private func savePost(_ post: CommunityPost) async throws {
        // Save post to CloudKit/API
    }
    
    private func saveComment(_ comment: Comment) async throws {
        // Save comment to CloudKit/API
    }
    
    private func saveSupportRequest(_ request: SupportRequest) async throws {
        // Save support request to CloudKit/API
    }
    
    private func saveMessage(_ message: DirectMessage) async throws {
        // Save message to CloudKit/API
    }
    
    private func saveGroupMembership(_ membership: GroupMembership) async throws {
        // Save group membership to CloudKit/API
    }
    
    private func saveEventAttendee(_ attendee: EventAttendee) async throws {
        // Save event attendee to CloudKit/API
    }
    
    private func saveContentReport(_ report: ContentReport) async throws {
        // Save content report to CloudKit/API
    }
    
    private func togglePostLike(_ postId: String, userId: String) async throws {
        // Toggle post like in CloudKit/API
    }
    
    private func updateConversation(_ conversationId: String, with message: DirectMessage) async {
        // Update conversation with latest message
    }
    
    private func notifyRelevantUsers(for post: CommunityPost) async {
        // Send notifications to relevant users
    }
    
    private func notifyPotentialSupporters(for request: SupportRequest) async {
        // Notify users who can provide support
    }
    
    private func handleCrisisRequest(_ request: SupportRequest) async {
        // Handle crisis support request with emergency protocols
    }
    
    private func sendMessageNotification(to recipientId: String, from sender: CommunityUser, message: DirectMessage) async {
        // Send push notification for new message
    }
    
    private func notifyModerators(of report: ContentReport) async {
        // Notify moderators of new content report
    }
    
    private func determineContentType(_ content: String, _ attachments: [PostAttachment]) -> PostContentType {
        if !attachments.isEmpty {
            return PostContentType(rawValue: attachments.first?.type.rawValue ?? "text") ?? .text
        }
        
        if content.contains("?") {
            return .question
        }
        
        return .text
    }
    
    private func generateConversationId(_ userId1: String, _ userId2: String) -> String {
        let sortedIds = [userId1, userId2].sorted()
        return "\(sortedIds[0])_\(sortedIds[1])"
    }
}

// MARK: - Supporting Types

enum ConnectionQuality {
    case unknown
    case cellular
    case wifi
    case ethernet
}

enum SortOption: String, CaseIterable {
    case recent = "recent"
    case popular = "popular"
    case helpful = "helpful"
    case trending = "trending"
    
    var displayName: String {
        switch self {
        case .recent: return "Most Recent"
        case .popular: return "Most Popular"
        case .helpful: return "Most Helpful"
        case .trending: return "Trending"
        }
    }
}

struct CommunityNotification: Codable, Identifiable {
    let id: String
    let type: NotificationType
    let title: String
    let message: String
    let data: [String: String]
    let createdAt: Date
    let isRead: Bool
    let actionURL: String?
}

enum NotificationType: String, Codable {
    case newMessage = "new_message"
    case newPost = "new_post"
    case newComment = "new_comment"
    case postLiked = "post_liked"
    case friendRequest = "friend_request"
    case groupInvite = "group_invite"
    case eventReminder = "event_reminder"
    case supportRequest = "support_request"
    case crisis = "crisis"
    case mention = "mention"
    case achievement = "achievement"
}

struct ReportedContent: Codable, Identifiable {
    let id: String
    let contentId: String
    let contentType: ContentType
    let reportCount: Int
    let lastReportedAt: Date
    let status: ModerationStatus
}

struct ModerationItem: Codable, Identifiable {
    let id: String
    let contentId: String
    let contentType: ContentType
    let priority: ModerationPriority
    let createdAt: Date
    let assignedModerator: String?
}

enum ContentType: String, Codable {
    case post = "post"
    case comment = "comment"
    case message = "message"
    case user = "user"
    case group = "group"
}

enum ReportReason: String, Codable, CaseIterable {
    case spam = "spam"
    case harassment = "harassment"
    case hateSpeech = "hate_speech"
    case violence = "violence"
    case misinformation = "misinformation"
    case inappropriateContent = "inappropriate_content"
    case copyright = "copyright"
    case other = "other"
    
    var displayName: String {
        switch self {
        case .spam: return "Spam"
        case .harassment: return "Harassment"
        case .hateSpeech: return "Hate Speech"
        case .violence: return "Violence or Threats"
        case .misinformation: return "Misinformation"
        case .inappropriateContent: return "Inappropriate Content"
        case .copyright: return "Copyright Violation"
        case .other: return "Other"
        }
    }
}

struct ContentReport: Codable, Identifiable {
    let id: String
    let contentId: String
    let contentType: ContentType
    let reporterId: String
    let reason: ReportReason
    let description: String?
    let createdAt: Date
    let status: ReportStatus
    let reviewedBy: String?
    let reviewedAt: Date?
    let action: ModerationAction?
}

enum ReportStatus: String, Codable {
    case pending = "pending"
    case reviewing = "reviewing"
    case resolved = "resolved"
    case dismissed = "dismissed"
}

enum ModerationAction: String, Codable {
    case noAction = "no_action"
    case warning = "warning"
    case contentRemoval = "content_removal"
    case userSuspension = "user_suspension"
    case userBan = "user_ban"
}

enum ModerationPriority: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case urgent = "urgent"
}

// MARK: - Message Encryption

class MessageEncryption {
    private let keychain = Keychain()
    
    func encrypt(_ message: DirectMessage) throws -> DirectMessage {
        // Implement end-to-end encryption
        return message
    }
    
    func decrypt(_ message: DirectMessage) throws -> DirectMessage {
        // Implement decryption
        return message
    }
}

// MARK: - Errors

enum SocialNetworkError: Error {
    case userNotAuthenticated
    case userNotFound
    case networkError
    case encryptionError
    case moderationError
    case permissionDenied
    
    var localizedDescription: String {
        switch self {
        case .userNotAuthenticated:
            return "User not authenticated"
        case .userNotFound:
            return "User not found"
        case .networkError:
            return "Network connection error"
        case .encryptionError:
            return "Message encryption error"
        case .moderationError:
            return "Content moderation error"
        case .permissionDenied:
            return "Permission denied"
        }
    }
}