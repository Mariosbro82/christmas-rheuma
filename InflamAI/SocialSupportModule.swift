//
//  SocialSupportModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import SwiftUI
import Combine
import CryptoKit
import Network

// MARK: - User Profile Models

struct UserProfile: Identifiable, Codable {
    let id: UUID
    var username: String
    var displayName: String
    var bio: String?
    var avatarURL: String?
    var location: String?
    var ageRange: AgeRange?
    var diagnosisDate: Date?
    var conditions: [MedicalCondition]
    var interests: [Interest]
    var privacySettings: PrivacySettings
    var supportPreferences: SupportPreferences
    var verificationStatus: VerificationStatus
    var joinDate: Date
    var lastActive: Date
    var isOnline: Bool
    var supporterRating: Double
    var helpfulnessScore: Int
    var totalPosts: Int
    var totalComments: Int
    var totalLikes: Int
    var badges: [UserBadge]
    var blockedUsers: [UUID]
    var reportedUsers: [UUID]
    
    init(id: UUID = UUID(), username: String, displayName: String) {
        self.id = id
        self.username = username
        self.displayName = displayName
        self.bio = nil
        self.avatarURL = nil
        self.location = nil
        self.ageRange = nil
        self.diagnosisDate = nil
        self.conditions = []
        self.interests = []
        self.privacySettings = PrivacySettings()
        self.supportPreferences = SupportPreferences()
        self.verificationStatus = .unverified
        self.joinDate = Date()
        self.lastActive = Date()
        self.isOnline = false
        self.supporterRating = 0.0
        self.helpfulnessScore = 0
        self.totalPosts = 0
        self.totalComments = 0
        self.totalLikes = 0
        self.badges = []
        self.blockedUsers = []
        self.reportedUsers = []
    }
}

enum AgeRange: String, CaseIterable, Codable {
    case teens = "13-19"
    case twenties = "20-29"
    case thirties = "30-39"
    case forties = "40-49"
    case fifties = "50-59"
    case sixties = "60-69"
    case seventies = "70+"
    case preferNotToSay = "prefer_not_to_say"
    
    var displayName: String {
        switch self {
        case .teens: return "13-19"
        case .twenties: return "20-29"
        case .thirties: return "30-39"
        case .forties: return "40-49"
        case .fifties: return "50-59"
        case .sixties: return "60-69"
        case .seventies: return "70+"
        case .preferNotToSay: return "Prefer not to say"
        }
    }
}

struct MedicalCondition: Identifiable, Codable, Hashable {
    let id: UUID
    let name: String
    let category: ConditionCategory
    let severity: ConditionSeverity?
    let diagnosisDate: Date?
    let isVisible: Bool
    
    init(id: UUID = UUID(), name: String, category: ConditionCategory, severity: ConditionSeverity? = nil, diagnosisDate: Date? = nil, isVisible: Bool = true) {
        self.id = id
        self.name = name
        self.category = category
        self.severity = severity
        self.diagnosisDate = diagnosisDate
        self.isVisible = isVisible
    }
}

enum ConditionCategory: String, CaseIterable, Codable {
    case rheumatoid = "rheumatoid"
    case osteoarthritis = "osteoarthritis"
    case fibromyalgia = "fibromyalgia"
    case lupus = "lupus"
    case psoriatic = "psoriatic"
    case ankylosing = "ankylosing"
    case gout = "gout"
    case other = "other"
    
    var displayName: String {
        switch self {
        case .rheumatoid: return "Rheumatoid Arthritis"
        case .osteoarthritis: return "Osteoarthritis"
        case .fibromyalgia: return "Fibromyalgia"
        case .lupus: return "Lupus"
        case .psoriatic: return "Psoriatic Arthritis"
        case .ankylosing: return "Ankylosing Spondylitis"
        case .gout: return "Gout"
        case .other: return "Other"
        }
    }
}

enum ConditionSeverity: String, CaseIterable, Codable {
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
    
    var displayName: String {
        switch self {
        case .mild: return "Mild"
        case .moderate: return "Moderate"
        case .severe: return "Severe"
        }
    }
    
    var color: String {
        switch self {
        case .mild: return "green"
        case .moderate: return "yellow"
        case .severe: return "red"
        }
    }
}

struct Interest: Identifiable, Codable, Hashable {
    let id: UUID
    let name: String
    let category: InterestCategory
    
    init(id: UUID = UUID(), name: String, category: InterestCategory) {
        self.id = id
        self.name = name
        self.category = category
    }
}

enum InterestCategory: String, CaseIterable, Codable {
    case exercise = "exercise"
    case nutrition = "nutrition"
    case mentalHealth = "mental_health"
    case hobbies = "hobbies"
    case career = "career"
    case family = "family"
    case travel = "travel"
    case technology = "technology"
    
    var displayName: String {
        switch self {
        case .exercise: return "Exercise & Fitness"
        case .nutrition: return "Nutrition"
        case .mentalHealth: return "Mental Health"
        case .hobbies: return "Hobbies"
        case .career: return "Career"
        case .family: return "Family"
        case .travel: return "Travel"
        case .technology: return "Technology"
        }
    }
}

struct PrivacySettings: Codable {
    var profileVisibility: ProfileVisibility
    var showLocation: Bool
    var showAge: Bool
    var showConditions: Bool
    var showDiagnosisDate: Bool
    var allowDirectMessages: Bool
    var allowGroupInvites: Bool
    var showOnlineStatus: Bool
    var dataSharing: DataSharingSettings
    
    init() {
        self.profileVisibility = .friendsOnly
        self.showLocation = false
        self.showAge = false
        self.showConditions = true
        self.showDiagnosisDate = false
        self.allowDirectMessages = true
        self.allowGroupInvites = true
        self.showOnlineStatus = true
        self.dataSharing = DataSharingSettings()
    }
}

enum ProfileVisibility: String, CaseIterable, Codable {
    case everyone = "everyone"
    case friendsOnly = "friends_only"
    case private = "private"
    
    var displayName: String {
        switch self {
        case .everyone: return "Everyone"
        case .friendsOnly: return "Friends Only"
        case .private: return "Private"
        }
    }
}

struct DataSharingSettings: Codable {
    var shareSymptomData: Bool
    var shareMedicationData: Bool
    var shareExerciseData: Bool
    var shareMoodData: Bool
    var allowResearchParticipation: Bool
    
    init() {
        self.shareSymptomData = false
        self.shareMedicationData = false
        self.shareExerciseData = false
        self.shareMoodData = false
        self.allowResearchParticipation = false
    }
}

struct SupportPreferences: Codable {
    var preferredCommunicationStyle: CommunicationStyle
    var availabilityHours: AvailabilityHours
    var supportTopics: [SupportTopic]
    var mentorshipInterest: MentorshipInterest
    var crisisSupport: Bool
    var professionalReferrals: Bool
    
    init() {
        self.preferredCommunicationStyle = .casual
        self.availabilityHours = AvailabilityHours()
        self.supportTopics = []
        self.mentorshipInterest = .none
        self.crisisSupport = false
        self.professionalReferrals = false
    }
}

enum CommunicationStyle: String, CaseIterable, Codable {
    case casual = "casual"
    case formal = "formal"
    case empathetic = "empathetic"
    case practical = "practical"
    
    var displayName: String {
        switch self {
        case .casual: return "Casual"
        case .formal: return "Formal"
        case .empathetic: return "Empathetic"
        case .practical: return "Practical"
        }
    }
}

struct AvailabilityHours: Codable {
    var startTime: String
    var endTime: String
    var timezone: String
    var availableDays: [DayOfWeek]
    
    init() {
        self.startTime = "09:00"
        self.endTime = "17:00"
        self.timezone = TimeZone.current.identifier
        self.availableDays = DayOfWeek.allCases
    }
}

enum DayOfWeek: String, CaseIterable, Codable {
    case monday = "monday"
    case tuesday = "tuesday"
    case wednesday = "wednesday"
    case thursday = "thursday"
    case friday = "friday"
    case saturday = "saturday"
    case sunday = "sunday"
    
    var displayName: String {
        switch self {
        case .monday: return "Monday"
        case .tuesday: return "Tuesday"
        case .wednesday: return "Wednesday"
        case .thursday: return "Thursday"
        case .friday: return "Friday"
        case .saturday: return "Saturday"
        case .sunday: return "Sunday"
        }
    }
}

enum SupportTopic: String, CaseIterable, Codable {
    case painManagement = "pain_management"
    case medicationSupport = "medication_support"
    case exerciseMotivation = "exercise_motivation"
    case emotionalSupport = "emotional_support"
    case practicalAdvice = "practical_advice"
    case careerSupport = "career_support"
    case relationshipAdvice = "relationship_advice"
    case dailyLiving = "daily_living"
    
    var displayName: String {
        switch self {
        case .painManagement: return "Pain Management"
        case .medicationSupport: return "Medication Support"
        case .exerciseMotivation: return "Exercise Motivation"
        case .emotionalSupport: return "Emotional Support"
        case .practicalAdvice: return "Practical Advice"
        case .careerSupport: return "Career Support"
        case .relationshipAdvice: return "Relationship Advice"
        case .dailyLiving: return "Daily Living"
        }
    }
}

enum MentorshipInterest: String, CaseIterable, Codable {
    case none = "none"
    case mentor = "mentor"
    case mentee = "mentee"
    case both = "both"
    
    var displayName: String {
        switch self {
        case .none: return "Not Interested"
        case .mentor: return "Want to Mentor"
        case .mentee: return "Want a Mentor"
        case .both: return "Both"
        }
    }
}

enum VerificationStatus: String, Codable {
    case unverified = "unverified"
    case pending = "pending"
    case verified = "verified"
    case rejected = "rejected"
    
    var displayName: String {
        switch self {
        case .unverified: return "Unverified"
        case .pending: return "Pending"
        case .verified: return "Verified"
        case .rejected: return "Rejected"
        }
    }
}

struct UserBadge: Identifiable, Codable {
    let id: UUID
    let name: String
    let description: String
    let iconName: String
    let category: BadgeCategory
    let earnedDate: Date
    let isVisible: Bool
    
    init(id: UUID = UUID(), name: String, description: String, iconName: String, category: BadgeCategory, earnedDate: Date = Date(), isVisible: Bool = true) {
        self.id = id
        self.name = name
        self.description = description
        self.iconName = iconName
        self.category = category
        self.earnedDate = earnedDate
        self.isVisible = isVisible
    }
}

enum BadgeCategory: String, CaseIterable, Codable {
    case helper = "helper"
    case supporter = "supporter"
    case contributor = "contributor"
    case mentor = "mentor"
    case veteran = "veteran"
    case specialist = "specialist"
    
    var displayName: String {
        switch self {
        case .helper: return "Helper"
        case .supporter: return "Supporter"
        case .contributor: return "Contributor"
        case .mentor: return "Mentor"
        case .veteran: return "Veteran"
        case .specialist: return "Specialist"
        }
    }
}

// MARK: - Community Models

struct CommunityGroup: Identifiable, Codable {
    let id: UUID
    var name: String
    var description: String
    var category: GroupCategory
    var type: GroupType
    var privacy: GroupPrivacy
    var imageURL: String?
    var createdBy: UUID
    var moderators: [UUID]
    var members: [UUID]
    var memberCount: Int
    var rules: [GroupRule]
    var tags: [String]
    var location: String?
    var meetingSchedule: MeetingSchedule?
    var createdDate: Date
    var lastActivity: Date
    var isActive: Bool
    var maxMembers: Int?
    var joinRequests: [UUID]
    var bannedUsers: [UUID]
    
    init(id: UUID = UUID(), name: String, description: String, category: GroupCategory, type: GroupType, privacy: GroupPrivacy, createdBy: UUID) {
        self.id = id
        self.name = name
        self.description = description
        self.category = category
        self.type = type
        self.privacy = privacy
        self.imageURL = nil
        self.createdBy = createdBy
        self.moderators = [createdBy]
        self.members = [createdBy]
        self.memberCount = 1
        self.rules = []
        self.tags = []
        self.location = nil
        self.meetingSchedule = nil
        self.createdDate = Date()
        self.lastActivity = Date()
        self.isActive = true
        self.maxMembers = nil
        self.joinRequests = []
        self.bannedUsers = []
    }
}

enum GroupCategory: String, CaseIterable, Codable {
    case condition = "condition"
    case support = "support"
    case exercise = "exercise"
    case nutrition = "nutrition"
    case medication = "medication"
    case lifestyle = "lifestyle"
    case local = "local"
    case research = "research"
    
    var displayName: String {
        switch self {
        case .condition: return "Condition-Specific"
        case .support: return "Support Groups"
        case .exercise: return "Exercise & Fitness"
        case .nutrition: return "Nutrition"
        case .medication: return "Medication"
        case .lifestyle: return "Lifestyle"
        case .local: return "Local Groups"
        case .research: return "Research"
        }
    }
}

enum GroupType: String, CaseIterable, Codable {
    case discussion = "discussion"
    case support = "support"
    case meetup = "meetup"
    case educational = "educational"
    case research = "research"
    
    var displayName: String {
        switch self {
        case .discussion: return "Discussion"
        case .support: return "Support"
        case .meetup: return "Meetup"
        case .educational: return "Educational"
        case .research: return "Research"
        }
    }
}

enum GroupPrivacy: String, CaseIterable, Codable {
    case `public` = "public"
    case `private` = "private"
    case secret = "secret"
    
    var displayName: String {
        switch self {
        case .public: return "Public"
        case .private: return "Private"
        case .secret: return "Secret"
        }
    }
}

struct GroupRule: Identifiable, Codable {
    let id: UUID
    let title: String
    let description: String
    let priority: RulePriority
    let createdDate: Date
    
    init(id: UUID = UUID(), title: String, description: String, priority: RulePriority = .medium) {
        self.id = id
        self.title = title
        self.description = description
        self.priority = priority
        self.createdDate = Date()
    }
}

enum RulePriority: String, CaseIterable, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
    
    var displayName: String {
        switch self {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .critical: return "Critical"
        }
    }
}

struct MeetingSchedule: Codable {
    var frequency: MeetingFrequency
    var dayOfWeek: DayOfWeek?
    var time: String
    var timezone: String
    var duration: Int // minutes
    var location: MeetingLocation
    var nextMeeting: Date?
    
    init(frequency: MeetingFrequency, dayOfWeek: DayOfWeek? = nil, time: String, timezone: String = TimeZone.current.identifier, duration: Int = 60, location: MeetingLocation) {
        self.frequency = frequency
        self.dayOfWeek = dayOfWeek
        self.time = time
        self.timezone = timezone
        self.duration = duration
        self.location = location
        self.nextMeeting = nil
    }
}

enum MeetingFrequency: String, CaseIterable, Codable {
    case weekly = "weekly"
    case biweekly = "biweekly"
    case monthly = "monthly"
    case irregular = "irregular"
    
    var displayName: String {
        switch self {
        case .weekly: return "Weekly"
        case .biweekly: return "Bi-weekly"
        case .monthly: return "Monthly"
        case .irregular: return "Irregular"
        }
    }
}

enum MeetingLocation: Codable {
    case online(url: String)
    case inPerson(address: String)
    case hybrid(onlineURL: String, address: String)
    
    var displayName: String {
        switch self {
        case .online: return "Online"
        case .inPerson: return "In-Person"
        case .hybrid: return "Hybrid"
        }
    }
}

// MARK: - Post and Discussion Models

struct CommunityPost: Identifiable, Codable {
    let id: UUID
    var title: String?
    var content: String
    var authorId: UUID
    var groupId: UUID?
    var category: PostCategory
    var type: PostType
    var tags: [String]
    var attachments: [PostAttachment]
    var likes: [UUID]
    var comments: [PostComment]
    var shares: [UUID]
    var reports: [PostReport]
    var isAnonymous: Bool
    var isPinned: Bool
    var isLocked: Bool
    var isDeleted: Bool
    var createdDate: Date
    var lastModified: Date
    var visibility: PostVisibility
    var triggerWarnings: [TriggerWarning]
    
    init(id: UUID = UUID(), title: String? = nil, content: String, authorId: UUID, groupId: UUID? = nil, category: PostCategory, type: PostType = .text, isAnonymous: Bool = false) {
        self.id = id
        self.title = title
        self.content = content
        self.authorId = authorId
        self.groupId = groupId
        self.category = category
        self.type = type
        self.tags = []
        self.attachments = []
        self.likes = []
        self.comments = []
        self.shares = []
        self.reports = []
        self.isAnonymous = isAnonymous
        self.isPinned = false
        self.isLocked = false
        self.isDeleted = false
        self.createdDate = Date()
        self.lastModified = Date()
        self.visibility = .public
        self.triggerWarnings = []
    }
}

enum PostCategory: String, CaseIterable, Codable {
    case question = "question"
    case experience = "experience"
    case advice = "advice"
    case support = "support"
    case celebration = "celebration"
    case vent = "vent"
    case resource = "resource"
    case research = "research"
    
    var displayName: String {
        switch self {
        case .question: return "Question"
        case .experience: return "Experience"
        case .advice: return "Advice"
        case .support: return "Support"
        case .celebration: return "Celebration"
        case .vent: return "Vent"
        case .resource: return "Resource"
        case .research: return "Research"
        }
    }
    
    var icon: String {
        switch self {
        case .question: return "questionmark.circle"
        case .experience: return "person.circle"
        case .advice: return "lightbulb"
        case .support: return "heart"
        case .celebration: return "party.popper"
        case .vent: return "cloud.rain"
        case .resource: return "book"
        case .research: return "magnifyingglass"
        }
    }
}

enum PostType: String, CaseIterable, Codable {
    case text = "text"
    case image = "image"
    case video = "video"
    case poll = "poll"
    case link = "link"
    case document = "document"
    
    var displayName: String {
        switch self {
        case .text: return "Text"
        case .image: return "Image"
        case .video: return "Video"
        case .poll: return "Poll"
        case .link: return "Link"
        case .document: return "Document"
        }
    }
}

enum PostVisibility: String, CaseIterable, Codable {
    case `public` = "public"
    case friends = "friends"
    case group = "group"
    case `private` = "private"
    
    var displayName: String {
        switch self {
        case .public: return "Public"
        case .friends: return "Friends"
        case .group: return "Group Only"
        case .private: return "Private"
        }
    }
}

enum TriggerWarning: String, CaseIterable, Codable {
    case depression = "depression"
    case anxiety = "anxiety"
    case selfHarm = "self_harm"
    case suicidalThoughts = "suicidal_thoughts"
    case medicalProcedures = "medical_procedures"
    case painDescription = "pain_description"
    case medicationSideEffects = "medication_side_effects"
    
    var displayName: String {
        switch self {
        case .depression: return "Depression"
        case .anxiety: return "Anxiety"
        case .selfHarm: return "Self-Harm"
        case .suicidalThoughts: return "Suicidal Thoughts"
        case .medicalProcedures: return "Medical Procedures"
        case .painDescription: return "Pain Description"
        case .medicationSideEffects: return "Medication Side Effects"
        }
    }
}

struct PostAttachment: Identifiable, Codable {
    let id: UUID
    let type: AttachmentType
    let url: String
    let filename: String?
    let fileSize: Int?
    let mimeType: String?
    let thumbnailURL: String?
    let uploadDate: Date
    
    init(id: UUID = UUID(), type: AttachmentType, url: String, filename: String? = nil, fileSize: Int? = nil, mimeType: String? = nil, thumbnailURL: String? = nil) {
        self.id = id
        self.type = type
        self.url = url
        self.filename = filename
        self.fileSize = fileSize
        self.mimeType = mimeType
        self.thumbnailURL = thumbnailURL
        self.uploadDate = Date()
    }
}

enum AttachmentType: String, CaseIterable, Codable {
    case image = "image"
    case video = "video"
    case audio = "audio"
    case document = "document"
    case link = "link"
    
    var displayName: String {
        switch self {
        case .image: return "Image"
        case .video: return "Video"
        case .audio: return "Audio"
        case .document: return "Document"
        case .link: return "Link"
        }
    }
}

struct PostComment: Identifiable, Codable {
    let id: UUID
    var content: String
    var authorId: UUID
    var postId: UUID
    var parentCommentId: UUID?
    var likes: [UUID]
    var replies: [PostComment]
    var reports: [CommentReport]
    var isAnonymous: Bool
    var isDeleted: Bool
    var createdDate: Date
    var lastModified: Date
    
    init(id: UUID = UUID(), content: String, authorId: UUID, postId: UUID, parentCommentId: UUID? = nil, isAnonymous: Bool = false) {
        self.id = id
        self.content = content
        self.authorId = authorId
        self.postId = postId
        self.parentCommentId = parentCommentId
        self.likes = []
        self.replies = []
        self.reports = []
        self.isAnonymous = isAnonymous
        self.isDeleted = false
        self.createdDate = Date()
        self.lastModified = Date()
    }
}

struct PostReport: Identifiable, Codable {
    let id: UUID
    let reporterId: UUID
    let postId: UUID
    let reason: ReportReason
    let description: String?
    let status: ReportStatus
    let createdDate: Date
    let reviewedDate: Date?
    let reviewedBy: UUID?
    
    init(id: UUID = UUID(), reporterId: UUID, postId: UUID, reason: ReportReason, description: String? = nil) {
        self.id = id
        self.reporterId = reporterId
        self.postId = postId
        self.reason = reason
        self.description = description
        self.status = .pending
        self.createdDate = Date()
        self.reviewedDate = nil
        self.reviewedBy = nil
    }
}

struct CommentReport: Identifiable, Codable {
    let id: UUID
    let reporterId: UUID
    let commentId: UUID
    let reason: ReportReason
    let description: String?
    let status: ReportStatus
    let createdDate: Date
    let reviewedDate: Date?
    let reviewedBy: UUID?
    
    init(id: UUID = UUID(), reporterId: UUID, commentId: UUID, reason: ReportReason, description: String? = nil) {
        self.id = id
        self.reporterId = reporterId
        self.commentId = commentId
        self.reason = reason
        self.description = description
        self.status = .pending
        self.createdDate = Date()
        self.reviewedDate = nil
        self.reviewedBy = nil
    }
}

enum ReportReason: String, CaseIterable, Codable {
    case spam = "spam"
    case harassment = "harassment"
    case inappropriateContent = "inappropriate_content"
    case misinformation = "misinformation"
    case violatesRules = "violates_rules"
    case other = "other"
    
    var displayName: String {
        switch self {
        case .spam: return "Spam"
        case .harassment: return "Harassment"
        case .inappropriateContent: return "Inappropriate Content"
        case .misinformation: return "Misinformation"
        case .violatesRules: return "Violates Community Rules"
        case .other: return "Other"
        }
    }
}

enum ReportStatus: String, CaseIterable, Codable {
    case pending = "pending"
    case reviewed = "reviewed"
    case actionTaken = "action_taken"
    case dismissed = "dismissed"
    
    var displayName: String {
        switch self {
        case .pending: return "Pending"
        case .reviewed: return "Reviewed"
        case .actionTaken: return "Action Taken"
        case .dismissed: return "Dismissed"
        }
    }
}

// MARK: - Messaging Models

struct DirectMessage: Identifiable, Codable {
    let id: UUID
    var content: String
    var senderId: UUID
    var recipientId: UUID
    var conversationId: UUID
    var type: MessageType
    var attachments: [MessageAttachment]
    var isRead: Bool
    var isDelivered: Bool
    var isDeleted: Bool
    var replyToMessageId: UUID?
    var createdDate: Date
    var readDate: Date?
    var deliveredDate: Date?
    
    init(id: UUID = UUID(), content: String, senderId: UUID, recipientId: UUID, conversationId: UUID, type: MessageType = .text) {
        self.id = id
        self.content = content
        self.senderId = senderId
        self.recipientId = recipientId
        self.conversationId = conversationId
        self.type = type
        self.attachments = []
        self.isRead = false
        self.isDelivered = false
        self.isDeleted = false
        self.replyToMessageId = nil
        self.createdDate = Date()
        self.readDate = nil
        self.deliveredDate = nil
    }
}

enum MessageType: String, CaseIterable, Codable {
    case text = "text"
    case image = "image"
    case video = "video"
    case audio = "audio"
    case document = "document"
    case location = "location"
    case contact = "contact"
    
    var displayName: String {
        switch self {
        case .text: return "Text"
        case .image: return "Image"
        case .video: return "Video"
        case .audio: return "Audio"
        case .document: return "Document"
        case .location: return "Location"
        case .contact: return "Contact"
        }
    }
}

struct MessageAttachment: Identifiable, Codable {
    let id: UUID
    let type: AttachmentType
    let url: String
    let filename: String?
    let fileSize: Int?
    let mimeType: String?
    let thumbnailURL: String?
    let duration: Double? // for audio/video
    let uploadDate: Date
    
    init(id: UUID = UUID(), type: AttachmentType, url: String, filename: String? = nil, fileSize: Int? = nil, mimeType: String? = nil, thumbnailURL: String? = nil, duration: Double? = nil) {
        self.id = id
        self.type = type
        self.url = url
        self.filename = filename
        self.fileSize = fileSize
        self.mimeType = mimeType
        self.thumbnailURL = thumbnailURL
        self.duration = duration
        self.uploadDate = Date()
    }
}

struct Conversation: Identifiable, Codable {
    let id: UUID
    var participants: [UUID]
    var lastMessage: DirectMessage?
    var lastActivity: Date
    var isArchived: Bool
    var isMuted: Bool
    var createdDate: Date
    var unreadCount: [UUID: Int] // userId: unreadCount
    
    init(id: UUID = UUID(), participants: [UUID]) {
        self.id = id
        self.participants = participants
        self.lastMessage = nil
        self.lastActivity = Date()
        self.isArchived = false
        self.isMuted = false
        self.createdDate = Date()
        self.unreadCount = [:]
    }
}

// MARK: - Social Support Manager

@MainActor
class SocialSupportManager: ObservableObject {
    // Published properties
    @Published var currentUser: UserProfile?
    @Published var userProfiles: [UUID: UserProfile] = [:]
    @Published var communityGroups: [CommunityGroup] = []
    @Published var communityPosts: [CommunityPost] = []
    @Published var conversations: [Conversation] = []
    @Published var directMessages: [UUID: [DirectMessage]] = [:] // conversationId: messages
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var connectionStatus: ConnectionStatus = .disconnected
    @Published var unreadMessageCount = 0
    @Published var notifications: [SocialNotification] = []
    
    // Private properties
    private let networkMonitor = NWPathMonitor()
    private let networkQueue = DispatchQueue(label: "NetworkMonitor")
    private var cancellables = Set<AnyCancellable>()
    private let dataManager = SocialDataManager()
    private let encryptionManager = SocialEncryptionManager()
    private let moderationManager = ModerationManager()
    private let notificationManager = SocialNotificationManager()
    
    // Timers
    private var heartbeatTimer: Timer?
    private var syncTimer: Timer?
    
    init() {
        setupNetworkMonitoring()
        loadUserData()
        setupTimers()
    }
    
    deinit {
        heartbeatTimer?.invalidate()
        syncTimer?.invalidate()
        networkMonitor.cancel()
    }
    
    // MARK: - Setup Methods
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.connectionStatus = path.status == .satisfied ? .connected : .disconnected
                if path.status == .satisfied {
                    self?.syncData()
                }
            }
        }
        networkMonitor.start(queue: networkQueue)
    }
    
    private func loadUserData() {
        currentUser = dataManager.loadCurrentUser()
        userProfiles = dataManager.loadUserProfiles()
        communityGroups = dataManager.loadCommunityGroups()
        communityPosts = dataManager.loadCommunityPosts()
        conversations = dataManager.loadConversations()
        directMessages = dataManager.loadDirectMessages()
        notifications = dataManager.loadNotifications()
        updateUnreadMessageCount()
    }
    
    private func setupTimers() {
        // Heartbeat timer for real-time updates
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            self?.sendHeartbeat()
        }
        
        // Sync timer for periodic data synchronization
        syncTimer = Timer.scheduledTimer(withTimeInterval: 300.0, repeats: true) { [weak self] _ in
            self?.syncData()
        }
    }
    
    // MARK: - User Profile Management
    
    func createUserProfile(username: String, displayName: String) async throws {
        guard currentUser == nil else {
            throw SocialError.userAlreadyExists
        }
        
        isLoading = true
        defer { isLoading = false }
        
        let profile = UserProfile(username: username, displayName: displayName)
        
        // Validate username availability
        if await isUsernameAvailable(username) {
            currentUser = profile
            userProfiles[profile.id] = profile
            dataManager.saveCurrentUser(profile)
            dataManager.saveUserProfiles(userProfiles)
        } else {
            throw SocialError.usernameNotAvailable
        }
    }
    
    func updateUserProfile(_ profile: UserProfile) async throws {
        guard let currentUser = currentUser, currentUser.id == profile.id else {
            throw SocialError.unauthorized
        }
        
        isLoading = true
        defer { isLoading = false }
        
        self.currentUser = profile
        userProfiles[profile.id] = profile
        dataManager.saveCurrentUser(profile)
        dataManager.saveUserProfiles(userProfiles)
    }
    
    func searchUsers(query: String, filters: UserSearchFilters? = nil) async -> [UserProfile] {
        return userProfiles.values.filter { profile in
            let matchesQuery = profile.username.localizedCaseInsensitiveContains(query) ||
                             profile.displayName.localizedCaseInsensitiveContains(query)
            
            if let filters = filters {
                return matchesQuery && matchesFilters(profile, filters: filters)
            }
            
            return matchesQuery
        }
    }
    
    private func matchesFilters(_ profile: UserProfile, filters: UserSearchFilters) -> Bool {
        if let ageRange = filters.ageRange, profile.ageRange != ageRange {
            return false
        }
        
        if let location = filters.location, profile.location?.localizedCaseInsensitiveContains(location) != true {
            return false
        }
        
        if !filters.conditions.isEmpty {
            let profileConditions = Set(profile.conditions.map { $0.category })
            let filterConditions = Set(filters.conditions)
            if profileConditions.intersection(filterConditions).isEmpty {
                return false
            }
        }
        
        return true
    }
    
    private func isUsernameAvailable(_ username: String) async -> Bool {
        // In a real implementation, this would check with the server
        return !userProfiles.values.contains { $0.username.lowercased() == username.lowercased() }
    }
    
    // MARK: - Community Group Management
    
    func createCommunityGroup(_ group: CommunityGroup) async throws {
        guard currentUser != nil else {
            throw SocialError.notAuthenticated
        }
        
        isLoading = true
        defer { isLoading = false }
        
        communityGroups.append(group)
        dataManager.saveCommunityGroups(communityGroups)
    }
    
    func joinGroup(_ groupId: UUID) async throws {
        guard let currentUser = currentUser else {
            throw SocialError.notAuthenticated
        }
        
        guard let groupIndex = communityGroups.firstIndex(where: { $0.id == groupId }) else {
            throw SocialError.groupNotFound
        }
        
        var group = communityGroups[groupIndex]
        
        if group.privacy == .private {
            // Add to join requests
            if !group.joinRequests.contains(currentUser.id) {
                group.joinRequests.append(currentUser.id)
            }
        } else {
            // Add directly to members
            if !group.members.contains(currentUser.id) {
                group.members.append(currentUser.id)
                group.memberCount += 1
            }
        }
        
        communityGroups[groupIndex] = group
        dataManager.saveCommunityGroups(communityGroups)
    }
    
    func leaveGroup(_ groupId: UUID) async throws {
        guard let currentUser = currentUser else {
            throw SocialError.notAuthenticated
        }
        
        guard let groupIndex = communityGroups.firstIndex(where: { $0.id == groupId }) else {
            throw SocialError.groupNotFound
        }
        
        var group = communityGroups[groupIndex]
        
        if let memberIndex = group.members.firstIndex(of: currentUser.id) {
            group.members.remove(at: memberIndex)
            group.memberCount -= 1
        }
        
        communityGroups[groupIndex] = group
        dataManager.saveCommunityGroups(communityGroups)
    }
    
    func searchGroups(query: String, filters: GroupSearchFilters? = nil) -> [CommunityGroup] {
        return communityGroups.filter { group in
            let matchesQuery = group.name.localizedCaseInsensitiveContains(query) ||
                             group.description.localizedCaseInsensitiveContains(query)
            
            if let filters = filters {
                return matchesQuery && matchesGroupFilters(group, filters: filters)
            }
            
            return matchesQuery
        }
    }
    
    private func matchesGroupFilters(_ group: CommunityGroup, filters: GroupSearchFilters) -> Bool {
        if let category = filters.category, group.category != category {
            return false
        }
        
        if let type = filters.type, group.type != type {
            return false
        }
        
        if let privacy = filters.privacy, group.privacy != privacy {
            return false
        }
        
        return true
    }
    
    // MARK: - Post Management
    
    func createPost(_ post: CommunityPost) async throws {
        guard currentUser != nil else {
            throw SocialError.notAuthenticated
        }
        
        isLoading = true
        defer { isLoading = false }
        
        // Content moderation
        let moderatedPost = try await moderationManager.moderatePost(post)
        
        communityPosts.insert(moderatedPost, at: 0)
        dataManager.saveCommunityPosts(communityPosts)
    }
    
    func likePost(_ postId: UUID) async throws {
        guard let currentUser = currentUser else {
            throw SocialError.notAuthenticated
        }
        
        guard let postIndex = communityPosts.firstIndex(where: { $0.id == postId }) else {
            throw SocialError.postNotFound
        }
        
        var post = communityPosts[postIndex]
        
        if post.likes.contains(currentUser.id) {
            post.likes.removeAll { $0 == currentUser.id }
        } else {
            post.likes.append(currentUser.id)
        }
        
        communityPosts[postIndex] = post
        dataManager.saveCommunityPosts(communityPosts)
    }
    
    func addComment(to postId: UUID, content: String, isAnonymous: Bool = false) async throws {
        guard let currentUser = currentUser else {
            throw SocialError.notAuthenticated
        }
        
        guard let postIndex = communityPosts.firstIndex(where: { $0.id == postId }) else {
            throw SocialError.postNotFound
        }
        
        let comment = PostComment(
            content: content,
            authorId: currentUser.id,
            postId: postId,
            isAnonymous: isAnonymous
        )
        
        // Content moderation
        let moderatedComment = try await moderationManager.moderateComment(comment)
        
        var post = communityPosts[postIndex]
        post.comments.append(moderatedComment)
        
        communityPosts[postIndex] = post
        dataManager.saveCommunityPosts(communityPosts)
    }
    
    func reportPost(_ postId: UUID, reason: ReportReason, description: String? = nil) async throws {
        guard let currentUser = currentUser else {
            throw SocialError.notAuthenticated
        }
        
        guard let postIndex = communityPosts.firstIndex(where: { $0.id == postId }) else {
            throw SocialError.postNotFound
        }
        
        let report = PostReport(
            reporterId: currentUser.id,
            postId: postId,
            reason: reason,
            description: description
        )
        
        var post = communityPosts[postIndex]
        post.reports.append(report)
        
        communityPosts[postIndex] = post
        dataManager.saveCommunityPosts(communityPosts)
        
        // Notify moderation team
        await moderationManager.handleReport(report)
    }
    
    // MARK: - Messaging
    
    func sendDirectMessage(to recipientId: UUID, content: String, type: MessageType = .text) async throws {
        guard let currentUser = currentUser else {
            throw SocialError.notAuthenticated
        }
        
        // Find or create conversation
        let conversationId = findOrCreateConversation(with: recipientId)
        
        let message = DirectMessage(
            content: content,
            senderId: currentUser.id,
            recipientId: recipientId,
            conversationId: conversationId,
            type: type
        )
        
        // Encrypt message content
        let encryptedMessage = try encryptionManager.encryptMessage(message)
        
        // Add to messages
        if directMessages[conversationId] == nil {
            directMessages[conversationId] = []
        }
        directMessages[conversationId]?.append(encryptedMessage)
        
        // Update conversation
        if let conversationIndex = conversations.firstIndex(where: { $0.id == conversationId }) {
            var conversation = conversations[conversationIndex]
            conversation.lastMessage = encryptedMessage
            conversation.lastActivity = Date()
            conversations[conversationIndex] = conversation
        }
        
        dataManager.saveDirectMessages(directMessages)
        dataManager.saveConversations(conversations)
        updateUnreadMessageCount()
    }
    
    private func findOrCreateConversation(with userId: UUID) -> UUID {
        // Find existing conversation
        if let conversation = conversations.first(where: { $0.participants.contains(userId) && $0.participants.count == 2 }) {
            return conversation.id
        }
        
        // Create new conversation
        let conversation = Conversation(participants: [currentUser!.id, userId])
        conversations.append(conversation)
        return conversation.id
    }
    
    func markMessageAsRead(_ messageId: UUID, in conversationId: UUID) async {
        guard let messages = directMessages[conversationId],
              let messageIndex = messages.firstIndex(where: { $0.id == messageId }) else {
            return
        }
        
        var message = messages[messageIndex]
        message.isRead = true
        message.readDate = Date()
        
        directMessages[conversationId]?[messageIndex] = message
        dataManager.saveDirectMessages(directMessages)
        updateUnreadMessageCount()
    }
    
    private func updateUnreadMessageCount() {
        guard let currentUser = currentUser else { return }
        
        var count = 0
        for (_, messages) in directMessages {
            count += messages.filter { !$0.isRead && $0.recipientId == currentUser.id }.count
        }
        unreadMessageCount = count
    }
    
    // MARK: - Network and Sync
    
    private func sendHeartbeat() {
        guard connectionStatus == .connected else { return }
        // Implementation would send heartbeat to server
    }
    
    private func syncData() {
        guard connectionStatus == .connected else { return }
        // Implementation would sync data with server
    }
    
    // MARK: - Utility Methods
    
    func getUserProfile(for userId: UUID) -> UserProfile? {
        return userProfiles[userId]
    }
    
    func getGroupsForUser(_ userId: UUID) -> [CommunityGroup] {
        return communityGroups.filter { $0.members.contains(userId) }
    }
    
    func getPostsForGroup(_ groupId: UUID) -> [CommunityPost] {
        return communityPosts.filter { $0.groupId == groupId }
    }
    
    func getConversationMessages(_ conversationId: UUID) -> [DirectMessage] {
        return directMessages[conversationId] ?? []
    }
}

// MARK: - Supporting Models

struct UserSearchFilters {
    var ageRange: AgeRange?
    var location: String?
    var conditions: [ConditionCategory]
    var interests: [InterestCategory]
    var verificationStatus: VerificationStatus?
    
    init(ageRange: AgeRange? = nil, location: String? = nil, conditions: [ConditionCategory] = [], interests: [InterestCategory] = [], verificationStatus: VerificationStatus? = nil) {
        self.ageRange = ageRange
        self.location = location
        self.conditions = conditions
        self.interests = interests
        self.verificationStatus = verificationStatus
    }
}

struct GroupSearchFilters {
    var category: GroupCategory?
    var type: GroupType?
    var privacy: GroupPrivacy?
    var location: String?
    var hasSchedule: Bool?
    
    init(category: GroupCategory? = nil, type: GroupType? = nil, privacy: GroupPrivacy? = nil, location: String? = nil, hasSchedule: Bool? = nil) {
        self.category = category
        self.type = type
        self.privacy = privacy
        self.location = location
        self.hasSchedule = hasSchedule
    }
}

enum ConnectionStatus {
    case connected
    case connecting
    case disconnected
    case error(String)
}

struct SocialNotification: Identifiable, Codable {
    let id: UUID
    let type: SocialNotificationType
    let title: String
    let message: String
    let userId: UUID
    let relatedId: UUID? // postId, groupId, etc.
    let isRead: Bool
    let createdDate: Date
    
    init(id: UUID = UUID(), type: SocialNotificationType, title: String, message: String, userId: UUID, relatedId: UUID? = nil, isRead: Bool = false) {
        self.id = id
        self.type = type
        self.title = title
        self.message = message
        self.userId = userId
        self.relatedId = relatedId
        self.isRead = isRead
        self.createdDate = Date()
    }
}

enum SocialNotificationType: String, CaseIterable, Codable {
    case newMessage = "new_message"
    case newComment = "new_comment"
    case newLike = "new_like"
    case groupInvite = "group_invite"
    case friendRequest = "friend_request"
    case mentionInPost = "mention_in_post"
    case groupUpdate = "group_update"
    case systemMessage = "system_message"
    
    var displayName: String {
        switch self {
        case .newMessage: return "New Message"
        case .newComment: return "New Comment"
        case .newLike: return "New Like"
        case .groupInvite: return "Group Invite"
        case .friendRequest: return "Friend Request"
        case .mentionInPost: return "Mentioned in Post"
        case .groupUpdate: return "Group Update"
        case .systemMessage: return "System Message"
        }
    }
}

// MARK: - Error Types

enum SocialError: Error, LocalizedError {
    case notAuthenticated
    case unauthorized
    case userAlreadyExists
    case usernameNotAvailable
    case userNotFound
    case groupNotFound
    case postNotFound
    case messageNotFound
    case networkError
    case encryptionError
    case moderationError
    case invalidInput
    
    var errorDescription: String? {
        switch self {
        case .notAuthenticated:
            return "User not authenticated"
        case .unauthorized:
            return "Unauthorized access"
        case .userAlreadyExists:
            return "User already exists"
        case .usernameNotAvailable:
            return "Username not available"
        case .userNotFound:
            return "User not found"
        case .groupNotFound:
            return "Group not found"
        case .postNotFound:
            return "Post not found"
        case .messageNotFound:
            return "Message not found"
        case .networkError:
            return "Network error"
        case .encryptionError:
            return "Encryption error"
        case .moderationError:
            return "Content moderation error"
        case .invalidInput:
            return "Invalid input"
        }
    }
}

// MARK: - Supporting Classes

class SocialDataManager {
    private let userDefaults = UserDefaults.standard
    private let fileManager = FileManager.default
    
    func loadCurrentUser() -> UserProfile? {
        // Implementation would load from secure storage
        return nil
    }
    
    func saveCurrentUser(_ user: UserProfile) {
        // Implementation would save to secure storage
    }
    
    func loadUserProfiles() -> [UUID: UserProfile] {
        // Implementation would load from local storage
        return [:]
    }
    
    func saveUserProfiles(_ profiles: [UUID: UserProfile]) {
        // Implementation would save to local storage
    }
    
    func loadCommunityGroups() -> [CommunityGroup] {
        // Implementation would load from local storage
        return []
    }
    
    func saveCommunityGroups(_ groups: [CommunityGroup]) {
        // Implementation would save to local storage
    }
    
    func loadCommunityPosts() -> [CommunityPost] {
        // Implementation would load from local storage
        return []
    }
    
    func saveCommunityPosts(_ posts: [CommunityPost]) {
        // Implementation would save to local storage
    }
    
    func loadConversations() -> [Conversation] {
        // Implementation would load from local storage
        return []
    }
    
    func saveConversations(_ conversations: [Conversation]) {
        // Implementation would save to local storage
    }
    
    func loadDirectMessages() -> [UUID: [DirectMessage]] {
        // Implementation would load from local storage
        return [:]
    }
    
    func saveDirectMessages(_ messages: [UUID: [DirectMessage]]) {
        // Implementation would save to local storage
    }
    
    func loadNotifications() -> [SocialNotification] {
        // Implementation would load from local storage
        return []
    }
    
    func saveNotifications(_ notifications: [SocialNotification]) {
        // Implementation would save to local storage
    }
}

class SocialEncryptionManager {
    private let keychain = Keychain(service: "com.inflamai.social")
    
    func encryptMessage(_ message: DirectMessage) throws -> DirectMessage {
        // Implementation would encrypt message content
        return message
    }
    
    func decryptMessage(_ message: DirectMessage) throws -> DirectMessage {
        // Implementation would decrypt message content
        return message
    }
    
    func generateKeyPair() throws -> (publicKey: Data, privateKey: Data) {
        // Implementation would generate encryption key pair
        return (Data(), Data())
    }
}

class ModerationManager {
    func moderatePost(_ post: CommunityPost) async throws -> CommunityPost {
        // Implementation would check content for inappropriate material
        return post
    }
    
    func moderateComment(_ comment: PostComment) async throws -> PostComment {
        // Implementation would check content for inappropriate material
        return comment
    }
    
    func handleReport(_ report: PostReport) async {
        // Implementation would handle content reports
    }
}

class SocialNotificationManager {
    func sendNotification(_ notification: SocialNotification) {
        // Implementation would send push notification
    }
    
  func scheduleNotification(_ notification: SocialNotification, at date: Date) {
        // Implementation would schedule notification
    }
}

// MARK: - Keychain Helper

struct Keychain {
    let service: String
    
    init(service: String) {
        self.service = service
    }
    
    func set(_ data: Data, for key: String) throws {
        // Implementation would store data in keychain
    }
    
    func get(_ key: String) throws -> Data? {
        // Implementation would retrieve data from keychain
        return nil
    }
    
    func delete(_ key: String) throws {
        // Implementation would delete data from keychain
    }
}

// MARK: - SwiftUI Views

struct SocialSupportView: View {
    @StateObject private var socialManager = SocialSupportManager()
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            CommunityFeedView()
                .tabItem {
                    Image(systemName: "house.fill")
                    Text("Feed")
                }
                .tag(0)
            
            GroupsView()
                .tabItem {
                    Image(systemName: "person.3.fill")
                    Text("Groups")
                }
                .tag(1)
            
            MessagesView()
                .tabItem {
                    Image(systemName: "message.fill")
                    Text("Messages")
                }
                .badge(socialManager.unreadMessageCount > 0 ? socialManager.unreadMessageCount : nil)
                .tag(2)
            
            ProfileView()
                .tabItem {
                    Image(systemName: "person.fill")
                    Text("Profile")
                }
                .tag(3)
        }
        .environmentObject(socialManager)
    }
}

struct CommunityFeedView: View {
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var showingCreatePost = false
    
    var body: some View {
        NavigationView {
            List(socialManager.communityPosts) { post in
                PostRowView(post: post)
            }
            .navigationTitle("Community")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingCreatePost = true }) {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showingCreatePost) {
                CreatePostView()
            }
        }
    }
}

struct PostRowView: View {
    let post: CommunityPost
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var showingComments = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Post header
            HStack {
                AsyncImage(url: URL(string: authorProfile?.avatarURL ?? "")) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } placeholder: {
                    Circle()
                        .fill(Color.gray.opacity(0.3))
                }
                .frame(width: 40, height: 40)
                .clipShape(Circle())
                
                VStack(alignment: .leading) {
                    Text(post.isAnonymous ? "Anonymous" : (authorProfile?.displayName ?? "Unknown"))
                        .font(.headline)
                    Text(post.createdDate, style: .relative)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Image(systemName: post.category.icon)
                    .foregroundColor(.blue)
            }
            
            // Post content
            if let title = post.title {
                Text(title)
                    .font(.title3)
                    .fontWeight(.semibold)
            }
            
            Text(post.content)
                .font(.body)
            
            // Post actions
            HStack {
                Button(action: { likePost() }) {
                    HStack {
                        Image(systemName: isLiked ? "heart.fill" : "heart")
                            .foregroundColor(isLiked ? .red : .primary)
                        Text("\(post.likes.count)")
                    }
                }
                
                Button(action: { showingComments = true }) {
                    HStack {
                        Image(systemName: "bubble.left")
                        Text("\(post.comments.count)")
                    }
                }
                
                Spacer()
                
                Menu {
                    Button("Share") { sharePost() }
                    Button("Report", role: .destructive) { reportPost() }
                } label: {
                    Image(systemName: "ellipsis")
                }
            }
            .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
        .sheet(isPresented: $showingComments) {
            CommentsView(post: post)
        }
    }
    
    private var authorProfile: UserProfile? {
        socialManager.getUserProfile(for: post.authorId)
    }
    
    private var isLiked: Bool {
        guard let currentUser = socialManager.currentUser else { return false }
        return post.likes.contains(currentUser.id)
    }
    
    private func likePost() {
        Task {
            try await socialManager.likePost(post.id)
        }
    }
    
    private func sharePost() {
        // Implementation for sharing post
    }
    
    private func reportPost() {
        // Implementation for reporting post
    }
}

struct CreatePostView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var title = ""
    @State private var content = ""
    @State private var selectedCategory: PostCategory = .question
    @State private var isAnonymous = false
    @State private var selectedGroup: CommunityGroup?
    
    var body: some View {
        NavigationView {
            Form {
                Section("Post Details") {
                    TextField("Title (optional)", text: $title)
                    
                    TextEditor(text: $content)
                        .frame(minHeight: 100)
                    
                    Picker("Category", selection: $selectedCategory) {
                        ForEach(PostCategory.allCases, id: \.self) { category in
                            Label(category.displayName, systemImage: category.icon)
                                .tag(category)
                        }
                    }
                }
                
                Section("Privacy") {
                    Toggle("Post anonymously", isOn: $isAnonymous)
                    
                    Picker("Post to group", selection: $selectedGroup) {
                        Text("General Community").tag(nil as CommunityGroup?)
                        ForEach(userGroups, id: \.id) { group in
                            Text(group.name).tag(group as CommunityGroup?)
                        }
                    }
                }
            }
            .navigationTitle("Create Post")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Post") {
                        createPost()
                    }
                    .disabled(content.isEmpty)
                }
            }
        }
    }
    
    private var userGroups: [CommunityGroup] {
        guard let currentUser = socialManager.currentUser else { return [] }
        return socialManager.getGroupsForUser(currentUser.id)
    }
    
    private func createPost() {
        guard let currentUser = socialManager.currentUser else { return }
        
        let post = CommunityPost(
            title: title.isEmpty ? nil : title,
            content: content,
            authorId: currentUser.id,
            groupId: selectedGroup?.id,
            category: selectedCategory,
            isAnonymous: isAnonymous
        )
        
        Task {
            try await socialManager.createPost(post)
            await MainActor.run {
                dismiss()
            }
        }
    }
}

struct GroupsView: View {
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var searchText = ""
    @State private var showingCreateGroup = false
    
    var body: some View {
        NavigationView {
            List {
                Section("My Groups") {
                    ForEach(myGroups) { group in
                        GroupRowView(group: group)
                    }
                }
                
                Section("Discover Groups") {
                    ForEach(filteredGroups) { group in
                        GroupRowView(group: group)
                    }
                }
            }
            .searchable(text: $searchText)
            .navigationTitle("Groups")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingCreateGroup = true }) {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showingCreateGroup) {
                CreateGroupView()
            }
        }
    }
    
    private var myGroups: [CommunityGroup] {
        guard let currentUser = socialManager.currentUser else { return [] }
        return socialManager.getGroupsForUser(currentUser.id)
    }
    
    private var filteredGroups: [CommunityGroup] {
        let allGroups = socialManager.communityGroups
        let myGroupIds = Set(myGroups.map { $0.id })
        
        return allGroups.filter { group in
            !myGroupIds.contains(group.id) &&
            (searchText.isEmpty || group.name.localizedCaseInsensitiveContains(searchText))
        }
    }
}

struct GroupRowView: View {
    let group: CommunityGroup
    @EnvironmentObject var socialManager: SocialSupportManager
    
    var body: some View {
        HStack {
            AsyncImage(url: URL(string: group.imageURL ?? "")) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.3))
            }
            .frame(width: 50, height: 50)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            
            VStack(alignment: .leading) {
                Text(group.name)
                    .font(.headline)
                Text(group.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                HStack {
                    Text("\(group.memberCount) members")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text(group.category.displayName)
                        .font(.caption2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color.blue.opacity(0.2))
                        .cornerRadius(4)
                }
            }
            
            Spacer()
            
            if isMember {
                Text("Joined")
                    .font(.caption)
                    .foregroundColor(.green)
            } else {
                Button("Join") {
                    joinGroup()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(.vertical, 4)
    }
    
    private var isMember: Bool {
        guard let currentUser = socialManager.currentUser else { return false }
        return group.members.contains(currentUser.id)
    }
    
    private func joinGroup() {
        Task {
            try await socialManager.joinGroup(group.id)
        }
    }
}

struct MessagesView: View {
    @EnvironmentObject var socialManager: SocialSupportManager
    
    var body: some View {
        NavigationView {
            List(socialManager.conversations) { conversation in
                ConversationRowView(conversation: conversation)
            }
            .navigationTitle("Messages")
        }
    }
}

struct ConversationRowView: View {
    let conversation: Conversation
    @EnvironmentObject var socialManager: SocialSupportManager
    
    var body: some View {
        NavigationLink(destination: ChatView(conversation: conversation)) {
            HStack {
                AsyncImage(url: URL(string: otherUser?.avatarURL ?? "")) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } placeholder: {
                    Circle()
                        .fill(Color.gray.opacity(0.3))
                }
                .frame(width: 40, height: 40)
                .clipShape(Circle())
                
                VStack(alignment: .leading) {
                    Text(otherUser?.displayName ?? "Unknown")
                        .font(.headline)
                    
                    if let lastMessage = conversation.lastMessage {
                        Text(lastMessage.content)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                }
                
                Spacer()
                
                VStack {
                    Text(conversation.lastActivity, style: .time)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    if unreadCount > 0 {
                        Text("\(unreadCount)")
                            .font(.caption2)
                            .foregroundColor(.white)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.red)
                            .clipShape(Capsule())
                    }
                }
            }
        }
    }
    
    private var otherUser: UserProfile? {
        guard let currentUser = socialManager.currentUser else { return nil }
        let otherUserId = conversation.participants.first { $0 != currentUser.id }
        return otherUserId.flatMap { socialManager.getUserProfile(for: $0) }
    }
    
    private var unreadCount: Int {
        guard let currentUser = socialManager.currentUser else { return 0 }
        return conversation.unreadCount[currentUser.id] ?? 0
    }
}

struct ChatView: View {
    let conversation: Conversation
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var messageText = ""
    
    var body: some View {
        VStack {
            ScrollView {
                LazyVStack {
                    ForEach(messages) { message in
                        MessageBubbleView(message: message)
                    }
                }
            }
            
            HStack {
                TextField("Type a message...", text: $messageText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Button("Send") {
                    sendMessage()
                }
                .disabled(messageText.isEmpty)
            }
            .padding()
        }
        .navigationTitle(otherUser?.displayName ?? "Chat")
        .navigationBarTitleDisplayMode(.inline)
    }
    
    private var messages: [DirectMessage] {
        socialManager.getConversationMessages(conversation.id)
    }
    
    private var otherUser: UserProfile? {
        guard let currentUser = socialManager.currentUser else { return nil }
        let otherUserId = conversation.participants.first { $0 != currentUser.id }
        return otherUserId.flatMap { socialManager.getUserProfile(for: $0) }
    }
    
    private func sendMessage() {
        guard let otherUser = otherUser else { return }
        
        Task {
            try await socialManager.sendDirectMessage(to: otherUser.id, content: messageText)
            await MainActor.run {
                messageText = ""
            }
        }
    }
}

struct MessageBubbleView: View {
    let message: DirectMessage
    @EnvironmentObject var socialManager: SocialSupportManager
    
    var body: some View {
        HStack {
            if isFromCurrentUser {
                Spacer()
            }
            
            Text(message.content)
                .padding()
                .background(isFromCurrentUser ? Color.blue : Color.gray.opacity(0.2))
                .foregroundColor(isFromCurrentUser ? .white : .primary)
                .cornerRadius(12)
            
            if !isFromCurrentUser {
                Spacer()
            }
        }
        .padding(.horizontal)
    }
    
    private var isFromCurrentUser: Bool {
        guard let currentUser = socialManager.currentUser else { return false }
        return message.senderId == currentUser.id
    }
}

struct ProfileView: View {
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var showingEditProfile = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Profile header
                    VStack {
                        AsyncImage(url: URL(string: currentUser?.avatarURL ?? "")) { image in
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        } placeholder: {
                            Circle()
                                .fill(Color.gray.opacity(0.3))
                        }
                        .frame(width: 100, height: 100)
                        .clipShape(Circle())
                        
                        Text(currentUser?.displayName ?? "Unknown")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        if let bio = currentUser?.bio {
                            Text(bio)
                                .font(.body)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                    }
                    
                    // Stats
                    HStack(spacing: 30) {
                        VStack {
                            Text("\(currentUser?.totalPosts ?? 0)")
                                .font(.title2)
                                .fontWeight(.bold)
                            Text("Posts")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        VStack {
                            Text("\(currentUser?.helpfulnessScore ?? 0)")
                                .font(.title2)
                                .fontWeight(.bold)
                            Text("Helpful")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        VStack {
                            Text(String(format: "%.1f", currentUser?.supporterRating ?? 0.0))
                                .font(.title2)
                                .fontWeight(.bold)
                            Text("Rating")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // Badges
                    if let badges = currentUser?.badges, !badges.isEmpty {
                        VStack(alignment: .leading) {
                            Text("Badges")
                                .font(.headline)
                            
                            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3)) {
                                ForEach(badges) { badge in
                                    VStack {
                                        Image(systemName: badge.iconName)
                                            .font(.title2)
                                            .foregroundColor(.blue)
                                        Text(badge.name)
                                            .font(.caption)
                                            .multilineTextAlignment(.center)
                                    }
                                    .padding()
                                    .background(Color.blue.opacity(0.1))
                                    .cornerRadius(8)
                                }
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Profile")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Edit") {
                        showingEditProfile = true
                    }
                }
            }
            .sheet(isPresented: $showingEditProfile) {
                EditProfileView()
            }
        }
    }
    
    private var currentUser: UserProfile? {
        socialManager.currentUser
    }
}

struct EditProfileView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var displayName = ""
    @State private var bio = ""
    @State private var selectedAgeRange: AgeRange?
    @State private var location = ""
    
    var body: some View {
        NavigationView {
            Form {
                Section("Basic Information") {
                    TextField("Display Name", text: $displayName)
                    TextField("Bio", text: $bio, axis: .vertical)
                        .lineLimit(3...6)
                }
                
                Section("Demographics") {
                    Picker("Age Range", selection: $selectedAgeRange) {
                        Text("Prefer not to say").tag(nil as AgeRange?)
                        ForEach(AgeRange.allCases, id: \.self) { range in
                            Text(range.displayName).tag(range as AgeRange?)
                        }
                    }
                    
                    TextField("Location", text: $location)
                }
            }
            .navigationTitle("Edit Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveProfile()
                    }
                }
            }
        }
        .onAppear {
            loadCurrentProfile()
        }
    }
    
    private func loadCurrentProfile() {
        guard let currentUser = socialManager.currentUser else { return }
        displayName = currentUser.displayName
        bio = currentUser.bio ?? ""
        selectedAgeRange = currentUser.ageRange
        location = currentUser.location ?? ""
    }
    
    private func saveProfile() {
        guard var currentUser = socialManager.currentUser else { return }
        
        currentUser.displayName = displayName
        currentUser.bio = bio.isEmpty ? nil : bio
        currentUser.ageRange = selectedAgeRange
        currentUser.location = location.isEmpty ? nil : location
        
        Task {
            try await socialManager.updateUserProfile(currentUser)
            await MainActor.run {
                dismiss()
            }
        }
    }
}

struct CreateGroupView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var name = ""
    @State private var description = ""
    @State private var selectedCategory: GroupCategory = .support
    @State private var selectedType: GroupType = .discussion
    @State private var selectedPrivacy: GroupPrivacy = .public
    
    var body: some View {
        NavigationView {
            Form {
                Section("Group Details") {
                    TextField("Group Name", text: $name)
                    TextField("Description", text: $description, axis: .vertical)
                        .lineLimit(3...6)
                }
                
                Section("Settings") {
                    Picker("Category", selection: $selectedCategory) {
                        ForEach(GroupCategory.allCases, id: \.self) { category in
                            Text(category.displayName).tag(category)
                        }
                    }
                    
                    Picker("Type", selection: $selectedType) {
                        ForEach(GroupType.allCases, id: \.self) { type in
                            Text(type.displayName).tag(type)
                        }
                    }
                    
                    Picker("Privacy", selection: $selectedPrivacy) {
                        ForEach(GroupPrivacy.allCases, id: \.self) { privacy in
                            Text(privacy.displayName).tag(privacy)
                        }
                    }
                }
            }
            .navigationTitle("Create Group")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Create") {
                        createGroup()
                    }
                    .disabled(name.isEmpty || description.isEmpty)
                }
            }
        }
    }
    
    private func createGroup() {
        guard let currentUser = socialManager.currentUser else { return }
        
        let group = CommunityGroup(
            name: name,
            description: description,
            category: selectedCategory,
            type: selectedType,
            privacy: selectedPrivacy,
            createdBy: currentUser.id
        )
        
        Task {
            try await socialManager.createCommunityGroup(group)
            await MainActor.run {
                dismiss()
            }
        }
    }
}

struct CommentsView: View {
    let post: CommunityPost
    @EnvironmentObject var socialManager: SocialSupportManager
    @State private var newComment = ""
    @State private var isAnonymous = false
    
    var body: some View {
        NavigationView {
            VStack {
                List(post.comments) { comment in
                    CommentRowView(comment: comment)
                }
                
                VStack {
                    Toggle("Comment anonymously", isOn: $isAnonymous)
                        .padding(.horizontal)
                    
                    HStack {
                        TextField("Add a comment...", text: $newComment)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                        
                        Button("Post") {
                            addComment()
                        }
                        .disabled(newComment.isEmpty)
                    }
                    .padding()
                }
            }
            .navigationTitle("Comments")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    private func addComment() {
        Task {
            try await socialManager.addComment(to: post.id, content: newComment, isAnonymous: isAnonymous)
            await MainActor.run {
                newComment = ""
            }
        }
    }
}

struct CommentRowView: View {
    let comment: PostComment
    @EnvironmentObject var socialManager: SocialSupportManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                AsyncImage(url: URL(string: authorProfile?.avatarURL ?? "")) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } placeholder: {
                    Circle()
                        .fill(Color.gray.opacity(0.3))
                }
                .frame(width: 30, height: 30)
                .clipShape(Circle())
                
                VStack(alignment: .leading) {
                    Text(comment.isAnonymous ? "Anonymous" : (authorProfile?.displayName ?? "Unknown"))
                        .font(.caption)
                        .fontWeight(.medium)
                    Text(comment.createdDate, style: .relative)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            Text(comment.content)
                .font(.body)
            
            HStack {
                Button(action: { likeComment() }) {
                    HStack {
                        Image(systemName: isLiked ? "heart.fill" : "heart")
                            .foregroundColor(isLiked ? .red : .secondary)
                        Text("\(comment.likes.count)")
                            .foregroundColor(.secondary)
                    }
                }
                .buttonStyle(PlainButtonStyle())
                
                Spacer()
            }
            .font(.caption)
        }
        .padding(.vertical, 4)
    }
    
    private var authorProfile: UserProfile? {
        socialManager.getUserProfile(for: comment.authorId)
    }
    
    private var isLiked: Bool {
        guard let currentUser = socialManager.currentUser else { return false }
        return comment.likes.contains(currentUser.id)
    }
    
    private func likeComment() {
        // Implementation for liking comment
    }
}