//
//  SocialSupportView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import Combine

struct SocialSupportView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var selectedTab = 0
    @State private var showingCreatePost = false
    @State private var showingCreateGroup = false
    @State private var showingProfile = false
    @State private var showingSettings = false
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // Feed Tab
                FeedView()
                    .tabItem {
                        Image(systemName: "house.fill")
                        Text("Feed")
                    }
                    .tag(0)
                
                // Groups Tab
                GroupsView()
                    .tabItem {
                        Image(systemName: "person.3.fill")
                        Text("Groups")
                    }
                    .tag(1)
                
                // Messages Tab
                MessagesView()
                    .tabItem {
                        Image(systemName: "message.fill")
                        Text("Messages")
                    }
                    .tag(2)
                
                // Discover Tab
                DiscoverView()
                    .tabItem {
                        Image(systemName: "magnifyingglass")
                        Text("Discover")
                    }
                    .tag(3)
                
                // Profile Tab
                ProfileView()
                    .tabItem {
                        Image(systemName: "person.fill")
                        Text("Profile")
                    }
                    .tag(4)
            }
            .navigationTitle(tabTitle)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: { showingSettings = true }) {
                        Image(systemName: "gearshape.fill")
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingCreatePost = true }) {
                        Image(systemName: "plus.circle.fill")
                    }
                }
            }
        }
        .sheet(isPresented: $showingCreatePost) {
            CreatePostView()
        }
        .sheet(isPresented: $showingCreateGroup) {
            CreateGroupView()
        }
        .sheet(isPresented: $showingSettings) {
            SocialSettingsView()
        }
        .task {
            await socialManager.loadAvailableGroups()
        }
    }
    
    private var tabTitle: String {
        switch selectedTab {
        case 0: return "Feed"
        case 1: return "Groups"
        case 2: return "Messages"
        case 3: return "Discover"
        case 4: return "Profile"
        default: return "Social"
        }
    }
}

// MARK: - Feed View

struct FeedView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var selectedFilter: FeedFilter = .all
    @State private var refreshing = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Filter Picker
            FeedFilterPicker(selectedFilter: $selectedFilter)
                .padding(.horizontal)
            
            // Posts List
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(filteredPosts) { post in
                        PostCard(post: post)
                            .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .refreshable {
                await refreshFeed()
            }
        }
        .overlay {
            if socialManager.recentPosts.isEmpty && !socialManager.isLoading {
                EmptyFeedView()
            }
        }
        .task {
            if socialManager.recentPosts.isEmpty {
                await loadFeed()
            }
        }
    }
    
    private var filteredPosts: [SupportPost] {
        switch selectedFilter {
        case .all:
            return socialManager.recentPosts
        case .myGroups:
            let groupIDs = Set(socialManager.joinedGroups.map { $0.id })
            return socialManager.recentPosts.filter { groupIDs.contains($0.groupID) }
        case .questions:
            return socialManager.recentPosts.filter { $0.type == .question }
        case .support:
            return socialManager.recentPosts.filter { $0.type == .support }
        case .celebrations:
            return socialManager.recentPosts.filter { $0.type == .celebration }
        }
    }
    
    private func loadFeed() async {
        // Load recent posts from all joined groups
        for group in socialManager.joinedGroups {
            let posts = await socialManager.loadGroupPosts(groupID: group.id)
            // Posts would be automatically updated via the manager
        }
    }
    
    private func refreshFeed() async {
        refreshing = true
        await loadFeed()
        refreshing = false
    }
}

enum FeedFilter: String, CaseIterable {
    case all = "All"
    case myGroups = "My Groups"
    case questions = "Questions"
    case support = "Support"
    case celebrations = "Celebrations"
}

struct FeedFilterPicker: View {
    @Binding var selectedFilter: FeedFilter
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(FeedFilter.allCases, id: \.self) { filter in
                    FilterChip(
                        title: filter.rawValue,
                        isSelected: selectedFilter == filter
                    ) {
                        selectedFilter = filter
                    }
                }
            }
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
    }
}

struct FilterChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline)
                .fontWeight(isSelected ? .semibold : .medium)
                .foregroundColor(isSelected ? .white : .primary)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 20)
                        .fill(isSelected ? Color.blue : Color(.systemGray6))
                )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct PostCard: View {
    let post: SupportPost
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var showingComments = false
    @State private var showingReactions = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            PostHeader(post: post)
            
            // Content
            PostContent(post: post)
            
            // Attachments
            if !post.attachments.isEmpty {
                PostAttachmentsView(attachments: post.attachments)
            }
            
            // Mood and Pain Level
            if post.mood != nil || post.painLevel != nil {
                PostMetricsView(mood: post.mood, painLevel: post.painLevel)
            }
            
            // Actions
            PostActions(
                post: post,
                onReaction: { reaction in
                    Task {
                        try? await socialManager.addReaction(to: post, reaction: reaction)
                    }
                },
                onComment: {
                    showingComments = true
                },
                onShare: {
                    // Handle share
                }
            )
            
            // Reaction Summary
            if !post.reactions.isEmpty {
                ReactionSummaryView(reactions: post.reactions)
            }
        }
        .padding(16)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        .sheet(isPresented: $showingComments) {
            PostCommentsView(post: post)
        }
    }
}

struct PostHeader: View {
    let post: SupportPost
    
    var body: some View {
        HStack {
            // Avatar
            Circle()
                .fill(Color.blue.gradient)
                .frame(width: 40, height: 40)
                .overlay {
                    Text(post.authorName.prefix(1).uppercased())
                        .font(.headline)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                }
            
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(post.isAnonymous ? "Anonymous" : post.authorName)
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    if post.isPinned {
                        Image(systemName: "pin.fill")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                }
                
                HStack(spacing: 4) {
                    PostTypeChip(type: post.type)
                    
                    Text("•")
                        .foregroundColor(.secondary)
                    
                    Text(post.createdDate, style: .relative)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    if let location = post.location {
                        Text("•")
                            .foregroundColor(.secondary)
                        
                        Text(location)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Spacer()
            
            Menu {
                Button("Report", systemImage: "flag") {
                    // Handle report
                }
                Button("Block User", systemImage: "person.slash") {
                    // Handle block
                }
            } label: {
                Image(systemName: "ellipsis")
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct PostTypeChip: View {
    let type: PostType
    
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: type.systemImage)
            Text(type.displayName)
        }
        .font(.caption)
        .fontWeight(.medium)
        .foregroundColor(type.color)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(type.color.opacity(0.1))
        )
    }
}

struct PostContent: View {
    let post: SupportPost
    @State private var isExpanded = false
    
    private let maxLines = 4
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(post.content)
                .font(.body)
                .lineLimit(isExpanded ? nil : maxLines)
                .animation(.easeInOut(duration: 0.2), value: isExpanded)
            
            if post.content.count > 200 {
                Button(isExpanded ? "Show less" : "Show more") {
                    isExpanded.toggle()
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
            
            // Tags
            if !post.tags.isEmpty {
                TagsView(tags: post.tags)
            }
        }
    }
}

struct TagsView: View {
    let tags: [String]
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(tags, id: \.self) { tag in
                    Text("#\(tag)")
                        .font(.caption)
                        .foregroundColor(.blue)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.blue.opacity(0.1))
                        )
                }
            }
            .padding(.horizontal)
        }
    }
}

struct PostAttachmentsView: View {
    let attachments: [PostAttachment]
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(attachments) { attachment in
                    AttachmentCard(attachment: attachment)
                }
            }
            .padding(.horizontal)
        }
    }
}

struct AttachmentCard: View {
    let attachment: PostAttachment
    
    var body: some View {
        VStack(spacing: 8) {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray5))
                .frame(width: 80, height: 80)
                .overlay {
                    Image(systemName: iconForAttachment)
                        .font(.title2)
                        .foregroundColor(.secondary)
                }
            
            Text(attachment.filename)
                .font(.caption)
                .lineLimit(2)
                .multilineTextAlignment(.center)
        }
        .frame(width: 80)
    }
    
    private var iconForAttachment: String {
        switch attachment.type {
        case .image: return "photo"
        case .document: return "doc"
        case .link: return "link"
        }
    }
}

struct PostMetricsView: View {
    let mood: MoodLevel?
    let painLevel: Int?
    
    var body: some View {
        HStack(spacing: 16) {
            if let mood = mood {
                HStack(spacing: 4) {
                    Image(systemName: "heart.circle.fill")
                        .foregroundColor(mood.color)
                    Text(mood.displayName)
                        .font(.caption)
                        .fontWeight(.medium)
                }
            }
            
            if let painLevel = painLevel {
                HStack(spacing: 4) {
                    Image(systemName: "bandage.fill")
                        .foregroundColor(.red)
                    Text("Pain: \(painLevel)/10")
                        .font(.caption)
                        .fontWeight(.medium)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray6))
        )
    }
}

struct PostActions: View {
    let post: SupportPost
    let onReaction: (ReactionType) -> Void
    let onComment: () -> Void
    let onShare: () -> Void
    
    var body: some View {
        HStack(spacing: 24) {
            // Reactions
            Menu {
                ForEach(ReactionType.allCases, id: \.self) { reaction in
                    Button("\(reaction.emoji) \(reaction.displayName)") {
                        onReaction(reaction)
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "heart")
                    Text("\(post.totalReactions)")
                }
                .font(.subheadline)
                .foregroundColor(.secondary)
            }
            
            // Comments
            Button(action: onComment) {
                HStack(spacing: 4) {
                    Image(systemName: "bubble.left")
                    Text("\(post.totalComments)")
                }
                .font(.subheadline)
                .foregroundColor(.secondary)
            }
            
            // Share
            Button(action: onShare) {
                Image(systemName: "square.and.arrow.up")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
    }
}

struct ReactionSummaryView: View {
    let reactions: [PostReaction]
    
    var body: some View {
        HStack(spacing: 8) {
            let reactionCounts = Dictionary(grouping: reactions, by: { $0.type })
                .mapValues { $0.count }
                .sorted { $0.value > $1.value }
            
            ForEach(Array(reactionCounts.prefix(3)), id: \.key) { reaction, count in
                HStack(spacing: 2) {
                    Text(reaction.emoji)
                    Text("\(count)")
                        .font(.caption)
                        .fontWeight(.medium)
                }
            }
            
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray6))
        )
    }
}

struct EmptyFeedView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "heart.circle")
                .font(.system(size: 60))
                .foregroundColor(.secondary)
            
            Text("Welcome to the Community")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Join groups and start connecting with others who understand your journey.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            
            NavigationLink("Discover Groups") {
                DiscoverView()
            }
            .buttonStyle(.borderedProminent)
            .tint(Colors.Primary.p500)
        }
    }
}

// MARK: - Groups View

struct GroupsView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var showingCreateGroup = false
    @State private var selectedGroup: SupportGroup?
    @State private var searchText = ""
    
    var body: some View {
        VStack(spacing: 0) {
            // Search Bar
            SearchBar(text: $searchText, placeholder: "Search groups...")
                .padding(.horizontal)
            
            // Groups List
            ScrollView {
                LazyVStack(spacing: 16) {
                    // My Groups Section
                    if !socialManager.joinedGroups.isEmpty {
                        GroupSection(
                            title: "My Groups",
                            groups: filteredJoinedGroups,
                            onGroupTap: { group in
                                selectedGroup = group
                            }
                        )
                    }
                    
                    // Suggested Groups Section
                    if !socialManager.suggestedGroups.isEmpty {
                        GroupSection(
                            title: "Suggested for You",
                            groups: filteredSuggestedGroups,
                            onGroupTap: { group in
                                selectedGroup = group
                            }
                        )
                    }
                    
                    // All Groups Section
                    GroupSection(
                        title: "All Groups",
                        groups: filteredAvailableGroups,
                        onGroupTap: { group in
                            selectedGroup = group
                        }
                    )
                }
                .padding(.vertical)
            }
        }
        .navigationDestination(item: $selectedGroup) { group in
            GroupDetailView(group: group)
        }
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Create") {
                    showingCreateGroup = true
                }
            }
        }
        .sheet(isPresented: $showingCreateGroup) {
            CreateGroupView()
        }
        .task {
            await socialManager.loadAvailableGroups()
            await socialManager.loadSuggestedGroups()
        }
    }
    
    private var filteredJoinedGroups: [SupportGroup] {
        if searchText.isEmpty {
            return socialManager.joinedGroups
        }
        return socialManager.joinedGroups.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.description.localizedCaseInsensitiveContains(searchText)
        }
    }
    
    private var filteredSuggestedGroups: [SupportGroup] {
        if searchText.isEmpty {
            return socialManager.suggestedGroups
        }
        return socialManager.suggestedGroups.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.description.localizedCaseInsensitiveContains(searchText)
        }
    }
    
    private var filteredAvailableGroups: [SupportGroup] {
        let joinedIDs = Set(socialManager.joinedGroups.map { $0.id })
        let suggestedIDs = Set(socialManager.suggestedGroups.map { $0.id })
        
        var groups = socialManager.availableGroups.filter {
            !joinedIDs.contains($0.id) && !suggestedIDs.contains($0.id)
        }
        
        if !searchText.isEmpty {
            groups = groups.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.description.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        return groups
    }
}

struct GroupSection: View {
    let title: String
    let groups: [SupportGroup]
    let onGroupTap: (SupportGroup) -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(title)
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }
            .padding(.horizontal)
            
            ForEach(groups) { group in
                GroupCard(group: group) {
                    onGroupTap(group)
                }
                .padding(.horizontal)
            }
        }
    }
}

struct GroupCard: View {
    let group: SupportGroup
    let onTap: () -> Void
    @StateObject private var socialManager = SocialSupportManager.shared
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    // Group Icon
                    RoundedRectangle(cornerRadius: 12)
                        .fill(group.category.color.gradient)
                        .frame(width: 50, height: 50)
                        .overlay {
                            Image(systemName: group.category.systemImage)
                                .font(.title2)
                                .foregroundColor(.white)
                        }
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text(group.name)
                            .font(.headline)
                            .fontWeight(.semibold)
                            .foregroundColor(.primary)
                        
                        Text(group.category.displayName)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        HStack(spacing: 12) {
                            Label("\(group.memberCount)", systemImage: "person.2")
                            Label(group.activityLevel.displayName, systemImage: "chart.line.uptrend.xyaxis")
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    VStack(spacing: 8) {
                        if socialManager.joinedGroups.contains(where: { $0.id == group.id }) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                        } else {
                            Image(systemName: "plus.circle")
                                .foregroundColor(.blue)
                        }
                        
                        if group.privacy == .private {
                            Image(systemName: "lock.fill")
                                .font(.caption)
                                .foregroundColor(.orange)
                        }
                    }
                }
                
                Text(group.description)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                
                // Tags
                if !group.tags.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(group.tags.prefix(3), id: \.self) { tag in
                                Text(tag)
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(
                                        RoundedRectangle(cornerRadius: 8)
                                            .fill(Color(.systemGray6))
                                    )
                            }
                        }
                    }
                }
            }
            .padding(16)
            .background(Color(.systemBackground))
            .cornerRadius(12)
            .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Messages View

struct MessagesView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var searchText = ""
    @State private var selectedConversation: Conversation?
    
    var body: some View {
        VStack(spacing: 0) {
            // Search Bar
            SearchBar(text: $searchText, placeholder: "Search conversations...")
                .padding(.horizontal)
            
            // Conversations List
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(filteredConversations) { conversation in
                        ConversationRow(conversation: conversation) {
                            selectedConversation = conversation
                        }
                        .padding(.horizontal)
                        
                        Divider()
                            .padding(.leading, 80)
                    }
                }
            }
        }
        .navigationDestination(item: $selectedConversation) { conversation in
            ConversationDetailView(conversation: conversation)
        }
        .overlay {
            if socialManager.conversations.isEmpty {
                EmptyMessagesView()
            }
        }
        .task {
            await socialManager.loadConversations()
        }
    }
    
    private var filteredConversations: [Conversation] {
        if searchText.isEmpty {
            return socialManager.conversations
        }
        return socialManager.conversations.filter {
            $0.participants.contains { participant in
                participant.name.localizedCaseInsensitiveContains(searchText)
            } || $0.lastMessage?.content.localizedCaseInsensitiveContains(searchText) == true
        }
    }
}

struct ConversationRow: View {
    let conversation: Conversation
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                // Avatar
                if conversation.isGroup {
                    ZStack {
                        Circle()
                            .fill(Color.blue.gradient)
                            .frame(width: 50, height: 50)
                        
                        Image(systemName: "person.2.fill")
                            .foregroundColor(.white)
                    }
                } else {
                    Circle()
                        .fill(Color.green.gradient)
                        .frame(width: 50, height: 50)
                        .overlay {
                            Text(conversation.participants.first?.name.prefix(1).uppercased() ?? "?")
                                .font(.headline)
                                .fontWeight(.semibold)
                                .foregroundColor(.white)
                        }
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(conversation.displayName)
                            .font(.headline)
                            .fontWeight(conversation.hasUnreadMessages ? .semibold : .medium)
                            .foregroundColor(.primary)
                        
                        Spacer()
                        
                        if let lastMessage = conversation.lastMessage {
                            Text(lastMessage.timestamp, style: .time)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    HStack {
                        if let lastMessage = conversation.lastMessage {
                            Text(lastMessage.content)
                                .font(.body)
                                .foregroundColor(conversation.hasUnreadMessages ? .primary : .secondary)
                                .lineLimit(2)
                        } else {
                            Text("No messages yet")
                                .font(.body)
                                .foregroundColor(.secondary)
                                .italic()
                        }
                        
                        Spacer()
                        
                        if conversation.hasUnreadMessages {
                            Circle()
                                .fill(Color.blue)
                                .frame(width: 8, height: 8)
                        }
                    }
                }
            }
            .padding(.vertical, 12)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct EmptyMessagesView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "message.circle")
                .font(.system(size: 60))
                .foregroundColor(.secondary)
            
            Text("No Messages Yet")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Start a conversation with someone from your support groups.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
    }
}

// MARK: - Discover View

struct DiscoverView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var searchText = ""
    @State private var selectedCategory: GroupCategory?
    @State private var selectedGroup: SupportGroup?
    
    var body: some View {
        VStack(spacing: 0) {
            // Search Bar
            SearchBar(text: $searchText, placeholder: "Search groups and topics...")
                .padding(.horizontal)
            
            // Category Filter
            CategoryFilterView(selectedCategory: $selectedCategory)
            
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Featured Groups
                    if !socialManager.featuredGroups.isEmpty {
                        DiscoverSection(
                            title: "Featured Groups",
                            subtitle: "Recommended by our community",
                            groups: filteredFeaturedGroups
                        ) { group in
                            selectedGroup = group
                        }
                    }
                    
                    // Popular Groups
                    if !socialManager.popularGroups.isEmpty {
                        DiscoverSection(
                            title: "Popular This Week",
                            subtitle: "Most active groups",
                            groups: filteredPopularGroups
                        ) { group in
                            selectedGroup = group
                        }
                    }
                    
                    // New Groups
                    if !socialManager.newGroups.isEmpty {
                        DiscoverSection(
                            title: "New Groups",
                            subtitle: "Recently created",
                            groups: filteredNewGroups
                        ) { group in
                            selectedGroup = group
                        }
                    }
                    
                    // All Groups
                    DiscoverSection(
                        title: "All Groups",
                        subtitle: "Browse all available groups",
                        groups: filteredAllGroups
                    ) { group in
                        selectedGroup = group
                    }
                }
                .padding(.vertical)
            }
        }
        .navigationDestination(item: $selectedGroup) { group in
            GroupDetailView(group: group)
        }
        .task {
            await socialManager.loadFeaturedGroups()
            await socialManager.loadPopularGroups()
            await socialManager.loadNewGroups()
        }
    }
    
    private var filteredFeaturedGroups: [SupportGroup] {
        filterGroups(socialManager.featuredGroups)
    }
    
    private var filteredPopularGroups: [SupportGroup] {
        filterGroups(socialManager.popularGroups)
    }
    
    private var filteredNewGroups: [SupportGroup] {
        filterGroups(socialManager.newGroups)
    }
    
    private var filteredAllGroups: [SupportGroup] {
        filterGroups(socialManager.availableGroups)
    }
    
    private func filterGroups(_ groups: [SupportGroup]) -> [SupportGroup] {
        var filtered = groups
        
        if let selectedCategory = selectedCategory {
            filtered = filtered.filter { $0.category == selectedCategory }
        }
        
        if !searchText.isEmpty {
            filtered = filtered.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.description.localizedCaseInsensitiveContains(searchText) ||
                $0.tags.contains { $0.localizedCaseInsensitiveContains(searchText) }
            }
        }
        
        return filtered
    }
}

struct CategoryFilterView: View {
    @Binding var selectedCategory: GroupCategory?
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                CategoryChip(
                    title: "All",
                    isSelected: selectedCategory == nil
                ) {
                    selectedCategory = nil
                }
                
                ForEach(GroupCategory.allCases, id: \.self) { category in
                    CategoryChip(
                        title: category.displayName,
                        isSelected: selectedCategory == category
                    ) {
                        selectedCategory = category
                    }
                }
            }
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
    }
}

struct CategoryChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline)
                .fontWeight(isSelected ? .semibold : .medium)
                .foregroundColor(isSelected ? .white : .primary)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 20)
                        .fill(isSelected ? Color.blue : Color(.systemGray6))
                )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct DiscoverSection: View {
    let title: String
    let subtitle: String
    let groups: [SupportGroup]
    let onGroupTap: (SupportGroup) -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal)
            
            if groups.isEmpty {
                Text("No groups found")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
            } else {
                ForEach(groups.prefix(5)) { group in
                    GroupCard(group: group) {
                        onGroupTap(group)
                    }
                    .padding(.horizontal)
                }
            }
        }
    }
}

// MARK: - Profile View

struct ProfileView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var showingSettings = false
    @State private var showingEditProfile = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Profile Header
                ProfileHeaderView()
                
                // Stats
                ProfileStatsView()
                
                // Recent Activity
                RecentActivityView()
                
                // Achievements
                AchievementsView()
                
                // Support Groups
                MyGroupsView()
            }
            .padding(.vertical)
        }
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Menu {
                    Button("Edit Profile", systemImage: "pencil") {
                        showingEditProfile = true
                    }
                    Button("Settings", systemImage: "gearshape") {
                        showingSettings = true
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .sheet(isPresented: $showingSettings) {
            SocialSettingsView()
        }
        .sheet(isPresented: $showingEditProfile) {
            EditProfileView()
        }
    }
}

struct ProfileHeaderView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    
    var body: some View {
        VStack(spacing: 16) {
            // Avatar
            Circle()
                .fill(Color.blue.gradient)
                .frame(width: 100, height: 100)
                .overlay {
                    if let profile = socialManager.currentUserProfile {
                        Text(profile.displayName.prefix(1).uppercased())
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                    }
                }
            
            VStack(spacing: 8) {
                if let profile = socialManager.currentUserProfile {
                    Text(profile.displayName)
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    if let bio = profile.bio, !bio.isEmpty {
                        Text(bio)
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }
                    
                    Text("Member since \(profile.joinDate.formatted(date: .abbreviated, time: .omitted))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.horizontal)
    }
}

struct ProfileStatsView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    
    var body: some View {
        if let profile = socialManager.currentUserProfile {
            HStack(spacing: 32) {
                StatItem(
                    title: "Posts",
                    value: "\(profile.supportStats.totalPosts)"
                )
                
                StatItem(
                    title: "Helpful",
                    value: "\(profile.supportStats.helpfulReactions)"
                )
                
                StatItem(
                    title: "Groups",
                    value: "\(socialManager.joinedGroups.count)"
                )
            }
            .padding(.horizontal)
        }
    }
}

struct StatItem: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

struct RecentActivityView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Activity")
                .font(.headline)
                .fontWeight(.semibold)
                .padding(.horizontal)
            
            if socialManager.userRecentPosts.isEmpty {
                Text("No recent activity")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
            } else {
                ForEach(socialManager.userRecentPosts.prefix(3)) { post in
                    PostCard(post: post)
                        .padding(.horizontal)
                }
            }
        }
    }
}

struct AchievementsView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Achievements")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                NavigationLink("View All") {
                    AllAchievementsView()
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
            .padding(.horizontal)
            
            if let profile = socialManager.currentUserProfile {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(profile.badges.prefix(5)) { badge in
                            AchievementBadge(badge: badge)
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
    }
}

struct AchievementBadge: View {
    let badge: UserBadge
    
    var body: some View {
        VStack(spacing: 8) {
            Circle()
                .fill(badge.color.gradient)
                .frame(width: 50, height: 50)
                .overlay {
                    Image(systemName: badge.systemImage)
                        .font(.title3)
                        .foregroundColor(.white)
                }
            
            Text(badge.title)
                .font(.caption)
                .fontWeight(.medium)
                .multilineTextAlignment(.center)
                .lineLimit(2)
        }
        .frame(width: 70)
    }
}

struct MyGroupsView: View {
    @StateObject private var socialManager = SocialSupportManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("My Groups")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                NavigationLink("View All") {
                    GroupsView()
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
            .padding(.horizontal)
            
            if socialManager.joinedGroups.isEmpty {
                VStack(spacing: 12) {
                    Text("You haven't joined any groups yet")
                        .font(.body)
                        .foregroundColor(.secondary)
                    
                    NavigationLink("Discover Groups") {
                        DiscoverView()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                }
                .padding(.horizontal)
            } else {
                ForEach(socialManager.joinedGroups.prefix(3)) { group in
                    GroupCard(group: group) {
                        // Navigate to group detail
                    }
                    .padding(.horizontal)
                }
            }
        }
    }
}

// MARK: - Supporting Views

struct SearchBar: View {
    @Binding var text: String
    let placeholder: String
    
    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)
            
            TextField(placeholder, text: $text)
                .textFieldStyle(PlainTextFieldStyle())
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(.systemGray6))
        )
    }
}

struct CreatePostView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var content = ""
    @State private var selectedGroup: SupportGroup?
    @State private var postType: PostType = .general
    @State private var isAnonymous = false
    @State private var mood: MoodLevel?
    @State private var painLevel: Int?
    @State private var tags: [String] = []
    @State private var newTag = ""
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Group Selection
                    GroupSelectionView(selectedGroup: $selectedGroup)
                    
                    // Post Type
                    PostTypeSelectionView(selectedType: $postType)
                    
                    // Content
                    VStack(alignment: .leading, spacing: 8) {
                        Text("What's on your mind?")
                            .font(.headline)
                        
                        TextEditor(text: $content)
                            .frame(minHeight: 120)
                            .padding(12)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color(.systemGray6))
                            )
                    }
                    
                    // Mood and Pain Level
                    MoodPainSelectionView(mood: $mood, painLevel: $painLevel)
                    
                    // Tags
                    TagsInputView(tags: $tags, newTag: $newTag)
                    
                    // Privacy
                    Toggle("Post anonymously", isOn: $isAnonymous)
                        .padding(.horizontal)
                }
                .padding()
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
                        Task {
                            await createPost()
                        }
                    }
                    .disabled(content.isEmpty || selectedGroup == nil)
                }
            }
        }
    }
    
    private func createPost() async {
        guard let selectedGroup = selectedGroup else { return }
        
        let post = SupportPost(
            id: UUID().uuidString,
            groupID: selectedGroup.id,
            authorID: socialManager.currentUserProfile?.id ?? "",
            authorName: socialManager.currentUserProfile?.displayName ?? "Anonymous",
            content: content,
            type: postType,
            isAnonymous: isAnonymous,
            mood: mood,
            painLevel: painLevel,
            tags: tags,
            attachments: [],
            reactions: [],
            comments: [],
            createdDate: Date(),
            updatedDate: Date(),
            isPinned: false,
            location: nil
        )
        
        do {
            try await socialManager.createPost(post)
            dismiss()
        } catch {
            // Handle error
        }
    }
}

struct CreateGroupView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var socialManager = SocialSupportManager.shared
    @State private var name = ""
    @State private var description = ""
    @State private var category: GroupCategory = .general
    @State private var privacy: GroupPrivacy = .public
    @State private var tags: [String] = []
    @State private var newTag = ""
    @State private var guidelines = ""
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Basic Info
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Basic Information")
                            .font(.headline)
                        
                        TextField("Group name", text: $name)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                        
                        TextField("Description", text: $description, axis: .vertical)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .lineLimit(3...6)
                    }
                    
                    // Category
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Category")
                            .font(.headline)
                        
                        Picker("Category", selection: $category) {
                            ForEach(GroupCategory.allCases, id: \.self) { category in
                                Text(category.displayName).tag(category)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                    }
                    
                    // Privacy
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Privacy")
                            .font(.headline)
                        
                        Picker("Privacy", selection: $privacy) {
                            ForEach(GroupPrivacy.allCases, id: \.self) { privacy in
                                Text(privacy.displayName).tag(privacy)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                    }
                    
                    // Tags
                    TagsInputView(tags: $tags, newTag: $newTag)
                    
                    // Guidelines
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Community Guidelines")
                            .font(.headline)
                        
                        TextEditor(text: $guidelines)
                            .frame(minHeight: 100)
                            .padding(12)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color(.systemGray6))
                            )
                    }
                }
                .padding()
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
                        Task {
                            await createGroup()
                        }
                    }
                    .disabled(name.isEmpty || description.isEmpty)
                }
            }
        }
    }
    
    private func createGroup() async {
        let group = SupportGroup(
            id: UUID().uuidString,
            name: name,
            description: description,
            category: category,
            privacy: privacy,
            memberCount: 1,
            activityLevel: .low,
            tags: tags,
            guidelines: guidelines.isEmpty ? nil : guidelines,
            createdDate: Date(),
            location: nil,
            meetingSchedule: nil,
            resources: []
        )
        
        do {
            try await socialManager.createGroup(group)
            dismiss()
        } catch {
            // Handle error
        }
    }
}

struct SocialSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var socialManager = SocialSupportManager.shared
    
    var body: some View {
        NavigationView {
            List {
                Section("Privacy") {
                    NavigationLink("Privacy Settings") {
                        PrivacySettingsView()
                    }
                    
                    NavigationLink("Blocked Users") {
                        BlockedUsersView()
                    }
                }
                
                Section("Notifications") {
                    NavigationLink("Notification Preferences") {
                        NotificationSettingsView()
                    }
                }
                
                Section("Content") {
                    NavigationLink("Content Preferences") {
                        ContentPreferencesView()
                    }
                }
                
                Section("Support") {
                    NavigationLink("Report a Problem") {
                        ReportProblemView()
                    }
                    
                    NavigationLink("Community Guidelines") {
                        CommunityGuidelinesView()
                    }
                }
            }
            .navigationTitle("Social Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Helper Views

struct GroupSelectionView: View {
    @Binding var selectedGroup: SupportGroup?
    @StateObject private var socialManager = SocialSupportManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Select Group")
                .font(.headline)
            
            Menu {
                ForEach(socialManager.joinedGroups) { group in
                    Button(group.name) {
                        selectedGroup = group
                    }
                }
            } label: {
                HStack {
                    Text(selectedGroup?.name ?? "Choose a group")
                        .foregroundColor(selectedGroup == nil ? .secondary : .primary)
                    Spacer()
                    Image(systemName: "chevron.down")
                        .foregroundColor(.secondary)
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color(.systemGray6))
                )
            }
        }
    }
}

struct PostTypeSelectionView: View {
    @Binding var selectedType: PostType
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Post Type")
                .font(.headline)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(PostType.allCases, id: \.self) { type in
                        Button {
                            selectedType = type
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: type.systemImage)
                                Text(type.displayName)
                            }
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(selectedType == type ? .white : .primary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 20)
                                    .fill(selectedType == type ? type.color : Color(.systemGray6))
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

struct MoodPainSelectionView: View {
    @Binding var mood: MoodLevel?
    @Binding var painLevel: Int?
    
    var body: some View {
        VStack(spacing: 16) {
            // Mood Selection
            VStack(alignment: .leading, spacing: 8) {
                Text("How are you feeling?")
                    .font(.headline)
                
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(MoodLevel.allCases, id: \.self) { moodOption in
                            Button {
                                mood = mood == moodOption ? nil : moodOption
                            } label: {
                                VStack(spacing: 4) {
                                    Text(moodOption.emoji)
                                        .font(.title2)
                                    Text(moodOption.displayName)
                                        .font(.caption)
                                        .fontWeight(.medium)
                                }
                                .padding(8)
                                .background(
                                    RoundedRectangle(cornerRadius: 10)
                                        .fill(mood == moodOption ? moodOption.color.opacity(0.2) : Color(.systemGray6))
                                )
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                    .padding(.horizontal)
                }
            }
            
            // Pain Level Selection
            VStack(alignment: .leading, spacing: 8) {
                Text("Pain Level (Optional)")
                    .font(.headline)
                
                HStack {
                    Text("0")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Slider(
                        value: Binding(
                            get: { Double(painLevel ?? 0) },
                            set: { painLevel = Int($0) }
                        ),
                        in: 0...10,
                        step: 1
                    )
                    
                    Text("10")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                if let painLevel = painLevel, painLevel > 0 {
                    Text("Pain Level: \(painLevel)/10")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }
}

struct TagsInputView: View {
    @Binding var tags: [String]
    @Binding var newTag: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Tags")
                .font(.headline)
            
            HStack {
                TextField("Add tag", text: $newTag)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .onSubmit {
                        addTag()
                    }
                
                Button("Add") {
                    addTag()
                }
                .disabled(newTag.isEmpty)
            }
            
            if !tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(tags, id: \.self) { tag in
                            HStack(spacing: 4) {
                                Text("#\(tag)")
                                    .font(.caption)
                                
                                Button {
                                    tags.removeAll { $0 == tag }
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.caption)
                                }
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.blue.opacity(0.1))
                            )
                            .foregroundColor(.blue)
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
    }
    
    private func addTag() {
        let trimmed = newTag.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty && !tags.contains(trimmed) {
            tags.append(trimmed)
            newTag = ""
        }
    }
}

// MARK: - Placeholder Views

struct GroupDetailView: View {
    let group: SupportGroup
    
    var body: some View {
        Text("Group Detail: \(group.name)")
            .navigationTitle(group.name)
    }
}

struct ConversationDetailView: View {
    let conversation: Conversation
    
    var body: some View {
        Text("Conversation: \(conversation.displayName)")
            .navigationTitle(conversation.displayName)
    }
}

struct PostCommentsView: View {
    let post: SupportPost
    
    var body: some View {
        Text("Comments for post")
            .navigationTitle("Comments")
    }
}

struct AllAchievementsView: View {
    var body: some View {
        Text("All Achievements")
            .navigationTitle("Achievements")
    }
}

struct EditProfileView: View {
    var body: some View {
        Text("Edit Profile")
            .navigationTitle("Edit Profile")
    }
}

struct PrivacySettingsView: View {
    var body: some View {
        Text("Privacy Settings")
            .navigationTitle("Privacy")
    }
}

struct BlockedUsersView: View {
    var body: some View {
        Text("Blocked Users")
            .navigationTitle("Blocked Users")
    }
}

struct NotificationSettingsView: View {
    var body: some View {
        Text("Notification Settings")
            .navigationTitle("Notifications")
    }
}

struct ContentPreferencesView: View {
    var body: some View {
        Text("Content Preferences")
            .navigationTitle("Content")
    }
}

struct ReportProblemView: View {
    var body: some View {
        Text("Report a Problem")
            .navigationTitle("Report Problem")
    }
}

struct CommunityGuidelinesView: View {
    var body: some View {
        Text("Community Guidelines")
            .navigationTitle("Guidelines")
    }
}