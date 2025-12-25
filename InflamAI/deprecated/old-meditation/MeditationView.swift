//
//  MeditationView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import AVFoundation
import HealthKit

struct MeditationView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var selectedTab = 0
    @State private var showingSessionDetail = false
    @State private var selectedSession: MeditationSession?
    @State private var showingSettings = false
    @State private var showingInsights = false
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // Discover Tab
                DiscoverView()
                    .tabItem {
                        Image(systemName: "sparkles")
                        Text("Discover")
                    }
                    .tag(0)
                
                // Library Tab
                LibraryView()
                    .tabItem {
                        Image(systemName: "books.vertical")
                        Text("Library")
                    }
                    .tag(1)
                
                // Player Tab
                PlayerView()
                    .tabItem {
                        Image(systemName: "play.circle")
                        Text("Player")
                    }
                    .tag(2)
                
                // Progress Tab
                ProgressView()
                    .tabItem {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                        Text("Progress")
                    }
                    .tag(3)
                
                // Profile Tab
                ProfileView()
                    .tabItem {
                        Image(systemName: "person.circle")
                        Text("Profile")
                    }
                    .tag(4)
            }
            .navigationTitle("Meditation")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingSettings = true }) {
                        Image(systemName: "gearshape")
                    }
                }
                
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: { showingInsights = true }) {
                        Image(systemName: "lightbulb")
                    }
                }
            }
        }
        .sheet(isPresented: $showingSettings) {
            MeditationSettingsView()
        }
        .sheet(isPresented: $showingInsights) {
            InsightsView()
        }
        .alert("Error", isPresented: .constant(meditationManager.error != nil)) {
            Button("OK") {
                meditationManager.error = nil
            }
        } message: {
            Text(meditationManager.error?.localizedDescription ?? "")
        }
    }
}

// MARK: - Discover View

struct DiscoverView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var searchText = ""
    @State private var selectedCategory: MeditationCategory?
    @State private var selectedType: MeditationType?
    
    var filteredSessions: [MeditationSession] {
        var sessions = meditationManager.availableSessions
        
        if !searchText.isEmpty {
            sessions = meditationManager.searchSessions(searchText)
        }
        
        if let category = selectedCategory {
            sessions = sessions.filter { $0.category == category }
        }
        
        if let type = selectedType {
            sessions = sessions.filter { $0.type == type }
        }
        
        return sessions
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Search Bar
                SearchBar(text: $searchText)
                
                // Quick Stats
                QuickStatsCard()
                
                // Recommended Sessions
                RecommendedSessionsSection()
                
                // Categories Filter
                CategoriesFilterSection(
                    selectedCategory: $selectedCategory,
                    selectedType: $selectedType
                )
                
                // Sessions Grid
                SessionsGridSection(sessions: filteredSessions)
            }
            .padding()
        }
        .refreshable {
            // Refresh content
        }
    }
}

struct SearchBar: View {
    @Binding var text: String
    
    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)
            
            TextField("Search meditations...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
        }
        .padding(.horizontal)
    }
}

struct QuickStatsCard: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Your Progress")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }
            
            HStack(spacing: 20) {
                StatItem(
                    title: "Current Streak",
                    value: "\(meditationManager.streak.currentStreak)",
                    unit: "days",
                    color: .orange
                )
                
                StatItem(
                    title: "Total Sessions",
                    value: "\(meditationManager.streak.totalSessions)",
                    unit: "",
                    color: .blue
                )
                
                StatItem(
                    title: "Total Time",
                    value: "\(Int(meditationManager.streak.totalMinutes / 60))",
                    unit: "hours",
                    color: .green
                )
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct StatItem: View {
    let title: String
    let value: String
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            if !unit.isEmpty {
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
    }
}

struct RecommendedSessionsSection: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recommended for You")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
                
                Button("See All") {
                    // Show all recommendations
                }
                .font(.subheadline)
                .foregroundColor(.accentColor)
            }
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    ForEach(meditationManager.getRecommendedSessions()) { session in
                        RecommendedSessionCard(session: session)
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

struct RecommendedSessionCard: View {
    let session: MeditationSession
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Session Image
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(session.type.color))
                .frame(width: 160, height: 100)
                .overlay {
                    Image(systemName: session.type.systemImage)
                        .font(.title)
                        .foregroundColor(.white)
                }
            
            // Session Info
            VStack(alignment: .leading, spacing: 4) {
                Text(session.title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .lineLimit(2)
                
                Text("\(Int(session.duration / 60)) min")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack {
                    DifficultyBadge(difficulty: session.difficulty)
                    Spacer()
                    
                    if meditationManager.isSessionDownloaded(session) {
                        Image(systemName: "arrow.down.circle.fill")
                            .foregroundColor(.green)
                            .font(.caption)
                    }
                }
            }
        }
        .frame(width: 160)
        .onTapGesture {
            meditationManager.startSession(session)
        }
    }
}

struct DifficultyBadge: View {
    let difficulty: DifficultyLevel
    
    var body: some View {
        Text(difficulty.displayName)
            .font(.caption2)
            .fontWeight(.medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Color(difficulty.color).opacity(0.2))
            .foregroundColor(Color(difficulty.color))
            .cornerRadius(4)
    }
}

struct CategoriesFilterSection: View {
    @Binding var selectedCategory: MeditationCategory?
    @Binding var selectedType: MeditationType?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Categories")
                .font(.headline)
                .fontWeight(.semibold)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    FilterChip(
                        title: "All",
                        isSelected: selectedCategory == nil && selectedType == nil
                    ) {
                        selectedCategory = nil
                        selectedType = nil
                    }
                    
                    ForEach(MeditationCategory.allCases, id: \.self) { category in
                        FilterChip(
                            title: category.displayName,
                            isSelected: selectedCategory == category
                        ) {
                            selectedCategory = selectedCategory == category ? nil : category
                            selectedType = nil
                        }
                    }
                    
                    ForEach(MeditationType.allCases, id: \.self) { type in
                        FilterChip(
                            title: type.displayName,
                            isSelected: selectedType == type
                        ) {
                            selectedType = selectedType == type ? nil : type
                            selectedCategory = nil
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
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
                .fontWeight(.medium)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(isSelected ? Color.accentColor : Color(.systemGray5))
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(20)
        }
    }
}

struct SessionsGridSection: View {
    let sessions: [MeditationSession]
    
    let columns = [
        GridItem(.flexible()),
        GridItem(.flexible())
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("All Sessions")
                .font(.headline)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: columns, spacing: 16) {
                ForEach(sessions) { session in
                    SessionCard(session: session)
                }
            }
        }
    }
}

struct SessionCard: View {
    let session: MeditationSession
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var showingDetail = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: session.type.systemImage)
                    .font(.title2)
                    .foregroundColor(Color(session.type.color))
                
                Spacer()
                
                Menu {
                    Button("Play") {
                        meditationManager.startSession(session)
                    }
                    
                    Button("View Details") {
                        showingDetail = true
                    }
                    
                    if meditationManager.isSessionDownloaded(session) {
                        Button("Remove Download", role: .destructive) {
                            meditationManager.deleteDownloadedSession(session)
                        }
                    } else {
                        Button("Download") {
                            meditationManager.downloadSession(session)
                        }
                    }
                } label: {
                    Image(systemName: "ellipsis")
                        .foregroundColor(.secondary)
                }
            }
            
            // Content
            VStack(alignment: .leading, spacing: 8) {
                Text(session.title)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .lineLimit(2)
                
                Text(session.description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(3)
                
                HStack {
                    Text("\(Int(session.duration / 60)) min")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    DifficultyBadge(difficulty: session.difficulty)
                }
            }
            
            // Download Progress
            if let progress = meditationManager.downloadProgress[session.id] {
                ProgressView(value: progress)
                    .progressViewStyle(LinearProgressViewStyle())
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        .onTapGesture {
            showingDetail = true
        }
        .sheet(isPresented: $showingDetail) {
            SessionDetailView(session: session)
        }
    }
}

// MARK: - Library View

struct LibraryView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var selectedFilter = LibraryFilter.all
    
    enum LibraryFilter: String, CaseIterable {
        case all = "All"
        case downloaded = "Downloaded"
        case favorites = "Favorites"
        case recent = "Recent"
        
        var systemImage: String {
            switch self {
            case .all: return "square.grid.2x2"
            case .downloaded: return "arrow.down.circle"
            case .favorites: return "heart"
            case .recent: return "clock"
            }
        }
    }
    
    var filteredSessions: [MeditationSession] {
        switch selectedFilter {
        case .all:
            return meditationManager.availableSessions
        case .downloaded:
            return meditationManager.availableSessions.filter { meditationManager.isSessionDownloaded($0) }
        case .favorites:
            return [] // Implement favorites functionality
        case .recent:
            return [] // Implement recent sessions functionality
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Filter Tabs
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    ForEach(LibraryFilter.allCases, id: \.self) { filter in
                        FilterTab(
                            title: filter.rawValue,
                            icon: filter.systemImage,
                            isSelected: selectedFilter == filter
                        ) {
                            selectedFilter = filter
                        }
                    }
                }
                .padding(.horizontal)
            }
            .padding(.vertical, 8)
            
            Divider()
            
            // Content
            if filteredSessions.isEmpty {
                EmptyLibraryView(filter: selectedFilter)
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(filteredSessions) { session in
                            LibrarySessionRow(session: session)
                        }
                    }
                    .padding()
                }
            }
        }
    }
}

struct FilterTab: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.caption)
                
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(isSelected ? Color.accentColor : Color.clear)
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(20)
            .overlay {
                if !isSelected {
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                }
            }
        }
    }
}

struct EmptyLibraryView: View {
    let filter: LibraryView.LibraryFilter
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: filter.systemImage)
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            
            Text(emptyMessage)
                .font(.headline)
                .multilineTextAlignment(.center)
            
            Text(emptyDescription)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private var emptyMessage: String {
        switch filter {
        case .all: return "No Sessions Available"
        case .downloaded: return "No Downloaded Sessions"
        case .favorites: return "No Favorite Sessions"
        case .recent: return "No Recent Sessions"
        }
    }
    
    private var emptyDescription: String {
        switch filter {
        case .all: return "Check your internet connection and try again."
        case .downloaded: return "Download sessions to access them offline."
        case .favorites: return "Mark sessions as favorites to see them here."
        case .recent: return "Your recently played sessions will appear here."
        }
    }
}

struct LibrarySessionRow: View {
    let session: MeditationSession
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        HStack(spacing: 12) {
            // Session Icon
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(session.type.color))
                .frame(width: 60, height: 60)
                .overlay {
                    Image(systemName: session.type.systemImage)
                        .font(.title2)
                        .foregroundColor(.white)
                }
            
            // Session Info
            VStack(alignment: .leading, spacing: 4) {
                Text(session.title)
                    .font(.headline)
                    .fontWeight(.medium)
                    .lineLimit(1)
                
                Text(session.description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                HStack {
                    Text("\(Int(session.duration / 60)) min")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("•")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(session.type.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            // Actions
            VStack(spacing: 8) {
                Button {
                    meditationManager.startSession(session)
                } label: {
                    Image(systemName: "play.fill")
                        .font(.title3)
                        .foregroundColor(.accentColor)
                }
                
                if meditationManager.isSessionDownloaded(session) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.05), radius: 1, x: 0, y: 1)
    }
}

// MARK: - Player View

struct PlayerView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var showingSessionPicker = false
    
    var body: some View {
        VStack(spacing: 0) {
            if let session = meditationManager.currentSession {
                ActivePlayerView(session: session)
            } else {
                EmptyPlayerView(showingSessionPicker: $showingSessionPicker)
            }
        }
        .sheet(isPresented: $showingSessionPicker) {
            SessionPickerView()
        }
    }
}

struct ActivePlayerView: View {
    let session: MeditationSession
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var showingControls = true
    
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                // Background
                LinearGradient(
                    colors: [Color(session.type.color), Color(session.type.color).opacity(0.3)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()
                .overlay {
                    // Session Visualization
                    SessionVisualizationView(session: session)
                        .opacity(showingControls ? 0.3 : 0.8)
                }
                .onTapGesture {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        showingControls.toggle()
                    }
                }
                
                // Player Controls
                if showingControls {
                    PlayerControlsView(session: session)
                        .background(Color(.systemBackground))
                        .transition(.move(edge: .bottom))
                }
            }
        }
    }
}

struct SessionVisualizationView: View {
    let session: MeditationSession
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var animationPhase = 0.0
    
    var body: some View {
        VStack(spacing: 40) {
            Spacer()
            
            // Session Icon with Animation
            ZStack {
                // Breathing circles
                ForEach(0..<3, id: \.self) { index in
                    Circle()
                        .stroke(Color.white.opacity(0.3), lineWidth: 2)
                        .frame(width: 100 + CGFloat(index * 40))
                        .scaleEffect(1 + sin(animationPhase + Double(index) * 0.5) * 0.1)
                }
                
                // Main icon
                Image(systemName: session.type.systemImage)
                    .font(.system(size: 48))
                    .foregroundColor(.white)
                    .scaleEffect(1 + sin(animationPhase) * 0.05)
            }
            .onAppear {
                withAnimation(.linear(duration: 4).repeatForever(autoreverses: false)) {
                    animationPhase = .pi * 2
                }
            }
            
            // Session Info
            VStack(spacing: 8) {
                Text(session.title)
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                
                Text(session.type.displayName)
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.8))
            }
            
            Spacer()
        }
        .padding()
    }
}

struct PlayerControlsView: View {
    let session: MeditationSession
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(spacing: 24) {
            // Progress Bar
            VStack(spacing: 8) {
                HStack {
                    Text(formatTime(meditationManager.currentTime))
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text(formatTime(session.duration))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                ProgressView(value: meditationManager.progress)
                    .progressViewStyle(LinearProgressViewStyle(tint: Color(session.type.color)))
            }
            
            // Main Controls
            HStack(spacing: 40) {
                // Previous/Rewind
                Button {
                    let newTime = max(0, meditationManager.currentTime - 15)
                    meditationManager.seekTo(newTime)
                } label: {
                    Image(systemName: "gobackward.15")
                        .font(.title2)
                        .foregroundColor(.primary)
                }
                
                // Play/Pause
                Button {
                    if meditationManager.isPlaying {
                        meditationManager.pauseSession()
                    } else if meditationManager.isPaused {
                        meditationManager.resumeSession()
                    } else {
                        meditationManager.startSession(session)
                    }
                } label: {
                    Image(systemName: meditationManager.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 64))
                        .foregroundColor(Color(session.type.color))
                }
                
                // Next/Forward
                Button {
                    let newTime = min(session.duration, meditationManager.currentTime + 15)
                    meditationManager.seekTo(newTime)
                } label: {
                    Image(systemName: "goforward.15")
                        .font(.title2)
                        .foregroundColor(.primary)
                }
            }
            
            // Secondary Controls
            HStack(spacing: 32) {
                // Volume
                HStack(spacing: 8) {
                    Image(systemName: "speaker.fill")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Slider(value: Binding(
                        get: { meditationManager.volume },
                        set: { meditationManager.setVolume($0) }
                    ), in: 0...1)
                    .frame(width: 80)
                }
                
                // Playback Rate
                Menu {
                    ForEach([0.5, 0.75, 1.0, 1.25, 1.5], id: \.self) { rate in
                        Button("\(rate, specifier: "%.2f")x") {
                            meditationManager.setPlaybackRate(Float(rate))
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("\(meditationManager.playbackRate, specifier: "%.2f")x")
                            .font(.caption)
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.caption2)
                    }
                    .foregroundColor(.secondary)
                }
                
                // Stop
                Button {
                    meditationManager.stopSession()
                } label: {
                    Image(systemName: "stop.fill")
                        .font(.title3)
                        .foregroundColor(.red)
                }
            }
        }
        .padding()
    }
    
    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

struct EmptyPlayerView: View {
    @Binding var showingSessionPicker: Bool
    
    var body: some View {
        VStack(spacing: 24) {
            Spacer()
            
            Image(systemName: "play.circle")
                .font(.system(size: 80))
                .foregroundColor(.secondary)
            
            VStack(spacing: 8) {
                Text("No Session Playing")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Choose a meditation session to begin your practice")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            
            Button("Choose Session") {
                showingSessionPicker = true
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            
            Spacer()
        }
        .padding()
    }
}

struct SessionPickerView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                ForEach(meditationManager.availableSessions) { session in
                    SessionPickerRow(session: session) {
                        meditationManager.startSession(session)
                        dismiss()
                    }
                }
            }
            .navigationTitle("Choose Session")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct SessionPickerRow: View {
    let session: MeditationSession
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: session.type.systemImage)
                    .font(.title2)
                    .foregroundColor(Color(session.type.color))
                    .frame(width: 32)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(session.title)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text("\(Int(session.duration / 60)) min • \(session.type.displayName)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                DifficultyBadge(difficulty: session.difficulty)
            }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Progress View

struct ProgressView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var selectedPeriod = InsightPeriod.weekly
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Period Selector
                PeriodSelector(selectedPeriod: $selectedPeriod)
                
                // Streak Card
                StreakCard()
                
                // Statistics Cards
                StatisticsSection()
                
                // Insights
                InsightsSection()
                
                // Achievements
                AchievementsSection()
            }
            .padding()
        }
    }
}

struct PeriodSelector: View {
    @Binding var selectedPeriod: InsightPeriod
    
    var body: some View {
        Picker("Period", selection: $selectedPeriod) {
            ForEach(InsightPeriod.allCases, id: \.self) { period in
                Text(period.displayName).tag(period)
            }
        }
        .pickerStyle(SegmentedPickerStyle())
    }
}

struct StreakCard: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Meditation Streak")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }
            
            HStack(spacing: 40) {
                VStack(spacing: 8) {
                    Text("\(meditationManager.streak.currentStreak)")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.orange)
                    
                    Text("Current Streak")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                VStack(spacing: 8) {
                    Text("\(meditationManager.streak.longestStreak)")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                    
                    Text("Longest Streak")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Weekly Progress
            WeeklyProgressView()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct WeeklyProgressView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text("This Week")
                    .font(.subheadline)
                    .fontWeight(.medium)
                Spacer()
                Text("\(completedDays)/7 days")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            HStack(spacing: 4) {
                ForEach(0..<7, id: \.self) { day in
                    Circle()
                        .fill(isDayCompleted(day) ? Color.green : Color(.systemGray4))
                        .frame(width: 32, height: 32)
                        .overlay {
                            Text(dayLetter(day))
                                .font(.caption2)
                                .fontWeight(.medium)
                                .foregroundColor(isDayCompleted(day) ? .white : .secondary)
                        }
                }
            }
        }
    }
    
    private var completedDays: Int {
        // Calculate completed days this week
        return 3 // Placeholder
    }
    
    private func isDayCompleted(_ day: Int) -> Bool {
        // Check if meditation was completed on this day
        return day < completedDays // Placeholder
    }
    
    private func dayLetter(_ day: Int) -> String {
        let days = ["S", "M", "T", "W", "T", "F", "S"]
        return days[day]
    }
}

struct StatisticsSection: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Statistics")
                .font(.headline)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                StatCard(
                    title: "Total Sessions",
                    value: "\(meditationManager.streak.totalSessions)",
                    icon: "play.circle",
                    color: .blue
                )
                
                StatCard(
                    title: "Total Time",
                    value: "\(Int(meditationManager.streak.totalMinutes / 60))h",
                    icon: "clock",
                    color: .green
                )
                
                StatCard(
                    title: "Average Session",
                    value: "\(Int(meditationManager.streak.averageSessionLength / 60))m",
                    icon: "chart.bar",
                    color: .orange
                )
                
                StatCard(
                    title: "Weekly Goal",
                    value: "\(completedThisWeek)/\(meditationManager.streak.weeklyGoal)",
                    icon: "target",
                    color: .purple
                )
            }
        }
    }
    
    private var completedThisWeek: Int {
        // Calculate sessions completed this week
        return 3 // Placeholder
    }
}

struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(color)
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 2) {
                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.05), radius: 1, x: 0, y: 1)
    }
}

struct InsightsSection: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Insights")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Button("View All") {
                    // Show all insights
                }
                .font(.subheadline)
                .foregroundColor(.accentColor)
            }
            
            if meditationManager.insights.isEmpty {
                Text("Complete more sessions to see personalized insights")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
            } else {
                ForEach(meditationManager.insights.prefix(3)) { insight in
                    InsightCard(insight: insight)
                }
            }
        }
    }
}

struct InsightCard: View {
    let insight: MeditationInsight
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: insight.category.systemImage)
                    .font(.title3)
                    .foregroundColor(.accentColor)
                
                Text(insight.title)
                    .font(.headline)
                    .fontWeight(.medium)
                
                Spacer()
            }
            
            Text(insight.description)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .lineLimit(3)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.05), radius: 1, x: 0, y: 1)
    }
}

struct AchievementsSection: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Achievements")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Button("View All") {
                    // Show all achievements
                }
                .font(.subheadline)
                .foregroundColor(.accentColor)
            }
            
            if meditationManager.achievements.isEmpty {
                Text("Start meditating to unlock achievements")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(meditationManager.achievements.prefix(5)) { achievement in
                            AchievementBadge(achievement: achievement)
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
    }
}

struct AchievementBadge: View {
    let achievement: MeditationAchievement
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: achievement.icon)
                .font(.title)
                .foregroundColor(achievement.isUnlocked ? .yellow : .gray)
                .frame(width: 60, height: 60)
                .background(Circle().fill(Color(.systemGray6)))
            
            Text(achievement.title)
                .font(.caption)
                .fontWeight(.medium)
                .multilineTextAlignment(.center)
                .lineLimit(2)
        }
        .frame(width: 80)
        .opacity(achievement.isUnlocked ? 1.0 : 0.5)
    }
}

// MARK: - Profile View

struct ProfileView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @State private var showingSettings = false
    @State private var showingAchievements = false
    @State private var showingInsights = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Profile Header
                ProfileHeaderView()
                
                // Quick Stats
                ProfileStatsView()
                
                // Menu Items
                VStack(spacing: 0) {
                    ProfileMenuItem(
                        title: "Settings",
                        icon: "gearshape",
                        action: { showingSettings = true }
                    )
                    
                    Divider().padding(.leading, 44)
                    
                    ProfileMenuItem(
                        title: "Achievements",
                        icon: "trophy",
                        action: { showingAchievements = true }
                    )
                    
                    Divider().padding(.leading, 44)
                    
                    ProfileMenuItem(
                        title: "Insights",
                        icon: "lightbulb",
                        action: { showingInsights = true }
                    )
                    
                    Divider().padding(.leading, 44)
                    
                    ProfileMenuItem(
                        title: "Export Data",
                        icon: "square.and.arrow.up",
                        action: { /* Export data */ }
                    )
                    
                    Divider().padding(.leading, 44)
                    
                    ProfileMenuItem(
                        title: "Help & Support",
                        icon: "questionmark.circle",
                        action: { /* Show help */ }
                    )
                }
                .background(Color(.systemBackground))
                .cornerRadius(12)
            }
            .padding()
        }
        .sheet(isPresented: $showingSettings) {
            MeditationSettingsView()
        }
        .sheet(isPresented: $showingAchievements) {
            AllAchievementsView()
        }
        .sheet(isPresented: $showingInsights) {
            InsightsView()
        }
    }
}

struct ProfileHeaderView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        VStack(spacing: 16) {
            // Avatar
            Circle()
                .fill(LinearGradient(
                    colors: [.blue, .purple],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ))
                .frame(width: 80, height: 80)
                .overlay {
                    Text("U")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }
            
            // User Info
            VStack(spacing: 4) {
                Text("Meditation User")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Member since \(memberSince)")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            // Level Badge
            HStack(spacing: 8) {
                Image(systemName: "star.fill")
                    .foregroundColor(.yellow)
                
                Text("Level \(currentLevel)")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text("•")
                    .foregroundColor(.secondary)
                
                Text("\(meditationManager.streak.totalSessions) sessions")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
    }
    
    private var memberSince: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: Date().addingTimeInterval(-86400 * 30)) // 30 days ago
    }
    
    private var currentLevel: Int {
        // Calculate level based on total sessions
        return max(1, meditationManager.streak.totalSessions / 10)
    }
}

struct ProfileStatsView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    
    var body: some View {
        HStack(spacing: 20) {
            ProfileStatItem(
                title: "Streak",
                value: "\(meditationManager.streak.currentStreak)",
                unit: "days",
                color: .orange
            )
            
            Divider()
                .frame(height: 40)
            
            ProfileStatItem(
                title: "Total Time",
                value: "\(Int(meditationManager.streak.totalMinutes / 60))",
                unit: "hours",
                color: .green
            )
            
            Divider()
                .frame(height: 40)
            
            ProfileStatItem(
                title: "Sessions",
                value: "\(meditationManager.streak.totalSessions)",
                unit: "",
                color: .blue
            )
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct ProfileStatItem: View {
    let title: String
    let value: String
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            if !unit.isEmpty {
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct ProfileMenuItem: View {
    let title: String
    let icon: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundColor(.accentColor)
                    .frame(width: 24)
                
                Text(title)
                    .font(.body)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Supporting Views

struct SessionDetailView: View {
    let session: MeditationSession
    @StateObject private var meditationManager = MeditationManager.shared
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: session.type.systemImage)
                                .font(.title)
                                .foregroundColor(Color(session.type.color))
                            
                            Spacer()
                            
                            DifficultyBadge(difficulty: session.difficulty)
                        }
                        
                        Text(session.title)
                            .font(.largeTitle)
                            .fontWeight(.bold)
                        
                        Text(session.type.displayName)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    // Description
                    Text(session.description)
                        .font(.body)
                    
                    // Details
                    VStack(alignment: .leading, spacing: 8) {
                        DetailRow(title: "Duration", value: "\(Int(session.duration / 60)) minutes")
                        DetailRow(title: "Category", value: session.category.displayName)
                        DetailRow(title: "Instructor", value: session.instructor ?? "Guided")
                    }
                    
                    // Benefits
                    if !session.benefits.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Benefits")
                                .font(.headline)
                                .fontWeight(.semibold)
                            
                            ForEach(session.benefits, id: \.self) { benefit in
                                HStack {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                    Text(benefit)
                                        .font(.subheadline)
                                }
                            }
                        }
                    }
                    
                    // Action Buttons
                    VStack(spacing: 12) {
                        Button("Start Session") {
                            meditationManager.startSession(session)
                            dismiss()
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .frame(maxWidth: .infinity)
                        
                        if meditationManager.isSessionDownloaded(session) {
                            Button("Remove Download") {
                                meditationManager.deleteDownloadedSession(session)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.large)
                            .frame(maxWidth: .infinity)
                        } else {
                            Button("Download for Offline") {
                                meditationManager.downloadSession(session)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.large)
                            .frame(maxWidth: .infinity)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Session Details")
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

struct DetailRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
    }
}

struct MeditationSettingsView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @Environment(\.dismiss) private var dismiss
    @State private var settings: MeditationSettings
    
    init() {
        _settings = State(initialValue: MeditationManager.shared.settings)
    }
    
    var body: some View {
        NavigationView {
            Form {
                // Reminders Section
                Section("Reminders") {
                    Toggle("Enable Reminders", isOn: $settings.reminderEnabled)
                    
                    if settings.reminderEnabled {
                        ForEach(settings.reminderTimes.indices, id: \.self) { index in
                            DatePicker(
                                "Reminder \(index + 1)",
                                selection: $settings.reminderTimes[index],
                                displayedComponents: .hourAndMinute
                            )
                        }
                        .onDelete { indexSet in
                            settings.reminderTimes.remove(atOffsets: indexSet)
                        }
                        
                        Button("Add Reminder") {
                            settings.reminderTimes.append(Date())
                        }
                    }
                }
                
                // Preferences Section
                Section("Preferences") {
                    Picker("Preferred Duration", selection: $settings.preferredDuration) {
                        Text("5 minutes").tag(TimeInterval(300))
                        Text("10 minutes").tag(TimeInterval(600))
                        Text("15 minutes").tag(TimeInterval(900))
                        Text("20 minutes").tag(TimeInterval(1200))
                        Text("30 minutes").tag(TimeInterval(1800))
                    }
                    
                    Toggle("Background Sounds", isOn: $settings.backgroundSoundsEnabled)
                    Toggle("Voice Guidance", isOn: $settings.voiceGuidanceEnabled)
                    Toggle("Vibration", isOn: $settings.vibrationEnabled)
                }
                
                // Health Section
                Section("Health") {
                    Toggle("Track Heart Rate", isOn: $settings.trackHeartRate)
                    Toggle("Share Progress", isOn: $settings.shareProgress)
                }
                
                // Download Section
                Section("Downloads") {
                    Toggle("Offline Mode", isOn: $settings.offlineMode)
                    
                    Picker("Download Quality", selection: $settings.downloadQuality) {
                        ForEach(AudioQuality.allCases, id: \.self) { quality in
                            Text(quality.displayName).tag(quality)
                        }
                    }
                }
                
                // Interface Section
                Section("Interface") {
                    Picker("Theme", selection: $settings.interfaceTheme) {
                        ForEach(InterfaceTheme.allCases, id: \.self) { theme in
                            Text(theme.displayName).tag(theme)
                        }
                    }
                    
                    Toggle("Auto Start Next", isOn: $settings.autoStartNext)
                }
                
                // Accessibility Section
                Section("Accessibility") {
                    Toggle("Large Text", isOn: $settings.accessibilityOptions.largeText)
                    Toggle("High Contrast", isOn: $settings.accessibilityOptions.highContrast)
                    Toggle("Reduce Motion", isOn: $settings.accessibilityOptions.reduceMotion)
                    Toggle("Haptic Feedback", isOn: $settings.accessibilityOptions.hapticFeedback)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        meditationManager.updateSettings(settings)
                        dismiss()
                    }
                }
            }
        }
    }
}

struct InsightsView: View {
    @StateObject private var meditationManager = MeditationManager.shared
    @Environment(\.dismiss) private var dismiss
    @State private var selectedCategory: InsightCategory?
    
    var filteredInsights: [MeditationInsight] {
        if let category = selectedCategory {
            return meditationManager.insights.filter { $0.category == category }
        }
        return meditationManager.insights
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Category Filter
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        FilterChip(
                            title: "All",
                            isSelected: selectedCategory == nil
                        ) {
                            selectedCategory = nil
                        }
                        
                        ForEach(InsightCategory.allCases, id: \.self) { category in
                            FilterChip(
                                title: category.displayName,
                                isSelected: selectedCategory == category
                            ) {
                                selectedCategory = selectedCategory == category ? nil : category
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 8)
                
                Divider()
                
                // Insights List
                if filteredInsights.isEmpty {
                    VStack(spacing: 16) {
                        Spacer()
                        
                        Image(systemName: "lightbulb")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        
                        Text("No Insights Yet")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Text("Complete more meditation sessions to receive personalized insights about your practice.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        
                        Spacer()
                    }
                } else {
                    ScrollView {
                        LazyVStack(spacing: 16) {
                            ForEach(filteredInsights) { insight in
                                DetailedInsightCard(insight: insight)
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Insights")
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

struct DetailedInsightCard: View {
    let insight: MeditationInsight
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: insight.category.systemImage)
                    .font(.title2)
                    .foregroundColor(.accentColor)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(insight.title)
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text(insight.category.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Text(insight.period.displayName)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(.systemGray5))
                    .cornerRadius(8)
            }
            
            // Description
            Text(insight.description)
                .font(.body)
            
            // Recommendations
            if !insight.recommendations.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Recommendations")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    ForEach(insight.recommendations, id: \.self) { recommendation in
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "lightbulb.fill")
                                .font(.caption)
                                .foregroundColor(.yellow)
                                .padding(.top, 2)
                            
                            Text(recommendation)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.05), radius: 2, x: 0, y: 1)
    }
}

#Preview {
    MeditationView()
}