//
//  DashboardView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData
import Charts

// MARK: - TrendDataPoint
struct TrendDataPoint: Identifiable {
    let id = UUID()
    let date: Date
    let value: Double
    
    init(date: Date, value: Double) {
        self.date = date
        self.value = value
    }
}

struct DashboardView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @StateObject private var errorHandler = CoreDataErrorHandler.shared
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: false)],
        predicate: NSPredicate(format: "timestamp >= %@", Calendar.current.startOfDay(for: Date()) as CVarArg),
        animation: .default)
    private var todaysPainEntries: FetchedResults<PainEntry>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: false)],
        predicate: NSPredicate(format: "date >= %@", Calendar.current.startOfDay(for: Date()) as CVarArg),
        animation: .default)
    private var todaysJournalEntries: FetchedResults<JournalEntry>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \BASSDAIAssessment.date, ascending: false)],
        predicate: NSPredicate(format: "date >= %@", Calendar.current.startOfDay(for: Date()) as CVarArg),
        animation: .default)
    private var todaysAssessments: FetchedResults<BASSDAIAssessment>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \Medication.startDate, ascending: true)],
        predicate: NSPredicate(format: "reminderEnabled == YES"),
        animation: .default)
    private var activeMedications: FetchedResults<Medication>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: false)],
        animation: .default)
    private var painEntries: FetchedResults<PainEntry>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: false)],
        animation: .default)
    private var journalEntries: FetchedResults<JournalEntry>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \BASSDAIAssessment.date, ascending: false)],
        animation: .default)
    private var assessments: FetchedResults<BASSDAIAssessment>
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Welcome Section
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Welcome back!")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.primary)
                        
                        Text("Here's your health summary for today")
                            .font(.title3)
                            .foregroundColor(.secondary)
                            .fontWeight(.medium)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 20)
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Welcome back! Here's your health summary for today")
                    
                    // Today's Summary
                    TodaySummaryCard(
                        painEntries: todaysPainEntries.count,
                        journalEntries: todaysJournalEntries.count,
                        assessments: todaysAssessments.count,
                        activeMedications: activeMedications.count
                    )
                    
                    // Analytics Overview
                    AnalyticsOverviewCard(
                        painEntries: painEntries,
                        journalEntries: journalEntries,
                        assessments: assessments
                    )
                    
                    // Insights Preview
                    InsightsPreviewCard()
                    
                    // Quick Actions
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                        NavigationLink(destination: PainTrackingView()) {
                            DashboardCard(
                                title: "Pain Tracking",
                                subtitle: "Log your pain",
                                icon: "figure.walk",
                                color: .red
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        NavigationLink(destination: MedicationView()) {
                            DashboardCard(
                                title: "Medications",
                                subtitle: "Manage meds",
                                icon: "pills.fill",
                                color: .blue
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        NavigationLink(destination: JournalView()) {
                            DashboardCard(
                                title: "Journal",
                                subtitle: "Write entry",
                                icon: "book.fill",
                                color: .green
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        NavigationLink(destination: BASSDAIView()) {
                            DashboardCard(
                                title: "BASDAI",
                                subtitle: "Assessment",
                                icon: "chart.bar.fill",
                                color: .orange
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                    .padding(.horizontal)
                    
                    // Intelligent Features
                    IntelligentFeaturesSection()
                    
                    // Recent Activity
                    RecentActivitySection(
                        painEntries: Array(painEntries.prefix(3)),
                        journalEntries: Array(journalEntries.prefix(3)),
                        assessments: Array(assessments.prefix(2))
                    )
                }
                .padding(.vertical)
            }
            .navigationTitle("Dashboard")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

struct TodaySummaryCard: View {
    let painEntries: Int
    let journalEntries: Int
    let assessments: Int
    let activeMedications: Int
    
    private func painLevelColor(_ level: Double) -> Color {
        switch level {
        case 0..<3: return .green
        case 3..<6: return .orange
        default: return .red
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Today's Summary")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Text(DateFormatter.dayFormatter.string(from: Date()))
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fontWeight(.medium)
            }
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                SummaryItem(value: painEntries, title: "Pain Entries", color: .red)
                SummaryItem(value: journalEntries, title: "Journal Entries", color: .blue)
                SummaryItem(value: assessments, title: "Assessments", color: .green)
                SummaryItem(value: activeMedications, title: "Active Medications", color: .orange)
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 8, x: 0, y: 4)
        )
        .padding(.horizontal, 20)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Today's health summary")
    }
}

struct RecentActivitySection: View {
    let painEntries: [PainEntry]
    let journalEntries: [JournalEntry]
    let assessments: [BASSDAIAssessment]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Recent Activity")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
                .padding(.horizontal, 20)
            
            if painEntries.isEmpty && journalEntries.isEmpty && assessments.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "clock")
                        .font(.system(size: 40, weight: .medium))
                        .foregroundColor(.secondary)
                    Text("No recent activity")
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(.secondary)
                    Text("Start tracking your health journey")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .fontWeight(.medium)
                }
                .frame(maxWidth: .infinity)
                .padding(32)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color(.systemGray6))
                )
                .padding(.horizontal, 20)
            } else {
                VStack(spacing: 16) {
                    ForEach(painEntries, id: \.objectID) { entry in
                        RecentActivityRow(
                            icon: "figure.walk",
                            title: "Pain Entry",
                            subtitle: "Level \(Int(entry.painLevel)) - \(entry.location ?? "Unknown")",
                            date: entry.timestamp ?? Date(),
                            color: .red
                        )
                    }
                    
                    ForEach(journalEntries, id: \.objectID) { entry in
                        RecentActivityRow(
                            icon: "book.fill",
                            title: "Journal Entry",
                            subtitle: String(entry.content?.prefix(50) ?? "No content"),
                            date: entry.date ?? Date(),
                            color: .green
                        )
                    }
                    
                    ForEach(assessments, id: \.objectID) { assessment in
                        RecentActivityRow(
                            icon: "chart.bar.fill",
                            title: "BASDAI Assessment",
                            subtitle: "Score: \(String(format: "%.1f", assessment.totalScore))",
                            date: assessment.date ?? Date(),
                            color: .orange
                        )
                    }
                }
                .padding(.horizontal, 20)
            }
        }
    }
}

struct RecentActivityRow: View {
    let icon: String
    let title: String
    let subtitle: String
    let date: Date
    let color: Color
    
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 22, weight: .medium))
                .foregroundColor(color)
                .frame(width: 40, height: 40)
                .background(
                    Circle()
                        .fill(color.opacity(0.12))
                )
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fontWeight(.medium)
                    .lineLimit(2)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 2) {
                Text(date, style: .time)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                Text(date, style: .date)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.05), radius: 4, x: 0, y: 2)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title): \(subtitle) at \(date.formatted(date: .omitted, time: .shortened))")
    }
}

struct DashboardCard: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 36, weight: .medium))
                .foregroundColor(color)
                .frame(width: 44, height: 44)
            
            VStack(spacing: 6) {
                Text(title)
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                    .multilineTextAlignment(.center)
                
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fontWeight(.medium)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 140)
        .padding(24)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 8, x: 0, y: 4)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title): \(subtitle)")
        .accessibilityHint("Tap to open \(title.lowercased())")
    }
}

// MARK: - Extensions

extension DateFormatter {
    static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEEE, MMM d"
        return formatter
    }()
    
    static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter
    }()
}

struct SummaryItem: View {
    let value: Int
    let title: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text("\(value)")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Analytics Overview Card

struct AnalyticsOverviewCard: View {
    let painEntries: FetchedResults<PainEntry>
    let journalEntries: FetchedResults<JournalEntry>
    let assessments: FetchedResults<BASSDAIAssessment>
    
    private var weeklyPainTrend: [TrendDataPoint] {
        let calendar = Calendar.current
        let weekAgo = calendar.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        
        let recentPain = painEntries.filter { entry in
            guard let timestamp = entry.timestamp else { return false }
            return timestamp >= weekAgo
        }
        
        return recentPain.compactMap { entry in
            guard let timestamp = entry.timestamp else { return nil }
            return TrendDataPoint(date: timestamp, value: Double(entry.painLevel))
        }.sorted { $0.date < $1.date }
    }
    
    private var averagePainLevel: Double {
        let recentPain = Array(painEntries.prefix(7))
        guard !recentPain.isEmpty else { return 0 }
        return recentPain.reduce(0) { $0 + Double($1.painLevel) } / Double(recentPain.count)
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Text("This Week")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                
                Spacer()
                
                HStack {
                    Text("View All")
                    Image(systemName: "chevron.right")
                }
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.blue)
            }
            
            VStack(spacing: 20) {
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Average Pain Level")
                            .font(.headline)
                            .fontWeight(.medium)
                            .foregroundColor(.secondary)
                        Text("\(String(format: "%.1f", averagePainLevel))")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.primary)
                    }
                    
                    Spacer()
                    
                    if !weeklyPainTrend.isEmpty {
                        Chart(weeklyPainTrend) { dataPoint in
                            LineMark(
                                x: .value("Date", dataPoint.date),
                                y: .value("Pain", dataPoint.value)
                            )
                            .foregroundStyle(.red)
                            .lineStyle(StrokeStyle(lineWidth: 2))
                        }
                        .frame(height: 100)
                        .chartXAxis(.hidden)
                        .chartYAxis(.hidden)
                        .chartYScale(domain: 0...10)
                    }
                }
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.08), radius: 8, x: 0, y: 4)
        )
        .padding(.horizontal, 20)
    }
    
    private func painLevelColor(_ level: Double) -> Color {
        switch level {
        case 0..<3: return .green
        case 3..<6: return .orange
        default: return .red
        }
    }
}

// MARK: - Insights Preview Card

struct InsightsPreviewCard: View {
    @Environment(\.managedObjectContext) private var viewContext
    @State private var todaysRecommendation: String = "Loading..."
    @State private var insightCount: Int = 0
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Text("Daily Insights")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Button("View All") {
                    // Navigate to insights
                }
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.blue)
            }
            
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 12) {
                    Image(systemName: "lightbulb.fill")
                        .font(.system(size: 20, weight: .medium))
                        .foregroundColor(.yellow)
                        .frame(width: 24, height: 24)
                    
                    Text("Daily Suggestion")
                        .font(.headline)
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                }
                
                Text("Consider gentle stretching exercises to help reduce morning stiffness.")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                    .lineLimit(3)
                    .padding(.leading, 36)
                
                Divider()
                    .padding(.vertical, 4)
                
                HStack(spacing: 12) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 20, weight: .medium))
                        .foregroundColor(.green)
                        .frame(width: 24, height: 24)
                    
                    Text("New Insight Available")
                        .font(.headline)
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                }
                
                Text("Your pain levels tend to be lower on days with moderate activity.")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                    .lineLimit(3)
                    .padding(.leading, 36)
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.08), radius: 8, x: 0, y: 4)
        )
        .padding(.horizontal, 20)
        .onAppear {
            loadTodaysRecommendation()
            loadInsightCount()
        }
    }
    
    private func loadTodaysRecommendation() {
        // let engine = PACESuggestionEngine(context: viewContext)
        // let suggestion = engine.generateDailySuggestion()
        // todaysRecommendation = suggestion.primaryActivity.name
        todaysRecommendation = "Take a 10-minute walk"
    }
    
    private func loadInsightCount() {
        // Simulate insight count - in real app, this would query actual insights
        insightCount = Int.random(in: 2...8)
    }
}

// MARK: - Intelligent Features Section

struct IntelligentFeaturesSection: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Intelligent Features")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
                .padding(.horizontal, 20)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 16), count: 2), spacing: 16) {
                IntelligentFeatureCard(
                    title: "Analytics",
                    subtitle: "View detailed reports",
                    icon: "chart.bar.fill",
                    color: .blue
                )
                
                IntelligentFeatureCard(
                    title: "Insights",
                    subtitle: "Personalized recommendations",
                    icon: "lightbulb.fill",
                    color: .orange
                )
                
                IntelligentFeatureCard(
                    title: "Health Nudges",
                    subtitle: "Smart reminders",
                    icon: "bell.fill",
                    color: .green
                )
                
                IntelligentFeatureCard(
                    title: "PACE Guide",
                    subtitle: "Activity management",
                    icon: "figure.walk",
                    color: .purple
                )
            }
            .padding(.horizontal, 20)
        }
    }
}

struct IntelligentFeatureCard: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 32, weight: .medium))
                .foregroundColor(color)
                .frame(width: 40, height: 40)
            
            VStack(spacing: 6) {
                Text(title)
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                    .multilineTextAlignment(.center)
                
                Text(subtitle)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 120)
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.08), radius: 6, x: 0, y: 3)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title): \(subtitle)")
        .accessibilityHint("Tap to access \(title.lowercased())")
    }
}

struct DashboardView_Previews: PreviewProvider {
    static var previews: some View {
        DashboardView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
            .coreDataErrorAlert()
    }
}