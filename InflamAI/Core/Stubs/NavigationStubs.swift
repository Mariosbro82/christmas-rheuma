//
//  NavigationStubs.swift
//  InflamAI
//
//  Lightweight stubs for navigation destination views used by HomeView
//  The full implementations are excluded due to cascading dependencies
//

import SwiftUI
import CoreData

// MARK: - TrendsView (Stub)

struct TrendsView: View {
    let context: NSManagedObjectContext

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        self.context = context
    }

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "chart.xyaxis.line")
                .font(.system(size: 60))
                .foregroundColor(.blue)

            Text("Trends & Insights")
                .font(.title2)
                .fontWeight(.bold)

            Text("This feature is under development")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Text("Track your symptoms over time and discover patterns")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Trends")
    }
}

// MARK: - CoachCompositorView (Stub)

struct CoachCompositorView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "sparkles")
                .font(.system(size: 60))
                .foregroundColor(.purple)

            Text("Exercise Coach")
                .font(.title2)
                .fontWeight(.bold)

            Text("This feature is under development")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Text("Get personalized exercise recommendations based on your condition")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Exercise Coach")
    }
}

// MARK: - ExerciseLibraryView (Stub)

struct ExerciseLibraryView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "figure.flexibility")
                .font(.system(size: 60))
                .foregroundColor(.green)

            Text("Exercise Library")
                .font(.title2)
                .fontWeight(.bold)

            Text("This feature is under development")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Text("Browse 52 AS-specific exercises with video guides")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Exercise Library")
    }
}

// MARK: - FlareTimelineView (Stub)

struct FlareTimelineView: View {
    let context: NSManagedObjectContext

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        self.context = context
    }

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "flame.fill")
                .font(.system(size: 60))
                .foregroundColor(.orange)

            Text("Flare Timeline")
                .font(.title2)
                .fontWeight(.bold)

            Text("This feature is under development")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Text("Track and manage your flare episodes over time")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Flare Timeline")
    }
}

// MARK: - JournalEntryDetailView (Stub)

struct JournalEntryDetailView: View {
    let entry: JournalEntry

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let date = entry.date {
                    Text(date, style: .date)
                        .font(.headline)
                        .foregroundColor(.secondary)
                }

                if let notes = entry.notes, !notes.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Notes")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(notes)
                            .font(.body)
                    }
                }

                if let symptoms = entry.symptoms, !symptoms.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Symptoms")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(symptoms)
                            .font(.body)
                    }
                }

                if let mood = entry.mood, !mood.isEmpty {
                    HStack {
                        Text("Mood:")
                            .fontWeight(.medium)
                        Text(mood)
                    }
                    .padding(.top)
                }
            }
            .padding()
        }
        .navigationTitle("Journal Entry")
    }
}

// MARK: - LoadingView (Stub)

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.5)
            Text("Loading...")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

// MARK: - DetailSection (Stub)

struct DetailSection: View {
    let title: String
    let content: String
    let icon: String

    init(title: String, content: String, icon: String) {
        self.title = title
        self.content = content
        self.icon = icon
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                Text(title)
                    .font(.headline)
            }
            Text(content)
                .font(.body)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
    }
}

// MARK: - SecureStorage (Stub)

class SecureStorage {
    static let shared = SecureStorage()

    private init() {}

    func get(_ key: String) -> String? {
        return UserDefaults.standard.string(forKey: "secure_\(key)")
    }

    func set(_ value: String?, forKey key: String) {
        if let value = value {
            UserDefaults.standard.set(value, forKey: "secure_\(key)")
        } else {
            UserDefaults.standard.removeObject(forKey: "secure_\(key)")
        }
    }

    func delete(_ key: String) {
        UserDefaults.standard.removeObject(forKey: "secure_\(key)")
    }

    func deleteAll() {
        let keys = UserDefaults.standard.dictionaryRepresentation().keys.filter { $0.hasPrefix("secure_") }
        for key in keys {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }

    func migrateAllPHIFromUserDefaults() {
        // Stub: No-op for migration - in production this would move PHI to secure storage
    }
}

// MARK: - PainTrackingView (Stub)

struct PainTrackingView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "hand.raised.fill")
                .font(.system(size: 60))
                .foregroundColor(.red)

            Text("Pain Tracking")
                .font(.title2)
                .fontWeight(.bold)

            Text("This feature is under development")
                .font(.body)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Pain Tracking")
    }
}

// MARK: - JournalView (Stub)

struct JournalView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "book.fill")
                .font(.system(size: 60))
                .foregroundColor(.purple)

            Text("Journal")
                .font(.title2)
                .fontWeight(.bold)

            Text("This feature is under development")
                .font(.body)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Journal")
    }
}

// MARK: - BASSDAIView (Stub)

struct BASSDAIView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "list.bullet.clipboard.fill")
                .font(.system(size: 60))
                .foregroundColor(.blue)

            Text("BASDAI Assessment")
                .font(.title2)
                .fontWeight(.bold)

            Text("This feature is under development")
                .font(.body)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
        .navigationTitle("BASDAI")
    }
}

// MARK: - MoodBadge (Stub)

struct MoodBadge: View {
    let mood: String

    var body: some View {
        Text(mood)
            .font(.caption)
            .fontWeight(.medium)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(moodColor.opacity(0.2))
            .foregroundColor(moodColor)
            .cornerRadius(8)
    }

    private var moodColor: Color {
        switch mood.lowercased() {
        case "great", "good": return .green
        case "okay", "neutral": return .blue
        case "bad", "poor": return .orange
        case "terrible", "awful": return .red
        default: return .gray
        }
    }
}
