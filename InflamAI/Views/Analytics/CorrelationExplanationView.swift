//
//  CorrelationExplanationView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI

struct CorrelationExplanationView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var themeManager: ThemeManager
    @State private var selectedTab = 0
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // What is Correlation?
                whatIsCorrelationView
                    .tag(0)
                
                // How to Interpret
                interpretationView
                    .tag(1)
                
                // Statistical Significance
                significanceView
                    .tag(2)
                
                // Limitations
                limitationsView
                    .tag(3)
            }
            .tabViewStyle(PageTabViewStyle(indexDisplayMode: .always))
            .navigationTitle("Understanding Correlations")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .foregroundColor(themeManager.colors.primary)
                }
            }
            .themedBackground()
        }
    }
    
    private var whatIsCorrelationView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .foregroundColor(themeManager.colors.primary)
                            .font(.largeTitle)
                        
                        Text("What is Correlation?")
                            .font(themeManager.typography.title1)
                            .fontWeight(.bold)
                            .foregroundColor(themeManager.colors.textPrimary)
                    }
                    
                    Text("Correlation measures the strength and direction of a relationship between two variables.")
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.textSecondary)
                }
                
                // Visual examples
                VStack(alignment: .leading, spacing: 16) {
                    Text("Visual Examples")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    VStack(spacing: 16) {
                        CorrelationExample(
                            title: "Positive Correlation (+0.8)",
                            description: "As one variable increases, the other tends to increase",
                            color: .green,
                            pattern: .positive
                        )
                        
                        CorrelationExample(
                            title: "Negative Correlation (-0.8)",
                            description: "As one variable increases, the other tends to decrease",
                            color: .red,
                            pattern: .negative
                        )
                        
                        CorrelationExample(
                            title: "No Correlation (0.0)",
                            description: "No clear relationship between the variables",
                            color: .gray,
                            pattern: .none
                        )
                    }
                }
                
                // Key points
                VStack(alignment: .leading, spacing: 12) {
                    Text("Key Points")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        KeyPoint(text: "Correlation values range from -1 to +1")
                        KeyPoint(text: "Closer to ±1 means stronger relationship")
                        KeyPoint(text: "Closer to 0 means weaker relationship")
                        KeyPoint(text: "Sign indicates direction (+ or -)")
                    }
                }
            }
            .padding()
        }
    }
    
    private var interpretationView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(themeManager.colors.primary)
                            .font(.largeTitle)
                        
                        Text("How to Interpret")
                            .font(themeManager.typography.title1)
                            .fontWeight(.bold)
                            .foregroundColor(themeManager.colors.textPrimary)
                    }
                    
                    Text("Understanding what correlation values mean for your health data.")
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.textSecondary)
                }
                
                // Strength guide
                VStack(alignment: .leading, spacing: 16) {
                    Text("Correlation Strength Guide")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    VStack(spacing: 12) {
                        StrengthGuide(range: "0.0 - 0.3", description: "Weak", color: .gray, example: "Sleep and mood might have a weak connection")
                        StrengthGuide(range: "0.3 - 0.5", description: "Moderate", color: .orange, example: "Exercise and energy levels show moderate relationship")
                        StrengthGuide(range: "0.5 - 0.7", description: "Strong", color: .yellow, example: "Stress and pain levels are strongly related")
                        StrengthGuide(range: "0.7 - 0.9", description: "Very Strong", color: .green, example: "Medication timing and symptom relief")
                        StrengthGuide(range: "0.9 - 1.0", description: "Extremely Strong", color: .blue, example: "Temperature and joint stiffness (rare)")
                    }
                }
                
                // Health context
                VStack(alignment: .leading, spacing: 12) {
                    Text("In Health Context")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        HealthExample(icon: "moon.fill", title: "Sleep Quality vs Pain", description: "Strong negative correlation (-0.7) suggests better sleep leads to less pain")
                        HealthExample(icon: "figure.walk", title: "Exercise vs Stiffness", description: "Moderate negative correlation (-0.4) indicates exercise may reduce stiffness")
                        HealthExample(icon: "cloud.rain.fill", title: "Weather vs Symptoms", description: "Weak positive correlation (0.2) shows minimal weather impact")
                    }
                }
            }
            .padding()
        }
    }
    
    private var significanceView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "checkmark.seal.fill")
                            .foregroundColor(themeManager.colors.primary)
                            .font(.largeTitle)
                        
                        Text("Statistical Significance")
                            .font(themeManager.typography.title1)
                            .fontWeight(.bold)
                            .foregroundColor(themeManager.colors.textPrimary)
                    }
                    
                    Text("Understanding p-values and confidence in your correlations.")
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.textSecondary)
                }
                
                // P-value explanation
                VStack(alignment: .leading, spacing: 16) {
                    Text("P-Value Explained")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    VStack(alignment: .leading, spacing: 12) {
                        Text("The p-value tells us how confident we can be that the correlation is real and not due to chance.")
                            .font(themeManager.typography.body)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        PValueGuide(threshold: "< 0.001", description: "Extremely significant", color: .green, confidence: "99.9%")
                        PValueGuide(threshold: "< 0.01", description: "Very significant", color: .blue, confidence: "99%")
                        PValueGuide(threshold: "< 0.05", description: "Significant", color: .orange, confidence: "95%")
                        PValueGuide(threshold: "> 0.05", description: "Not significant", color: .red, confidence: "< 95%")
                    }
                }
                
                // Sample size importance
                VStack(alignment: .leading, spacing: 12) {
                    Text("Sample Size Matters")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        SampleSizePoint(icon: "calendar", text: "More days of data = more reliable results")
                        SampleSizePoint(icon: "chart.bar.fill", text: "At least 30 data points recommended")
                        SampleSizePoint(icon: "exclamationmark.triangle", text: "Small samples can show false correlations")
                        SampleSizePoint(icon: "checkmark.circle", text: "Larger samples give more confidence")
                    }
                }
                
                // Confidence intervals
                VStack(alignment: .leading, spacing: 12) {
                    Text("Confidence Intervals")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    Text("The range where we expect the true correlation to lie. Narrower intervals indicate more precise estimates.")
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.textSecondary)
                    
                    HStack {
                        Text("Example: [0.65, 0.85]")
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(themeManager.colors.primary)
                        
                        Text("means we're 95% confident the true correlation is between 0.65 and 0.85")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                    }
                }
            }
            .padding()
        }
    }
    
    private var limitationsView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                            .font(.largeTitle)
                        
                        Text("Important Limitations")
                            .font(themeManager.typography.title1)
                            .fontWeight(.bold)
                            .foregroundColor(themeManager.colors.textPrimary)
                    }
                    
                    Text("Understanding what correlations can and cannot tell us.")
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.textSecondary)
                }
                
                // Causation warning
                VStack(alignment: .leading, spacing: 16) {
                    Text("Correlation ≠ Causation")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.red)
                    
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Just because two things are correlated doesn't mean one causes the other.")
                            .font(themeManager.typography.body)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        CausationExample(
                            correlation: "Ice cream sales and drowning deaths are correlated",
                            reality: "Both increase in summer due to hot weather",
                            lesson: "A third factor (temperature) affects both"
                        )
                        
                        CausationExample(
                            correlation: "Pain levels and stress are correlated",
                            reality: "Pain might cause stress, stress might worsen pain, or both might be caused by inflammation",
                            lesson: "The relationship could be bidirectional or have a common cause"
                        )
                    }
                }
                
                // Other limitations
                VStack(alignment: .leading, spacing: 16) {
                    Text("Other Important Considerations")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    VStack(spacing: 12) {
                        LimitationCard(
                            icon: "arrow.up.and.down",
                            title: "Non-linear Relationships",
                            description: "Correlation only measures linear relationships. Some health factors might have curved or complex relationships."
                        )
                        
                        LimitationCard(
                            icon: "person.3.fill",
                            title: "Individual Variation",
                            description: "Population correlations might not apply to you personally. Your body is unique."
                        )
                        
                        LimitationCard(
                            icon: "clock.fill",
                            title: "Time Delays",
                            description: "Some effects might be delayed. Today's stress might affect tomorrow's pain."
                        )
                        
                        LimitationCard(
                            icon: "questionmark.circle",
                            title: "Missing Variables",
                            description: "Important factors might not be tracked, leading to incomplete understanding."
                        )
                    }
                }
                
                // Best practices
                VStack(alignment: .leading, spacing: 12) {
                    Text("Best Practices")
                        .font(themeManager.typography.title2)
                        .fontWeight(.bold)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        BestPractice(text: "Use correlations as starting points for investigation")
                        BestPractice(text: "Discuss findings with your healthcare provider")
                        BestPractice(text: "Look for patterns over time, not single data points")
                        BestPractice(text: "Consider multiple factors together")
                        BestPractice(text: "Keep tracking consistently for better insights")
                    }
                }
            }
            .padding()
        }
    }
}

// MARK: - Supporting Views

struct CorrelationExample: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let description: String
    let color: Color
    let pattern: CorrelationPattern
    
    enum CorrelationPattern {
        case positive, negative, none
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(themeManager.typography.body)
                .fontWeight(.medium)
                .foregroundColor(color)
            
            Text(description)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
            
            // Simple visual representation
            HStack(spacing: 4) {
                ForEach(0..<10, id: \.self) { i in
                    Circle()
                        .fill(color.opacity(0.7))
                        .frame(width: 6, height: 6)
                        .offset(y: offsetForPattern(pattern, index: i))
                }
            }
            .frame(height: 20)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(color.opacity(0.1))
        )
    }
    
    private func offsetForPattern(_ pattern: CorrelationPattern, index: Int) -> CGFloat {
        switch pattern {
        case .positive:
            return CGFloat(-index) * 1.5
        case .negative:
            return CGFloat(index) * 1.5
        case .none:
            return CGFloat.random(in: -8...8)
        }
    }
}

struct KeyPoint: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let text: String
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
                .font(.caption)
            
            Text(text)
                .font(themeManager.typography.body)
                .foregroundColor(themeManager.colors.textSecondary)
        }
    }
}

struct StrengthGuide: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let range: String
    let description: String
    let color: Color
    let example: String
    
    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(range)
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    Text(description)
                        .font(themeManager.typography.body)
                        .fontWeight(.bold)
                        .foregroundColor(color)
                }
                
                Text(example)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .italic()
            }
            
            Spacer()
            
            Circle()
                .fill(color)
                .frame(width: 12, height: 12)
        }
        .padding(.vertical, 4)
    }
}

struct HealthExample: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(themeManager.colors.primary)
                .font(.title3)
                .frame(width: 24, height: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(themeManager.typography.body)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Text(description)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
            }
        }
        .padding(.vertical, 4)
    }
}

struct PValueGuide: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let threshold: String
    let description: String
    let color: Color
    let confidence: String
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("p \(threshold)")
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    Text(description)
                        .font(themeManager.typography.body)
                        .foregroundColor(color)
                }
                
                Text("Confidence: \(confidence)")
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
            }
            
            Spacer()
            
            Circle()
                .fill(color)
                .frame(width: 12, height: 12)
        }
        .padding(.vertical, 4)
    }
}

struct SampleSizePoint: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundColor(themeManager.colors.primary)
                .font(.caption)
                .frame(width: 16, height: 16)
            
            Text(text)
                .font(themeManager.typography.body)
                .foregroundColor(themeManager.colors.textSecondary)
        }
    }
}

struct CausationExample: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let correlation: String
    let reality: String
    let lesson: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "link")
                    .foregroundColor(.orange)
                Text("Correlation:")
                    .font(themeManager.typography.caption)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            Text(correlation)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
                .italic()
            
            HStack {
                Image(systemName: "lightbulb")
                    .foregroundColor(.blue)
                Text("Reality:")
                    .font(themeManager.typography.caption)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            Text(reality)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
            
            HStack {
                Image(systemName: "graduationcap")
                    .foregroundColor(.green)
                Text("Lesson:")
                    .font(themeManager.typography.caption)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            Text(lesson)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(themeManager.colors.background)
        )
    }
}

struct LimitationCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.orange)
                .font(.title3)
                .frame(width: 24, height: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(themeManager.typography.body)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Text(description)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .lineLimit(nil)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(.orange.opacity(0.1))
        )
    }
}

struct BestPractice: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let text: String
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "star.fill")
                .foregroundColor(.yellow)
                .font(.caption)
            
            Text(text)
                .font(themeManager.typography.body)
                .foregroundColor(themeManager.colors.textSecondary)
        }
    }
}

#Preview {
    CorrelationExplanationView()
        .environmentObject(ThemeManager())
}