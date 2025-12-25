//
//  CorrelationsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts

struct CorrelationsView: View {
    @Environment(\.dismiss) private var dismiss
    // Removed ThemeManager dependency
    @StateObject private var analyticsManager = AdvancedAnalyticsManager()
    @State private var selectedCorrelation: CorrelationData?
    @State private var selectedTimeRange: TimeRange = .month
    @State private var showingDetailSheet = false
    
    enum TimeRange: String, CaseIterable {
        case week = "Week"
        case month = "Month"
        case quarter = "3 Months"
        case year = "Year"
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Header with time range selector
                    headerSection
                    
                    // Correlation Matrix
                    correlationMatrix
                    
                    // Top Correlations
                    topCorrelations
                    
                    // Weather Correlations
                    weatherCorrelations
                    
                    // Activity Correlations
                    activityCorrelations
                    
                    // Medication Correlations
                    medicationCorrelations
                }
                .padding(.vertical)
            }
            .navigationTitle("Correlations")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .foregroundColor(.blue)
                }
            }
            .background(Color(.systemBackground))
            .sheet(isPresented: $showingDetailSheet) {
                if let correlation = selectedCorrelation {
                    CorrelationDetailView(correlation: correlation)
                        // Removed ThemeManager environment object
                }
            }
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(.blue)
                    .font(.title)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Correlation Analysis")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                    
                    Text("Discover patterns between symptoms and lifestyle factors")
                        .font(.body)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            // Time range picker
            Picker("Time Range", selection: $selectedTimeRange) {
                ForEach(TimeRange.allCases, id: \.self) { range in
                    Text(range.rawValue).tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .onChange(of: selectedTimeRange) { _ in
                analyticsManager.loadAnalytics(for: selectedTimeRange.rawValue)
            }
        }
        .padding()
        .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(.systemBackground))
                    .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
            )
        .padding(.horizontal)
    }
    
    private var correlationMatrix: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Correlation Matrix")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            ScrollView(.horizontal, showsIndicators: false) {
                VStack(spacing: 0) {
                    // Header row
                    HStack(spacing: 0) {
                        Text("")
                            .frame(width: 80, height: 40)
                        
                        ForEach(["Pain", "Sleep", "Stress", "Weather", "Activity"], id: \.self) { factor in
                            Text(factor)
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(.primary)
                                .frame(width: 60, height: 40)
                                .background(Color(.systemBackground))
                        }
                    }
                    
                    // Data rows
                    ForEach(["Pain", "Sleep", "Stress", "Weather", "Activity"], id: \.self) { rowFactor in
                        HStack(spacing: 0) {
                            Text(rowFactor)
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(.primary)
                                .frame(width: 80, height: 40)
                                .background(Color(.systemBackground))
                            
                            ForEach(["Pain", "Sleep", "Stress", "Weather", "Activity"], id: \.self) { colFactor in
                                let correlation = getCorrelationValue(row: rowFactor, col: colFactor)
                                
                                Text(String(format: "%.2f", correlation))
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .foregroundColor(.white)
                                    .frame(width: 60, height: 40)
                                    .background(correlationColor(correlation))
                            }
                        }
                    }
                }
            }
            
            // Legend
            HStack {
                Text("Strong Negative")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Rectangle()
                    .fill(LinearGradient(
                        colors: [.red, .orange, .yellow, .green, .blue],
                        startPoint: .leading,
                        endPoint: .trailing
                    ))
                    .frame(height: 8)
                    .cornerRadius(4)
                
                Text("Strong Positive")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var topCorrelations: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Strongest Correlations")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            LazyVStack(spacing: 12) {
                ForEach(analyticsManager.correlations.prefix(5), id: \.id) { correlation in
                    CorrelationRow(correlation: correlation)
                        .onTapGesture {
                            selectedCorrelation = correlation
                            showingDetailSheet = true
                        }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var weatherCorrelations: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "cloud.sun.fill")
                    .foregroundColor(.blue)
                    .font(.title2)
                
                Text("Weather Impact")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
            }
            
            LazyVStack(spacing: 12) {
                WeatherCorrelationRow(
                    factor: "Barometric Pressure",
                    correlation: -0.73,
                    description: "Lower pressure increases pain"
                )
                
                WeatherCorrelationRow(
                    factor: "Humidity",
                    correlation: 0.45,
                    description: "Higher humidity worsens stiffness"
                )
                
                WeatherCorrelationRow(
                    factor: "Temperature",
                    correlation: -0.32,
                    description: "Cold weather increases joint pain"
                )
                
                WeatherCorrelationRow(
                    factor: "Precipitation",
                    correlation: 0.28,
                    description: "Rainy days correlate with higher pain"
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var activityCorrelations: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "figure.walk")
                    .foregroundColor(.green)
                    .font(.title2)
                
                Text("Activity Impact")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
            }
            
            Chart {
                ForEach(getActivityCorrelations(), id: \.activity) { data in
                    BarMark(
                        x: .value("Correlation", data.correlation),
                        y: .value("Activity", data.activity)
                    )
                    .foregroundStyle(data.correlation > 0 ? Color.red : Color.green)
                    .cornerRadius(4)
                }
            }
            .frame(height: 200)
            .chartXScale(domain: -1...1)
            .chartXAxis {
                AxisMarks(position: .bottom) { value in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisValueLabel()
                }
            }
            
            Text("Positive values indicate activities that increase pain, negative values indicate activities that reduce pain.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var medicationCorrelations: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "pills.fill")
                    .foregroundColor(.purple)
                    .font(.title2)
                
                Text("Medication Effectiveness")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
            }
            
            LazyVStack(spacing: 12) {
                MedicationCorrelationRow(
                    medication: "Methotrexate",
                    effectiveness: 0.78,
                    timing: "Morning",
                    description: "Most effective when taken with breakfast"
                )
                
                MedicationCorrelationRow(
                    medication: "Prednisone",
                    effectiveness: 0.65,
                    timing: "Evening",
                    description: "Better pain control with evening dose"
                )
                
                MedicationCorrelationRow(
                    medication: "Ibuprofen",
                    effectiveness: 0.42,
                    timing: "As needed",
                    description: "Moderate effectiveness for breakthrough pain"
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    // Helper functions
    private func getCorrelationValue(row: String, col: String) -> Double {
        if row == col { return 1.0 }
        
        let correlations: [String: [String: Double]] = [
            "Pain": ["Sleep": -0.65, "Stress": 0.72, "Weather": 0.58, "Activity": -0.43],
            "Sleep": ["Pain": -0.65, "Stress": -0.54, "Weather": 0.23, "Activity": 0.38],
            "Stress": ["Pain": 0.72, "Sleep": -0.54, "Weather": 0.15, "Activity": -0.29],
            "Weather": ["Pain": 0.58, "Sleep": 0.23, "Stress": 0.15, "Activity": -0.12],
            "Activity": ["Pain": -0.43, "Sleep": 0.38, "Stress": -0.29, "Weather": -0.12]
        ]
        
        return correlations[row]?[col] ?? 0.0
    }
    
    private func correlationColor(_ value: Double) -> Color {
        let normalizedValue = (value + 1) / 2 // Normalize to 0-1
        
        if normalizedValue < 0.2 {
            return .red
        } else if normalizedValue < 0.4 {
            return .orange
        } else if normalizedValue < 0.6 {
            return .yellow
        } else if normalizedValue < 0.8 {
            return .green
        } else {
            return .blue
        }
    }
    
    private func getActivityCorrelations() -> [(activity: String, correlation: Double)] {
        return [
            ("High Intensity Exercise", 0.45),
            ("Prolonged Sitting", 0.32),
            ("Heavy Lifting", 0.28),
            ("Gentle Stretching", -0.52),
            ("Walking", -0.38),
            ("Swimming", -0.61),
            ("Yoga", -0.47),
            ("Meditation", -0.34)
        ]
    }
}

// MARK: - Supporting Views

struct WeatherCorrelationRow: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let factor: String
    let correlation: Double
    let description: String
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(factor)
                    .font(.body)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(String(format: "%.2f", correlation))
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(correlation > 0 ? .red : .green)
                
                Text(abs(correlation) > 0.5 ? "Strong" : "Moderate")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 8)
    }
}

struct MedicationCorrelationRow: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let medication: String
    let effectiveness: Double
    let timing: String
    let description: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(medication)
                    .font(.body)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Text("\(Int(effectiveness * 100))%")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(effectivenessColor(effectiveness))
            }
            
            HStack {
                Text("Best timing: \(timing)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
            }
            
            Text(description)
                .font(.caption)
                .foregroundColor(.secondary)
            
            // Effectiveness bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color(.systemGray5))
                        .frame(height: 4)
                        .cornerRadius(2)
                    
                    Rectangle()
                        .fill(effectivenessColor(effectiveness))
                        .frame(width: geometry.size.width * effectiveness, height: 4)
                        .cornerRadius(2)
                }
            }
            .frame(height: 4)
        }
        .padding(.vertical, 8)
    }
    
    private func effectivenessColor(_ value: Double) -> Color {
        switch value {
        case 0..<0.3: return .red
        case 0.3..<0.6: return .orange
        case 0.6..<0.8: return .yellow
        default: return .green
        }
    }
}

#Preview {
    CorrelationsView()
        // .environmentObject(ThemeManager())
}