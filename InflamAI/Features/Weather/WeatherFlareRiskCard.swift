//
//  WeatherFlareRiskCard.swift
//  InflamAI
//
//  Main weather dashboard card with current conditions and flare risk
//  Uses Apple WeatherKit for real weather data
//  Section 9.1: Skeleton loading state, friendly error state
//

import SwiftUI
import CoreLocation

struct WeatherFlareRiskCard: View {
    @StateObject private var viewModel = WeatherFlareRiskViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            // Header with risk indicator
            headerSection

            if viewModel.isLoading {
                // Section 9.1: Skeleton loading state instead of spinner
                weatherSkeletonView
            } else if let error = viewModel.errorMessage {
                // Section 9.1: Friendly error state with retry
                WeatherErrorView(message: error) {
                    Task { await viewModel.refresh() }
                }
            } else {
                // Current weather conditions
                currentWeatherSection

                Divider()

                // Pressure trend chart
                if !viewModel.pressureForecast.isEmpty {
                    VStack(alignment: .leading, spacing: Spacing.xs) {
                        Text("48-Hour Pressure Trend")
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)

                        PressureTrendChart(dataPoints: viewModel.pressureForecast)
                            .frame(height: 120)
                    }
                }

                // Active weather alerts
                if !viewModel.weatherAlerts.isEmpty {
                    alertsSection
                }

                // Recommendations
                recommendationSection
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
        .task {
            await viewModel.loadWeatherData()
        }
    }

    // MARK: - Header Section

    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text("Weather & Symptom Patterns")
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                Text("Next 48 hours")
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)
            }

            Spacer()

            // Risk badge
            FlareRiskBadge(level: viewModel.riskLevel)
        }
    }

    // MARK: - Skeleton Loading View (Section 9.1)

    private var weatherSkeletonView: some View {
        VStack(spacing: Spacing.md) {
            // Weather display skeleton
            HStack(spacing: Spacing.md) {
                // Icon skeleton
                VStack(spacing: Spacing.xs) {
                    SkeletonView(width: 50, height: 50, cornerRadius: Radii.full)
                    SkeletonView(width: 60, height: 12, cornerRadius: Radii.sm)
                }
                .frame(width: 80)

                // Temperature skeleton
                VStack(alignment: .leading, spacing: Spacing.xs) {
                    SkeletonView(width: 80, height: 36, cornerRadius: Radii.sm)
                    SkeletonView(width: 100, height: 14, cornerRadius: Radii.sm)
                }

                Spacer()
            }

            // Metrics row skeleton
            HStack(spacing: Spacing.md) {
                ForEach(0..<3, id: \.self) { _ in
                    VStack(spacing: Spacing.xs) {
                        SkeletonView(width: 24, height: 24, cornerRadius: Radii.sm)
                        SkeletonView(width: 40, height: 16, cornerRadius: Radii.sm)
                        SkeletonView(width: 50, height: 10, cornerRadius: Radii.sm)
                    }
                    .frame(maxWidth: .infinity)
                }
            }

            // Chart skeleton
            SkeletonView(height: 80, cornerRadius: Radii.md)
        }
        .padding(.vertical, Spacing.sm)
    }

    // MARK: - Current Weather Section (Redesigned to match error state design language)

    private var currentWeatherSection: some View {
        VStack(spacing: Spacing.lg) {
            // Main weather display - centered, clean layout
            HStack(spacing: Spacing.xl) {
                // Condition icon in soft circle (matches error state design)
                ZStack {
                    Circle()
                        .fill(viewModel.currentCondition.color.opacity(0.15))
                        .frame(width: 72, height: 72)

                    Image(systemName: viewModel.currentCondition.iconName)
                        .font(.system(size: 32))
                        .foregroundColor(viewModel.currentCondition.color)
                }

                // Temperature & condition
                VStack(alignment: .leading, spacing: Spacing.xs) {
                    Text(viewModel.currentTemperature)
                        .font(.system(size: 40, weight: .semibold, design: .rounded))
                        .foregroundColor(Colors.Gray.g900)

                    Text(viewModel.currentCondition.displayName)
                        .font(.system(size: Typography.sm, weight: .medium))
                        .foregroundColor(Colors.Gray.g600)

                    if let weather = viewModel.currentWeather {
                        Text("Feels like \(String(format: "%.0f°C", weather.feelsLike))")
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }

                Spacer()
            }

            // Weather metrics - clean pill-style cards
            HStack(spacing: Spacing.sm) {
                WeatherMetricPill(
                    icon: "humidity.fill",
                    value: viewModel.currentHumidity,
                    label: "Humidity",
                    tint: Colors.Primary.p500
                )

                WeatherMetricPill(
                    icon: "gauge.with.dots.needle.bottom.50percent",
                    value: viewModel.currentPressure,
                    label: "Pressure",
                    tint: Colors.Primary.p500
                )

                WeatherMetricPill(
                    icon: viewModel.pressureChange12h >= 0 ? "arrow.up.circle.fill" : "arrow.down.circle.fill",
                    value: String(format: "%+.1f", viewModel.pressureChange12h),
                    label: "12h Change",
                    tint: pressureChangeColor(viewModel.pressureChange12h)
                )
            }
        }
    }

    // MARK: - Alerts Section

    private var alertsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            Text("Weather Alerts")
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g500)

            ForEach(viewModel.weatherAlerts.prefix(2)) { alert in
                WeatherAlertRow(alert: alert)
            }
        }
    }

    // MARK: - Recommendation Section

    private var recommendationSection: some View {
        RecommendationBanner(
            icon: "lightbulb.fill",
            text: viewModel.riskRecommendation,
            color: viewModel.riskLevel.color
        )
    }

    // MARK: - Helpers

    private func pressureChangeColor(_ change: Double) -> Color {
        if change < -10 { return Colors.Semantic.error }
        else if change < -5 { return Colors.Semantic.warning }
        else if change > 5 { return Colors.Primary.p500 }
        else { return Colors.Gray.g900 }
    }
}

// MARK: - Supporting Views

struct FlareRiskBadge: View {
    let level: FlareRiskLevel

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            Circle()
                .fill(level.color)
                .frame(width: 8, height: 8)

            Text(level.rawValue)
                .font(.system(size: Typography.xs, weight: .semibold))
        }
        .padding(.horizontal, Spacing.sm)
        .padding(.vertical, Spacing.xs)
        .background(level.color.opacity(0.2))
        .cornerRadius(Radii.lg)
    }
}

struct WeatherMetric: View {
    let icon: String
    let label: String
    let value: String
    var valueColor: Color = Colors.Gray.g900

    var body: some View {
        VStack(spacing: Spacing.xxs) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(Colors.Primary.p500)

            Text(value)
                .font(.system(size: Typography.sm, weight: .semibold))
                .foregroundColor(valueColor)

            Text(label)
                .font(.system(size: Typography.xxs))
                .foregroundColor(Colors.Gray.g500)
        }
        .frame(maxWidth: .infinity)
    }
}

/// Clean pill-style metric card matching app design language
struct WeatherMetricPill: View {
    let icon: String
    let value: String
    let label: String
    let tint: Color

    var body: some View {
        VStack(spacing: Spacing.xs) {
            HStack(spacing: Spacing.xxs) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(tint)

                Text(value)
                    .font(.system(size: Typography.sm, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)
            }

            Text(label)
                .font(.system(size: Typography.xxs))
                .foregroundColor(Colors.Gray.g500)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Spacing.sm)
        .padding(.horizontal, Spacing.xs)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.md)
    }
}

struct WeatherAlertRow: View {
    let alert: WeatherFlareAlert

    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(alert.severity.color)

            VStack(alignment: .leading, spacing: Spacing.xxxs) {
                Text(alert.message)
                    .font(.system(size: Typography.sm, weight: .medium))
                    .foregroundColor(Colors.Gray.g900)

                Text(alert.timeframe)
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)
            }

            Spacer()
        }
        .padding(Spacing.sm)
        .background(alert.severity.color.opacity(0.1))
        .cornerRadius(Radii.md)
    }
}

struct RecommendationBanner: View {
    let icon: String
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: icon)
                .foregroundColor(color)

            Text(text)
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g900)

            Spacer()
        }
        .padding(Spacing.sm)
        .background(color.opacity(0.15))
        .cornerRadius(Radii.md)
    }
}

/// Section 9.1: Friendly error state with retry
struct WeatherErrorView: View {
    let message: String
    let retryAction: () -> Void

    var body: some View {
        VStack(spacing: Spacing.md) {
            // Friendly illustration
            ZStack {
                Circle()
                    .fill(Colors.Semantic.warning.opacity(0.15))
                    .frame(width: 64, height: 64)

                Image(systemName: "cloud.sun")
                    .font(.system(size: 28))
                    .foregroundColor(Colors.Semantic.warning)
            }

            VStack(spacing: Spacing.xs) {
                Text("Weather Unavailable")
                    .font(.system(size: Typography.base, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                Text(message)
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)
                    .multilineTextAlignment(.center)
            }

            Button(action: retryAction) {
                HStack(spacing: Spacing.xs) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 12, weight: .semibold))
                    Text("Try Again")
                        .font(.system(size: Typography.sm, weight: .semibold))
                }
                .foregroundColor(.white)
                .padding(.horizontal, Spacing.lg)
                .padding(.vertical, Spacing.sm)
                .background(Colors.Primary.p500)
                .cornerRadius(Radii.full)
            }
        }
        .padding(Spacing.lg)
        .frame(maxWidth: .infinity)
    }
}

// Keep the old ErrorView for backward compatibility
struct ErrorView: View {
    let message: String
    let retryAction: () -> Void

    var body: some View {
        WeatherErrorView(message: message, retryAction: retryAction)
    }
}

// MARK: - Current Weather Card (Standalone)

struct CurrentWeatherCard: View {
    @ObservedObject var weatherService = OpenMeteoService.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Current Weather")
                .font(.headline)

            if let weather = weatherService.currentWeather {
                HStack(spacing: 16) {
                    // Condition
                    VStack {
                        Image(systemName: weather.condition.iconName)
                            .font(.system(size: 36))
                            .foregroundColor(weather.condition.color)

                        Text(weather.condition.displayName)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    // Temperature
                    VStack(alignment: .leading) {
                        Text(String(format: "%.0f°C", weather.temperature))
                            .font(.system(size: 32, weight: .light))

                        Text("Feels like \(String(format: "%.0f°C", weather.feelsLike))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    // Details
                    VStack(alignment: .trailing, spacing: 4) {
                        HStack {
                            Image(systemName: "humidity.fill")
                                .foregroundColor(.blue)
                            Text("\(weather.humidity)%")
                        }
                        .font(.caption)

                        HStack {
                            Image(systemName: "barometer")
                                .foregroundColor(.blue)
                            Text(String(format: "%.0f mmHg", weather.pressure))
                        }
                        .font(.caption)

                        HStack {
                            Image(systemName: "wind")
                                .foregroundColor(.blue)
                            Text(String(format: "%.0f km/h %@", weather.windSpeed, weather.windDirection))
                        }
                        .font(.caption)
                    }
                }
            } else if weatherService.isLoading {
                ProgressView()
                    .frame(maxWidth: .infinity)
            } else {
                Text("Weather data unavailable")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 3)
    }
}

// MARK: - Preview

struct WeatherFlareRiskCard_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 16) {
            WeatherFlareRiskCard()
            CurrentWeatherCard()
        }
        .padding()
        .background(Color(.systemGroupedBackground))
    }
}
