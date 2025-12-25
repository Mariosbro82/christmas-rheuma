//
//  SkeletonViews.swift
//  InflamAI
//
//  Reusable skeleton loading views for consistent loading states
//  Use these instead of ProgressView for data-heavy screens
//
//  CRIT-005: Implement proper loading states across all data-dependent screens
//

import SwiftUI

// MARK: - Skeleton Modifier (Shimmer Animation)

struct SkeletonModifier: ViewModifier {
    @State private var isAnimating = false

    func body(content: Content) -> some View {
        content
            .overlay(
                GeometryReader { geo in
                    Rectangle()
                        .fill(
                            LinearGradient(
                                colors: [
                                    Color.clear,
                                    Color.white.opacity(0.4),
                                    Color.clear
                                ],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geo.size.width * 0.5)
                        .offset(x: isAnimating ? geo.size.width * 1.5 : -geo.size.width)
                }
                .mask(content)
            )
            .onAppear {
                withAnimation(
                    .linear(duration: 1.5)
                    .repeatForever(autoreverses: false)
                ) {
                    isAnimating = true
                }
            }
    }
}

extension View {
    func skeleton() -> some View {
        modifier(SkeletonModifier())
    }
}

// MARK: - Base Skeleton Shape

struct SkeletonShape: View {
    var height: CGFloat = 20
    var cornerRadius: CGFloat = Radii.xs

    var body: some View {
        RoundedRectangle(cornerRadius: cornerRadius)
            .fill(Colors.Gray.g200)
            .frame(height: height)
            .skeleton()
    }
}

// MARK: - Skeleton Card View

/// Skeleton placeholder for card layouts
struct SkeletonCardView: View {
    var showImage: Bool = true
    var lineCount: Int = 3

    var body: some View {
        HStack(spacing: Spacing.sm) {
            if showImage {
                RoundedRectangle(cornerRadius: Radii.lg)
                    .fill(Colors.Gray.g200)
                    .frame(width: 80, height: 80)
                    .skeleton()
            }

            VStack(alignment: .leading, spacing: Spacing.xs) {
                // Title
                SkeletonShape(height: 18)
                    .frame(width: 150)

                // Subtitle lines
                ForEach(0..<lineCount, id: \.self) { index in
                    SkeletonShape(height: 14)
                        .frame(width: index == lineCount - 1 ? 100 : nil)
                }
            }

            Spacer()
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.lg)
        .dshadow(Shadows.sm)
    }
}

// MARK: - Skeleton List Row

/// Skeleton placeholder for list items
struct SkeletonListRow: View {
    var showIcon: Bool = true

    var body: some View {
        HStack(spacing: Spacing.sm) {
            if showIcon {
                Circle()
                    .fill(Colors.Gray.g200)
                    .frame(width: 44, height: 44)
                    .skeleton()
            }

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                SkeletonShape(height: 16)
                    .frame(width: 120)

                SkeletonShape(height: 12)
                    .frame(width: 80)
            }

            Spacer()

            SkeletonShape(height: 12)
                .frame(width: 40)
        }
        .padding(.vertical, Spacing.xs)
    }
}

// MARK: - Exercise Card Skeleton

/// Skeleton specifically for exercise library cards
struct ExerciseCardSkeleton: View {
    var body: some View {
        HStack(spacing: Spacing.sm) {
            // Thumbnail placeholder
            RoundedRectangle(cornerRadius: Radii.lg)
                .fill(Colors.Gray.g200)
                .frame(width: 80, height: 80)
                .skeleton()

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                // Exercise name
                SkeletonShape(height: 16)
                    .frame(width: 140)

                // Difficulty & duration
                HStack(spacing: Spacing.xs) {
                    SkeletonShape(height: 14)
                        .frame(width: 60)

                    SkeletonShape(height: 14)
                        .frame(width: 50)
                }

                // Target areas
                SkeletonShape(height: 12)
                    .frame(width: 100)
            }

            Spacer()
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.lg)
        .dshadow(Shadows.sm)
    }
}

// MARK: - Medication Card Skeleton

/// Skeleton specifically for medication cards
struct MedicationCardSkeleton: View {
    var body: some View {
        HStack(spacing: Spacing.sm) {
            // Pill icon placeholder
            Circle()
                .fill(Colors.Gray.g200)
                .frame(width: 48, height: 48)
                .skeleton()

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                // Medication name
                SkeletonShape(height: 16)
                    .frame(width: 120)

                // Dosage
                SkeletonShape(height: 12)
                    .frame(width: 80)
            }

            Spacer()

            // Time/status
            SkeletonShape(height: 28)
                .frame(width: 70)
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.lg)
    }
}

// MARK: - Chart Skeleton

/// Skeleton placeholder for charts
struct ChartSkeleton: View {
    var height: CGFloat = 200

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            // Chart bars
            HStack(alignment: .bottom, spacing: Spacing.xs) {
                ForEach(0..<12, id: \.self) { index in
                    RoundedRectangle(cornerRadius: Radii.xs)
                        .fill(Colors.Gray.g200)
                        .frame(height: randomHeight(for: index))
                        .skeleton()
                }
            }
            .frame(height: height)

            // X-axis labels
            HStack {
                ForEach(0..<4, id: \.self) { _ in
                    SkeletonShape(height: 10)
                        .frame(width: 30)
                    Spacer()
                }
            }
        }
    }

    private func randomHeight(for index: Int) -> CGFloat {
        // Pseudo-random heights based on index for consistent appearance
        let heights: [CGFloat] = [0.4, 0.6, 0.8, 0.5, 0.7, 0.9, 0.6, 0.4, 0.7, 0.8, 0.5, 0.6]
        return height * heights[index % heights.count]
    }
}

// MARK: - Flare Timeline Skeleton

/// Skeleton for flare timeline entries
struct FlareTimelineSkeleton: View {
    var body: some View {
        VStack(spacing: Spacing.md) {
            ForEach(0..<3, id: \.self) { _ in
                HStack(alignment: .top, spacing: Spacing.sm) {
                    // Date indicator
                    VStack(spacing: Spacing.xxs) {
                        Circle()
                            .fill(Colors.Gray.g200)
                            .frame(width: 12, height: 12)
                            .skeleton()

                        Rectangle()
                            .fill(Colors.Gray.g200)
                            .frame(width: 2, height: 60)
                    }

                    // Flare card
                    VStack(alignment: .leading, spacing: Spacing.xs) {
                        SkeletonShape(height: 18)
                            .frame(width: 100)

                        SkeletonShape(height: 14)

                        HStack {
                            SkeletonShape(height: 24)
                                .frame(width: 60)
                            SkeletonShape(height: 24)
                                .frame(width: 60)
                        }
                    }
                    .padding(Spacing.sm)
                    .background(Colors.Gray.g100)
                    .cornerRadius(Radii.lg)
                }
            }
        }
    }
}

// MARK: - Loading State Wrapper

/// Wraps content with loading/empty/error state handling
struct LoadingStateView<Content: View, EmptyContent: View>: View {
    let isLoading: Bool
    let isEmpty: Bool
    let errorMessage: String?
    let retryAction: (() -> Void)?
    let loadingView: AnyView
    let emptyView: EmptyContent
    let content: Content

    init(
        isLoading: Bool,
        isEmpty: Bool,
        errorMessage: String? = nil,
        retryAction: (() -> Void)? = nil,
        loadingView: some View = ProgressView(),
        @ViewBuilder emptyView: () -> EmptyContent,
        @ViewBuilder content: () -> Content
    ) {
        self.isLoading = isLoading
        self.isEmpty = isEmpty
        self.errorMessage = errorMessage
        self.retryAction = retryAction
        self.loadingView = AnyView(loadingView)
        self.emptyView = emptyView()
        self.content = content()
    }

    var body: some View {
        Group {
            if isLoading {
                loadingView
            } else if let error = errorMessage {
                ErrorStateView(
                    title: "Something went wrong",
                    message: error,
                    retryAction: retryAction
                )
            } else if isEmpty {
                emptyView
            } else {
                content
            }
        }
    }
}

// MARK: - Previews

#if DEBUG
struct SkeletonViews_Previews: PreviewProvider {
    static var previews: some View {
        ScrollView {
            VStack(spacing: Spacing.lg) {
                Text("Skeleton Components")
                    .font(.headline)

                Group {
                    Text("Card Skeleton").font(.caption)
                    SkeletonCardView()
                }

                Group {
                    Text("Exercise Card Skeleton").font(.caption)
                    ExerciseCardSkeleton()
                }

                Group {
                    Text("Medication Card Skeleton").font(.caption)
                    MedicationCardSkeleton()
                }

                Group {
                    Text("List Row Skeleton").font(.caption)
                    VStack(spacing: 0) {
                        SkeletonListRow()
                        Divider()
                        SkeletonListRow()
                        Divider()
                        SkeletonListRow()
                    }
                }

                Group {
                    Text("Chart Skeleton").font(.caption)
                    ChartSkeleton()
                }

                Group {
                    Text("Flare Timeline Skeleton").font(.caption)
                    FlareTimelineSkeleton()
                }
            }
            .padding()
        }
        .background(Colors.Gray.g50)
    }
}
#endif
