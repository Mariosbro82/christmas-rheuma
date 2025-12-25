//
//  OnboardingWithMascotView.swift
//  InflamAI-Swift
//
//  Example onboarding flow featuring the Ankylosaurus mascot
//

import SwiftUI

struct OnboardingWithMascotView: View {
    @State private var currentPage = 0
    @State private var showMascot = false
    @Environment(\.dismiss) private var dismiss

    let pages = [
        OnboardingPage(
            title: "Welcome to InflamAI!",
            description: "Hi! I'm Anky, your friendly companion on your health journey. Let me show you around!",
            expression: .waving,
            animation: .bouncing
        ),
        OnboardingPage(
            title: "Track Your Symptoms",
            description: "I'll help you log your pain levels, stiffness, and other symptoms easily every day.",
            expression: .encouraging,
            animation: .none
        ),
        OnboardingPage(
            title: "Discover Patterns",
            description: "Together, we'll find connections between weather, activities, and how you feel!",
            expression: .excited,
            animation: .waving
        ),
        OnboardingPage(
            title: "Stay on Track",
            description: "I'll remind you about medications and help you export reports for your doctor.",
            expression: .happy,
            animation: .bouncing
        )
    ]

    var body: some View {
        ZStack {
            // Gradient background
            LinearGradient(
                colors: [
                    Color(red: 0.95, green: 0.97, blue: 0.99),
                    Color(red: 0.88, green: 0.94, blue: 0.96)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                // Mascot area
                ZStack {
                    // Decorative circles
                    Circle()
                        .fill(Color(red: 0.4, green: 0.7, blue: 0.6).opacity(0.1))
                        .frame(width: 280, height: 280)
                        .blur(radius: 20)
                        .offset(x: showMascot ? 0 : -100, y: 0)

                    // The mascot
                    Group {
                        let page = pages[currentPage]
                        let mascot = AnkylosaurusMascot(expression: page.expression, size: 220)

                        switch page.animation {
                        case .bouncing:
                            mascot.bouncing()
                        case .waving:
                            mascot.waving()
                        case .none:
                            mascot
                        }
                    }
                    .scaleEffect(showMascot ? 1 : 0.5)
                    .opacity(showMascot ? 1 : 0)
                }
                .frame(height: 300)
                .padding(.top, 60)

                // Content area
                VStack(spacing: 20) {
                    TabView(selection: $currentPage) {
                        ForEach(0..<pages.count, id: \.self) { index in
                            VStack(spacing: 16) {
                                Text(pages[index].title)
                                    .font(.system(size: 28, weight: .bold))
                                    .multilineTextAlignment(.center)
                                    .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.4))

                                Text(pages[index].description)
                                    .font(.system(size: 17))
                                    .multilineTextAlignment(.center)
                                    .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                    .padding(.horizontal, 32)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .tag(index)
                        }
                    }
                    .tabViewStyle(.page(indexDisplayMode: .always))
                    .frame(height: 200)

                    // Page indicator dots (custom styled)
                    HStack(spacing: 8) {
                        ForEach(0..<pages.count, id: \.self) { index in
                            Circle()
                                .fill(currentPage == index ?
                                      Color(red: 0.4, green: 0.7, blue: 0.6) :
                                        Color(red: 0.4, green: 0.7, blue: 0.6).opacity(0.3))
                                .frame(width: 8, height: 8)
                                .scaleEffect(currentPage == index ? 1.2 : 1.0)
                                .animation(.spring(response: 0.3), value: currentPage)
                        }
                    }
                    .padding(.bottom, 20)

                    // Action buttons
                    VStack(spacing: 12) {
                        Button {
                            if currentPage < pages.count - 1 {
                                withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
                                    currentPage += 1
                                }
                            } else {
                                // Complete onboarding
                                dismiss()
                            }
                        } label: {
                            HStack {
                                Text(currentPage < pages.count - 1 ? "Next" : "Get Started")
                                    .font(.system(size: 18, weight: .semibold))
                                if currentPage < pages.count - 1 {
                                    Image(systemName: "arrow.right")
                                } else {
                                    Image(systemName: "checkmark")
                                }
                            }
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .frame(height: 56)
                            .background(
                                LinearGradient(
                                    colors: [
                                        Color(red: 0.4, green: 0.7, blue: 0.6),
                                        Color(red: 0.35, green: 0.65, blue: 0.55)
                                    ],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .cornerRadius(16)
                            .shadow(color: Color(red: 0.4, green: 0.7, blue: 0.6).opacity(0.3),
                                    radius: 12, y: 6)
                        }

                        if currentPage > 0 {
                            Button {
                                withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
                                    currentPage -= 1
                                }
                            } label: {
                                HStack {
                                    Image(systemName: "arrow.left")
                                    Text("Back")
                                        .font(.system(size: 16, weight: .medium))
                                }
                                .foregroundColor(Color(red: 0.4, green: 0.7, blue: 0.6))
                            }
                        }
                    }
                    .padding(.horizontal, 32)
                    .padding(.bottom, 40)
                }

                Spacer()
            }
        }
        .onAppear {
            withAnimation(.spring(response: 0.8, dampingFraction: 0.7).delay(0.2)) {
                showMascot = true
            }
        }
        .onChange(of: currentPage) { oldValue, newValue in
            // Re-animate mascot when page changes
            showMascot = false
            withAnimation(.spring(response: 0.6, dampingFraction: 0.7).delay(0.1)) {
                showMascot = true
            }
        }
    }
}

// MARK: - Supporting Types

struct OnboardingPage {
    enum Animation {
        case bouncing
        case waving
        case none
    }

    let title: String
    let description: String
    let expression: AnkylosaurusMascot.Expression
    let animation: Animation
}

// MARK: - Preview

#Preview {
    OnboardingWithMascotView()
}
