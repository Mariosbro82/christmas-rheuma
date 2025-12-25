//
//  LottieView.swift
//  InflamAI
//
//  SwiftUI wrapper for Lottie animations
//

import SwiftUI
#if canImport(Lottie)
import Lottie
#endif

#if os(iOS)
struct LottieView: UIViewRepresentable {
    enum Source {
        case bundle(String)
        case url(URL)
    }

    let source: Source
    let loopMode: LottieLoopMode
    let animationSpeed: CGFloat
    let contentMode: UIView.ContentMode

    init(
        animationName: String,
        loopMode: LottieLoopMode = .loop,
        animationSpeed: CGFloat = 1.0,
        contentMode: UIView.ContentMode = .scaleAspectFit
    ) {
        self.source = .bundle(animationName)
        self.loopMode = loopMode
        self.animationSpeed = animationSpeed
        self.contentMode = contentMode
    }

    init(
        url: URL,
        loopMode: LottieLoopMode = .loop,
        animationSpeed: CGFloat = 1.0,
        contentMode: UIView.ContentMode = .scaleAspectFit
    ) {
        self.source = .url(url)
        self.loopMode = loopMode
        self.animationSpeed = animationSpeed
        self.contentMode = contentMode
    }

    func makeUIView(context: Context) -> LottieAnimationView {
        let animationView = LottieAnimationView()

        animationView.contentMode = contentMode
        animationView.loopMode = loopMode
        animationView.animationSpeed = animationSpeed
        animationView.backgroundBehavior = .pauseAndRestore

        // Load animation based on source
        switch source {
        case .bundle(let animationName):
            // Load from bundle
            if let path = Bundle.main.path(forResource: animationName, ofType: "json", inDirectory: "Resources/Animations") {
                animationView.animation = LottieAnimation.filepath(path)
                animationView.play() // Start playing immediately for bundle animations
            } else {
                print("⚠️ Lottie animation '\(animationName).json' not found in bundle at Resources/Animations/")
            }
        case .url(let url):
            // Load from URL asynchronously (supports both .json and .lottie formats)
            Task {
                do {
                    // Lottie automatically handles both JSON and DotLottie formats
                    let animation = try await LottieAnimation.loadedFrom(url: url)
                    await MainActor.run {
                        animationView.animation = animation
                        animationView.play()
                    }
                } catch {
                    print("❌ Failed to load Lottie animation from \(url): \(error.localizedDescription)")
                    await MainActor.run {
                        // Show placeholder instead of blank space
                        let placeholderLabel = UILabel()
                        placeholderLabel.text = "Animation unavailable"
                        placeholderLabel.textColor = .systemGray
                        placeholderLabel.font = .preferredFont(forTextStyle: .caption1)
                        placeholderLabel.textAlignment = .center
                        placeholderLabel.frame = animationView.bounds
                        placeholderLabel.autoresizingMask = [.flexibleWidth, .flexibleHeight]
                        animationView.addSubview(placeholderLabel)
                    }
                }
            }
        }

        return animationView
    }

    func updateUIView(_ uiView: LottieAnimationView, context: Context) {
        if !uiView.isAnimationPlaying && uiView.animation != nil {
            uiView.play()
        }
    }
}

// MARK: - Convenience Extensions

extension LottieView {
    // MARK: Bundle-based

    /// Plays the animation once and stops
    static func playOnce(
        _ animationName: String,
        speed: CGFloat = 1.0,
        contentMode: UIView.ContentMode = .scaleAspectFit
    ) -> LottieView {
        LottieView(
            animationName: animationName,
            loopMode: .playOnce,
            animationSpeed: speed,
            contentMode: contentMode
        )
    }

    /// Loops the animation continuously
    static func loop(
        _ animationName: String,
        speed: CGFloat = 1.0,
        contentMode: UIView.ContentMode = .scaleAspectFit
    ) -> LottieView {
        LottieView(
            animationName: animationName,
            loopMode: .loop,
            animationSpeed: speed,
            contentMode: contentMode
        )
    }

    /// Auto-reverses the animation (plays forward then backward)
    static func autoReverse(
        _ animationName: String,
        speed: CGFloat = 1.0,
        contentMode: UIView.ContentMode = .scaleAspectFit
    ) -> LottieView {
        LottieView(
            animationName: animationName,
            loopMode: .autoReverse,
            animationSpeed: speed,
            contentMode: contentMode
        )
    }

    // MARK: URL-based

    /// Loads and loops animation from URL
    static func fromURL(
        _ url: URL,
        loopMode: LottieLoopMode = .loop,
        speed: CGFloat = 1.0,
        contentMode: UIView.ContentMode = .scaleAspectFit
    ) -> LottieView {
        LottieView(
            url: url,
            loopMode: loopMode,
            animationSpeed: speed,
            contentMode: contentMode
        )
    }

    /// Loads and loops animation from URL string
    static func fromURL(
        _ urlString: String,
        loopMode: LottieLoopMode = .loop,
        speed: CGFloat = 1.0,
        contentMode: UIView.ContentMode = .scaleAspectFit
    ) -> LottieView? {
        guard let url = URL(string: urlString) else { return nil }
        return fromURL(url, loopMode: loopMode, speed: speed, contentMode: contentMode)
    }
}

// MARK: - Preview

#Preview("Looping Animation") {
    LottieView.loop("sleeping-dino")
        .frame(width: 300, height: 300)
        .background(Color.gray.opacity(0.1))
}
#endif
