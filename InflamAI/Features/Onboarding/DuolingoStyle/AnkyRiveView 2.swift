//
//  AnkyRiveView.swift
//  InflamAI
//
//  Rive-powered Anky mascot with state machine control
//  Falls back to SwiftUI Canvas animation when Rive unavailable
//
//  Available Animations (from rive-first-animations):
//  - idle: Default breathing state
//  - wave: Greeting animation
//
//  Missing animations use fallback to AnkyAnimatedMascot (Canvas)
//

import SwiftUI
import RiveRuntime

// MARK: - Anky Static Image Assets

/// Maps states to static dino image assets from Assets.xcassets
/// Use these for non-animated contexts (feature screens, tips, empty states)
enum AnkyImageAsset: String {
    // Emotional states
    case happy = "dino-happy 1"
    case sad = "dino-sad"
    case tired = "dino-tired 1"
    case sleeping = "dino-sleeping 1"
    case scared = "dino-sweat-little-scared"

    // Activity states
    case meditating = "dino-meditating"
    case walking = "dino-walking 1"
    case exercising = "dino-walking-fast"
    case strong = "dino-strong-mussel"

    // Informational states
    case explaining = "dino-showing-whiteboard"
    case spineShowing = "dino-spine-showing"
    case medications = "dino-medications"
    case health = "dino-hearth 1"
    case privacy = "dino-privacy"
    case stop = "dino-stop-hand"

    // Default
    case normal = "dino-stading-normal"

    /// Get appropriate image for an AnkyState
    static func forState(_ state: AnkyState) -> AnkyImageAsset {
        switch state {
        case .idle, .attentive: return .normal
        case .waving: return .happy
        case .happy, .proud: return .happy
        case .celebrating: return .happy
        case .thinking: return .meditating
        case .concerned, .sympathetic: return .sad
        case .encouraging: return .strong
        case .explaining: return .explaining
        case .curious: return .meditating
        case .sleeping, .waking: return .sleeping
        }
    }
}

/// SwiftUI View for displaying static Anky images
struct AnkyImageView: View {
    let asset: AnkyImageAsset
    var size: CGFloat = 120

    var body: some View {
        Image(asset.rawValue)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(width: size, height: size)
    }
}

// MARK: - Anky Rive State

/// Maps app states to Rive state machine inputs
enum AnkyRiveState: String, CaseIterable {
    case idle
    case waving
    case happy
    case celebrating
    case thinking
    case concerned
    case encouraging
    case sleeping

    /// Whether this state has a Rive animation
    var hasRiveAnimation: Bool {
        switch self {
        case .idle, .waving:
            return true
        default:
            return false // Use fallback for these
        }
    }

    /// Rive animation name
    var riveAnimationName: String {
        switch self {
        case .idle: return "idle"
        case .waving: return "wave"
        default: return "idle"
        }
    }

    /// Convert from legacy AnkyState
    static func from(_ legacyState: AnkyState) -> AnkyRiveState {
        switch legacyState {
        case .idle, .attentive: return .idle
        case .waving: return .waving
        case .happy: return .happy
        case .celebrating: return .celebrating
        case .thinking: return .thinking
        case .concerned, .sympathetic: return .concerned
        case .encouraging, .explaining: return .encouraging
        case .sleeping, .waking: return .sleeping
        case .proud, .curious: return .happy
        }
    }
}

// MARK: - Rive Availability Check

/// Check if Rive file is available in bundle
private func isRiveFileAvailable() -> Bool {
    let available = Bundle.main.url(forResource: "anky_mascot", withExtension: "riv") != nil
    #if DEBUG
    print("🎬 Rive file available in bundle: \(available)")
    #endif
    return available
}

// MARK: - Anky Rive View

/// Smart mascot view that uses Rive when available, falls back to static image
/// NOTE: Canvas fallback (AnkyAnimatedMascot aka "Optimus Prime") is REMOVED
struct AnkyRiveView: View {
    let size: CGFloat
    let state: AnkyState
    var showShadow: Bool = true
    var onTap: (() -> Void)?

    @State private var useImageFallback: Bool

    private var riveState: AnkyRiveState {
        AnkyRiveState.from(state)
    }

    init(size: CGFloat, state: AnkyState, showShadow: Bool = true, onTap: (() -> Void)? = nil) {
        self.size = size
        self.state = state
        self.showShadow = showShadow
        self.onTap = onTap
        // Start with image fallback if Rive file not available
        self._useImageFallback = State(initialValue: !isRiveFileAvailable())
    }

    var body: some View {
        ZStack {
            if riveState.hasRiveAnimation && !useImageFallback {
                // Use Rive animation (only for idle + waving)
                RiveMascotView(
                    state: riveState,
                    size: size,
                    showShadow: showShadow,
                    onLoadError: {
                        print("🎬 Rive load error - falling back to static image")
                        useImageFallback = true
                    }
                )
                .onTapGesture {
                    HapticFeedback.light()
                    onTap?()
                }
            } else {
                // Fallback to static image (NO Canvas "Optimus Prime"!)
                staticImageFallback
            }
        }
        .frame(width: size, height: size)
    }

    // MARK: - Static Image Fallback

    /// Static image fallback when Rive isn't available or state not supported
    private var staticImageFallback: some View {
        ZStack {
            // Shadow
            if showShadow {
                Ellipse()
                    .fill(
                        RadialGradient(
                            colors: [Color.black.opacity(0.15), Color.clear],
                            center: .center,
                            startRadius: 0,
                            endRadius: size * 0.35
                        )
                    )
                    .frame(width: size * 0.5, height: size * 0.12)
                    .offset(y: size * 0.42)
            }

            // Static dino image based on state
            Image(AnkyImageAsset.forState(state).rawValue)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: size * 0.9, height: size * 0.9)
        }
        .onTapGesture {
            HapticFeedback.light()
            onTap?()
        }
    }
}

// MARK: - Rive Mascot View

/// Observable wrapper for RiveViewModel to ensure proper SwiftUI lifecycle
@MainActor
class RiveViewModelWrapper: ObservableObject {
    @Published var viewModel: RiveViewModel?
    @Published var loadFailed = false
    @Published var hasLoaded = false
    @Published var isLoading = false

    init() {
        // Don't load in init - defer to onAppear for safer SwiftUI lifecycle
    }

    func loadRiveFileIfNeeded() {
        // Ensure we're not already loading or loaded
        guard viewModel == nil && !loadFailed && !isLoading else { return }

        isLoading = true

        #if DEBUG
        print("🎬 RiveViewModelWrapper: Attempting to load anky_mascot.riv")
        #endif

        // Defer loading slightly to ensure SwiftUI view is ready
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            do {
                let vm = try RiveViewModel(fileName: "anky_mascot")
                self.viewModel = vm
                self.hasLoaded = true
                self.isLoading = false
                #if DEBUG
                print("🎬 RiveViewModelWrapper: Successfully loaded Rive file")
                #endif
            } catch {
                #if DEBUG
                print("🎬 RiveViewModelWrapper: Failed to load Rive file: \(error)")
                #endif
                self.loadFailed = true
                self.isLoading = false
            }
        }
    }

    func playAnimation(_ name: String) {
        guard let vm = viewModel else { return }
        do {
            try vm.play(animationName: name)
            #if DEBUG
            print("🎬 Playing animation: \(name)")
            #endif
        } catch {
            #if DEBUG
            print("🎬 Animation '\(name)' failed: \(error)")
            #endif
            // Try idle as fallback
            try? vm.play(animationName: "idle")
        }
    }
}

/// Rive-powered mascot view with animation support
struct RiveMascotView: View {
    let state: AnkyRiveState
    let size: CGFloat
    let showShadow: Bool
    var onLoadError: (() -> Void)?

    @StateObject private var wrapper = RiveViewModelWrapper()

    var body: some View {
        ZStack {
            // Shadow beneath character
            if showShadow {
                Ellipse()
                    .fill(
                        RadialGradient(
                            colors: [Color.black.opacity(0.15), Color.clear],
                            center: .center,
                            startRadius: 0,
                            endRadius: size * 0.35
                        )
                    )
                    .frame(width: size * 0.5, height: size * 0.12)
                    .offset(y: size * 0.42)
            }

            // Rive animation view
            if let viewModel = wrapper.viewModel, wrapper.hasLoaded {
                viewModel.view()
                    .frame(width: size, height: size)
                    .onChange(of: state) { _, newState in
                        wrapper.playAnimation(newState.riveAnimationName)
                    }
                    .onAppear {
                        // Play initial animation
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                            wrapper.playAnimation(state.riveAnimationName)
                        }
                    }
            } else if wrapper.loadFailed {
                // Show nothing - parent will show fallback
                EmptyView()
                    .onAppear {
                        onLoadError?()
                    }
            } else {
                // Loading state
                ProgressView()
                    .frame(width: size, height: size)
            }
        }
        .onAppear {
            // Trigger deferred loading when view appears
            wrapper.loadRiveFileIfNeeded()
        }
        .onReceive(wrapper.$loadFailed) { failed in
            if failed {
                onLoadError?()
            }
        }
    }
}

// MARK: - Previews

#Preview("Rive Anky - Idle") {
    VStack(spacing: 20) {
        AnkyRiveView(size: 200, state: .idle, showShadow: true)

        Text("Idle Animation")
            .font(.headline)
            .foregroundColor(Colors.Gray.g700)
    }
    .padding()
    .background(Colors.Gray.g50)
}

#Preview("Rive Anky - Waving") {
    VStack(spacing: 20) {
        AnkyRiveView(size: 200, state: .waving, showShadow: true)

        Text("Wave Animation")
            .font(.headline)
            .foregroundColor(Colors.Gray.g700)
    }
    .padding()
    .background(Colors.Gray.g50)
}

#Preview("All States") {
    ScrollView {
        VStack(spacing: 24) {
            ForEach(AnkyRiveState.allCases, id: \.self) { riveState in
                HStack(spacing: 16) {
                    let ankyState: AnkyState = {
                        switch riveState {
                        case .idle: return .idle
                        case .waving: return .waving
                        case .happy: return .happy
                        case .celebrating: return .celebrating
                        case .thinking: return .thinking
                        case .concerned: return .concerned
                        case .encouraging: return .encouraging
                        case .sleeping: return .sleeping
                        }
                    }()

                    AnkyRiveView(size: 80, state: ankyState, showShadow: true)

                    VStack(alignment: .leading, spacing: 4) {
                        Text(riveState.rawValue.capitalized)
                            .font(.headline)

                        HStack(spacing: 4) {
                            Circle()
                                .fill(riveState.hasRiveAnimation ? Colors.Semantic.success : Colors.Semantic.warning)
                                .frame(width: 8, height: 8)

                            Text(riveState.hasRiveAnimation ? "Rive" : "Canvas Fallback")
                                .font(.caption)
                                .foregroundColor(Colors.Gray.g500)
                        }
                    }

                    Spacer()
                }
                .padding(.horizontal)
            }
        }
        .padding(.vertical)
    }
    .background(Colors.Gray.g50)
}

#Preview("Interactive Demo") {
    struct InteractiveDemo: View {
        @State private var currentState: AnkyState = .idle

        var body: some View {
            VStack(spacing: 32) {
                AnkyRiveView(
                    size: 200,
                    state: currentState,
                    showShadow: true
                ) {
                    withAnimation {
                        currentState = currentState == .waving ? .idle : .waving
                    }
                }

                Text("Tap Anky to wave!")
                    .font(.subheadline)
                    .foregroundColor(Colors.Gray.g500)

                HStack(spacing: 12) {
                    Button("Idle") {
                        currentState = .idle
                    }
                    .buttonStyle(.bordered)

                    Button("Wave") {
                        currentState = .waving
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
            .background(Colors.Gray.g50)
        }
    }

    return InteractiveDemo()
}

#Preview("Static Image Assets") {
    ScrollView {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))], spacing: 16) {
            ForEach([
                AnkyImageAsset.happy,
                .sad,
                .tired,
                .sleeping,
                .scared,
                .meditating,
                .walking,
                .exercising,
                .strong,
                .explaining,
                .spineShowing,
                .medications,
                .health,
                .privacy,
                .stop,
                .normal
            ], id: \.rawValue) { asset in
                VStack(spacing: 8) {
                    AnkyImageView(asset: asset, size: 80)
                    Text(asset.rawValue.replacingOccurrences(of: "dino-", with: ""))
                        .font(.caption2)
                        .foregroundColor(Colors.Gray.g500)
                        .lineLimit(1)
                }
            }
        }
        .padding()
    }
    .background(Colors.Gray.g50)
}
