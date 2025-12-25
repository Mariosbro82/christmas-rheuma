//
//  YouTubePlayerView.swift
//  InflamAI
//
//  SwiftUI wrapper for YouTube video player using WKWebView
//  Plays YouTube videos embedded in the app
//

import SwiftUI
import WebKit

/// YouTube video player that embeds videos using WKWebView
struct YouTubePlayerView: View {
    let videoID: String
    @State private var isLoading = true
    @State private var loadError: String?

    var body: some View {
        ZStack {
            // Video player
            YouTubeWebView(
                videoID: videoID,
                isLoading: $isLoading,
                loadError: $loadError
            )

            // Loading indicator
            if isLoading {
                ZStack {
                    Color.black.opacity(0.3)
                    ProgressView()
                        .scaleEffect(1.5)
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                }
            }

            // Error state
            if let error = loadError {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 40))
                        .foregroundColor(.orange)

                    Text("Video Unavailable")
                        .font(.headline)

                    Text(error)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.systemBackground))
            }
        }
        .aspectRatio(16/9, contentMode: .fit)
        .cornerRadius(12)
        .accessibilityLabel("YouTube video player")
        .accessibilityHint("Double tap to play exercise tutorial video")
    }
}

/// WKWebView wrapper for YouTube embed
struct YouTubeWebView: UIViewRepresentable {
    let videoID: String
    @Binding var isLoading: Bool
    @Binding var loadError: String?

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        configuration.mediaTypesRequiringUserActionForPlayback = []

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.scrollView.isScrollEnabled = false
        webView.backgroundColor = .black
        webView.isOpaque = false
        webView.navigationDelegate = context.coordinator

        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        guard let embedURL = YouTubeHelper.embedURL(for: videoID) else {
            loadError = "Invalid video ID"
            isLoading = false
            return
        }

        // Create HTML with responsive iframe
        let html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <style>
                * {
                    margin: 0;
                    padding: 0;
                }
                html, body {
                    width: 100%;
                    height: 100%;
                    background-color: #000;
                    overflow: hidden;
                }
                .video-container {
                    position: relative;
                    width: 100%;
                    height: 100%;
                }
                iframe {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    border: none;
                }
            </style>
        </head>
        <body>
            <div class="video-container">
                <iframe
                    src="\(embedURL.absoluteString)"
                    frameborder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowfullscreen>
                </iframe>
            </div>
        </body>
        </html>
        """

        // Use youtube-nocookie.com as baseURL to avoid security sandbox issues
        let baseURL = URL(string: "https://www.youtube-nocookie.com")
        webView.loadHTMLString(html, baseURL: baseURL)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, WKNavigationDelegate {
        var parent: YouTubeWebView

        init(_ parent: YouTubeWebView) {
            self.parent = parent
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            // Remove artificial delay - mark as loaded immediately
            self.parent.isLoading = false
            print("✅ YouTube video loaded successfully for ID: \(parent.videoID)")
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            parent.isLoading = false
            let nsError = error as NSError
            print("❌ YouTube video load failed: \(nsError.domain) - \(nsError.code) - \(nsError.localizedDescription)")
            parent.loadError = "Failed to load video: \(nsError.localizedDescription)"
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            parent.isLoading = false
            let nsError = error as NSError
            print("❌ YouTube provisional navigation failed: \(nsError.domain) - \(nsError.code) - \(nsError.localizedDescription)")

            // Provide more specific error messages
            if nsError.code == NSURLErrorNotConnectedToInternet {
                parent.loadError = "No internet connection"
            } else if nsError.code == NSURLErrorTimedOut {
                parent.loadError = "Connection timed out"
            } else if nsError.code == NSURLErrorCannotFindHost {
                parent.loadError = "Cannot reach YouTube servers"
            } else {
                parent.loadError = "Network error: \(nsError.localizedDescription)"
            }
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            // Allow all YouTube-related navigation
            if let url = navigationAction.request.url {
                let host = url.host ?? ""
                if host.contains("youtube") || host.contains("googlevideo") || host.contains("ytimg") || host.contains("ggpht") {
                    decisionHandler(.allow)
                    return
                }
            }
            decisionHandler(.allow)
        }
    }
}

// MARK: - Convenience Initializers

extension YouTubePlayerView {
    /// Initialize with full YouTube URL
    init(url: String) {
        if let videoID = YouTubeHelper.extractVideoID(from: url) {
            self.videoID = videoID
        } else {
            // Fallback to empty video ID (will show error)
            self.videoID = ""
        }
    }
}

// MARK: - Preview

#Preview {
    VStack {
        Text("Cat-Cow Stretch")
            .font(.headline)
            .padding()

        YouTubePlayerView(videoID: "y39PrKY_4JM")
            .padding()

        Spacer()
    }
}
