//
//  YouTubeHelper.swift
//  InflamAI
//
//  Helper utilities for YouTube video handling
//

import Foundation

struct YouTubeHelper {
    /// Extract video ID from various YouTube URL formats
    /// Supports:
    /// - https://www.youtube.com/watch?v=VIDEO_ID
    /// - https://youtu.be/VIDEO_ID
    /// - https://www.youtube.com/shorts/VIDEO_ID
    /// - https://m.youtube.com/watch?v=VIDEO_ID
    static func extractVideoID(from urlString: String) -> String? {
        guard let url = URL(string: urlString) else { return nil }

        // Handle youtube.com/watch?v=VIDEO_ID
        if url.host?.contains("youtube.com") == true,
           let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
           let queryItems = components.queryItems,
           let videoID = queryItems.first(where: { $0.name == "v" })?.value {
            return videoID
        }

        // Handle youtu.be/VIDEO_ID
        if url.host?.contains("youtu.be") == true {
            return url.lastPathComponent
        }

        // Handle youtube.com/shorts/VIDEO_ID
        if url.host?.contains("youtube.com") == true,
           url.pathComponents.contains("shorts"),
           let videoID = url.pathComponents.last {
            return videoID
        }

        // If it's already just a video ID (11 characters)
        if urlString.count == 11 && !urlString.contains("/") {
            return urlString
        }

        return nil
    }

    /// Create embed URL for video player
    static func embedURL(for videoID: String) -> URL? {
        // Use youtube-nocookie.com for better privacy
        // Parameters:
        // - playsinline=1: Play inline on iOS (not fullscreen)
        // - modestbranding=1: Minimal YouTube branding
        // - rel=0: Don't show related videos from other channels
        // - enablejsapi=1: Enable JavaScript API for better control
        // - origin: Required for CORS validation (using youtube-nocookie.com)
        // - fs=1: Allow fullscreen
        // - controls=1: Show video controls
        let origin = "https://www.youtube-nocookie.com"
        let urlString = "https://www.youtube-nocookie.com/embed/\(videoID)?playsinline=1&modestbranding=1&rel=0&enablejsapi=1&origin=\(origin)&fs=1&controls=1"
        return URL(string: urlString)
    }

    /// Create thumbnail URL for video
    static func thumbnailURL(for videoID: String, quality: ThumbnailQuality = .medium) -> URL? {
        let urlString = "https://img.youtube.com/vi/\(videoID)/\(quality.rawValue).jpg"
        return URL(string: urlString)
    }

    enum ThumbnailQuality: String {
        case maxres = "maxresdefault"  // 1280x720
        case high = "hqdefault"         // 480x360
        case medium = "mqdefault"       // 320x180
        case `default` = "default"      // 120x90
    }
}
