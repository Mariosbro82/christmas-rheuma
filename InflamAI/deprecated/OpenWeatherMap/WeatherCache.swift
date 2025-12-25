//
//  WeatherCache.swift
//  InflamAI-Swift
//
//  Persistent cache for weather data with automatic cleanup
//  Handles both current weather and forecast data
//

import Foundation
import CoreLocation

class WeatherCache {
    static let shared = WeatherCache()

    private let fileManager = FileManager.default
    private let cacheDirectory: URL
    private let maxCacheAge: TimeInterval = 86400 // 24 hours
    private let maxCacheSize: Int = 50 * 1024 * 1024 // 50 MB

    private init() {
        let urls = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)
        cacheDirectory = urls[0].appendingPathComponent("WeatherCache", isDirectory: true)

        // Create cache directory if needed
        try? fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        // Clean up old cache on init
        Task {
            await cleanExpiredCache()
        }
    }

    // MARK: - Save Methods

    func saveWeather(_ weather: WeatherData, for location: CLLocation) {
        let key = cacheKey(for: location)
        let fileURL = cacheDirectory.appendingPathComponent("\(key)_current.json")

        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(weather)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            print("❌ Failed to cache weather data: \(error)")
        }
    }

    func saveForecast(_ forecast: WeatherForecast, for location: CLLocation) {
        let key = cacheKey(for: location)
        let fileURL = cacheDirectory.appendingPathComponent("\(key)_forecast.json")

        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(forecast)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            print("❌ Failed to cache forecast data: \(error)")
        }
    }

    // MARK: - Load Methods

    func loadWeather(for location: CLLocation) -> WeatherData? {
        let key = cacheKey(for: location)
        let fileURL = cacheDirectory.appendingPathComponent("\(key)_current.json")

        guard fileManager.fileExists(atPath: fileURL.path) else {
            return nil
        }

        // Check file age
        guard let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
              let modificationDate = attributes[.modificationDate] as? Date,
              Date().timeIntervalSince(modificationDate) < maxCacheAge else {
            // Cache expired
            try? fileManager.removeItem(at: fileURL)
            return nil
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(WeatherData.self, from: data)
        } catch {
            print("❌ Failed to load cached weather: \(error)")
            // Remove corrupted cache
            try? fileManager.removeItem(at: fileURL)
            return nil
        }
    }

    func loadForecast(for location: CLLocation) -> WeatherForecast? {
        let key = cacheKey(for: location)
        let fileURL = cacheDirectory.appendingPathComponent("\(key)_forecast.json")

        guard fileManager.fileExists(atPath: fileURL.path) else {
            return nil
        }

        // Check file age
        guard let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
              let modificationDate = attributes[.modificationDate] as? Date,
              Date().timeIntervalSince(modificationDate) < maxCacheAge else {
            try? fileManager.removeItem(at: fileURL)
            return nil
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(WeatherForecast.self, from: data)
        } catch {
            print("❌ Failed to load cached forecast: \(error)")
            try? fileManager.removeItem(at: fileURL)
            return nil
        }
    }

    // MARK: - Cache Management

    func clearCache() {
        do {
            let contents = try fileManager.contentsOfDirectory(at: cacheDirectory, includingPropertiesForKeys: nil)
            for file in contents {
                try? fileManager.removeItem(at: file)
            }
            print("✅ Weather cache cleared")
        } catch {
            print("❌ Failed to clear cache: \(error)")
        }
    }

    func getCacheSize() -> Int {
        var totalSize = 0

        do {
            let contents = try fileManager.contentsOfDirectory(
                at: cacheDirectory,
                includingPropertiesForKeys: [.fileSizeKey]
            )

            for file in contents {
                let attributes = try fileManager.attributesOfItem(atPath: file.path)
                if let fileSize = attributes[.size] as? Int {
                    totalSize += fileSize
                }
            }
        } catch {
            print("❌ Failed to calculate cache size: \(error)")
        }

        return totalSize
    }

    func getCacheAge(for location: CLLocation) -> TimeInterval? {
        let key = cacheKey(for: location)
        let fileURL = cacheDirectory.appendingPathComponent("\(key)_current.json")

        guard fileManager.fileExists(atPath: fileURL.path),
              let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
              let modificationDate = attributes[.modificationDate] as? Date else {
            return nil
        }

        return Date().timeIntervalSince(modificationDate)
    }

    func cleanExpiredCache() async {
        do {
            let contents = try fileManager.contentsOfDirectory(
                at: cacheDirectory,
                includingPropertiesForKeys: [.contentModificationDateKey]
            )

            let now = Date()
            var removedCount = 0

            for file in contents {
                let attributes = try fileManager.attributesOfItem(atPath: file.path)
                if let modDate = attributes[.modificationDate] as? Date,
                   now.timeIntervalSince(modDate) > maxCacheAge {
                    try fileManager.removeItem(at: file)
                    removedCount += 1
                }
            }

            if removedCount > 0 {
                print("✅ Cleaned \(removedCount) expired cache files")
            }

            // Check total size and remove oldest if needed
            await enforceCacheSizeLimit()

        } catch {
            print("❌ Failed to clean expired cache: \(error)")
        }
    }

    private func enforceCacheSizeLimit() async {
        let currentSize = getCacheSize()

        guard currentSize > maxCacheSize else { return }

        do {
            let contents = try fileManager.contentsOfDirectory(
                at: cacheDirectory,
                includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]
            )

            // Sort by modification date (oldest first)
            let sortedFiles = contents.sorted { file1, file2 in
                let date1 = (try? fileManager.attributesOfItem(atPath: file1.path)[.modificationDate] as? Date) ?? Date.distantPast
                let date2 = (try? fileManager.attributesOfItem(atPath: file2.path)[.modificationDate] as? Date) ?? Date.distantPast
                return date1 < date2
            }

            var sizeFreed = 0
            let targetSize = maxCacheSize / 2 // Free up to 50% of limit

            for file in sortedFiles {
                guard sizeFreed < (currentSize - targetSize) else { break }

                let attributes = try fileManager.attributesOfItem(atPath: file.path)
                let fileSize = attributes[.size] as? Int ?? 0

                try fileManager.removeItem(at: file)
                sizeFreed += fileSize
            }

            print("✅ Freed \(sizeFreed / 1024) KB of cache space")

        } catch {
            print("❌ Failed to enforce cache size limit: \(error)")
        }
    }

    // MARK: - Helpers

    private func cacheKey(for location: CLLocation) -> String {
        // Round to 2 decimal places to group nearby locations
        let lat = (location.coordinate.latitude * 100).rounded() / 100
        let lon = (location.coordinate.longitude * 100).rounded() / 100
        return "\(lat)_\(lon)"
    }
}
