//
//  UIApplication+TopMost.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import UIKit

extension UIApplication {
    static func presentTopViewController(_ controller: UIViewController) {
        guard var topController = UIApplication.shared.connectedScenes
            .compactMap({ ($0 as? UIWindowScene)?.keyWindow?.rootViewController })
            .first else { return }
        while let presented = topController.presentedViewController {
            topController = presented
        }
        topController.present(controller, animated: true)
    }
}
