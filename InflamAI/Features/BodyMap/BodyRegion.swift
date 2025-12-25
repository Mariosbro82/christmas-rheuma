//
//  BodyRegion.swift
//  InflamAI
//
//  Anatomically accurate body region mapping (47 regions)
//

import Foundation
import SwiftUI

/// Body regions for pain tracking - anatomically accurate
enum BodyRegion: String, CaseIterable, Identifiable {
    // MARK: - Cervical Spine (7 vertebrae)
    case c1, c2, c3, c4, c5, c6, c7

    // MARK: - Thoracic Spine (12 vertebrae)
    case t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12

    // MARK: - Lumbar Spine (5 vertebrae)
    case l1, l2, l3, l4, l5

    // MARK: - Sacrum & SI Joints
    case sacrum
    case siLeft = "si_left"
    case siRight = "si_right"

    // MARK: - Upper Extremities
    case shoulderLeft = "shoulder_left"
    case shoulderRight = "shoulder_right"
    case elbowLeft = "elbow_left"
    case elbowRight = "elbow_right"
    case wristLeft = "wrist_left"
    case wristRight = "wrist_right"
    case handLeft = "hand_left"
    case handRight = "hand_right"

    // MARK: - Lower Extremities
    case hipLeft = "hip_left"
    case hipRight = "hip_right"
    case kneeLeft = "knee_left"
    case kneeRight = "knee_right"
    case ankleLeft = "ankle_left"
    case ankleRight = "ankle_right"
    case footLeft = "foot_left"
    case footRight = "foot_right"

    // MARK: - Identifiable

    var id: String { rawValue }

    // MARK: - Display Properties

    var displayName: String {
        switch self {
        // Cervical
        case .c1: return "C1 (Atlas)"
        case .c2: return "C2 (Axis)"
        case .c3: return "C3"
        case .c4: return "C4"
        case .c5: return "C5"
        case .c6: return "C6"
        case .c7: return "C7"

        // Thoracic
        case .t1: return "T1"
        case .t2: return "T2"
        case .t3: return "T3"
        case .t4: return "T4"
        case .t5: return "T5"
        case .t6: return "T6"
        case .t7: return "T7"
        case .t8: return "T8"
        case .t9: return "T9"
        case .t10: return "T10"
        case .t11: return "T11"
        case .t12: return "T12"

        // Lumbar
        case .l1: return "L1"
        case .l2: return "L2"
        case .l3: return "L3"
        case .l4: return "L4"
        case .l5: return "L5"

        // Sacrum & SI
        case .sacrum: return "Sacrum"
        case .siLeft: return "Left SI Joint"
        case .siRight: return "Right SI Joint"

        // Upper extremities
        case .shoulderLeft: return "Left Shoulder"
        case .shoulderRight: return "Right Shoulder"
        case .elbowLeft: return "Left Elbow"
        case .elbowRight: return "Right Elbow"
        case .wristLeft: return "Left Wrist"
        case .wristRight: return "Right Wrist"
        case .handLeft: return "Left Hand"
        case .handRight: return "Right Hand"

        // Lower extremities
        case .hipLeft: return "Left Hip"
        case .hipRight: return "Right Hip"
        case .kneeLeft: return "Left Knee"
        case .kneeRight: return "Right Knee"
        case .ankleLeft: return "Left Ankle"
        case .ankleRight: return "Right Ankle"
        case .footLeft: return "Left Foot"
        case .footRight: return "Right Foot"
        }
    }

    /// Position on front/back body view (normalized 0-1 coordinates)
    /// Origin (0,0) = top-left, (1,1) = bottom-right
    func position(forFrontView: Bool) -> CGPoint {
        if forFrontView {
            return frontViewPosition
        } else {
            return backViewPosition
        }
    }

    /// Front view positions
    private var frontViewPosition: CGPoint {
        switch self {
        // Front view shows shoulders, elbows, wrists, hands, hips, knees, ankles, feet
        case .shoulderLeft: return CGPoint(x: 0.35, y: 0.18)
        case .shoulderRight: return CGPoint(x: 0.65, y: 0.18)
        case .elbowLeft: return CGPoint(x: 0.25, y: 0.32)
        case .elbowRight: return CGPoint(x: 0.75, y: 0.32)
        case .wristLeft: return CGPoint(x: 0.20, y: 0.45)
        case .wristRight: return CGPoint(x: 0.80, y: 0.45)
        case .handLeft: return CGPoint(x: 0.18, y: 0.52)
        case .handRight: return CGPoint(x: 0.82, y: 0.52)
        case .hipLeft: return CGPoint(x: 0.42, y: 0.48)
        case .hipRight: return CGPoint(x: 0.58, y: 0.48)
        case .kneeLeft: return CGPoint(x: 0.42, y: 0.68)
        case .kneeRight: return CGPoint(x: 0.58, y: 0.68)
        case .ankleLeft: return CGPoint(x: 0.42, y: 0.88)
        case .ankleRight: return CGPoint(x: 0.58, y: 0.88)
        case .footLeft: return CGPoint(x: 0.42, y: 0.95)
        case .footRight: return CGPoint(x: 0.58, y: 0.95)
        default: return CGPoint(x: 0.5, y: 0.5) // Hidden on front view
        }
    }

    /// Back view positions (spine + SI joints)
    private var backViewPosition: CGPoint {
        switch self {
        // Cervical spine (top of neck to base)
        case .c1: return CGPoint(x: 0.50, y: 0.08)
        case .c2: return CGPoint(x: 0.50, y: 0.10)
        case .c3: return CGPoint(x: 0.50, y: 0.12)
        case .c4: return CGPoint(x: 0.50, y: 0.14)
        case .c5: return CGPoint(x: 0.50, y: 0.16)
        case .c6: return CGPoint(x: 0.50, y: 0.18)
        case .c7: return CGPoint(x: 0.50, y: 0.20)

        // Thoracic spine
        case .t1: return CGPoint(x: 0.50, y: 0.22)
        case .t2: return CGPoint(x: 0.50, y: 0.24)
        case .t3: return CGPoint(x: 0.50, y: 0.26)
        case .t4: return CGPoint(x: 0.50, y: 0.28)
        case .t5: return CGPoint(x: 0.50, y: 0.30)
        case .t6: return CGPoint(x: 0.50, y: 0.32)
        case .t7: return CGPoint(x: 0.50, y: 0.34)
        case .t8: return CGPoint(x: 0.50, y: 0.36)
        case .t9: return CGPoint(x: 0.50, y: 0.38)
        case .t10: return CGPoint(x: 0.50, y: 0.40)
        case .t11: return CGPoint(x: 0.50, y: 0.42)
        case .t12: return CGPoint(x: 0.50, y: 0.44)

        // Lumbar spine
        case .l1: return CGPoint(x: 0.50, y: 0.46)
        case .l2: return CGPoint(x: 0.50, y: 0.48)
        case .l3: return CGPoint(x: 0.50, y: 0.50)
        case .l4: return CGPoint(x: 0.50, y: 0.52)
        case .l5: return CGPoint(x: 0.50, y: 0.54)

        // Sacrum & SI joints
        case .sacrum: return CGPoint(x: 0.50, y: 0.58)
        case .siLeft: return CGPoint(x: 0.45, y: 0.58)
        case .siRight: return CGPoint(x: 0.55, y: 0.58)

        // Peripheral joints also visible from back
        case .shoulderLeft: return CGPoint(x: 0.35, y: 0.20)
        case .shoulderRight: return CGPoint(x: 0.65, y: 0.20)
        case .elbowLeft: return CGPoint(x: 0.25, y: 0.34)
        case .elbowRight: return CGPoint(x: 0.75, y: 0.34)
        case .hipLeft: return CGPoint(x: 0.42, y: 0.56)
        case .hipRight: return CGPoint(x: 0.58, y: 0.56)
        case .kneeLeft: return CGPoint(x: 0.42, y: 0.70)
        case .kneeRight: return CGPoint(x: 0.58, y: 0.70)

        default: return CGPoint(x: 0.5, y: 0.5) // Hidden on back view
        }
    }

    /// Anatomical category
    var category: RegionCategory {
        switch self {
        case .c1, .c2, .c3, .c4, .c5, .c6, .c7:
            return .cervical
        case .t1, .t2, .t3, .t4, .t5, .t6, .t7, .t8, .t9, .t10, .t11, .t12:
            return .thoracic
        case .l1, .l2, .l3, .l4, .l5:
            return .lumbar
        case .sacrum, .siLeft, .siRight:
            return .sacroiliac
        case .shoulderLeft, .shoulderRight, .elbowLeft, .elbowRight, .wristLeft, .wristRight, .handLeft, .handRight:
            return .upperExtremity
        case .hipLeft, .hipRight, .kneeLeft, .kneeRight, .ankleLeft, .ankleRight, .footLeft, .footRight:
            return .lowerExtremity
        }
    }

    /// Is visible on front view?
    var isVisibleOnFrontView: Bool {
        switch self {
        case .shoulderLeft, .shoulderRight, .elbowLeft, .elbowRight,
             .wristLeft, .wristRight, .handLeft, .handRight,
             .hipLeft, .hipRight, .kneeLeft, .kneeRight,
             .ankleLeft, .ankleRight, .footLeft, .footRight:
            return true
        default:
            return false
        }
    }

    /// Is visible on back view?
    var isVisibleOnBackView: Bool {
        // Spine and SI joints only on back, but shoulders/hips also visible
        return !isVisibleOnFrontView || category == .sacroiliac ||
               self == .shoulderLeft || self == .shoulderRight ||
               self == .hipLeft || self == .hipRight ||
               self == .elbowLeft || self == .elbowRight ||
               self == .kneeLeft || self == .kneeRight ||
               category == .cervical || category == .thoracic || category == .lumbar
    }
}

// MARK: - Region Category

enum RegionCategory: String {
    case cervical = "Cervical Spine"
    case thoracic = "Thoracic Spine"
    case lumbar = "Lumbar Spine"
    case sacroiliac = "Sacroiliac"
    case upperExtremity = "Upper Extremity"
    case lowerExtremity = "Lower Extremity"
}
