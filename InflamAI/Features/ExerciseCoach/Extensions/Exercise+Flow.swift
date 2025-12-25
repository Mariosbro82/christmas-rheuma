//
//  Exercise+Flow.swift
//  InflamAI
//
//  Converter from legacy Exercise model to new Flow-based mannequin coach system
//  Generates default keyframes and phases for existing exercises
//

import Foundation

extension Exercise {
    /// Convert Exercise to Flow for mannequin coach
    /// Generates basic keyframes and breath-based phases
    func toFlow() -> Flow {
        // Map target areas to joints
        let joints = mapTargetAreasToJoints(targetAreas)

        // Calculate cycle count from duration (8 cycles per minute avg)
        let cycleCount = max(4, (duration * 60) / 8)

        // Generate phases based on category
        let phases = generatePhases(for: category, duration: duration)

        // Generate keyframes based on target areas
        let keyframes = generateKeyframes(for: joints, category: category)

        return Flow(
            title: name,
            subtitle: benefits.first ?? "",
            estMinutes: Double(duration),
            targetAreas: joints,
            cycleCount: cycleCount,
            phases: phases,
            keyframes: keyframes,
            instructions: instructions,
            safetyTips: safetyTips,
            benefits: benefits
        )
    }

    // MARK: - Private Helpers

    private func mapTargetAreasToJoints(_ areas: [String]) -> [Joint] {
        var joints: Set<Joint> = []

        for area in areas {
            let areaLower = area.lowercased()

            // Spine regions
            if areaLower.contains("neck") || areaLower.contains("cervical") {
                joints.insert(.cervicalSpine)
            }
            if areaLower.contains("upper back") || areaLower.contains("thoracic") {
                joints.insert(.thoracicSpine)
            }
            if areaLower.contains("lower back") || areaLower.contains("lumbar") {
                joints.insert(.lumbarSpine)
            }
            if areaLower.contains("si joint") || areaLower.contains("sacroiliac") || areaLower.contains("pelvis") {
                joints.insert(.sacroiliacJoints)
            }
            if areaLower.contains("entire spine") || areaLower.contains("whole spine") {
                joints.formUnion([.cervicalSpine, .thoracicSpine, .lumbarSpine])
            }

            // Limbs
            if areaLower.contains("shoulder") {
                joints.insert(.shoulders)
            }
            if areaLower.contains("elbow") || areaLower.contains("arm") {
                joints.insert(.elbows)
            }
            if areaLower.contains("wrist") || areaLower.contains("hand") {
                joints.insert(.wrists)
            }
            if areaLower.contains("hip") {
                joints.insert(.hips)
            }
            if areaLower.contains("knee") || areaLower.contains("leg") {
                joints.insert(.knees)
            }
            if areaLower.contains("ankle") || areaLower.contains("foot") {
                joints.insert(.ankles)
            }
            if areaLower.contains("chest") || areaLower.contains("rib") {
                joints.insert(.ribCage)
            }
        }

        // Fallback: if no joints mapped, default to spine
        if joints.isEmpty {
            joints = [.lumbarSpine, .thoracicSpine]
        }

        return Array(joints)
    }

    private func generatePhases(for category: ExerciseCategory, duration: Int) -> [Phase] {
        switch category {
        case .stretching:
            return [
                Phase(
                    role: .setup,
                    durationSec: 5.0,
                    cue: Cue(
                        now: "Get into starting position",
                        next: "Begin the stretch on your next breath out",
                        breathCue: .exhale
                    ),
                    keyframeIndex: 0
                ),
                Phase(
                    role: .move,
                    breaths: 2,
                    cue: Cue(
                        now: "Slowly move into the stretch",
                        next: "Hold the position",
                        breathCue: .exhale
                    ),
                    keyframeIndex: 1
                ),
                Phase(
                    role: .hold,
                    durationSec: 4.0,
                    cue: Cue(
                        now: "Hold the stretch gently",
                        next: "Release on your exhale",
                        breathCue: .inhale
                    ),
                    keyframeIndex: 1
                ),
                Phase(
                    role: .returnSlow,
                    breaths: 2,
                    cue: Cue(
                        now: "Slowly return to starting position",
                        next: "Rest and breathe",
                        breathCue: .exhale
                    ),
                    keyframeIndex: 0
                ),
                Phase(
                    role: .rest,
                    durationSec: 3.0,
                    cue: Cue(
                        now: "Rest and breathe naturally",
                        next: "Prepare for next cycle",
                        breathCue: nil
                    ),
                    keyframeIndex: 0
                )
            ]

        case .breathing:
            return [
                Phase(
                    role: .setup,
                    durationSec: 3.0,
                    cue: Cue(
                        now: "Sit comfortably with straight spine",
                        next: "Begin inhaling deeply",
                        breathCue: nil
                    ),
                    keyframeIndex: 0
                ),
                Phase(
                    role: .move,
                    breaths: 1,
                    cue: Cue(
                        now: "Breathe in deeply through your nose",
                        next: "Hold your breath",
                        breathCue: .inhale
                    ),
                    keyframeIndex: 1
                ),
                Phase(
                    role: .hold,
                    durationSec: 4.0,
                    cue: Cue(
                        now: "Hold your breath gently",
                        next: "Exhale slowly",
                        breathCue: nil
                    ),
                    keyframeIndex: 1
                ),
                Phase(
                    role: .returnSlow,
                    breaths: 1,
                    cue: Cue(
                        now: "Exhale slowly through your mouth",
                        next: "Rest between breaths",
                        breathCue: .exhale
                    ),
                    keyframeIndex: 0
                ),
                Phase(
                    role: .rest,
                    durationSec: 2.0,
                    cue: Cue(
                        now: "Rest and breathe naturally",
                        next: "Prepare for next cycle",
                        breathCue: nil
                    ),
                    keyframeIndex: 0
                )
            ]

        case .mobility, .all:
            return [
                Phase(
                    role: .setup,
                    durationSec: 4.0,
                    cue: Cue(
                        now: "Get into starting position",
                        next: "Begin the movement",
                        breathCue: nil
                    ),
                    keyframeIndex: 0
                ),
                Phase(
                    role: .move,
                    breaths: 2,
                    cue: Cue(
                        now: "Move slowly and controlled",
                        next: "Return to start",
                        breathCue: .exhale
                    ),
                    keyframeIndex: 1
                ),
                Phase(
                    role: .returnSlow,
                    breaths: 2,
                    cue: Cue(
                        now: "Return to starting position",
                        next: "Brief rest",
                        breathCue: .inhale
                    ),
                    keyframeIndex: 0
                ),
                Phase(
                    role: .rest,
                    durationSec: 2.0,
                    cue: Cue(
                        now: "Rest briefly",
                        next: "Prepare for next cycle",
                        breathCue: nil
                    ),
                    keyframeIndex: 0
                )
            ]

        default:
            // Generic phases for other categories
            return [
                Phase(
                    role: .setup,
                    durationSec: 5.0,
                    cue: Cue(
                        now: "Prepare for exercise",
                        next: "Begin movement",
                        breathCue: nil
                    ),
                    keyframeIndex: 0
                ),
                Phase(
                    role: .move,
                    durationSec: 6.0,
                    cue: Cue(
                        now: "Perform the movement",
                        next: "Return to start",
                        breathCue: .exhale
                    ),
                    keyframeIndex: 1
                ),
                Phase(
                    role: .rest,
                    durationSec: 3.0,
                    cue: Cue(
                        now: "Rest",
                        next: "Repeat",
                        breathCue: nil
                    ),
                    keyframeIndex: 0
                )
            ]
        }
    }

    private func generateKeyframes(for joints: [Joint], category: ExerciseCategory) -> [Keyframe] {
        // Keyframe 0: Neutral starting position
        let startPose = Pose.neutral

        // Keyframe 1: Exercise position based on joints and category
        var endAngles: [Joint: Double] = [:]

        for joint in joints {
            switch (joint, category) {
            // Stretching: gentle flexion/extension
            case (.cervicalSpine, .stretching):
                endAngles[joint] = -15.0  // Forward neck stretch
            case (.thoracicSpine, .stretching):
                endAngles[joint] = -10.0  // Upper back stretch
            case (.lumbarSpine, .stretching):
                endAngles[joint] = -20.0  // Lower back stretch
            case (.shoulders, .stretching):
                endAngles[joint] = 45.0   // Shoulder elevation
            case (.hips, .stretching):
                endAngles[joint] = 30.0   // Hip flexion

            // Breathing: chest expansion
            case (.ribCage, .breathing):
                endAngles[joint] = 15.0   // Chest expansion
            case (.cervicalSpine, .breathing):
                endAngles[joint] = -5.0   // Slight neck extension

            // Mobility: full range motion
            case (.cervicalSpine, .mobility):
                endAngles[joint] = -25.0  // Neck rotation
            case (.thoracicSpine, .mobility):
                endAngles[joint] = 15.0   // Thoracic rotation
            case (.lumbarSpine, .mobility):
                endAngles[joint] = 25.0   // Lumbar rotation
            case (.hips, .mobility):
                endAngles[joint] = 45.0   // Hip circle

            // Default: mild flexion
            default:
                endAngles[joint] = 15.0
            }
        }

        let endPose = Pose(angles: endAngles)

        return [
            Keyframe(t: 0.0, pose: startPose, description: "Start"),
            Keyframe(t: 1.0, pose: endPose, description: "Peak")
        ]
    }
}
