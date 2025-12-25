//
//  SharedTypes.swift
//  InflamAI-Swift
//
//  Central type definitions to resolve module visibility issues
//

import Foundation
import SwiftUI
import CoreData

// Re-export types from Features to make them visible everywhere
// This file should be compiled before other files that depend on these types

// Note: The actual implementations are in:
// - Features/Questionnaires/QuestionnaireModels.swift
// - Features/Questionnaires/QuestionnaireFormView.swift
// - Features/Questionnaires/QuestionnaireHomeSection.swift
// - Features/BodyMap/BodyMapModels.swift

// This file just ensures build order and visibility
