//
//  StandardizedMedications.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import Foundation
import SwiftUI

// MARK: - Medication Categories
enum MedicationCategory: String, CaseIterable {
    case corticosteroid = "Corticosteroid"
    case biologic = "Biologic"
    case dmard = "DMARD"
    case nsaid = "NSAID"
    
    var color: Color {
        switch self {
        case .corticosteroid:
            return .orange
        case .biologic:
            return .purple
        case .dmard:
            return .blue
        case .nsaid:
            return .green
        }
    }
    
    var emoji: String {
        switch self {
        case .corticosteroid:
            return "💊"
        case .biologic:
            return "💉"
        case .dmard:
            return "💊"
        case .nsaid:
            return "💊"
        }
    }
}

// MARK: - Predefined Medication
struct PredefinedMedication: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let category: MedicationCategory
    let emoji: String
    let standardDosages: [String]
    let standardFrequencies: [String]
    let commonInstructions: String
    let interactions: [String]
    let sideEffects: [String]
    
    static func == (lhs: PredefinedMedication, rhs: PredefinedMedication) -> Bool {
        lhs.name == rhs.name
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
    }
}

// MARK: - Standardized Medications Database
class StandardizedMedicationsDatabase {
    static let shared = StandardizedMedicationsDatabase()
    
    private init() {}
    
    let predefinedMedications: [PredefinedMedication] = [
        // MARK: - Corticosteroids
        PredefinedMedication(
            name: "Prednisone",
            category: .corticosteroid,
            emoji: "💊",
            standardDosages: ["5mg", "10mg", "20mg", "40mg"],
            standardFrequencies: ["Daily", "Twice Daily", "As Needed"],
            commonInstructions: "Take with food to reduce stomach irritation",
            interactions: ["NSAIDs", "Blood thinners", "Diabetes medications"],
            sideEffects: ["Weight gain", "Mood changes", "Increased appetite", "Sleep disturbances"]
        ),
        PredefinedMedication(
            name: "Prednisolone",
            category: .corticosteroid,
            emoji: "💊",
            standardDosages: ["5mg", "10mg", "25mg"],
            standardFrequencies: ["Daily", "Twice Daily"],
            commonInstructions: "Take with food, do not stop abruptly",
            interactions: ["NSAIDs", "Anticoagulants"],
            sideEffects: ["Fluid retention", "High blood pressure", "Bone thinning"]
        ),
        PredefinedMedication(
            name: "Methylprednisolone",
            category: .corticosteroid,
            emoji: "💊",
            standardDosages: ["4mg", "8mg", "16mg", "32mg"],
            standardFrequencies: ["Daily", "Twice Daily"],
            commonInstructions: "Take in the morning to reduce sleep interference",
            interactions: ["Live vaccines", "NSAIDs"],
            sideEffects: ["Increased infection risk", "Mood changes", "Osteoporosis"]
        ),
        
        // MARK: - Biologics
        PredefinedMedication(
            name: "Adalimumab (Humira)",
            category: .biologic,
            emoji: "💉",
            standardDosages: ["40mg"],
            standardFrequencies: ["Every 2 weeks", "Weekly"],
            commonInstructions: "Inject subcutaneously, rotate injection sites",
            interactions: ["Live vaccines", "Other biologics"],
            sideEffects: ["Injection site reactions", "Increased infection risk", "Headache"]
        ),
        PredefinedMedication(
            name: "Etanercept (Enbrel)",
            category: .biologic,
            emoji: "💉",
            standardDosages: ["25mg", "50mg"],
            standardFrequencies: ["Twice Weekly", "Weekly"],
            commonInstructions: "Store in refrigerator, allow to reach room temperature before injection",
            interactions: ["Live vaccines", "Immunosuppressants"],
            sideEffects: ["Upper respiratory infections", "Injection site reactions"]
        ),
        PredefinedMedication(
            name: "Infliximab (Remicade)",
            category: .biologic,
            emoji: "💉",
            standardDosages: ["3mg/kg", "5mg/kg", "10mg/kg"],
            standardFrequencies: ["Every 8 weeks", "Every 6 weeks"],
            commonInstructions: "IV infusion in clinical setting, premedication may be required",
            interactions: ["Live vaccines", "Other TNF inhibitors"],
            sideEffects: ["Infusion reactions", "Increased infection risk", "Liver problems"]
        ),
        PredefinedMedication(
            name: "Rituximab",
            category: .biologic,
            emoji: "💉",
            standardDosages: ["1000mg"],
            standardFrequencies: ["Every 6 months", "As prescribed"],
            commonInstructions: "IV infusion with premedication, monitor for reactions",
            interactions: ["Live vaccines", "Other immunosuppressants"],
            sideEffects: ["Infusion reactions", "Low blood counts", "Increased infection risk"]
        ),
        
        // MARK: - DMARDs
        PredefinedMedication(
            name: "Methotrexate (MTX)",
            category: .dmard,
            emoji: "💊",
            standardDosages: ["7.5mg", "10mg", "15mg", "20mg", "25mg"],
            standardFrequencies: ["Weekly"],
            commonInstructions: "Take with folic acid, avoid alcohol, monitor liver function",
            interactions: ["NSAIDs", "Trimethoprim", "Proton pump inhibitors"],
            sideEffects: ["Nausea", "Fatigue", "Mouth sores", "Liver toxicity"]
        ),
        PredefinedMedication(
            name: "Sulfasalazine",
            category: .dmard,
            emoji: "💊",
            standardDosages: ["500mg", "1000mg"],
            standardFrequencies: ["Twice Daily", "Three Times Daily"],
            commonInstructions: "Take with food, increase dose gradually",
            interactions: ["Warfarin", "Digoxin"],
            sideEffects: ["Nausea", "Headache", "Rash", "Orange urine"]
        ),
        PredefinedMedication(
            name: "Hydroxychloroquine",
            category: .dmard,
            emoji: "💊",
            standardDosages: ["200mg", "400mg"],
            standardFrequencies: ["Daily", "Twice Daily"],
            commonInstructions: "Take with food, regular eye exams required",
            interactions: ["Digoxin", "Insulin"],
            sideEffects: ["Nausea", "Headache", "Skin rash", "Retinal toxicity (rare)"]
        ),
        PredefinedMedication(
            name: "Leflunomide",
            category: .dmard,
            emoji: "💊",
            standardDosages: ["10mg", "20mg"],
            standardFrequencies: ["Daily"],
            commonInstructions: "Monitor liver function, avoid pregnancy",
            interactions: ["Warfarin", "Rifampin"],
            sideEffects: ["Diarrhea", "Hair loss", "Liver problems", "High blood pressure"]
        ),
        
        // MARK: - NSAIDs
        PredefinedMedication(
            name: "Ibuprofen",
            category: .nsaid,
            emoji: "💊",
            standardDosages: ["200mg", "400mg", "600mg", "800mg"],
            standardFrequencies: ["Three Times Daily", "Four Times Daily", "As Needed"],
            commonInstructions: "Take with food, limit duration of use",
            interactions: ["Blood thinners", "ACE inhibitors", "Lithium"],
            sideEffects: ["Stomach upset", "Heartburn", "Kidney problems", "High blood pressure"]
        ),
        PredefinedMedication(
            name: "Naproxen",
            category: .nsaid,
            emoji: "💊",
            standardDosages: ["220mg", "375mg", "500mg"],
            standardFrequencies: ["Twice Daily", "As Needed"],
            commonInstructions: "Take with food, monitor for GI bleeding",
            interactions: ["Warfarin", "Methotrexate", "Diuretics"],
            sideEffects: ["Stomach pain", "Heartburn", "Dizziness", "Fluid retention"]
        ),
        PredefinedMedication(
            name: "Celecoxib",
            category: .nsaid,
            emoji: "💊",
            standardDosages: ["100mg", "200mg"],
            standardFrequencies: ["Daily", "Twice Daily"],
            commonInstructions: "May be taken with or without food",
            interactions: ["Blood thinners", "ACE inhibitors"],
            sideEffects: ["Headache", "Dizziness", "Stomach upset", "Fluid retention"]
        )
    ]
    
    // MARK: - Helper Methods
    func getMedicationsByCategory(_ category: MedicationCategory) -> [PredefinedMedication] {
        return predefinedMedications.filter { $0.category == category }
    }
    
    func searchMedications(query: String) -> [PredefinedMedication] {
        guard !query.isEmpty else { return predefinedMedications }
        return predefinedMedications.filter { 
            $0.name.localizedCaseInsensitiveContains(query)
        }
    }
    
    func getMedicationByName(_ name: String) -> PredefinedMedication? {
        return predefinedMedications.first { $0.name.localizedCaseInsensitiveContains(name) }
    }
    
    func checkInteractions(medications: [String]) -> [String] {
        var allInteractions: Set<String> = []
        
        for medicationName in medications {
            if let medication = getMedicationByName(medicationName) {
                for interaction in medication.interactions {
                    if medications.contains(where: { $0.localizedCaseInsensitiveContains(interaction) }) {
                        allInteractions.insert("⚠️ \(medicationName) may interact with \(interaction)")
                    }
                }
            }
        }
        
        return Array(allInteractions)
    }
}

// MARK: - Medication Helper Extensions
extension PredefinedMedication {
    var displayName: String {
        return "\(emoji) \(name)"
    }
    
    var categoryBadge: some View {
        Text(category.rawValue)
            .font(.caption2)
            .fontWeight(.semibold)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(category.color.opacity(0.2))
            .foregroundColor(category.color)
            .cornerRadius(8)
    }
}