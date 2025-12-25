//
//  MedicationCard.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData

struct MedicationCard: View {
    let medication: Medication
    private let medicationsDB = StandardizedMedicationsDatabase.shared
    
    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()
    
    private var predefinedMedication: PredefinedMedication? {
        medicationsDB.getMedicationByName(medication.name ?? "")
    }
    
    private var medicationEmoji: String {
        predefinedMedication?.emoji ?? "💊"
    }
    
    private var medicationCategory: MedicationCategory? {
        predefinedMedication?.category
    }
    
    var body: some View {
        HStack(spacing: 12) {
            // Medication emoji/icon
            VStack {
                Text(medicationEmoji)
                    .font(.title)
                    .frame(width: 44, height: 44)
                    .background(
                        Group {
                            if let category = medicationCategory {
                                category.color.opacity(0.15)
                            } else {
                                Color.blue.opacity(0.1)
                            }
                        }
                    )
                    .clipShape(Circle())
                
                // Category badge
                if let category = medicationCategory {
                    Text(category.rawValue)
                        .font(.system(size: 8, weight: .bold))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(category.color.opacity(0.2))
                        .foregroundColor(category.color)
                        .cornerRadius(4)
                }
            }
            
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(medication.name ?? "Unknown")
                        .font(.headline)
                        .fontWeight(.semibold)
                        .lineLimit(1)
                    
                    Spacer()
                    
                    // Status indicator
                    Circle()
                        .fill(Color.green)
                        .frame(width: 8, height: 8)
                }
                
                HStack {
                    Text("\(String(medication.dosage)) • \(medication.frequency ?? "")")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                }
                
                if let instructions = medication.notes, !instructions.isEmpty {
                    Text(instructions)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                }
                
                if let nextDose = getNextDoseTime() {
                    HStack {
                        Image(systemName: "clock")
                            .font(.caption2)
                            .foregroundColor(.blue)
                        
                        Text("Next: \(nextDose, formatter: dateFormatter)")
                            .font(.caption)
                            .foregroundColor(.blue)
                            .fontWeight(.medium)
                        
                        Spacer()
                    }
                }
                
                // Show interaction warning if applicable
                if let predefined = predefinedMedication, !predefined.interactions.isEmpty {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption2)
                            .foregroundColor(.orange)
                        
                        Text("Check interactions")
                            .font(.caption2)
                            .foregroundColor(.orange)
                            .fontWeight(.medium)
                        
                        Spacer()
                    }
                }
            }
            
            Image(systemName: "chevron.right")
                .foregroundColor(.secondary)
                .font(.caption)
                .fontWeight(.semibold)
        }
        .padding(16)
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.08), radius: 4, x: 0, y: 2)
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(medicationCategory?.color.opacity(0.2) ?? Color.clear, lineWidth: 1)
        )
    }
    
    private func getNextDoseTime() -> Date? {
        // Simple logic to calculate next dose based on frequency
        guard let frequency = medication.frequency else { return nil }
        
        let calendar = Calendar.current
        let now = Date()
        
        switch frequency.lowercased() {
        case "daily", "once daily":
            return calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: now))
        case "twice daily", "2x daily":
            let morning = calendar.date(bySettingHour: 8, minute: 0, second: 0, of: now)
            let evening = calendar.date(bySettingHour: 20, minute: 0, second: 0, of: now)

            if let morning = morning, now < morning {
                return morning
            } else if let evening = evening, now < evening {
                return evening
            } else {
                // Calculate next morning dose (tomorrow at 8 AM)
                guard let tomorrowMorning = calendar.date(bySettingHour: 8, minute: 0, second: 0, of: now),
                      let nextMorning = calendar.date(byAdding: .day, value: 1, to: tomorrowMorning) else {
                    return nil
                }
                return nextMorning
            }
        case "three times daily", "3x daily":
            let morning = calendar.date(bySettingHour: 8, minute: 0, second: 0, of: now)
            let afternoon = calendar.date(bySettingHour: 14, minute: 0, second: 0, of: now)
            let evening = calendar.date(bySettingHour: 20, minute: 0, second: 0, of: now)

            if let morning = morning, now < morning {
                return morning
            } else if let afternoon = afternoon, now < afternoon {
                return afternoon
            } else if let evening = evening, now < evening {
                return evening
            } else {
                // Calculate next morning dose (tomorrow at 8 AM)
                guard let tomorrowMorning = calendar.date(bySettingHour: 8, minute: 0, second: 0, of: now),
                      let nextMorning = calendar.date(byAdding: .day, value: 1, to: tomorrowMorning) else {
                    return nil
                }
                return nextMorning
            }
        default:
            return nil
        }
    }
}

#if DEBUG
struct MedicationCard_Previews: PreviewProvider {
    static var previews: some View {
        // Create a mock medication for preview using Core Data context
        let context = InflamAIPersistenceController.preview.container.viewContext
        let medication = Medication(context: context)
        medication.name = "Sample Medication"
        medication.dosage = 10.0
        medication.frequency = "Daily"
        medication.unit = "mg"
        medication.notes = "Take with food"
        
        return MedicationCard(medication: medication)
            .padding()
            .previewLayout(.sizeThatFits)
            .environment(\.managedObjectContext, context)
    }
}
#endif