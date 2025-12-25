//
//  TodayMedicationCard.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData
import UserNotifications

struct TodayMedicationCard: View {
    @Environment(\.managedObjectContext) private var viewContext
    
    let medication: Medication
    @State private var isTaken = false
    
    var body: some View {
        HStack(spacing: 15) {
            // Medication Icon
            ZStack {
                Circle()
                    .fill(isTaken ? Color.green : Color.blue)
                    .frame(width: 50, height: 50)
                
                Image(systemName: isTaken ? "checkmark" : "pills")
                    .foregroundColor(.white)
                    .font(.title2)
            }
            
            // Medication Info
            VStack(alignment: .leading, spacing: 4) {
                Text(medication.name ?? "Unknown Medication")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                
                Text("\(Int(medication.dosage)) \(medication.unit ?? "mg")")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                if let frequency = medication.frequency, !frequency.isEmpty {
                    Text(frequency)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                if let notes = medication.notes, !notes.isEmpty {
                    Text(notes)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
            }
            
            Spacer()
            
            // Take Medication Button
            Button(action: {
                toggleMedicationTaken()
            }) {
                Image(systemName: isTaken ? "checkmark.circle.fill" : "circle")
                    .font(.title2)
                    .foregroundColor(isTaken ? .green : .gray)
            }
        }
        .padding()
        .background(Color(UIColor.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
        .onAppear {
            checkIfTakenToday()
        }
    }
    
    private func toggleMedicationTaken() {
        if isTaken {
            // Remove today's intake
            removeTodayIntake()
        } else {
            // Add today's intake
            addTodayIntake()
        }
        isTaken.toggle()
    }
    
    private func addTodayIntake() {
        let newIntake = MedicationIntake(context: viewContext)
        newIntake.medicationName = medication.name
        newIntake.timestamp = Date()
        newIntake.dosageTaken = medication.dosage

        do {
            try viewContext.save()
        } catch {
            print("Error saving medication intake: \(error)")
        }
    }
    
    private func removeTodayIntake() {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: today)!
        
        let request = NSFetchRequest<MedicationIntake>(entityName: "MedicationIntake")
        request.predicate = NSPredicate(format: "medicationName == %@ AND timestamp >= %@ AND timestamp < %@", medication.name ?? "", today as NSDate, tomorrow as NSDate)
        
        do {
            let intakes = try viewContext.fetch(request)
            for intake in intakes {
                viewContext.delete(intake)
            }
            try viewContext.save()
        } catch {
            print("Error removing medication intake: \(error)")
        }
    }
    
    private func checkIfTakenToday() {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: today)!
        
        let request = NSFetchRequest<MedicationIntake>(entityName: "MedicationIntake")
        request.predicate = NSPredicate(format: "medicationName == %@ AND timestamp >= %@ AND timestamp < %@", medication.name ?? "", today as NSDate, tomorrow as NSDate)
        
        do {
            let intakes = try viewContext.fetch(request)
            isTaken = !intakes.isEmpty
        } catch {
            print("Error checking medication intake: \(error)")
            isTaken = false
        }
    }
}

struct TodayMedicationCard_Previews: PreviewProvider {
    static var previews: some View {
        let context = InflamAIPersistenceController.preview.container.viewContext
        let medication = Medication(context: context)
        medication.name = "Sample Medication"
        medication.dosage = 10.0
        medication.frequency = "Daily"
        medication.unit = "mg"
        
        return TodayMedicationCard(medication: medication)
            .environment(\.managedObjectContext, context)
    }
}