//
//  MedicationView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData
import UserNotifications

// Import Core Data for Medication entity

struct MedicationView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \Medication.name, ascending: true)],
        animation: .default)
    private var medications: FetchedResults<Medication>
    
    @State private var showingAddMedication = false
    @State private var showingNotificationAlert = false
    @State private var searchText = ""
    @State private var showingInteractionWarnings = false
    @State private var selectedMedication: Medication?
    @State private var showingMedicationDetail = false
    @State private var todayIntakes: [NSManagedObject] = []
    
    // Simplified medication management
    
    private var filteredMedications: [Medication] {
        if searchText.isEmpty {
            return Array(medications)
        } else {
            return medications.filter { medication in
                (medication.name ?? "").localizedCaseInsensitiveContains(searchText) ||
                String(medication.dosage).localizedCaseInsensitiveContains(searchText) ||
                (medication.frequency ?? "").localizedCaseInsensitiveContains(searchText)
            }
        }
    }
    
    private var medicationInteractions: [String] {
        let medicationsDB = StandardizedMedicationsDatabase.shared
        let medicationNames = medications.compactMap { $0.name }
        return medicationsDB.checkInteractions(medications: medicationNames)
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                searchBarView
                interactionWarningsView
                todayScheduleView
                medicationListView
            }
            .navigationTitle("Medications")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showingAddMedication = true
                    } label: {
                        Image(systemName: "plus.circle.fill")
                            .font(.title2)
                    }
                }
            }
            .sheet(isPresented: $showingAddMedication) {
                addMedicationSheet
            }
            .sheet(isPresented: $showingMedicationDetail) {
                medicationDetailSheet
            }
        }
    }
    
    private var searchBarView: some View {
        HStack {
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search medications...", text: $searchText)
                    .textFieldStyle(PlainTextFieldStyle())
            }
            .padding(8)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
            
            if !searchText.isEmpty {
                Button("Cancel") {
                    searchText = ""
                    hideKeyboard()
                }
                .foregroundColor(.blue)
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }
    
    private var interactionWarningsView: some View {
        Group {
            if !medicationInteractions.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Medication Interactions")
                            .font(.headline)
                            .fontWeight(.semibold)
                        Spacer()
                        Button(showingInteractionWarnings ? "Hide" : "Show") {
                            showingInteractionWarnings.toggle()
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                    
                    if showingInteractionWarnings {
                        ForEach(medicationInteractions, id: \.self) { interaction in
                            Text(interaction)
                                .font(.caption)
                                .foregroundColor(.orange)
                                .padding(.leading, 20)
                        }
                    }
                }
                .padding()
                .background(Color.orange.opacity(0.1))
                .cornerRadius(12)
                .padding(.horizontal)
                .padding(.top, 8)
            }
        }
    }
    
    private var todayScheduleView: some View {
         VStack {
             if !medications.isEmpty && searchText.isEmpty {
                 todayScheduleContent
             }
         }
     }
     
     private var todayScheduleContent: some View {
         VStack(alignment: .leading, spacing: 12) {
             Text("Today's Schedule")
                 .font(.title2)
                 .fontWeight(.bold)
                 .padding(.horizontal)
             
             ScrollView(.horizontal, showsIndicators: false) {
                 HStack(spacing: 12) {
                     ForEach(medications, id: \.id) { medication in
                         medicationScheduleCard(medication: medication)
                     }
                 }
                 .padding(.horizontal)
             }
         }
         .padding(.vertical)
         .background(Color.gray.opacity(0.1))
     }
     
     private func medicationScheduleCard(medication: Medication) -> some View {
         let medicationsDB = StandardizedMedicationsDatabase.shared
         let predefinedMed = medicationsDB.getMedicationByName(medication.name ?? "")
         
         return VStack(spacing: 8) {
             Text(predefinedMed?.emoji ?? "💊")
                 .font(.title)
             Text(medication.name ?? "Unknown")
                 .font(.caption)
                 .fontWeight(.medium)
                 .multilineTextAlignment(.center)
             Text("\(String(medication.dosage)) \(medication.unit ?? "")")
                 .font(.caption2)
                 .foregroundColor(.secondary)
         }
         .frame(width: 80, height: 100)
         .background((predefinedMed?.category.color ?? Color.blue).opacity(0.1))
         .cornerRadius(12)
     }
    
    private var medicationListView: some View {
        List {
            Section(header: Text(searchText.isEmpty ? "All Medications" : "Search Results")) {
                if filteredMedications.isEmpty {
                    emptyStateView
                } else {
                    ForEach(filteredMedications, id: \.id) { medication in
                        medicationRowView(medication: medication)
                    }
                    .onDelete(perform: deleteMedications)
                }
            }
        }
        // Using default list style for macOS compatibility
    }
    
    private var emptyStateView: some View {
        Group {
            if searchText.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "pills")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("No medications added yet")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    Text("Tap the + button to add your first medication")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.vertical, 32)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 32))
                        .foregroundColor(.secondary)
                    Text("No medications found")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    Text("Try adjusting your search terms")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 24)
            }
        }
    }
    
    private func medicationRowView(medication: Medication) -> some View {
         let medicationsDB = StandardizedMedicationsDatabase.shared
         let predefinedMed = medicationsDB.getMedicationByName(medication.name ?? "")
        
        return Button(action: {
            selectedMedication = medication
            showingMedicationDetail = true
        }) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(medication.name ?? "Unknown")
                            .font(.headline)
                            .foregroundColor(.primary)
                        
                        if let predefinedMed = predefinedMed {
                            predefinedMed.categoryBadge
                        }
                    }
                    
                    Text("\(String(medication.dosage)) \(medication.unit ?? "")")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Text(medication.frequency ?? "")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Text(predefinedMed?.emoji ?? "💊")
                    .font(.title2)
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(PlainButtonStyle())
    }
    
    private var addMedicationSheet: some View {
        Text("Add Medication Coming Soon")
            .environment(\.managedObjectContext, viewContext)
    }
    
    private var medicationDetailSheet: some View {
        Group {
            if selectedMedication != nil {
                NavigationView {
                    Text("Medication Detail View")
                        .navigationTitle("Medication Details")
                        .toolbar {
                            ToolbarItem(placement: .cancellationAction) {
                                Button("Done") {
                                    showingMedicationDetail = false
                                }
                            }
                        }
                }
            }
        }
    }
    
    private func hideKeyboard() {
        // SwiftUI handles keyboard dismissal automatically
    }
    
    private func deleteMedications(offsets: IndexSet) {
         withAnimation {
             // Remove from the filtered list for now
             // In a real app, this would remove from Core Data
             print("Delete medications at indices: \(offsets)")
         }
     }
}

#if DEBUG
struct MedicationView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            MedicationView()
        }
    }
}
#endif