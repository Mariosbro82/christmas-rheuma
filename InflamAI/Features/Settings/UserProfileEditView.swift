//
//  UserProfileEditView.swift
//  InflamAI
//
//  User profile data entry for demographics required by ML features
//  Collects: gender, height, weight, BMI, smoking status, date of birth, diagnosis date
//

import SwiftUI
import CoreData

struct UserProfileEditView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss

    @StateObject private var viewModel: UserProfileEditViewModel

    // Callback for when profile is saved
    var onSave: (() -> Void)?

    init(onSave: (() -> Void)? = nil) {
        self.onSave = onSave
        _viewModel = StateObject(wrappedValue: UserProfileEditViewModel())
    }

    var body: some View {
        NavigationView {
            Form {
                // Personal Information Section
                Section {
                    TextField("Name (optional)", text: $viewModel.name)

                    DatePicker("Date of Birth",
                               selection: $viewModel.dateOfBirth,
                               in: ...Date(),
                               displayedComponents: .date)

                    Picker("Gender", selection: $viewModel.gender) {
                        Text("Select...").tag("")
                        Text("Male").tag("male")
                        Text("Female").tag("female")
                        Text("Other").tag("other")
                        Text("Prefer not to say").tag("unknown")
                    }
                } header: {
                    Text("Personal Information")
                } footer: {
                    Text("This information helps personalize your health insights.")
                }

                // Body Measurements Section
                Section {
                    HStack {
                        Text("Height")
                        Spacer()
                        TextField("cm", value: $viewModel.heightCm, format: .number)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                        Text("cm")
                            .foregroundColor(.secondary)
                    }

                    HStack {
                        Text("Weight")
                        Spacer()
                        TextField("kg", value: $viewModel.weightKg, format: .number)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                        Text("kg")
                            .foregroundColor(.secondary)
                    }

                    if viewModel.calculatedBMI > 0 {
                        HStack {
                            Text("BMI")
                            Spacer()
                            Text(String(format: "%.1f", viewModel.calculatedBMI))
                                .foregroundColor(viewModel.bmiColor)
                                .fontWeight(.medium)
                            Text(viewModel.bmiCategory)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                } header: {
                    Text("Body Measurements")
                } footer: {
                    Text("BMI helps personalize your data visualizations.")
                }

                // AS-Specific Information Section
                Section {
                    DatePicker("AS Diagnosis Date",
                               selection: $viewModel.diagnosisDate,
                               in: ...Date(),
                               displayedComponents: .date)

                    Toggle("HLA-B27 Positive", isOn: $viewModel.hlaB27Positive)

                    Toggle("Previous Biologic Treatment", isOn: $viewModel.biologicExperienced)
                } header: {
                    Text("AS Information")
                } footer: {
                    Text("Disease duration and HLA-B27 status help personalize your experience.")
                }

                // Lifestyle Section
                Section {
                    Picker("Smoking Status", selection: $viewModel.smokingStatus) {
                        Text("Never smoked").tag("never")
                        Text("Former smoker").tag("former")
                        Text("Current smoker").tag("current")
                    }
                } header: {
                    Text("Lifestyle")
                } footer: {
                    Text("Smoking affects disease progression and treatment response.")
                }

                // Healthcare Providers Section (Optional)
                Section {
                    TextField("Primary Physician", text: $viewModel.primaryPhysicianName)
                    TextField("Rheumatologist", text: $viewModel.rheumatologistName)
                } header: {
                    Text("Healthcare Providers (Optional)")
                }

                // Data Quality Indicator
                Section {
                    HStack {
                        Image(systemName: "chart.bar.fill")
                            .foregroundColor(.blue)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Profile Completeness")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            Text("\(viewModel.completenessPercentage)% complete")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        ProgressView(value: Double(viewModel.completenessPercentage) / 100.0)
                            .frame(width: 60)
                    }

                    if viewModel.completenessPercentage < 100 {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Missing fields:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            ForEach(viewModel.missingFields, id: \.self) { field in
                                HStack(spacing: 4) {
                                    Image(systemName: "circle")
                                        .font(.system(size: 6))
                                    Text(field)
                                }
                                .font(.caption)
                                .foregroundColor(.orange)
                            }
                        }
                    }
                } header: {
                    Text("ML Feature Coverage")
                } footer: {
                    Text("Complete profile data improves personalization.")
                }
            }
            .navigationTitle("Edit Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        viewModel.saveProfile()
                        onSave?()
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
}

// MARK: - UserProfileEditViewModel is now in Core/ViewModels/UserProfileEditViewModel.swift
// This avoids duplicate class definitions and ensures visibility across OnboardingFlow, SettingsView, etc.

// MARK: - Compact Profile Setup for Onboarding

struct ProfileSetupOnboardingPage: View {
    @ObservedObject var viewModel: UserProfileEditViewModel

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 12) {
                    Image(systemName: "person.crop.circle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.blue)

                    Text("Tell Us About You")
                        .font(.system(size: 28, weight: .bold))

                    Text("This helps personalize your experience and data visualizations")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 40)

                // Essential Fields
                VStack(spacing: 16) {
                    // Gender
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Gender")
                            .font(.headline)

                        Picker("Gender", selection: $viewModel.gender) {
                            Text("Select...").tag("")
                            Text("Male").tag("male")
                            Text("Female").tag("female")
                            Text("Other").tag("other")
                        }
                        .pickerStyle(.segmented)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Height & Weight
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Height")
                                .font(.headline)
                            HStack {
                                TextField("170", value: $viewModel.heightCm, format: .number)
                                    .keyboardType(.decimalPad)
                                    .textFieldStyle(.roundedBorder)
                                Text("cm")
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Weight")
                                .font(.headline)
                            HStack {
                                TextField("70", value: $viewModel.weightKg, format: .number)
                                    .keyboardType(.decimalPad)
                                    .textFieldStyle(.roundedBorder)
                                Text("kg")
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }

                    // BMI Display
                    if viewModel.calculatedBMI > 0 {
                        HStack {
                            Text("Your BMI:")
                                .foregroundColor(.secondary)
                            Text(String(format: "%.1f", viewModel.calculatedBMI))
                                .fontWeight(.bold)
                                .foregroundColor(viewModel.bmiColor)
                            Text("(\(viewModel.bmiCategory))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(viewModel.bmiColor.opacity(0.1))
                        .cornerRadius(12)
                    }

                    // Date of Birth
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Date of Birth")
                            .font(.headline)
                        DatePicker("", selection: $viewModel.dateOfBirth, in: ...Date(), displayedComponents: .date)
                            .datePickerStyle(.compact)
                            .labelsHidden()
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Smoking Status
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Smoking Status")
                            .font(.headline)
                        Picker("Smoking", selection: $viewModel.smokingStatus) {
                            Text("Never").tag("never")
                            Text("Former").tag("former")
                            Text("Current").tag("current")
                        }
                        .pickerStyle(.segmented)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Diagnosis Date
                    VStack(alignment: .leading, spacing: 8) {
                        Text("AS Diagnosis Date")
                            .font(.headline)
                        DatePicker("", selection: $viewModel.diagnosisDate, in: ...Date(), displayedComponents: .date)
                            .datePickerStyle(.compact)
                            .labelsHidden()
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // HLA-B27 Status
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("HLA-B27 Positive?")
                                .font(.headline)
                            Text("Common genetic marker for AS")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Toggle("", isOn: $viewModel.hlaB27Positive)
                            .labelsHidden()
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }

                // Privacy Note
                HStack(spacing: 8) {
                    Image(systemName: "lock.shield.fill")
                        .foregroundColor(.green)
                    Text("All data stays on your device. Never shared.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()

                Spacer(minLength: 120)
            }
            .padding(.horizontal, 24)
        }
    }
}

// MARK: - Preview

struct UserProfileEditView_Previews: PreviewProvider {
    static var previews: some View {
        UserProfileEditView()
    }
}
