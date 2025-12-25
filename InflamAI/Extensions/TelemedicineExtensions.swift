//
//  TelemedicineExtensions.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI

// MARK: - MedicalSpecialty Extensions

extension MedicalSpecialty {
    var icon: String {
        switch self {
        case .rheumatology:
            return "figure.walk"
        case .cardiology:
            return "heart.fill"
        case .dermatology:
            return "hand.raised.fill"
        case .endocrinology:
            return "drop.fill"
        case .gastroenterology:
            return "stomach.fill"
        case .neurology:
            return "brain.head.profile"
        case .orthopedics:
            return "figure.strengthtraining.traditional"
        case .psychiatry:
            return "brain.fill"
        case .pulmonology:
            return "lungs.fill"
        case .urology:
            return "drop.circle.fill"
        case .oncology:
            return "cross.case.fill"
        case .pediatrics:
            return "figure.and.child.holdinghands"
        case .gynecology:
            return "person.fill"
        case .ophthalmology:
            return "eye.fill"
        case .otolaryngology:
            return "ear.fill"
        case .generalMedicine:
            return "stethoscope"
        }
    }
    
    var color: Color {
        switch self {
        case .rheumatology:
            return .orange
        case .cardiology:
            return .red
        case .dermatology:
            return .pink
        case .endocrinology:
            return .blue
        case .gastroenterology:
            return .green
        case .neurology:
            return .purple
        case .orthopedics:
            return .brown
        case .psychiatry:
            return .indigo
        case .pulmonology:
            return .cyan
        case .urology:
            return .teal
        case .oncology:
            return .gray
        case .pediatrics:
            return .yellow
        case .gynecology:
            return .mint
        case .ophthalmology:
            return .blue
        case .otolaryngology:
            return .orange
        case .generalMedicine:
            return .primary
        }
    }
}

// MARK: - Consultation Type Extensions

extension Consultation.ConsultationType {
    var icon: String {
        switch self {
        case .videoCall:
            return "video.fill"
        case .audioCall:
            return "phone.fill"
        case .chat:
            return "message.fill"
        case .inPerson:
            return "person.2.fill"
        }
    }
    
    var color: Color {
        switch self {
        case .videoCall:
            return .blue
        case .audioCall:
            return .green
        case .chat:
            return .orange
        case .inPerson:
            return .purple
        }
    }
    
    var description: String {
        switch self {
        case .videoCall:
            return "Face-to-face video consultation"
        case .audioCall:
            return "Voice-only consultation"
        case .chat:
            return "Text-based consultation"
        case .inPerson:
            return "In-person office visit"
        }
    }
}

// MARK: - Consultation Status Extensions

extension Consultation.ConsultationStatus {
    var color: String {
        switch self {
        case .scheduled:
            return "blue"
        case .inProgress:
            return "green"
        case .completed:
            return "gray"
        case .cancelled:
            return "red"
        case .noShow:
            return "orange"
        case .rescheduled:
            return "yellow"
        }
    }
    
    var displayName: String {
        switch self {
        case .scheduled:
            return "Scheduled"
        case .inProgress:
            return "In Progress"
        case .completed:
            return "Completed"
        case .cancelled:
            return "Cancelled"
        case .noShow:
            return "No Show"
        case .rescheduled:
            return "Rescheduled"
        }
    }
    
    var systemImage: String {
        switch self {
        case .scheduled:
            return "calendar"
        case .inProgress:
            return "video.circle.fill"
        case .completed:
            return "checkmark.circle.fill"
        case .cancelled:
            return "xmark.circle.fill"
        case .noShow:
            return "exclamationmark.triangle.fill"
        case .rescheduled:
            return "arrow.clockwise.circle.fill"
        }
    }
}

// MARK: - Urgency Level Extensions

extension Consultation.UrgencyLevel {
    var color: String {
        switch self {
        case .low:
            return "green"
        case .medium:
            return "yellow"
        case .high:
            return "orange"
        case .urgent:
            return "red"
        }
    }
    
    var displayName: String {
        switch self {
        case .low:
            return "Low Priority"
        case .medium:
            return "Medium Priority"
        case .high:
            return "High Priority"
        case .urgent:
            return "Urgent"
        }
    }
    
    var description: String {
        switch self {
        case .low:
            return "Routine consultation, can wait"
        case .medium:
            return "Standard consultation"
        case .high:
            return "Important, schedule soon"
        case .urgent:
            return "Needs immediate attention"
        }
    }
    
    var systemImage: String {
        switch self {
        case .low:
            return "circle.fill"
        case .medium:
            return "circle.lefthalf.filled"
        case .high:
            return "exclamationmark.circle.fill"
        case .urgent:
            return "exclamationmark.triangle.fill"
        }
    }
}

// MARK: - Payment Status Extensions

extension Consultation.PaymentStatus {
    var color: Color {
        switch self {
        case .pending:
            return .orange
        case .paid:
            return .green
        case .failed:
            return .red
        case .refunded:
            return .blue
        case .partiallyPaid:
            return .yellow
        }
    }
    
    var displayName: String {
        switch self {
        case .pending:
            return "Payment Pending"
        case .paid:
            return "Paid"
        case .failed:
            return "Payment Failed"
        case .refunded:
            return "Refunded"
        case .partiallyPaid:
            return "Partially Paid"
        }
    }
    
    var systemImage: String {
        switch self {
        case .pending:
            return "clock.fill"
        case .paid:
            return "checkmark.circle.fill"
        case .failed:
            return "xmark.circle.fill"
        case .refunded:
            return "arrow.counterclockwise.circle.fill"
        case .partiallyPaid:
            return "circle.lefthalf.filled"
        }
    }
}

// MARK: - Connection Quality Extensions

extension VideoCallManager.ConnectionQuality {
    var displayName: String {
        switch self {
        case .excellent:
            return "Excellent"
        case .good:
            return "Good"
        case .fair:
            return "Fair"
        case .poor:
            return "Poor"
        case .disconnected:
            return "Disconnected"
        }
    }
    
    var color: String {
        switch self {
        case .excellent:
            return "green"
        case .good:
            return "blue"
        case .fair:
            return "yellow"
        case .poor:
            return "orange"
        case .disconnected:
            return "red"
        }
    }
    
    var systemImage: String {
        switch self {
        case .excellent:
            return "wifi"
        case .good:
            return "wifi"
        case .fair:
            return "wifi"
        case .poor:
            return "wifi.exclamationmark"
        case .disconnected:
            return "wifi.slash"
        }
    }
}

// MARK: - Vital Sign Type Extensions

extension HealthDataSummary.VitalSignReading.VitalSignType {
    var displayName: String {
        switch self {
        case .heartRate:
            return "Heart Rate"
        case .bloodPressure:
            return "Blood Pressure"
        case .temperature:
            return "Temperature"
        case .oxygenSaturation:
            return "Oxygen Saturation"
        case .respiratoryRate:
            return "Respiratory Rate"
        case .weight:
            return "Weight"
        case .height:
            return "Height"
        case .bmi:
            return "BMI"
        }
    }
    
    var unit: String {
        switch self {
        case .heartRate:
            return "bpm"
        case .bloodPressure:
            return "mmHg"
        case .temperature:
            return "°F"
        case .oxygenSaturation:
            return "%"
        case .respiratoryRate:
            return "breaths/min"
        case .weight:
            return "lbs"
        case .height:
            return "in"
        case .bmi:
            return ""
        }
    }
    
    var systemImage: String {
        switch self {
        case .heartRate:
            return "heart.fill"
        case .bloodPressure:
            return "drop.fill"
        case .temperature:
            return "thermometer"
        case .oxygenSaturation:
            return "lungs.fill"
        case .respiratoryRate:
            return "wind"
        case .weight:
            return "scalemass.fill"
        case .height:
            return "ruler.fill"
        case .bmi:
            return "person.fill"
        }
    }
    
    var color: Color {
        switch self {
        case .heartRate:
            return .red
        case .bloodPressure:
            return .blue
        case .temperature:
            return .orange
        case .oxygenSaturation:
            return .cyan
        case .respiratoryRate:
            return .mint
        case .weight:
            return .purple
        case .height:
            return .brown
        case .bmi:
            return .indigo
        }
    }
    
    func isNormalRange(_ value: Double) -> Bool {
        switch self {
        case .heartRate:
            return value >= 60 && value <= 100
        case .bloodPressure:
            return value <= 120 // Systolic
        case .temperature:
            return value >= 97.0 && value <= 99.5
        case .oxygenSaturation:
            return value >= 95
        case .respiratoryRate:
            return value >= 12 && value <= 20
        case .weight:
            return true // Varies by person
        case .height:
            return true // Varies by person
        case .bmi:
            return value >= 18.5 && value <= 24.9
        }
    }
}

// MARK: - Prescription Extensions

extension Prescription {
    var statusColor: Color {
        if Date() > prescribedDate.addingTimeInterval(TimeInterval(refills * 30 * 24 * 60 * 60)) {
            return .red // Expired
        } else if refills == 0 {
            return .orange // No refills left
        } else {
            return .green // Active
        }
    }
    
    var statusText: String {
        if Date() > prescribedDate.addingTimeInterval(TimeInterval(refills * 30 * 24 * 60 * 60)) {
            return "Expired"
        } else if refills == 0 {
            return "No Refills"
        } else {
            return "Active"
        }
    }
    
    var isExpired: Bool {
        Date() > prescribedDate.addingTimeInterval(TimeInterval(refills * 30 * 24 * 60 * 60))
    }
    
    var needsRefill: Bool {
        refills <= 1 && !isExpired
    }
}

// MARK: - Provider Availability Extensions

extension ProviderAvailability {
    var displayTimeRange: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return "\(formatter.string(from: startTime)) - \(formatter.string(from: endTime))"
    }
    
    var dayName: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEEE"
        return formatter.string(from: date)
    }
    
    var isToday: Bool {
        Calendar.current.isDateInToday(date)
    }
    
    var isTomorrow: Bool {
        Calendar.current.isDateInTomorrow(date)
    }
    
    var displayDate: String {
        if isToday {
            return "Today"
        } else if isTomorrow {
            return "Tomorrow"
        } else {
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            return formatter.string(from: date)
        }
    }
}

// MARK: - Healthcare Provider Extensions

extension HealthcareProvider {
    var displayRating: String {
        return String(format: "%.1f", rating)
    }
    
    var ratingStars: String {
        let fullStars = Int(rating)
        let hasHalfStar = rating - Double(fullStars) >= 0.5
        let emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0)
        
        return String(repeating: "★", count: fullStars) +
               (hasHalfStar ? "☆" : "") +
               String(repeating: "☆", count: emptyStars)
    }
    
    var experienceText: String {
        return "\(experience) year\(experience == 1 ? "" : "s") of experience"
    }
    
    var availabilityStatus: String {
        if isOnline {
            return "Available now"
        } else {
            return "Offline"
        }
    }
    
    var nextAvailableSlot: Date? {
        let now = Date()
        return availability
            .filter { $0.date >= now && $0.isAvailable }
            .sorted { $0.date < $1.date }
            .first?.date
    }
    
    var nextAvailableText: String {
        guard let nextSlot = nextAvailableSlot else {
            return "No upcoming availability"
        }
        
        let formatter = DateFormatter()
        if Calendar.current.isDateInToday(nextSlot) {
            formatter.timeStyle = .short
            return "Next available today at \(formatter.string(from: nextSlot))"
        } else if Calendar.current.isDateInTomorrow(nextSlot) {
            formatter.timeStyle = .short
            return "Next available tomorrow at \(formatter.string(from: nextSlot))"
        } else {
            formatter.dateStyle = .medium
            formatter.timeStyle = .short
            return "Next available \(formatter.string(from: nextSlot))"
        }
    }
}

// MARK: - Consultation Extensions

extension Consultation {
    var timeUntilAppointment: TimeInterval {
        scheduledDate.timeIntervalSinceNow
    }
    
    var canJoin: Bool {
        status == .scheduled && timeUntilAppointment <= 900 && timeUntilAppointment >= -300 // 15 min before to 5 min after
    }
    
    var isUpcoming: Bool {
        status == .scheduled && scheduledDate > Date()
    }
    
    var isPast: Bool {
        status == .completed || scheduledDate < Date()
    }
    
    var displayDuration: String {
        let hours = Int(duration) / 3600
        let minutes = (Int(duration) % 3600) / 60
        
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else {
            return "\(minutes) minutes"
        }
    }
    
    var timeUntilText: String {
        let interval = timeUntilAppointment
        
        if interval < 0 {
            return "Started \(Int(-interval / 60)) minutes ago"
        } else if interval < 3600 {
            return "Starts in \(Int(interval / 60)) minutes"
        } else if interval < 86400 {
            return "Starts in \(Int(interval / 3600)) hours"
        } else {
            return "Starts in \(Int(interval / 86400)) days"
        }
    }
}

// MARK: - Consultation Fee Extensions

extension ConsultationFee {
    var displayAmount: String {
        return String(format: "$%.2f", amount)
    }
    
    var displayInsuranceCopay: String {
        if let copay = insuranceCopay {
            return String(format: "$%.2f", copay)
        } else {
            return "N/A"
        }
    }
    
    var savings: Double? {
        guard let copay = insuranceCopay else { return nil }
        return amount - copay
    }
    
    var displaySavings: String {
        if let savings = savings {
            return String(format: "Save $%.2f", savings)
        } else {
            return ""
        }
    }
}

// MARK: - Consultation Rating Extensions

extension ConsultationRating {
    var displayRating: String {
        return String(format: "%.1f", rating)
    }
    
    var starRating: String {
        let fullStars = Int(rating)
        let hasHalfStar = rating - Double(fullStars) >= 0.5
        let emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0)
        
        return String(repeating: "★", count: fullStars) +
               (hasHalfStar ? "☆" : "") +
               String(repeating: "☆", count: emptyStars)
    }
    
    var qualityDescription: String {
        switch rating {
        case 4.5...5.0:
            return "Excellent"
        case 3.5..<4.5:
            return "Good"
        case 2.5..<3.5:
            return "Average"
        case 1.5..<2.5:
            return "Poor"
        default:
            return "Very Poor"
        }
    }
}

// MARK: - Date Extensions for Telemedicine

extension Date {
    var isInNext15Minutes: Bool {
        let interval = timeIntervalSinceNow
        return interval > 0 && interval <= 900 // 15 minutes
    }
    
    var isInPast5Minutes: Bool {
        let interval = timeIntervalSinceNow
        return interval < 0 && interval >= -300 // 5 minutes ago
    }
    
    var canJoinCall: Bool {
        isInNext15Minutes || isInPast5Minutes
    }
    
    func timeSlots(duration: TimeInterval = 1800) -> [Date] {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: self)
        let startHour = 9 // 9 AM
        let endHour = 17 // 5 PM
        
        var slots: [Date] = []
        
        for hour in startHour..<endHour {
            for minute in stride(from: 0, to: 60, by: Int(duration / 60)) {
                if let slot = calendar.date(bySettingHour: hour, minute: minute, second: 0, of: startOfDay) {
                    slots.append(slot)
                }
            }
        }
        
        return slots.filter { $0 > Date() } // Only future slots
    }
}

// MARK: - String Extensions for Telemedicine

extension String {
    var isValidPhoneNumber: Bool {
        let phoneRegex = "^[\\+]?[1-9]?[0-9]{7,14}$"
        let phoneTest = NSPredicate(format: "SELF MATCHES %@", phoneRegex)
        return phoneTest.evaluate(with: self)
    }
    
    var isValidEmail: Bool {
        let emailRegex = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,64}"
        let emailTest = NSPredicate(format: "SELF MATCHES %@", emailRegex)
        return emailTest.evaluate(with: self)
    }
    
    var formattedPhoneNumber: String {
        let cleanNumber = components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
        
        if cleanNumber.count == 10 {
            let areaCode = String(cleanNumber.prefix(3))
            let firstThree = String(cleanNumber.dropFirst(3).prefix(3))
            let lastFour = String(cleanNumber.suffix(4))
            return "(\(areaCode)) \(firstThree)-\(lastFour)"
        }
        
        return self
    }
}

// MARK: - Color Extensions for Telemedicine

extension Color {
    static let telemedicineBlue = Color(red: 0.0, green: 0.48, blue: 0.8)
    static let telemedicineGreen = Color(red: 0.0, green: 0.7, blue: 0.4)
    static let telemedicineOrange = Color(red: 1.0, green: 0.6, blue: 0.0)
    static let telemedicineRed = Color(red: 0.9, green: 0.2, blue: 0.2)
    static let telemedicinePurple = Color(red: 0.6, green: 0.2, blue: 0.8)
    
    init(_ colorName: String) {
        switch colorName.lowercased() {
        case "blue":
            self = .blue
        case "green":
            self = .green
        case "red":
            self = .red
        case "orange":
            self = .orange
        case "yellow":
            self = .yellow
        case "purple":
            self = .purple
        case "pink":
            self = .pink
        case "gray", "grey":
            self = .gray
        default:
            self = .primary
        }
    }
}