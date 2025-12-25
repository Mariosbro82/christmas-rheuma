//
//  PDFExportService.swift
//  InflamAI
//
//  Production-grade 7-page clinical PDF report for rheumatologists
//  Includes executive summary, clinical assessments, functional capacity,
//  medication response, patient-reported outcomes, charts, heatmaps, and statistics
//

import Foundation
import PDFKit
import UIKit
import CoreData
import SwiftUI

/// Clinician-grade PDF report generator
class PDFExportService {

    // MARK: - Constants

    private let pageSize = CGRect(x: 0, y: 0, width: 612, height: 792) // Letter size
    private let margin: CGFloat = 50
    private let headerHeight: CGFloat = 100
    private let footerHeight: CGFloat = 40

    // MARK: - Public API

    /// Generate comprehensive PDF report with assessments
    /// - Parameters:
    ///   - patientName: Patient's name for the report
    ///   - diagnosis: Primary diagnosis (defaults to Ankylosing Spondylitis)
    ///   - dateRange: Date range for the report data
    ///   - logs: Array of symptom logs
    ///   - flareEvents: Array of flare events
    ///   - medications: Array of medications
    ///   - assessmentResponses: Array of questionnaire responses
    ///   - userProfile: Optional user profile for executive summary
    ///   - exerciseSessions: Array of exercise sessions for functional capacity
    func generateReport(
        patientName: String?,
        diagnosis: String = "Ankylosing Spondylitis",
        dateRange: (start: Date, end: Date),
        logs: [SymptomLog],
        flareEvents: [FlareEvent],
        medications: [Medication],
        assessmentResponses: [QuestionnaireResponse] = [],
        userProfile: UserProfile? = nil,
        exerciseSessions: [ExerciseSession] = []
    ) -> Data {
        let renderer = UIGraphicsPDFRenderer(bounds: pageSize)
        let totalPages = calculateTotalPages(assessmentResponses: assessmentResponses)

        return renderer.pdfData { context in
            // Page 1: Executive Summary (NEW)
            context.beginPage()
            drawExecutiveSummaryPage(
                context: context,
                patientName: patientName,
                diagnosis: diagnosis,
                dateRange: dateRange,
                logs: logs,
                flareEvents: flareEvents,
                medications: medications,
                userProfile: userProfile,
                totalPages: totalPages
            )

            // Page 2: Clinical Assessment (NEW)
            context.beginPage()
            drawClinicalAssessmentPage(
                context: context,
                logs: logs,
                dateRange: dateRange,
                totalPages: totalPages
            )

            // Page 3: Functional Capacity (NEW)
            context.beginPage()
            drawFunctionalCapacityPage(
                context: context,
                logs: logs,
                exerciseSessions: exerciseSessions,
                totalPages: totalPages
            )

            // Page 4: Medication Response (NEW)
            context.beginPage()
            drawMedicationResponsePage(
                context: context,
                medications: medications,
                logs: logs,
                totalPages: totalPages
            )

            // Page 5: Patient-Reported Outcomes (NEW)
            context.beginPage()
            drawPatientReportedOutcomesPage(
                context: context,
                logs: logs,
                assessmentResponses: assessmentResponses,
                totalPages: totalPages
            )

            // Page 6: BASDAI Trends & Pain Heatmap (existing)
            context.beginPage()
            drawChartsPage(context: context, logs: logs, dateRange: dateRange, pageNumber: 6, totalPages: totalPages)

            // Page 7: Flare Events & Medication List (existing)
            context.beginPage()
            drawInsightsPage(context: context, flareEvents: flareEvents, medications: medications, logs: logs, pageNumber: 7, totalPages: totalPages)

            // Additional pages for Assessment Details (existing)
            if !assessmentResponses.isEmpty {
                drawAssessmentPages(context: context, assessmentResponses: assessmentResponses, startPage: 8, totalPages: totalPages)
            }
        }
    }

    // MARK: - Page Count Helper

    /// Calculate total number of pages in the report
    private func calculateTotalPages(assessmentResponses: [QuestionnaireResponse]) -> Int {
        // Base pages: Executive Summary, Clinical Assessment, Functional Capacity,
        // Medication Response, Patient-Reported Outcomes, Charts, Insights = 7
        return 7 + assessmentResponses.count
    }

    // MARK: - Page 1: Executive Summary (NEW)

    private func drawExecutiveSummaryPage(
        context: UIGraphicsPDFRendererContext,
        patientName: String?,
        diagnosis: String,
        dateRange: (start: Date, end: Date),
        logs: [SymptomLog],
        flareEvents: [FlareEvent],
        medications: [Medication],
        userProfile: UserProfile?,
        totalPages: Int
    ) {
        let ctx = context.cgContext

        // Header with logo placeholder
        drawHeader(ctx: ctx, title: "Executive Summary")

        var yPosition = headerHeight + 20

        // 1. Patient Demographics Box
        yPosition = drawPatientDemographicsBox(
            ctx: ctx,
            patientName: patientName,
            diagnosis: diagnosis,
            dateRange: dateRange,
            userProfile: userProfile,
            yPosition: yPosition
        )

        yPosition += 15

        // 2. Current Disease State Box (BASDAI & ASDAS with badges)
        yPosition = drawCurrentDiseaseStateBox(
            ctx: ctx,
            logs: logs,
            yPosition: yPosition
        )

        yPosition += 15

        // 3. Key Metrics Box (90-day metrics)
        yPosition = drawKeyMetricsBox(
            ctx: ctx,
            logs: logs,
            flareEvents: flareEvents,
            medications: medications,
            dateRange: dateRange,
            yPosition: yPosition
        )

        yPosition += 15

        // 4. Changes Since Last Period Box
        yPosition = drawChangesBox(
            ctx: ctx,
            logs: logs,
            flareEvents: flareEvents,
            medications: medications,
            dateRange: dateRange,
            yPosition: yPosition
        )

        yPosition += 15

        // 5. Red Flags Section
        yPosition = drawRedFlagsSection(
            ctx: ctx,
            logs: logs,
            flareEvents: flareEvents,
            medications: medications,
            dateRange: dateRange,
            yPosition: yPosition
        )

        // Add disclaimer at bottom
        let disclaimerY = pageSize.height - margin - footerHeight - 60
        drawDisclaimer(ctx: ctx, yPosition: disclaimerY)

        // Footer
        drawFooter(ctx: ctx, pageNumber: 1, totalPages: totalPages)
    }

    // MARK: - Executive Summary: Patient Demographics Box

    private func drawPatientDemographicsBox(
        ctx: CGContext,
        patientName: String?,
        diagnosis: String,
        dateRange: (start: Date, end: Date),
        userProfile: UserProfile?,
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Patient Demographics", yPosition: yPosition)

        // Draw box background
        let boxHeight: CGFloat = 80
        let boxRect = CGRect(x: margin, y: y, width: pageSize.width - 2 * margin, height: boxHeight)
        ctx.setFillColor(UIColor.systemGray6.cgColor)
        ctx.fill(boxRect)
        ctx.setStrokeColor(UIColor.systemGray4.cgColor)
        ctx.setLineWidth(1)
        ctx.stroke(boxRect)

        // 3-column layout
        let columnWidth = (pageSize.width - 2 * margin) / 3
        let contentY = y + 8
        let lineHeight: CGFloat = 16

        // Column 1: Basic Info
        let col1X = margin + 10
        drawCompactKeyValue(ctx: ctx, key: "Name", value: patientName ?? "Not Provided", x: col1X, y: contentY)
        drawCompactKeyValue(ctx: ctx, key: "DOB", value: formatDOBString(userProfile?.dateOfBirth), x: col1X, y: contentY + lineHeight)
        drawCompactKeyValue(ctx: ctx, key: "Gender", value: formatGenderString(userProfile?.gender), x: col1X, y: contentY + lineHeight * 2)
        drawCompactKeyValue(ctx: ctx, key: "Report Period", value: formatShortDate(dateRange.start) + " - " + formatShortDate(dateRange.end), x: col1X, y: contentY + lineHeight * 3)

        // Column 2: Diagnosis Info
        let col2X = margin + columnWidth + 10
        drawCompactKeyValue(ctx: ctx, key: "Diagnosis", value: diagnosis, x: col2X, y: contentY)
        drawCompactKeyValue(ctx: ctx, key: "Dx Date", value: formatDiagnosisDateString(userProfile?.diagnosisDate), x: col2X, y: contentY + lineHeight)
        drawCompactKeyValue(ctx: ctx, key: "Duration", value: calculateDiseaseDurationString(userProfile?.diagnosisDate), x: col2X, y: contentY + lineHeight * 2)

        // Column 3: Clinical Markers
        let col3X = margin + columnWidth * 2 + 10
        drawCompactKeyValue(ctx: ctx, key: "HLA-B27", value: formatHLAB27StatusString(userProfile?.hlaB27Positive), x: col3X, y: contentY)
        drawCompactKeyValue(ctx: ctx, key: "Prior Biologics", value: formatBiologicExperienceString(userProfile?.biologicExperienced), x: col3X, y: contentY + lineHeight)

        return y + boxHeight + 5
    }

    // MARK: - Executive Summary: Current Disease State Box

    private func drawCurrentDiseaseStateBox(
        ctx: CGContext,
        logs: [SymptomLog],
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Current Disease State", yPosition: yPosition)

        // Draw box background
        let boxHeight: CGFloat = 70
        let boxRect = CGRect(x: margin, y: y, width: pageSize.width - 2 * margin, height: boxHeight)
        ctx.setFillColor(UIColor.systemGray6.cgColor)
        ctx.fill(boxRect)
        ctx.setStrokeColor(UIColor.systemGray4.cgColor)
        ctx.setLineWidth(1)
        ctx.stroke(boxRect)

        // Calculate current scores (most recent log or average of last 7 days)
        let recentLogs = logs.sorted { ($0.timestamp ?? Date.distantPast) > ($1.timestamp ?? Date.distantPast) }
        let currentBASDai = calculateCurrentBASDaiScore(logs: recentLogs)
        let currentASDA = calculateCurrentASDAScore(logs: recentLogs)

        let contentY = y + 12
        let columnWidth = (pageSize.width - 2 * margin) / 2

        // BASDAI Score with colored badge
        let basdaiX = margin + 20
        let basdaiInterpretation = BASDAICalculator.interpretation(score: currentBASDai)
        drawScoreWithColorBadge(
            ctx: ctx,
            label: "BASDAI",
            score: currentBASDai,
            interpretation: (category: basdaiInterpretation.category, color: basdaiInterpretation.color),
            x: basdaiX,
            y: contentY
        )

        // ASDAS-CRP Score with colored badge
        let asdasX = margin + columnWidth + 20
        let asdasInterpretation = ASDACalculator.interpretation(score: currentASDA)
        drawScoreWithColorBadge(
            ctx: ctx,
            label: "ASDAS-CRP",
            score: currentASDA,
            interpretation: (category: asdasInterpretation.category, color: asdasInterpretation.color),
            x: asdasX,
            y: contentY
        )

        return y + boxHeight + 5
    }

    // MARK: - Executive Summary: Key Metrics Box

    private func drawKeyMetricsBox(
        ctx: CGContext,
        logs: [SymptomLog],
        flareEvents: [FlareEvent],
        medications: [Medication],
        dateRange: (start: Date, end: Date),
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Key Metrics (Last 90 Days)", yPosition: yPosition)

        // Draw box background
        let boxHeight: CGFloat = 55
        let boxRect = CGRect(x: margin, y: y, width: pageSize.width - 2 * margin, height: boxHeight)
        ctx.setFillColor(UIColor.systemGray6.cgColor)
        ctx.fill(boxRect)
        ctx.setStrokeColor(UIColor.systemGray4.cgColor)
        ctx.setLineWidth(1)
        ctx.stroke(boxRect)

        // Filter for last 90 days
        let ninetyDaysAgo = Calendar.current.date(byAdding: .day, value: -90, to: Date()) ?? Date()
        let recentLogs = logs.filter { ($0.timestamp ?? Date.distantPast) >= ninetyDaysAgo }
        let recentFlares = flareEvents.filter { ($0.startDate ?? Date.distantPast) >= ninetyDaysAgo }

        // Calculate metrics
        let flareCount = recentFlares.count
        let adherenceRate = calculateMedicationAdherenceRate(medications: medications, dateRange: dateRange)
        let avgMorningStiffness = calculateAverageMorningStiffnessMinutes(logs: recentLogs)
        let avgFatigue = calculateAverageFatigueScore(logs: recentLogs)

        // 4-column layout
        let columnWidth = (pageSize.width - 2 * margin) / 4
        let contentY = y + 10

        // Column 1: Flare Count
        drawMetricCard(ctx: ctx, title: "Flare Events", value: "\(flareCount)", x: margin + 10, y: contentY)

        // Column 2: Medication Adherence
        drawMetricCard(ctx: ctx, title: "Med Adherence", value: String(format: "%.0f%%", adherenceRate), x: margin + columnWidth + 10, y: contentY)

        // Column 3: Morning Stiffness
        drawMetricCard(ctx: ctx, title: "Avg Stiffness", value: "\(avgMorningStiffness) min", x: margin + columnWidth * 2 + 10, y: contentY)

        // Column 4: Fatigue
        drawMetricCard(ctx: ctx, title: "Avg Fatigue", value: String(format: "%.1f/10", avgFatigue), x: margin + columnWidth * 3 + 10, y: contentY)

        return y + boxHeight + 5
    }

    // MARK: - Executive Summary: Changes Box

    private func drawChangesBox(
        ctx: CGContext,
        logs: [SymptomLog],
        flareEvents: [FlareEvent],
        medications: [Medication],
        dateRange: (start: Date, end: Date),
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Changes Since Last Period (30 Days)", yPosition: yPosition)

        // Draw box background
        let boxHeight: CGFloat = 60
        let boxRect = CGRect(x: margin, y: y, width: pageSize.width - 2 * margin, height: boxHeight)
        ctx.setFillColor(UIColor.systemGray6.cgColor)
        ctx.fill(boxRect)
        ctx.setStrokeColor(UIColor.systemGray4.cgColor)
        ctx.setLineWidth(1)
        ctx.stroke(boxRect)

        // Split logs into current vs previous period
        let thirtyDaysAgo = Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date()
        let sixtyDaysAgo = Calendar.current.date(byAdding: .day, value: -60, to: Date()) ?? Date()

        let currentPeriodLogs = logs.filter { ($0.timestamp ?? Date.distantPast) >= thirtyDaysAgo }
        let previousPeriodLogs = logs.filter {
            let ts = $0.timestamp ?? Date.distantPast
            return ts >= sixtyDaysAgo && ts < thirtyDaysAgo
        }

        let currentPeriodFlares = flareEvents.filter { ($0.startDate ?? Date.distantPast) >= thirtyDaysAgo }
        let previousPeriodFlares = flareEvents.filter {
            let ts = $0.startDate ?? Date.distantPast
            return ts >= sixtyDaysAgo && ts < thirtyDaysAgo
        }

        // Calculate changes
        let currentBASDai = currentPeriodLogs.isEmpty ? 0 : currentPeriodLogs.reduce(0.0) { $0 + $1.basdaiScore } / Double(currentPeriodLogs.count)
        let previousBASDai = previousPeriodLogs.isEmpty ? 0 : previousPeriodLogs.reduce(0.0) { $0 + $1.basdaiScore } / Double(previousPeriodLogs.count)
        let basdaiChange = currentBASDai - previousBASDai

        let flareChange = currentPeriodFlares.count - previousPeriodFlares.count

        // Find new and stopped medications (simplified - based on isActive status)
        let activeMeds = medications.filter { $0.isActive }
        let newMedCount = activeMeds.filter {
            guard let startDate = $0.startDate else { return false }
            return startDate >= thirtyDaysAgo
        }.count

        let stoppedMedCount = medications.filter {
            guard let endDate = $0.endDate else { return false }
            return endDate >= thirtyDaysAgo && !$0.isActive
        }.count

        // 3-column layout
        let columnWidth = (pageSize.width - 2 * margin) / 3
        let contentY = y + 10

        // Column 1: BASDAI Change
        drawChangeMetric(ctx: ctx, title: "BASDAI Change", change: basdaiChange, format: "%.1f", x: margin + 20, y: contentY)

        // Column 2: Flare Frequency Change
        drawChangeMetric(ctx: ctx, title: "Flare Frequency", change: Double(flareChange), format: "%.0f", x: margin + columnWidth + 20, y: contentY)

        // Column 3: Medication Changes
        let medChangeText = newMedCount > 0 || stoppedMedCount > 0 ? "+\(newMedCount) new, -\(stoppedMedCount) stopped" : "No changes"
        drawCompactKeyValue(ctx: ctx, key: "Medications", value: medChangeText, x: margin + columnWidth * 2 + 20, y: contentY + 10)

        return y + boxHeight + 5
    }

    // MARK: - Executive Summary: Red Flags Section

    private func drawRedFlagsSection(
        ctx: CGContext,
        logs: [SymptomLog],
        flareEvents: [FlareEvent],
        medications: [Medication],
        dateRange: (start: Date, end: Date),
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Clinical Alerts", yPosition: yPosition)

        var alerts: [(severity: String, message: String, color: UIColor)] = []

        // Calculate current BASDAI
        let recentLogs = logs.sorted { ($0.timestamp ?? Date.distantPast) > ($1.timestamp ?? Date.distantPast) }
        let currentBASDai = calculateCurrentBASDaiScore(logs: recentLogs)

        // Check for high disease activity (BASDAI >= 4)
        if currentBASDai >= 4 {
            alerts.append((
                severity: "HIGH",
                message: "High disease activity (BASDAI \(String(format: "%.1f", currentBASDai))) - Consider treatment escalation",
                color: .systemRed
            ))
        }

        // Check for medication non-adherence
        let adherenceRate = calculateMedicationAdherenceRate(medications: medications, dateRange: dateRange)
        if adherenceRate < 80 && !medications.isEmpty {
            alerts.append((
                severity: "MEDIUM",
                message: "Medication adherence below 80% (\(String(format: "%.0f%%", adherenceRate))) - Discuss barriers",
                color: .systemOrange
            ))
        }

        // Check for frequent flares (3+ in 90 days)
        let ninetyDaysAgo = Calendar.current.date(byAdding: .day, value: -90, to: Date()) ?? Date()
        let recentFlares = flareEvents.filter { ($0.startDate ?? Date.distantPast) >= ninetyDaysAgo }
        if recentFlares.count >= 3 {
            alerts.append((
                severity: "MEDIUM",
                message: "Frequent flares (\(recentFlares.count) in 90 days) - Review triggers and treatment plan",
                color: .systemOrange
            ))
        }

        // Check for severe morning stiffness (>60 min average)
        let avgStiffness = calculateAverageMorningStiffnessMinutes(logs: recentLogs)
        if avgStiffness > 60 {
            alerts.append((
                severity: "MEDIUM",
                message: "Prolonged morning stiffness (avg \(avgStiffness) min) - May indicate active inflammation",
                color: .systemOrange
            ))
        }

        // Draw alerts or "no alerts" message
        if alerts.isEmpty {
            let noAlertsRect = CGRect(x: margin, y: y, width: pageSize.width - 2 * margin, height: 30)
            ctx.setFillColor(UIColor.systemGreen.withAlphaComponent(0.1).cgColor)
            ctx.fill(noAlertsRect)
            ctx.setStrokeColor(UIColor.systemGreen.cgColor)
            ctx.setLineWidth(1)
            ctx.stroke(noAlertsRect)

            let checkAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 11, weight: .medium),
                .foregroundColor: UIColor.systemGreen
            ]
            let checkString = NSAttributedString(string: "No clinical alerts at this time", attributes: checkAttributes)
            checkString.draw(at: CGPoint(x: margin + 10, y: y + 8))

            y += 35
        } else {
            for alert in alerts {
                y = drawAlertRow(ctx: ctx, severity: alert.severity, message: alert.message, color: alert.color, yPosition: y)
            }
        }

        return y
    }

    // MARK: - Executive Summary Helper Methods

    private func drawCompactKeyValue(ctx: CGContext, key: String, value: String, x: CGFloat, y: CGFloat) {
        let keyAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9, weight: .medium),
            .foregroundColor: UIColor.darkGray
        ]
        let valueAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9),
            .foregroundColor: UIColor.black
        ]

        let keyString = NSAttributedString(string: "\(key): ", attributes: keyAttributes)
        keyString.draw(at: CGPoint(x: x, y: y))

        let valueString = NSAttributedString(string: value, attributes: valueAttributes)
        valueString.draw(at: CGPoint(x: x + keyString.size().width, y: y))
    }

    private func drawScoreWithColorBadge(
        ctx: CGContext,
        label: String,
        score: Double,
        interpretation: (category: String, color: SwiftUI.Color),
        x: CGFloat,
        y: CGFloat
    ) {
        // Convert SwiftUI Color to UIColor
        let uiColor = UIColor(interpretation.color)

        // Draw label
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 12, weight: .semibold),
            .foregroundColor: UIColor.black
        ]
        let labelString = NSAttributedString(string: label, attributes: labelAttributes)
        labelString.draw(at: CGPoint(x: x, y: y))

        // Draw score
        let scoreAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 24, weight: .bold),
            .foregroundColor: uiColor
        ]
        let scoreString = NSAttributedString(string: String(format: "%.1f", score), attributes: scoreAttributes)
        scoreString.draw(at: CGPoint(x: x, y: y + 15))

        // Draw colored badge with category
        let badgeX = x + 70
        let badgeY = y + 18
        let badgeWidth: CGFloat = 100
        let badgeHeight: CGFloat = 22

        let badgeRect = CGRect(x: badgeX, y: badgeY, width: badgeWidth, height: badgeHeight)
        ctx.setFillColor(uiColor.withAlphaComponent(0.2).cgColor)
        let path = UIBezierPath(roundedRect: badgeRect, cornerRadius: 4)
        ctx.addPath(path.cgPath)
        ctx.fillPath()

        ctx.setStrokeColor(uiColor.cgColor)
        ctx.setLineWidth(1)
        ctx.addPath(path.cgPath)
        ctx.strokePath()

        let categoryAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9, weight: .medium),
            .foregroundColor: uiColor
        ]
        let categoryString = NSAttributedString(string: interpretation.category, attributes: categoryAttributes)
        let categorySize = categoryString.size()
        categoryString.draw(at: CGPoint(x: badgeX + (badgeWidth - categorySize.width) / 2, y: badgeY + 5))
    }

    private func drawMetricCard(ctx: CGContext, title: String, value: String, x: CGFloat, y: CGFloat) {
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8, weight: .medium),
            .foregroundColor: UIColor.gray
        ]
        let titleString = NSAttributedString(string: title, attributes: titleAttributes)
        titleString.draw(at: CGPoint(x: x, y: y))

        let valueAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 16, weight: .bold),
            .foregroundColor: UIColor.black
        ]
        let valueString = NSAttributedString(string: value, attributes: valueAttributes)
        valueString.draw(at: CGPoint(x: x, y: y + 12))
    }

    private func drawChangeMetric(ctx: CGContext, title: String, change: Double, format: String, x: CGFloat, y: CGFloat) {
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9, weight: .medium),
            .foregroundColor: UIColor.darkGray
        ]
        let titleString = NSAttributedString(string: title, attributes: titleAttributes)
        titleString.draw(at: CGPoint(x: x, y: y))

        // Determine color and arrow based on change direction
        let color: UIColor
        let arrow: String
        if change < -0.1 {
            color = .systemGreen
            arrow = " (improving)"
        } else if change > 0.1 {
            color = .systemRed
            arrow = " (worsening)"
        } else {
            color = .systemGray
            arrow = " (stable)"
        }

        let changeText = (change >= 0 ? "+" : "") + String(format: format, change) + arrow

        let valueAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 14, weight: .semibold),
            .foregroundColor: color
        ]
        let valueString = NSAttributedString(string: changeText, attributes: valueAttributes)
        valueString.draw(at: CGPoint(x: x, y: y + 14))
    }

    private func drawAlertRow(ctx: CGContext, severity: String, message: String, color: UIColor, yPosition: CGFloat) -> CGFloat {
        let rowHeight: CGFloat = 28
        let rowRect = CGRect(x: margin, y: yPosition, width: pageSize.width - 2 * margin, height: rowHeight)

        // Background
        ctx.setFillColor(color.withAlphaComponent(0.1).cgColor)
        ctx.fill(rowRect)

        // Left border (severity indicator)
        ctx.setFillColor(color.cgColor)
        ctx.fill(CGRect(x: margin, y: yPosition, width: 4, height: rowHeight))

        // Severity badge
        let badgeAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8, weight: .bold),
            .foregroundColor: color
        ]
        let badgeString = NSAttributedString(string: "[\(severity)]", attributes: badgeAttributes)
        badgeString.draw(at: CGPoint(x: margin + 10, y: yPosition + 8))

        // Message
        let messageAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9),
            .foregroundColor: UIColor.black
        ]
        let messageString = NSAttributedString(string: message, attributes: messageAttributes)
        messageString.draw(at: CGPoint(x: margin + 55, y: yPosition + 8))

        return yPosition + rowHeight + 3
    }

    // MARK: - Executive Summary Calculation Helpers

    private func calculateCurrentBASDaiScore(logs: [SymptomLog]) -> Double {
        // Use most recent log or average of last 7 days
        let sevenDaysAgo = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        let recentLogs = logs.filter { ($0.timestamp ?? Date.distantPast) >= sevenDaysAgo }

        if recentLogs.isEmpty {
            return logs.first?.basdaiScore ?? 0
        }
        return recentLogs.reduce(0.0) { $0 + $1.basdaiScore } / Double(recentLogs.count)
    }

    private func calculateCurrentASDAScore(logs: [SymptomLog]) -> Double {
        // Use most recent log with CRP data or calculate from available data
        let sevenDaysAgo = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        let recentLogs = logs.filter { ($0.timestamp ?? Date.distantPast) >= sevenDaysAgo }

        if let mostRecent = recentLogs.first {
            return ASDACalculator.calculate(from: mostRecent) ?? mostRecent.asdasScore
        } else if let firstLog = logs.first {
            return ASDACalculator.calculate(from: firstLog) ?? firstLog.asdasScore
        }
        return 0
    }

    private func calculateMedicationAdherenceRate(medications: [Medication], dateRange: (start: Date, end: Date)) -> Double {
        guard !medications.isEmpty else { return 0 }

        var totalExpected = 0
        var totalTaken = 0

        for medication in medications where medication.isActive {
            if let doseLogs = medication.doseLogs as? Set<DoseLog> {
                let logsInRange = doseLogs.filter {
                    guard let ts = $0.timestamp else { return false }
                    return ts >= dateRange.start && ts <= dateRange.end
                }

                // Calculate expected doses based on frequency
                let daysInRange = Calendar.current.dateComponents([.day], from: dateRange.start, to: dateRange.end).day ?? 1
                let expectedPerDay = medication.frequency?.lowercased().contains("twice") == true ? 2 : 1
                totalExpected += daysInRange * expectedPerDay
                totalTaken += logsInRange.filter { !$0.wasSkipped }.count
            }
        }

        guard totalExpected > 0 else { return 100 }
        return Double(totalTaken) / Double(totalExpected) * 100
    }

    private func calculateAverageMorningStiffnessMinutes(logs: [SymptomLog]) -> Int {
        guard !logs.isEmpty else { return 0 }
        let total = logs.reduce(0) { $0 + Int($1.morningStiffnessMinutes) }
        return total / logs.count
    }

    private func calculateAverageFatigueScore(logs: [SymptomLog]) -> Double {
        guard !logs.isEmpty else { return 0 }
        let total = logs.reduce(0.0) { $0 + Double($1.fatigueLevel) }
        return total / Double(logs.count)
    }

    // MARK: - Executive Summary Formatting Helpers

    private func formatDOBString(_ date: Date?) -> String {
        guard let date = date else { return "N/A" }
        let formatter = DateFormatter()
        formatter.dateFormat = "MM/dd/yyyy"
        return formatter.string(from: date)
    }

    private func formatGenderString(_ gender: String?) -> String {
        guard let gender = gender, !gender.isEmpty else { return "N/A" }
        return gender.capitalized
    }

    private func formatDiagnosisDateString(_ date: Date?) -> String {
        guard let date = date else { return "N/A" }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM yyyy"
        return formatter.string(from: date)
    }

    private func formatShortDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MM/dd/yy"
        return formatter.string(from: date)
    }

    private func calculateDiseaseDurationString(_ diagnosisDate: Date?) -> String {
        guard let diagnosisDate = diagnosisDate else { return "N/A" }
        let components = Calendar.current.dateComponents([.year, .month], from: diagnosisDate, to: Date())
        if let years = components.year, years > 0 {
            return "\(years) year\(years == 1 ? "" : "s")"
        } else if let months = components.month, months > 0 {
            return "\(months) month\(months == 1 ? "" : "s")"
        }
        return "< 1 month"
    }

    private func formatHLAB27StatusString(_ status: Bool?) -> String {
        guard let status = status else { return "Unknown" }
        return status ? "Positive" : "Negative"
    }

    private func formatBiologicExperienceString(_ experienced: Bool?) -> String {
        guard let experienced = experienced else { return "Unknown" }
        return experienced ? "Yes" : "No"
    }

    // MARK: - Page 2: Clinical Assessment (for Rheumatologists)

    /// Draw a clinical assessment page for rheumatologists with ASDAS-CRP, lab values, and joint counts
    /// - Parameters:
    ///   - context: PDF renderer context
    ///   - logs: Array of symptom logs
    ///   - dateRange: Date range for the report
    ///   - pageNumber: Current page number (default 2)
    ///   - totalPages: Total pages in report
    func drawClinicalAssessmentPage(
        context: UIGraphicsPDFRendererContext,
        logs: [SymptomLog],
        dateRange: (start: Date, end: Date),
        pageNumber: Int = 2,
        totalPages: Int
    ) {
        let ctx = context.cgContext

        drawHeader(ctx: ctx, title: "Patient Self-Report Summary")

        var yPosition = headerHeight + 20

        // Sort logs chronologically
        let sortedLogs = logs.sorted { ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) }

        // SECTION 1: ASDAS-CRP Section
        yPosition = drawSection(ctx: ctx, title: "ASDAS-CRP Self-Reported Scores", yPosition: yPosition)
        yPosition = drawASDASSection(ctx: ctx, logs: sortedLogs, yPosition: yPosition)

        yPosition += 10

        // SECTION 2: Laboratory Values Table
        yPosition = drawSection(ctx: ctx, title: "Laboratory Values (CRP)", yPosition: yPosition)
        yPosition = drawLabValuesTable(ctx: ctx, logs: sortedLogs, yPosition: yPosition)

        yPosition += 10

        // SECTION 3: Joint Assessment Section
        yPosition = drawSection(ctx: ctx, title: "Joint Assessment", yPosition: yPosition)
        yPosition = drawJointAssessmentSection(ctx: ctx, logs: sortedLogs, yPosition: yPosition)

        yPosition += 10

        // SECTION 4: Self-Reported Classification Box
        yPosition = drawSection(ctx: ctx, title: "Self-Reported Classification", yPosition: yPosition)
        _ = drawDiseaseActivityClassification(ctx: ctx, logs: sortedLogs, yPosition: yPosition)

        drawFooter(ctx: ctx, pageNumber: pageNumber, totalPages: totalPages)
    }

    // MARK: - ASDAS Section Helpers

    private func drawASDASSection(ctx: CGContext, logs: [SymptomLog], yPosition: CGFloat) -> CGFloat {
        var currentY = yPosition

        // Draw ASDAS trend chart with color zones
        let chartRect = CGRect(
            x: margin,
            y: currentY,
            width: pageSize.width - 2 * margin,
            height: 100
        )

        drawASDATrendChart(ctx: ctx, logs: logs, rect: chartRect)
        currentY += 110

        // Draw ASDAS statistics and interpretation
        let asdasValues = logs.compactMap { $0.asdasScore > 0 ? $0.asdasScore : nil }

        if !asdasValues.isEmpty {
            let avgASDAS = asdasValues.reduce(0, +) / Double(asdasValues.count)
            let minASDAS = asdasValues.min() ?? 0
            let maxASDAS = asdasValues.max() ?? 0

            // Statistics row
            let statsText = "Average: \(String(format: "%.2f", avgASDAS)) | Min: \(String(format: "%.2f", minASDAS)) | Max: \(String(format: "%.2f", maxASDAS))"
            currentY = drawText(ctx: ctx, text: statsText, yPosition: currentY, font: .systemFont(ofSize: 10), color: .darkGray)

            // Interpretation with color
            let interpretation = interpretASDAS(avgASDAS)
            currentY = drawText(ctx: ctx, text: "Interpretation: \(interpretation)", yPosition: currentY, font: .systemFont(ofSize: 10, weight: .medium), color: asdasColor(for: avgASDAS))

            // Check for clinically important improvement/worsening (>= 1.1 major, >= 0.6 clinically important)
            if asdasValues.count >= 2 {
                let firstASDAS = asdasValues.first ?? 0
                let lastASDAS = asdasValues.last ?? 0
                let change = lastASDAS - firstASDAS

                var changeText = ""
                var changeColor = UIColor.darkGray

                if change <= -1.1 {
                    changeText = "Major Improvement (change: \(String(format: "%.2f", change)))"
                    changeColor = .systemGreen
                } else if change <= -0.6 {
                    changeText = "Clinically Important Improvement (change: \(String(format: "%.2f", change)))"
                    changeColor = .systemGreen
                } else if change >= 1.1 {
                    changeText = "Major Worsening (change: +\(String(format: "%.2f", change)))"
                    changeColor = .systemRed
                } else if change >= 0.6 {
                    changeText = "Clinically Important Worsening (change: +\(String(format: "%.2f", change)))"
                    changeColor = .systemOrange
                } else {
                    changeText = "Stable (change: \(String(format: "%+.2f", change)))"
                    changeColor = .darkGray
                }

                currentY = drawText(ctx: ctx, text: changeText, yPosition: currentY, font: .systemFont(ofSize: 10), color: changeColor)
            }
        } else {
            currentY = drawText(ctx: ctx, text: "No ASDAS data available for this period.", yPosition: currentY, font: .systemFont(ofSize: 10), color: .gray)
        }

        return currentY
    }

    private func drawASDATrendChart(ctx: CGContext, logs: [SymptomLog], rect: CGRect) {
        // Draw chart background with ASDAS color zones
        drawASDASZones(ctx: ctx, rect: rect)

        // Draw axes
        ctx.setStrokeColor(UIColor.black.cgColor)
        ctx.setLineWidth(1)

        // Y-axis
        ctx.move(to: CGPoint(x: rect.minX, y: rect.minY))
        ctx.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))

        // X-axis
        ctx.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        ctx.strokePath()

        // Y-axis labels (0-5 for ASDAS)
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 7),
            .foregroundColor: UIColor.black
        ]

        let maxASDAS: CGFloat = 5.0
        for i in 0...5 {
            let y = rect.maxY - (CGFloat(i) / maxASDAS) * rect.height
            let label = NSAttributedString(string: "\(i)", attributes: labelAttributes)
            label.draw(at: CGPoint(x: rect.minX - 12, y: y - 5))
        }

        // Plot data
        let logsWithASDAS = logs.filter { $0.asdasScore > 0 }
        guard logsWithASDAS.count > 1 else {
            // Draw "Insufficient Data" message
            let msgAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 10),
                .foregroundColor: UIColor.gray
            ]
            let msg = NSAttributedString(string: "Insufficient ASDAS data for trend chart", attributes: msgAttributes)
            msg.draw(at: CGPoint(x: rect.midX - 70, y: rect.midY - 8))
            return
        }

        let xStep = rect.width / CGFloat(max(logsWithASDAS.count - 1, 1))

        ctx.setStrokeColor(UIColor.systemPurple.cgColor)
        ctx.setLineWidth(2)

        for (index, log) in logsWithASDAS.enumerated() {
            let x = rect.minX + CGFloat(index) * xStep
            let normalizedScore = min(log.asdasScore, 5.0) / 5.0
            let y = rect.maxY - normalizedScore * rect.height

            if index == 0 {
                ctx.move(to: CGPoint(x: x, y: y))
            } else {
                ctx.addLine(to: CGPoint(x: x, y: y))
            }
        }

        ctx.strokePath()

        // Plot points with color coding based on ASDAS zone
        for (index, log) in logsWithASDAS.enumerated() {
            let x = rect.minX + CGFloat(index) * xStep
            let normalizedScore = min(log.asdasScore, 5.0) / 5.0
            let y = rect.maxY - normalizedScore * rect.height

            ctx.setFillColor(asdasColor(for: log.asdasScore).cgColor)
            ctx.fillEllipse(in: CGRect(x: x - 4, y: y - 4, width: 8, height: 8))
        }

        // Chart title
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9, weight: .medium),
            .foregroundColor: UIColor.black
        ]

        let title = NSAttributedString(string: "ASDAS-CRP Score Over Time", attributes: titleAttributes)
        title.draw(at: CGPoint(x: rect.minX, y: rect.minY - 12))
    }

    private func drawASDASZones(ctx: CGContext, rect: CGRect) {
        let maxASDAS: CGFloat = 5.0

        // Zone boundaries: Inactive (<1.3), Moderate (1.3-2.1), High (2.1-3.5), Very High (>=3.5)
        let inactiveTop = 1.3 / maxASDAS
        let moderateTop = 2.1 / maxASDAS
        let highTop = 3.5 / maxASDAS

        // Inactive Disease (<1.3) - Green
        ctx.setFillColor(UIColor.systemGreen.withAlphaComponent(0.15).cgColor)
        let inactiveRect = CGRect(
            x: rect.minX,
            y: rect.maxY - inactiveTop * rect.height,
            width: rect.width,
            height: inactiveTop * rect.height
        )
        ctx.fill(inactiveRect)

        // Moderate Activity (1.3-2.1) - Yellow
        ctx.setFillColor(UIColor.systemYellow.withAlphaComponent(0.15).cgColor)
        let moderateRect = CGRect(
            x: rect.minX,
            y: rect.maxY - moderateTop * rect.height,
            width: rect.width,
            height: (moderateTop - inactiveTop) * rect.height
        )
        ctx.fill(moderateRect)

        // High Activity (2.1-3.5) - Orange
        ctx.setFillColor(UIColor.systemOrange.withAlphaComponent(0.15).cgColor)
        let highRect = CGRect(
            x: rect.minX,
            y: rect.maxY - highTop * rect.height,
            width: rect.width,
            height: (highTop - moderateTop) * rect.height
        )
        ctx.fill(highRect)

        // Very High Activity (>=3.5) - Red
        ctx.setFillColor(UIColor.systemRed.withAlphaComponent(0.15).cgColor)
        let veryHighRect = CGRect(
            x: rect.minX,
            y: rect.minY,
            width: rect.width,
            height: (1.0 - highTop) * rect.height
        )
        ctx.fill(veryHighRect)

        // Draw threshold lines with dashes
        ctx.setStrokeColor(UIColor.gray.withAlphaComponent(0.5).cgColor)
        ctx.setLineWidth(0.5)
        ctx.setLineDash(phase: 0, lengths: [3, 3])

        for threshold in [1.3, 2.1, 3.5] {
            let y = rect.maxY - (CGFloat(threshold) / maxASDAS) * rect.height
            ctx.move(to: CGPoint(x: rect.minX, y: y))
            ctx.addLine(to: CGPoint(x: rect.maxX, y: y))
            ctx.strokePath()
        }

        ctx.setLineDash(phase: 0, lengths: [])
    }

    private func interpretASDAS(_ score: Double) -> String {
        switch score {
        case ..<1.3:
            return "Self-Reported: Low"
        case 1.3..<2.1:
            return "Self-Reported: Moderate"
        case 2.1..<3.5:
            return "Self-Reported: Elevated"
        default:
            return "Self-Reported: High"
        }
    }

    private func asdasColor(for score: Double) -> UIColor {
        switch score {
        case ..<1.3:
            return .systemGreen
        case 1.3..<2.1:
            return .systemYellow
        case 2.1..<3.5:
            return .systemOrange
        default:
            return .systemRed
        }
    }

    // MARK: - Laboratory Values Table

    private func drawLabValuesTable(ctx: CGContext, logs: [SymptomLog], yPosition: CGFloat) -> CGFloat {
        var currentY = yPosition

        // Filter logs with CRP values
        let logsWithCRP = logs.filter { $0.crpValue > 0 }

        if logsWithCRP.isEmpty {
            return drawText(ctx: ctx, text: "No CRP values recorded in this period.", yPosition: currentY, font: .systemFont(ofSize: 10), color: .gray)
        }

        // Table header
        let columnWidths: [CGFloat] = [100, 90, 130]
        let headers = ["Date", "CRP (mg/L)", "Status"]

        let headerAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9, weight: .semibold),
            .foregroundColor: UIColor.black
        ]

        // Draw header background
        ctx.setFillColor(UIColor.systemGray5.cgColor)
        ctx.fill(CGRect(x: margin, y: currentY, width: columnWidths.reduce(0, +), height: 16))

        var xPosition = margin
        for (index, header) in headers.enumerated() {
            let headerString = NSAttributedString(string: header, attributes: headerAttributes)
            headerString.draw(at: CGPoint(x: xPosition + 4, y: currentY + 2))
            xPosition += columnWidths[index]
        }

        currentY += 17

        // Table rows (limit to most recent 5 entries to fit on page)
        let recentLogs = Array(logsWithCRP.suffix(5))
        let rowAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8),
            .foregroundColor: UIColor.darkGray
        ]

        for (rowIndex, log) in recentLogs.enumerated() {
            // Alternate row background
            if rowIndex % 2 == 0 {
                ctx.setFillColor(UIColor.systemGray6.cgColor)
                ctx.fill(CGRect(x: margin, y: currentY, width: columnWidths.reduce(0, +), height: 14))
            }

            xPosition = margin

            // Date
            let dateString = NSAttributedString(string: formatDate(log.timestamp ?? Date()), attributes: rowAttributes)
            dateString.draw(at: CGPoint(x: xPosition + 4, y: currentY + 2))
            xPosition += columnWidths[0]

            // CRP Value
            let crpString = NSAttributedString(string: String(format: "%.1f", log.crpValue), attributes: rowAttributes)
            crpString.draw(at: CGPoint(x: xPosition + 4, y: currentY + 2))
            xPosition += columnWidths[1]

            // Status: Normal (<5), Elevated (5-10), High (>10)
            let status = crpStatus(for: log.crpValue)
            let statusColor = crpStatusColor(for: log.crpValue)
            let statusAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 8, weight: .medium),
                .foregroundColor: statusColor
            ]
            let statusString = NSAttributedString(string: status, attributes: statusAttributes)
            statusString.draw(at: CGPoint(x: xPosition + 4, y: currentY + 2))

            currentY += 14
        }

        // Draw table border
        ctx.setStrokeColor(UIColor.lightGray.cgColor)
        ctx.setLineWidth(0.5)
        ctx.stroke(CGRect(x: margin, y: yPosition, width: columnWidths.reduce(0, +), height: currentY - yPosition))

        return currentY + 5
    }

    private func crpStatus(for value: Double) -> String {
        switch value {
        case ..<5:
            return "Normal"
        case 5..<10:
            return "Elevated"
        default:
            return "High"
        }
    }

    private func crpStatusColor(for value: Double) -> UIColor {
        switch value {
        case ..<5:
            return .systemGreen
        case 5..<10:
            return .systemOrange
        default:
            return .systemRed
        }
    }

    // MARK: - Joint Assessment Section

    private func drawJointAssessmentSection(ctx: CGContext, logs: [SymptomLog], yPosition: CGFloat) -> CGFloat {
        var currentY = yPosition

        // Check if we have joint count data
        let logsWithJointData = logs.filter { $0.tenderJointCount > 0 || $0.swollenJointCount > 0 || $0.enthesitisCount > 0 }

        if logsWithJointData.isEmpty {
            return drawText(ctx: ctx, text: "No joint assessment data recorded in this period.", yPosition: currentY, font: .systemFont(ofSize: 10), color: .gray)
        }

        // Draw bar chart for joint counts
        let chartRect = CGRect(
            x: margin,
            y: currentY,
            width: pageSize.width - 2 * margin,
            height: 70
        )

        drawJointCountBarChart(ctx: ctx, logs: logsWithJointData, rect: chartRect)
        currentY += 80

        // Summary statistics
        let avgTender = Double(logsWithJointData.map { Int($0.tenderJointCount) }.reduce(0, +)) / Double(logsWithJointData.count)
        let avgSwollen = Double(logsWithJointData.map { Int($0.swollenJointCount) }.reduce(0, +)) / Double(logsWithJointData.count)
        let avgEnthesitis = Double(logsWithJointData.map { Int($0.enthesitisCount) }.reduce(0, +)) / Double(logsWithJointData.count)

        let summaryText = "Averages - Tender: \(String(format: "%.1f", avgTender)) | Swollen: \(String(format: "%.1f", avgSwollen)) | Enthesitis: \(String(format: "%.1f", avgEnthesitis))"
        currentY = drawText(ctx: ctx, text: summaryText, yPosition: currentY, font: .systemFont(ofSize: 9), color: .darkGray)

        return currentY
    }

    private func drawJointCountBarChart(ctx: CGContext, logs: [SymptomLog], rect: CGRect) {
        // Draw axes
        ctx.setStrokeColor(UIColor.black.cgColor)
        ctx.setLineWidth(1)

        // Y-axis
        ctx.move(to: CGPoint(x: rect.minX, y: rect.minY))
        ctx.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))

        // X-axis
        ctx.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        ctx.strokePath()

        // Find max count for scaling
        let maxCount = logs.map { max(Int($0.tenderJointCount), Int($0.swollenJointCount), Int($0.enthesitisCount)) }.max() ?? 10
        let scaledMax = max(maxCount, 10)

        // Y-axis labels
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 6),
            .foregroundColor: UIColor.black
        ]

        for i in stride(from: 0, through: scaledMax, by: max(scaledMax / 4, 1)) {
            let y = rect.maxY - (CGFloat(i) / CGFloat(scaledMax)) * rect.height
            let label = NSAttributedString(string: "\(i)", attributes: labelAttributes)
            label.draw(at: CGPoint(x: rect.minX - 12, y: y - 4))
        }

        // Draw bars for each log entry (limit to 8 most recent)
        let recentLogs = Array(logs.suffix(8))
        let groupWidth = rect.width / CGFloat(recentLogs.count)
        let barWidth = groupWidth * 0.25
        let gap: CGFloat = 2

        for (index, log) in recentLogs.enumerated() {
            let groupX = rect.minX + CGFloat(index) * groupWidth + groupWidth * 0.1

            // Tender Joint Count - Blue
            let tenderHeight = (CGFloat(log.tenderJointCount) / CGFloat(scaledMax)) * rect.height
            ctx.setFillColor(UIColor.systemBlue.cgColor)
            ctx.fill(CGRect(x: groupX, y: rect.maxY - tenderHeight, width: barWidth, height: tenderHeight))

            // Swollen Joint Count - Orange
            let swollenHeight = (CGFloat(log.swollenJointCount) / CGFloat(scaledMax)) * rect.height
            ctx.setFillColor(UIColor.systemOrange.cgColor)
            ctx.fill(CGRect(x: groupX + barWidth + gap, y: rect.maxY - swollenHeight, width: barWidth, height: swollenHeight))

            // Enthesitis Count - Purple
            let enthesitisHeight = (CGFloat(log.enthesitisCount) / CGFloat(scaledMax)) * rect.height
            ctx.setFillColor(UIColor.systemPurple.cgColor)
            ctx.fill(CGRect(x: groupX + 2 * (barWidth + gap), y: rect.maxY - enthesitisHeight, width: barWidth, height: enthesitisHeight))
        }

        // Legend
        let legendY = rect.minY - 10
        let legendItems: [(String, UIColor)] = [
            ("Tender", .systemBlue),
            ("Swollen", .systemOrange),
            ("Enthesitis", .systemPurple)
        ]

        var legendX = rect.minX
        for (label, color) in legendItems {
            ctx.setFillColor(color.cgColor)
            ctx.fill(CGRect(x: legendX, y: legendY, width: 8, height: 8))

            let legendLabel = NSAttributedString(string: label, attributes: [
                .font: UIFont.systemFont(ofSize: 7),
                .foregroundColor: UIColor.darkGray
            ])
            legendLabel.draw(at: CGPoint(x: legendX + 10, y: legendY - 1))

            legendX += 60
        }
    }

    // MARK: - Disease Activity Classification

    private func drawDiseaseActivityClassification(ctx: CGContext, logs: [SymptomLog], yPosition: CGFloat) -> CGFloat {
        var currentY = yPosition

        // Draw classification box
        let boxRect = CGRect(
            x: margin,
            y: currentY,
            width: pageSize.width - 2 * margin,
            height: 75
        )

        // Box background
        ctx.setFillColor(UIColor.systemGray6.cgColor)
        ctx.fill(boxRect)

        // Box border
        ctx.setStrokeColor(UIColor.systemGray4.cgColor)
        ctx.setLineWidth(1)
        ctx.stroke(boxRect)

        currentY += 8

        // Get latest ASDAS for classification
        let logsWithASDAS = logs.filter { $0.asdasScore > 0 }

        if let latestLog = logsWithASDAS.last {
            let currentASDAS = latestLog.asdasScore
            let classification = interpretASDAS(currentASDAS)
            let classificationColor = asdasColor(for: currentASDAS)

            // Current Classification
            let classificationTitle = "Current Classification: "
            let titleAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 10, weight: .medium),
                .foregroundColor: UIColor.black
            ]

            let titleString = NSAttributedString(string: classificationTitle, attributes: titleAttributes)
            titleString.draw(at: CGPoint(x: margin + 8, y: currentY))

            let classificationAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 10, weight: .bold),
                .foregroundColor: classificationColor
            ]

            let classificationString = NSAttributedString(string: classification, attributes: classificationAttributes)
            classificationString.draw(at: CGPoint(x: margin + 8 + titleString.size().width, y: currentY))

            currentY += 15

            // Latest ASDAS Value
            let valueText = "Latest ASDAS-CRP: \(String(format: "%.2f", currentASDAS))"
            currentY = drawTextInBox(ctx: ctx, text: valueText, xPosition: margin + 8, yPosition: currentY)

            // Treatment Target Status (target is ASDAS < 1.3)
            let atTarget = currentASDAS < 1.3
            let targetStatus = atTarget ? "At Treatment Target (ASDAS < 1.3)" : "Not at Treatment Target"
            let targetColor = atTarget ? UIColor.systemGreen : UIColor.systemOrange

            let targetAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 9),
                .foregroundColor: targetColor
            ]
            let targetString = NSAttributedString(string: targetStatus, attributes: targetAttributes)
            targetString.draw(at: CGPoint(x: margin + 8, y: currentY))

            currentY += 14

            // Improvement Threshold Assessment (>= 1.1 major, >= 0.6 clinically important)
            if logsWithASDAS.count >= 2 {
                let firstASDAS = logsWithASDAS.first!.asdasScore
                let change = currentASDAS - firstASDAS

                var thresholdText = ""
                var thresholdColor = UIColor.darkGray

                if change <= -1.1 {
                    thresholdText = "Major Improvement Threshold Met (>= 1.1 decrease)"
                    thresholdColor = .systemGreen
                } else if change <= -0.6 {
                    thresholdText = "Clinically Important Improvement (>= 0.6 decrease)"
                    thresholdColor = .systemGreen
                } else if change >= 0.6 {
                    thresholdText = "Warning: Worsening Detected (>= 0.6 increase)"
                    thresholdColor = .systemRed
                } else {
                    thresholdText = "Stable: No significant change detected"
                    thresholdColor = .darkGray
                }

                let thresholdAttributes: [NSAttributedString.Key: Any] = [
                    .font: UIFont.systemFont(ofSize: 9),
                    .foregroundColor: thresholdColor
                ]
                let thresholdString = NSAttributedString(string: thresholdText, attributes: thresholdAttributes)
                thresholdString.draw(at: CGPoint(x: margin + 8, y: currentY))
            }
        } else {
            currentY = drawTextInBox(ctx: ctx, text: "Insufficient ASDAS data for classification.", xPosition: margin + 8, yPosition: currentY)
            _ = drawTextInBox(ctx: ctx, text: "At least one ASDAS-CRP measurement is required.", xPosition: margin + 8, yPosition: currentY)
        }

        return boxRect.maxY + 8
    }

    private func drawTextInBox(ctx: CGContext, text: String, xPosition: CGFloat, yPosition: CGFloat) -> CGFloat {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9),
            .foregroundColor: UIColor.darkGray
        ]
        let textString = NSAttributedString(string: text, attributes: attributes)
        textString.draw(at: CGPoint(x: xPosition, y: yPosition))
        return yPosition + 14
    }

    // MARK: - Page 3: Functional Capacity (for Rheumatologists)

    /// Draws the Functional Capacity page focusing on patient function and physical activity
    /// - Parameters:
    ///   - context: The PDF renderer context
    ///   - logs: Symptom log entries for the reporting period
    ///   - exerciseSessions: Exercise sessions for the reporting period
    ///   - assessmentResponses: Questionnaire responses (BASFI, etc.)
    ///   - dateRange: The start and end dates for the report
    ///   - pageNumber: Current page number
    ///   - totalPages: Total number of pages in the report
    func drawFunctionalCapacityPage(
        context: UIGraphicsPDFRendererContext,
        logs: [SymptomLog],
        exerciseSessions: [ExerciseSession],
        assessmentResponses: [QuestionnaireResponse],
        dateRange: (start: Date, end: Date),
        pageNumber: Int,
        totalPages: Int
    ) {
        let ctx = context.cgContext

        drawHeader(ctx: ctx, title: "Functional Capacity Assessment")

        var yPosition = headerHeight + 20

        // MARK: - 1. BASFI Section (Bath AS Functional Index)
        yPosition = drawSection(ctx: ctx, title: "BASFI (Bath AS Functional Index)", yPosition: yPosition)

        // Get BASFI scores from QuestionnaireResponse where questionnaireID == "basfi"
        let basfiResponses = assessmentResponses.filter { $0.questionnaireID == "basfi" }
            .sorted { ($0.createdAt ?? Date.distantPast) < ($1.createdAt ?? Date.distantPast) }

        if basfiResponses.isEmpty {
            yPosition = drawText(
                ctx: ctx,
                text: "No BASFI assessments recorded in this period.",
                yPosition: yPosition,
                font: .systemFont(ofSize: 11),
                color: .gray
            )
        } else {
            // Current BASFI score with interpretation
            if let latestBASFI = basfiResponses.last {
                let score = latestBASFI.score
                let interpretation = interpretBASFI(score: score)

                yPosition = drawKeyValue(ctx: ctx, key: "Current Score", value: String(format: "%.1f / 10", score), yPosition: yPosition)
                yPosition = drawKeyValue(ctx: ctx, key: "Interpretation", value: interpretation.text, yPosition: yPosition)
                yPosition = drawKeyValue(ctx: ctx, key: "Last Assessed", value: formatDate(latestBASFI.createdAt ?? Date()), yPosition: yPosition)
                yPosition = drawKeyValue(ctx: ctx, key: "Assessments in Period", value: "\(basfiResponses.count)", yPosition: yPosition)

                // Draw color indicator for severity
                drawBASFISeverityIndicator(ctx: ctx, score: score, x: pageSize.width - margin - 60, y: yPosition - 70)
            }

            yPosition += 10

            // BASFI Trend Chart (if multiple assessments)
            if basfiResponses.count > 1 {
                let chartRect = CGRect(
                    x: margin,
                    y: yPosition,
                    width: pageSize.width - 2 * margin,
                    height: 90
                )
                drawBASFITrendChart(ctx: ctx, responses: basfiResponses, rect: chartRect)
                yPosition += 100
            }
        }

        yPosition += 10

        // MARK: - 2. Physical Activity Summary
        yPosition = drawSection(ctx: ctx, title: "Physical Activity Summary", yPosition: yPosition)

        let activityStats = calculatePhysicalActivityStats(sessions: exerciseSessions, dateRange: dateRange)

        let activityInfo: [(String, String)] = [
            ("Exercise Sessions", "\(activityStats.totalSessions)"),
            ("Total Exercise Time", "\(activityStats.totalMinutes) minutes"),
            ("Avg Sessions/Week", String(format: "%.1f", activityStats.avgSessionsPerWeek)),
            ("Most Common Type", activityStats.mostCommonType ?? "N/A")
        ]

        for (label, value) in activityInfo {
            yPosition = drawKeyValue(ctx: ctx, key: label, value: value, yPosition: yPosition)
        }

        // Exercise type breakdown (top 3)
        if !activityStats.exerciseTypeBreakdown.isEmpty {
            yPosition += 5
            yPosition = drawText(
                ctx: ctx,
                text: "Exercise Types:",
                yPosition: yPosition,
                font: .systemFont(ofSize: 10, weight: .medium),
                color: .black
            )

            for (type, count) in activityStats.exerciseTypeBreakdown.prefix(3) {
                let percentage = activityStats.totalSessions > 0 ? Double(count) / Double(activityStats.totalSessions) * 100 : 0
                yPosition = drawText(
                    ctx: ctx,
                    text: "  - \(type): \(count) (\(String(format: "%.0f", percentage))%)",
                    yPosition: yPosition,
                    font: .systemFont(ofSize: 9),
                    color: .darkGray
                )
            }
        }

        yPosition += 10

        // MARK: - 3. Exercise Adherence Metrics
        yPosition = drawSection(ctx: ctx, title: "Exercise Adherence Metrics", yPosition: yPosition)

        let adherenceStats = calculateExerciseAdherenceStats(sessions: exerciseSessions)

        let adherenceInfo: [(String, String)] = [
            ("Completion Rate", String(format: "%.0f%%", adherenceStats.completionRate)),
            ("Sessions Stopped Early", "\(adherenceStats.stoppedEarlyCount)"),
            ("Pain Incidents", "\(adherenceStats.painAlertCount)")
        ]

        for (label, value) in adherenceInfo {
            yPosition = drawKeyValue(ctx: ctx, key: label, value: value, yPosition: yPosition)
        }

        // Pain before/after comparison
        if adherenceStats.avgPainBefore > 0 || adherenceStats.avgPainAfter > 0 {
            let painChange = adherenceStats.avgPainAfter - adherenceStats.avgPainBefore
            let changeText: String
            let changeColor: UIColor

            if painChange < -0.5 {
                changeText = String(format: "Pain improved by %.1f pts after exercise", abs(painChange))
                changeColor = .systemGreen
            } else if painChange > 0.5 {
                changeText = String(format: "Pain increased by %.1f pts after exercise", painChange)
                changeColor = .systemRed
            } else {
                changeText = "Pain unchanged after exercise"
                changeColor = .darkGray
            }

            yPosition = drawText(ctx: ctx, text: changeText, yPosition: yPosition, font: .systemFont(ofSize: 10), color: changeColor)
        }

        yPosition += 10

        // MARK: - 4. Mobility Indicators (from SymptomLog)
        yPosition = drawSection(ctx: ctx, title: "Mobility Indicators", yPosition: yPosition)

        let mobilityStats = calculateMobilityStats(logs: logs)

        let mobilityInfo: [(String, String)] = [
            ("Avg Morning Stiffness", "\(mobilityStats.avgMorningStiffnessMinutes) min"),
            ("Max Morning Stiffness", "\(mobilityStats.maxMorningStiffnessMinutes) min"),
            ("Avg Physical Function", String(format: "%.1f / 10", mobilityStats.avgPhysicalFunction)),
            ("Avg Activity Limitation", String(format: "%.1f / 10", mobilityStats.avgActivityLimitation)),
            ("Days >60min Stiffness", "\(mobilityStats.daysWithSevereStiffness)")
        ]

        for (label, value) in mobilityInfo {
            yPosition = drawKeyValue(ctx: ctx, key: label, value: value, yPosition: yPosition)
        }

        // Stiffness trend indicator
        if let trend = mobilityStats.stiffnessTrend {
            let trendText: String
            let trendColor: UIColor

            switch trend {
            case .improving:
                trendText = "Stiffness Trend: Improving"
                trendColor = .systemGreen
            case .worsening:
                trendText = "Stiffness Trend: Worsening"
                trendColor = .systemRed
            case .stable:
                trendText = "Stiffness Trend: Stable"
                trendColor = .darkGray
            }

            yPosition = drawText(ctx: ctx, text: trendText, yPosition: yPosition, font: .systemFont(ofSize: 10, weight: .medium), color: trendColor)
        }

        drawFooter(ctx: ctx, pageNumber: pageNumber, totalPages: totalPages)
    }

    // Overload for backward compatibility with existing call signature
    private func drawFunctionalCapacityPage(
        context: UIGraphicsPDFRendererContext,
        logs: [SymptomLog],
        exerciseSessions: [ExerciseSession],
        totalPages: Int
    ) {
        drawFunctionalCapacityPage(
            context: context,
            logs: logs,
            exerciseSessions: exerciseSessions,
            assessmentResponses: [],
            dateRange: (start: Date().addingTimeInterval(-30 * 24 * 3600), end: Date()),
            pageNumber: 3,
            totalPages: totalPages
        )
    }

    // MARK: - BASFI Interpretation

    private struct BASFIInterpretation {
        let text: String
        let color: UIColor
    }

    private func interpretBASFI(score: Double) -> BASFIInterpretation {
        switch score {
        case 0..<2:
            return BASFIInterpretation(text: "Good function", color: .systemGreen)
        case 2..<4:
            return BASFIInterpretation(text: "Mild limitation", color: .systemYellow)
        case 4..<6:
            return BASFIInterpretation(text: "Moderate limitation", color: .systemOrange)
        case 6..<8:
            return BASFIInterpretation(text: "Severe limitation", color: .systemRed)
        default:
            return BASFIInterpretation(text: "Very severe limitation", color: .systemRed)
        }
    }

    private func drawBASFISeverityIndicator(ctx: CGContext, score: Double, x: CGFloat, y: CGFloat) {
        let interpretation = interpretBASFI(score: score)

        ctx.setFillColor(interpretation.color.cgColor)
        ctx.fillEllipse(in: CGRect(x: x, y: y, width: 50, height: 50))

        // Draw score text in circle
        let scoreAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 14, weight: .bold),
            .foregroundColor: UIColor.white
        ]

        let scoreText = NSAttributedString(string: String(format: "%.1f", score), attributes: scoreAttributes)
        let textSize = scoreText.size()
        scoreText.draw(at: CGPoint(x: x + 25 - textSize.width / 2, y: y + 25 - textSize.height / 2))
    }

    // MARK: - BASFI Trend Chart

    private func drawBASFITrendChart(ctx: CGContext, responses: [QuestionnaireResponse], rect: CGRect) {
        // Draw chart background
        ctx.setFillColor(UIColor.systemGray6.cgColor)
        ctx.fill(rect)

        let chartLeft = rect.minX + 25
        let chartBottom = rect.maxY - 12
        let chartTop = rect.minY + 12
        let chartRight = rect.maxX - 10
        let chartHeight = chartBottom - chartTop
        let chartWidth = chartRight - chartLeft

        // Draw axes
        ctx.setStrokeColor(UIColor.black.cgColor)
        ctx.setLineWidth(1)
        ctx.move(to: CGPoint(x: chartLeft, y: chartTop))
        ctx.addLine(to: CGPoint(x: chartLeft, y: chartBottom))
        ctx.addLine(to: CGPoint(x: chartRight, y: chartBottom))
        ctx.strokePath()

        // Y-axis labels (0-10)
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 7),
            .foregroundColor: UIColor.black
        ]

        for i in stride(from: 0, through: 10, by: 5) {
            let y = chartBottom - (CGFloat(i) / 10.0) * chartHeight
            let label = NSAttributedString(string: "\(i)", attributes: labelAttributes)
            label.draw(at: CGPoint(x: rect.minX + 8, y: y - 5))

            ctx.setStrokeColor(UIColor.lightGray.withAlphaComponent(0.3).cgColor)
            ctx.move(to: CGPoint(x: chartLeft, y: y))
            ctx.addLine(to: CGPoint(x: chartRight, y: y))
            ctx.strokePath()
        }

        // Draw threshold zones
        let zones: [(range: ClosedRange<Double>, color: UIColor)] = [
            (0...2, UIColor.systemGreen.withAlphaComponent(0.15)),
            (2...4, UIColor.systemYellow.withAlphaComponent(0.15)),
            (4...6, UIColor.systemOrange.withAlphaComponent(0.15)),
            (6...10, UIColor.systemRed.withAlphaComponent(0.15))
        ]

        for zone in zones {
            let zoneMinY = chartBottom - (CGFloat(zone.range.upperBound) / 10.0) * chartHeight
            let zoneMaxY = chartBottom - (CGFloat(zone.range.lowerBound) / 10.0) * chartHeight
            ctx.setFillColor(zone.color.cgColor)
            ctx.fill(CGRect(x: chartLeft, y: zoneMinY, width: chartWidth, height: zoneMaxY - zoneMinY))
        }

        // Plot data
        guard responses.count > 0 else { return }

        let xStep = chartWidth / CGFloat(max(responses.count - 1, 1))

        // Draw line
        ctx.setStrokeColor(UIColor.systemBlue.cgColor)
        ctx.setLineWidth(2)

        for (index, response) in responses.enumerated() {
            let x = chartLeft + CGFloat(index) * xStep
            let y = chartBottom - (response.score / 10.0) * chartHeight

            if index == 0 {
                ctx.move(to: CGPoint(x: x, y: y))
            } else {
                ctx.addLine(to: CGPoint(x: x, y: y))
            }
        }
        ctx.strokePath()

        // Draw points
        ctx.setFillColor(UIColor.systemBlue.cgColor)
        for (index, response) in responses.enumerated() {
            let x = chartLeft + CGFloat(index) * xStep
            let y = chartBottom - (response.score / 10.0) * chartHeight
            ctx.fillEllipse(in: CGRect(x: x - 3, y: y - 3, width: 6, height: 6))
        }

        // Title
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8, weight: .medium),
            .foregroundColor: UIColor.black
        ]
        let title = NSAttributedString(string: "BASFI Trend (0-10, higher = worse)", attributes: titleAttributes)
        title.draw(at: CGPoint(x: chartLeft, y: rect.minY))
    }

    // MARK: - Physical Activity Statistics

    private struct PhysicalActivityStats {
        let totalSessions: Int
        let totalMinutes: Int
        let avgSessionsPerWeek: Double
        let mostCommonType: String?
        let exerciseTypeBreakdown: [(String, Int)]
    }

    private func calculatePhysicalActivityStats(sessions: [ExerciseSession], dateRange: (start: Date, end: Date)) -> PhysicalActivityStats {
        let totalSessions = sessions.count
        let totalMinutes = sessions.reduce(0) { $0 + Int($1.durationMinutes) }

        // Calculate weeks in date range
        let calendar = Calendar.current
        let days = calendar.dateComponents([.day], from: dateRange.start, to: dateRange.end).day ?? 1
        let weeks = max(1, Double(days) / 7.0)
        let avgSessionsPerWeek = Double(totalSessions) / weeks

        // Calculate exercise type distribution
        var typeCount: [String: Int] = [:]
        for session in sessions {
            let type = session.routineType ?? "Unknown"
            typeCount[type, default: 0] += 1
        }

        let sortedTypes = typeCount.sorted { $0.value > $1.value }
        let mostCommonType = sortedTypes.first?.key

        return PhysicalActivityStats(
            totalSessions: totalSessions,
            totalMinutes: totalMinutes,
            avgSessionsPerWeek: avgSessionsPerWeek,
            mostCommonType: mostCommonType,
            exerciseTypeBreakdown: sortedTypes
        )
    }

    // MARK: - Exercise Adherence Statistics

    private struct ExerciseAdherenceStats {
        let completionRate: Double
        let stoppedEarlyCount: Int
        let painAlertCount: Int
        let avgPainBefore: Double
        let avgPainAfter: Double
    }

    private func calculateExerciseAdherenceStats(sessions: [ExerciseSession]) -> ExerciseAdherenceStats {
        guard !sessions.isEmpty else {
            return ExerciseAdherenceStats(
                completionRate: 0,
                stoppedEarlyCount: 0,
                painAlertCount: 0,
                avgPainBefore: 0,
                avgPainAfter: 0
            )
        }

        let completedCount = sessions.filter { $0.completedSuccessfully }.count
        let completionRate = Double(completedCount) / Double(sessions.count) * 100

        let stoppedEarlyCount = sessions.filter { $0.stoppedEarly }.count

        // Count pain alerts from related ExercisePainAlert entities
        var totalPainAlerts = 0
        for session in sessions {
            if let painAlerts = session.painAlerts as? Set<ExercisePainAlert> {
                totalPainAlerts += painAlerts.count
            }
        }

        // Calculate average pain before and after
        let sessionsWithPainData = sessions.filter { $0.painBefore > 0 || $0.painAfter > 0 }
        let avgPainBefore: Double
        let avgPainAfter: Double

        if sessionsWithPainData.isEmpty {
            avgPainBefore = 0
            avgPainAfter = 0
        } else {
            avgPainBefore = sessionsWithPainData.reduce(0.0) { $0 + Double($1.painBefore) } / Double(sessionsWithPainData.count)
            avgPainAfter = sessionsWithPainData.reduce(0.0) { $0 + Double($1.painAfter) } / Double(sessionsWithPainData.count)
        }

        return ExerciseAdherenceStats(
            completionRate: completionRate,
            stoppedEarlyCount: stoppedEarlyCount,
            painAlertCount: totalPainAlerts,
            avgPainBefore: avgPainBefore,
            avgPainAfter: avgPainAfter
        )
    }

    // MARK: - Mobility Statistics

    private enum StiffnessTrend {
        case improving
        case worsening
        case stable
    }

    private struct MobilityStats {
        let avgMorningStiffnessMinutes: Int
        let maxMorningStiffnessMinutes: Int
        let avgPhysicalFunction: Double
        let avgActivityLimitation: Double
        let daysWithSevereStiffness: Int
        let stiffnessTrend: StiffnessTrend?
    }

    private func calculateMobilityStats(logs: [SymptomLog]) -> MobilityStats {
        guard !logs.isEmpty else {
            return MobilityStats(
                avgMorningStiffnessMinutes: 0,
                maxMorningStiffnessMinutes: 0,
                avgPhysicalFunction: 0,
                avgActivityLimitation: 0,
                daysWithSevereStiffness: 0,
                stiffnessTrend: nil
            )
        }

        // Morning stiffness calculations
        let stiffnessValues = logs.map { Int($0.morningStiffnessMinutes) }
        let avgMorningStiffness = stiffnessValues.reduce(0, +) / max(1, stiffnessValues.count)
        let maxMorningStiffness = stiffnessValues.max() ?? 0
        let daysWithSevereStiffness = stiffnessValues.filter { $0 > 60 }.count

        // Physical function (from physicalFunctionScore, 0-10 scale)
        let physicalFunctionValues = logs.map { Double($0.physicalFunctionScore) }
        let avgPhysicalFunction = physicalFunctionValues.reduce(0, +) / Double(max(1, physicalFunctionValues.count))

        // Activity limitation (from activityLimitationScore, 0-10 scale)
        let activityLimitationValues = logs.map { Double($0.activityLimitationScore) }
        let avgActivityLimitation = activityLimitationValues.reduce(0, +) / Double(max(1, activityLimitationValues.count))

        // Calculate stiffness trend (compare first half vs second half of period)
        var stiffnessTrend: StiffnessTrend?
        if logs.count >= 10 {
            let sortedLogs = logs.sorted { ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) }
            let midpoint = sortedLogs.count / 2

            let firstHalf = sortedLogs.prefix(midpoint)
            let secondHalf = sortedLogs.suffix(midpoint)

            let firstHalfAvg = firstHalf.map { Int($0.morningStiffnessMinutes) }.reduce(0, +) / max(1, firstHalf.count)
            let secondHalfAvg = secondHalf.map { Int($0.morningStiffnessMinutes) }.reduce(0, +) / max(1, secondHalf.count)

            let change = secondHalfAvg - firstHalfAvg
            if change < -10 {
                stiffnessTrend = .improving
            } else if change > 10 {
                stiffnessTrend = .worsening
            } else {
                stiffnessTrend = .stable
            }
        }

        return MobilityStats(
            avgMorningStiffnessMinutes: avgMorningStiffness,
            maxMorningStiffnessMinutes: maxMorningStiffness,
            avgPhysicalFunction: avgPhysicalFunction,
            avgActivityLimitation: avgActivityLimitation,
            daysWithSevereStiffness: daysWithSevereStiffness,
            stiffnessTrend: stiffnessTrend
        )
    }

    // MARK: - Page 4: Medication Response (Critical for Rheumatologists)

    /// Draws comprehensive medication response analysis page - critical for treatment decisions
    /// Includes:
    /// 1. Current Medications Table (name, category, dosage, route, biologic flag)
    /// 2. Medication Adherence Details (visual bars, skip reasons from DoseLog)
    /// 3. Treatment Response Analysis (BASDAI 4 weeks before vs 4 weeks after medication start)
    /// 4. Medication Timeline (start/stop events)
    /// 5. Treatment History Summary (biologics, NSAIDs, DMARDs counts)
    private func drawMedicationResponsePage(
        context: UIGraphicsPDFRendererContext,
        medications: [Medication],
        logs: [SymptomLog],
        totalPages: Int
    ) {
        let ctx = context.cgContext

        drawHeader(ctx: ctx, title: "Medication Response Analysis")

        var yPosition = headerHeight + 18

        // Section 1: Current Medications Table
        yPosition = drawMedResponseCurrentMedicationsTable(ctx: ctx, medications: medications, yPosition: yPosition)
        yPosition += 10

        // Section 2: Medication Adherence Details
        yPosition = drawMedResponseAdherenceDetails(ctx: ctx, medications: medications, yPosition: yPosition)
        yPosition += 10

        // Section 3: Treatment Response Analysis (KEY FEATURE)
        yPosition = drawMedResponseTreatmentAnalysis(ctx: ctx, medications: medications, logs: logs, yPosition: yPosition)
        yPosition += 10

        // Section 4: Medication Timeline
        yPosition = drawMedResponseTimeline(ctx: ctx, medications: medications, yPosition: yPosition)
        yPosition += 10

        // Section 5: Treatment History Summary
        _ = drawMedResponseHistorySummary(ctx: ctx, medications: medications, yPosition: yPosition)

        drawFooter(ctx: ctx, pageNumber: 4, totalPages: totalPages)
    }

    // MARK: - Medication Response: Section 1 - Current Medications Table

    private func drawMedResponseCurrentMedicationsTable(ctx: CGContext, medications: [Medication], yPosition: CGFloat) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Current Medications", yPosition: yPosition)

        let activeMeds = medications.filter { $0.isActive }

        if activeMeds.isEmpty {
            return drawText(ctx: ctx, text: "No active medications recorded.", yPosition: y, font: .systemFont(ofSize: 10), color: .gray)
        }

        // Table header
        let tableLeft = margin
        let colWidths: [CGFloat] = [115, 60, 75, 60, 48, 48]
        let headerTitles = ["Medication", "Category", "Dosage", "Started", "Route", "Biologic"]

        ctx.setFillColor(UIColor.systemBlue.withAlphaComponent(0.1).cgColor)
        ctx.fill(CGRect(x: tableLeft, y: y, width: pageSize.width - 2 * margin, height: 15))

        let headerAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8, weight: .semibold),
            .foregroundColor: UIColor.black
        ]

        var xOffset: CGFloat = tableLeft + 3
        for (index, title) in headerTitles.enumerated() {
            NSAttributedString(string: title, attributes: headerAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[index]
        }

        y += 16

        let rowAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8),
            .foregroundColor: UIColor.black
        ]

        let biologicAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8, weight: .bold),
            .foregroundColor: UIColor.systemPurple
        ]

        for (index, med) in activeMeds.prefix(5).enumerated() {
            if index % 2 == 1 {
                ctx.setFillColor(UIColor.systemGray6.cgColor)
                ctx.fill(CGRect(x: tableLeft, y: y, width: pageSize.width - 2 * margin, height: 13))
            }

            if med.isBiologic {
                ctx.setFillColor(UIColor.systemPurple.withAlphaComponent(0.08).cgColor)
                ctx.fill(CGRect(x: tableLeft, y: y, width: pageSize.width - 2 * margin, height: 13))
            }

            xOffset = tableLeft + 3

            // Name
            NSAttributedString(string: truncateMedString(med.name ?? "Unknown", maxLength: 17), attributes: med.isBiologic ? biologicAttributes : rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[0]

            // Category
            NSAttributedString(string: truncateMedString(med.category ?? "-", maxLength: 9), attributes: rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[1]

            // Dosage
            let dosage = "\(Int(med.dosage))\(med.dosageUnit ?? "mg") \(truncateMedString(med.frequency ?? "", maxLength: 5))"
            NSAttributedString(string: truncateMedString(dosage, maxLength: 11), attributes: rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[2]

            // Start Date
            let startDate = med.startDate != nil ? formatMedShortDate(med.startDate!) : "-"
            NSAttributedString(string: startDate, attributes: rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[3]

            // Route
            NSAttributedString(string: truncateMedString((med.route ?? "oral").capitalized, maxLength: 7), attributes: rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[4]

            // Biologic flag
            let bioText = med.isBiologic ? "YES" : "No"
            NSAttributedString(string: bioText, attributes: med.isBiologic ? biologicAttributes : rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))

            y += 13
        }

        ctx.setStrokeColor(UIColor.lightGray.cgColor)
        ctx.setLineWidth(0.5)
        ctx.stroke(CGRect(x: tableLeft, y: yPosition + 35, width: pageSize.width - 2 * margin, height: y - yPosition - 35))

        return y + 3
    }

    // MARK: - Medication Response: Section 2 - Adherence Details

    private func drawMedResponseAdherenceDetails(ctx: CGContext, medications: [Medication], yPosition: CGFloat) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Medication Adherence", yPosition: yPosition)

        let activeMeds = medications.filter { $0.isActive }

        if activeMeds.isEmpty {
            return drawText(ctx: ctx, text: "No active medications to analyze.", yPosition: y, font: .systemFont(ofSize: 10), color: .gray)
        }

        for med in activeMeds.prefix(4) {
            let name = med.name ?? "Unknown"
            let doseLogs = (med.doseLogs as? Set<DoseLog>) ?? []
            let totalDoses = doseLogs.count
            let takenDoses = doseLogs.filter { !$0.wasSkipped }.count
            let adherencePercent = totalDoses > 0 ? Double(takenDoses) / Double(totalDoses) * 100 : 0

            let adherenceColor = adherencePercent >= 80 ? UIColor.systemGreen :
                                 (adherencePercent >= 60 ? UIColor.systemOrange : UIColor.systemRed)

            // Name
            let nameAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 9, weight: .medium), .foregroundColor: UIColor.black]
            NSAttributedString(string: truncateMedString(name, maxLength: 18), attributes: nameAttributes).draw(at: CGPoint(x: margin, y: y))

            // Percentage
            let percentAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 9, weight: .bold), .foregroundColor: adherenceColor]
            let percentText = totalDoses > 0 ? String(format: "%.0f%%", adherencePercent) : "N/A"
            NSAttributedString(string: percentText, attributes: percentAttributes).draw(at: CGPoint(x: margin + 125, y: y))

            // Adherence bar
            let barX = margin + 160
            let barWidth: CGFloat = 110
            let barHeight: CGFloat = 8

            ctx.setFillColor(UIColor.systemGray5.cgColor)
            ctx.fill(CGRect(x: barX, y: y + 2, width: barWidth, height: barHeight))

            if totalDoses > 0 {
                ctx.setFillColor(adherenceColor.cgColor)
                ctx.fill(CGRect(x: barX, y: y + 2, width: barWidth * CGFloat(adherencePercent / 100), height: barHeight))
            }

            ctx.setStrokeColor(UIColor.lightGray.cgColor)
            ctx.setLineWidth(0.5)
            ctx.stroke(CGRect(x: barX, y: y + 2, width: barWidth, height: barHeight))

            // Dose count
            let countAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 7), .foregroundColor: UIColor.darkGray]
            NSAttributedString(string: "(\(takenDoses)/\(totalDoses))", attributes: countAttributes).draw(at: CGPoint(x: barX + barWidth + 5, y: y + 1))

            y += 12

            // Skip reasons (from DoseLog.skipReason)
            let skippedDoses = doseLogs.filter { $0.wasSkipped }
            if !skippedDoses.isEmpty {
                let skipReasons = Dictionary(grouping: skippedDoses) { $0.skipReason ?? "Not specified" }
                let topReasons = skipReasons.sorted { $0.value.count > $1.value.count }.prefix(2)
                if !topReasons.isEmpty {
                    let reasonsText = topReasons.map { "\($0.key) (\($0.value.count))" }.joined(separator: ", ")
                    let reasonsAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 7), .foregroundColor: UIColor.darkGray]
                    NSAttributedString(string: "  Skip reasons: \(truncateMedString(reasonsText, maxLength: 50))", attributes: reasonsAttributes).draw(at: CGPoint(x: margin, y: y))
                    y += 9
                }
            }
        }

        return y
    }

    // MARK: - Medication Response: Section 3 - Treatment Response Analysis (KEY FEATURE)

    /// Analyzes BASDAI 4 weeks before vs 4 weeks after each medication start date
    /// This helps doctors see if treatments are working!
    private func drawMedResponseTreatmentAnalysis(ctx: CGContext, medications: [Medication], logs: [SymptomLog], yPosition: CGFloat) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Treatment Response Analysis", yPosition: yPosition)

        let medsWithStartDate = medications.filter { $0.startDate != nil }

        if medsWithStartDate.isEmpty {
            return drawText(ctx: ctx, text: "No medications with start dates for response analysis.", yPosition: y, font: .systemFont(ofSize: 10), color: .gray)
        }

        let sortedLogs = logs.sorted { ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) }

        // Subtitle
        let subtitleAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.italicSystemFont(ofSize: 8), .foregroundColor: UIColor.darkGray]
        NSAttributedString(string: "BASDAI comparison: 4 weeks before vs 4 weeks after medication start", attributes: subtitleAttributes).draw(at: CGPoint(x: margin, y: y))
        y += 13

        // Table header
        let tableLeft = margin
        let colWidths: [CGFloat] = [105, 55, 55, 50, 95]
        let headerTitles = ["Medication", "Before", "After", "Delta", "Response"]

        ctx.setFillColor(UIColor.systemGreen.withAlphaComponent(0.1).cgColor)
        ctx.fill(CGRect(x: tableLeft, y: y, width: pageSize.width - 2 * margin, height: 13))

        let headerAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 8, weight: .semibold), .foregroundColor: UIColor.black]
        var xOffset: CGFloat = tableLeft + 3
        for (index, title) in headerTitles.enumerated() {
            NSAttributedString(string: title, attributes: headerAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[index]
        }
        y += 14

        let rowAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 8), .foregroundColor: UIColor.black]
        let medsToAnalyze = medsWithStartDate.sorted { ($0.startDate ?? Date.distantPast) < ($1.startDate ?? Date.distantPast) }

        for (index, med) in medsToAnalyze.prefix(5).enumerated() {
            guard let startDate = med.startDate else { continue }

            if index % 2 == 1 {
                ctx.setFillColor(UIColor.systemGray6.cgColor)
                ctx.fill(CGRect(x: tableLeft, y: y, width: pageSize.width - 2 * margin, height: 13))
            }

            xOffset = tableLeft + 3

            // Name
            NSAttributedString(string: truncateMedString(med.name ?? "Unknown", maxLength: 15), attributes: rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[0]

            // Calculate BASDAI 4 weeks before medication start
            let fourWeeksBefore = Calendar.current.date(byAdding: .day, value: -28, to: startDate) ?? startDate
            let logsBefore = sortedLogs.filter { log in
                guard let timestamp = log.timestamp else { return false }
                return timestamp >= fourWeeksBefore && timestamp < startDate
            }
            let avgBefore = logsBefore.isEmpty ? nil : logsBefore.reduce(0.0) { $0 + $1.basdaiScore } / Double(logsBefore.count)

            // Calculate BASDAI 4 weeks after medication start
            let fourWeeksAfter = Calendar.current.date(byAdding: .day, value: 28, to: startDate) ?? startDate
            let logsAfter = sortedLogs.filter { log in
                guard let timestamp = log.timestamp else { return false }
                return timestamp > startDate && timestamp <= fourWeeksAfter
            }
            let avgAfter = logsAfter.isEmpty ? nil : logsAfter.reduce(0.0) { $0 + $1.basdaiScore } / Double(logsAfter.count)

            // Before value
            NSAttributedString(string: avgBefore != nil ? String(format: "%.1f", avgBefore!) : "N/A", attributes: rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[1]

            // After value
            NSAttributedString(string: avgAfter != nil ? String(format: "%.1f", avgAfter!) : "N/A", attributes: rowAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[2]

            // Delta (improvement/worsening)
            var delta: Double? = nil
            var deltaColor = UIColor.gray
            if let before = avgBefore, let after = avgAfter {
                delta = after - before
                // Green = improvement (BASDAI decreased), Red = worsening, Orange = stable
                deltaColor = delta! < -0.5 ? .systemGreen : (delta! > 0.5 ? .systemRed : .systemOrange)
            }

            let deltaAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 8, weight: .bold), .foregroundColor: deltaColor]
            NSAttributedString(string: delta != nil ? String(format: "%+.1f", delta!) : "N/A", attributes: deltaAttributes).draw(at: CGPoint(x: xOffset, y: y + 2))
            xOffset += colWidths[3]

            // Visual response indicator (arrow)
            drawMedResponseIndicator(ctx: ctx, x: xOffset, y: y + 2, delta: delta, color: deltaColor)

            y += 13
        }

        ctx.setStrokeColor(UIColor.lightGray.cgColor)
        ctx.setLineWidth(0.5)
        ctx.stroke(CGRect(x: tableLeft, y: yPosition + 48, width: pageSize.width - 2 * margin, height: y - yPosition - 48))

        // Legend
        y += 4
        let legendAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 7), .foregroundColor: UIColor.darkGray]
        NSAttributedString(string: "Legend: Green down arrow = BASDAI improved (treatment effective) | Red up arrow = BASDAI worsened | Orange line = Stable", attributes: legendAttributes).draw(at: CGPoint(x: margin, y: y))

        return y + 10
    }

    /// Draws visual arrow indicator for treatment response
    private func drawMedResponseIndicator(ctx: CGContext, x: CGFloat, y: CGFloat, delta: Double?, color: UIColor) {
        guard let delta = delta else {
            let naAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 7), .foregroundColor: UIColor.gray]
            NSAttributedString(string: "Insufficient data", attributes: naAttributes).draw(at: CGPoint(x: x, y: y))
            return
        }

        ctx.setFillColor(color.cgColor)
        let arrowX = x + 3
        let arrowY = y + 4

        if delta < -0.5 {
            // Green down arrow (improvement - BASDAI decreased)
            ctx.move(to: CGPoint(x: arrowX, y: arrowY - 3))
            ctx.addLine(to: CGPoint(x: arrowX + 5, y: arrowY - 3))
            ctx.addLine(to: CGPoint(x: arrowX + 2.5, y: arrowY + 3))
            ctx.closePath()
            ctx.fillPath()
            NSAttributedString(string: "Improved", attributes: [.font: UIFont.systemFont(ofSize: 7, weight: .medium), .foregroundColor: color]).draw(at: CGPoint(x: arrowX + 10, y: y))
        } else if delta > 0.5 {
            // Red up arrow (worsening - BASDAI increased)
            ctx.move(to: CGPoint(x: arrowX, y: arrowY + 3))
            ctx.addLine(to: CGPoint(x: arrowX + 5, y: arrowY + 3))
            ctx.addLine(to: CGPoint(x: arrowX + 2.5, y: arrowY - 3))
            ctx.closePath()
            ctx.fillPath()
            NSAttributedString(string: "Worsened", attributes: [.font: UIFont.systemFont(ofSize: 7, weight: .medium), .foregroundColor: color]).draw(at: CGPoint(x: arrowX + 10, y: y))
        } else {
            // Orange horizontal line (stable)
            ctx.setLineWidth(2)
            ctx.setStrokeColor(color.cgColor)
            ctx.move(to: CGPoint(x: arrowX, y: arrowY))
            ctx.addLine(to: CGPoint(x: arrowX + 8, y: arrowY))
            ctx.strokePath()
            NSAttributedString(string: "Stable", attributes: [.font: UIFont.systemFont(ofSize: 7, weight: .medium), .foregroundColor: color]).draw(at: CGPoint(x: arrowX + 12, y: y))
        }
    }

    // MARK: - Medication Response: Section 4 - Timeline

    /// Draws medication timeline showing start/stop events
    private func drawMedResponseTimeline(ctx: CGContext, medications: [Medication], yPosition: CGFloat) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Medication Timeline", yPosition: yPosition)

        var events: [(date: Date, name: String, type: String)] = []
        for med in medications {
            if let startDate = med.startDate {
                events.append((startDate, med.name ?? "Unknown", "Started"))
            }
            if let endDate = med.endDate {
                events.append((endDate, med.name ?? "Unknown", "Stopped"))
            }
        }

        if events.isEmpty {
            return drawText(ctx: ctx, text: "No medication timeline events recorded.", yPosition: y, font: .systemFont(ofSize: 10), color: .gray)
        }

        events.sort { $0.date < $1.date }

        let timelineX = margin + 65
        ctx.setStrokeColor(UIColor.systemGray3.cgColor)
        ctx.setLineWidth(2)
        ctx.move(to: CGPoint(x: timelineX, y: y))
        ctx.addLine(to: CGPoint(x: timelineX, y: y + CGFloat(min(events.count, 5)) * 16 + 5))
        ctx.strokePath()

        for event in events.prefix(5) {
            // Date label
            let dateAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 8), .foregroundColor: UIColor.darkGray]
            NSAttributedString(string: formatMedShortDate(event.date), attributes: dateAttributes).draw(at: CGPoint(x: margin, y: y))

            // Event dot (green for started, red for stopped)
            let dotColor = event.type == "Started" ? UIColor.systemGreen : UIColor.systemRed
            ctx.setFillColor(dotColor.cgColor)
            ctx.fillEllipse(in: CGRect(x: timelineX - 4, y: y + 2, width: 8, height: 8))

            // Event description
            let eventAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 9, weight: .medium), .foregroundColor: dotColor]
            NSAttributedString(string: "\(event.type): \(truncateMedString(event.name, maxLength: 35))", attributes: eventAttributes).draw(at: CGPoint(x: timelineX + 10, y: y))

            y += 16
        }

        return y + 3
    }

    // MARK: - Medication Response: Section 5 - Treatment History Summary

    /// Draws summary of medications including biologic counts (important for insurance/prior auth)
    private func drawMedResponseHistorySummary(ctx: CGContext, medications: [Medication], yPosition: CGFloat) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Treatment Summary", yPosition: yPosition)

        let activeMeds = medications.filter { $0.isActive }
        let biologics = medications.filter { $0.isBiologic }
        let activeBiologics = activeMeds.filter { $0.isBiologic }

        let summaryItems: [(String, String, UIColor?)] = [
            ("Total Medications", "\(medications.count)", nil),
            ("Currently Active", "\(activeMeds.count)", nil),
            ("Biologics (Total)", "\(biologics.count)", biologics.isEmpty ? nil : UIColor.systemPurple),
            ("Active Biologics", "\(activeBiologics.count)", activeBiologics.isEmpty ? nil : UIColor.systemPurple),
            ("NSAIDs", "\(medications.filter { ($0.category ?? "").lowercased().contains("nsaid") }.count)", nil),
            ("DMARDs", "\(medications.filter { ($0.category ?? "").lowercased().contains("dmard") }.count)", nil)
        ]

        let labelAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 9), .foregroundColor: UIColor.darkGray]
        var col1Y = y
        var col2Y = y

        for (index, item) in summaryItems.enumerated() {
            let xPos = index % 2 == 0 ? margin : margin + 175
            let currentY = index % 2 == 0 ? col1Y : col2Y

            NSAttributedString(string: "\(item.0):", attributes: labelAttributes).draw(at: CGPoint(x: xPos, y: currentY))

            let valueAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 9, weight: .semibold), .foregroundColor: item.2 ?? UIColor.black]
            NSAttributedString(string: item.1, attributes: valueAttributes).draw(at: CGPoint(x: xPos + 90, y: currentY))

            if index % 2 == 0 { col1Y += 13 } else { col2Y += 13 }
        }

        y = max(col1Y, col2Y) + 5

        // Biologic medications note (important for insurance/prior authorization)
        if !biologics.isEmpty {
            let biologicNote = biologics.map { $0.name ?? "Unknown" }.joined(separator: ", ")
            let noteAttributes: [NSAttributedString.Key: Any] = [.font: UIFont.italicSystemFont(ofSize: 8), .foregroundColor: UIColor.systemPurple]
            NSAttributedString(string: "Biologic medications: \(truncateMedString(biologicNote, maxLength: 65))", attributes: noteAttributes).draw(at: CGPoint(x: margin, y: y))
            y += 11

            // Prior authorization note for insurance
            let priorAuthNote = NSAttributedString(string: "Note: Biologic medications typically require prior authorization for insurance coverage.", attributes: [.font: UIFont.italicSystemFont(ofSize: 7), .foregroundColor: UIColor.gray])
            priorAuthNote.draw(at: CGPoint(x: margin, y: y))
            y += 10
        }

        return y
    }

    // MARK: - Medication Response: Helper Methods

    private func formatMedShortDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        return formatter.string(from: date)
    }

    private func truncateMedString(_ string: String, maxLength: Int) -> String {
        if string.count <= maxLength { return string }
        return String(string.prefix(maxLength - 2)) + ".."
    }

    // MARK: - Page 5: Patient-Reported Outcomes (PRO)

    /// Draws the comprehensive Patient-Reported Outcomes page for rheumatologists.
    /// This captures the patient's perspective on their condition including:
    /// - Quality of Life (ASQOL)
    /// - Mental Health Summary with traffic light indicators
    /// - Fatigue Analysis with correlation note
    /// - Sleep Quality assessment
    /// - Patient Global Assessment with trends
    /// - Well-being Summary with energy and coping metrics
    private func drawPatientReportedOutcomesPage(
        context: UIGraphicsPDFRendererContext,
        logs: [SymptomLog],
        assessmentResponses: [QuestionnaireResponse],
        totalPages: Int
    ) {
        let ctx = context.cgContext
        let pageNumber = 5

        drawHeader(ctx: ctx, title: "Patient-Reported Outcomes (PRO)")

        var yPosition = headerHeight + 15

        // 1. Quality of Life Section (ASQOL)
        yPosition = drawPROASQOLSection(
            ctx: ctx,
            assessmentResponses: assessmentResponses,
            yPosition: yPosition
        )

        // 2. Mental Health Summary
        yPosition = drawPROMentalHealthSection(
            ctx: ctx,
            logs: logs,
            yPosition: yPosition
        )

        // 3. Fatigue Analysis
        yPosition = drawPROFatigueSection(
            ctx: ctx,
            logs: logs,
            yPosition: yPosition
        )

        // 4. Sleep Quality
        yPosition = drawPROSleepQualitySection(
            ctx: ctx,
            logs: logs,
            yPosition: yPosition
        )

        // 5. Patient Global Assessment
        yPosition = drawPROPatientGlobalSection(
            ctx: ctx,
            logs: logs,
            yPosition: yPosition
        )

        // 6. Well-being Summary Box
        _ = drawPROWellbeingSummaryBox(
            ctx: ctx,
            logs: logs,
            yPosition: yPosition
        )

        // Footer
        drawFooter(ctx: ctx, pageNumber: pageNumber, totalPages: totalPages)
    }

    // MARK: - PRO Section: ASQOL (Quality of Life)

    private func drawPROASQOLSection(
        ctx: CGContext,
        assessmentResponses: [QuestionnaireResponse],
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Quality of Life (ASQOL)", yPosition: yPosition)

        // Filter ASQOL responses
        let asqolResponses = assessmentResponses
            .filter { $0.questionnaireID?.lowercased() == "asqol" }
            .sorted { ($0.createdAt ?? Date.distantPast) < ($1.createdAt ?? Date.distantPast) }

        if asqolResponses.isEmpty {
            y = drawText(
                ctx: ctx,
                text: "No ASQOL assessments recorded in this period.",
                yPosition: y,
                font: .systemFont(ofSize: 11),
                color: .gray
            )
        } else {
            // Current score
            if let latestResponse = asqolResponses.last {
                let score = latestResponse.score
                let interpretation = proASQOLInterpretation(score: score)

                y = drawKeyValue(ctx: ctx, key: "Latest Score", value: String(format: "%.1f / 18", score), yPosition: y)
                y = drawKeyValue(ctx: ctx, key: "Interpretation", value: interpretation.text, yPosition: y)

                // Draw interpretation color indicator
                drawPROColorIndicator(ctx: ctx, color: interpretation.color, x: margin + 320, y: y - 35)

                y = drawKeyValue(ctx: ctx, key: "Assessment Date", value: formatDate(latestResponse.createdAt ?? Date()), yPosition: y)
            }

            // If multiple assessments, show trend summary
            if asqolResponses.count > 1 {
                let firstScore = asqolResponses.first?.score ?? 0
                let lastScore = asqolResponses.last?.score ?? 0
                let change = lastScore - firstScore
                let trendText = change < -1 ? "Improving" : (change > 1 ? "Declining" : "Stable")
                y = drawKeyValue(ctx: ctx, key: "Trend", value: "\(trendText) (\(asqolResponses.count) assessments)", yPosition: y)
            }
        }

        return y + 8
    }

    private func proASQOLInterpretation(score: Double) -> (text: String, color: UIColor) {
        // ASQOL: 0-18 scale, lower = better QoL
        switch score {
        case 0..<5:
            return ("Good Quality of Life", .systemGreen)
        case 5..<10:
            return ("Moderate Impact on QoL", .systemYellow)
        case 10..<14:
            return ("Significant Impact on QoL", .systemOrange)
        default:
            return ("Severe Impact on QoL", .systemRed)
        }
    }

    // MARK: - PRO Section: Mental Health Summary

    private func drawPROMentalHealthSection(
        ctx: CGContext,
        logs: [SymptomLog],
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Mental Health Summary", yPosition: yPosition)

        guard !logs.isEmpty else {
            y = drawText(ctx: ctx, text: "No symptom logs available.", yPosition: y, font: .systemFont(ofSize: 11), color: .gray)
            return y + 8
        }

        // Calculate averages
        let stressAvg = logs.map { Double($0.stressLevel) }.reduce(0, +) / Double(logs.count)
        let anxietyAvg = logs.map { Double($0.anxietyLevel) }.reduce(0, +) / Double(logs.count)
        let moodAvg = logs.map { Double($0.moodScore) }.reduce(0, +) / Double(logs.count)

        // Draw traffic light indicators in a row
        let indicatorWidth: CGFloat = 130
        let indicatorSpacing: CGFloat = 12

        // Stress Level (higher = worse)
        drawPROTrafficLightIndicator(
            ctx: ctx,
            label: "Stress",
            value: stressAvg,
            maxValue: 10,
            x: margin,
            y: y,
            width: indicatorWidth,
            isInverseScale: true
        )

        // Anxiety Level (higher = worse)
        drawPROTrafficLightIndicator(
            ctx: ctx,
            label: "Anxiety",
            value: anxietyAvg,
            maxValue: 10,
            x: margin + indicatorWidth + indicatorSpacing,
            y: y,
            width: indicatorWidth,
            isInverseScale: true
        )

        // Mood Score (higher = better)
        drawPROTrafficLightIndicator(
            ctx: ctx,
            label: "Mood",
            value: moodAvg,
            maxValue: 10,
            x: margin + 2 * (indicatorWidth + indicatorSpacing),
            y: y,
            width: indicatorWidth,
            isInverseScale: false
        )

        y += 55

        // Overall mental health risk assessment
        let mentalHealthRisk = calculatePROMentalHealthRisk(stress: stressAvg, anxiety: anxietyAvg, mood: moodAvg)
        y = drawPROMentalHealthRiskBox(ctx: ctx, risk: mentalHealthRisk, yPosition: y)

        return y + 8
    }

    private func calculatePROMentalHealthRisk(stress: Double, anxiety: Double, mood: Double) -> PROMentalHealthRisk {
        // Calculate composite risk score
        // Stress and anxiety contribute positively to risk, mood contributes inversely
        let riskScore = (stress + anxiety + (10 - mood)) / 3

        if riskScore < 3 {
            return .low
        } else if riskScore < 6 {
            return .moderate
        } else {
            return .high
        }
    }

    private enum PROMentalHealthRisk {
        case low, moderate, high

        var color: UIColor {
            switch self {
            case .low: return .systemGreen
            case .moderate: return .systemYellow
            case .high: return .systemRed
            }
        }

        var label: String {
            switch self {
            case .low: return "Low Risk"
            case .moderate: return "Moderate - Monitor"
            case .high: return "High Risk - Needs Attention"
            }
        }

        var recommendation: String {
            switch self {
            case .low:
                return "Mental health indicators within normal range."
            case .moderate:
                return "Consider discussing stress management strategies."
            case .high:
                return "Recommend mental health screening or referral."
            }
        }
    }

    private func drawPROMentalHealthRiskBox(ctx: CGContext, risk: PROMentalHealthRisk, yPosition: CGFloat) -> CGFloat {
        let boxRect = CGRect(x: margin, y: yPosition, width: pageSize.width - 2 * margin, height: 36)

        // Background
        ctx.setFillColor(risk.color.withAlphaComponent(0.1).cgColor)
        ctx.fill(boxRect)

        // Border
        ctx.setStrokeColor(risk.color.cgColor)
        ctx.setLineWidth(2)
        ctx.stroke(boxRect)

        // Traffic light circle
        ctx.setFillColor(risk.color.cgColor)
        ctx.fillEllipse(in: CGRect(x: margin + 8, y: yPosition + 8, width: 18, height: 18))

        // Risk label
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 10, weight: .bold),
            .foregroundColor: UIColor.black
        ]
        let labelString = NSAttributedString(string: risk.label, attributes: labelAttributes)
        labelString.draw(at: CGPoint(x: margin + 34, y: yPosition + 5))

        // Recommendation
        let recAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9),
            .foregroundColor: UIColor.darkGray
        ]
        let recString = NSAttributedString(string: risk.recommendation, attributes: recAttributes)
        recString.draw(at: CGPoint(x: margin + 34, y: yPosition + 19))

        return yPosition + 44
    }

    // MARK: - PRO Section: Fatigue Analysis

    private func drawPROFatigueSection(
        ctx: CGContext,
        logs: [SymptomLog],
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Fatigue Analysis", yPosition: yPosition)

        guard !logs.isEmpty else {
            y = drawText(ctx: ctx, text: "No fatigue data available.", yPosition: y, font: .systemFont(ofSize: 11), color: .gray)
            return y + 8
        }

        // Calculate fatigue statistics
        let fatigueLevels = logs.map { Double($0.fatigueLevel) }
        let avgFatigue = fatigueLevels.reduce(0, +) / Double(fatigueLevels.count)
        let maxFatigue = fatigueLevels.max() ?? 0
        let minFatigue = fatigueLevels.min() ?? 0

        // Draw statistics with color indicator
        let fatigueColor = proColorForFatigue(avgFatigue)

        y = drawKeyValue(ctx: ctx, key: "Average Fatigue", value: String(format: "%.1f / 10", avgFatigue), yPosition: y)
        drawPROColorIndicator(ctx: ctx, color: fatigueColor, x: margin + 200, y: y - 15)

        y = drawKeyValue(ctx: ctx, key: "Range", value: String(format: "%.0f - %.0f", minFatigue, maxFatigue), yPosition: y)

        // Clinical note about fatigue in AS
        let noteAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.italicSystemFont(ofSize: 9),
            .foregroundColor: UIColor.gray
        ]
        let noteString = NSAttributedString(
            string: "Note: Fatigue often correlates with disease activity in AS.",
            attributes: noteAttributes
        )
        noteString.draw(at: CGPoint(x: margin, y: y))

        return y + 18
    }

    private func proColorForFatigue(_ value: Double) -> UIColor {
        switch value {
        case 0..<3: return .systemGreen
        case 3..<6: return .systemYellow
        case 6..<8: return .systemOrange
        default: return .systemRed
        }
    }

    // MARK: - PRO Section: Sleep Quality

    private func drawPROSleepQualitySection(
        ctx: CGContext,
        logs: [SymptomLog],
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Sleep Quality", yPosition: yPosition)

        guard !logs.isEmpty else {
            y = drawText(ctx: ctx, text: "No sleep data available.", yPosition: y, font: .systemFont(ofSize: 11), color: .gray)
            return y + 8
        }

        // Calculate sleep statistics
        let sleepQualities = logs.map { Double($0.sleepQuality) }
        let avgSleepQuality = sleepQualities.reduce(0, +) / Double(sleepQualities.count)

        let sleepDurations = logs.map { $0.sleepDurationHours }
        let avgSleepDuration = sleepDurations.reduce(0, +) / Double(sleepDurations.count)

        // Draw statistics
        let qualityColor = proColorForSleepQuality(avgSleepQuality)

        y = drawKeyValue(ctx: ctx, key: "Avg Sleep Quality", value: String(format: "%.1f / 10", avgSleepQuality), yPosition: y)
        drawPROColorIndicator(ctx: ctx, color: qualityColor, x: margin + 200, y: y - 15)

        y = drawKeyValue(ctx: ctx, key: "Avg Sleep Duration", value: String(format: "%.1f hours", avgSleepDuration), yPosition: y)

        // Sleep duration assessment
        let durationAssessment: String
        if avgSleepDuration < 6 {
            durationAssessment = "Below recommended (7-9h)"
        } else if avgSleepDuration < 7 {
            durationAssessment = "Slightly below recommended"
        } else if avgSleepDuration <= 9 {
            durationAssessment = "Within recommended range"
        } else {
            durationAssessment = "Above typical range"
        }

        y = drawKeyValue(ctx: ctx, key: "Assessment", value: durationAssessment, yPosition: y)

        return y + 8
    }

    private func proColorForSleepQuality(_ value: Double) -> UIColor {
        switch value {
        case 7...10: return .systemGreen
        case 5..<7: return .systemYellow
        case 3..<5: return .systemOrange
        default: return .systemRed
        }
    }

    // MARK: - PRO Section: Patient Global Assessment

    private func drawPROPatientGlobalSection(
        ctx: CGContext,
        logs: [SymptomLog],
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Patient Global Assessment", yPosition: yPosition)

        guard !logs.isEmpty else {
            y = drawText(ctx: ctx, text: "No patient global data available.", yPosition: y, font: .systemFont(ofSize: 11), color: .gray)
            return y + 8
        }

        // Patient Global (VAS-style)
        let patientGlobals = logs.map { Double($0.patientGlobal) }
        let avgPatientGlobal = patientGlobals.reduce(0, +) / Double(patientGlobals.count)

        // Overall Feeling
        let overallFeelings = logs.map { Double($0.overallFeeling) }
        let avgOverallFeeling = overallFeelings.reduce(0, +) / Double(overallFeelings.count)

        // Day Quality
        let dayQualities = logs.map { Double($0.dayQuality) }
        let avgDayQuality = dayQualities.reduce(0, +) / Double(dayQualities.count)

        // Draw statistics
        if avgPatientGlobal > 0 {
            y = drawKeyValue(ctx: ctx, key: "Patient Global (VAS)", value: String(format: "%.1f / 10", avgPatientGlobal), yPosition: y)
            drawPROColorIndicator(ctx: ctx, color: proColorForPatientGlobal(avgPatientGlobal), x: margin + 250, y: y - 15)
        }

        y = drawKeyValue(ctx: ctx, key: "Overall Feeling", value: String(format: "%.1f / 10", avgOverallFeeling), yPosition: y)
        drawPROColorIndicator(ctx: ctx, color: proColorForWellbeing(avgOverallFeeling), x: margin + 250, y: y - 15)

        y = drawKeyValue(ctx: ctx, key: "Day Quality", value: String(format: "%.1f / 10", avgDayQuality), yPosition: y)
        drawPROColorIndicator(ctx: ctx, color: proColorForWellbeing(avgDayQuality), x: margin + 250, y: y - 15)

        // Trend indicator if sufficient data
        if logs.count >= 7 {
            let sortedLogs = logs.sorted { ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) }
            let halfCount = max(sortedLogs.count / 2, 1)
            let recentHalf = Array(sortedLogs.suffix(halfCount))
            let earlierHalf = Array(sortedLogs.prefix(halfCount))

            guard !recentHalf.isEmpty && !earlierHalf.isEmpty else {
                return y + 8
            }

            let recentAvg = recentHalf.map { Double($0.overallFeeling) }.reduce(0, +) / Double(recentHalf.count)
            let earlierAvg = earlierHalf.map { Double($0.overallFeeling) }.reduce(0, +) / Double(earlierHalf.count)

            let trend: String
            let trendColor: UIColor
            if recentAvg > earlierAvg + 0.5 {
                trend = "Improving"
                trendColor = .systemGreen
            } else if recentAvg < earlierAvg - 0.5 {
                trend = "Declining"
                trendColor = .systemRed
            } else {
                trend = "Stable"
                trendColor = .systemBlue
            }

            let trendAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 10, weight: .medium),
                .foregroundColor: trendColor
            ]
            let trendString = NSAttributedString(string: "Trend: \(trend)", attributes: trendAttributes)
            trendString.draw(at: CGPoint(x: margin, y: y))
            y += 15
        }

        return y + 8
    }

    private func proColorForPatientGlobal(_ value: Double) -> UIColor {
        // Patient Global: higher = worse (disease activity scale)
        switch value {
        case 0..<3: return .systemGreen
        case 3..<5: return .systemYellow
        case 5..<7: return .systemOrange
        default: return .systemRed
        }
    }

    private func proColorForWellbeing(_ value: Double) -> UIColor {
        // Wellbeing: higher = better
        switch value {
        case 7...10: return .systemGreen
        case 5..<7: return .systemYellow
        case 3..<5: return .systemOrange
        default: return .systemRed
        }
    }

    // MARK: - PRO Section: Well-being Summary Box

    private func drawPROWellbeingSummaryBox(
        ctx: CGContext,
        logs: [SymptomLog],
        yPosition: CGFloat
    ) -> CGFloat {
        var y = drawSection(ctx: ctx, title: "Well-being Summary", yPosition: yPosition)

        guard !logs.isEmpty else {
            y = drawText(ctx: ctx, text: "Insufficient data for well-being summary.", yPosition: y, font: .systemFont(ofSize: 11), color: .gray)
            return y + 8
        }

        // Calculate well-being metrics
        let energyLevels = logs.map { Double($0.energyLevel) }
        let avgEnergy = energyLevels.reduce(0, +) / Double(energyLevels.count)

        let copingAbilities = logs.map { Double($0.copingAbility) }
        let avgCoping = copingAbilities.reduce(0, +) / Double(copingAbilities.count)

        // Draw summary box
        let boxRect = CGRect(x: margin, y: y, width: pageSize.width - 2 * margin, height: 65)

        // Background
        ctx.setFillColor(UIColor.systemGray6.cgColor)
        let path = UIBezierPath(roundedRect: boxRect, cornerRadius: 8)
        ctx.addPath(path.cgPath)
        ctx.fillPath()

        // Border
        ctx.setStrokeColor(UIColor.systemGray4.cgColor)
        ctx.setLineWidth(1)
        ctx.addPath(path.cgPath)
        ctx.strokePath()

        // Content inside box
        let contentX = margin + 12
        var contentY = y + 10

        // Energy Level
        let energyLabel = NSAttributedString(
            string: "Energy Level:",
            attributes: [.font: UIFont.systemFont(ofSize: 10, weight: .semibold), .foregroundColor: UIColor.black]
        )
        energyLabel.draw(at: CGPoint(x: contentX, y: contentY))

        let energyValue = NSAttributedString(
            string: String(format: "%.1f / 10", avgEnergy),
            attributes: [.font: UIFont.systemFont(ofSize: 10), .foregroundColor: UIColor.darkGray]
        )
        energyValue.draw(at: CGPoint(x: contentX + 80, y: contentY))

        // Energy bar
        drawPROProgressBar(ctx: ctx, value: avgEnergy, maxValue: 10, x: contentX + 145, y: contentY + 2, width: 100, height: 10)

        contentY += 18

        // Coping Ability
        let copingLabel = NSAttributedString(
            string: "Coping Ability:",
            attributes: [.font: UIFont.systemFont(ofSize: 10, weight: .semibold), .foregroundColor: UIColor.black]
        )
        copingLabel.draw(at: CGPoint(x: contentX, y: contentY))

        let copingValue = NSAttributedString(
            string: String(format: "%.1f / 10", avgCoping),
            attributes: [.font: UIFont.systemFont(ofSize: 10), .foregroundColor: UIColor.darkGray]
        )
        copingValue.draw(at: CGPoint(x: contentX + 80, y: contentY))

        // Coping bar
        drawPROProgressBar(ctx: ctx, value: avgCoping, maxValue: 10, x: contentX + 145, y: contentY + 2, width: 100, height: 10)

        contentY += 18

        // Overall assessment
        let overallScore = (avgEnergy + avgCoping) / 2
        let assessment: String
        let assessmentColor: UIColor

        if overallScore >= 7 {
            assessment = "Good overall well-being"
            assessmentColor = .systemGreen
        } else if overallScore >= 5 {
            assessment = "Moderate well-being - may benefit from support"
            assessmentColor = .systemYellow
        } else {
            assessment = "Low well-being - consider supportive interventions"
            assessmentColor = .systemRed
        }

        let assessmentAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9, weight: .medium),
            .foregroundColor: assessmentColor
        ]
        let assessmentString = NSAttributedString(string: assessment, attributes: assessmentAttributes)
        assessmentString.draw(at: CGPoint(x: contentX, y: contentY))

        return y + 75
    }

    // MARK: - PRO Helper Drawing Methods

    private func drawPROTrafficLightIndicator(
        ctx: CGContext,
        label: String,
        value: Double,
        maxValue: Double,
        x: CGFloat,
        y: CGFloat,
        width: CGFloat,
        isInverseScale: Bool
    ) {
        // Determine color based on value and scale direction
        let normalizedValue = isInverseScale ? value : (maxValue - value)
        let color: UIColor
        if normalizedValue < maxValue * 0.3 {
            color = .systemGreen
        } else if normalizedValue < maxValue * 0.6 {
            color = .systemYellow
        } else {
            color = .systemRed
        }

        // Draw background
        let boxRect = CGRect(x: x, y: y, width: width, height: 45)
        ctx.setFillColor(color.withAlphaComponent(0.1).cgColor)
        ctx.fill(boxRect)

        // Draw border
        ctx.setStrokeColor(color.cgColor)
        ctx.setLineWidth(1.5)
        ctx.stroke(boxRect)

        // Draw traffic light circle
        ctx.setFillColor(color.cgColor)
        ctx.fillEllipse(in: CGRect(x: x + 6, y: y + 6, width: 12, height: 12))

        // Draw label
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9, weight: .medium),
            .foregroundColor: UIColor.black
        ]
        let labelString = NSAttributedString(string: label, attributes: labelAttributes)
        labelString.draw(at: CGPoint(x: x + 24, y: y + 5))

        // Draw value
        let valueAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 14, weight: .bold),
            .foregroundColor: color
        ]
        let valueString = NSAttributedString(string: String(format: "%.1f", value), attributes: valueAttributes)
        valueString.draw(at: CGPoint(x: x + 6, y: y + 22))

        // Draw max value
        let maxAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8),
            .foregroundColor: UIColor.gray
        ]
        let maxString = NSAttributedString(string: "/ \(Int(maxValue))", attributes: maxAttributes)
        maxString.draw(at: CGPoint(x: x + 40, y: y + 26))
    }

    private func drawPROColorIndicator(ctx: CGContext, color: UIColor, x: CGFloat, y: CGFloat) {
        ctx.setFillColor(color.cgColor)
        ctx.fillEllipse(in: CGRect(x: x, y: y, width: 10, height: 10))
    }

    private func drawPROProgressBar(ctx: CGContext, value: Double, maxValue: Double, x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat) {
        // Background
        ctx.setFillColor(UIColor.systemGray5.cgColor)
        ctx.fill(CGRect(x: x, y: y, width: width, height: height))

        // Determine color based on value
        let color: UIColor
        let ratio = value / maxValue
        if ratio >= 0.7 {
            color = .systemGreen
        } else if ratio >= 0.4 {
            color = .systemYellow
        } else {
            color = .systemRed
        }

        // Filled portion
        let filledWidth = width * CGFloat(min(ratio, 1.0))
        ctx.setFillColor(color.cgColor)
        ctx.fill(CGRect(x: x, y: y, width: filledWidth, height: height))

        // Border
        ctx.setStrokeColor(UIColor.systemGray4.cgColor)
        ctx.setLineWidth(0.5)
        ctx.stroke(CGRect(x: x, y: y, width: width, height: height))
    }

    // MARK: - Page 6: Charts (BASDAI Trends & Pain Heatmap)

    private func drawChartsPage(context: UIGraphicsPDFRendererContext, logs: [SymptomLog], dateRange: (start: Date, end: Date), pageNumber: Int = 6, totalPages: Int) {
        let ctx = context.cgContext

        drawHeader(ctx: ctx, title: "Symptom Trends & Analysis")

        var yPosition = headerHeight + 20

        // BASDAI Trend Chart
        yPosition = drawSection(ctx: ctx, title: "BASDAI Trend", yPosition: yPosition)

        let chartRect = CGRect(
            x: margin,
            y: yPosition,
            width: pageSize.width - 2 * margin,
            height: 200
        )

        drawBASDATrendChart(ctx: ctx, logs: logs, rect: chartRect)
        yPosition += 220

        // Body Heatmap
        yPosition = drawSection(ctx: ctx, title: "Pain Distribution Heatmap", yPosition: yPosition)

        let heatmapRect = CGRect(
            x: margin,
            y: yPosition,
            width: pageSize.width - 2 * margin,
            height: 300
        )

        drawBodyHeatmap(ctx: ctx, logs: logs, rect: heatmapRect)

        drawFooter(ctx: ctx, pageNumber: pageNumber, totalPages: totalPages)
    }

    // MARK: - Page 7: Insights (Flare Events & Medication List)

    private func drawInsightsPage(context: UIGraphicsPDFRendererContext, flareEvents: [FlareEvent], medications: [Medication], logs: [SymptomLog], pageNumber: Int = 7, totalPages: Int) {
        let ctx = context.cgContext

        drawHeader(ctx: ctx, title: "Insights & Medication Adherence")

        var yPosition = headerHeight + 20

        // Flare Events Summary
        yPosition = drawSection(ctx: ctx, title: "Flare Events", yPosition: yPosition)

        if flareEvents.isEmpty {
            yPosition = drawText(ctx: ctx, text: "No flare events recorded in this period.", yPosition: yPosition, font: .systemFont(ofSize: 12), color: .gray)
        } else {
            yPosition = drawText(ctx: ctx, text: "Total flare events: \(flareEvents.count)", yPosition: yPosition, font: .systemFont(ofSize: 12), color: .black)

            // Group by severity
            let severityCounts = Dictionary(grouping: flareEvents) { $0.severity }.mapValues { $0.count }
            for (severity, count) in severityCounts.sorted(by: { $0.key > $1.key }) {
                let severityText = "Severity \(severity): \(count) event(s)"
                yPosition = drawText(ctx: ctx, text: "  • \(severityText)", yPosition: yPosition, font: .systemFont(ofSize: 11), color: .darkGray)
            }
        }

        yPosition += 20

        // Medication Adherence
        yPosition = drawSection(ctx: ctx, title: "Medication Adherence", yPosition: yPosition)

        if medications.isEmpty {
            yPosition = drawText(ctx: ctx, text: "No medications recorded.", yPosition: yPosition, font: .systemFont(ofSize: 12), color: .gray)
        } else {
            for med in medications {
                yPosition = drawMedicationRow(ctx: ctx, medication: med, yPosition: yPosition)
            }
        }

        yPosition += 20

        // Clinical Notes
        yPosition = drawSection(ctx: ctx, title: "Notes for Healthcare Provider", yPosition: yPosition)

        let notes = generateClinicalNotes(logs: logs, flareEvents: flareEvents)
        yPosition = drawText(ctx: ctx, text: notes, yPosition: yPosition, font: .systemFont(ofSize: 11), color: .darkGray, maxWidth: pageSize.width - 2 * margin)

        drawFooter(ctx: ctx, pageNumber: pageNumber, totalPages: totalPages)
    }

    // MARK: - Drawing Utilities

    private func drawHeader(ctx: CGContext, title: String) {
        // Background
        ctx.setFillColor(UIColor.systemBlue.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: pageSize.width, height: headerHeight))

        // Title
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 24, weight: .bold),
            .foregroundColor: UIColor.white
        ]

        let titleString = NSAttributedString(string: title, attributes: titleAttributes)
        titleString.draw(at: CGPoint(x: margin, y: 40))

        // Logo placeholder (InflamAI icon)
        let logoRect = CGRect(x: pageSize.width - margin - 60, y: 20, width: 60, height: 60)
        ctx.setFillColor(UIColor.white.withAlphaComponent(0.3).cgColor)
        ctx.fillEllipse(in: logoRect)
    }

    private func drawFooter(ctx: CGContext, pageNumber: Int, totalPages: Int) {
        let footerY = pageSize.height - footerHeight

        // Divider line
        ctx.setStrokeColor(UIColor.lightGray.cgColor)
        ctx.setLineWidth(1)
        ctx.move(to: CGPoint(x: margin, y: footerY))
        ctx.addLine(to: CGPoint(x: pageSize.width - margin, y: footerY))
        ctx.strokePath()

        // Footer text
        let footerAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 10),
            .foregroundColor: UIColor.gray
        ]

        let leftText = NSAttributedString(string: "Generated by InflamAI • \(formatDate(Date()))", attributes: footerAttributes)
        leftText.draw(at: CGPoint(x: margin, y: footerY + 10))

        let rightText = NSAttributedString(string: "Page \(pageNumber) of \(totalPages)", attributes: footerAttributes)
        let rightTextSize = rightText.size()
        rightText.draw(at: CGPoint(x: pageSize.width - margin - rightTextSize.width, y: footerY + 10))
    }

    private func drawSection(ctx: CGContext, title: String, yPosition: CGFloat) -> CGFloat {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 16, weight: .semibold),
            .foregroundColor: UIColor.black
        ]

        let titleString = NSAttributedString(string: title, attributes: attributes)
        titleString.draw(at: CGPoint(x: margin, y: yPosition))

        // Underline
        ctx.setStrokeColor(UIColor.systemBlue.cgColor)
        ctx.setLineWidth(2)
        ctx.move(to: CGPoint(x: margin, y: yPosition + 22))
        ctx.addLine(to: CGPoint(x: margin + titleString.size().width, y: yPosition + 22))
        ctx.strokePath()

        return yPosition + 35
    }

    private func drawKeyValue(ctx: CGContext, key: String, value: String, yPosition: CGFloat) -> CGFloat {
        let keyAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 12, weight: .medium),
            .foregroundColor: UIColor.darkGray
        ]

        let valueAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 12),
            .foregroundColor: UIColor.black
        ]

        let keyString = NSAttributedString(string: "\(key):", attributes: keyAttributes)
        keyString.draw(at: CGPoint(x: margin, y: yPosition))

        let valueString = NSAttributedString(string: value, attributes: valueAttributes)
        valueString.draw(at: CGPoint(x: margin + 150, y: yPosition))

        return yPosition + 20
    }

    private func drawText(ctx: CGContext, text: String, yPosition: CGFloat, font: UIFont, color: UIColor, maxWidth: CGFloat? = nil) -> CGFloat {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: color
        ]

        let textString = NSAttributedString(string: text, attributes: attributes)

        if let maxWidth = maxWidth {
            let rect = CGRect(x: margin, y: yPosition, width: maxWidth, height: CGFloat.greatestFiniteMagnitude)
            textString.draw(with: rect, options: [.usesLineFragmentOrigin], context: nil)
            let size = textString.boundingRect(with: CGSize(width: maxWidth, height: .greatestFiniteMagnitude), options: [.usesLineFragmentOrigin], context: nil)
            return yPosition + size.height + 10
        } else {
            textString.draw(at: CGPoint(x: margin, y: yPosition))
            return yPosition + textString.size().height + 10
        }
    }

    private func drawDisclaimer(ctx: CGContext, yPosition: CGFloat) {
        let disclaimer = "DISCLAIMER: This report contains self-documented patient data to support clinical conversations. InflamAI is not a medical device and does not provide clinical assessments, diagnoses, or medical recommendations. All data should be verified during clinical examination. This app does not replace professional medical advice."

        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 9),
            .foregroundColor: UIColor.gray
        ]

        let disclaimerString = NSAttributedString(string: disclaimer, attributes: attributes)
        let rect = CGRect(x: margin, y: yPosition, width: pageSize.width - 2 * margin, height: 80)
        disclaimerString.draw(with: rect, options: [.usesLineFragmentOrigin], context: nil)
    }

    // MARK: - Chart Drawing

    private func drawBASDATrendChart(ctx: CGContext, logs: [SymptomLog], rect: CGRect) {
        // Draw axes
        ctx.setStrokeColor(UIColor.black.cgColor)
        ctx.setLineWidth(1)

        // Y-axis
        ctx.move(to: CGPoint(x: rect.minX, y: rect.minY))
        ctx.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))

        // X-axis
        ctx.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        ctx.strokePath()

        // Y-axis labels (0-10)
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 8),
            .foregroundColor: UIColor.black
        ]

        for i in 0...10 {
            let y = rect.maxY - (CGFloat(i) / 10.0) * rect.height
            let label = NSAttributedString(string: "\(i)", attributes: labelAttributes)
            label.draw(at: CGPoint(x: rect.minX - 20, y: y - 5))

            // Grid line
            ctx.setStrokeColor(UIColor.lightGray.withAlphaComponent(0.3).cgColor)
            ctx.move(to: CGPoint(x: rect.minX, y: y))
            ctx.addLine(to: CGPoint(x: rect.maxX, y: y))
            ctx.strokePath()
        }

        // Plot data
        guard logs.count > 1 else { return }

        let sortedLogs = logs.sorted { ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) }
        let xStep = rect.width / CGFloat(max(sortedLogs.count - 1, 1))

        ctx.setStrokeColor(UIColor.systemBlue.cgColor)
        ctx.setLineWidth(2)

        for (index, log) in sortedLogs.enumerated() {
            let x = rect.minX + CGFloat(index) * xStep
            let y = rect.maxY - (log.basdaiScore / 10.0) * rect.height

            if index == 0 {
                ctx.move(to: CGPoint(x: x, y: y))
            } else {
                ctx.addLine(to: CGPoint(x: x, y: y))
            }
        }

        ctx.strokePath()

        // Plot points
        ctx.setFillColor(UIColor.systemBlue.cgColor)
        for (index, log) in sortedLogs.enumerated() {
            let x = rect.minX + CGFloat(index) * xStep
            let y = rect.maxY - (log.basdaiScore / 10.0) * rect.height

            ctx.fillEllipse(in: CGRect(x: x - 3, y: y - 3, width: 6, height: 6))

            // Mark flares
            if log.isFlareEvent {
                ctx.setFillColor(UIColor.red.cgColor)
                ctx.fillEllipse(in: CGRect(x: x - 5, y: y - 5, width: 10, height: 10))
                ctx.setFillColor(UIColor.systemBlue.cgColor)
            }
        }

        // Chart title
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 10, weight: .medium),
            .foregroundColor: UIColor.black
        ]

        let title = NSAttributedString(string: "BASDAI Score Over Time", attributes: titleAttributes)
        title.draw(at: CGPoint(x: rect.minX, y: rect.minY - 20))
    }

    private func drawBodyHeatmap(ctx: CGContext, logs: [SymptomLog], rect: CGRect) {
        // Draw simplified body outline
        let bodyWidth: CGFloat = 100
        let bodyHeight: CGFloat = 250
        let centerX = rect.midX
        let startY = rect.minY + 20

        // Head
        ctx.setFillColor(UIColor.systemGray5.cgColor)
        ctx.fillEllipse(in: CGRect(x: centerX - 20, y: startY, width: 40, height: 40))

        // Torso
        ctx.fill(CGRect(x: centerX - 30, y: startY + 40, width: 60, height: 100))

        // Legs
        ctx.fill(CGRect(x: centerX - 30, y: startY + 140, width: 25, height: 80))
        ctx.fill(CGRect(x: centerX + 5, y: startY + 140, width: 25, height: 80))

        // Spine overlay (simplified - show C, T, L regions)
        let spineX = centerX
        let spineTop = startY + 50

        // Calculate average pain per region
        var regionPain: [String: Double] = [:]

        for log in logs {
            if let bodyRegions = log.bodyRegionLogs as? Set<BodyRegionLog> {
                for region in bodyRegions {
                    let regionID = region.regionID ?? ""
                    regionPain[regionID, default: 0] += Double(region.painLevel)
                }
            }
        }

        // Average out
        for (key, total) in regionPain {
            let count = logs.filter { log in
                (log.bodyRegionLogs as? Set<BodyRegionLog>)?.contains { $0.regionID == key } ?? false
            }.count
            regionPain[key] = total / Double(max(count, 1))
        }

        // Draw heat indicators for cervical, thoracic, lumbar
        let cervicalPain = averagePainForCategory(regionPain: regionPain, prefix: "c")
        let thoracicPain = averagePainForCategory(regionPain: regionPain, prefix: "t")
        let lumbarPain = averagePainForCategory(regionPain: regionPain, prefix: "l")

        drawHeatCircle(ctx: ctx, x: spineX, y: spineTop, pain: cervicalPain, label: "C")
        drawHeatCircle(ctx: ctx, x: spineX, y: spineTop + 40, pain: thoracicPain, label: "T")
        drawHeatCircle(ctx: ctx, x: spineX, y: spineTop + 80, pain: lumbarPain, label: "L")

        // Legend
        drawHeatmapLegend(ctx: ctx, rect: rect)
    }

    private func averagePainForCategory(regionPain: [String: Double], prefix: String) -> Double {
        let values = regionPain.filter { $0.key.lowercased().hasPrefix(prefix) }.values
        return values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
    }

    private func drawHeatCircle(ctx: CGContext, x: CGFloat, y: CGFloat, pain: Double, label: String) {
        let color = heatColor(for: pain)
        ctx.setFillColor(color.cgColor)
        ctx.fillEllipse(in: CGRect(x: x - 15, y: y - 15, width: 30, height: 30))

        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 12, weight: .bold),
            .foregroundColor: UIColor.white
        ]

        let labelString = NSAttributedString(string: label, attributes: labelAttributes)
        labelString.draw(at: CGPoint(x: x - 5, y: y - 7))
    }

    private func drawHeatmapLegend(ctx: CGContext, rect: CGRect) {
        let legendY = rect.maxY - 30

        let labels = ["0-3", "4-6", "7-10"]
        let colors = [UIColor.green, UIColor.orange, UIColor.red]

        for (index, (label, color)) in zip(labels, colors).enumerated() {
            let x = rect.minX + CGFloat(index) * 120

            ctx.setFillColor(color.cgColor)
            ctx.fillEllipse(in: CGRect(x: x, y: legendY, width: 20, height: 20))

            let labelAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 10),
                .foregroundColor: UIColor.black
            ]

            let labelString = NSAttributedString(string: label, attributes: labelAttributes)
            labelString.draw(at: CGPoint(x: x + 25, y: legendY + 3))
        }
    }

    private func heatColor(for pain: Double) -> UIColor {
        switch pain {
        case 0..<3: return .green
        case 3..<7: return .orange
        default: return .red
        }
    }

    // MARK: - Row Drawing

    private func drawMedicationRow(ctx: CGContext, medication: Medication, yPosition: CGFloat) -> CGFloat {
        let name = medication.name ?? "Unknown"
        let dosage = "\(medication.dosage) \(medication.dosageUnit ?? "mg")"

        let nameAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 11, weight: .medium),
            .foregroundColor: UIColor.black
        ]

        let dosageAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 10),
            .foregroundColor: UIColor.darkGray
        ]

        let nameString = NSAttributedString(string: name, attributes: nameAttributes)
        nameString.draw(at: CGPoint(x: margin, y: yPosition))

        let dosageString = NSAttributedString(string: dosage, attributes: dosageAttributes)
        dosageString.draw(at: CGPoint(x: margin + 200, y: yPosition))

        return yPosition + 18
    }

    // MARK: - Clinical Notes

    private func generateClinicalNotes(logs: [SymptomLog], flareEvents: [FlareEvent]) -> String {
        var notes = ""

        // Average BASDAI
        let avgBASDai = logs.isEmpty ? 0 : logs.reduce(0.0) { $0 + $1.basdaiScore } / Double(logs.count)
        notes += "Average BASDAI: \(String(format: "%.1f", avgBASDai)). "

        // Disease activity
        if avgBASDai < 2 {
            notes += "Patient shows remission/low disease activity. "
        } else if avgBASDai < 4 {
            notes += "Patient shows moderate disease activity. "
        } else {
            notes += "Patient shows high disease activity. Consider treatment escalation. "
        }

        // Flare frequency
        notes += "Flare events: \(flareEvents.count) in reporting period. "

        // Flare severity analysis
        if !flareEvents.isEmpty {
            let avgSeverity = flareEvents.reduce(0.0) { $0 + Double($1.severity) } / Double(flareEvents.count)
            notes += "Average flare severity: \(String(format: "%.1f", avgSeverity))/4. "
        }

        notes += "\n\nRecommendation: Review treatment plan and consider patient-reported flare patterns in management strategy."

        return notes
    }

    // MARK: - Helpers

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        return formatter.string(from: date)
    }

    // MARK: - Assessment Detail Pages (Page 8+)

    private func drawAssessmentPages(context: UIGraphicsPDFRendererContext, assessmentResponses: [QuestionnaireResponse], startPage: Int, totalPages: Int) {
        var currentPage = startPage
        let sortedResponses = assessmentResponses.sorted { ($0.createdAt ?? Date.distantPast) > ($1.createdAt ?? Date.distantPast) }

        for (index, response) in sortedResponses.enumerated() {
            context.beginPage()
            let ctx = context.cgContext

            drawHeader(ctx: ctx, title: "Assessment Details")

            var yPosition = headerHeight + 20

            // Assessment Information
            let assessmentTitle = response.questionnaireID?.uppercased() ?? "Assessment"
            yPosition = drawSection(ctx: ctx, title: assessmentTitle, yPosition: yPosition)

            let assessmentInfo = [
                ("Date", formatDate(response.createdAt ?? Date())),
                ("Score", String(format: "%.2f", response.score)),
                ("Duration", formatDuration(response.durationMs))
            ]

            for (label, value) in assessmentInfo {
                yPosition = drawKeyValue(ctx: ctx, key: label, value: value, yPosition: yPosition)
            }

            yPosition += 20

            // Individual Answers
            yPosition = drawSection(ctx: ctx, title: "Individual Responses", yPosition: yPosition)

            if let answersData = response.answersData,
               let answers = try? JSONDecoder().decode([String: Double].self, from: answersData) {

                for (questionID, answerValue) in answers.sorted(by: { $0.key < $1.key }) {
                    // Check if we need a new page
                    if yPosition > pageSize.height - margin - footerHeight - 60 {
                        drawFooter(ctx: ctx, pageNumber: currentPage, totalPages: totalPages)
                        currentPage += 1
                        context.beginPage()
                        drawHeader(ctx: ctx, title: "Assessment Details (continued)")
                        yPosition = headerHeight + 20
                    }

                    let questionText = "Q\(questionID)"
                    let answerText = String(format: "%.1f", answerValue)

                    yPosition = drawKeyValue(ctx: ctx, key: questionText, value: answerText, yPosition: yPosition)
                }
            }

            // Notes if available
            if let note = response.note, !note.isEmpty {
                yPosition += 20
                yPosition = drawSection(ctx: ctx, title: "Notes", yPosition: yPosition)
                yPosition = drawText(ctx: ctx, text: note, yPosition: yPosition, font: .systemFont(ofSize: 11), color: .darkGray, maxWidth: pageSize.width - 2 * margin)
            }

            drawFooter(ctx: ctx, pageNumber: currentPage, totalPages: totalPages)
            currentPage += 1
        }
    }

    private func formatDuration(_ durationMs: Double) -> String {
        let seconds = Int(durationMs / 1000)
        let minutes = seconds / 60
        let remainingSeconds = seconds % 60

        if minutes > 0 {
            return "\(minutes)m \(remainingSeconds)s"
        } else {
            return "\(remainingSeconds)s"
        }
    }
}
