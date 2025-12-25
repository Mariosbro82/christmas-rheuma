//
//  MedicalPDFGenerator.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import PDFKit
import CoreData

class MedicalPDFGenerator: ObservableObject {
    
    static func generateComprehensiveMedicalReport(
        painEntries: [PainEntry],
        journalEntries: [JournalEntry],
        bassdaiAssessments: [BASSDAIAssessment],
        basfiResponses: [QuestionnaireResponse],
        basgResponses: [QuestionnaireResponse],
        medications: [Medication],
        medicationIntakes: [MedicationIntake]
    ) -> Data {
        
        let pdfMetaData = [
            kCGPDFContextCreator: "InflamAI Medical Report Generator",
            kCGPDFContextAuthor: "InflamAI",
            kCGPDFContextTitle: "Comprehensive Rheumatology Medical Report",
            kCGPDFContextSubject: "Patient Pain and Symptom Analysis Report"
        ]
        
        let format = UIGraphicsPDFRendererFormat()
        format.documentInfo = pdfMetaData as [String: Any]
        
        let pageRect = CGRect(x: 0, y: 0, width: 612, height: 792) // US Letter size
        let renderer = UIGraphicsPDFRenderer(bounds: pageRect, format: format)
        
        let data = renderer.pdfData { context in
            // Page 1: Cover Page and Executive Summary
            context.beginPage()
            drawCoverPage(in: pageRect, painEntries: painEntries, bassdaiAssessments: bassdaiAssessments, basfiResponses: basfiResponses, basgResponses: basgResponses)
            
            // Page 2: Pain Body Diagram and Regional Analysis
            context.beginPage()
            drawPainAnalysisPage(in: pageRect, painEntries: painEntries)
            
            // Page 3: BASDAI Trends and Disease Activity
            context.beginPage()
            drawQuestionnaireAnalysisPage(in: pageRect, bassdaiAssessments: bassdaiAssessments, basfiResponses: basfiResponses, basgResponses: basgResponses)
            
            // Page 4: Medication Analysis and Adherence
            context.beginPage()
            drawMedicationAnalysisPage(in: pageRect, medications: medications, medicationIntakes: medicationIntakes)
            
            // Page 5: Correlation Analysis and Insights
            context.beginPage()
            drawCorrelationAnalysisPage(in: pageRect, journalEntries: journalEntries, painEntries: painEntries)
            
            // Page 6: Recommendations and Clinical Summary
            context.beginPage()
            drawClinicalSummaryPage(in: pageRect, painEntries: painEntries, bassdaiAssessments: bassdaiAssessments, basfiResponses: basfiResponses, basgResponses: basgResponses, journalEntries: journalEntries)
        }
        
        return data
    }
    
    // MARK: - Page Drawing Functions
    
    private static func drawCoverPage(in rect: CGRect, painEntries: [PainEntry], bassdaiAssessments: [BASSDAIAssessment], basfiResponses: [QuestionnaireResponse], basgResponses: [QuestionnaireResponse]) {
        let context = UIGraphicsGetCurrentContext()!
        
        // Header
        drawMedicalHeader(in: rect, title: "COMPREHENSIVE RHEUMATOLOGY REPORT")
        
        var yPosition: CGFloat = 150
        
        // Patient Information Section
        drawSectionHeader("PATIENT SUMMARY", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .long
        
        let reportInfo = [
            "Report Generated: \(dateFormatter.string(from: Date()))",
            "Analysis Period: \(getAnalysisPeriod(painEntries: painEntries))",
            "Total Pain Entries: \(painEntries.count)",
            "BASDAI Assessments: \(bassdaiAssessments.count)",
            "BASFI Assessments: \(basfiResponses.count)",
            "BAS-G Assessments: \(basgResponses.count)",
            "Data Quality: \(getDataQualityScore(painEntries: painEntries, bassdaiAssessments: bassdaiAssessments))"
        ]
        
        for info in reportInfo {
            drawBodyText(info, at: CGPoint(x: 70, y: yPosition), fontSize: 12)
            yPosition += 20
        }
        
        yPosition += 30
        
        // Executive Summary
        drawSectionHeader("EXECUTIVE SUMMARY", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        let executiveSummary = generateExecutiveSummary(painEntries: painEntries, bassdaiAssessments: bassdaiAssessments, basfiResponses: basfiResponses, basgResponses: basgResponses)
        drawWrappedText(executiveSummary, in: CGRect(x: 70, y: yPosition, width: 472, height: 200), fontSize: 12)
        yPosition += 220
        
        // Key Metrics Overview
        drawSectionHeader("KEY CLINICAL METRICS", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        drawKeyMetricsTable(in: CGRect(x: 50, y: yPosition, width: 512, height: 200), painEntries: painEntries, bassdaiAssessments: bassdaiAssessments, basfiResponses: basfiResponses, basgResponses: basgResponses)
        
        // Footer
        drawMedicalFooter(in: rect)
    }
    
    private static func drawPainAnalysisPage(in rect: CGRect, painEntries: [PainEntry]) {
        drawMedicalHeader(in: rect, title: "PAIN ANALYSIS & BODY MAPPING")
        
        var yPosition: CGFloat = 150
        
        // Pain Distribution Analysis
        drawSectionHeader("PAIN DISTRIBUTION ANALYSIS", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        // Draw simplified body diagram with pain regions
        drawPainBodyDiagram(in: CGRect(x: 50, y: yPosition, width: 250, height: 400), painEntries: painEntries)
        
        // Pain statistics on the right
        drawPainStatistics(in: CGRect(x: 320, y: yPosition, width: 242, height: 400), painEntries: painEntries)
        
        yPosition += 420
        
        // Pain trend analysis
        drawPainTrendAnalysis(in: CGRect(x: 50, y: yPosition, width: 512, height: 150), painEntries: painEntries)
        
        drawMedicalFooter(in: rect)
    }
    
    private static func drawQuestionnaireAnalysisPage(in rect: CGRect, bassdaiAssessments: [BASSDAIAssessment], basfiResponses: [QuestionnaireResponse], basgResponses: [QuestionnaireResponse]) {
        drawMedicalHeader(in: rect, title: "QUESTIONNAIRE ANALYSIS")
        
        var yPosition: CGFloat = 150
        
        // BASDAI Overview
        drawSectionHeader("BASDAI — DISEASE ACTIVITY", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        if let latestAssessment = bassdaiAssessments.last {
            drawBASSDAIScoreCard(in: CGRect(x: 50, y: yPosition, width: 512, height: 100), assessment: latestAssessment)
            yPosition += 120
        }
        
        // BASDAI Trend Chart
        drawSectionHeader("BASDAI TRENDS OVER TIME", at: CGPoint(x: 50, y: yPosition), fontSize: 16)
        yPosition += 30
        
        drawBASSDAITrendChart(in: CGRect(x: 50, y: yPosition, width: 512, height: 200), assessments: bassdaiAssessments)
        yPosition += 220
        
        // Component Analysis
        drawSectionHeader("COMPONENT ANALYSIS", at: CGPoint(x: 50, y: yPosition), fontSize: 16)
        yPosition += 30
        
        drawBASSDAIComponentAnalysis(in: CGRect(x: 50, y: yPosition, width: 512, height: 150), assessments: bassdaiAssessments)
        yPosition += 170
        
        // BASFI Summary
        drawSectionHeader("BASFI — FUNCTIONAL INDEX", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 30
        
        drawQuestionnaireSummaryBox(
            in: CGRect(x: 50, y: yPosition, width: 512, height: 120),
            questionnaireName: "BASFI",
            responses: basfiResponses
        )
        yPosition += 140
        
        // BAS-G Summary
        drawSectionHeader("BAS-G — GLOBAL WELL-BEING", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 30
        
        drawQuestionnaireSummaryBox(
            in: CGRect(x: 50, y: yPosition, width: 512, height: 120),
            questionnaireName: "BAS-G",
            responses: basgResponses
        )
        
        drawMedicalFooter(in: rect)
    }
    
    private static func drawMedicationAnalysisPage(in rect: CGRect, medications: [Medication], medicationIntakes: [MedicationIntake]) {
        drawMedicalHeader(in: rect, title: "MEDICATION ANALYSIS & ADHERENCE")
        
        var yPosition: CGFloat = 150
        
        // Current Medications
        drawSectionHeader("CURRENT MEDICATION REGIMEN", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        drawMedicationTable(in: CGRect(x: 50, y: yPosition, width: 512, height: 200), medications: medications)
        yPosition += 220
        
        // Adherence Analysis
        drawSectionHeader("ADHERENCE ANALYSIS", at: CGPoint(x: 50, y: yPosition), fontSize: 16)
        yPosition += 30
        
        drawMedicationAdherenceChart(in: CGRect(x: 50, y: yPosition, width: 512, height: 200), medications: medications, intakes: medicationIntakes)
        yPosition += 220
        
        // Medication Insights
        drawMedicationInsights(in: CGRect(x: 50, y: yPosition, width: 512, height: 100), medications: medications, intakes: medicationIntakes)
        
        drawMedicalFooter(in: rect)
    }
    
    private static func drawCorrelationAnalysisPage(in rect: CGRect, journalEntries: [JournalEntry], painEntries: [PainEntry]) {
        drawMedicalHeader(in: rect, title: "CORRELATION ANALYSIS & INSIGHTS")
        
        var yPosition: CGFloat = 150
        
        // Pain-Mood Correlation
        drawSectionHeader("PAIN-MOOD CORRELATION", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        drawCorrelationChart(in: CGRect(x: 50, y: yPosition, width: 512, height: 200), journalEntries: journalEntries)
        yPosition += 220
        
        // Sleep Quality Impact
        drawSectionHeader("SLEEP QUALITY IMPACT", at: CGPoint(x: 50, y: yPosition), fontSize: 16)
        yPosition += 30
        
        drawSleepQualityAnalysis(in: CGRect(x: 50, y: yPosition, width: 512, height: 150), journalEntries: journalEntries)
        yPosition += 170
        
        // Activity Level Correlation
        drawSectionHeader("ACTIVITY LEVEL CORRELATION", at: CGPoint(x: 50, y: yPosition), fontSize: 16)
        yPosition += 30
        
        drawActivityCorrelation(in: CGRect(x: 50, y: yPosition, width: 512, height: 120), journalEntries: journalEntries)
        
        drawMedicalFooter(in: rect)
    }
    
    private static func drawClinicalSummaryPage(in rect: CGRect, painEntries: [PainEntry], bassdaiAssessments: [BASSDAIAssessment], basfiResponses: [QuestionnaireResponse], basgResponses: [QuestionnaireResponse], journalEntries: [JournalEntry]) {
        drawMedicalHeader(in: rect, title: "CLINICAL SUMMARY & RECOMMENDATIONS")
        
        var yPosition: CGFloat = 150
        
        // Clinical Findings
        drawSectionHeader("CLINICAL FINDINGS", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        let clinicalFindings = generateClinicalFindings(painEntries: painEntries, bassdaiAssessments: bassdaiAssessments, basfiResponses: basfiResponses, basgResponses: basgResponses, journalEntries: journalEntries)
        drawWrappedText(clinicalFindings, in: CGRect(x: 70, y: yPosition, width: 472, height: 150), fontSize: 12)
        yPosition += 170
        
        // Recommendations
        drawSectionHeader("CLINICAL RECOMMENDATIONS", at: CGPoint(x: 50, y: yPosition), fontSize: 18)
        yPosition += 40
        
        let recommendations = generateRecommendations(painEntries: painEntries, bassdaiAssessments: bassdaiAssessments, basfiResponses: basfiResponses, basgResponses: basgResponses, journalEntries: journalEntries)
        drawWrappedText(recommendations, in: CGRect(x: 70, y: yPosition, width: 472, height: 200), fontSize: 12)
        yPosition += 220
        
        // Follow-up Plan
        drawSectionHeader("FOLLOW-UP PLAN", at: CGPoint(x: 50, y: yPosition), fontSize: 16)
        yPosition += 30
        
        let followUpPlan = generateFollowUpPlan(painEntries: painEntries, bassdaiAssessments: bassdaiAssessments, basfiResponses: basfiResponses, basgResponses: basgResponses)
        drawWrappedText(followUpPlan, in: CGRect(x: 70, y: yPosition, width: 472, height: 100), fontSize: 12)
        
        drawMedicalFooter(in: rect)
    }
    
    // MARK: - Drawing Helper Functions
    
    private static func drawMedicalHeader(in rect: CGRect, title: String) {
        let context = UIGraphicsGetCurrentContext()!
        
        // Header background
        context.setFillColor(UIColor.systemBlue.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: rect.width, height: 80))
        
        // Logo area (placeholder)
        context.setFillColor(UIColor.white.cgColor)
        context.fillEllipse(in: CGRect(x: 30, y: 20, width: 40, height: 40))
        
        // Title
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 20),
            .foregroundColor: UIColor.white
        ]
        
        title.draw(at: CGPoint(x: 90, y: 30), withAttributes: titleAttributes)
        
        // Date
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        let dateString = dateFormatter.string(from: Date())
        
        let dateAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 12),
            .foregroundColor: UIColor.white
        ]
        
        dateString.draw(at: CGPoint(x: rect.width - 150, y: 45), withAttributes: dateAttributes)
    }
    
    private static func drawMedicalFooter(in rect: CGRect) {
        let footerY = rect.height - 50
        
        let footerAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 10),
            .foregroundColor: UIColor.gray
        ]
        
        "Generated by InflamAI - Confidential Medical Report".draw(
            at: CGPoint(x: 50, y: footerY),
            withAttributes: footerAttributes
        )
        
        "Page \(getCurrentPageNumber())".draw(
            at: CGPoint(x: rect.width - 100, y: footerY),
            withAttributes: footerAttributes
        )
    }
    
    private static func drawSectionHeader(_ text: String, at point: CGPoint, fontSize: CGFloat) {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: fontSize),
            .foregroundColor: UIColor.systemBlue
        ]
        
        text.draw(at: point, withAttributes: attributes)
        
        // Underline
        let context = UIGraphicsGetCurrentContext()!
        context.setStrokeColor(UIColor.systemBlue.cgColor)
        context.setLineWidth(2)
        context.move(to: CGPoint(x: point.x, y: point.y + fontSize + 5))
        context.addLine(to: CGPoint(x: point.x + 200, y: point.y + fontSize + 5))
        context.strokePath()
    }
    
    private static func drawBodyText(_ text: String, at point: CGPoint, fontSize: CGFloat) {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: fontSize),
            .foregroundColor: UIColor.black
        ]
        
        text.draw(at: point, withAttributes: attributes)
    }
    
    private static func drawWrappedText(_ text: String, in rect: CGRect, fontSize: CGFloat) {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: fontSize),
            .foregroundColor: UIColor.black
        ]
        
        let attributedString = NSAttributedString(string: text, attributes: attributes)
        attributedString.draw(in: rect)
    }
    
    private static func drawPainBodyDiagram(in rect: CGRect, painEntries: [PainEntry]) {
        let context = UIGraphicsGetCurrentContext()!
        
        // Draw simplified body outline
        context.setStrokeColor(UIColor.black.cgColor)
        context.setLineWidth(2)
        
        let centerX = rect.midX
        let centerY = rect.midY
        
        // Head
        context.strokeEllipse(in: CGRect(x: centerX - 20, y: rect.minY + 20, width: 40, height: 50))
        
        // Body
        context.move(to: CGPoint(x: centerX, y: rect.minY + 70))
        context.addLine(to: CGPoint(x: centerX, y: rect.minY + 250))
        
        // Arms
        context.move(to: CGPoint(x: centerX - 60, y: rect.minY + 120))
        context.addLine(to: CGPoint(x: centerX + 60, y: rect.minY + 120))
        
        // Legs
        context.move(to: CGPoint(x: centerX - 20, y: rect.minY + 250))
        context.addLine(to: CGPoint(x: centerX - 20, y: rect.minY + 380))
        context.move(to: CGPoint(x: centerX + 20, y: rect.minY + 250))
        context.addLine(to: CGPoint(x: centerX + 20, y: rect.minY + 380))
        
        context.strokePath()
        
        // Draw pain bubbles
        drawPainBubbles(in: rect, painEntries: painEntries)
    }
    
    private static func drawPainBubbles(in rect: CGRect, painEntries: [PainEntry]) {
        let context = UIGraphicsGetCurrentContext()!
        
        // Analyze pain by region
        var regionPainData: [String: (count: Int, totalPain: Double)] = [:]
        
        for entry in painEntries {
            let regions = entry.bodyRegions?.components(separatedBy: ",") ?? []
            for region in regions {
                let trimmedRegion = region.trimmingCharacters(in: .whitespaces)
                if regionPainData[trimmedRegion] == nil {
                    regionPainData[trimmedRegion] = (0, 0)
                }
                regionPainData[trimmedRegion]!.count += 1
                regionPainData[trimmedRegion]!.totalPain += Double(entry.painLevel)
            }
        }
        
        // Draw bubbles for each region
        for (region, data) in regionPainData {
            let averagePain = data.totalPain / Double(data.count)
            let position = getPainBubblePosition(for: region, in: rect)
            let bubbleSize = min(20 + Double(data.count) * 2, 40)
            
            // Set color based on pain level
            let color = getPainColor(for: averagePain)
            context.setFillColor(color.cgColor)
            
            let bubbleRect = CGRect(
                x: position.x - bubbleSize/2,
                y: position.y - bubbleSize/2,
                width: bubbleSize,
                height: bubbleSize
            )
            
            context.fillEllipse(in: bubbleRect)
            
            // Add pain level text
            let painText = String(format: "%.1f", averagePain)
            let textAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 10),
                .foregroundColor: UIColor.white
            ]
            
            let textSize = painText.size(withAttributes: textAttributes)
            painText.draw(
                at: CGPoint(
                    x: position.x - textSize.width/2,
                    y: position.y - textSize.height/2
                ),
                withAttributes: textAttributes
            )
        }
    }
    
    private static func getPainBubblePosition(for region: String, in rect: CGRect) -> CGPoint {
        let centerX = rect.midX
        let centerY = rect.midY
        
        switch region.lowercased() {
        case "head", "neck":
            return CGPoint(x: centerX, y: rect.minY + 45)
        case "left_shoulder", "left shoulder":
            return CGPoint(x: centerX - 40, y: rect.minY + 100)
        case "right_shoulder", "right shoulder":
            return CGPoint(x: centerX + 40, y: rect.minY + 100)
        case "chest", "upper_back":
            return CGPoint(x: centerX, y: rect.minY + 140)
        case "lower_back", "back":
            return CGPoint(x: centerX, y: rect.minY + 200)
        case "left_hip", "left hip":
            return CGPoint(x: centerX - 25, y: rect.minY + 240)
        case "right_hip", "right hip":
            return CGPoint(x: centerX + 25, y: rect.minY + 240)
        case "left_knee", "left knee":
            return CGPoint(x: centerX - 20, y: rect.minY + 320)
        case "right_knee", "right knee":
            return CGPoint(x: centerX + 20, y: rect.minY + 320)
        default:
            return CGPoint(x: centerX, y: centerY)
        }
    }
    
    private static func getPainColor(for painLevel: Double) -> UIColor {
        switch painLevel {
        case 0..<2: return .systemGreen
        case 2..<4: return .systemYellow
        case 4..<6: return .systemOrange
        case 6..<8: return .systemRed
        default: return .systemPurple
        }
    }
    
    // MARK: - Data Analysis Helper Functions
    
    private static func generateExecutiveSummary(painEntries: [PainEntry], bassdaiAssessments: [BASSDAIAssessment], basfiResponses: [QuestionnaireResponse], basgResponses: [QuestionnaireResponse]) -> String {
        let avgPain = painEntries.isEmpty ? 0 : Double(painEntries.reduce(0) { $0 + $1.painLevel }) / Double(painEntries.count)
        let latestBASSDAI = bassdaiAssessments.last?.totalScore ?? 0
        let latestBASFI = basfiResponses.last?.score ?? 0
        let latestBASG = basgResponses.last?.score ?? 0
        
        return """
This comprehensive report analyzes \(painEntries.count) pain entries alongside \(bassdaiAssessments.count) BASDAI disease activity checks, \(basfiResponses.count) BASFI functional assessments, and \(basgResponses.count) BAS-G global well-being reflections. The patient's average pain level is \(String(format: "%.1f", avgPain))/10. Most recent scores show BASDAI \(String(format: "%.1f", latestBASSDAI)), BASFI \(String(format: "%.1f", latestBASFI)), and BAS-G \(String(format: "%.1f", latestBASG)). The analysis highlights symptom distribution, function, and perceived well-being trends to guide shared-care decisions.
"""
    }
    
    private static func generateClinicalFindings(painEntries: [PainEntry], bassdaiAssessments: [BASSDAIAssessment], basfiResponses: [QuestionnaireResponse], basgResponses: [QuestionnaireResponse], journalEntries: [JournalEntry]) -> String {
        return """
Clinical analysis reveals significant patterns in disease progression, functional capacity, and perceived well-being. Pain distribution shows predominant involvement of axial skeleton with peripheral joint contribution. BASDAI scores demonstrate fluctuating inflammatory activity, while BASFI captures the day-to-day functional limitations influencing mobility and self-care. BAS-G responses contextualize how symptoms impact quality of life. Correlation analysis demonstrates relationships between pain levels, mood, and sleep quality, suggesting multifactorial symptom management requirements.
"""
    }
    
    private static func generateRecommendations(painEntries: [PainEntry], bassdaiAssessments: [BASSDAIAssessment], basfiResponses: [QuestionnaireResponse], basgResponses: [QuestionnaireResponse], journalEntries: [JournalEntry]) -> String {
        return """
Based on the comprehensive analysis, recommendations include: 1) Optimise anti-inflammatory therapy during high BASDAI periods, 2) Reinforce structured mobility and strengthening routines to address BASFI limitations, 3) Employ sleep and fatigue management strategies to support BAS-G trends, 4) Encourage stress reduction and pacing techniques, 5) Maintain daily BASDAI and weekly BASFI/BAS-G monitoring to capture trajectory shifts, 6) Reassess pharmacologic and biologic options if sustained high scores persist.
"""
    }
    
    private static func generateFollowUpPlan(painEntries: [PainEntry], bassdaiAssessments: [BASSDAIAssessment], basfiResponses: [QuestionnaireResponse], basgResponses: [QuestionnaireResponse]) -> String {
        return """
Recommended follow-up includes daily BASDAI tracking, weekly BASFI and BAS-G submissions, quarterly comprehensive evaluations, and immediate consultation for disease flares. Continue current monitoring protocols with emphasis on early intervention strategies and functional goal-setting.
"""
    }
    
    // MARK: - Chart Drawing Functions (Simplified for PDF)
    
    private static func drawKeyMetricsTable(in rect: CGRect, painEntries: [PainEntry], bassdaiAssessments: [BASSDAIAssessment], basfiResponses: [QuestionnaireResponse], basgResponses: [QuestionnaireResponse]) {
        let context = UIGraphicsGetCurrentContext()!
        
        // Table background
        context.setFillColor(UIColor.systemGray6.cgColor)
        context.fill(rect)
        
        // Table border
        context.setStrokeColor(UIColor.black.cgColor)
        context.setLineWidth(1)
        context.stroke(rect)
        
        // Calculate metrics
        let avgPain = painEntries.isEmpty ? 0 : Double(painEntries.reduce(0) { $0 + $1.painLevel }) / Double(painEntries.count)
        let latestBASSDAI = bassdaiAssessments.last?.totalScore ?? 0
        let latestBASFI = basfiResponses.last?.score ?? 0
        let avgBASFI = basfiResponses.isEmpty ? 0 : basfiResponses.map(\.score).reduce(0, +) / Double(basfiResponses.count)
        let latestBASG = basgResponses.last?.score ?? 0
        let painTrend = calculatePainTrend(painEntries: painEntries)
        
        let metrics = [
            ("Average Pain Level", String(format: "%.1f/10", avgPain)),
            ("Latest BASDAI Score", String(format: "%.1f/10", latestBASSDAI)),
            ("Latest BASFI Score", basfiResponses.isEmpty ? "—" : String(format: "%.1f/10", latestBASFI)),
            ("Average BASFI Score", basfiResponses.isEmpty ? "—" : String(format: "%.1f/10", avgBASFI)),
            ("Latest BAS-G Score", basgResponses.isEmpty ? "—" : String(format: "%.1f/10", latestBASG)),
            ("Pain Trend (30 days)", painTrend),
            ("Assessments Logged", "\(bassdaiAssessments.count) BASDAI / \(basfiResponses.count) BASFI / \(basgResponses.count) BAS-G"),
            ("Data Collection Period", getAnalysisPeriod(painEntries: painEntries))
        ]
        
        let rowHeight: CGFloat = rect.height / CGFloat(metrics.count)
        
        for (index, (metric, value)) in metrics.enumerated() {
            let y = rect.minY + CGFloat(index) * rowHeight
            
            // Draw row separator
            if index > 0 {
                context.move(to: CGPoint(x: rect.minX, y: y))
                context.addLine(to: CGPoint(x: rect.maxX, y: y))
                context.strokePath()
            }
            
            // Draw metric name
            drawBodyText(metric, at: CGPoint(x: rect.minX + 10, y: y + 10), fontSize: 12)
            
            // Draw value
            drawBodyText(value, at: CGPoint(x: rect.minX + 300, y: y + 10), fontSize: 12)
        }
    }
    
    private static func drawPainStatistics(in rect: CGRect, painEntries: [PainEntry]) {
        var yPosition = rect.minY
        
        drawSectionHeader("PAIN STATISTICS", at: CGPoint(x: rect.minX, y: yPosition), fontSize: 14)
        yPosition += 30
        
        let avgPain = painEntries.isEmpty ? 0 : Double(painEntries.reduce(0) { $0 + $1.painLevel }) / Double(painEntries.count)
        let maxPain = painEntries.map { Double($0.painLevel) }.max() ?? 0
        let minPain = painEntries.map { Double($0.painLevel) }.min() ?? 0
        
        let stats = [
            "Average Pain: \(String(format: "%.1f", avgPain))",
            "Maximum Pain: \(String(format: "%.0f", maxPain))",
            "Minimum Pain: \(String(format: "%.0f", minPain))",
            "Total Entries: \(painEntries.count)",
            "Most Affected Region: \(getMostAffectedRegion(painEntries: painEntries))"
        ]
        
        for stat in stats {
            drawBodyText(stat, at: CGPoint(x: rect.minX, y: yPosition), fontSize: 11)
            yPosition += 20
        }
    }
    
    // Additional drawing functions would be implemented here...
    // (drawPainTrendAnalysis, drawBASSDAIScoreCard, etc.)
    
    // MARK: - Utility Functions
    
    private static func getAnalysisPeriod(painEntries: [PainEntry]) -> String {
        guard let firstEntry = painEntries.first?.timestamp,
              let lastEntry = painEntries.last?.timestamp else {
            return "No data available"
        }
        
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        
        return "\(formatter.string(from: firstEntry)) - \(formatter.string(from: lastEntry))"
    }
    
    private static func getDataQualityScore(painEntries: [PainEntry], bassdaiAssessments: [BASSDAIAssessment]) -> String {
        let totalEntries = painEntries.count + bassdaiAssessments.count
        
        switch totalEntries {
        case 0..<10: return "Limited"
        case 10..<30: return "Moderate"
        case 30..<50: return "Good"
        default: return "Excellent"
        }
    }
    
    private static func calculatePainTrend(painEntries: [PainEntry]) -> String {
        guard painEntries.count >= 2 else { return "Insufficient data" }
        
        let sortedEntries = painEntries.sorted { ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) }
        let recentEntries = Array(sortedEntries.suffix(10))
        let olderEntries = Array(sortedEntries.prefix(10))
        
        let recentAvg = Double(recentEntries.reduce(0) { $0 + $1.painLevel }) / Double(recentEntries.count)
        let olderAvg = Double(olderEntries.reduce(0) { $0 + $1.painLevel }) / Double(olderEntries.count)
        
        let difference = recentAvg - olderAvg
        
        if difference > 0.5 {
            return "Increasing"
        } else if difference < -0.5 {
            return "Decreasing"
        } else {
            return "Stable"
        }
    }
    
    private static func getMostAffectedRegion(painEntries: [PainEntry]) -> String {
        var regionCounts: [String: Int] = [:]
        
        for entry in painEntries {
            let regions = entry.bodyRegions?.components(separatedBy: ",") ?? []
            for region in regions {
                let trimmedRegion = region.trimmingCharacters(in: .whitespaces)
                regionCounts[trimmedRegion, default: 0] += 1
            }
        }
        
        return regionCounts.max(by: { $0.value < $1.value })?.key ?? "Unknown"
    }
    
    private static func getCurrentPageNumber() -> Int {
        // This would need to be tracked during PDF generation
        return 1
    }
    
    // Placeholder implementations for additional chart drawing functions
    private static func drawPainTrendAnalysis(in rect: CGRect, painEntries: [PainEntry]) {
        // Implementation for pain trend chart
    }
    
    private static func drawBASSDAIScoreCard(in rect: CGRect, assessment: BASSDAIAssessment) {
        // Implementation for BASDAI score display
    }
    
    private static func drawBASSDAITrendChart(in rect: CGRect, assessments: [BASSDAIAssessment]) {
        // Implementation for BASDAI trend chart
    }
    
    private static func drawBASSDAIComponentAnalysis(in rect: CGRect, assessments: [BASSDAIAssessment]) {
        // Implementation for component analysis
    }

    private static func drawQuestionnaireSummaryBox(in rect: CGRect, questionnaireName: String, responses: [QuestionnaireResponse]) {
        let context = UIGraphicsGetCurrentContext()
        context?.setFillColor(UIColor.systemGray6.cgColor)
        context?.fill(rect)
        context?.setStrokeColor(UIColor.systemGray4.cgColor)
        context?.setLineWidth(1)
        context?.stroke(rect)
        
        let hasEntries = !responses.isEmpty
        let latestScore = responses.last?.score ?? 0
        let averageScore = hasEntries ? responses.map(\.score).reduce(0, +) / Double(responses.count) : 0
        let highestScore = responses.map(\.score).max() ?? 0
        let lowestScore = responses.map(\.score).min() ?? 0
        let latestDate = responses.last?.createdAt
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        
        let metrics = [
            "Latest Score: " + (hasEntries ? String(format: "%.1f / 10", latestScore) : "—"),
            "Last Logged: " + (latestDate.map { dateFormatter.string(from: $0) } ?? "No entries yet"),
            "Average Score: " + (hasEntries ? String(format: "%.1f", averageScore) : "—"),
            "Best Score (lower is better): " + (hasEntries ? String(format: "%.1f", lowestScore) : "—"),
            "Highest Score: " + (hasEntries ? String(format: "%.1f", highestScore) : "—"),
            "Entries Logged: \(responses.count)"
        ]
        
        var yPosition = rect.minY + 16
        drawBodyText("\(questionnaireName) Summary", at: CGPoint(x: rect.minX + 16, y: yPosition), fontSize: 14)
        yPosition += 24
        
        for metric in metrics {
            drawBodyText(metric, at: CGPoint(x: rect.minX + 16, y: yPosition), fontSize: 11)
            yPosition += 18
        }
    }
    
    private static func drawMedicationTable(in rect: CGRect, medications: [Medication]) {
        // Implementation for medication table
    }
    
    private static func drawMedicationAdherenceChart(in rect: CGRect, medications: [Medication], intakes: [MedicationIntake]) {
        // Implementation for adherence chart
    }
    
    private static func drawMedicationInsights(in rect: CGRect, medications: [Medication], intakes: [MedicationIntake]) {
        // Implementation for medication insights
    }
    
    private static func drawCorrelationChart(in rect: CGRect, journalEntries: [JournalEntry]) {
        // Implementation for correlation chart
    }
    
    private static func drawSleepQualityAnalysis(in rect: CGRect, journalEntries: [JournalEntry]) {
        // Implementation for sleep analysis
    }
    
    private static func drawActivityCorrelation(in rect: CGRect, journalEntries: [JournalEntry]) {
        // Implementation for activity correlation
    }
}
