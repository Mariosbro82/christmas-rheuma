//
//  ClinicianReportComposer.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import PDFKit
import UIKit

typealias MedicationSummary = String

final class ClinicianReportComposer {
    func makePDF(entries: [SymptomEntry], mobilityCompletionRate: Int, medicationSummaries: [MedicationSummary]) -> Data {
        let format = UIGraphicsPDFRendererFormat()
        let rect = CGRect(x: 0, y: 0, width: 595, height: 842)
        let renderer = UIGraphicsPDFRenderer(bounds: rect, format: format)
        return renderer.pdfData { context in
            context.beginPage()
            drawHeader(in: context.cgContext)
            drawSummary(entries: entries, in: context.cgContext, pageRect: rect)
            drawMobility(rate: mobilityCompletionRate, in: context.cgContext, pageRect: rect)
            drawMedication(medicationSummaries, in: context.cgContext, pageRect: rect)
            drawFooter(in: context.cgContext, pageRect: rect)
        }
    }
    
    private func drawHeader(in context: CGContext) {
        let title = NSLocalizedString("clinician.header.title", comment: "")
        title.draw(at: CGPoint(x: 32, y: 28), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 22)])
        let disclaimer = NSLocalizedString("disclaimer.general_info", comment: "")
        disclaimer.draw(in: CGRect(x: 32, y: 50, width: 531, height: 40), withAttributes: [.font: UIFont.systemFont(ofSize: 11)])
    }
    
    private func drawSummary(entries: [SymptomEntry], in context: CGContext, pageRect: CGRect) {
        let sectionTitle = NSLocalizedString("clinician.section.symptom", comment: "")
        sectionTitle.draw(at: CGPoint(x: 32, y: 90), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 16)])
        guard !entries.isEmpty else {
            NSLocalizedString("clinician.no_entries", comment: "").draw(at: CGPoint(x: 32, y: 110), withAttributes: [.font: UIFont.systemFont(ofSize: 12)])
            return
        }
        let pain = entries.map(\.pain).averageFormatted()
        let stiffness = entries.map(\.stiffnessMinutes).averageIntFormatted()
        let fatigue = entries.map(\.fatigue).averageFormatted()
        let body = String(format: NSLocalizedString("clinician.summary.body", comment: ""), entries.count, pain, stiffness, fatigue)
        body.draw(in: CGRect(x: 32, y: 110, width: pageRect.width - 64, height: 80), withAttributes: [.font: UIFont.systemFont(ofSize: 12)])
    }
    
    private func drawMobility(rate: Int, in context: CGContext, pageRect: CGRect) {
        let title = NSLocalizedString("clinician.section.mobility", comment: "")
        title.draw(at: CGPoint(x: 32, y: 200), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 16)])
        let detail = String(format: NSLocalizedString("clinician.mobility.rate", comment: ""), rate)
        detail.draw(at: CGPoint(x: 32, y: 220), withAttributes: [.font: UIFont.systemFont(ofSize: 12)])
    }
    
    private func drawMedication(_ medications: [MedicationSummary], in context: CGContext, pageRect: CGRect) {
        let sectionTitle = NSLocalizedString("clinician.section.medication", comment: "")
        sectionTitle.draw(at: CGPoint(x: 32, y: 260), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 16)])
        if medications.isEmpty {
            NSLocalizedString("clinician.medication.none", comment: "").draw(at: CGPoint(x: 32, y: 280), withAttributes: [.font: UIFont.systemFont(ofSize: 12)])
        } else {
            var y = 280.0
            medications.forEach { summary in
                summary.draw(at: CGPoint(x: 32, y: y), withAttributes: [.font: UIFont.systemFont(ofSize: 12)])
                y += 18
            }
        }
    }
    
    private func drawFooter(in context: CGContext, pageRect: CGRect) {
        let caution = NSLocalizedString("mother.pregnancy.caution", comment: "")
        caution.draw(in: CGRect(x: 32, y: pageRect.height - 80, width: pageRect.width - 64, height: 60), withAttributes: [.font: UIFont.italicSystemFont(ofSize: 11), .foregroundColor: UIColor.systemRed])
    }
}

private extension Array where Element == Double {
    func averageFormatted() -> String {
        guard !isEmpty else { return "--" }
        let value = reduce(0, +) / Double(count)
        return String(format: "%.1f", value)
    }
}

private extension Array where Element == Int {
    func averageIntFormatted() -> Int {
        guard !isEmpty else { return 0 }
        let value = Double(reduce(0, +)) / Double(count)
        return Int(value.rounded())
    }
}
