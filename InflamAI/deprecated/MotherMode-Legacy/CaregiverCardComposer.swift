//
//  CaregiverCardComposer.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import PDFKit
import UIKit

struct CaregiverCardData {
    var whatHelps: String
    var avoid: String
    var medsHandled: String
    var emergencyInfo: String
    var contacts: String
}

final class CaregiverCardComposer {
    func makePDF(data: CaregiverCardData) -> Data {
        let format = UIGraphicsPDFRendererFormat()
        let rect = CGRect(x: 0, y: 0, width: 595, height: 842) // A4
        let renderer = UIGraphicsPDFRenderer(bounds: rect, format: format)
        
        return renderer.pdfData { context in
            context.beginPage()
            drawHeader(in: context.cgContext)
            draw(sectionTitle: NSLocalizedString("mother.caregiver.what_helps", comment: ""), body: data.whatHelps, y: 80, context: context.cgContext, pageRect: rect)
            draw(sectionTitle: NSLocalizedString("mother.caregiver.avoid", comment: ""), body: data.avoid, y: 190, context: context.cgContext, pageRect: rect)
            draw(sectionTitle: NSLocalizedString("mother.caregiver.medications", comment: ""), body: data.medsHandled, y: 300, context: context.cgContext, pageRect: rect)
            draw(sectionTitle: NSLocalizedString("mother.caregiver.emergency", comment: ""), body: data.emergencyInfo, y: 410, context: context.cgContext, pageRect: rect)
            draw(sectionTitle: NSLocalizedString("mother.caregiver.contacts", comment: ""), body: data.contacts, y: 520, context: context.cgContext, pageRect: rect)
            drawFooter(in: context.cgContext, pageRect: rect)
        }
    }
    
    private func drawHeader(in context: CGContext) {
        let title = NSLocalizedString("mother.caregiver.title", comment: "")
        title.draw(at: CGPoint(x: 32, y: 32), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 22)])
        
        let disclaimer = NSLocalizedString("disclaimer.general_info", comment: "")
        disclaimer.draw(in: CGRect(x: 32, y: 52, width: 531, height: 40), withAttributes: [.font: UIFont.systemFont(ofSize: 11)])
    }
    
    private func draw(sectionTitle: String, body: String, y: CGFloat, context: CGContext, pageRect: CGRect) {
        sectionTitle.draw(at: CGPoint(x: 32, y: y), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 14)])
        body.draw(in: CGRect(x: 32, y: y + 18, width: pageRect.width - 64, height: 100), withAttributes: [.font: UIFont.systemFont(ofSize: 12)])
    }
    
    private func drawFooter(in context: CGContext, pageRect: CGRect) {
        let caution = NSLocalizedString("mother.pregnancy.caution", comment: "")
        caution.draw(in: CGRect(x: 32, y: pageRect.height - 80, width: pageRect.width - 64, height: 60), withAttributes: [
            .font: UIFont.italicSystemFont(ofSize: 11),
            .foregroundColor: UIColor.systemRed
        ])
    }
}
