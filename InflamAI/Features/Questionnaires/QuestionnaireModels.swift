//
//  QuestionnaireModels.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-06-09.
//

import Foundation

// MARK: - Disease Categories

enum DiseaseCategory: String, CaseIterable, Identifiable, Codable {
    case axialSpa = "Axial Spondyloarthritis / AS"
    case rheumatoidArthritis = "Rheumatoid Arthritis"
    case psoriaticArthritis = "Psoriatic Arthritis"
    case lupus = "Systemic Lupus Erythematosus"
    case sjogrens = "Sjögren's Syndrome"
    case scleroderma = "Systemic Sclerosis"
    case vasculitis = "Vasculitis"
    case gout = "Gout"
    case osteoarthritis = "Osteoarthritis"
    case fibromyalgia = "Fibromyalgia"
    case myopathies = "Inflammatory Myopathies"
    case pediatric = "Pediatric / Juvenile"
    case generic = "Cross-Disease / Generic"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .axialSpa: return "figure.walk"
        case .rheumatoidArthritis: return "hand.raised"
        case .psoriaticArthritis: return "person.crop.circle.badge.exclamationmark"
        case .lupus: return "heart.circle"
        case .sjogrens: return "eye"
        case .scleroderma: return "bandage"
        case .vasculitis: return "waveform.path.ecg"
        case .gout: return "foot.circle"
        case .osteoarthritis: return "figure.flexibility"
        case .fibromyalgia: return "brain.head.profile"
        case .myopathies: return "figure.arms.open"
        case .pediatric: return "figure.and.child.holdinghands"
        case .generic: return "stethoscope"
        }
    }
}

// MARK: - Questionnaire ID

enum QuestionnaireID: String, CaseIterable, Identifiable, Codable {
    // Axial Spondyloarthritis / AS
    case basdai
    case basfi
    case basg
    case asqol
    case asdas
    case asasHI = "asas_hi"

    // Rheumatoid Arthritis
    case rapid3
    case haqDI = "haq_di"
    case raid

    // Psoriatic Arthritis
    case psaid12 = "psaid_12"
    case psaid9 = "psaid_9"
    case psaqol

    // Systemic Lupus Erythematosus
    case slaq
    case qSlaq = "q_slaq"
    case lupusPRO = "lupus_pro"
    case lupusQoL = "lupus_qol"

    // Sjögren's Syndrome
    case esspri
    case profadSSI = "profad_ssi"

    // Systemic Sclerosis / Scleroderma
    case shaq
    case uclaGIT = "ucla_git"
    case raynaudsCS = "raynauds_cs"

    // Vasculitis
    case aavPRO = "aav_pro"
    case bdcaf
    case bdQoL = "bd_qol"

    // Gout
    case gaq2 = "gaq_2"
    case gis

    // Osteoarthritis
    case womac
    case koos
    case hoos

    // Fibromyalgia
    case fiqr
    case acr2010 = "acr_2010"

    // Inflammatory Myopathies
    case map
    case ibmHI = "ibm_hi"

    // Pediatric / Juvenile
    case chaq
    case jamar
    case pedsQL = "peds_ql"
    case jadas
    case cJadas = "c_jadas"

    // Generic / Cross-Disease
    case mdhaq
    case promis
    case sf36 = "sf_36"
    case eq5d = "eq_5d"
    case facitFatigue = "facit_fatigue"

    var id: String { rawValue }

    var category: DiseaseCategory {
        switch self {
        // Axial SpA/AS
        case .basdai, .basfi, .basg, .asqol, .asdas, .asasHI:
            return .axialSpa
        // RA
        case .rapid3, .haqDI, .raid:
            return .rheumatoidArthritis
        // PsA
        case .psaid12, .psaid9, .psaqol:
            return .psoriaticArthritis
        // SLE
        case .slaq, .qSlaq, .lupusPRO, .lupusQoL:
            return .lupus
        // Sjögren's
        case .esspri, .profadSSI:
            return .sjogrens
        // Scleroderma
        case .shaq, .uclaGIT, .raynaudsCS:
            return .scleroderma
        // Vasculitis
        case .aavPRO, .bdcaf, .bdQoL:
            return .vasculitis
        // Gout
        case .gaq2, .gis:
            return .gout
        // OA
        case .womac, .koos, .hoos:
            return .osteoarthritis
        // Fibromyalgia
        case .fiqr, .acr2010:
            return .fibromyalgia
        // Myopathies
        case .map, .ibmHI:
            return .myopathies
        // Pediatric
        case .chaq, .jamar, .pedsQL, .jadas, .cJadas:
            return .pediatric
        // Generic
        case .mdhaq, .promis, .sf36, .eq5d, .facitFatigue:
            return .generic
        }
    }

    var titleKey: String {
        return "questionnaire.\(rawValue).title"
    }

    var descriptionKey: String {
        return "questionnaire.\(rawValue).description"
    }

    var isDefault: Bool {
        return self == .basdai
    }
    
    var defaultSchedule: QuestionnaireSchedule {
        switch self {
        // Daily questionnaires - disease activity monitoring
        case .basdai, .rapid3, .esspri:
            return QuestionnaireSchedule(
                frequency: .daily(time: DateComponents(hour: 20, minute: 0)),
                windowHours: 4,
                timezoneIdentifier: "Europe/Berlin"
            )

        // Weekly questionnaires - function and impact assessments
        case .basfi, .haqDI, .raid, .psaid12, .psaid9, .fiqr, .map, .chaq, .mdhaq:
            return QuestionnaireSchedule(
                frequency: .weekly(weekday: 1, time: DateComponents(hour: 18, minute: 0)),
                windowHours: 72,
                timezoneIdentifier: "Europe/Berlin"
            )

        // Weekly questionnaires - with prerequisites
        case .basg:
            return QuestionnaireSchedule(
                frequency: .weekly(weekday: 1, time: DateComponents(hour: 18, minute: 5)),
                windowHours: 72,
                timezoneIdentifier: "Europe/Berlin",
                prerequisites: [.basfi]
            )

        // Monthly questionnaires - quality of life and comprehensive assessments
        case .asqol, .psaqol, .lupusQoL, .bdQoL, .profadSSI, .lupusPRO,
             .sf36, .eq5d:
            return QuestionnaireSchedule(
                frequency: .monthly(day: 1, time: DateComponents(hour: 18, minute: 0)),
                windowHours: 168, // 1 week window
                timezoneIdentifier: "Europe/Berlin"
            )

        // On-demand questionnaires - clinical/research use
        case .asdas, .asasHI, .slaq, .qSlaq, .shaq, .uclaGIT, .raynaudsCS,
             .aavPRO, .bdcaf, .gaq2, .gis, .womac, .koos, .hoos, .acr2010,
             .ibmHI, .jamar, .pedsQL, .jadas, .cJadas, .promis, .facitFatigue:
            return QuestionnaireSchedule(
                frequency: .onDemand,
                windowHours: 0,
                timezoneIdentifier: "Europe/Berlin"
            )
        }
    }
}

struct QuestionnaireSchedule: Hashable, Codable {
    enum Frequency: Hashable, Codable {
        case daily(time: DateComponents)
        case weekly(weekday: Int, time: DateComponents)
        case monthly(day: Int, time: DateComponents)
        case onDemand

        private enum CodingKeys: String, CodingKey {
            case type, hour, minute, weekday, day
        }

        enum FrequencyType: String, Codable {
            case daily
            case weekly
            case monthly
            case onDemand
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(FrequencyType.self, forKey: .type)
            switch type {
            case .daily:
                let hour = try container.decode(Int.self, forKey: .hour)
                let minute = try container.decode(Int.self, forKey: .minute)
                let components = DateComponents(hour: hour, minute: minute)
                self = .daily(time: components)
            case .weekly:
                let hour = try container.decode(Int.self, forKey: .hour)
                let minute = try container.decode(Int.self, forKey: .minute)
                let weekday = try container.decode(Int.self, forKey: .weekday)
                let components = DateComponents(hour: hour, minute: minute)
                self = .weekly(weekday: weekday, time: components)
            case .monthly:
                let hour = try container.decode(Int.self, forKey: .hour)
                let minute = try container.decode(Int.self, forKey: .minute)
                let day = try container.decode(Int.self, forKey: .day)
                let components = DateComponents(hour: hour, minute: minute)
                self = .monthly(day: day, time: components)
            case .onDemand:
                self = .onDemand
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .daily(let time):
                try container.encode(FrequencyType.daily, forKey: .type)
                try container.encode(time.hour ?? 0, forKey: .hour)
                try container.encode(time.minute ?? 0, forKey: .minute)
            case .weekly(let weekday, let time):
                try container.encode(FrequencyType.weekly, forKey: .type)
                try container.encode(weekday, forKey: .weekday)
                try container.encode(time.hour ?? 0, forKey: .hour)
                try container.encode(time.minute ?? 0, forKey: .minute)
            case .monthly(let day, let time):
                try container.encode(FrequencyType.monthly, forKey: .type)
                try container.encode(day, forKey: .day)
                try container.encode(time.hour ?? 0, forKey: .hour)
                try container.encode(time.minute ?? 0, forKey: .minute)
            case .onDemand:
                try container.encode(FrequencyType.onDemand, forKey: .type)
            }
        }
    }
    
    var frequency: Frequency
    var windowHours: Int
    var timezoneIdentifier: String
    var prerequisites: [QuestionnaireID]
    
    init(
        frequency: Frequency,
        windowHours: Int,
        timezoneIdentifier: String,
        prerequisites: [QuestionnaireID] = []
    ) {
        self.frequency = frequency
        self.windowHours = windowHours
        self.timezoneIdentifier = timezoneIdentifier
        self.prerequisites = prerequisites
    }
    
    var timezone: TimeZone {
        TimeZone(identifier: timezoneIdentifier) ?? TimeZone(identifier: "Europe/Berlin") ?? .current
    }
}

struct QuestionnaireItem: Identifiable, Hashable, Codable {
    let id: String
    let promptKey: String
    let minimum: Int
    let maximum: Int
    let anchors: [Int: String]
}

struct QuestionnaireDefinition: Hashable {
    let id: QuestionnaireID
    let version: String
    let items: [QuestionnaireItem]
    let periodDescriptionKey: String
    let notesAllowed: Bool
    
    static let basdai = QuestionnaireDefinition(
        id: .basdai,
        version: "1.0.0",
        items: [
            QuestionnaireItem(
                id: "Q1",
                promptKey: "basdai.q1.prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            ),
            QuestionnaireItem(
                id: "Q2",
                promptKey: "basdai.q2.prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            ),
            QuestionnaireItem(
                id: "Q3",
                promptKey: "basdai.q3.prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            ),
            QuestionnaireItem(
                id: "Q4",
                promptKey: "basdai.q4.prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            ),
            QuestionnaireItem(
                id: "Q5",
                promptKey: "basdai.q5.prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "basdai.anchor.duration.none", 5: "basdai.anchor.duration.oneHour", 10: "basdai.anchor.duration.twoHours"]
            ),
            QuestionnaireItem(
                id: "Q6",
                promptKey: "basdai.q6.prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            )
        ],
        periodDescriptionKey: "basdai.period",
        notesAllowed: true
    )
    
    static let basfi = QuestionnaireDefinition(
        id: .basfi,
        version: "1.0.0",
        items: (1...10).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "basfi.q\(index).prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.easy", 5: "scale.anchor.difficult", 10: "scale.anchor.impossible"]
            )
        },
        periodDescriptionKey: "basfi.period",
        notesAllowed: true
    )
    
    static let basg = QuestionnaireDefinition(
        id: .basg,
        version: "1.0.0",
        items: [
            QuestionnaireItem(
                id: "Q1",
                promptKey: "basg.q1.prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.veryWell", 5: "scale.anchor.moderate", 10: "scale.anchor.veryPoor"]
            ),
            QuestionnaireItem(
                id: "Q2",
                promptKey: "basg.q2.prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.veryWell", 5: "scale.anchor.moderate", 10: "scale.anchor.veryPoor"]
            )
        ],
        periodDescriptionKey: "basg.period",
        notesAllowed: true
    )

    static let asqol = QuestionnaireDefinition(
        id: .asqol,
        version: "1.0.0",
        items: (1...18).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "asqol.q\(index).prompt",
                minimum: 0,
                maximum: 1,
                anchors: [0: "scale.anchor.no", 1: "scale.anchor.yes"]
            )
        },
        periodDescriptionKey: "asqol.period",
        notesAllowed: true
    )

    static let asasHI = QuestionnaireDefinition(
        id: .asasHI,
        version: "1.0.0",
        items: (1...17).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "asashi.q\(index).prompt",
                minimum: 0,
                maximum: 1,
                anchors: [0: "scale.anchor.no", 1: "scale.anchor.yes"]
            )
        },
        periodDescriptionKey: "asashi.period",
        notesAllowed: true
    )

    // MARK: - Rheumatoid Arthritis

    static let rapid3 = QuestionnaireDefinition(
        id: .rapid3,
        version: "1.0.0",
        items: [
            QuestionnaireItem(id: "function", promptKey: "rapid3.function.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "pain", promptKey: "rapid3.pain.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "global", promptKey: "rapid3.global.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.veryWell", 5: "scale.anchor.moderate", 10: "scale.anchor.veryPoor"])
        ],
        periodDescriptionKey: "rapid3.period",
        notesAllowed: true
    )

    static let haqDI = QuestionnaireDefinition(
        id: .haqDI,
        version: "1.0.0",
        items: (1...20).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "haq.q\(index).prompt",
                minimum: 0,
                maximum: 3,
                anchors: [0: "haq.anchor.without", 1: "haq.anchor.some", 2: "haq.anchor.much", 3: "haq.anchor.unable"]
            )
        },
        periodDescriptionKey: "haq.period",
        notesAllowed: true
    )

    static let raid = QuestionnaireDefinition(
        id: .raid,
        version: "1.0.0",
        items: [
            QuestionnaireItem(id: "pain", promptKey: "raid.pain.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "function", promptKey: "raid.function.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "fatigue", promptKey: "raid.fatigue.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "sleep", promptKey: "raid.sleep.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "physical", promptKey: "raid.physical.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "emotional", promptKey: "raid.emotional.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "coping", promptKey: "raid.coping.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.veryWell", 5: "scale.anchor.moderate", 10: "scale.anchor.veryPoor"])
        ],
        periodDescriptionKey: "raid.period",
        notesAllowed: true
    )

    // MARK: - Psoriatic Arthritis

    static let psaid12 = QuestionnaireDefinition(
        id: .psaid12,
        version: "1.0.0",
        items: (1...12).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "psaid12.q\(index).prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            )
        },
        periodDescriptionKey: "psaid.period",
        notesAllowed: true
    )

    static let psaid9 = QuestionnaireDefinition(
        id: .psaid9,
        version: "1.0.0",
        items: (1...9).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "psaid9.q\(index).prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            )
        },
        periodDescriptionKey: "psaid.period",
        notesAllowed: true
    )

    static let psaqol = QuestionnaireDefinition(
        id: .psaqol,
        version: "1.0.0",
        items: (1...20).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "psaqol.q\(index).prompt",
                minimum: 0,
                maximum: 1,
                anchors: [0: "scale.anchor.no", 1: "scale.anchor.yes"]
            )
        },
        periodDescriptionKey: "psaqol.period",
        notesAllowed: true
    )

    // MARK: - Lupus

    static let slaq = QuestionnaireDefinition(
        id: .slaq,
        version: "1.0.0",
        items: (1...24).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "slaq.q\(index).prompt",
                minimum: 0,
                maximum: 3,
                anchors: [0: "scale.anchor.none", 1: "scale.anchor.mild", 2: "scale.anchor.moderate", 3: "scale.anchor.severe"]
            )
        },
        periodDescriptionKey: "slaq.period",
        notesAllowed: true
    )

    static let lupusQoL = QuestionnaireDefinition(
        id: .lupusQoL,
        version: "1.0.0",
        items: (1...34).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "lupusqol.q\(index).prompt",
                minimum: 0,
                maximum: 4,
                anchors: [0: "scale.anchor.never", 1: "scale.anchor.rarely", 2: "scale.anchor.sometimes", 3: "scale.anchor.often", 4: "scale.anchor.always"]
            )
        },
        periodDescriptionKey: "lupusqol.period",
        notesAllowed: true
    )

    // MARK: - Sjögren's Syndrome

    static let esspri = QuestionnaireDefinition(
        id: .esspri,
        version: "1.0.0",
        items: [
            QuestionnaireItem(id: "dryness", promptKey: "esspri.dryness.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "fatigue", promptKey: "esspri.fatigue.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]),
            QuestionnaireItem(id: "pain", promptKey: "esspri.pain.prompt",
                            minimum: 0, maximum: 10,
                            anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"])
        ],
        periodDescriptionKey: "esspri.period",
        notesAllowed: true
    )

    // MARK: - Scleroderma

    static let shaq = QuestionnaireDefinition(
        id: .shaq,
        version: "1.0.0",
        items: (1...20).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "shaq.q\(index).prompt",
                minimum: 0,
                maximum: 3,
                anchors: [0: "haq.anchor.without", 1: "haq.anchor.some", 2: "haq.anchor.much", 3: "haq.anchor.unable"]
            )
        },
        periodDescriptionKey: "shaq.period",
        notesAllowed: true
    )

    // MARK: - Gout

    static let gaq2 = QuestionnaireDefinition(
        id: .gaq2,
        version: "1.0.0",
        items: (1...20).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "gaq2.q\(index).prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            )
        },
        periodDescriptionKey: "gaq2.period",
        notesAllowed: true
    )

    static let gis = QuestionnaireDefinition(
        id: .gis,
        version: "1.0.0",
        items: (1...24).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "gis.q\(index).prompt",
                minimum: 0,
                maximum: 5,
                anchors: [0: "scale.anchor.never", 5: "scale.anchor.always"]
            )
        },
        periodDescriptionKey: "gis.period",
        notesAllowed: true
    )

    // MARK: - Osteoarthritis

    static let womac = QuestionnaireDefinition(
        id: .womac,
        version: "1.0.0",
        items: (1...24).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "womac.q\(index).prompt",
                minimum: 0,
                maximum: 4,
                anchors: [0: "scale.anchor.none", 1: "scale.anchor.mild", 2: "scale.anchor.moderate", 3: "scale.anchor.severe", 4: "scale.anchor.extreme"]
            )
        },
        periodDescriptionKey: "womac.period",
        notesAllowed: true
    )

    // MARK: - Fibromyalgia

    static let fiqr = QuestionnaireDefinition(
        id: .fiqr,
        version: "1.0.0",
        items: (1...21).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "fiqr.q\(index).prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            )
        },
        periodDescriptionKey: "fiqr.period",
        notesAllowed: true
    )

    // MARK: - Pediatric

    static let chaq = QuestionnaireDefinition(
        id: .chaq,
        version: "1.0.0",
        items: (1...30).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "chaq.q\(index).prompt",
                minimum: 0,
                maximum: 3,
                anchors: [0: "chaq.anchor.without", 1: "chaq.anchor.some", 2: "chaq.anchor.much", 3: "chaq.anchor.unable"]
            )
        },
        periodDescriptionKey: "chaq.period",
        notesAllowed: true
    )

    // MARK: - Generic

    static let mdhaq = QuestionnaireDefinition(
        id: .mdhaq,
        version: "1.0.0",
        items: (1...10).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "mdhaq.q\(index).prompt",
                minimum: 0,
                maximum: 10,
                anchors: [0: "scale.anchor.none", 5: "scale.anchor.moderate", 10: "scale.anchor.severe"]
            )
        },
        periodDescriptionKey: "mdhaq.period",
        notesAllowed: true
    )

    static let sf36 = QuestionnaireDefinition(
        id: .sf36,
        version: "1.0.0",
        items: (1...36).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "sf36.q\(index).prompt",
                minimum: 0,
                maximum: 5,
                anchors: [0: "scale.anchor.none", 2: "scale.anchor.moderate", 5: "scale.anchor.severe"]
            )
        },
        periodDescriptionKey: "sf36.period",
        notesAllowed: true
    )

    static let eq5d = QuestionnaireDefinition(
        id: .eq5d,
        version: "1.0.0",
        items: [
            QuestionnaireItem(id: "mobility", promptKey: "eq5d.mobility.prompt",
                            minimum: 1, maximum: 5,
                            anchors: [1: "eq5d.anchor.noProblems", 3: "eq5d.anchor.moderate", 5: "eq5d.anchor.extreme"]),
            QuestionnaireItem(id: "selfCare", promptKey: "eq5d.selfcare.prompt",
                            minimum: 1, maximum: 5,
                            anchors: [1: "eq5d.anchor.noProblems", 3: "eq5d.anchor.moderate", 5: "eq5d.anchor.extreme"]),
            QuestionnaireItem(id: "usualActivities", promptKey: "eq5d.activities.prompt",
                            minimum: 1, maximum: 5,
                            anchors: [1: "eq5d.anchor.noProblems", 3: "eq5d.anchor.moderate", 5: "eq5d.anchor.extreme"]),
            QuestionnaireItem(id: "painDiscomfort", promptKey: "eq5d.pain.prompt",
                            minimum: 1, maximum: 5,
                            anchors: [1: "eq5d.anchor.noProblems", 3: "eq5d.anchor.moderate", 5: "eq5d.anchor.extreme"]),
            QuestionnaireItem(id: "anxietyDepression", promptKey: "eq5d.anxiety.prompt",
                            minimum: 1, maximum: 5,
                            anchors: [1: "eq5d.anchor.noProblems", 3: "eq5d.anchor.moderate", 5: "eq5d.anchor.extreme"]),
            QuestionnaireItem(id: "healthVAS", promptKey: "eq5d.healthvas.prompt",
                            minimum: 0, maximum: 100,
                            anchors: [0: "eq5d.anchor.worstHealth", 50: "eq5d.anchor.moderate", 100: "eq5d.anchor.bestHealth"])
        ],
        periodDescriptionKey: "eq5d.period",
        notesAllowed: true
    )

    static let facitFatigue = QuestionnaireDefinition(
        id: .facitFatigue,
        version: "1.0.0",
        items: (1...13).map { index in
            QuestionnaireItem(
                id: "Q\(index)",
                promptKey: "facit.q\(index).prompt",
                minimum: 0,
                maximum: 4,
                anchors: [0: "scale.anchor.never", 2: "scale.anchor.sometimes", 4: "scale.anchor.always"]
            )
        },
        periodDescriptionKey: "facit.period",
        notesAllowed: true
    )

    // Helper method to get definition by ID
    static func definition(for id: QuestionnaireID) -> QuestionnaireDefinition? {
        switch id {
        // Axial SpA/AS
        case .basdai: return .basdai
        case .basfi: return .basfi
        case .basg: return .basg
        case .asqol: return .asqol
        case .asasHI: return .asasHI
        // RA
        case .rapid3: return .rapid3
        case .haqDI: return .haqDI
        case .raid: return .raid
        // PsA
        case .psaid12: return .psaid12
        case .psaid9: return .psaid9
        case .psaqol: return .psaqol
        // Lupus
        case .slaq: return .slaq
        case .lupusQoL: return .lupusQoL
        // Sjögren's
        case .esspri: return .esspri
        // Scleroderma
        case .shaq: return .shaq
        // Gout
        case .gaq2: return .gaq2
        case .gis: return .gis
        // Osteoarthritis
        case .womac: return .womac
        // Fibromyalgia
        case .fiqr: return .fiqr
        // Pediatric
        case .chaq: return .chaq
        // Generic
        case .mdhaq: return .mdhaq
        case .sf36: return .sf36
        case .eq5d: return .eq5d
        case .facitFatigue: return .facitFatigue
        // Placeholder for questionnaires not yet fully implemented
        default: return nil
        }
    }

    static let all: [QuestionnaireDefinition] = [
        .basdai, .basfi, .basg, .asqol, .asasHI,
        .rapid3, .haqDI, .raid,
        .psaid12, .psaid9, .psaqol,
        .slaq, .lupusQoL,
        .esspri,
        .shaq,
        .gaq2, .gis,
        .womac,
        .fiqr,
        .chaq,
        .mdhaq, .sf36, .eq5d, .facitFatigue
    ]
}

enum QuestionnaireScoring {
    static func score(for questionnaireID: QuestionnaireID, answers: [String: Double]) -> Double {
        switch questionnaireID {
        // MARK: - Axial SpA/AS

        case .basdai:
            guard
                let q1 = answers["Q1"],
                let q2 = answers["Q2"],
                let q3 = answers["Q3"],
                let q4 = answers["Q4"],
                let q5 = answers["Q5"],
                let q6 = answers["Q6"]
            else { return .nan }

            let averageStiffness = (q5 + q6) / 2.0
            return (q1 + q2 + q3 + q4 + averageStiffness) / 5.0

        case .basfi:
            let def = QuestionnaireDefinition.basfi
            let values = def.items.compactMap { answers[$0.id] }
            guard values.count == def.items.count else { return .nan }
            return values.reduce(0, +) / Double(values.count)

        case .basg:
            let def = QuestionnaireDefinition.basg
            let values = def.items.compactMap { answers[$0.id] }
            guard values.count == def.items.count else { return .nan }
            return values.reduce(0, +) / Double(values.count)

        // MARK: - Rheumatoid Arthritis

        case .rapid3:
            // RAPID3 = Function + Pain + Global (sum of 0-10 scales, total 0-30)
            guard
                let function = answers["function"],
                let pain = answers["pain"],
                let global = answers["global"]
            else { return .nan }
            return function + pain + global

        case .haqDI:
            // HAQ-DI: Average of 8 categories (each category is max of questions in that category)
            // Simplified: average all 20 questions for now (0-3 scale)
            let def = QuestionnaireDefinition.haqDI
            let values = def.items.compactMap { answers[$0.id] }
            guard values.count == def.items.count else { return .nan }
            return values.reduce(0, +) / Double(values.count)

        case .raid:
            // RAID: Weighted average of 7 domains (0-10 each)
            // For now, simple average (weights: pain 21%, function 16%, fatigue 15%, sleep 12%, physical 12%, emotional 12%, coping 12%)
            return 0.0 // Placeholder - needs full implementation

        // MARK: - Psoriatic Arthritis

        case .psaid12:
            // PsAID-12: Sum of 12 items (0-10 each), then average
            let def = QuestionnaireDefinition.psaid12
            let values = def.items.compactMap { answers[$0.id] }
            guard values.count == def.items.count else { return .nan }
            return values.reduce(0, +) / Double(values.count)

        case .psaid9, .psaqol:
            return 0.0 // Placeholder

        // MARK: - Sjögren's

        case .esspri:
            // ESSPRI: Average of 3 numeric ratings (dryness, fatigue, pain) 0-10
            guard
                let dryness = answers["dryness"],
                let fatigue = answers["fatigue"],
                let pain = answers["pain"]
            else { return .nan }
            return (dryness + fatigue + pain) / 3.0

        case .profadSSI:
            return 0.0 // Placeholder

        // MARK: - Fibromyalgia

        case .fiqr:
            // FIQ-R: Sum of 21 items (0-10 each), total 0-100
            let def = QuestionnaireDefinition.fiqr
            let values = def.items.compactMap { answers[$0.id] }
            guard values.count == def.items.count else { return .nan }
            return values.reduce(0, +) // Sum, not average

        case .acr2010:
            return 0.0 // Placeholder

        // MARK: - Pediatric

        case .chaq:
            // CHAQ: Average of 30 items (0-3 scale)
            let def = QuestionnaireDefinition.chaq
            let values = def.items.compactMap { answers[$0.id] }
            guard values.count == def.items.count else { return .nan }
            return values.reduce(0, +) / Double(values.count)

        case .jamar, .pedsQL, .jadas, .cJadas:
            return 0.0 // Placeholder

        // MARK: - Generic

        case .mdhaq:
            // MDHAQ: Average of 10 items
            let def = QuestionnaireDefinition.mdhaq
            let values = def.items.compactMap { answers[$0.id] }
            guard values.count == def.items.count else { return .nan }
            return values.reduce(0, +) / Double(values.count)

        case .promis, .sf36, .eq5d, .facitFatigue:
            return 0.0 // Placeholder

        // MARK: - Other (not yet implemented)

        default:
            return 0.0
        }
    }
}

struct QuestionnaireThreshold {
    let lowerBound: Double
    let colorHex: String
    let messageKey: String
    
    static func basdaiThresholds() -> [QuestionnaireThreshold] {
        [
            QuestionnaireThreshold(lowerBound: 0.0, colorHex: "#3FB16B", messageKey: "basdai.threshold.low"),
            QuestionnaireThreshold(lowerBound: 4.0, colorHex: "#FFC857", messageKey: "basdai.threshold.moderate"),
            QuestionnaireThreshold(lowerBound: 6.0, colorHex: "#FF8C42", messageKey: "basdai.threshold.high"),
            QuestionnaireThreshold(lowerBound: 8.0, colorHex: "#D9534F", messageKey: "basdai.threshold.veryHigh")
        ]
    }
}

struct QuestionnaireAnswerSet: Codable, Equatable {
    private(set) var values: [String: Double]
    
    init(values: [String: Double]) {
        self.values = values
    }
    
    subscript(_ itemID: String) -> Double? {
        get { values[itemID] }
        set { values[itemID] = newValue }
    }
}

struct QuestionnaireMetaPayload: Codable, Equatable {
    var appVersion: String
    var durationMs: Double
    var isDraft: Bool
    var deviceLocale: String
}

