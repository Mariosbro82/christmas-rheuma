//
//  BASSDAIPracticeView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI

struct BASSDAIPracticeView: View {
    @State private var currentQuestionIndex = 0
    @State private var selectedAnswers: [Int] = Array(repeating: -1, count: 32)
    @State private var showingResults = false
    @Environment(\.dismiss) private var dismiss
    
    private let practiceQuestions = [
        PracticeQuestion(
            question: "How long does morning stiffness typically last in ankylosing spondylitis?",
            options: [
                "Less than 15 minutes",
                "30 minutes to 1 hour",
                "More than 1 hour",
                "It varies day to day"
            ],
            correctAnswer: 2,
            explanation: "Morning stiffness in ankylosing spondylitis typically lasts more than 1 hour, which is a key diagnostic criterion that distinguishes it from mechanical back pain."
        ),
        PracticeQuestion(
            question: "What is the most effective way to reduce morning stiffness?",
            options: [
                "Complete bed rest",
                "Hot shower and gentle exercise",
                "Cold therapy",
                "Avoiding all movement"
            ],
            correctAnswer: 1,
            explanation: "Heat therapy (like a hot shower) combined with gentle exercise and stretching is most effective for reducing morning stiffness in AS patients."
        ),
        PracticeQuestion(
            question: "Morning stiffness severity should be rated based on:",
            options: [
                "Pain level only",
                "Duration only",
                "Impact on daily activities",
                "Weather conditions"
            ],
            correctAnswer: 2,
            explanation: "Morning stiffness severity in BASDAI is rated based on how much it impacts your ability to perform daily activities and function normally."
        ),
        PracticeQuestion(
            question: "Which joints are most commonly affected by morning stiffness in ankylosing spondylitis?",
            options: [
                "Fingers and toes only",
                "Spine and sacroiliac joints",
                "Shoulders and elbows only",
                "Knees and ankles only"
            ],
            correctAnswer: 1,
            explanation: "Morning stiffness in AS primarily affects the spine and sacroiliac joints, though peripheral joints can also be involved in some patients."
        ),
        PracticeQuestion(
            question: "How does morning stiffness in AS typically change with movement?",
            options: [
                "Gets worse with any movement",
                "Improves with gentle activity",
                "Remains constant regardless of activity",
                "Only improves with complete rest"
            ],
            correctAnswer: 1,
            explanation: "Morning stiffness in AS characteristically improves with gentle movement and exercise, which helps distinguish it from mechanical back problems."
        ),
        PracticeQuestion(
            question: "What time of day is morning stiffness typically most severe in AS patients?",
            options: [
                "Late afternoon",
                "Evening before bed",
                "Upon waking and early morning",
                "After meals"
            ],
            correctAnswer: 2,
            explanation: "Morning stiffness is most severe upon waking and in the early morning hours, often requiring 1-2 hours of activity to improve significantly."
        ),
        PracticeQuestion(
            question: "Which factor most commonly worsens morning stiffness in AS?",
            options: [
                "Eating breakfast",
                "Prolonged inactivity or bed rest",
                "Drinking water",
                "Bright light exposure"
            ],
            correctAnswer: 1,
            explanation: "Prolonged inactivity, especially during sleep, worsens morning stiffness in AS. This is why patients often feel stiffest after lying in bed all night."
        ),
        PracticeQuestion(
            question: "How should patients rate morning stiffness duration in the BASDAI assessment?",
            options: [
                "Only count the most severe episodes",
                "Average duration over the past week",
                "Longest duration in the past month",
                "Duration on the assessment day only"
            ],
            correctAnswer: 1,
            explanation: "BASDAI morning stiffness should be rated based on the average duration experienced over the past week, providing a more accurate representation of disease activity."
        ),
        PracticeQuestion(
            question: "What distinguishes AS morning stiffness from normal age-related stiffness?",
            options: [
                "AS stiffness is shorter in duration",
                "AS stiffness affects different joints",
                "AS stiffness lasts longer and improves with exercise",
                "AS stiffness only occurs on weekends"
            ],
            correctAnswer: 2,
            explanation: "AS morning stiffness typically lasts much longer (>1 hour) than normal age-related stiffness and characteristically improves with exercise and movement."
        ),
        PracticeQuestion(
            question: "Which of the following best describes morning stiffness in ankylosing spondylitis?",
            options: [
                "Stiffness that gets worse throughout the day",
                "Stiffness that improves with movement and activity",
                "Stiffness only in the hands and feet",
                "Stiffness that only occurs after exercise"
            ],
            correctAnswer: 1,
            explanation: "Morning stiffness in AS typically improves with movement and activity, which is a key characteristic that helps differentiate it from other conditions."
        ),
        PracticeQuestion(
            question: "Tom reports that his morning stiffness lasts 3 hours every day. On the BASDAI duration scale, this would be rated as:",
            options: ["5-6 (moderate-high)", "7-8 (high-very high)", "9-10 (maximum)", "3-4 (low-moderate)"],
            correctAnswer: 2,
            explanation: "3 hours (180 minutes) of daily morning stiffness represents severe, prolonged stiffness and would rate 9-10 on the BASDAI scale."
        ),
        PracticeQuestion(
            question: "When assessing morning stiffness for BASDAI, which time period should you consider?",
            options: [
                "Only today's stiffness",
                "The past week's average",
                "The worst day in the past month",
                "The best day in the past week"
            ],
            correctAnswer: 1,
            explanation: "BASDAI assessment should reflect the average experience over the past week, not just one day or extreme days."
        ),
        PracticeQuestion(
            question: "Emma takes a warm shower immediately upon waking, which reduces her stiffness from 90 minutes to 30 minutes. What duration should she report?",
            options: [
                "90 minutes (without intervention)",
                "30 minutes (with intervention)",
                "60 minutes (average of both)",
                "It depends on her usual routine"
            ],
            correctAnswer: 3,
            explanation: "Report based on your usual morning routine. If warm showers are part of your regular routine, report the duration with intervention."
        ),
        PracticeQuestion(
            question: "Which factor does NOT typically influence morning stiffness duration in AS patients?",
            options: [
                "Weather conditions",
                "Sleep quality",
                "Time of last meal",
                "Stress levels"
            ],
            correctAnswer: 2,
            explanation: "While weather, sleep quality, and stress can affect morning stiffness, the time of your last meal typically does not have a direct impact on AS morning stiffness."
        ),
        PracticeQuestion(
            question: "Maria experiences morning stiffness that affects her ability to perform daily activities. She struggles to bend down to put on shoes and needs 2 hours before feeling normal. How would this impact her BASDAI assessment?",
            options: [
                "Only duration should be considered (8-9 rating)",
                "Both duration and functional impact should be noted (high severity)",
                "Functional impact is not part of BASDAI scoring",
                "This should be rated as mild since she eventually improves"
            ],
            correctAnswer: 1,
            explanation: "BASDAI considers both duration and severity. Functional impairment lasting 2 hours indicates significant morning stiffness that should be rated highly on both duration and overall wellbeing scales."
        ),
        PracticeQuestion(
            question: "David notices his morning stiffness is worse on cold, rainy days (120 minutes) compared to warm, sunny days (45 minutes). For his weekly BASDAI assessment, he should:",
            options: [
                "Report only the worst days (120 minutes)",
                "Report only the best days (45 minutes)",
                "Calculate the average based on actual weather experienced",
                "Always report 90 minutes as a compromise"
            ],
            correctAnswer: 2,
            explanation: "BASDAI should reflect the actual average experience during the assessment week, taking into account all days including weather variations that occurred."
        ),
        PracticeQuestion(
            question: "Which morning stiffness pattern is most characteristic of ankylosing spondylitis?",
            options: [
                "Brief stiffness (5-10 minutes) that resolves quickly",
                "Prolonged stiffness (>30 minutes) that improves with movement",
                "Stiffness that worsens with any movement or activity",
                "Stiffness that only affects small joints like fingers"
            ],
            correctAnswer: 1,
            explanation: "AS typically causes prolonged morning stiffness (usually >30 minutes, often >1 hour) that characteristically improves with movement and activity, distinguishing it from other conditions."
        ),
        PracticeQuestion(
            question: "Jennifer takes anti-inflammatory medication before bed, which reduces her morning stiffness from 90 minutes to 30 minutes. For BASDAI reporting, she should:",
            options: [
                "Report 90 minutes (pre-medication duration)",
                "Report 30 minutes (current managed duration)",
                "Report 60 minutes (average of both)",
                "Ask her doctor which duration to report"
            ],
            correctAnswer: 1,
            explanation: "BASDAI should reflect your current symptom experience with your current treatment regimen. If medication is part of your regular routine, report the managed symptoms."
        ),
        PracticeQuestion(
            question: "Robert's morning stiffness varies by location: spine (2 hours), hips (45 minutes), shoulders (30 minutes). For the BASDAI morning stiffness question, he should report:",
            options: [
                "2 hours (longest duration)",
                "45 minutes (average duration)",
                "30 minutes (shortest duration)",
                "The duration until overall mobility feels normal"
            ],
            correctAnswer: 3,
            explanation: "Morning stiffness duration should reflect when you feel you can move and function normally overall, not just when one area improves. Consider your overall functional capacity."
        ),
        PracticeQuestion(
            question: "Which intervention is most likely to help reduce morning stiffness duration in AS patients?",
            options: [
                "Staying in bed longer to rest the joints",
                "Gentle stretching and movement exercises",
                "Applying ice packs to stiff areas",
                "Avoiding all physical activity until stiffness resolves"
            ],
            correctAnswer: 1,
            explanation: "Gentle movement and stretching exercises are typically most effective for reducing AS morning stiffness. Rest and inactivity usually worsen stiffness, while appropriate movement helps improve it."
        ),
        PracticeQuestion(
            question: "Anna works night shifts and sleeps during the day. She experiences stiffness when waking up at 4 PM that lasts 75 minutes. How should this be recorded for BASDAI?",
            options: [
                "This doesn't count as morning stiffness since it's afternoon",
                "Record as 75 minutes of morning stiffness",
                "Convert to equivalent morning time and estimate duration",
                "Only count stiffness that occurs before 10 AM"
            ],
            correctAnswer: 1,
            explanation: "'Morning stiffness' in BASDAI refers to stiffness upon awakening, regardless of the actual time of day. For shift workers, this would be stiffness after their main sleep period."
        ),
        PracticeQuestion(
            question: "Which statement about morning stiffness assessment is most accurate?",
            options: [
                "Mild morning stiffness (under 30 minutes) is not clinically significant",
                "Morning stiffness duration directly correlates with disease activity",
                "Any morning stiffness over 1 hour indicates severe AS",
                "Morning stiffness patterns can help monitor treatment effectiveness"
            ],
            correctAnswer: 3,
            explanation: "Morning stiffness patterns and changes over time are valuable for monitoring treatment effectiveness and disease management, making consistent tracking important for optimal care."
        ),
        PracticeQuestion(
            question: "Sarah has been tracking her morning stiffness for 2 weeks. Week 1 average: 90 minutes, Week 2 average: 45 minutes. This improvement most likely indicates:",
            options: [
                "Her AS is getting worse",
                "Treatment adjustments are working effectively",
                "She should stop her current medications",
                "The measurements are inaccurate"
            ],
            correctAnswer: 1,
            explanation: "A significant reduction in morning stiffness duration typically indicates that treatment adjustments (medication, exercise, lifestyle changes) are effectively managing disease activity."
        ),
        PracticeQuestion(
            question: "Which of the following scenarios represents the most severe morning stiffness for BASDAI scoring?",
            options: [
                "30 minutes of mild stiffness that doesn't affect daily activities",
                "2 hours of severe stiffness preventing normal morning routine",
                "45 minutes of moderate stiffness with some difficulty dressing",
                "15 minutes of stiffness that resolves with gentle movement"
            ],
            correctAnswer: 1,
            explanation: "Severe morning stiffness lasting 2 hours that prevents normal activities represents the highest severity level (9-10) on the BASDAI scale, indicating significant disease activity."
        ),
        PracticeQuestion(
            question: "Mark notices his morning stiffness is consistently worse on Mondays after weekend inactivity. This pattern suggests:",
            options: [
                "He should avoid weekend activities",
                "Consistent daily movement is important for AS management",
                "Monday stiffness is normal and not related to AS",
                "He needs stronger pain medication"
            ],
            correctAnswer: 1,
            explanation: "Increased stiffness after periods of inactivity is characteristic of AS. This pattern emphasizes the importance of maintaining consistent daily movement and exercise routines."
        ),
        PracticeQuestion(
            question: "Lisa experiences morning stiffness that varies seasonally: winter (120 min), summer (60 min). For accurate BASDAI tracking, she should:",
            options: [
                "Only report summer measurements",
                "Only report winter measurements",
                "Track seasonal patterns and report current season averages",
                "Always report 90 minutes as a year-round average"
            ],
            correctAnswer: 2,
            explanation: "Seasonal variations in AS symptoms are common. Accurate BASDAI tracking should reflect current seasonal patterns, helping healthcare providers understand environmental triggers and adjust treatment accordingly."
        ),
        PracticeQuestion(
            question: "Which morning routine modification is most likely to reduce AS stiffness duration?",
            options: [
                "Staying in bed an extra hour to rest joints",
                "Taking a hot shower followed by gentle stretching",
                "Immediately engaging in vigorous exercise",
                "Applying ice packs to stiff areas"
            ],
            correctAnswer: 1,
            explanation: "Heat therapy (hot shower) followed by gentle stretching is the most effective morning routine for AS patients. Heat relaxes muscles and joints, while gentle movement helps restore mobility without causing injury."
        ),
        PracticeQuestion(
            question: "When should AS patients be most concerned about changes in their morning stiffness pattern?",
            options: [
                "Any day-to-day variation in stiffness duration",
                "Sudden, sustained increase in duration or severity over several weeks",
                "Occasional days with no morning stiffness",
                "Stiffness that improves with their usual routine"
            ],
            correctAnswer: 1,
            explanation: "Sudden, sustained increases in morning stiffness duration or severity over several weeks may indicate disease flare or need for treatment adjustment and should prompt consultation with healthcare providers."
        ),
        PracticeQuestion(
            question: "Alex works rotating shifts (days/nights). How should he assess morning stiffness for BASDAI?",
            options: [
                "Only count stiffness during day shifts",
                "Assess stiffness upon awakening regardless of time",
                "Skip BASDAI assessment during night shift weeks",
                "Estimate what morning stiffness would be on a normal schedule"
            ],
            correctAnswer: 1,
            explanation: "BASDAI morning stiffness assessment should be based on stiffness upon awakening from the main sleep period, regardless of the actual time of day, to accommodate shift workers and varying schedules."
        ),
        PracticeQuestion(
            question: "Which factor is most important when rating morning stiffness severity (not duration) in BASDAI?",
            options: [
                "The exact time stiffness begins",
                "How much the stiffness interferes with daily functioning",
                "Whether stiffness affects spine or peripheral joints",
                "The weather conditions during the assessment week"
            ],
            correctAnswer: 1,
            explanation: "BASDAI severity rating focuses on functional impact - how much the morning stiffness interferes with your ability to perform daily activities and maintain normal functioning."
        ),
        PracticeQuestion(
            question: "Rachel's morning stiffness improved from 2 hours to 30 minutes after starting a new exercise program. This change most likely reflects:",
            options: [
                "Temporary improvement that will reverse soon",
                "Positive impact of increased physical activity on AS symptoms",
                "Measurement error in her initial assessments",
                "Natural disease remission unrelated to exercise"
            ],
            correctAnswer: 1,
            explanation: "Regular, appropriate exercise is one of the most effective non-pharmacological treatments for AS. Significant improvement in morning stiffness after starting an exercise program typically reflects the positive impact of increased physical activity."
        )
    ]
    
    var body: some View {
        NavigationView {
            if showingResults {
                ResultsView(
                    questions: practiceQuestions,
                    answers: selectedAnswers,
                    onRestart: {
                        currentQuestionIndex = 0
                        selectedAnswers = Array(repeating: -1, count: practiceQuestions.count)
                        showingResults = false
                    }
                )
            } else {
                VStack(spacing: 24) {
                    // Progress indicator
                    ProgressView(value: Double(currentQuestionIndex + 1), total: Double(practiceQuestions.count))
                        .progressViewStyle(LinearProgressViewStyle())
                        .padding(.horizontal)
                    
                    Text("Question \(currentQuestionIndex + 1) of \(practiceQuestions.count)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    // Question content
                    VStack(spacing: 20) {
                        Text(practiceQuestions[currentQuestionIndex].question)
                            .font(.title3)
                            .multilineTextAlignment(.leading)
                            .padding(.horizontal)
                        
                        // Answer options
                        VStack(spacing: 12) {
                            ForEach(0..<practiceQuestions[currentQuestionIndex].options.count, id: \.self) { optionIndex in
                                Button {
                                    selectedAnswers[currentQuestionIndex] = optionIndex
                                } label: {
                                    HStack {
                                        Text(practiceQuestions[currentQuestionIndex].options[optionIndex])
                                            .multilineTextAlignment(.leading)
                                            .foregroundColor(selectedAnswers[currentQuestionIndex] == optionIndex ? .white : .primary)
                                        Spacer()
                                        if selectedAnswers[currentQuestionIndex] == optionIndex {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundColor(.white)
                                        }
                                    }
                                    .padding()
                                    .background(
                                        RoundedRectangle(cornerRadius: 12)
                                            .fill(selectedAnswers[currentQuestionIndex] == optionIndex ? Color.blue : Color(.systemGray6))
                                    )
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(selectedAnswers[currentQuestionIndex] == optionIndex ? Color.blue : Color.clear, lineWidth: 2)
                                    )
                                }
                                .buttonStyle(PlainButtonStyle())
                            }
                        }
                        .padding(.horizontal)
                    }
                    
                    Spacer()
                    
                    // Navigation buttons
                    HStack(spacing: 16) {
                        if currentQuestionIndex > 0 {
                            Button("Previous") {
                                currentQuestionIndex -= 1
                            }
                            .buttonStyle(.bordered)
                        }
                        
                        Spacer()
                        
                        if currentQuestionIndex < practiceQuestions.count - 1 {
                            Button("Next") {
                                currentQuestionIndex += 1
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(Colors.Primary.p500)
                            .disabled(selectedAnswers[currentQuestionIndex] == -1)
                        } else {
                            Button("View Results") {
                                showingResults = true
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(Colors.Primary.p500)
                            .disabled(selectedAnswers[currentQuestionIndex] == -1)
                        }
                    }
                    .padding(.horizontal)
                }
                .padding()
            }
        }
        .navigationTitle("BASDAI Practice")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button("Close") {
                    dismiss()
                }
            }
        }
    }
}

struct PracticeQuestion {
    let question: String
    let options: [String]
    let correctAnswer: Int
    let explanation: String
}

struct ResultsView: View {
    let questions: [PracticeQuestion]
    let answers: [Int]
    let onRestart: () -> Void
    
    private var score: Int {
        zip(questions, answers).reduce(0) { result, pair in
            result + (pair.0.correctAnswer == pair.1 ? 1 : 0)
        }
    }
    
    private var percentage: Double {
        Double(score) / Double(questions.count) * 100
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Score summary
                VStack(spacing: 16) {
                    Text("Practice Complete!")
                        .font(.title)
                        .fontWeight(.bold)
                    
                    VStack(spacing: 8) {
                        Text("\(score)/\(questions.count)")
                            .font(.system(size: 48, weight: .bold, design: .rounded))
                            .foregroundColor(scoreColor)
                        
                        Text(String(format: "%.0f%% Correct", percentage))
                            .font(.title2)
                            .foregroundColor(scoreColor)
                        
                        Text(scoreMessage)
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(16)
                
                // Question review
                VStack(alignment: .leading, spacing: 16) {
                    Text("Review")
                        .font(.headline)
                    
                    ForEach(0..<questions.count, id: \.self) { index in
                        QuestionReviewCard(
                            question: questions[index],
                            userAnswer: answers[index],
                            questionNumber: index + 1
                        )
                    }
                }
                
                // Action buttons
                VStack(spacing: 12) {
                    Button("Try Again") {
                        onRestart()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                    .frame(maxWidth: .infinity)
                    
                    Button("Take Real Assessment") {
                        // This would navigate to the actual BASDAI assessment
                    }
                    .buttonStyle(.bordered)
                    .frame(maxWidth: .infinity)
                }
            }
            .padding()
        }
    }
    
    private var scoreColor: Color {
        switch percentage {
        case 80...100: return .green
        case 60..<80: return .orange
        default: return .red
        }
    }
    
    private var scoreMessage: String {
        switch percentage {
        case 80...100: return "Excellent! You have a strong understanding of BASDAI morning stiffness assessment."
        case 60..<80: return "Good work! Review the explanations to improve your understanding."
        default: return "Keep practicing! Understanding these concepts will help you provide more accurate assessments."
        }
    }
}

struct QuestionReviewCard: View {
    let question: PracticeQuestion
    let userAnswer: Int
    let questionNumber: Int
    
    private var isCorrect: Bool {
        userAnswer == question.correctAnswer
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Question header
            HStack {
                Text("Question \(questionNumber)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Image(systemName: isCorrect ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundColor(isCorrect ? .green : .red)
            }
            
            // Question text
            Text(question.question)
                .font(.body)
                .fontWeight(.medium)
            
            // User's answer
            if userAnswer >= 0 && userAnswer < question.options.count {
                HStack {
                    Text("Your answer:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(question.options[userAnswer])
                        .font(.caption)
                        .foregroundColor(isCorrect ? .green : .red)
                        .fontWeight(.medium)
                }
            }
            
            // Correct answer (if wrong)
            if !isCorrect {
                HStack {
                    Text("Correct answer:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(question.options[question.correctAnswer])
                        .font(.caption)
                        .foregroundColor(.green)
                        .fontWeight(.medium)
                }
            }
            
            // Explanation
            Text(question.explanation)
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.top, 4)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isCorrect ? Color.green.opacity(0.3) : Color.red.opacity(0.3), lineWidth: 1)
        )
    }
}

struct BASSDAIPracticeView_Previews: PreviewProvider {
    static var previews: some View {
        BASSDAIPracticeView()
    }
}