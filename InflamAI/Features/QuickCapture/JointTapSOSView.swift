//  JointTapSOSView - Simplified version
import SwiftUI
import CoreData

struct JointTapSOSView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            VStack {
                Text("Quick Flare Capture")
                    .font(.title)
                Text("Feature coming soon")
            }
            .navigationTitle("SOS")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
