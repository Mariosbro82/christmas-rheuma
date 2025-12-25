//
//  PressureTrendChart.swift
//  InflamAI-Swift
//
//  Barometric pressure trend visualization
//  Shows 48-hour forecast with significant drops highlighted
//

import SwiftUI
import Charts

struct PressureTrendChart: View {
    let dataPoints: [PressureDataPoint]

    var body: some View {
        Chart {
            // Pressure line only (no area fill to avoid rendering artifacts)
            ForEach(dataPoints) { point in
                LineMark(
                    x: .value("Time", point.timestamp),
                    y: .value("Pressure", point.pressure)
                )
                .foregroundStyle(Color.cyan)
                .lineStyle(StrokeStyle(lineWidth: 2.5))
                .interpolationMethod(.catmullRom)
            }

            // Highlight significant drops (change < -5 hPa)
            ForEach(dataPoints.filter { $0.change < -5 }) { point in
                PointMark(
                    x: .value("Time", point.timestamp),
                    y: .value("Pressure", point.pressure)
                )
                .foregroundStyle(.red)
                .symbolSize(80)
            }

            // Average pressure reference line
            if !dataPoints.isEmpty {
                RuleMark(y: .value("Average", averagePressure))
                    .foregroundStyle(.gray.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 5]))
            }
        }
        .chartXAxis {
            AxisMarks(values: .stride(by: .hour, count: 6)) { value in
                if let date = value.as(Date.self) {
                    AxisValueLabel {
                        Text(date, format: .dateTime.hour())
                            .font(.caption2)
                    }
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                        .foregroundStyle(Color.gray.opacity(0.3))
                }
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 5)) { value in
                if let pressure = value.as(Double.self) {
                    AxisValueLabel {
                        Text(String(format: "%.0f", pressure))
                            .font(.caption2)
                    }
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                        .foregroundStyle(Color.gray.opacity(0.3))
                }
            }
        }
        .chartYScale(domain: yAxisMin...yAxisMax)
        .chartPlotStyle { plotArea in
            plotArea
                .background(Color.clear)
        }
    }

    // MARK: - Computed Properties

    private var averagePressure: Double {
        guard !dataPoints.isEmpty else { return 1013.25 }
        return dataPoints.map(\.pressure).reduce(0, +) / Double(dataPoints.count)
    }

    private var yAxisMin: Double {
        let minPressure = dataPoints.map(\.pressure).min() ?? 1000.0
        return minPressure - 10
    }

    private var yAxisMax: Double {
        let maxPressure = dataPoints.map(\.pressure).max() ?? 1030.0
        return maxPressure + 10
    }
}

// MARK: - Preview

struct PressureTrendChart_Previews: PreviewProvider {
    static var previews: some View {
        // Generate mock data
        let mockData = (0..<48).map { hour in
            PressureDataPoint(
                timestamp: Date().addingTimeInterval(Double(hour) * 3600),
                pressure: 1013.25 + Double.random(in: -15...15),
                change: Double.random(in: -8...8)
            )
        }

        PressureTrendChart(dataPoints: mockData)
            .frame(height: 200)
            .padding()
    }
}
