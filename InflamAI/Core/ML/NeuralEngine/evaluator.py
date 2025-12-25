#!/usr/bin/env python3
"""
evaluator.py - The Merciless Judge
Grades model performance against hard metrics for AS flare prediction
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

class PerformanceEvaluator:
    """Ruthless performance grader for the OVERLORD system"""

    # Hard thresholds for medical-grade performance
    THRESHOLDS = {
        'accuracy': {
            'excellent': 0.95,   # 50 points
            'good': 0.92,        # 40 points (target)
            'acceptable': 0.90,  # 30 points
            'poor': 0.85,        # 10 points
        },
        'f1_score': {
            'excellent': 0.90,   # 40 points
            'good': 0.85,        # 35 points (target)
            'acceptable': 0.80,  # 25 points
            'poor': 0.75,        # 10 points
        },
        'latency_ms': {
            'excellent': 25,     # 10 points
            'good': 50,          # 8 points (target)
            'acceptable': 100,   # 5 points
            'poor': 200,         # 2 points
        }
    }

    MINIMUM_PASSING_SCORE = 80  # Must achieve this to proceed

    def __init__(self, protocol_path: str = "OVERLORD_PROTOCOL.md"):
        self.protocol_path = Path(protocol_path)
        self.log_path = Path("logs/evaluation_log.json")
        self.log_path.parent.mkdir(exist_ok=True)

    def grade_performance(
        self,
        accuracy: float,
        f1_score: float,
        latency_ms: float,
        additional_metrics: Optional[Dict] = None
    ) -> Tuple[int, Dict[str, any]]:
        """
        Grade model performance with ruthless precision

        Returns:
            Tuple of (total_score, detailed_report)
        """
        score = 0
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': accuracy,
                'f1_score': f1_score,
                'latency_ms': latency_ms
            },
            'scores': {},
            'feedback': []
        }

        # Grade accuracy (50 points max)
        if accuracy >= self.THRESHOLDS['accuracy']['excellent']:
            score += 50
            report['scores']['accuracy'] = 50
            report['feedback'].append("✅ EXCELLENT: Accuracy exceeds 95%!")
        elif accuracy >= self.THRESHOLDS['accuracy']['good']:
            score += 40
            report['scores']['accuracy'] = 40
            report['feedback'].append("✅ GOOD: Accuracy meets target (>92%)")
        elif accuracy >= self.THRESHOLDS['accuracy']['acceptable']:
            score += 30
            report['scores']['accuracy'] = 30
            report['feedback'].append("⚠️ ACCEPTABLE: Accuracy at 90%, needs improvement")
        elif accuracy >= self.THRESHOLDS['accuracy']['poor']:
            score += 10
            report['scores']['accuracy'] = 10
            report['feedback'].append("❌ POOR: Accuracy below target, consider architecture change")
        else:
            report['scores']['accuracy'] = 0
            report['feedback'].append("💀 FAILURE: Accuracy < 85%, major intervention needed")

        # Grade F1 Score (40 points max) - Critical for medical imbalance!
        if f1_score >= self.THRESHOLDS['f1_score']['excellent']:
            score += 40
            report['scores']['f1_score'] = 40
            report['feedback'].append("✅ EXCELLENT: F1 score shows great balance!")
        elif f1_score >= self.THRESHOLDS['f1_score']['good']:
            score += 35
            report['scores']['f1_score'] = 35
            report['feedback'].append("✅ GOOD: F1 score meets target (>0.85)")
        elif f1_score >= self.THRESHOLDS['f1_score']['acceptable']:
            score += 25
            report['scores']['f1_score'] = 25
            report['feedback'].append("⚠️ ACCEPTABLE: F1 at 0.80, class imbalance issues?")
        elif f1_score >= self.THRESHOLDS['f1_score']['poor']:
            score += 10
            report['scores']['f1_score'] = 10
            report['feedback'].append("❌ POOR: F1 below target, adjust focal loss gamma")
        else:
            report['scores']['f1_score'] = 0
            report['feedback'].append("💀 FAILURE: F1 < 0.75, severe class imbalance")

        # Grade latency (10 points max)
        if latency_ms <= self.THRESHOLDS['latency_ms']['excellent']:
            score += 10
            report['scores']['latency_ms'] = 10
            report['feedback'].append("✅ EXCELLENT: Lightning fast inference!")
        elif latency_ms <= self.THRESHOLDS['latency_ms']['good']:
            score += 8
            report['scores']['latency_ms'] = 8
            report['feedback'].append("✅ GOOD: Latency meets target (<50ms)")
        elif latency_ms <= self.THRESHOLDS['latency_ms']['acceptable']:
            score += 5
            report['scores']['latency_ms'] = 5
            report['feedback'].append("⚠️ ACCEPTABLE: Latency OK but could be optimized")
        elif latency_ms <= self.THRESHOLDS['latency_ms']['poor']:
            score += 2
            report['scores']['latency_ms'] = 2
            report['feedback'].append("❌ POOR: Latency high, consider model pruning")
        else:
            report['scores']['latency_ms'] = 0
            report['feedback'].append("💀 FAILURE: Latency >200ms, unusable for real-time")

        # Final verdict
        report['total_score'] = score
        report['passed'] = score >= self.MINIMUM_PASSING_SCORE

        if score >= 95:
            report['verdict'] = "🏆 SUPREME VICTORY: Model exceeds all expectations!"
        elif score >= 90:
            report['verdict'] = "🎯 EXCELLENT: Model ready for deployment!"
        elif score >= self.MINIMUM_PASSING_SCORE:
            report['verdict'] = "✅ PASSED: Model meets minimum requirements"
        else:
            report['verdict'] = f"❌ FAILED: Score {score} < {self.MINIMUM_PASSING_SCORE}. Must improve!"

        # Add improvement suggestions based on weaknesses
        if accuracy < self.THRESHOLDS['accuracy']['good']:
            report['suggestions'] = report.get('suggestions', [])
            report['suggestions'].append("TRY: Switch LSTM → Transformer architecture")
            report['suggestions'].append("TRY: Increase model capacity (hidden_dim, layers)")
            report['suggestions'].append("TRY: Implement attention mechanisms")

        if f1_score < self.THRESHOLDS['f1_score']['good']:
            report['suggestions'] = report.get('suggestions', [])
            report['suggestions'].append("TRY: Adjust focal loss gamma (current → higher)")
            report['suggestions'].append("TRY: Implement SMOTE for synthetic minority oversampling")
            report['suggestions'].append("TRY: Use weighted sampling in DataLoader")

        if latency_ms > self.THRESHOLDS['latency_ms']['good']:
            report['suggestions'] = report.get('suggestions', [])
            report['suggestions'].append("TRY: Quantization (FP32 → INT8)")
            report['suggestions'].append("TRY: Model pruning (remove 20% lowest weights)")
            report['suggestions'].append("TRY: Knowledge distillation to smaller model")

        # Save evaluation log
        self._save_log(report)

        # Update protocol file
        self._update_protocol(accuracy, f1_score, score)

        return score, report

    def _save_log(self, report: Dict):
        """Save evaluation report to JSON log"""
        logs = []
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                logs = json.load(f)

        logs.append(report)

        # Keep only last 100 evaluations
        if len(logs) > 100:
            logs = logs[-100:]

        with open(self.log_path, 'w') as f:
            json.dump(logs, f, indent=2)

    def _update_protocol(self, accuracy: float, f1_score: float, score: int):
        """Update OVERLORD_PROTOCOL.md with latest metrics"""
        if not self.protocol_path.exists():
            return

        content = self.protocol_path.read_text()
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if line.startswith('BEST_ACCURACY:'):
                current_best = float(line.split(':')[1].strip())
                if accuracy > current_best:
                    lines[i] = f'BEST_ACCURACY: {accuracy:.4f}'
            elif line.startswith('CURRENT_ITERATION:'):
                current_iter = int(line.split(':')[1].strip())
                lines[i] = f'CURRENT_ITERATION: {current_iter + 1}'
            elif line.startswith('LAST_UPDATE:'):
                lines[i] = f'LAST_UPDATE: {datetime.now().isoformat()}'

        self.protocol_path.write_text('\n'.join(lines))

    def check_convergence(self, history_file: str = "logs/training_history.json") -> str:
        """Analyze training history to detect issues"""
        history_path = Path(history_file)
        if not history_path.exists():
            return "No history available"

        with open(history_path, 'r') as f:
            history = json.load(f)

        if len(history) < 5:
            return "Insufficient history for analysis"

        recent = history[-5:]
        accuracies = [h['accuracy'] for h in recent]

        # Check for plateau
        if max(accuracies) - min(accuracies) < 0.01:
            return "PLATEAU_DETECTED: Accuracy stuck, major change needed"

        # Check for oscillation
        if all(accuracies[i] > accuracies[i+1] if i%2==0 else accuracies[i] < accuracies[i+1]
               for i in range(len(accuracies)-1)):
            return "OSCILLATION_DETECTED: Learning rate might be too high"

        # Check for degradation
        if accuracies[-1] < accuracies[0] - 0.05:
            return "DEGRADATION_DETECTED: Model getting worse, check for bugs"

        return "HEALTHY: Model improving normally"

def main():
    """CLI interface for manual evaluation"""
    parser = argparse.ArgumentParser(description='Grade model performance')
    parser.add_argument('--accuracy', type=float, required=True, help='Validation accuracy (0-1)')
    parser.add_argument('--f1', type=float, required=True, help='F1 score (0-1)')
    parser.add_argument('--latency', type=float, required=True, help='Inference latency in ms')
    parser.add_argument('--check-convergence', action='store_true', help='Analyze training convergence')

    args = parser.parse_args()

    evaluator = PerformanceEvaluator()

    if args.check_convergence:
        status = evaluator.check_convergence()
        print(f"Convergence Status: {status}")
        return

    score, report = evaluator.grade_performance(
        accuracy=args.accuracy,
        f1_score=args.f1,
        latency_ms=args.latency
    )

    # Print detailed report
    print("\n" + "="*60)
    print("OVERLORD PERFORMANCE EVALUATION")
    print("="*60)
    print(f"Accuracy: {args.accuracy:.4f} → {report['scores']['accuracy']} points")
    print(f"F1 Score: {args.f1:.4f} → {report['scores']['f1_score']} points")
    print(f"Latency: {args.latency:.1f}ms → {report['scores']['latency_ms']} points")
    print("-"*60)
    print(f"TOTAL SCORE: {score}/100")
    print(f"VERDICT: {report['verdict']}")
    print("-"*60)

    if 'feedback' in report:
        print("\nFeedback:")
        for fb in report['feedback']:
            print(f"  {fb}")

    if 'suggestions' in report:
        print("\nImprovement Suggestions:")
        for suggestion in report['suggestions']:
            print(f"  • {suggestion}")

    # Exit with error if failed (for automation)
    if not report['passed']:
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()