#!/usr/bin/env python3
"""
Decibel Formula Monitor and Analysis Tool

This script monitors and compares two different decibel calculation formulas:
1. calcDecibell: 8.6859 * ln(athiwat) + 25.6699
2. compute_db: Reference-based method with multiple options

The script provides:
- Formula comparison and validation
- Input/output analysis
- Performance testing
- Real-world signal testing
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DecibelFormulaMonitor:
    """Monitor and analyze decibel calculation formulas."""
    
    def __init__(self):
        self.results = {}
        self.test_signals = {}
        self.performance_data = {}
        
    def calcDecibell(self, athiwat: float) -> float:
        """
        Formula 1: 8.6859 * ln(athiwat) + 25.6699
        Simulates C++ formula using natural log
        """
        athiwat = max(float(athiwat), 1e-12)  # Prevent log(0)
        return 8.6859 * np.log(athiwat) + 25.6699

    def _rms_counts(
        self,
        sig: np.ndarray,
        *,
        gain_factor: float,
        subtract_dc: bool,
    ) -> float:
        """
        Prepare signal in "counts" units (int16) and calculate RMS
        - If sig is float [-1,1], multiply by 32768 to scale to int16
        - (Optional) Remove DC component
        - Multiply by gain_factor to simulate firmware
        """
        x = sig.astype(np.float32, copy=False)
        if np.issubdtype(sig.dtype, np.floating):
            x *= 32768.0
        if subtract_dc:
            x = x - np.mean(x)
        x *= float(gain_factor)
        rms = float(np.sqrt(np.mean(np.maximum(x * x, 0.0))))
        return max(rms, 1e-12)

    def compute_db(
        self,
        sig: np.ndarray,
        calib_offset: float = 0.0,
        *,
        method: str = "ln",           # "ref" or "ln"
        gain_factor: float = 3.0,     # Matches GAIN_FACTOR in C++ example
        ref_rms: float = 1000.0,      # Matches REF in C++ example
        subtract_dc: bool = True,
        clamp_min: float = 0.0,       # Choose minimum clamp (e.g., 0 dB)
    ) -> float:
        """
        Calculate dB from one signal window
        - method="ref": dB = 20*log10(rms / ref_rms) (compare with measure_dB example)
        - method="ln" : dB = calcDecibell(rms) (compare with calcDecibell function)
        Both add calib_offset afterward for field calibration
        """
        rms = self._rms_counts(sig, gain_factor=gain_factor, subtract_dc=subtract_dc)

        if method == "ref":
            db = 20.0 * np.log10(rms / float(ref_rms))
        elif method == "ln":
            db = self.calcDecibell(rms)
        else:
            raise ValueError("method must be 'ref' or 'ln'")

        if clamp_min is not None:
            db = max(db, float(clamp_min))

        return float(db + float(calib_offset))

    def generate_test_signals(self) -> Dict[str, np.ndarray]:
        """Generate various test signals for formula validation."""
        logger.info("Generating test signals...")
        
        # Test signal parameters
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        signals = {}
        
        # 1. Pure sine wave (1 kHz)
        signals['sine_1khz'] = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        # 2. Pure sine wave (500 Hz)
        signals['sine_500hz'] = 0.3 * np.sin(2 * np.pi * 500 * t)
        
        # 3. White noise
        signals['white_noise'] = 0.1 * np.random.randn(samples)
        
        # 4. Pink noise (simulated)
        signals['pink_noise'] = 0.05 * np.random.randn(samples) * np.exp(-t/2)
        
        # 5. Impulse signal
        signals['impulse'] = np.zeros(samples)
        signals['impulse'][samples//2] = 1.0
        
        # 6. Square wave
        signals['square_wave'] = 0.4 * np.sign(np.sin(2 * np.pi * 200 * t))
        
        # 7. Low amplitude signal
        signals['low_amplitude'] = 0.01 * np.sin(2 * np.pi * 800 * t)
        
        # 8. High amplitude signal
        signals['high_amplitude'] = 0.9 * np.sin(2 * np.pi * 1200 * t)
        
        # 9. DC signal
        signals['dc_signal'] = 0.2 * np.ones(samples)
        
        # 10. Mixed frequency signal
        signals['mixed_freq'] = (0.3 * np.sin(2 * np.pi * 300 * t) + 
                               0.2 * np.sin(2 * np.pi * 600 * t) + 
                               0.1 * np.sin(2 * np.pi * 900 * t))
        
        self.test_signals = signals
        logger.info(f"Generated {len(signals)} test signals")
        return signals

    def test_formula_consistency(self) -> Dict[str, Dict]:
        """Test formula consistency across different inputs."""
        logger.info("Testing formula consistency...")
        
        if not self.test_signals:
            self.generate_test_signals()
        
        results = {}
        
        for signal_name, signal in self.test_signals.items():
            logger.info(f"Testing signal: {signal_name}")
            
            # Test different methods and parameters
            test_configs = [
                {'method': 'ln', 'gain_factor': 3.0, 'subtract_dc': True},
                {'method': 'ln', 'gain_factor': 3.0, 'subtract_dc': False},
                {'method': 'ref', 'gain_factor': 3.0, 'subtract_dc': True, 'ref_rms': 1000.0},
                {'method': 'ref', 'gain_factor': 3.0, 'subtract_dc': False, 'ref_rms': 1000.0},
                {'method': 'ref', 'gain_factor': 1.0, 'subtract_dc': True, 'ref_rms': 1000.0},
                {'method': 'ref', 'gain_factor': 5.0, 'subtract_dc': True, 'ref_rms': 1000.0},
            ]
            
            signal_results = {}
            
            for i, config in enumerate(test_configs):
                try:
                    db_result = self.compute_db(signal, **config)
                    signal_results[f"config_{i+1}"] = {
                        'config': config,
                        'db_value': db_result,
                        'status': 'success'
                    }
                except Exception as e:
                    signal_results[f"config_{i+1}"] = {
                        'config': config,
                        'error': str(e),
                        'status': 'error'
                    }
            
            results[signal_name] = signal_results
        
        self.results['consistency'] = results
        return results

    def analyze_formula_behavior(self) -> Dict[str, Dict]:
        """Analyze formula behavior across input ranges."""
        logger.info("Analyzing formula behavior...")
        
        # Test RMS values across different ranges
        rms_values = np.logspace(-3, 3, 100)  # 0.001 to 1000
        
        analysis = {}
        
        # Test calcDecibell formula
        db_values_ln = []
        for rms in rms_values:
            db_values_ln.append(self.calcDecibell(rms))
        
        # Test reference-based formula with different ref_rms values
        ref_rms_values = [100, 500, 1000, 2000]
        db_values_ref = {}
        
        for ref_rms in ref_rms_values:
            db_values_ref[ref_rms] = []
            for rms in rms_values:
                db_values_ref[ref_rms].append(20.0 * np.log10(rms / ref_rms))
        
        analysis['rms_range'] = rms_values.tolist()
        analysis['db_values_ln'] = db_values_ln
        analysis['db_values_ref'] = db_values_ref
        
        # Calculate differences between formulas
        differences = {}
        for ref_rms in ref_rms_values:
            differences[ref_rms] = np.array(db_values_ln) - np.array(db_values_ref[ref_rms])
        
        analysis['differences'] = {str(k): v.tolist() for k, v in differences.items()}
        
        self.results['behavior_analysis'] = analysis
        return analysis

    def performance_test(self, iterations: int = 1000) -> Dict[str, float]:
        """Test performance of both formulas."""
        logger.info(f"Running performance test with {iterations} iterations...")
        
        # Generate test signal
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        test_signal = 0.5 * np.sin(2 * np.pi * 1000 * np.linspace(0, duration, samples))
        
        performance = {}
        
        # Test calcDecibell performance
        start_time = time.time()
        for _ in range(iterations):
            rms = self._rms_counts(test_signal, gain_factor=3.0, subtract_dc=True)
            _ = self.calcDecibell(rms)
        calcDecibell_time = time.time() - start_time
        
        # Test compute_db with ln method
        start_time = time.time()
        for _ in range(iterations):
            _ = self.compute_db(test_signal, method='ln')
        compute_db_ln_time = time.time() - start_time
        
        # Test compute_db with ref method
        start_time = time.time()
        for _ in range(iterations):
            _ = self.compute_db(test_signal, method='ref')
        compute_db_ref_time = time.time() - start_time
        
        performance['calcDecibell_time'] = calcDecibell_time
        performance['compute_db_ln_time'] = compute_db_ln_time
        performance['compute_db_ref_time'] = compute_db_ref_time
        performance['iterations'] = iterations
        
        # Calculate operations per second
        performance['calcDecibell_ops_per_sec'] = iterations / calcDecibell_time
        performance['compute_db_ln_ops_per_sec'] = iterations / compute_db_ln_time
        performance['compute_db_ref_ops_per_sec'] = iterations / compute_db_ref_time
        
        self.performance_data = performance
        return performance

    def create_visualization(self, save_plots: bool = True) -> None:
        """Create visualizations of formula behavior."""
        logger.info("Creating visualizations...")
        
        if 'behavior_analysis' not in self.results:
            self.analyze_formula_behavior()
        
        analysis = self.results['behavior_analysis']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Decibel Formula Analysis and Comparison', fontsize=16)
        
        # Plot 1: Formula comparison
        ax1.semilogx(analysis['rms_range'], analysis['db_values_ln'], 
                    'b-', linewidth=2, label='calcDecibell (ln)')
        
        for ref_rms, db_values in analysis['db_values_ref'].items():
            ax1.semilogx(analysis['rms_range'], db_values, 
                        '--', linewidth=1, label=f'ref (ref_rms={ref_rms})')
        
        ax1.set_xlabel('RMS Value')
        ax1.set_ylabel('dB')
        ax1.set_title('Formula Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Differences between formulas
        for ref_rms, diff in analysis['differences'].items():
            ax2.semilogx(analysis['rms_range'], diff, 
                        linewidth=1, label=f'ref_rms={ref_rms}')
        
        ax2.set_xlabel('RMS Value')
        ax2.set_ylabel('Difference (calcDecibell - ref)')
        ax2.set_title('Formula Differences')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance comparison
        if self.performance_data:
            methods = ['calcDecibell', 'compute_db_ln', 'compute_db_ref']
            ops_per_sec = [
                self.performance_data['calcDecibell_ops_per_sec'],
                self.performance_data['compute_db_ln_ops_per_sec'],
                self.performance_data['compute_db_ref_ops_per_sec']
            ]
            
            bars = ax3.bar(methods, ops_per_sec, color=['blue', 'green', 'orange'])
            ax3.set_ylabel('Operations per Second')
            ax3.set_title('Performance Comparison')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, ops_per_sec):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ops_per_sec)*0.01,
                        f'{value:.0f}', ha='center', va='bottom')
        
        # Plot 4: Signal test results
        if 'consistency' in self.results:
            signal_names = list(self.results['consistency'].keys())
            success_counts = []
            
            for signal_name in signal_names:
                signal_results = self.results['consistency'][signal_name]
                success_count = sum(1 for result in signal_results.values() 
                                  if result['status'] == 'success')
                success_counts.append(success_count)
            
            bars = ax4.bar(range(len(signal_names)), success_counts, color='lightblue')
            ax4.set_xlabel('Test Signal')
            ax4.set_ylabel('Successful Configurations')
            ax4.set_title('Formula Success Rate by Signal Type')
            ax4.set_xticks(range(len(signal_names)))
            ax4.set_xticklabels(signal_names, rotation=45, ha='right')
            ax4.set_ylim(0, max(success_counts) + 1)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'db_formula_analysis_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to: {filename}")
        
        plt.show()

    def generate_report(self, output_file: str = "db_formula_report.json") -> Dict:
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive report...")
        
        # Run all analyses
        self.test_formula_consistency()
        self.analyze_formula_behavior()
        self.performance_test()
        
        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_test_signals': len(self.test_signals),
                'formula_methods': ['calcDecibell', 'compute_db_ln', 'compute_db_ref'],
                'test_configurations': 6
            },
            'consistency_results': self.results.get('consistency', {}),
            'behavior_analysis': self.results.get('behavior_analysis', {}),
            'performance_data': self.performance_data,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved to: {output_file}")
        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Analyze consistency results
        if 'consistency' in self.results:
            total_tests = 0
            successful_tests = 0
            
            for signal_results in self.results['consistency'].values():
                for result in signal_results.values():
                    total_tests += 1
                    if result['status'] == 'success':
                        successful_tests += 1
            
            success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
            recommendations.append(f"Overall success rate: {success_rate:.1f}%")
            
            if success_rate < 90:
                recommendations.append("Consider investigating failed configurations")
        
        # Analyze performance
        if self.performance_data:
            fastest_method = min([
                ('calcDecibell', self.performance_data['calcDecibell_ops_per_sec']),
                ('compute_db_ln', self.performance_data['compute_db_ln_ops_per_sec']),
                ('compute_db_ref', self.performance_data['compute_db_ref_ops_per_sec'])
            ], key=lambda x: x[1])
            
            recommendations.append(f"Fastest method: {fastest_method[0]} ({fastest_method[1]:.0f} ops/sec)")
        
        # Analyze behavior differences
        if 'behavior_analysis' in self.results:
            analysis = self.results['behavior_analysis']
            if 'differences' in analysis:
                max_diff = max([
                    max(abs(np.array(diff))) 
                    for diff in analysis['differences'].values()
                ])
                recommendations.append(f"Maximum difference between formulas: {max_diff:.2f} dB")
        
        return recommendations

    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        print("\n" + "=" * 60)
        print("DECIBEL FORMULA MONITORING SUMMARY")
        print("=" * 60)
        
        if self.test_signals:
            print(f"Test signals generated: {len(self.test_signals)}")
        
        if 'consistency' in self.results:
            total_tests = sum(len(results) for results in self.results['consistency'].values())
            successful_tests = sum(
                sum(1 for result in signal_results.values() if result['status'] == 'success')
                for signal_results in self.results['consistency'].values()
            )
            print(f"Consistency tests: {successful_tests}/{total_tests} successful")
        
        if self.performance_data:
            print(f"Performance test iterations: {self.performance_data['iterations']}")
            print(f"calcDecibell performance: {self.performance_data['calcDecibell_ops_per_sec']:.0f} ops/sec")
            print(f"compute_db_ln performance: {self.performance_data['compute_db_ln_ops_per_sec']:.0f} ops/sec")
            print(f"compute_db_ref performance: {self.performance_data['compute_db_ref_ops_per_sec']:.0f} ops/sec")
        
        if 'behavior_analysis' in self.results:
            print("Behavior analysis completed")
        
        print("=" * 60)

def main():
    """Main function to run the decibel formula monitoring."""
    monitor = DecibelFormulaMonitor()
    
    try:
        print("🎵 Starting Decibel Formula Monitoring and Analysis")
        print("=" * 60)
        
        # Run comprehensive analysis
        monitor.generate_test_signals()
        monitor.test_formula_consistency()
        monitor.analyze_formula_behavior()
        monitor.performance_test()
        
        # Generate report and visualization
        report = monitor.generate_report()
        monitor.create_visualization()
        
        # Print summary
        monitor.print_summary()
        
        print("\n✅ Analysis completed successfully!")
        print("📊 Check the generated report and visualization files for detailed results.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
