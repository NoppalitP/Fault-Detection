#!/usr/bin/env python3
"""
Decibel Formula Monitor - Simplified Version
Monitors and compares the two decibel calculation formulas from audio.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

def calcDecibell(athiwat: float) -> float:
    """Formula 1: 8.6859 * ln(athiwat) + 25.6699"""
    athiwat = max(float(athiwat), 1e-12)
    return 8.6859 * np.log(athiwat) + 25.6699

def _rms_counts(sig: np.ndarray, *, gain_factor: float, subtract_dc: bool) -> float:
    """Calculate RMS in counts units"""
    x = sig.astype(np.float32, copy=False)
    if np.issubdtype(sig.dtype, np.floating):
        x *= 32768.0
    if subtract_dc:
        x = x - np.mean(x)
    x *= float(gain_factor)
    rms = float(np.sqrt(np.mean(np.maximum(x * x, 0.0))))
    return max(rms, 1e-12)

def compute_db(sig: np.ndarray, calib_offset: float = 0.0, *, 
               method: str = "ln", gain_factor: float = 3.0, 
               ref_rms: float = 1000.0, subtract_dc: bool = True, 
               clamp_min: float = 0.0) -> float:
    """Formula 2: Reference-based dB calculation"""
    rms = _rms_counts(sig, gain_factor=gain_factor, subtract_dc=subtract_dc)
    
    if method == "ref":
        db = 20.0 * np.log10(rms / float(ref_rms))
    elif method == "ln":
        db = calcDecibell(rms)
    else:
        raise ValueError("method must be 'ref' or 'ln'")
    
    if clamp_min is not None:
        db = max(db, float(clamp_min))
    
    return float(db + float(calib_offset))

def generate_test_signals():
    """Generate test signals for analysis"""
    sample_rate = 16000
    duration = 0.1
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    
    signals = {
        'sine_1khz': 0.5 * np.sin(2 * np.pi * 1000 * t),
        'sine_500hz': 0.3 * np.sin(2 * np.pi * 500 * t),
        'white_noise': 0.1 * np.random.randn(samples),
        'low_amplitude': 0.01 * np.sin(2 * np.pi * 800 * t),
        'high_amplitude': 0.9 * np.sin(2 * np.pi * 1200 * t),
        'dc_signal': 0.2 * np.ones(samples)
    }
    
    return signals

def test_formulas():
    """Test both formulas with various signals"""
    print("Testing Decibel Formulas...")
    print("=" * 50)
    
    signals = generate_test_signals()
    results = {}
    
    for signal_name, signal in signals.items():
        print(f"\nTesting: {signal_name}")
        
        # Test calcDecibell directly
        rms = _rms_counts(signal, gain_factor=3.0, subtract_dc=True)
        db_ln = calcDecibell(rms)
        
        # Test compute_db with ln method
        db_compute_ln = compute_db(signal, method='ln')
        
        # Test compute_db with ref method
        db_compute_ref = compute_db(signal, method='ref')
        
        results[signal_name] = {
            'rms': rms,
            'calcDecibell': db_ln,
            'compute_db_ln': db_compute_ln,
            'compute_db_ref': db_compute_ref,
            'diff_ln': abs(db_ln - db_compute_ln),
            'diff_ref': abs(db_ln - db_compute_ref)
        }
        
        print(f"  RMS: {rms:.2f}")
        print(f"  calcDecibell: {db_ln:.2f} dB")
        print(f"  compute_db_ln: {db_compute_ln:.2f} dB")
        print(f"  compute_db_ref: {db_compute_ref:.2f} dB")
        print(f"  Difference (ln): {results[signal_name]['diff_ln']:.4f} dB")
        print(f"  Difference (ref): {results[signal_name]['diff_ref']:.4f} dB")
    
    return results

def analyze_formula_behavior():
    """Analyze formula behavior across input ranges"""
    print("\nAnalyzing Formula Behavior...")
    print("=" * 50)
    
    # Test RMS values from 0.001 to 1000
    rms_values = np.logspace(-3, 3, 100)
    
    db_ln_values = [calcDecibell(rms) for rms in rms_values]
    db_ref_values = [20.0 * np.log10(rms / 1000.0) for rms in rms_values]
    
    # Calculate differences
    differences = np.array(db_ln_values) - np.array(db_ref_values)
    
    print(f"RMS range: {rms_values[0]:.3f} to {rms_values[-1]:.1f}")
    print(f"calcDecibell range: {min(db_ln_values):.2f} to {max(db_ln_values):.2f} dB")
    print(f"Reference range: {min(db_ref_values):.2f} to {max(db_ref_values):.2f} dB")
    print(f"Max difference: {max(abs(differences)):.2f} dB")
    print(f"Mean difference: {np.mean(differences):.2f} dB")
    
    return {
        'rms_values': rms_values,
        'db_ln_values': db_ln_values,
        'db_ref_values': db_ref_values,
        'differences': differences
    }

def performance_test(iterations=1000):
    """Test performance of both formulas"""
    print(f"\nPerformance Test ({iterations} iterations)...")
    print("=" * 50)
    
    # Generate test signal
    sample_rate = 16000
    duration = 0.1
    samples = int(sample_rate * duration)
    test_signal = 0.5 * np.sin(2 * np.pi * 1000 * np.linspace(0, duration, samples))
    
    # Test calcDecibell
    start_time = time.time()
    for _ in range(iterations):
        rms = _rms_counts(test_signal, gain_factor=3.0, subtract_dc=True)
        _ = calcDecibell(rms)
    calcDecibell_time = time.time() - start_time
    
    # Test compute_db_ln
    start_time = time.time()
    for _ in range(iterations):
        _ = compute_db(test_signal, method='ln')
    compute_ln_time = time.time() - start_time
    
    # Test compute_db_ref
    start_time = time.time()
    for _ in range(iterations):
        _ = compute_db(test_signal, method='ref')
    compute_ref_time = time.time() - start_time
    
    print(f"calcDecibell: {calcDecibell_time:.4f}s ({iterations/calcDecibell_time:.0f} ops/sec)")
    print(f"compute_db_ln: {compute_ln_time:.4f}s ({iterations/compute_ln_time:.0f} ops/sec)")
    print(f"compute_db_ref: {compute_ref_time:.4f}s ({iterations/compute_ref_time:.0f} ops/sec)")
    
    return {
        'calcDecibell_time': calcDecibell_time,
        'compute_ln_time': compute_ln_time,
        'compute_ref_time': compute_ref_time,
        'iterations': iterations
    }

def create_plots(behavior_data):
    """Create visualization plots"""
    print("\nCreating plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Formula comparison
    ax1.semilogx(behavior_data['rms_values'], behavior_data['db_ln_values'], 
                'b-', linewidth=2, label='calcDecibell (ln)')
    ax1.semilogx(behavior_data['rms_values'], behavior_data['db_ref_values'], 
                'r--', linewidth=2, label='Reference (ref_rms=1000)')
    ax1.set_xlabel('RMS Value')
    ax1.set_ylabel('dB')
    ax1.set_title('Formula Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Differences
    ax2.semilogx(behavior_data['rms_values'], behavior_data['differences'], 
                'g-', linewidth=2)
    ax2.set_xlabel('RMS Value')
    ax2.set_ylabel('Difference (calcDecibell - Reference)')
    ax2.set_title('Formula Differences')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'db_formula_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filename}")
    
    plt.show()

def main():
    """Main function"""
    print("🎵 Decibel Formula Monitor")
    print("=" * 60)
    
    # Run tests
    test_results = test_formulas()
    behavior_data = analyze_formula_behavior()
    performance_data = performance_test()
    
    # Create plots
    create_plots(behavior_data)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_results': test_results,
        'behavior_data': {
            'rms_range': behavior_data['rms_values'].tolist(),
            'max_difference': float(max(abs(behavior_data['differences']))),
            'mean_difference': float(np.mean(behavior_data['differences']))
        },
        'performance_data': performance_data
    }
    
    with open('db_formula_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Analysis completed!")
    print("📊 Results saved to: db_formula_results.json")

if __name__ == "__main__":
    main()
