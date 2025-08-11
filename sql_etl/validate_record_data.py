#!/usr/bin/env python3
"""
Validation script for offline record data in data/record directory.
This script analyzes the audio files to validate data quality and provide insights.
"""

import os
import wave
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecordDataValidator:
    """Validates and analyzes offline record data."""
    
    def __init__(self, record_dir: str = "data/record"):
        self.record_dir = Path(record_dir)
        self.normal_dir = self.record_dir / "normal"
        self.anormal_dir = self.record_dir / "anormal"
        
    def validate_audio_file(self, file_path: Path) -> dict:
        """Validate a single audio file and extract metadata."""
        try:
            with wave.open(str(file_path), 'rb') as wav_file:
                # Extract audio properties
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                duration = frames / sample_rate
                
                # Extract timestamp from filename
                filename = file_path.stem
                timestamp_str = filename.split('_')[1:4]  # Extract date, time, milliseconds
                
                return {
                    'filename': filename,
                    'file_path': str(file_path),
                    'frames': frames,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'sample_width': sample_width,
                    'duration': duration,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'timestamp': '_'.join(timestamp_str),
                    'status': 'valid'
                }
                
        except Exception as e:
            logger.warning(f"Failed to validate {file_path}: {e}")
            return {
                'filename': file_path.stem,
                'file_path': str(file_path),
                'status': 'invalid',
                'error': str(e)
            }
    
    def analyze_directory(self, dir_path: Path, category: str) -> list:
        """Analyze all files in a directory."""
        logger.info(f"Analyzing {category} directory: {dir_path}")
        
        if not dir_path.exists():
            logger.error(f"Directory does not exist: {dir_path}")
            return []
        
        files = list(dir_path.glob("*.wav"))
        logger.info(f"Found {len(files)} WAV files in {category}")
        
        results = []
        for file_path in files:
            result = self.validate_audio_file(file_path)
            result['category'] = category
            results.append(result)
            
        return results
    
    def generate_statistics(self, all_results: list) -> dict:
        """Generate comprehensive statistics from validation results."""
        df = pd.DataFrame(all_results)
        
        # Filter valid files
        valid_df = df[df['status'] == 'valid']
        invalid_df = df[df['status'] == 'invalid']
        
        stats = {
            'total_files': len(df),
            'valid_files': len(valid_df),
            'invalid_files': len(invalid_df),
            'validation_rate': len(valid_df) / len(df) * 100 if len(df) > 0 else 0,
            
            # File size statistics
            'avg_file_size_mb': valid_df['file_size_mb'].mean() if len(valid_df) > 0 else 0,
            'min_file_size_mb': valid_df['file_size_mb'].min() if len(valid_df) > 0 else 0,
            'max_file_size_mb': valid_df['file_size_mb'].max() if len(valid_df) > 0 else 0,
            
            # Duration statistics
            'avg_duration_sec': valid_df['duration'].mean() if len(valid_df) > 0 else 0,
            'min_duration_sec': valid_df['duration'].min() if len(valid_df) > 0 else 0,
            'max_duration_sec': valid_df['duration'].max() if len(valid_df) > 0 else 0,
            
            # Audio format statistics
            'sample_rates': valid_df['sample_rate'].unique().tolist() if len(valid_df) > 0 else [],
            'channels': valid_df['channels'].unique().tolist() if len(valid_df) > 0 else [],
            'sample_widths': valid_df['sample_width'].unique().tolist() if len(valid_df) > 0 else [],
            
            # Category breakdown
            'normal_files': len(df[df['category'] == 'normal']),
            'anormal_files': len(df[df['category'] == 'anormal']),
        }
        
        return stats
    
    def check_data_consistency(self, all_results: list) -> dict:
        """Check for data consistency issues."""
        df = pd.DataFrame(all_results)
        valid_df = df[df['status'] == 'valid']
        
        issues = []
        
        # Check for consistent sample rates
        if len(valid_df) > 0:
            unique_sample_rates = valid_df['sample_rate'].unique()
            if len(unique_sample_rates) > 1:
                issues.append(f"Multiple sample rates detected: {unique_sample_rates}")
            
            # Check for consistent channels
            unique_channels = valid_df['channels'].unique()
            if len(unique_channels) > 1:
                issues.append(f"Multiple channel configurations: {unique_channels}")
            
            # Check for consistent sample widths
            unique_sample_widths = valid_df['sample_width'].unique()
            if len(unique_sample_widths) > 1:
                issues.append(f"Multiple sample widths: {unique_sample_widths}")
            
            # Check for duration consistency
            duration_std = valid_df['duration'].std()
            if duration_std > 1.0:  # More than 1 second variation
                issues.append(f"High duration variation: std={duration_std:.2f}s")
        
        return {
            'issues_found': len(issues),
            'issues': issues
        }
    
    def generate_report(self, output_file: str = "record_data_validation_report.txt"):
        """Generate a comprehensive validation report."""
        logger.info("Starting record data validation...")
        
        # Analyze both directories
        normal_results = self.analyze_directory(self.normal_dir, "normal")
        anormal_results = self.analyze_directory(self.anormal_dir, "anormal")
        
        all_results = normal_results + anormal_results
        
        # Generate statistics
        stats = self.generate_statistics(all_results)
        consistency = self.check_data_consistency(all_results)
        
        # Generate report
        report_lines = [
            "=" * 60,
            "RECORD DATA VALIDATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY STATISTICS",
            "-" * 30,
            f"Total files: {stats['total_files']}",
            f"Valid files: {stats['valid_files']}",
            f"Invalid files: {stats['invalid_files']}",
            f"Validation rate: {stats['validation_rate']:.1f}%",
            "",
            "CATEGORY BREAKDOWN",
            "-" * 30,
            f"Normal files: {stats['normal_files']}",
            f"Anormal files: {stats['anormal_files']}",
            "",
            "FILE CHARACTERISTICS",
            "-" * 30,
            f"Average file size: {stats['avg_file_size_mb']:.2f} MB",
            f"File size range: {stats['min_file_size_mb']:.2f} - {stats['max_file_size_mb']:.2f} MB",
            f"Average duration: {stats['avg_duration_sec']:.2f} seconds",
            f"Duration range: {stats['min_duration_sec']:.2f} - {stats['max_duration_sec']:.2f} seconds",
            "",
            "AUDIO FORMAT",
            "-" * 30,
            f"Sample rates: {stats['sample_rates']}",
            f"Channels: {stats['channels']}",
            f"Sample widths: {stats['sample_widths']}",
            "",
            "DATA CONSISTENCY",
            "-" * 30,
            f"Issues found: {consistency['issues_found']}",
        ]
        
        if consistency['issues']:
            report_lines.append("Issues:")
            for issue in consistency['issues']:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.append("No consistency issues detected.")
        
        # Add detailed file information
        report_lines.extend([
            "",
            "DETAILED FILE ANALYSIS",
            "-" * 30,
        ])
        
        # Group by category
        df = pd.DataFrame(all_results)
        for category in ['normal', 'anormal']:
            cat_df = df[df['category'] == category]
            if len(cat_df) > 0:
                report_lines.extend([
                    f"\n{category.upper()} FILES:",
                    f"  Count: {len(cat_df)}",
                    f"  Valid: {len(cat_df[cat_df['status'] == 'valid'])}",
                    f"  Invalid: {len(cat_df[cat_df['status'] == 'invalid'])}",
                ])
                
                # Show first few files as examples
                valid_files = cat_df[cat_df['status'] == 'valid'].head(3)
                for _, file_info in valid_files.iterrows():
                    report_lines.append(
                        f"  Example: {file_info['filename']} - "
                        f"{file_info['duration']:.2f}s, "
                        f"{file_info['sample_rate']}Hz, "
                        f"{file_info['channels']}ch"
                    )
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Validation report written to: {output_file}")
        
        return {
            'stats': stats,
            'consistency': consistency,
            'total_files': len(all_results)
        }
    
    def print_summary(self, results: dict):
        """Print a summary of validation results."""
        print("\n" + "=" * 50)
        print("RECORD DATA VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total files analyzed: {results['total_files']}")
        print(f"Validation success rate: {results['stats']['validation_rate']:.1f}%")
        print(f"Data consistency issues: {results['consistency']['issues_found']}")
        print(f"Normal files: {results['stats']['normal_files']}")
        print(f"Anormal files: {results['stats']['anormal_files']}")
        print(f"Average file duration: {results['stats']['avg_duration_sec']:.2f} seconds")
        print(f"Average file size: {results['stats']['avg_file_size_mb']:.2f} MB")
        
        if results['consistency']['issues']:
            print("\nIssues detected:")
            for issue in results['consistency']['issues']:
                print(f"  - {issue}")

def main():
    """Main function to run the validation."""
    validator = RecordDataValidator()
    
    try:
        results = validator.generate_report()
        validator.print_summary(results)
        
        print(f"\nDetailed report saved to: record_data_validation_report.txt")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()
