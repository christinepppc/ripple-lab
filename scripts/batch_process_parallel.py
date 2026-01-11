#!/usr/bin/env python3
"""
Parallel batch processing for ripple detection pipeline.
Processes multiple trials simultaneously using multiprocessing.

Usage:
    python scripts/batch_process_parallel.py --sessions 1 2 3 4 5 --workers 12
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'packages' / 'ripple_core'))

import argparse
import subprocess
from multiprocessing import Pool, cpu_count
from datetime import datetime
import time

# Define trial list for sessions 1-5
TRIALS = [
    (1, 1),
    (1, 12),
    (2, 1),
    (3, 5),
    (4, 1),
    (4, 6),
    (5, 1),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 11),
]

def process_single_trial(trial_info):
    """Process a single trial through the full pipeline."""
    sess, trial = trial_info
    log_file = f"/tmp/sess{sess:03d}_trial{trial:03d}.log"
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as log:
            log.write(f"{'#'*70}\n")
            log.write(f"# Processing Session {sess}, Trial {trial}\n")
            log.write(f"{'#'*70}\n\n")
            
            # Step 1: Bipolar re-referencing
            log.write(">>> STEP 1/4: Bipolar re-referencing\n")
            log.flush()
            result = subprocess.run(
                ['python3', 'scripts/preprocessing/process_bipolar_trial.py', 
                 '--session', str(sess), '--trial', str(trial)],
                capture_output=True, text=True, check=False
            )
            log.write(result.stdout)
            log.write(result.stderr)
            if result.returncode != 0:
                log.write(f"ERROR: Bipolar processing failed\n")
                return (sess, trial, False, time.time() - start_time)
            
            # Step 2: Ripple detection
            log.write("\n>>> STEP 2/4: Ripple detection\n")
            log.flush()
            result = subprocess.run(
                ['python3', 'scripts/detection/detect_ripples_bipolar.py',
                 '--session', str(sess), '--trial', str(trial)],
                capture_output=True, text=True, check=False
            )
            log.write(result.stdout)
            log.write(result.stderr)
            if result.returncode != 0:
                log.write(f"ERROR: Ripple detection failed\n")
                return (sess, trial, False, time.time() - start_time)
            
            # Step 3: Create channel labels
            log.write("\n>>> STEP 3/4: Create channel labels\n")
            log.flush()
            result = subprocess.run(
                ['python3', 'scripts/analysis/create_channel_labels.py',
                 '--session', str(sess), '--trial', str(trial)],
                capture_output=True, text=True, check=False
            )
            log.write(result.stdout)
            log.write(result.stderr)
            if result.returncode != 0:
                log.write(f"ERROR: Channel labels creation failed\n")
                return (sess, trial, False, time.time() - start_time)
            
            # Step 4: Synchrony analysis
            log.write("\n>>> STEP 4/4: Synchrony analysis\n")
            log.flush()
            result = subprocess.run(
                ['python3', 'scripts/analysis/run_synchrony_analysis.py',
                 '--session', str(sess), '--trial', str(trial), '--z_low', '2.5'],
                capture_output=True, text=True, check=False
            )
            log.write(result.stdout)
            log.write(result.stderr)
            if result.returncode != 0:
                log.write(f"ERROR: Synchrony analysis failed\n")
                return (sess, trial, False, time.time() - start_time)
            
            elapsed = time.time() - start_time
            log.write(f"\n✓ COMPLETED: Session {sess}, Trial {trial} in {elapsed:.1f}s\n")
            
        return (sess, trial, True, elapsed)
        
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"\nEXCEPTION: {str(e)}\n")
        return (sess, trial, False, time.time() - start_time)


def main():
    parser = argparse.ArgumentParser(description='Parallel batch processing')
    parser.add_argument('--workers', type=int, default=12, 
                        help='Number of parallel workers (default: 12)')
    parser.add_argument('--trials', nargs='*', 
                        help='Specific trials to process (format: sess:trial, e.g., 1:1 2:5)')
    
    args = parser.parse_args()
    
    # Determine which trials to process
    if args.trials:
        trials_to_process = []
        for t in args.trials:
            sess, trial = map(int, t.split(':'))
            trials_to_process.append((sess, trial))
    else:
        trials_to_process = TRIALS
    
    print("="*80)
    print("PARALLEL BATCH PROCESSING PIPELINE")
    print("="*80)
    print(f"Total trials: {len(trials_to_process)}")
    print(f"Parallel workers: {args.workers}")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    start_time = time.time()
    
    # Process trials in parallel
    with Pool(processes=args.workers) as pool:
        results = pool.map(process_single_trial, trials_to_process)
    
    elapsed_total = time.time() - start_time
    
    # Summary
    print()
    print("="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    
    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]
    
    print(f"Total time: {elapsed_total/60:.1f} minutes ({elapsed_total:.1f}s)")
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    if successful:
        avg_time = sum(r[3] for r in successful) / len(successful)
        print(f"Average time per trial: {avg_time:.1f}s")
        print(f"Speedup vs sequential: {avg_time * len(successful) / elapsed_total:.1f}x")
    
    if failed:
        print("\nFailed trials:")
        for sess, trial, _, _ in failed:
            print(f"  - Session {sess}, Trial {trial}")
    
    print(f"\nIndividual logs: /tmp/sess*_trial*.log")
    print("="*80)


if __name__ == "__main__":
    main()
