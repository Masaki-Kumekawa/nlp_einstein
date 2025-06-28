"""
Check the current progress of the experiment.
"""

import json
import os
from datetime import datetime

def check_progress():
    """Display current experiment progress."""
    
    # Check if status file exists
    if not os.path.exists('experiment_status.json'):
        print("❌ No experiment is currently running.")
        print("   Run 'python run_experiment_with_progress.py' to start.")
        return
    
    # Load status
    with open('experiment_status.json', 'r') as f:
        status = json.load(f)
    
    # Display header
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT PROGRESS REPORT")
    print("=" * 60)
    
    # Basic info
    print(f"🕐 Started: {status['start_time']}")
    print(f"⏱️  Elapsed: {status['elapsed_time']}")
    print(f"📈 Overall Progress: {status['overall_progress']}")
    print(f"🔄 Status: {'Running' if status['is_running'] else 'Completed'}")
    print("\n" + "-" * 60)
    
    # Step details
    print("STEP PROGRESS:")
    for step in status['steps']:
        # Status icon
        if step['status'] == 'completed':
            icon = "✅"
        elif step['status'] == 'running':
            icon = "🔄"
        elif step['status'] == 'failed':
            icon = "❌"
        else:
            icon = "⏳"
        
        # Progress bar
        progress = int(step['progress'])
        bar_length = 30
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Step name formatting
        step_name = step['name'].replace('_', ' ').title()
        
        print(f"{icon} {step_name:<20} [{bar}] {progress:3d}%")
        
        # Show details for current step
        if step['status'] == 'running' and 'details' in step:
            print(f"   └─ {step['details']}")
    
    print("-" * 60)
    
    # Recent log entries
    if os.path.exists('experiment_progress.log'):
        print("\nRECENT ACTIVITY:")
        with open('experiment_progress.log', 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-5:] if len(lines) >= 5 else lines
            for line in recent_lines:
                print(f"  {line.strip()}")
    
    print("=" * 60 + "\n")
    
    # Next update hint
    if status['is_running']:
        print("💡 Run this script again to check updated progress.")
    else:
        print("✨ Experiment completed! Check output/paper.tex for results.")

if __name__ == "__main__":
    check_progress()