import sys
import subprocess
import time
import os

def run_step(script_name):
    """Run a python script and check for errors"""
    print(f"\n{'='*50}")
    print(f"▶️ RUNNING: {script_name}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    # Use the current python interpreter
    python_exe = sys.executable
    
    # Run the script from the root directory so imports work correctly
    # We execute it as a module or file path relative to root
    script_path = f"src/pipeline/{script_name}"
    
    # We need to set PYTHONPATH to include src so imports work
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"src{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = "src"
        
    # We run the script directly as a file, imports inside step scripts handle path
    result = subprocess.run(
        [python_exe, script_path], 
        cwd=".",  # run from project root
        env=env
    )
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {script_name} (Exit code: {result.returncode})")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ COMPLETED: {script_name} in {duration:.2f}s")

def main():
    """Orchestrate the ML Pipeline"""
    print("\n🤖 STARTING SMART RESALE ML PIPELINE")
    
    steps = [
        "step1_data_prep.py",
        "step2_features.py", 
        "step3_training.py",
        "step4_evaluation.py"
    ]
    
    for step in steps:
        run_step(step)
    
    print("\n" + "🎉"*20)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉"*20)

if __name__ == "__main__":
    main()
