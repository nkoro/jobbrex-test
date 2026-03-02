import os
import subprocess

def reset_lab():
    print("Stopping any existing n8n instances on port 5678...")
    subprocess.run("fuser -k 5678/tcp", shell=True, stderr=subprocess.DEVNULL)
    
    print("Starting n8n with Execute Command node unlocked...")
    env = os.environ.copy()
    env["N8N_BLOCK_NODES"] = "none"
    
    # The fix: Redirect stdin/out/err to files so it doesn't crash the terminal
    with open("n8n_output.log", "w") as log_file:
        subprocess.Popen(
            ["n8n", "start"], 
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True
        )
    
    print("\nSuccess! Refresh Firefox at http://localhost:5678 in 5 seconds.")
    print("Logs are being written to ~/evalops-lab/n8n_output.log")

if __name__ == "__main__":
    reset_lab()
