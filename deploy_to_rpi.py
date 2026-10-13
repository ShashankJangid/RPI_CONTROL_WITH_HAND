import os
import sys
import paramiko

RPI_HOSTS = ["192.168.1.111", "ailab.local", "ailab"]
USERNAME = "ailab"
PASSWORD = "ailab"
REMOTE_DIR = "/home/ailab/RPI_CONTROL_WITH_HAND"

FILES_TO_TRANSFER = [
    "config.py",
    "camera_stream.py",
    "hand_detector.py",
    "hardware_controller.py",
    "main.py",
    "test_gesture_control.py",
    "requirements.txt",
    "README.md",
]


def connect_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connected = False
    last_err = None

    for host in RPI_HOSTS:
        print(f"Attempting connection to {host}...")
        try:
            ssh.connect(
                hostname=host,
                port=22,
                username=USERNAME,
                password=PASSWORD,
                timeout=10,
                look_for_keys=False,
                allow_agent=False
            )
            print(f"Successfully connected to Raspberry Pi at {host}!")
            return ssh
        except Exception as e:
            print(f"Failed connecting to {host}: {e}")
            last_err = e

    raise RuntimeError(f"Could not connect to Raspberry Pi: {last_err}")


def deploy():
    ssh = connect_ssh()
    sftp = ssh.open_sftp()

    print(f"Creating remote directory: {REMOTE_DIR}...")
    stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {REMOTE_DIR}")
    stdout.channel.recv_exit_status()

    local_base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n--- Transferring Files ---")
    for filename in FILES_TO_TRANSFER:
        local_path = os.path.join(local_base_dir, filename)
        remote_path = f"{REMOTE_DIR}/{filename}"

        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            print(f"Uploading: {filename} ({file_size} bytes) -> {remote_path}")
            sftp.put(local_path, remote_path)
        else:
            print(f"WARNING: File not found locally: {local_path}")

    sftp.close()

    print("\n--- Verifying Remote Files ---")
    stdin, stdout, stderr = ssh.exec_command(f"ls -lh {REMOTE_DIR}")
    print(stdout.read().decode())

    ssh.close()
    print("All files transferred successfully to Raspberry Pi!")


if __name__ == "__main__":
    deploy()
