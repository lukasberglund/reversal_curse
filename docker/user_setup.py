import os
import json
import subprocess


for user in os.listdir("/root/situational-awareness/docker/users/"):
    user = user.replace(".json", "")
    print(user)
    os.system("useradd -m " + user)
    os.system(
        f"cp /root/situational-awareness/docker/default_bash_rc /home/{user}/.bashrc"
    )
    os.system(f"usermod -aG sudo {user}")
    os.system(f"mkdir /home/{user}/.ssh")
    os.system(f"usermod --shell /bin/bash {user}")
    subprocess.run(f"echo '{user}:pass' | chpasswd", shell=True, check=True)
    with open("/root/situational-awareness/docker/users/" + user + ".json", "r") as f:
        json_data = json.load(f)
        json_data["bashrc"] = json_data["bashrc"]
        with open("/home/" + user + "/.bashrc", "a+") as f:
            f.write(json_data["bashrc"])
        authorised_keys = json_data["authorized_keys"]

        for key in authorised_keys:
            with open("/home/" + user + "/.ssh/authorized_keys", "a+") as f:
                f.write(key)
