import os
import gdown

def download_data():
    os.makedirs("data", exist_ok=True)

    files = {
        "ncf_model_state.pth": "1-5JsDtm_EF3qwvTopoWDEUu17FVfYEaV",
        "ratings.csv": "1Xuh5ZuI2RnBZ2f-nuKAkrI29dwaRzSXP",
        "movies.csv": "1P5p6bpyVA_uTGGeYq5PiK-MwIUZ5sIi1",
    }

    for filename, file_id in files.items():
        output_path = os.path.join("data", filename)
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {filename} from {url}")
            gdown.download(url, output_path, quiet=False)
        else:
            print(f"{filename} already exists. Skipping download.")
