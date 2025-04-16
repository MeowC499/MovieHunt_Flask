import os
import gdown

def download_data():
    os.makedirs("data", exist_ok=True)

    files = {
        "ncf_model_state.pth": "1-5JsDtm_EF3qwvTopoWDEUu17FVfYEaV",
        "ratings.csv": "1Xuh5ZuI2RnBZ2f-nuKAkrI29dwaRzSXP",
        "movies.csv": "1P5p6bpyVA_uTGGeYq5PiK-MwIUZ5sIi1",
        "links.csv": "1AtRiQ0-X5KuZFjnPWcPj1kpqDe3KZUWq",
        "genome-tags.csv": "1ijNcsOK2b0Yenl2q8AgGYJwqLAbbtBEb",
        "genome-scores.csv": "1VTdzDeknOLCuO6CQq0QyJYjZvFj8SG7r",
        "tags.csv": "1F56JAP1jC5Ia1-3eB1tJajpFOiuY0Gfb"
    }

    for filename, file_id in files.items():
        output_path = os.path.join("data", filename)
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"⬇️  Downloading {filename}...")
            try:
                gdown.download(url, output_path, quiet=False)
                print(f"✅ Successfully downloaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"📦 {filename} already exists. Skipping.")

if __name__ == "__main__":
    download_data()
