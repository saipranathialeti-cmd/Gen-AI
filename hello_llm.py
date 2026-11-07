
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load .env file
env_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=env_path)

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not env_path.exists():
    print(".env file not found!")

elif not api_key:
    print("API key not found. Check your .env file.")

else:
    client = genai.Client(api_key=api_key)

    print("Connected to Gemini successfully!")

    # Try up to 3 times
    for attempt in range(1, 4):
        try:
            print(f"\nAttempt {attempt}...")

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents="Hello Gemini! Introduce yourself."
            )

            print("\nGemini Response:")
            print(response.text)
            break

        except Exception as e:
            error = str(e)

            if "503" in error and attempt < 3:
                wait_time = 2 ** attempt
                print("Gemini server is busy.")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("\nRequest failed:", e)
                break
