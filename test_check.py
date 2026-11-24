# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import pytesseract
# from PIL import Image
# import speech_recognition as sr

# def load_environment():
#     """Loads environment variables from a .env file."""
#     load_dotenv(override=True)
# load_environment()

# def get_google_api_key():
#     """Fetches the Google API key from environment variables."""
#     api_key = os.getenv("GOOGLE_API_KEY")
#     print(api_key)
#     print('printed api key')
#     if not api_key:
#         raise ValueError("GOOGLE_API_KEY not found in environment variables.")
#     return api_key
# get_google_api_key()



# import os

# print("--- 1. CHECKING REAL FILENAMES ---")
# # This prints the actual names of files (revealing hidden extensions)
# files = os.listdir()
# env_files = [f for f in files if ".env" in f]

# if not env_files:
#     print("❌ NO .env file found at all!")
# else:
#     for f in env_files:
#         print(f"Found file: '{f}' (Real Name)")

# print("\n--- 2. TRYING TO READ CONTENT ---")
# try:
#     # Try to open the file exactly as code expects
#     with open(".env", "r", encoding="utf-8") as f:
#         content = f.read()
#         print("✅ Success! File found. Content preview:")
#         print(f"[{content.strip()}]")
# except FileNotFoundError:
#     print("❌ Python cannot find '.env'. This confirms the file is named .env.txt or something else.")
# except Exception as e:
#     print(f"❌ Error reading file: {e}")