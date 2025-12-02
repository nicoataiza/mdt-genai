import os
import sys
import time
import magic
import mimetypes
from dotenv import load_dotenv
import google.generativeai as genai
from prompts import PROMPT_GENERATE_MDT, PROMPT_FACTUAL_CORRECTNESS, PROMPT_PLAUSIBILITY

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Choose an available model (prefer 2.0 flash, then 1.5 variants)
def pick_available_model():
    try:
        available = list(genai.list_models())
        # Keep only models that support text generation
        candidates = [
            m
            for m in available
            if hasattr(m, "supported_generation_methods")
            and "generateContent" in m.supported_generation_methods
        ]
        priority_suffixes = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        # Prefer a model whose name ends with one of the priority suffixes
        for suf in priority_suffixes:
            for m in candidates:
                if m.name.endswith(suf):
                    print(f"Using Gemini model: {m.name}")
                    return m.name
        # Fallback to the first candidate
        if candidates:
            print(f"Using fallback Gemini model: {candidates[0].name}")
            return candidates[0].name
    except Exception as e:
        print(
            f"Warning: could not list models, defaulting to gemini-1.5-flash-8b. Error: {e}"
        )
    return "gemini-1.5-flash-8b"


def upload_file(file_path):
    """Uploads a file to Google for use with Gemini."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    print(f"Uploading file: {file_path}...")
    # Use python-magic to determine the mime type
    mime_type = magic.from_file(file_path, mime=True)

    # Fallback if mime type is not detected
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(file_path)

    if not mime_type:
        print(
            f"Could not determine mime type for {file_path}. Defaulting to application/octet-stream."
        )
        mime_type = "application/octet-stream"

    uploaded_file = genai.upload_file(path=file_path, mime_type=mime_type)
    print(f"File uploaded. Name: {uploaded_file.name}")
    return uploaded_file


def get_model_response(prompt, file_obj, model_name=None):
    """Gets a response from the Gemini model."""
    print("Generating response from Gemini...")
    model_name = model_name or pick_available_model()
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content([prompt, file_obj])
    print("Response received.")
    return response.text


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/gemini.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # 1. Upload File
    file_obj = upload_file(pdf_path)

    try:
        # --- TASK 1: Generate MDT Transcript ---
        print("\n--- Task 1: Generating MDT Transcript ---")
        transcript = get_model_response(PROMPT_GENERATE_MDT, file_obj)
        print("\n[GENERATED TRANSCRIPT]\n")
        print(transcript)
        print("\n" + "=" * 50 + "\n")

        # --- TASK 2: Factual Correctness ---
        print("\n--- Task 2: Evaluating Factual Correctness ---")
        formatted_prompt_factual = PROMPT_FACTUAL_CORRECTNESS.format(
            transcript=transcript
        )
        factual_feedback = get_model_response(formatted_prompt_factual, file_obj)
        print("\n[FACTUAL CORRECTNESS FEEDBACK]\n")
        print(factual_feedback)
        print("\n" + "=" * 50 + "\n")

        # --- TASK 3: Plausibility ---
        print("\n--- Task 3: Evaluating Plausibility ---")
        formatted_prompt_plausibility = PROMPT_PLAUSIBILITY.format(
            transcript=transcript
        )
        # For plausibility, we don't need the PDF file.
        model_name = pick_available_model()
        model = genai.GenerativeModel(model_name=model_name)
        plausibility_feedback = model.generate_content(
            formatted_prompt_plausibility
        ).text
        print("\n[PLAUSIBILITY FEEDBACK]\n")
        print(plausibility_feedback)
        print("\n" + "=" * 50 + "\n")

    finally:
        # Clean up the uploaded file
        print("Cleaning up resources...")
        genai.delete_file(file_obj.name)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
