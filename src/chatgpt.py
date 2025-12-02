import os
import time
import sys
from openai import OpenAI
from dotenv import load_dotenv
from prompts import PROMPT_GENERATE_MDT, PROMPT_FACTUAL_CORRECTNESS, PROMPT_PLAUSIBILITY

# Load environment variables
load_dotenv()

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def upload_file(file_path):
    """Uploads a file to OpenAI for use with assistants."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    print(f"Uploading file: {file_path}...")
    file = client.files.create(file=open(file_path, "rb"), purpose="assistants")
    print(f"File uploaded. ID: {file.id}")
    return file


def create_assistant():
    """Creates an assistant with file search capabilities."""
    print("Creating Assistant...")
    assistant = client.beta.assistants.create(
        name="MDT Generator & Evaluator",
        instructions="You are a helpful medical AI assistant capable of analyzing case studies and generating transcripts.",
        model="gpt-4o",
        tools=[{"type": "file_search"}],
    )
    print(f"Assistant created. ID: {assistant.id}")
    return assistant


def run_thread(assistant_id, thread_id):
    """Runs the thread and waits for completion."""
    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id
    )

    print(f"Run started (ID: {run.id}). Waiting for completion...")
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run.id
        )
        if run_status.status == "completed":
            print("Run completed.")
            break
        elif run_status.status in ["failed", "cancelled", "expired"]:
            print(f"Run failed with status: {run_status.status}")
            sys.exit(1)
        time.sleep(2)

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    # Return the latest message content
    return messages.data[0].content[0].text.value


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/chatgpt.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # 1. Upload File
    file_obj = upload_file(pdf_path)

    # 2. Create Assistant
    assistant = create_assistant()

    # 3. Create Thread
    print("Creating Thread...")
    thread = client.beta.threads.create()

    try:
        # --- TASK 1: Generate MDT Transcript ---
        print("\n--- Task 1: Generating MDT Transcript ---")

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=PROMPT_GENERATE_MDT,
            attachments=[{"file_id": file_obj.id, "tools": [{"type": "file_search"}]}],
        )

        transcript = run_thread(assistant.id, thread.id)
        print("\n[GENERATED TRANSCRIPT]\n")
        print(transcript)
        print("\n" + "=" * 50 + "\n")

        # --- TASK 2: Factual Correctness ---
        print("\n--- Task 2: Evaluating Factual Correctness ---")

        formatted_prompt_factual = PROMPT_FACTUAL_CORRECTNESS.format(
            transcript=transcript
        )

        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=formatted_prompt_factual
        )

        factual_feedback = run_thread(assistant.id, thread.id)
        print("\n[FACTUAL CORRECTNESS FEEDBACK]\n")
        print(factual_feedback)
        print("\n" + "=" * 50 + "\n")

        # --- TASK 3: Plausibility ---
        print("\n--- Task 3: Evaluating Plausibility ---")

        formatted_prompt_plausibility = PROMPT_PLAUSIBILITY.format(
            transcript=transcript
        )

        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=formatted_prompt_plausibility
        )

        plausibility_feedback = run_thread(assistant.id, thread.id)
        print("\n[PLAUSIBILITY FEEDBACK]\n")
        print(plausibility_feedback)
        print("\n" + "=" * 50 + "\n")

    finally:
        # Clean up the created file and assistant
        print("Cleaning up resources...")
        client.files.delete(file_obj.id)
        client.beta.assistants.delete(assistant.id)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
