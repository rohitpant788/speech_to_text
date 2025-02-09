import streamlit as st
import whisper
import os
import time
import pandas as pd

# Set directories
ROOT_DIR = os.getcwd()
WHISPER_MODELS_DIR = os.path.join(ROOT_DIR, "models")  # Folder for Whisper models
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")  # Folder for uploaded audio files
TRANSCRIPT_DIR = os.path.join(ROOT_DIR, "transcripts")  # Folder for transcriptions

# Ensure directories exist
os.makedirs(WHISPER_MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# Set Whisper environment variable
os.environ["WHISPER_MODELS_DIR"] = WHISPER_MODELS_DIR
st.info(f"Whisper models will be loaded from: {WHISPER_MODELS_DIR}")


# Function to check if a model exists or download it
def ensure_model_exists(model_name):
    model_path = os.path.join(WHISPER_MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        st.warning(f"The model '{model_name}' is not available locally. Downloading now...")
        whisper.load_model(model_name)  # Downloads and caches the model
        st.success(f"The model '{model_name}' has been downloaded to: {WHISPER_MODELS_DIR}")
    else:
        st.info(f"The model '{model_name}' is already available at: {model_path}")


# Function to transcribe audio
def transcribe_audio(file_path, model_name):
    model = whisper.load_model(model_name)
    with st.spinner("Transcribing... Please wait..."):
        result = model.transcribe(file_path)
        return result["text"]


# Function to list uploaded files
def list_uploaded_files():
    files = os.listdir(UPLOAD_DIR)
    return [file for file in files if file.endswith((".m4a", ".mp3", ".wav"))]


# Function to list transcriptions with metadata
def list_transcriptions():
    transcriptions = []
    for file in os.listdir(TRANSCRIPT_DIR):
        if file.endswith(".txt"):
            file_parts = file.replace(".txt", "").split("_")
            if len(file_parts) >= 2:
                model_name = file_parts[-1]  # Last part is the model name
                original_file = "_".join(file_parts[:-1])  # Rest is the original filename
                transcriptions.append({"Audio File": original_file, "Model": model_name, "Transcript File": file})
    return transcriptions


# Function to delete a file (uploads or transcripts)
def delete_file(file_name, directory):
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        st.warning(f"Deleted file: {file_name}")


# Streamlit App
def main():
    st.title("Audio Transcription App")
    st.write("This app allows you to transcribe audio files using OpenAI's Whisper.")

    # Model selection
    st.subheader("Choose Whisper Model")
    model_name = st.selectbox("Select a Whisper model:", ["tiny", "base", "small", "medium", "large"])
    st.info(f"You have selected the '{model_name}' model.")

    # Ensure the model exists
    ensure_model_exists(model_name)

    # File upload
    uploaded_file = st.file_uploader("Upload your audio file", type=["m4a", "mp3", "wav"])

    # If a file is uploaded, save it to the uploads folder
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.audio(file_path, format="audio/mpeg")

    # Display uploaded files
    st.subheader("Uploaded Files")
    uploaded_files = list_uploaded_files()

    if uploaded_files:
        for file_name in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, file_name)

            col1, col2 = st.columns([0.8, 0.2])
            col1.write(f"📁 {file_name}")
            if col2.button("❌ Delete", key=f"delete_{file_name}"):
                delete_file(file_name, UPLOAD_DIR)
                st.experimental_rerun()  # Refresh the app to update file list

    # Transcription section
    st.subheader("Transcription")
    if uploaded_files:
        selected_file = st.selectbox("Select a file to transcribe:", uploaded_files)

        if st.button("Transcribe and Save"):
            file_path = os.path.join(UPLOAD_DIR, selected_file)

            # Start the timer
            start_time = time.time()

            # Transcribe audio
            transcription = transcribe_audio(file_path, model_name)

            # End the timer
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            # Save transcription to a file
            transcript_filename = f"{selected_file}_{model_name}.txt"
            transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)
            with open(transcript_path, "w") as f:
                f.write(transcription)

            st.success(
                f"Transcription completed and saved to: {transcript_path}\n"
                f"Completed in {int(minutes)} minutes and {int(seconds)} seconds."
            )

            # Provide a download link for the transcription
            with open(transcript_path, "r") as f:
                st.download_button(
                    label="Download Transcription",
                    data=f.read(),
                    file_name=transcript_filename,
                    mime="text/plain",
                )

    # Display transcriptions table
    st.subheader("Saved Transcriptions")
    transcription_records = list_transcriptions()

    if transcription_records:
        df = pd.DataFrame(transcription_records)
        st.dataframe(df)

        # Option to delete transcripts
        selected_transcript = st.selectbox("Select a transcription to delete:", df["Transcript File"].tolist())
        if st.button("Delete Selected Transcription"):
            delete_file(selected_transcript, TRANSCRIPT_DIR)
            st.experimental_rerun()  # Refresh the app to update list


if __name__ == "__main__":
    main()
