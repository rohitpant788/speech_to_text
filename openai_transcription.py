import streamlit as st
import openai
import os

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Default output directory
DEFAULT_OUTPUT_DIR = os.getcwd()

# Function to transcribe audio using OpenAI API
def transcribe_audio_openai(file_path):
    with open(file_path, "rb") as audio_file:
        st.spinner("Uploading and transcribing...")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

# Streamlit App
def main():
    st.title("Audio Transcription App with OpenAI Whisper API")
    st.write("This app allows you to transcribe audio files using OpenAI's Whisper API.")

    # File upload
    uploaded_file = st.file_uploader("Upload your audio file", type=["m4a", "mp3", "wav"])

    # Output directory
    st.subheader("Output Directory")
    output_dir = st.text_input("Specify the directory to save the transcription:", DEFAULT_OUTPUT_DIR)

    # Transcription and saving
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mpeg")

        # Perform transcription
        if st.button("Transcribe and Save"):
            # Save uploaded file to a temporary location
            temp_audio_path = os.path.join(output_dir, uploaded_file.name)
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # Transcribe audio using OpenAI Whisper API
            transcription = transcribe_audio_openai(temp_audio_path)

            # Save transcription to a file
            text_file_path = os.path.join(output_dir, "transcription.txt")
            with open(text_file_path, "w") as f:
                f.write(transcription)

            st.success(f"Transcription completed and saved to: {text_file_path}")

            # Provide a download link for the transcription
            with open(text_file_path, "r") as f:
                st.download_button(
                    label="Download Transcription",
                    data=f.read(),
                    file_name="transcription.txt",
                    mime="text/plain",
                )

if __name__ == "__main__":
    main()
