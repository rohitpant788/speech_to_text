import streamlit as st
import whisper
import os
import time

# Set the new cache directory for Whisper models
WHISPER_MODELS_DIR = r"D:\26. my_models\whisper"
os.environ["WHISPER_MODELS_DIR"] = WHISPER_MODELS_DIR
st.info(f"Whisper models will be loaded from: {WHISPER_MODELS_DIR}")

# Default output directory
DEFAULT_OUTPUT_DIR = os.getcwd()


# Function to ensure the model is downloaded
def ensure_model_exists(model_name):
    # Construct the path where the model should exist
    model_path = os.path.join(WHISPER_MODELS_DIR, model_name)
    st.write(f"Checking for model at: {model_path}")

    # Check if the model directory exists
    if not os.path.exists(model_path):
        st.warning(f"The model '{model_name}' is not available locally in {WHISPER_MODELS_DIR}. Downloading now...")
        whisper.load_model(model_name)  # Downloads and caches the model
        st.success(f"The model '{model_name}' has been downloaded and cached at {WHISPER_MODELS_DIR}.")
    else:
        st.info(f"The model '{model_name}' is already available at {model_path}.")


# Function to transcribe audio
def transcribe_audio(file_path, model_name):
    # Load the Whisper model dynamically based on user selection
    model = whisper.load_model(model_name)
    with st.spinner("Transcribing... Please wait..."):
        result = model.transcribe(file_path)
        return result["text"]


# Streamlit App
def main():
    st.title("Audio Transcription App")
    st.write("This app allows you to transcribe audio files using OpenAI's Whisper.")

    # Model selection
    st.subheader("Choose Whisper Model")
    model_name = st.selectbox(
        "Select a Whisper model:", ["tiny", "base", "small", "medium", "large"]
    )
    st.info(f"You have selected the '{model_name}' model.")

    # Ensure the model exists locally
    ensure_model_exists(model_name)

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

            # Start the timer
            start_time = time.time()

            # Transcribe audio
            transcription = transcribe_audio(temp_audio_path, model_name)

            # End the timer
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            # Save transcription to a file
            text_file_path = os.path.join(output_dir, "transcription.txt")
            with open(text_file_path, "w") as f:
                f.write(transcription)

            st.success(
                f"Transcription completed and saved to: {text_file_path}\n"
                f"Completed in {int(minutes)} minutes and {int(seconds)} seconds."
            )

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
