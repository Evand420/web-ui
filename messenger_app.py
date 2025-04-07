import openai
import gradio as gr
from gtts import gTTS
import tempfile
import speech_recognition as sr
import cv2
import os

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Function to process user input and generate AI response
def chat_with_ai(user_input=None, audio_input=None, video_input=None):
    try:
        # If audio input is provided, convert it to text
        if audio_input:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_input) as source:
                audio_data = recognizer.record(source)
                user_input = recognizer.recognize_google(audio_data)

        # Generate AI response using OpenAI GPT
        if user_input:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            ai_response = response['choices'][0]['message']['content']
        else:
            ai_response = "Please provide a text or audio input."

        # Convert AI response to audio
        tts = gTTS(ai_response)
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_file.name)

        # Handle video input (if provided)
        if video_input:
            cap = cv2.VideoCapture(video_input)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Display the video frame (optional)
                cv2.imshow("Video Input", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        return ai_response, audio_file.name
    except Exception as e:
        return f"Error: {str(e)}", None

# Gradio UI
with gr.Blocks() as messenger_app:
    gr.Markdown("## 🗨️ AI Messenger - Chat with AI via Text, Audio, or Video")

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...")
            audio_input = gr.Audio(source="microphone", type="filepath", label="Or Speak Your Message")
            video_input = gr.Video(source="webcam", label="Or Record a Video")
            send_button = gr.Button("Send")

        with gr.Column():
            ai_response = gr.Textbox(label="AI Response", interactive=False)
            audio_output = gr.Audio(label="AI Response (Audio)", interactive=False)

    send_button.click(
        fn=chat_with_ai,
        inputs=[user_input, audio_input, video_input],
        outputs=[ai_response, audio_output]
    )

# Launch the app
if __name__ == "__main__":
    messenger_app.launch()