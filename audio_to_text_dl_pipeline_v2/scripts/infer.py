from inference.infer import transcribe_audio

if __name__ == "__main__":
    audio_path = "sample.wav"
    print(transcribe_audio(audio_path))
