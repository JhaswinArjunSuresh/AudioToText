import torchaudio
from models.simple_asr import SimpleASRModel

def transcribe_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    features = torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0).transpose(0, 1)
    model = SimpleASRModel(128, 100)
    model.load_state_dict(torch.load("results/best_model.pth"))
    model.eval()
    with torch.no_grad():
        logits = model(features.unsqueeze(0))
        prediction = logits.argmax(dim=-1).squeeze().tolist()
    return prediction
