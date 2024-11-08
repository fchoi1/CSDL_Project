# multi_speaker_tts.py

import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts
import torch
import os
from torch.utils.data import Dataset, DataLoader


# Step 1: Setup and Install NeMo (run this command in the terminal, not in the script)
# pip install nemo_toolkit[all]


# Step 2: Load Pre-trained Speaker Verification Model to Extract Embeddings
def extract_speaker_embedding(audio_file_path):
    # Load a pre-trained speaker verification model from NVIDIA NeMo
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="titanet_large"
    )

    # Extract speaker embedding for the input audio file
    embedding = speaker_model.get_embedding(audio_file_path)

    return embedding


# Step 3: Load a Pre-trained TTS Model and Configure for Speaker Embeddings
def load_tts_model():
    # Load a pre-trained FastPitch model
    tts_model = nemo_tts.models.FastPitchModel.from_pretrained(
        model_name="tts_en_fastpitch"
    )

    return tts_model


# Step 4: Dataset Class for Loading Text, Audio, and Speaker Embeddings
class SpeakerDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, audio_file, speaker_audio_file = self.dataset[idx]
        speaker_embedding = extract_speaker_embedding(speaker_audio_file)

        return {
            "text": text,
            "audio": audio_file,
            "speaker_embedding": speaker_embedding
        }


# Step 5: Training Function
def train_tts_model_with_speaker_embeddings(tts_model, dataset, batch_size=1):
    # Create a DataLoader for batching the dataset
    data_loader = DataLoader(SpeakerDataset(dataset), batch_size=batch_size, shuffle=True)

    # Set the model to training mode
    tts_model.train()

    # Loop through the dataset
    for batch in data_loader:
        text = batch["text"]
        audio = batch["audio"]
        speaker_embedding = batch["speaker_embedding"]

        # Perform the training step
        # Note: training_step may require a batch_idx argument
        # In practice, we would need a more complex setup for inputting audio and text
        # This is a simplified example
        loss = tts_model.training_step(batch, 0)

        # Backpropagate the loss and update the model weights
        loss.backward()

        # Define the optimizer and perform an optimization step
        optimizer = torch.optim.Adam(tts_model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()

    print("Training completed.")
    return tts_model


# Step 6: Synthesize Speech with Speaker Embeddings
def synthesize_speech(tts_model, text, speaker_audio_file):
    # Extract speaker embedding for the target speaker
    speaker_embedding = extract_speaker_embedding(speaker_audio_file)

    # Generate speech with the provided text and speaker embedding
    with torch.no_grad():
        generated_audio = tts_model.generate_speech(text, speaker_embedding)

    print("Synthesis completed.")
    return generated_audio


# Main function to demonstrate the workflow
def main():
    # Step 1: Load the TTS model
    tts_model = load_tts_model()

    # Example dataset format - list of tuples (text, audio_file, speaker_audio_file)
    dataset = [
        (
            "GO DO YOU HEAR",
            "TestAudio/one.m4a",
            "LibriSpeech/dev-clean/84/121123/84-121123-0000.flac",
        ),
        (
            "BUT IN LESS THAN FIVE MINUTES THE STAIRCASE GROANED BENEATH AN EXTRAORDINARY WEIGHT",
            "path/to/audio2.wav",
            "LibriSpeech/dev-clean/84/121123/84-121123-0001.flac",
        ),
        # Add more samples as needed
    ]

    # Step 2: Fine-tune the model with the multi-speaker dataset
    trained_tts_model = train_tts_model_with_speaker_embeddings(tts_model, dataset)

    # Step 3: Synthesize speech with a specific speaker embedding
    generated_audio = synthesize_speech(
        trained_tts_model, "This is a test sentence.", "path/to/speaker1_audio.wav"
    )

    # Save generated audio to file
    generated_audio_file = "generated_speech.wav"
    with open(generated_audio_file, 'wb') as f:
        f.write(generated_audio)
    
    print(f"Generated audio saved to {generated_audio_file}")


if __name__ == "__main__":
    main()
