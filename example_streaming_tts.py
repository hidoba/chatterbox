"""
Example of using ChatterboxTTS with streaming support.

This example demonstrates both non-streaming and streaming modes.
"""

import torch
import torchaudio
from chatterbox import ChatterboxTTS

def main():
    # Initialize the TTS model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tts = ChatterboxTTS.from_pretrained(device=device)

    # Example text
    text = "Hello! This is a demonstration of streaming text to speech synthesis. The audio will be generated in chunks, allowing for real-time playback."

    print("\n=== Non-Streaming Mode (Original) ===")
    # Non-streaming: generates complete audio at once
    audio = tts.generate(
        text=text,
        temperature=0.8,
        cfg_weight=0.5,
    )
    torchaudio.save("output_non_streaming.wav", audio, tts.sr)
    print(f"Saved complete audio to output_non_streaming.wav")

    print("\n=== Streaming Mode with Progressive Chunk Sizes ===")
    # Streaming: generates audio in chunks
    # You can use:
    # - Fixed chunk size (int): stream_tokens_per_slice=200
    # - Progressive chunk sizes (list): stream_tokens_per_slice=[2, 2, 3, 4, 6, 8, 12, 20, 30, 50, 80, 120, 200]
    # Progressive sizes start small for low latency, then increase for efficiency
    import time

    # Define progressive chunk schedule
    chunk_schedule = [10,10,15,20,40,80,100]

    audio_chunks = []
    start_time = time.time()
    last_chunk_time = start_time
    first_chunk_latency = None

    for i, audio_chunk in enumerate(tts.generate(
        text=text,
        temperature=0.8,
        cfg_weight=0.5,
        stream_tokens_per_slice=chunk_schedule,  # Progressive chunk sizes
        stream_remove_milliseconds_end=35,  # original 45ms
        stream_remove_milliseconds_start=15,  # original 25ms
    )):
        current_time = time.time()

        if i == 0:
            first_chunk_latency = current_time - start_time
            print(f"⏱️  Time to first chunk: {first_chunk_latency:.3f}s")

        inter_chunk_time = current_time - last_chunk_time
        audio_duration = audio_chunk.shape[1] / tts.sr

        # Show which chunk size was used (estimate from schedule)
        chunk_size_used = chunk_schedule[i] if i < len(chunk_schedule) else chunk_schedule[-1]

        print(f"Chunk {i+1} (target: {chunk_size_used} tokens): {audio_chunk.shape} | "
              f"Inter-chunk: {inter_chunk_time:.3f}s | "
              f"Duration: {audio_duration:.3f}s")

        audio_chunks.append(audio_chunk)
        last_chunk_time = current_time

        # In a real application, you could play this chunk immediately
        # while waiting for the next one
        torchaudio.save(f"output_chunk_{i+1}.wav", audio_chunk, tts.sr)

    total_time = time.time() - start_time
    total_audio_duration = sum(chunk.shape[1] for chunk in audio_chunks) / tts.sr

    # Concatenate all chunks to verify completeness
    full_audio = torch.cat(audio_chunks, dim=1)
    torchaudio.save("output_streaming_combined.wav", full_audio, tts.sr)

    print(f"\n📊 Streaming Statistics:")
    print(f"   Total chunks: {len(audio_chunks)}")
    print(f"   Time to first chunk: {first_chunk_latency:.3f}s")
    print(f"   Total generation time: {total_time:.3f}s")
    print(f"   Total audio duration: {total_audio_duration:.3f}s")
    print(f"   Real-time factor: {total_audio_duration / total_time:.2f}x")

if __name__ == "__main__":
    main()
