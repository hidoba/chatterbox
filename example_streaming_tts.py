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

    print("\n=== Streaming Mode ===")
    # Streaming: generates audio in chunks
    # Set stream_tokens_per_slice to enable streaming
    # Lower values = faster initial response, more chunks
    # Higher values = fewer chunks, longer initial delay
    import time

    audio_chunks = []
    start_time = time.time()
    last_chunk_time = start_time
    first_chunk_latency = None

    for i, audio_chunk in enumerate(tts.generate(
        text=text,
        temperature=0.8,
        cfg_weight=0.5,
        stream_tokens_per_slice=100,
        stream_remove_milliseconds_end=45,  # Trim 45ms from end of each chunk
        stream_remove_milliseconds_start=25,  # Trim 25ms from start of each chunk
    )):
        current_time = time.time()

        if i == 0:
            first_chunk_latency = current_time - start_time
            print(f"⏱️  Time to first chunk: {first_chunk_latency:.3f}s")

        inter_chunk_time = current_time - last_chunk_time
        audio_duration = audio_chunk.shape[1] / tts.sr

        print(f"Received chunk {i+1}: {audio_chunk.shape} | "
              f"Inter-chunk: {inter_chunk_time:.3f}s | "
              f"Audio duration: {audio_duration:.3f}s")

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

    print("\n=== Streaming with Different Parameters ===")
    # Try different chunk sizes
    for chunk_size in [100, 300, 500]:
        print(f"\nTesting with chunk_size={chunk_size}")
        chunk_count = 0
        start = time.time()
        first_chunk = None

        for i, _ in enumerate(tts.generate(
            text=text,
            stream_tokens_per_slice=chunk_size,
        )):
            if i == 0:
                first_chunk = time.time() - start
            chunk_count += 1

        total = time.time() - start
        print(f"  Generated {chunk_count} chunks | "
              f"First chunk: {first_chunk:.3f}s | "
              f"Total: {total:.3f}s")

if __name__ == "__main__":
    main()
