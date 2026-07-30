#!/usr/bin/env python3
"""One-time OuteTTS speaker profile creation. Requires the optional outetts package."""
import argparse

p = argparse.ArgumentParser(description="Create an OuteTTS 0.3 speaker JSON from reference audio.")
p.add_argument("audio")
p.add_argument("output")
p.add_argument("--transcript", help="exact reference transcript; avoids Whisper transcription errors")
p.add_argument("--whisper-model", default="turbo")
p.add_argument("--whisper-device", default="cuda")
a = p.parse_args()

try:
    import outetts
except ImportError:
    raise SystemExit("Install the optional voice-cloning package first:  python -m pip install -U outetts")

interface = outetts.Interface(outetts.ModelConfig.auto_config(
    model=outetts.Models.VERSION_0_3_SIZE_500M,
    backend=outetts.Backend.LLAMACPP,
    quantization=outetts.LlamaCppQuantization.Q4_0,
))
kwargs = {"audio_path": a.audio}
if a.transcript:
    kwargs.update(transcript=a.transcript, whisper_model=a.whisper_model, whisper_device=a.whisper_device)
speaker = interface.create_speaker(**kwargs)
interface.save_speaker(speaker, a.output)
print(f"Wrote {a.output}")
