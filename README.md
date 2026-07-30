# srt2speech

srt2speech is an open-source project by Waversense from r/LocalTextToSpeech community. It is a lightweight SRT-to-speech converter with speech-duration adaptation support. On GPU up to 15x realtime performance, multi-speaker templating support. It uses llama.cpp as its engine, providing fast local performance across most platforms and GPUs.

## Install

1. Download a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases)
   with support for your GPU or CPU.

   For CUDA, unpack the matching CUDA DLL package into the llama.cpp directory,
   for example `llama-b10182-bin-win-cuda-13.3-x64`.

2. Download the OuteTTS 0.3 models:

   - [OuteTTS-0.3-500M-Q4_0.gguf](https://huggingface.co/OuteAI/OuteTTS-0.3-500M-GGUF/resolve/main/OuteTTS-0.3-500M-Q4_0.gguf?download=true)
   - [WavTokenizer-Large-75-Q5_1.gguf](https://huggingface.co/ggml-org/WavTokenizer/resolve/main/WavTokenizer-Large-75-Q5_1.gguf?download=true)
   You may choose a higher quality quantization from https://huggingface.co/OuteAI/OuteTTS-0.3-500M-GGUF
3. Install Python if needed, then install the only dependency:

   ```powershell
   winget install -e --id Python.Python.3.11 --scope machine
   python -m pip install numpy
   ```

4. [Download srt2speech.py](https://github.com/Waversense/srt2speech/blob/master/srt2speech.py)
   or clone the repository. You can use
   [demo.srt](https://github.com/Waversense/srt2speech/blob/master/demo.srt)
   for testing.

5. You are ready to create speech from an SRT file.

## Usage

Create one WAV:

```powershell
python .\srt2speech.py .\demo.srt .\output.wav `
  --server C:\llama.cpp\llama-server.exe `
  --model C:\models\OuteTTS-0.3-500M-Q4_0.gguf `
  --vocoder C:\models\WavTokenizer-Large-75-Q5_1.gguf
```

Create separate cue and silence WAVs in `output-audio`:

```powershell
python .\srt2speech.py .\demo.srt .\output-audio --split `
  --server C:\llama.cpp\llama-server.exe `
  --model C:\models\OuteTTS-0.3-500M-Q4_0.gguf `
  --vocoder C:\models\WavTokenizer-Large-75-Q5_1.gguf
```

Optionally switch speakers at the start of an SRT cue. The selection remains
active until the next switch:

```srt
{{en_male_1}} Spoken by the male voice.
{{en_female_1}} Spoken by the female voice.
```

Run `python .\srt2speech.py -h` for all options.

You can use the `clone_voice.py` script to create custom cloned voices, it will need outetts as python package.


## Features and background
- pitch-corrected speed adjustment
- automatic regeneration
- modification of pauses between words
- exact placement of silence between subtitle cues

- SRT does not support multiple speakers, so I also added simple templating.
Adding {{speaker_name}} to a subtitle automatically switches voices.

Of course voice cloning is supported, I added a small helper script.

- Dependencies are minimal: Python, NumPy, llama.cpp, and the required GGUF speech models. I tested it with Q4 quantization, which works well.

## Performance on my laptop:
RTX 4080 Laptop GPU: around 12–13× real time
CPU only: around 1.5–2.0× real time

## Languages supported:
- English (en)
- Japanese (jp)
- Korean (ko)
- Chinese (zh)
- French (fr)
- German (de)

It should work on almost any hardware, including old PCs, Linux or Mac.

It may be useful for anyone generating narration, translated audio tracks, accessibility audio, or quick video voiceovers.

The project is open source under the Apache 2.0 license. Attribution and license notices must be preserved.
