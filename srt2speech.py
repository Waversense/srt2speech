#!/usr/bin/env python3
"""Fast SRT to speech with persistent llama.cpp servers and OuteTTS."""

import argparse
import html
import json
import math
import os
import re
import socket
import shlex
import subprocess
import threading
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise SystemExit("srt2speech needs numpy:  python -m pip install numpy")

VERSION = "0.5.2"


TIME_RE = re.compile(
    r"^\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d+):(\d{2}):(\d{2})[,.](\d{3})"
)
SPEAKER_RE = re.compile(r"^\{\{([A-Za-z0-9][A-Za-z0-9_.-]*)\}\}\s*")


def seconds(parts):
    h, m, s, ms = map(int, parts)
    return h * 3600 + m * 60 + s + ms / 1000


def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>|\{\\[^}]+\}", "", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def read_srt(path):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = block.splitlines()
        for i, line in enumerate(lines):
            match = TIME_RE.match(line)
            if match:
                start = seconds(match.groups()[:4])
                end = seconds(match.groups()[4:])
                spoken = clean_text(" ".join(lines[i + 1 :]))
                if spoken and end > start:
                    cues.append((len(cues) + 1, start, end, spoken))
                break
    if not cues:
        raise ValueError("No valid subtitle cues found")
    return sorted(cues, key=lambda cue: cue[1])


def number_words(value):
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def integer(n):
        if n < 20:
            return ones[n]
        if n < 100:
            return tens[n // 10] + ((" " + ones[n % 10]) if n % 10 else "")
        if n < 1000:
            return ones[n // 100] + " hundred" + ((" " + integer(n % 100)) if n % 100 else "")
        for size, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
            if n >= size:
                return integer(n // size) + " " + name + ((" " + integer(n % size)) if n % size else "")
        return str(n)

    whole, dot, fraction = value.partition(".")
    result = integer(int(whole))
    if dot:
        result += " point " + " ".join(ones[int(d)] for d in fraction)
    return result


def process_text(text, version):
    text = re.sub(r"\d+(?:\.\d+)?", lambda m: number_words(m.group()), text.lower())
    text = re.sub(r"[-_/,\.\\]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "<|space|>" if version == "0.3" else "<|text_sep|>")


def load_speaker(path):
    speaker = json.loads(Path(path).read_text(encoding="utf-8"))
    version = str(speaker.get("version", "0.3"))
    if version not in ("0.2", "0.3") or not speaker.get("words"):
        raise ValueError("Speaker JSON must contain version 0.2/0.3 and a words array")
    sep = "<|space|>" if version == "0.3" else "<|text_sep|>"
    code_start = "" if version == "0.3" else "<|code_start|>"
    code_end = sep if version == "0.3" else "<|code_end|>"
    text = "<|text_start|>" + "".join(str(w["word"]) + sep for w in speaker["words"])
    audio = ["<|audio_start|>"]
    for word in speaker["words"]:
        codes = "".join(f"<|{int(code)}|>" for code in word["codes"])
        audio.append(f'{word["word"]}<|t_{float(word["duration"]):.2f}|>{code_start}{codes}{code_end}')
    return version, text, "\n".join(audio) + "\n"


def select_speakers(cues, default_path):
    """Apply persistent {{name}} speaker changes and load each profile once."""
    directory = Path(default_path).parent
    current = Path(default_path)
    loaded = {}
    selected = {}
    spoken_cues = []

    for number, start, end, text in cues:
        marker = SPEAKER_RE.match(text)
        if marker:
            current = directory / f"{marker.group(1)}.json"
            text = text[marker.end():].strip()
        if not current.exists():
            raise FileNotFoundError(f"Cue {number}: speaker not found: {current}")
        key = str(current.resolve())
        if key not in loaded:
            loaded[key] = load_speaker(current)
        if text:
            spoken_cues.append((number, start, end, text))
            selected[number] = loaded[key]

    if not spoken_cues:
        raise ValueError("No subtitle text remains after speaker markers")
    return spoken_cues, selected


def http_json(url, data=None, timeout=600):
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail[-2000:]}") from error


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_server(url, process, log_path, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            tail = Path(log_path).read_text(errors="replace")[-4000:]
            raise RuntimeError(f"llama-server stopped during startup\n{tail}")
        try:
            http_json(url + "/health", timeout=2)
            return
        except Exception:
            time.sleep(0.25)
    tail = Path(log_path).read_text(errors="replace")[-4000:]
    raise RuntimeError(f"Timed out starting {url}\n{tail}")


def split_args(text):
    return shlex.split(text, posix=True) if text else []


def start_server(exe, model, port, jobs, ctx_per_slot, args, vocoder, log_path, label):
    command = [
        exe, "-m", model, "--host", "127.0.0.1", "--port", str(port),
        "--parallel", str(jobs), "--cont-batching",
        "--ctx-size", str(ctx_per_slot * jobs), "--cache-ram", "0", "--no-cache-idle-slots",
        "--log-colors", "off", "--verbosity", "2" if args.verbose else "1",
    ]
    if vocoder:
        command += ["--embeddings", "--pooling", "none", "--embd-normalize", "-1",
                    "--batch-size", str(ctx_per_slot),
                    "--ubatch-size", str(ctx_per_slot)]
    if args.threads:
        command += ["--threads", str(args.threads), "--threads-batch", str(args.threads)]
    if args.cpu:
        command += ["--device", "none"]
    command += split_args(args.llama_args)
    command += split_args(args.vocoder_args if vocoder else args.llm_args)
    command += args.server_args

    log = open(log_path, "w", encoding="utf-8", errors="replace")
    # Keep the child attached to the current console in verbose mode so Windows
    # cannot suppress its live output. Quiet mode hides the server windows.
    flags = 0 if args.verbose else (subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if args.verbose:
        print(f"[{label}] command: {subprocess.list2cmdline(command)}", flush=True)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=flags,
        )

        def copy_output():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{label}] {line}", end="", flush=True)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        process._srt2speech_reader = reader
    else:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        process._srt2speech_reader = None
    process._srt2speech_log = log
    return process, command


def stop_server(process):
    if not process:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process._srt2speech_reader:
        process._srt2speech_reader.join(timeout=2)
    process._srt2speech_log.close()


def embeddings_to_audio(embedding):
    embd = np.asarray(embedding, dtype=np.float32)
    if embd.ndim != 2 or embd.shape[1] < 2:
        raise ValueError(f"Unexpected vocoder embedding shape: {embd.shape}")
    n_codes, n_embd = embd.shape
    half = n_embd // 2
    magnitude = np.clip(np.exp(embd[:, :half]), 0, 100)
    spectrum = magnitude * np.exp(1j * embd[:, half:half * 2])
    frames = np.fft.irfft(spectrum, n=1280, axis=1)
    hann = np.hanning(1281)[:-1]
    frames *= hann

    out_length = (n_codes - 1) * 320 + 1280
    audio = np.zeros(out_length, dtype=np.float64)
    envelope = np.zeros(out_length, dtype=np.float64)
    hann2 = hann * hann
    for i, frame in enumerate(frames):
        start = i * 320
        audio[start:start + 1280] += frame
        envelope[start:start + 1280] += hann2
    audio = audio[480:-480]
    envelope = envelope[480:-480]
    valid = envelope > 1e-10
    audio[valid] /= envelope[valid]
    audio[:min(6000, len(audio))] = 0
    return np.clip(audio * 32767, -32768, 32767).astype("<i2")


def trim_silence(samples, rate=24000, threshold=180, pad_ms=35):
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > threshold)
    if not len(loud):
        return samples
    pad = rate * pad_ms // 1000
    return samples[max(0, loud[0] - pad):min(len(samples), loud[-1] + pad + 1)]


def compact_pauses(samples, remove, rate=24000):
    """Shorten internal pauses while leaving at least 40 ms of each one."""
    if remove <= 0 or len(samples) < rate // 5:
        return samples, 0

    window = max(1, rate // 100)
    level = np.abs(samples.astype(np.int32))
    total = np.concatenate(([0], np.cumsum(level, dtype=np.int64)))
    average = (total[window:] - total[:-window]) / window
    quiet = np.zeros(len(samples), dtype=bool)
    quiet[window // 2:window // 2 + len(average)] = average < 220

    edges = np.flatnonzero(np.diff(np.pad(quiet.astype(np.int8), (1, 1))))
    minimum, keep = rate * 70 // 1000, rate * 40 // 1000
    runs = [(start, end, min(end - start - keep, (end - start) // 2))
            for start, end in zip(edges[::2], edges[1::2])
            if end - start >= minimum and start > window and end < len(samples) - window]
    capacity = sum(item[2] for item in runs)
    target = min(remove, capacity)
    if target <= 0:
        return samples, 0

    cuts = [min(cap, target * cap // capacity) for _, _, cap in runs]
    missing = target - sum(cuts)
    for index, (_, _, cap) in enumerate(runs):
        extra = min(missing, cap - cuts[index])
        cuts[index] += extra
        missing -= extra
        if not missing:
            break

    keep_samples = np.ones(len(samples), dtype=bool)
    for (start, end, _), cut in zip(runs, cuts):
        middle = (start + end) // 2
        cut_start = middle - cut // 2
        keep_samples[cut_start:cut_start + cut] = False
    return samples[keep_samples], target


def time_compress(samples, output_length):
    """Pitch-preserving WSOLA compression for mono speech."""
    if output_length >= len(samples):
        return samples.copy()
    if output_length < 2:
        return samples[:output_length].copy()

    frame = min(960, output_length, len(samples) // 2)
    frame -= frame % 2
    if frame < 32:
        # Too short to contain a useful pitch period.
        return samples[:output_length].copy()
    hop, search = frame // 2, frame // 4

    source = samples.astype(np.float64)
    result = np.zeros(output_length + frame, dtype=np.float64)
    result[:frame] = source[:frame]
    speed = len(samples) / output_length
    fade = np.arange(hop, dtype=np.float64) / hop

    output_start = hop
    while output_start < output_length:
        expected = round(output_start * speed)
        first = max(0, min(len(source) - frame, expected - search))
        last = max(first, min(len(source) - frame, expected + search))
        reference = result[output_start:output_start + hop]
        reference_norm = np.linalg.norm(reference)
        best, best_score = first, -math.inf
        for candidate in range(first, last + 1, 4):
            overlap = source[candidate:candidate + hop]
            score = np.dot(reference, overlap) / (reference_norm * np.linalg.norm(overlap) + 1e-12)
            if score > best_score:
                best, best_score = candidate, score

        available = min(frame, output_length - output_start)
        overlap = min(hop, available)
        result[output_start:output_start + overlap] = (
            result[output_start:output_start + overlap] * (1 - fade[:overlap])
            + source[best:best + overlap] * fade[:overlap]
        )
        if available > overlap:
            result[output_start + overlap:output_start + available] = source[
                best + overlap:best + available]
        output_start += hop

    return np.clip(result[:output_length], -32768, 32767).astype(np.int16)


def fit_duration(samples, wanted, max_speed, rate=24000):
    samples, pause_removed = compact_pauses(samples, max(0, len(samples) - wanted), rate)
    required_speed = max(1.0, len(samples) / wanted)
    applied_speed = min(required_speed, max_speed)
    trimmed = 0

    if len(samples) > wanted:
        fitted_length = max(wanted, round(len(samples) / applied_speed))
        samples = time_compress(samples, fitted_length)
        trimmed = max(0, len(samples) - wanted)
        samples = samples[:wanted]
    else:
        samples = samples.copy()

    fade = min(len(samples) // 2, max(1, rate // 100))
    if fade:
        ramp = np.arange(fade) / fade
        samples[:fade] = (samples[:fade] * ramp).astype(np.int16)
        samples[-fade:] = (samples[-fade:] * ramp[::-1]).astype(np.int16)
    return samples, required_speed, applied_speed, pause_removed / rate, trimmed / rate


def generate_codes(cue, attempt, args, llm_url, speakers):
    number, start, end, text = cue
    version, speaker_text, speaker_audio = speakers[number]
    prompt = "<|im_start|>\n" + speaker_text + process_text(text, version) + "<|text_end|>\n" + speaker_audio
    seed = args.seed + number * 1009 + attempt
    predict = args.predict or max(384, round((end - start) * 160 + 256))
    response = http_json(llm_url + "/completion", {
        "prompt": prompt,
        "n_predict": predict,
        "cache_prompt": True,
        "return_tokens": True,
        "samplers": ["top_k"],
        "top_k": args.top_k,
        "seed": seed,
    })
    codes = [token - 151672 for token in response.get("tokens", []) if 151672 <= token <= 155772]
    if not codes:
        raise RuntimeError(f"Cue {number}: OuteTTS produced no audio codes")
    return number, attempt, codes


def decode_codes(item, vocoder_url):
    number, attempt, codes = item
    decoded = http_json(vocoder_url + "/embeddings", {"input": codes})
    if isinstance(decoded, dict):
        decoded = decoded.get("data", decoded)
    if not (isinstance(decoded, list) and decoded and isinstance(decoded[0], dict)):
        raise RuntimeError(f"Cue {number}: unexpected vocoder response")
    samples = trim_silence(embeddings_to_audio(decoded[0]["embedding"]))
    return number, attempt, samples, len(samples) / 24000


def batch_map(function, items, jobs, *extra):
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(function, item, *extra) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def write_zeros(wav, count):
    block = b"\0\0" * 65536
    while count:
        size = min(count, 65536)
        wav.writeframesraw(block[:size * 2])
        count -= size


def write_wav(path, samples=(), frames=None):
    data = np.asarray(samples, dtype="<i2")
    if frames is not None:
        data = data[:frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        if len(data):
            wav.writeframesraw(data.tobytes())
        if frames is not None and frames > len(data):
            write_zeros(wav, frames - len(data))


def write_merged(path, cues, clips):
    pending = np.zeros(0, dtype=np.int32)
    position = 0
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        for number, start, end, _ in cues:
            start_frame = round(start * 24000)
            slot_frames = max(1, round(end * 24000) - start_frame)
            advance = max(0, start_frame - position)
            take = min(advance, len(pending))
            if take:
                wav.writeframesraw(np.clip(pending[:take], -32768, 32767).astype("<i2").tobytes())
                pending = pending[take:]
                position += take
            if position < start_frame:
                write_zeros(wav, start_frame - position)
                position = start_frame
            needed = max(slot_frames, len(clips[number]))
            if len(pending) < needed:
                pending = np.pad(pending, (0, needed - len(pending)))
            pending[:len(clips[number])] += clips[number].astype(np.int32)
        if len(pending):
            wav.writeframesraw(np.clip(pending, -32768, 32767).astype("<i2").tobytes())


def write_split(folder, cues, clips):
    if any(current[1] < previous[2] for previous, current in zip(cues, cues[1:])):
        raise ValueError("--split cannot represent overlapping cues; use merged WAV output")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    playlist = ["#EXTM3U"]
    position = 0
    item = 1
    for number, start, end, _ in cues:
        start_frame, end_frame = round(start * 24000), round(end * 24000)
        if start_frame > position:
            frames = start_frame - position
            name = f"{item:06d}-silence.wav"
            write_wav(folder / name, frames=frames)
            playlist += [f"#EXTINF:{frames / 24000:.3f},Silence", name]
            item += 1
        frames = max(1, end_frame - start_frame, len(clips[number]))
        name = f"{item:06d}-cue-{number:06d}.wav"
        write_wav(folder / name, clips[number], frames)
        playlist += [f"#EXTINF:{frames / 24000:.3f},Cue {number}", name]
        item += 1
        position = start_frame + frames
    path = folder / "playlist.m3u"
    path.write_text("\n".join(playlist) + "\n", encoding="utf-8")
    return path


def resolve_server(args, parser):
    if args.server:
        return args.server
    if args.tts:
        tts = Path(args.tts)
        name = "llama-server.exe" if os.name == "nt" else "llama-server"
        candidate = tts.with_name(name)
        if candidate.exists():
            return str(candidate)
    parser.error("use --server PATH (or --tts PATH when llama-server is beside it)")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        prog="srt2speech",
        description="Fast SRT speech with persistent, continuously batched llama.cpp servers.",
        epilog="Example diagnostics: srt2speech input.srt output.wav ... --verbose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("input", help="input .srt file")
    parser.add_argument("output", nargs="?", help="merged .wav, or output folder with --split")

    required = parser.add_argument_group("models and voice")
    required.add_argument("--server", help="path to llama-server executable")
    required.add_argument("--tts", help="compatibility: infer llama-server beside llama-tts")
    required.add_argument("--model", help="OuteTTS 0.2/0.3 GGUF")
    required.add_argument("--vocoder", help="WavTokenizer GGUF")
    required.add_argument("--speaker", default=str(here / "voices" / "en_male_1.json"),
                          help="initial speaker JSON; {{name}} in a cue selects name.json beside it")
    required.add_argument("--llm-url", help="reuse an already running OuteTTS llama-server")
    required.add_argument("--vocoder-url", help="reuse an already running WavTokenizer llama-server")

    timing = parser.add_argument_group("duration matching")
    timing.add_argument("--retries", type=int, default=1,
                        help="extra generations for cues outside --tolerance; models stay loaded")
    timing.add_argument("--tolerance", type=float, default=0.12, metavar="RATIO",
                        help="retry threshold as a fraction of the SRT slot; 0.12 means 12%%")
    timing.add_argument("--max-speed", type=float, default=1.30, metavar="RATIO",
                        help="maximum pitch-preserving speech compression; remaining overflow is tail-trimmed")
    timing.add_argument("--max-pitch-shift", type=float, default=None, metavar="SEMITONES",
                        help=argparse.SUPPRESS)

    batch = parser.add_argument_group("batching and generation")
    batch.add_argument("-j", "--jobs", type=int, default=4,
                       help="concurrent llama.cpp slots and HTTP requests")
    batch.add_argument("-t", "--threads", type=int, default=0,
                       help="CPU threads per server; 0 lets llama.cpp choose")
    batch.add_argument("--ctx-per-slot", type=int, default=4096,
                       help="LLM context tokens reserved per concurrent cue")
    batch.add_argument("--seed", type=int, default=1, help="base sampling seed")
    batch.add_argument("--top-k", type=int, default=4, help="OuteTTS top-k sampler value")
    batch.add_argument("--predict", type=int, default=0,
                       help="maximum generated tokens per cue; 0 uses a duration-based cap")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--cpu", action="store_true",
                         help="force CPU; otherwise llama.cpp automatically uses available GPU acceleration")
    runtime.add_argument("--llama-args", default="", metavar="TEXT",
                         help='extra llama-server arguments for both models, e.g. "--main-gpu 1"')
    runtime.add_argument("--llm-args", default="", metavar="TEXT",
                         help="extra llama-server arguments for the OuteTTS model only")
    runtime.add_argument("--vocoder-args", default="", metavar="TEXT",
                         help="extra llama-server arguments for WavTokenizer only")

    output_group = parser.add_argument_group("output and diagnostics")
    output_group.add_argument("--split", action="store_true",
                              help="write cue/silence WAVs and playlist.m3u instead of one WAV")
    output_group.add_argument("--limit", type=int, default=0,
                              help="process only the first N cues; 0 processes all")
    output_group.add_argument("--keep-logs", action="store_true",
                              help="copy llama-server logs beside the output")
    output_group.add_argument("-v", "--verbose", "--llama-output", dest="verbose",
                              action="store_true",
                              help="show raw live output from both llama-server processes, their commands, and per-stage details")

    # Kept for old commands. Prefer the quoted --llama-args options above.
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[], help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.jobs = max(1, args.jobs)
    args.retries = max(0, args.retries)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be a finite value of zero or greater")
    if args.max_pitch_shift is not None:
        if not math.isfinite(args.max_pitch_shift) or args.max_pitch_shift < 0:
            parser.error("--max-pitch-shift must be a finite value of zero or greater")
        args.max_speed = 2 ** (args.max_pitch_shift / 12.0)
    if not math.isfinite(args.max_speed) or args.max_speed < 1:
        parser.error("--max-speed must be a finite value of 1 or greater")

    input_path = Path(args.input)
    output = Path(args.output or (input_path.stem + "-audio" if args.split else input_path.with_suffix(".wav")))
    cues = read_srt(input_path)
    if args.limit:
        cues = cues[:args.limit]
    cues, speakers = select_speakers(cues, args.speaker)

    managed = not (args.llm_url or args.vocoder_url)
    if bool(args.llm_url) != bool(args.vocoder_url):
        parser.error("--llm-url and --vocoder-url must be used together")
    llm_process = vocoder_process = None
    work = Path(tempfile.mkdtemp(prefix="srt2speech-"))
    llm_log, vocoder_log = work / "llm-server.log", work / "vocoder-server.log"

    failed = False
    try:
        if args.verbose:
            print("[verbose] enabled: raw llama-server output will be streamed live", flush=True)
            backend = "CPU forced" if args.cpu else "llama.cpp automatic GPU/CPU offload"
            print(f"[config] version={VERSION}, backend={backend}, jobs={args.jobs}, retries={args.retries}")
            print(f"[config] tolerance={args.tolerance:g}, max-speed={args.max_speed:g}x, "
                  f"top-k={args.top_k}, predict={args.predict or 'auto'}")
            print(f"[config] speaker={args.speaker}")

        if managed:
            exe = resolve_server(args, parser)
            for name in ("server", "model", "vocoder", "speaker"):
                value = exe if name == "server" else getattr(args, name)
                if not value or not Path(value).exists():
                    parser.error(f"--{name} not found: {value}")
            llm_port, vocoder_port = free_port(), free_port()
            llm_url, vocoder_url = f"http://127.0.0.1:{llm_port}", f"http://127.0.0.1:{vocoder_port}"
            backend = "CPU" if args.cpu else "llama.cpp automatic offload"
            print(f"Loading OuteTTS once ({args.jobs} slots, {backend})...")
            llm_process, _ = start_server(exe, args.model, llm_port, args.jobs, args.ctx_per_slot,
                                          args, False, llm_log, "llm")
            wait_server(llm_url, llm_process, llm_log)
            print(f"Loading WavTokenizer once ({args.jobs} slots, {backend})...")
            vocoder_process, _ = start_server(exe, args.vocoder, vocoder_port, args.jobs, 2048,
                                              args, True, vocoder_log, "vocoder")
            wait_server(vocoder_url, vocoder_process, vocoder_log)
            print("Servers ready; submitting cues as a continuous batch.")
        else:
            llm_url, vocoder_url = args.llm_url.rstrip("/"), args.vocoder_url.rstrip("/")
            if args.verbose:
                print("[verbose] Reusing external servers; server output remains in their consoles.")

        cue_by_number = {cue[0]: cue for cue in cues}
        candidates = {cue[0]: [] for cue in cues}
        active = list(cues)
        for attempt in range(args.retries + 1):
            print(f"Generating codes: {len(active)} cue(s), batch size {args.jobs}...")
            generated = batch_map(generate_codes, active, args.jobs, attempt, args, llm_url, speakers)
            if args.verbose:
                for number, try_number, codes in sorted(generated):
                    print(f"[tts] cue {number}, attempt {try_number + 1}: {len(codes)} audio tokens")
            print(f"Decoding audio: {len(generated)} cue(s), batch size {args.jobs}...")
            decoded = batch_map(decode_codes, generated, args.jobs, vocoder_url)
            if args.verbose:
                for number, try_number, _, duration in sorted(decoded):
                    print(f"[vocoder] cue {number}, attempt {try_number + 1}: {duration:.3f}s decoded")

            for number, _, samples, duration in decoded:
                cue = cue_by_number[number]
                target = cue[2] - cue[1]
                error = abs(math.log(max(duration, 0.001) / target))
                candidates[number].append((error, duration, samples))

            if attempt == args.retries:
                break
            active = [cue for cue in active
                      if abs(candidates[cue[0]][-1][1] - (cue[2] - cue[1])) / (cue[2] - cue[1]) > args.tolerance]
            if not active:
                break
            print(f"Retrying {len(active)} duration mismatch(es)...")

        clips = {}
        for count, (number, start, end, _) in enumerate(cues, 1):
            _, duration, samples = min(candidates[number], key=lambda item: item[0])
            cue_index = count - 1
            next_start = cues[cue_index + 1][1] if cue_index + 1 < len(cues) else end
            deadline = end + max(0, next_start - end) / 2
            slot_frames = max(1, round(end * 24000) - round(start * 24000))
            available_frames = max(slot_frames, round(deadline * 24000) - round(start * 24000))
            wanted = min(max(slot_frames, len(samples)), available_frames)
            clip, required_speed, applied_speed, pause_removed, trimmed = fit_duration(
                samples, wanted, args.max_speed)
            clips[number] = clip
            status = f"speed {applied_speed:.2f}x (pitch preserved)"
            if wanted > slot_frames:
                status += f", used {(wanted - slot_frames) / 24000:.2f}s gap"
            if pause_removed:
                status += f", shortened pauses {pause_removed:.2f}s"
            if trimmed:
                status += f", WARNING trimmed {trimmed:.2f}s after speed cap"
            print(f"[{count}/{len(cues)}] cue {number}: {duration:.2f}s -> {end - start:.2f}s, "
                  f"tries {len(candidates[number])}, {status}")
            if args.verbose and required_speed > applied_speed:
                print(f"[fit] cue {number}: required {required_speed:.2f}x before pause shortening, "
                      f"capped at {applied_speed:.2f}x by --max-speed {args.max_speed:g}")

        if args.split:
            print(f"Wrote {write_split(output, cues, clips)}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            write_merged(output, cues, clips)
            print(f"Wrote {output}")
    except Exception:
        failed = True
        if llm_log.exists():
            print(f"LLM log: {llm_log}", file=sys.stderr)
        if vocoder_log.exists():
            print(f"Vocoder log: {vocoder_log}", file=sys.stderr)
        raise
    finally:
        stop_server(vocoder_process)
        stop_server(llm_process)
        if args.keep_logs:
            log_dir = output.with_suffix("").with_name(output.stem + "-logs") if not args.split else output / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            for log in (llm_log, vocoder_log):
                if log.exists():
                    (log_dir / log.name).write_bytes(log.read_bytes())
        elif not failed:
            for path in work.glob("*"):
                path.unlink(missing_ok=True)
            work.rmdir()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"srt2speech: {error}", file=sys.stderr)
        raise SystemExit(1)
