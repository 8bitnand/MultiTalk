import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr


APP_DIR = Path(__file__).resolve().parent
if APP_DIR.name == "MultiTalk" and (APP_DIR.parent / "InfiniteTalk").exists():
    ROOT = APP_DIR.parent
    MULTITALK_ROOT = APP_DIR
    INFINITETALK_ROOT = ROOT / "InfiniteTalk"
else:
    ROOT = APP_DIR
    MULTITALK_ROOT = ROOT / "MultiTalk"
    INFINITETALK_ROOT = ROOT / "InfiniteTalk"

JOBS_ROOT = APP_DIR / "super_app_jobs"
LOCAL_BIN = ROOT / ".local" / "bin"
PYTHON = sys.executable

MODELS = ["InfiniteTalk", "MultiTalk"]
TASKS = {
    "InfiniteTalk": ["SingleImageDriven", "VideoDubbing"],
    "MultiTalk": ["ImageDriven"],
}
SIZES = {
    "InfiniteTalk": ["infinitetalk-480", "infinitetalk-720"],
    "MultiTalk": ["multitalk-480", "multitalk-720"],
}
MODEL_ROOTS = {
    "InfiniteTalk": INFINITETALK_ROOT,
    "MultiTalk": MULTITALK_ROOT,
}
SCRIPT_NAMES = {
    "InfiniteTalk": "generate_infinitetalk.py",
    "MultiTalk": "generate_multitalk.py",
}
TASK_NAMES = {
    "InfiniteTalk": "infinitetalk-14B",
    "MultiTalk": "multitalk-14B",
}
DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


@dataclass
class PreparedJob:
    job_id: str
    job_dir: Path
    repo: Path
    command: list[str]
    manifest: dict[str, Any]
    output_file: Path


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{LOCAL_BIN}:{env.get('PATH', '')}"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "asset"


def _as_file_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "name", "video", "audio"):
            if value.get(key):
                return str(value[key])
    if isinstance(value, (list, tuple)) and value:
        return _as_file_path(value[0])
    return str(value)


def _copy_input(value: Any, job_dir: Path, prefix: str) -> str | None:
    path_value = _as_file_path(value)
    if not path_value:
        return None
    source = Path(path_value)
    if not source.exists():
        return path_value
    target_dir = job_dir / "inputs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{prefix}_{_clean_name(source.name)}"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return str(target)


def _job_dir() -> tuple[str, Path]:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def _split_cli_values(value: str | None) -> list[str]:
    if not value:
        return []
    return shlex.split(value)


def _strip_mp4(path: Path) -> Path:
    if path.suffix.lower() == ".mp4":
        return path.with_suffix("")
    return path


def _default_paths(model: str, speaker_mode: str) -> dict[str, str]:
    repo = MODEL_ROOTS[model]
    defaults = {
        "ckpt_dir": str(repo / "weights" / "Wan2.1-I2V-14B-480P"),
        "wav2vec_dir": str(repo / "weights" / "chinese-wav2vec2-base"),
        "quant_dir": "",
        "lora_dir": "",
        "dit_path": "",
    }
    if model == "InfiniteTalk":
        subdir = "multi" if "Multi" in speaker_mode else "single"
        defaults["infinitetalk_dir"] = str(repo / "weights" / "InfiniteTalk" / subdir / "infinitetalk.safetensors")
        defaults["voice_1"] = str(repo / "weights" / "Kokoro-82M" / "voices" / "am_adam.pt")
        defaults["voice_2"] = str(repo / "weights" / "Kokoro-82M" / "voices" / "af_heart.pt")
    else:
        defaults["infinitetalk_dir"] = ""
        defaults["voice_1"] = str(repo / "weights" / "Kokoro-82M" / "voices" / "am_adam.pt")
        defaults["voice_2"] = str(repo / "weights" / "Kokoro-82M" / "voices" / "af_heart.pt")
    return defaults


def _status_line() -> str:
    bits = [f"python {sys.version.split()[0]}"]
    bits.append("ffmpeg ok" if shutil.which("ffmpeg") else "ffmpeg missing")
    try:
        probe = subprocess.run(
            [PYTHON, "-c", "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = probe.stdout.strip().splitlines()
        if len(lines) >= 2 and lines[0] == "True":
            bits.append(f"cuda {lines[1]} gpu")
        else:
            bits.append("cuda unavailable")
    except Exception as exc:
        bits.append(f"cuda check failed: {exc}")
    return " | ".join(bits)


def _path_state(label: str, value: str) -> str:
    if not value:
        return f"{label}: not set"
    return f"{label}: {'ok' if Path(value).exists() else 'missing'} ({value})"


def check_setup(model: str, speaker_mode: str, ckpt_dir: str, wav2vec_dir: str, infinitetalk_dir: str, quant_dir: str, lora_dir: str) -> str:
    defaults = _default_paths(model, speaker_mode)
    lines = [_status_line()]
    lines.append(_path_state("base checkpoint", ckpt_dir or defaults["ckpt_dir"]))
    lines.append(_path_state("wav2vec", wav2vec_dir or defaults["wav2vec_dir"]))
    if model == "InfiniteTalk":
        lines.append(_path_state("InfiniteTalk weights", infinitetalk_dir or defaults["infinitetalk_dir"]))
    if quant_dir:
        lines.append(_path_state("quant path", quant_dir))
    for idx, item in enumerate(_split_cli_values(lora_dir), start=1):
        lines.append(_path_state(f"LoRA {idx}", item))
    return "\n".join(lines)


@contextmanager
def _temporary_import_path(path: Path):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path = old_path


def _resolve_repo_path(repo: Path, value: str) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(repo / path)


def _render_tts_audio(repo: Path, job_dir: Path, text: str, voice_1: str, voice_2: str | None) -> dict[str, Any]:
    if not text.strip():
        raise gr.Error("TTS text is required.")
    with _temporary_import_path(repo):
        import soundfile as sf
        import torch
        from kokoro import KPipeline

        tts_dir = job_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        pipeline = KPipeline(lang_code="a", repo_id=str(repo / "weights" / "Kokoro-82M"), device="cpu")

        def synth(segment_text: str, voice_path: str):
            voice_tensor = torch.load(_resolve_repo_path(repo, voice_path), weights_only=True, map_location="cpu")
            chunks = []
            for _, _, audio in pipeline(segment_text, voice=voice_tensor, speed=1, split_pattern=r"\n+"):
                chunks.append(audio)
            if not chunks:
                raise gr.Error("TTS produced no audio.")
            return torch.concat(chunks, dim=0)

        if not voice_2:
            audio = synth(text, voice_1)
            audio_path = tts_dir / "person1.wav"
            sf.write(audio_path, audio.detach().cpu().numpy(), 24000)
            return {"person1": str(audio_path)}

        matches = re.findall(r"\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)", text, re.DOTALL)
        if not matches:
            raise gr.Error("Multi-speaker TTS needs markers like '(s1) hello (s2) hi'.")
        p1_parts = []
        p2_parts = []
        for speaker, content in matches:
            if speaker == "1":
                audio = synth(content.strip(), voice_1)
                p1_parts.append(audio)
                p2_parts.append(torch.zeros_like(audio))
            elif speaker == "2":
                audio = synth(content.strip(), voice_2)
                p1_parts.append(torch.zeros_like(audio))
                p2_parts.append(audio)
        p1 = torch.concat(p1_parts, dim=0)
        p2 = torch.concat(p2_parts, dim=0)
        p1_path = tts_dir / "person1.wav"
        p2_path = tts_dir / "person2.wav"
        sf.write(p1_path, p1.detach().cpu().numpy(), 24000)
        sf.write(p2_path, p2.detach().cpu().numpy(), 24000)
        return {"person1": str(p1_path), "person2": str(p2_path), "audio_type": "para"}


def _audio_mode_details(audio_mode: str) -> tuple[bool, bool, str]:
    is_tts = "TTS" in audio_mode
    is_multi = "Multi" in audio_mode
    audio_type = "para" if "parallel" in audio_mode or is_tts else "add"
    return is_tts, is_multi, audio_type


def _prepare_input_json(
    model: str,
    task_mode: str,
    prompt: str,
    image_input: Any,
    video_input: Any,
    audio_mode: str,
    audio_1: Any,
    audio_2: Any,
    tts_text: str,
    voice_1: str,
    voice_2: str,
    bbox_json: str,
    job_dir: Path,
    execute: bool,
) -> tuple[Path, str]:
    if not prompt.strip():
        raise gr.Error("Prompt is required.")

    repo = MODEL_ROOTS[model]
    is_tts, is_multi, audio_type = _audio_mode_details(audio_mode)
    input_data: dict[str, Any] = {"prompt": prompt.strip()}

    if model == "MultiTalk":
        image = _copy_input(image_input, job_dir, "image")
        if not image:
            raise gr.Error("MultiTalk requires an input image.")
        input_data["cond_image"] = image
    else:
        if task_mode == "VideoDubbing":
            video = _copy_input(video_input, job_dir, "video")
            if not video:
                raise gr.Error("InfiniteTalk VideoDubbing requires an input video.")
            input_data["cond_video"] = video
        else:
            image = _copy_input(image_input, job_dir, "image")
            if not image:
                raise gr.Error("InfiniteTalk SingleImageDriven requires an input image.")
            input_data["cond_video"] = image

    if is_tts:
        if model == "MultiTalk":
            tts_audio = {
                "text": tts_text.strip(),
                "human1_voice": voice_1.strip() or _default_paths(model, audio_mode)["voice_1"],
            }
            if is_multi:
                tts_audio["human2_voice"] = voice_2.strip() or _default_paths(model, audio_mode)["voice_2"]
                input_data["audio_type"] = "para"
            input_data["tts_audio"] = tts_audio
            input_data["cond_audio"] = {}
            audio_arg = "tts"
        else:
            if execute:
                rendered = _render_tts_audio(
                    repo,
                    job_dir,
                    tts_text,
                    voice_1.strip() or _default_paths(model, audio_mode)["voice_1"],
                    (voice_2.strip() or _default_paths(model, audio_mode)["voice_2"]) if is_multi else None,
                )
                input_data["cond_audio"] = {"person1": rendered["person1"]}
                if is_multi:
                    input_data["cond_audio"]["person2"] = rendered["person2"]
                    input_data["audio_type"] = rendered.get("audio_type", "para")
            else:
                input_data["cond_audio"] = {"person1": "<rendered_tts_person1.wav>"}
                if is_multi:
                    input_data["cond_audio"]["person2"] = "<rendered_tts_person2.wav>"
                    input_data["audio_type"] = "para"
            audio_arg = "localfile"
    else:
        audio1 = _copy_input(audio_1, job_dir, "audio1")
        if not audio1:
            raise gr.Error("Audio 1 is required.")
        input_data["cond_audio"] = {"person1": audio1}
        if is_multi:
            audio2 = _copy_input(audio_2, job_dir, "audio2")
            if not audio2:
                raise gr.Error("Audio 2 is required for multi-person mode.")
            input_data["cond_audio"]["person2"] = audio2
            input_data["audio_type"] = audio_type
        audio_arg = "localfile"

    if bbox_json.strip():
        try:
            input_data["bbox"] = json.loads(bbox_json)
        except json.JSONDecodeError as exc:
            raise gr.Error(f"Invalid bbox JSON: {exc}") from exc

    input_json = job_dir / "input.json"
    input_json.write_text(json.dumps(input_data, indent=2), encoding="utf-8")
    return input_json, audio_arg


def _append_if(command: list[str], flag: str, value: Any, *, skip_empty: bool = True):
    if skip_empty and (value is None or value == ""):
        return
    command.extend([flag, str(value)])


def _prepare_job(
    model: str,
    task_mode: str,
    prompt: str,
    negative_prompt: str,
    image_input: Any,
    video_input: Any,
    audio_mode: str,
    audio_1: Any,
    audio_2: Any,
    tts_text: str,
    voice_1: str,
    voice_2: str,
    bbox_json: str,
    resolution: str,
    mode: str,
    frame_num: int,
    max_frame_num: int,
    motion_frame: int,
    sample_steps: int,
    sample_shift: float,
    text_cfg: float,
    audio_cfg: float,
    seed: int,
    negative_enabled: bool,
    offload_model: bool,
    num_persistent: int,
    use_teacache: bool,
    teacache_thresh: float,
    use_apg: bool,
    apg_momentum: float,
    apg_norm_threshold: float,
    color_correction: float,
    quant: str,
    ckpt_dir: str,
    wav2vec_dir: str,
    infinitetalk_dir: str,
    quant_dir: str,
    dit_path: str,
    lora_dir: str,
    lora_scale: str,
    t5_cpu: bool,
    dit_fsdp: bool,
    t5_fsdp: bool,
    ulysses_size: int,
    ring_size: int,
    scene_seg: bool,
    save_file: str,
    audio_save_dir: str,
    execute: bool,
) -> PreparedJob:
    repo = MODEL_ROOTS[model]
    job_id, job_dir = _job_dir()
    defaults = _default_paths(model, audio_mode)
    input_json, audio_arg = _prepare_input_json(
        model,
        task_mode,
        prompt,
        image_input,
        video_input,
        audio_mode,
        audio_1,
        audio_2,
        tts_text,
        voice_1,
        voice_2,
        bbox_json,
        job_dir,
        execute,
    )

    script = SCRIPT_NAMES[model]
    if int(ulysses_size) > 1 or int(ring_size) > 1 or dit_fsdp or t5_fsdp:
        gpu_count = max(int(ulysses_size), int(ring_size), 1)
        command = ["torchrun", f"--nproc_per_node={gpu_count}", "--standalone", script]
    else:
        command = [PYTHON, script]

    output_base = _strip_mp4(Path(save_file).expanduser()) if save_file.strip() else job_dir / "result"
    if not output_base.is_absolute():
        output_base = (ROOT / output_base).resolve()
    output_file = Path(str(output_base) + ".mp4")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    command.extend(["--task", TASK_NAMES[model]])
    _append_if(command, "--ckpt_dir", ckpt_dir.strip() or defaults["ckpt_dir"])
    _append_if(command, "--wav2vec_dir", wav2vec_dir.strip() or defaults["wav2vec_dir"])
    _append_if(command, "--input_json", input_json)
    _append_if(command, "--save_file", output_base)
    _append_if(command, "--audio_save_dir", audio_save_dir.strip() or str(job_dir / "audio"))
    _append_if(command, "--size", resolution)
    _append_if(command, "--mode", mode)
    _append_if(command, "--frame_num", int(frame_num))
    _append_if(command, "--motion_frame", int(motion_frame))
    _append_if(command, "--sample_steps", int(sample_steps))
    _append_if(command, "--sample_shift", sample_shift)
    _append_if(command, "--sample_text_guide_scale", text_cfg)
    _append_if(command, "--sample_audio_guide_scale", audio_cfg)
    _append_if(command, "--base_seed", int(seed))
    _append_if(command, "--offload_model", "true" if offload_model else "false")
    _append_if(command, "--ulysses_size", int(ulysses_size))
    _append_if(command, "--ring_size", int(ring_size))
    _append_if(command, "--color_correction_strength", color_correction)
    _append_if(command, "--apg_momentum", apg_momentum)
    _append_if(command, "--apg_norm_threshold", apg_norm_threshold)
    _append_if(command, "--audio_mode", audio_arg)

    if negative_enabled and negative_prompt.strip():
        _append_if(command, "--n_prompt", negative_prompt.strip())

    if model == "InfiniteTalk":
        _append_if(command, "--max_frame_num", int(max_frame_num))
        _append_if(command, "--infinitetalk_dir", infinitetalk_dir.strip() or defaults["infinitetalk_dir"])
        _append_if(command, "--dit_path", dit_path.strip())
        if scene_seg and task_mode == "VideoDubbing":
            command.append("--scene_seg")

    if t5_cpu:
        command.append("--t5_cpu")
    if dit_fsdp:
        command.append("--dit_fsdp")
    if t5_fsdp:
        command.append("--t5_fsdp")
    if int(num_persistent) >= 0:
        _append_if(command, "--num_persistent_param_in_dit", int(num_persistent))
    if use_teacache:
        command.append("--use_teacache")
        _append_if(command, "--teacache_thresh", teacache_thresh)
    if use_apg:
        command.append("--use_apg")
    if quant != "None":
        _append_if(command, "--quant", quant)
    if quant_dir.strip():
        _append_if(command, "--quant_dir", quant_dir.strip())

    loras = _split_cli_values(lora_dir)
    if loras:
        command.append("--lora_dir")
        command.extend(loras)
        scales = _split_cli_values(lora_scale) or ["1.0"]
        command.append("--lora_scale")
        command.extend(scales)

    manifest = {
        "job_id": job_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "task_mode": task_mode,
        "audio_mode": audio_mode,
        "input_json": str(input_json),
        "output_file": str(output_file),
        "command": command,
        "note": "Core model dtype is configured by the upstream repos; None uses bf16/fp16 mixed weights, quant selects int8/fp8 weights when present.",
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedJob(job_id, job_dir, repo, command, manifest, output_file)


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def preview_job(*values):
    try:
        job = _prepare_job(*values, execute=False)
    except Exception as exc:
        return "Preview failed", "", "", str(exc), None
    return "Preview ready", _command_text(job.command), json.dumps(job.manifest, indent=2), "", None


def run_job(*values):
    try:
        job = _prepare_job(*values, execute=True)
    except Exception as exc:
        yield "Could not start", "", "", str(exc), None
        return

    command_text = _command_text(job.command)
    manifest_text = json.dumps(job.manifest, indent=2)
    log_lines = [f"Job {job.job_id}", command_text, ""]
    yield "Running", command_text, manifest_text, "\n".join(log_lines), None

    process = subprocess.Popen(
        job.command,
        cwd=job.repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log_lines.append(line.rstrip())
        yield "Running", command_text, manifest_text, "\n".join(log_lines[-180:]), None
    code = process.wait()
    if code == 0 and job.output_file.exists():
        log_lines.append(f"Finished: {job.output_file}")
        yield "Finished", command_text, manifest_text, "\n".join(log_lines[-180:]), str(job.output_file)
    else:
        log_lines.append(f"Exited with code {code}")
        yield "Failed", command_text, manifest_text, "\n".join(log_lines[-220:]), None


def update_model(model: str, task_mode: str, audio_mode: str):
    tasks = TASKS[model]
    task_value = task_mode if task_mode in tasks else tasks[0]
    sizes = SIZES[model]
    defaults = _default_paths(model, audio_mode)
    is_infinite = model == "InfiniteTalk"
    show_video = is_infinite and task_value == "VideoDubbing"
    show_image = model == "MultiTalk" or task_value == "SingleImageDriven"
    return (
        gr.update(choices=tasks, value=task_value, visible=is_infinite),
        gr.update(choices=sizes, value=sizes[0]),
        gr.update(visible=show_image),
        gr.update(visible=show_video),
        gr.update(visible=is_infinite),
        gr.update(visible=is_infinite),
        gr.update(visible=is_infinite and task_value == "VideoDubbing"),
        defaults["ckpt_dir"],
        defaults["wav2vec_dir"],
        defaults["infinitetalk_dir"],
        defaults["voice_1"],
        defaults["voice_2"],
    )


def update_audio_mode(audio_mode: str):
    is_tts, is_multi, _ = _audio_mode_details(audio_mode)
    return (
        gr.update(visible=not is_tts),
        gr.update(visible=(not is_tts and is_multi)),
        gr.update(visible=is_tts),
        gr.update(visible=is_tts),
        gr.update(visible=(is_tts and is_multi)),
        gr.update(visible=is_multi),
    )


def update_task(model: str, task_mode: str):
    show_video = model == "InfiniteTalk" and task_mode == "VideoDubbing"
    show_image = model == "MultiTalk" or task_mode == "SingleImageDriven"
    return (
        gr.update(visible=show_image),
        gr.update(visible=show_video),
        gr.update(visible=show_video),
    )


def list_jobs() -> str:
    if not JOBS_ROOT.exists():
        return "No jobs yet."
    rows = []
    for manifest in sorted(JOBS_ROOT.glob("*/manifest.json"), reverse=True)[:40]:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        output = Path(data.get("output_file", ""))
        state = "done" if output.exists() else "not finished"
        rows.append(f"{data.get('job_id')} | {state} | {data.get('model')} | {data.get('task_mode')} | {output}")
    return "\n".join(rows) if rows else "No jobs yet."


CSS = """
.topbar {
  border-bottom: 1px solid #30343a;
  padding: 10px 0 14px 0;
  margin-bottom: 12px;
}
.title {
  font-size: 25px;
  font-weight: 760;
  letter-spacing: 0;
}
.subtitle {
  color: #9fa8a3;
  font-size: 13px;
  margin-top: 4px;
}
.logbox textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 12px !important;
}
.commandbox textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 12px !important;
}
"""


def build_app():
    defaults = _default_paths("InfiniteTalk", "Single Person(Local File)")
    with gr.Blocks(title="Talk Video Super App") as app:
        gr.HTML(
            f"""
            <div class="topbar">
              <div class="title">Talk Video Super App</div>
              <div class="subtitle">One control room for InfiniteTalk and MultiTalk. {_status_line()}</div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Generate"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=360):
                        model = gr.Radio(MODELS, value="InfiniteTalk", label="Model")
                        task_mode = gr.Radio(TASKS["InfiniteTalk"], value="VideoDubbing", label="Task")
                        image_input = gr.Image(type="filepath", label="Reference image", visible=False, height=260)
                        video_input = gr.Video(label="Reference video", visible=True, height=260)
                        prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Describe the scene, performance, camera, motion, and interaction.")
                        negative_enabled = gr.Checkbox(value=True, label="Use negative prompt")
                        negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=4)

                        with gr.Accordion("Audio", open=True):
                            audio_mode = gr.Radio(
                                [
                                    "Single Person(Local File)",
                                    "Single Person(TTS)",
                                    "Multi Person(Local File, audio add)",
                                    "Multi Person(Local File, audio parallel)",
                                    "Multi Person(TTS)",
                                ],
                                value="Single Person(Local File)",
                                label="Speaker/audio mode",
                            )
                            with gr.Row():
                                audio_1 = gr.Audio(label="Audio 1", type="filepath", visible=True)
                                audio_2 = gr.Audio(label="Audio 2", type="filepath", visible=False)
                            tts_text = gr.Textbox(label="TTS text", lines=4, visible=False, placeholder="Single speaker text, or use (s1)/(s2) markers for multi-speaker TTS.")
                            with gr.Row():
                                voice_1 = gr.Textbox(label="Voice 1", value=defaults["voice_1"], visible=False)
                                voice_2 = gr.Textbox(label="Voice 2", value=defaults["voice_2"], visible=False)
                            bbox_json = gr.Textbox(
                                label="Optional bbox JSON",
                                lines=3,
                                visible=False,
                                placeholder='{"person1":[160,120,1280,1080],"person2":[160,1320,1280,2280]}',
                            )

                    with gr.Column(scale=5, min_width=360):
                        with gr.Accordion("Generation", open=True):
                            with gr.Row():
                                resolution = gr.Radio(SIZES["InfiniteTalk"], value="infinitetalk-480", label="Resolution")
                                mode = gr.Radio(["clip", "streaming"], value="streaming", label="Mode")
                            with gr.Row():
                                frame_num = gr.Slider(17, 201, value=81, step=4, label="Frames per clip")
                                max_frame_num = gr.Slider(81, 3000, value=1000, step=1, label="Max frames")
                            with gr.Row():
                                motion_frame = gr.Slider(1, 33, value=9, step=1, label="Motion frames")
                                sample_steps = gr.Slider(1, 1000, value=8, step=1, label="Diffusion steps")
                            with gr.Row():
                                sample_shift = gr.Slider(0, 16, value=2, step=0.5, label="Sample shift")
                                seed = gr.Slider(-1, 2147483647, value=42, step=1, label="Seed")
                            with gr.Row():
                                text_cfg = gr.Slider(0, 20, value=1, step=0.1, label="Text CFG")
                                audio_cfg = gr.Slider(0, 20, value=2, step=0.1, label="Audio CFG")
                            color_correction = gr.Slider(0, 1, value=1, step=0.05, label="Color correction")

                        with gr.Accordion("Memory And Acceleration", open=True):
                            with gr.Row():
                                offload_model = gr.Checkbox(value=True, label="Offload model")
                                t5_cpu = gr.Checkbox(value=False, label="T5 on CPU")
                                use_teacache = gr.Checkbox(value=False, label="TeaCache")
                                use_apg = gr.Checkbox(value=False, label="APG")
                            with gr.Row():
                                num_persistent = gr.Number(value=-1, precision=0, label="Persistent DiT params (-1 off, 0 low VRAM)")
                                teacache_thresh = gr.Slider(0.1, 0.8, value=0.2, step=0.05, label="TeaCache threshold")
                            with gr.Row():
                                apg_momentum = gr.Number(value=-0.75, label="APG momentum")
                                apg_norm_threshold = gr.Number(value=55, label="APG norm threshold")
                            quant = gr.Radio(["None", "int8", "fp8"], value="None", label="Quantized weights")

                        with gr.Accordion("Model Files", open=False):
                            ckpt_dir = gr.Textbox(label="Wan base checkpoint", value=defaults["ckpt_dir"])
                            wav2vec_dir = gr.Textbox(label="Wav2Vec checkpoint", value=defaults["wav2vec_dir"])
                            infinitetalk_dir = gr.Textbox(label="InfiniteTalk weights", value=defaults["infinitetalk_dir"])
                            quant_dir = gr.Textbox(label="Quant dir / safetensors")
                            dit_path = gr.Textbox(label="DiT path override", visible=True)
                            lora_dir = gr.Textbox(label="LoRA paths", placeholder="Use shell-style quoting for multiple paths")
                            lora_scale = gr.Textbox(label="LoRA scales", value="1.0")

                        with gr.Accordion("Distributed And Output", open=False):
                            with gr.Row():
                                dit_fsdp = gr.Checkbox(value=False, label="DiT FSDP")
                                t5_fsdp = gr.Checkbox(value=False, label="T5 FSDP")
                            with gr.Row():
                                ulysses_size = gr.Slider(1, 8, value=1, step=1, label="Ulysses size")
                                ring_size = gr.Slider(1, 8, value=1, step=1, label="Ring size")
                            scene_seg = gr.Checkbox(value=False, label="Scene segmentation", visible=True)
                            save_file = gr.Textbox(label="Save file base", placeholder="Blank saves under super_app_jobs")
                            audio_save_dir = gr.Textbox(label="Audio embedding dir", placeholder="Blank saves inside the job")

                    with gr.Column(scale=6, min_width=420):
                        with gr.Row():
                            preview_btn = gr.Button("Preview command")
                            run_btn = gr.Button("Generate", variant="primary")
                            check_btn = gr.Button("Check setup")
                        status = gr.Textbox(label="Status", value="Idle", interactive=False)
                        output = gr.Video(label="Output", height=300)
                        command = gr.Textbox(label="Command", lines=5, elem_classes=["commandbox"])
                        manifest = gr.Code(label="Manifest", language="json", lines=12)
                        logs = gr.Textbox(label="Logs", lines=18, elem_classes=["logbox"])

                all_inputs = [
                    model,
                    task_mode,
                    prompt,
                    negative_prompt,
                    image_input,
                    video_input,
                    audio_mode,
                    audio_1,
                    audio_2,
                    tts_text,
                    voice_1,
                    voice_2,
                    bbox_json,
                    resolution,
                    mode,
                    frame_num,
                    max_frame_num,
                    motion_frame,
                    sample_steps,
                    sample_shift,
                    text_cfg,
                    audio_cfg,
                    seed,
                    negative_enabled,
                    offload_model,
                    num_persistent,
                    use_teacache,
                    teacache_thresh,
                    use_apg,
                    apg_momentum,
                    apg_norm_threshold,
                    color_correction,
                    quant,
                    ckpt_dir,
                    wav2vec_dir,
                    infinitetalk_dir,
                    quant_dir,
                    dit_path,
                    lora_dir,
                    lora_scale,
                    t5_cpu,
                    dit_fsdp,
                    t5_fsdp,
                    ulysses_size,
                    ring_size,
                    scene_seg,
                    save_file,
                    audio_save_dir,
                ]

                preview_btn.click(preview_job, inputs=all_inputs, outputs=[status, command, manifest, logs, output])
                run_btn.click(run_job, inputs=all_inputs, outputs=[status, command, manifest, logs, output])
                check_btn.click(
                    check_setup,
                    inputs=[model, audio_mode, ckpt_dir, wav2vec_dir, infinitetalk_dir, quant_dir, lora_dir],
                    outputs=logs,
                )
                model.change(
                    update_model,
                    inputs=[model, task_mode, audio_mode],
                    outputs=[
                        task_mode,
                        resolution,
                        image_input,
                        video_input,
                        max_frame_num,
                        infinitetalk_dir,
                        scene_seg,
                        ckpt_dir,
                        wav2vec_dir,
                        infinitetalk_dir,
                        voice_1,
                        voice_2,
                    ],
                )
                task_mode.change(update_task, inputs=[model, task_mode], outputs=[image_input, video_input, scene_seg])
                audio_mode.change(update_audio_mode, inputs=audio_mode, outputs=[audio_1, audio_2, tts_text, voice_1, voice_2, bbox_json])
                audio_mode.change(
                    lambda m, a: (_default_paths(m, a)["infinitetalk_dir"], _default_paths(m, a)["voice_1"], _default_paths(m, a)["voice_2"]),
                    inputs=[model, audio_mode],
                    outputs=[infinitetalk_dir, voice_1, voice_2],
                )

            with gr.Tab("Jobs"):
                refresh = gr.Button("Refresh jobs")
                jobs = gr.Textbox(label="Recent jobs", value=list_jobs, lines=24)
                refresh.click(list_jobs, outputs=jobs)

        app.queue(max_size=8, default_concurrency_limit=1)
    return app


def main():
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    build_app().launch(server_name="0.0.0.0", server_port=8420, show_error=True, css=CSS)


if __name__ == "__main__":
    main()
