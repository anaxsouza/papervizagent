"""
generate.py

Responsibility: Provide a CLI to generate and refine PaperVizAgent figures.
Depends on: PaperVizProcessor, agent classes, and utils/generation_utils gemini client.
Does not depend on: Streamlit UI.
"""

import argparse
import asyncio
import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from google.genai import types
from PIL import Image

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.polish_agent import PolishAgent
from agents.retriever_agent import RetrieverAgent
from agents.stylist_agent import StylistAgent
from agents.vanilla_agent import VanillaAgent
from agents.visualizer_agent import VisualizerAgent
from utils import config as config_utils
from utils.paperviz_processor import PaperVizProcessor
from utils.generation_utils import gemini_client


AVAILABLE_MODES = [
    "vanilla",
    "dev_planner",
    "dev_planner_stylist",
    "dev_planner_critic",
    "demo_planner_critic",
    "dev_full",
    "demo_full",
    "dev_polish",
    "dev_retriever",
]


def _sanitize_filename(name: str) -> str:
    keep = []
    for ch in name.lower().strip():
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch in {" ", "/", "\\", ":"}:
            keep.append("_")
    sanitized = "".join(keep).strip("_")
    return sanitized or "figure"


def _split_filters(figure_filters: list[str] | None) -> list[str]:
    if not figure_filters:
        return []

    merged: list[str] = []
    for raw in figure_filters:
        for token in raw.split(","):
            token = token.strip().lower()
            if token:
                merged.append(token)
    return merged


def _matches_filter(entry: dict[str, Any], filters: list[str]) -> bool:
    if not filters:
        return True

    filename = str(entry.get("filename", "")).lower()
    caption = str(entry.get("caption", "")).lower()
    for needle in filters:
        if filename == needle or filename.startswith(needle) or needle in filename:
            return True
        if needle in caption:
            return True
    return False


def _load_json_inputs(input_path: Path, filters: list[str]) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict):
        loaded = [loaded]
    if not isinstance(loaded, list):
        raise ValueError("Input JSON must be a list or object.")

    filtered = [
        entry
        for entry in loaded
        if isinstance(entry, dict) and _matches_filter(entry, filters)
    ]
    if not filtered:
        raise ValueError("No input entries matched the selection.")
    return filtered


def _build_direct_input(content: str, caption: str) -> list[dict[str, Any]]:
    return [
        {
            "filename": "direct_input",
            "caption": caption,
            "content": content,
            "visual_intent": caption,
        }
    ]


def _prepare_data_list(
    entries: list[dict[str, Any]],
    aspect_ratio: str,
    candidates: int,
    critic_rounds: int,
) -> list[dict[str, Any]]:
    data_list: list[dict[str, Any]] = []
    global_candidate_id = 0

    for entry_idx, entry in enumerate(entries):
        base_filename = _sanitize_filename(
            str(entry.get("filename", f"input_{entry_idx}"))
        )
        caption = str(entry.get("caption", "")).strip()
        content = entry.get("content", "")
        visual_intent = str(entry.get("visual_intent", caption)).strip() or caption

        for local_candidate_id in range(candidates):
            if candidates > 1:
                item_filename = f"{base_filename}_candidate_{local_candidate_id}"
            else:
                item_filename = base_filename

            data_list.append(
                {
                    "filename": item_filename,
                    "source_filename": base_filename,
                    "caption": caption,
                    "content": content,
                    "visual_intent": visual_intent,
                    "candidate_id": global_candidate_id,
                    "max_critic_rounds": critic_rounds,
                    "additional_info": {"rounded_ratio": aspect_ratio},
                }
            )
            global_candidate_id += 1

    return data_list


def _build_processor(exp_config: config_utils.ExpConfig) -> PaperVizProcessor:
    return PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )


def _resolve_final_image_key(result: dict[str, Any], task_name: str) -> str | None:
    eval_key = result.get("eval_image_field")
    if isinstance(eval_key, str) and result.get(eval_key):
        return eval_key

    for idx in range(9, -1, -1):
        key = f"target_{task_name}_critic_desc{idx}_base64_jpg"
        if result.get(key):
            return key

    fallbacks = [
        f"target_{task_name}_stylist_desc0_base64_jpg",
        f"target_{task_name}_desc0_base64_jpg",
        f"vanilla_{task_name}_base64_jpg",
        f"polished_{task_name}_base64_jpg",
    ]
    for key in fallbacks:
        if result.get(key):
            return key
    return None


def _decode_image_from_b64(image_b64: str) -> Image.Image | None:
    if not image_b64:
        return None
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_b64)
        return Image.open(BytesIO(raw))
    except Exception:
        return None


def _save_results_json(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", errors="surrogateescape") as f:
        json_string = json.dumps(payload, ensure_ascii=False, indent=2)
        json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
        f.write(json_string)


def _build_results_payload(
    args: argparse.Namespace,
    timestamp: str,
    all_results: list[dict[str, Any]],
    saved_images: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata": {
            "timestamp": timestamp,
            "mode": args.mode,
            "task": args.task,
            "retrieval": args.retrieval,
            "aspect_ratio": args.aspect_ratio,
            "resolution": args.resolution,
            "temperature": args.temperature,
            "critic_rounds": args.critic_rounds,
            "candidates": args.candidates,
            "concurrency": args.concurrency,
            "input": args.input,
        },
        "saved_images": saved_images,
        "results": all_results,
    }


async def run_generate(args: argparse.Namespace) -> None:
    direct_content = args.content
    if args.content_file:
        content_file = Path(args.content_file)
        if not content_file.exists():
            raise FileNotFoundError(f"Content file not found: {content_file}")
        direct_content = content_file.read_text(encoding="utf-8")

    if direct_content:
        caption = args.caption or "Generated figure"
        entries = _build_direct_input(direct_content, caption)
    else:
        filters = _split_filters(args.figure)
        entries = _load_json_inputs(Path(args.input), filters)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not direct_content:
        pending_entries: list[dict[str, Any]] = []
        skipped_existing = 0
        for entry in entries:
            entry_filename = _sanitize_filename(str(entry.get("filename", "")))
            if (output_dir / f"{entry_filename}.png").exists():
                skipped_existing += 1
                continue
            pending_entries.append(entry)

        if skipped_existing:
            print(
                f"Skipping {skipped_existing} already-generated figure(s) found in {output_dir}."
            )
        entries = pending_entries

    if not entries:
        print("All selected figures are already generated. Nothing to do.")
        return

    if args.max_figures > 0 and len(entries) > args.max_figures:
        print(
            f"Limiting to first {args.max_figures} figure(s) out of {len(entries)} remaining entries."
        )
        entries = entries[: args.max_figures]

    data_list = _prepare_data_list(
        entries=entries,
        aspect_ratio=args.aspect_ratio,
        candidates=args.candidates,
        critic_rounds=args.critic_rounds,
    )

    exp_config = config_utils.ExpConfig(
        dataset_name="PaperBananaBench",
        task_name=args.task,
        split_name="cli",
        temperature=args.temperature,
        exp_mode=args.mode,
        retrieval_setting=args.retrieval,
        max_critic_rounds=args.critic_rounds,
        model_name=args.model_name,
        image_model_name=args.image_model_name,
        image_size=args.resolution,
        work_dir=Path(__file__).parent,
    )

    processor = _build_processor(exp_config)

    print(f"Running mode={args.mode} task={args.task} retrieval={args.retrieval}")
    print(
        f"Inputs={len(entries)} candidates_per_input={args.candidates} total_jobs={len(data_list)}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json_path = output_dir / "results.json"

    all_results: list[dict[str, Any]] = []
    saved_images: list[dict[str, Any]] = []
    async for result_data in processor.process_queries_batch(
        data_list,
        max_concurrent=args.concurrency,
        do_eval=False,
    ):
        all_results.append(result_data)

        idx = len(all_results) - 1
        image_key = _resolve_final_image_key(result_data, args.task)
        if not image_key:
            _save_results_json(
                results_json_path,
                _build_results_payload(args, timestamp, all_results, saved_images),
            )
            continue

        image_obj = _decode_image_from_b64(str(result_data.get(image_key, "")))
        if image_obj is None:
            _save_results_json(
                results_json_path,
                _build_results_payload(args, timestamp, all_results, saved_images),
            )
            continue

        filename = _sanitize_filename(
            str(result_data.get("filename", f"candidate_{idx}"))
        )
        png_path = output_dir / f"{filename}.png"
        if png_path.exists():
            png_path = output_dir / f"{filename}_{idx}.png"

        image_obj.convert("RGB").save(png_path, format="PNG")
        print(f"Saved image {idx + 1}/{len(data_list)}: {png_path}")
        saved_images.append(
            {
                "candidate_index": idx,
                "filename": result_data.get("filename"),
                "image_key": image_key,
                "path": str(png_path),
            }
        )
        _save_results_json(
            results_json_path,
            _build_results_payload(args, timestamp, all_results, saved_images),
        )

    _save_results_json(
        results_json_path,
        _build_results_payload(args, timestamp, all_results, saved_images),
    )

    print(f"Saved {len(saved_images)} PNG files to {output_dir}")
    print(f"Saved detailed results to {output_dir / 'results.json'}")


async def run_refine(args: argparse.Namespace) -> None:
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client is not initialized. Check your API key or config."
        )

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_name = args.image_model_name
    if not model_name:
        exp_config = config_utils.ExpConfig(
            dataset_name="PaperBananaBench",
            task_name="diagram",
            split_name="cli",
            exp_mode="demo_planner_critic",
            retrieval_setting="none",
            work_dir=Path(__file__).parent,
        )
        model_name = exp_config.image_model_name

    with Image.open(image_path) as img:
        buffer = BytesIO()
        img.convert("RGB").save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    contents = [
        types.Part.from_text(text=args.prompt),
        types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes),
    ]

    gen_config = types.GenerateContentConfig(
        temperature=args.temperature,
        max_output_tokens=8192,
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=args.aspect_ratio,
            image_size=args.resolution,
        ),
    )

    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model=model_name,
        contents=contents,
        config=gen_config,
    )

    image_data: bytes | None = None
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data:
                data = part.inline_data.data
                if isinstance(data, bytes):
                    image_data = data
                elif isinstance(data, str):
                    image_data = base64.b64decode(data)
                break

    if not image_data:
        raise RuntimeError("No image data returned by refinement model.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{_sanitize_filename(image_path.stem)}_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = output_dir / out_name

    with Image.open(BytesIO(image_data)) as refined_img:
        refined_img.convert("RGB").save(out_path, format="PNG")

    print(f"Refined image saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PaperVizAgent CLI for generation and refinement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Generate figures with PaperVizAgent pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate.add_argument(
        "-i",
        "--input",
        default="data/sentinel_figures.json",
        help="Input JSON path containing one or more figure entries",
    )
    generate.add_argument(
        "--content",
        default="",
        help="Inline content text (overrides --input and --content_file)",
    )
    generate.add_argument(
        "--content_file",
        default="",
        help="Path to UTF-8 text file used as content (overrides --input)",
    )
    generate.add_argument(
        "--caption",
        default="",
        help="Caption/visual intent for --content or --content_file mode",
    )
    generate.add_argument(
        "-o",
        "--output_dir",
        default="results/cli_generate",
        help="Directory where generated PNG files and results.json are saved",
    )
    generate.add_argument(
        "--mode",
        default="dev_full",
        choices=AVAILABLE_MODES,
        help="Pipeline mode (dev_full recommended for planner+critic workflow)",
    )
    generate.add_argument(
        "-n",
        "--candidates",
        type=int,
        default=1,
        help="Number of candidates per input figure",
    )
    generate.add_argument(
        "--resolution",
        default="4K",
        choices=["1K", "2K", "4K"],
        help="Image resolution passed to the generation model",
    )
    generate.add_argument(
        "--aspect_ratio",
        default="16:9",
        help="Target aspect ratio for generated images",
    )
    generate.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for text planning and image prompt generation",
    )
    generate.add_argument(
        "--critic_rounds",
        type=int,
        default=3,
        help="Maximum number of critic refinement rounds",
    )
    generate.add_argument(
        "--task",
        default="diagram",
        choices=["diagram", "plot"],
        help="Task type for prompt templates and render behavior",
    )
    generate.add_argument(
        "--model_name",
        default="",
        help="Text model override (empty = use configs/model_config.yaml default)",
    )
    generate.add_argument(
        "--image_model_name",
        default="",
        help="Image model override (empty = use configs/model_config.yaml default)",
    )
    generate.add_argument(
        "--retrieval",
        default="none",
        choices=["auto", "manual", "random", "none"],
        help="Retriever strategy for reference examples",
    )
    generate.add_argument(
        "--figure",
        action="append",
        default=[],
        help="Filter entries by filename/caption (repeatable or comma-separated)",
    )
    generate.add_argument(
        "--max_figures",
        type=int,
        default=0,
        help="Maximum number of input figures to process in one run (0 = no limit)",
    )
    generate.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent generation jobs (1 = fully sequential)",
    )

    refine = subparsers.add_parser(
        "refine",
        help="Refine an existing image using a user prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    refine.add_argument(
        "--image",
        required=True,
        help="Path to source image that will be refined",
    )
    refine.add_argument(
        "--prompt",
        required=True,
        help="User refinement instruction (what to change in the image)",
    )
    refine.add_argument(
        "-o",
        "--output_dir",
        default="results/cli_refine",
        help="Directory where refined PNG files are saved",
    )
    refine.add_argument(
        "--resolution",
        default="4K",
        choices=["1K", "2K", "4K"],
        help="Target refinement resolution",
    )
    refine.add_argument(
        "--aspect_ratio",
        default="16:9",
        help="Target aspect ratio for the refined image",
    )
    refine.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for refinement",
    )
    refine.add_argument(
        "--image_model_name",
        default="",
        help="Image model override (empty = use configs/model_config.yaml default)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        if args.content and args.content_file:
            raise ValueError("Use only one of --content or --content_file.")
        asyncio.run(run_generate(args))
        return

    if args.command == "refine":
        asyncio.run(run_refine(args))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
