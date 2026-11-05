"""Standalone CLI for the MOLT research module."""

import argparse
import json
from pathlib import Path

from .benchmark import DemoConfig, run_demo
from .store import load_skill


def main() -> None:
    parser = argparse.ArgumentParser(description="KiteWatt: neural-to-program efficiency research")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Train a tiny teacher and attempt skill compilation")
    demo.add_argument("--operator", choices=("add_mod", "xor", "max", "affine_mod"), default="max")
    demo.add_argument("--modulus", type=int, default=8)
    demo.add_argument("--steps", type=int, default=300)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--requests", type=int, default=256)
    demo.add_argument("--output", default="runs/molt")
    text = subparsers.add_parser("text-lab", help="Learn, validate, and evaluate text skills without paid APIs")
    text.add_argument("--seed", type=int, default=7)
    text.add_argument("--output", default="runs/text-lab")
    text.add_argument("--max-queries", type=int, default=18)
    ollama = subparsers.add_parser("ollama-demo", help="Compile from a local Ollama model (optional)")
    ollama.add_argument("--model", default="llama3.2:1b")
    ollama.add_argument("--task", choices=("slug", "email_domain", "url_segment"), default="slug")
    ollama.add_argument("--port", type=int, default=11434)
    ollama.add_argument("--seed", type=int, default=7)
    ollama.add_argument("--max-queries", type=int, default=18)
    ollama.add_argument("--output", default="runs/ollama-demo")
    serve = subparsers.add_parser("serve", help="Open the local dashboard (http://127.0.0.1:8765)")
    serve.add_argument("--port", type=int, default=8765)
    query = subparsers.add_parser("query", help="Query a locally stored certified skill")
    query.add_argument("--skill", type=Path, required=True)
    query.add_argument("--left", type=int, required=True)
    query.add_argument("--right", type=int, required=True)
    args = parser.parse_args()
    if args.command == "ollama-demo":
        from .ollama import run_ollama_demo
        report = run_ollama_demo(args.task, model=args.model, seed=args.seed,
                                 output=args.output, max_queries=args.max_queries,
                                 port=args.port)
        print(f"Local model: {report['model']} · status: {report['status']}")
        print(f"Teacher calls: {report['model_calls']} · held-out accuracy: {report['heldout_accuracy']}")
        print("Report written to:", Path(args.output) / "ollama_report.json")
        return
    if args.command == "serve":
        from .web import serve as serve_dashboard
        serve_dashboard(port=args.port)
        return
    if args.command == "text-lab":
        from .lab import LabConfig, run_text_lab
        report = run_text_lab(LabConfig(seed=args.seed, output=args.output,
                                        max_queries=args.max_queries))
        for name, task in report["tasks"].items():
            print(f"{name:14} {task['status']:10} held-out={task['heldout_test_accuracy']:.3f} "
                  f"queries={task['teacher_calls_for_discovery']} "
                  f"ops={task['program']}")
        print("Report written to:", Path(args.output) / "report.json")
        return
    if args.command == "query":
        skill = load_skill(args.skill)
        print(f"{skill.name}({args.left}, {args.right}) = {skill.predict(args.left, args.right)}")
        return
    report = run_demo(DemoConfig(
        operator=args.operator, modulus=args.modulus, steps=args.steps,
        seed=args.seed, benchmark_requests=args.requests, output_dir=args.output,
    ))
    print("MOLT compiler:", report["compiler"]["status"])
    print("Teacher full-domain accuracy:", report["training"]["teacher_full_domain_accuracy"])
    for name, result in report["results"].items():
        print(f"{name:19s}: accuracy={result['accuracy']:.3f}, "
              f"median latency={result['median_ms']:.5f} ms")
    print("Results written to:", Path(args.output) / "report.json")


if __name__ == "__main__":
    main()
