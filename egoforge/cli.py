from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        prog="egoforge", description="Egocentric motion annotation and quality control"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="audit, clean, annotate and export the pinned source subset"
    )
    run.add_argument("--raw", default="data/raw/humanola")
    run.add_argument("--output", default="output/run")
    run.add_argument("--config", default="configs/default.json")
    run.add_argument("--skip-media", action="store_true")
    serve = sub.add_parser("serve", help="launch local annotation review console")
    serve.add_argument("--raw", default="data/raw/humanola")
    serve.add_argument("--output", default="output/run")
    serve.add_argument("--port", type=int, default=8765)
    verify = sub.add_parser("verify", help="verify derived artifact checksums")
    verify.add_argument("--output", default="output/run")
    args = parser.parse_args()
    if args.command == "run":
        from .pipeline import build

        build(args.raw, args.output, args.config, not args.skip_media)
    elif args.command == "serve":
        from .review import serve

        serve(Path(args.output), Path(args.raw), args.port)
    elif args.command == "verify":
        from .data import sha256

        root = Path(args.output)
        manifest = json.loads((root / "artifact_manifest.json").read_text())
        bad = [
            p
            for p, h in manifest["artifacts"].items()
            if not (root / p).is_file() or sha256(root / p) != h
        ]
        print(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "artifacts_checked": len(manifest["artifacts"]),
                    "mismatches": bad,
                },
                indent=2,
            )
        )
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
