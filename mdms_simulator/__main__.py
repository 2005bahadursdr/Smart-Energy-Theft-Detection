"""
CLI entry point for mdms_simulator.
Allows running:
  python -m mdms_simulator --mode demo
  python -m mdms_simulator --mode serve --port 8001
"""

import argparse
import sys
import uvicorn

from mdms_simulator.config import LEGAL_DISCLAIMER
from mdms_simulator.demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mdms_simulator",
        description="Smart Grid Meter Data Management System (MDMS) Simulator"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "serve"],
        default="demo",
        help="Execution mode: 'demo' for end-to-end local demonstration, 'serve' to run FastAPI REST API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for FastAPI server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port number for FastAPI server"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        success = run_demo()
        sys.exit(0 if success else 1)
    elif args.mode == "serve":
        print("================================================================================")
        print(" [SMART GRID] MDMS SIMULATOR REST API SERVER")
        print("================================================================================")
        print(f" Listening on : http://{args.host}:{args.port}")
        print(f" API Docs     : http://{args.host}:{args.port}/docs")
        print(f" Notice       : {LEGAL_DISCLAIMER}")
        print("================================================================================")
        uvicorn.run("mdms_simulator.api.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
