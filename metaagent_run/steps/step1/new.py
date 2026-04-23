import asyncio
import sys

from .orchestrator import main_async


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        _ = reconfigure(line_buffering=True)
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
