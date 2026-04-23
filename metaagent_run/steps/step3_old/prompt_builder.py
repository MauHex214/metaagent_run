import json
from functools import lru_cache
from pathlib import Path

from .config import RuntimeConfig
from .schema import format_categorized_schema_for_prompt

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SECTION_INSTRUCTION_PLACEHOLDER = "__SECTION_INSTRUCTION__"
SCHEMA_NOTE_PLACEHOLDER = "__SCHEMA_NOTE__"
SCHEMA_STR_PLACEHOLDER = "__SCHEMA_STR__"


@lru_cache(maxsize=16)
def load_prompt_template(prompt_version: str) -> str:
    prompt_file = PROMPT_DIR / f"{prompt_version}.txt"
    if not prompt_file.exists():
        available = sorted(path.stem for path in PROMPT_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_version}. Available: {available}"
        )
    with prompt_file.open("r", encoding="utf-8") as file:
        return file.read().strip()


def build_discovery_messages(
    paragraph_text: str,
    current_schema: set[str],
    section_type: str,
    runtime_config: RuntimeConfig,
) -> list[dict[str, str]]:
    schema_str, schema_note = format_categorized_schema_for_prompt(
        current_schema,
        max_per_category=80,
        max_total=400,
    )
    section_instruction = ""
    if section_type == "TABLE":
        section_instruction = (
            "ADDITIONAL RULES FOR TABLE/SUPPLEMENTARY CONTENT:\n"
            "This text is from a data table or supplementary file.\n"
            "- Column HEADERS describing SAMPLE METADATA -> valid metadata attribute names\n"
            "- Cell VALUES (specific numbers, dates, coordinates, species names, location names etc.) -> EXCLUDE\n"
            "- File names, sheet names, table captions, footnotes -> EXCLUDE\n"
            "- If unsure whether something is a column header or a cell value -> EXCLUDE"
        )

    system_prompt = load_prompt_template(runtime_config.prompt_version)
    system_prompt = system_prompt.replace(
        SECTION_INSTRUCTION_PLACEHOLDER, section_instruction
    )
    system_prompt = system_prompt.replace(SCHEMA_NOTE_PLACEHOLDER, schema_note)
    system_prompt = system_prompt.replace(SCHEMA_STR_PLACEHOLDER, schema_str)
    payload = json.dumps({"text": paragraph_text}, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]
