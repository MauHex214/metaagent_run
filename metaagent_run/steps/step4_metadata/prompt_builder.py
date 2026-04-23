import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from .config import PROMPT_VERSION

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
MIXS_BLOCK_PLACEHOLDER = "__MIXS_BLOCK__"


@lru_cache(maxsize=8)
def load_prompt_template(prompt_version: str) -> str:
    prompt_file = PROMPT_DIR / f"{prompt_version}.txt"
    if not prompt_file.exists():
        available = sorted(path.stem for path in PROMPT_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_version}. Available: {available}"
        )
    with prompt_file.open("r", encoding="utf-8") as file:
        return file.read().strip()


def build_mixs_reference_block(mixs_standards: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for entry in mixs_standards:
        title = str(entry.get("Title", ""))
        slot = str(entry.get("Slot_Name", ""))
        desc = str(entry.get("Description", ""))
        if len(desc) > 150:
            desc = desc[:147] + "..."
        lines.append(f'- Slot: "{slot}"  Title: "{title}"  Desc: "{desc}"')
    return "\n".join(lines)


def build_mapping_system_prompt(
    mixs_standards: List[Dict[str, str]],
    prompt_version: str = PROMPT_VERSION,
) -> str:
    template = load_prompt_template(prompt_version)
    return template.replace(
        MIXS_BLOCK_PLACEHOLDER, build_mixs_reference_block(mixs_standards)
    )


def build_mapping_user_prompt(
    fields_batch: List[str],
    synonym_groups: Dict[str, List[str]],
) -> str:
    field_blocks: List[str] = []
    for field in fields_batch:
        norm = field.lower()
        aliases: List[str] = [field]
        for canonical, members in synonym_groups.items():
            members_lower = {member.lower() for member in members}
            if norm in members_lower or canonical.lower() == norm:
                seen = set()
                aliases = []
                for candidate in members:
                    candidate_norm = candidate.lower()
                    if candidate_norm not in seen:
                        aliases.append(candidate)
                        seen.add(candidate_norm)
                if norm not in seen:
                    aliases.insert(0, field)
                break

        field_blocks.append(
            'Canonical field key: "{}"\nAliases in literature: {}\nInterpret the environmental concept represented by this field group, then map it to the best MIxS term or UNMAPPED.'.format(
                field,
                json.dumps(aliases, ensure_ascii=False),
            )
        )

    return "## Field Groups to Map ({} groups)\n\n{}".format(
        len(fields_batch),
        "\n\n".join(field_blocks),
    )


def build_mapping_messages(
    fields_batch: List[str],
    mixs_standards: List[Dict[str, str]],
    synonym_groups: Dict[str, List[str]],
    prompt_version: str = PROMPT_VERSION,
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_mapping_system_prompt(mixs_standards, prompt_version),
        },
        {
            "role": "user",
            "content": build_mapping_user_prompt(fields_batch, synonym_groups),
        },
    ]
