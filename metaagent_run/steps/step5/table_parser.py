"""Layer 1 — 结构化表格确定性解析。

零 LLM 调用，通过表头列名分类 + 逐行解析提取 AtomicRelation。
列分类使用动态 keyword 集（从上游 MIxS target fields + synonyms 构建）。
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .schemas import AtomicRelation

from metaagent_run.core import INSDC_ACCESSION_RE as ACC_PATTERN

# ── 列角色关键词（accession / label / exclude 为静态通用集） ──
_ACC_COL_KEYWORDS = {
    "run", "bioproject", "biosample", "accession", "sra", "srr", "err",
    "drr", "genbank", "ddbj", "embl", "ena", "experiment", "erx", "srx",
    "sample_accession", "genome_id",
}
_LABEL_COL_KEYWORDS = {
    "sample", "station", "site", "label", "name", "dataset", "id",
    "identifier", "library_id", "genome", "bin", "mag", "clone",
}
_EXCLUDE_COL_KEYWORDS = {
    "instrument", "platform", "avgspotlen", "spots", "bases", "layout",
    "library_strategy", "library_source", "library_selection",
    "assembly", "contig", "scaffold", "n50", "completeness", "contamination",
    "gc", "gene", "cds", "rrna", "trna", "coding_density", "heterogeneity",
    "primer", "reads", "taxonomy", "otu", "phylum", "class", "order",
    "family", "genus", "species", "annotation", "protein", "ec", "ko",
}


# ═══════════════════════════════════════════════════════════
#  normalize_table_text — 标准化分隔符（移植自 step5）
# ═══════════════════════════════════════════════════════════

def normalize_table_text(text: str) -> str:
    """标准化表格文本的分隔符。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.count("\n") > 200:
        return text
    original_text = text
    avg_line_len = len(text) / (text.count("\n") + 1)
    if avg_line_len > 500:
        text = re.sub(r"\t[ ]+\t(?=\w)", "\n", text)
    if text.count("\n") < original_text.count("\n") * 1.5 + 5:
        text = re.sub(r"[ ]{4,}", "\n", text)
    if text.count("\n") > original_text.count("\n") + 2:
        text = re.sub(r"[ ]{2,3}", "\t", text)
    return text


# ═══════════════════════════════════════════════════════════
#  列名归一化与分类
# ═══════════════════════════════════════════════════════════

def _normalize_col_name(raw: str) -> str:
    """列名归一化：去单位、去括号、转下划线小写。"""
    s = raw.strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = s.strip("_").lower()
    return s


def classify_column(
    raw_name: str,
    target_field_names_lower: Set[str],
    metadata_keywords: Set[str],
) -> Tuple[str, str]:
    """将一个列名分类为 accession / label / metadata / exclude。

    优先级：
    1. accession 列（关键词匹配）
    2. target_fields 精确匹配
    3. 排除列
    4. metadata 动态关键词（从 MIxS synonyms 构建）
    5. label 列
    6. 默认 exclude（不再默认归 metadata）
    """
    norm = _normalize_col_name(raw_name)
    display = raw_name.strip()
    tokens = set(norm.split("_"))

    # 优先级 1: accession 列
    if tokens & _ACC_COL_KEYWORDS:
        return "accession", display

    # 优先级 2: target_fields 精确匹配
    if norm in target_field_names_lower:
        return "metadata", display

    # 优先级 3: 排除列
    if tokens & _EXCLUDE_COL_KEYWORDS:
        return "exclude", display

    # 优先级 4: metadata 动态关键词（MIxS synonyms）
    if tokens & metadata_keywords:
        return "metadata", display

    # 优先级 5: label 列
    if tokens & _LABEL_COL_KEYWORDS:
        return "label", display

    # 默认排除（宁可少提取，避免噪音）
    return "exclude", display


# ═══════════════════════════════════════════════════════════
#  StructuredTableParser
# ═══════════════════════════════════════════════════════════

class StructuredTableParser:
    """结构化表格解析器。"""

    def is_structured_table(
        self,
        section: Dict[str, Any],
        min_tabs: int = 5,
        min_rows: int = 3,
    ) -> bool:
        """判断一个 section 是否是可解析的结构化表格。"""
        text = normalize_table_text(section.get("text", ""))
        if not text:
            return False
        tab_count = text.count("\t")
        if tab_count < min_tabs:
            return False
        lines = [l for l in text.split("\n") if l.strip() and "\t" in l]
        return len(lines) >= min_rows

    def count_columns(self, section: Dict[str, Any]) -> int:
        """获取表格的最大列数。"""
        text = normalize_table_text(section.get("text", ""))
        lines = [l for l in text.split("\n") if l.strip() and "\t" in l]
        if not lines:
            return 0
        return max(len(l.split("\t")) for l in lines[:5])

    def is_transposed_table(
        self, section: Dict[str, Any], metadata_keywords: Set[str],
    ) -> bool:
        """Detect if a table has field names in the first column (transposed layout).

        Uses dynamic metadata_keywords instead of static set.
        Heuristic: if >30% of first-column values match metadata keywords.
        """
        text = normalize_table_text(section.get("text", ""))
        lines = [l for l in text.split("\n") if l.strip() and "\t" in l]
        if len(lines) < 3:
            return False
        first_col_values = []
        for line in lines[2:]:
            cols = line.split("\t")
            if cols:
                first_col_values.append(cols[0].strip())
        if not first_col_values:
            return False
        meta_hits = sum(1 for v in first_col_values
                        if set(_normalize_col_name(v).split("_")) & metadata_keywords)
        return meta_hits / len(first_col_values) > 0.3

    def extract(
        self,
        section: Dict[str, Any],
        target_field_names: List[str],
        metadata_keywords: Set[str],
        pmid: str = "",
        verified_accessions: Optional[Set[str]] = None,
    ) -> List[AtomicRelation]:
        """解析普通结构化表格。"""
        text = normalize_table_text(section.get("text", ""))
        pmid = pmid or str(section.get("pmid", ""))
        section_key = "%s::%s" % (section.get("section_type", ""), section.get("index", 0))
        verified = verified_accessions or set()

        target_lower = {f.strip().lower() for f in target_field_names}

        lines = self._split_to_lines(text)
        if len(lines) < 2:
            return []

        header_line, data_lines = self._detect_header_and_data(lines)
        if not header_line:
            return []

        cols = header_line.split("\t")
        col_roles = {}
        for i, raw in enumerate(cols):
            if not raw.strip():
                continue
            col_roles[i] = classify_column(raw, target_lower, metadata_keywords)

        acc_indices = [i for i, (r, _) in col_roles.items() if r == "accession"]
        label_indices = [i for i, (r, _) in col_roles.items() if r == "label"]
        meta_indices = [i for i, (r, _) in col_roles.items() if r == "metadata"]

        if not acc_indices and not label_indices and not meta_indices:
            return []

        relations: List[AtomicRelation] = []

        for row_text in data_lines:
            cells = row_text.split("\t")

            accessions = self._extract_accessions_from_cells(cells, acc_indices, verified)
            labels = self._extract_labels_from_cells(cells, label_indices)
            metadata = self._extract_metadata_from_cells(cells, meta_indices, col_roles)


            if accessions and labels and metadata:
                for acc in accessions:
                    for lbl in labels:
                        relations.append(AtomicRelation(
                            pmid=pmid, section_key=section_key,
                            relation_type="accession_label_metadata",
                            accession=acc, label=lbl, metadata=metadata,
                            source="table_parse",
                        ))

            for acc in accessions:
                for lbl in labels:
                    relations.append(AtomicRelation(
                        pmid=pmid, section_key=section_key,
                        relation_type="accession_label",
                        accession=acc, label=lbl,
                        source="table_parse",
                    ))

            if metadata:
                for acc in accessions:
                    relations.append(AtomicRelation(
                        pmid=pmid, section_key=section_key,
                        relation_type="accession_metadata",
                        accession=acc, metadata=metadata,
                        source="table_parse",
                    ))

            if metadata:
                for lbl in labels:
                    relations.append(AtomicRelation(
                        pmid=pmid, section_key=section_key,
                        relation_type="label_metadata",
                        label=lbl, metadata=metadata,
                        source="table_parse",
                    ))

        return relations

    def extract_transposed(
        self,
        section: Dict[str, Any],
        target_field_names: List[str],
        metadata_keywords: Set[str],
        pmid: str = "",
        verified_accessions: Optional[Set[str]] = None,
    ) -> List[AtomicRelation]:
        """Parse a transposed table (fields in rows, samples in columns).

        Filters single-column rows (titles, section headers), handles
        multi-level headers, transposes, then delegates to extract().
        """
        text = normalize_table_text(section.get("text", ""))
        lines = [l for l in text.split("\n") if l.strip() and "\t" in l]
        if len(lines) < 3:
            return []

        # Filter single-column rows (title lines, section headers)
        col_counts = [len(l.split("\t")) for l in lines]
        expected_cols = max(col_counts)
        multi_col_lines = [l for l, c in zip(lines, col_counts)
                           if c >= expected_cols - 1]
        if len(multi_col_lines) < 3:
            return []

        # Build matrix from filtered lines
        matrix = []
        max_cols = 0
        for line in multi_col_lines:
            cols = [c.strip() for c in line.split("\t")]
            matrix.append(cols)
            max_cols = max(max_cols, len(cols))

        for row in matrix:
            while len(row) < max_cols:
                row.append("")

        # Multi-level header detection
        row0_non_empty = [c for c in matrix[0][1:] if c]
        row1_non_empty = [c for c in matrix[1][1:] if c] if len(matrix) > 1 else []

        has_multi_header = False
        if len(row0_non_empty) < len(matrix[0]) - 1 and row1_non_empty:
            has_multi_header = True
        elif row0_non_empty and all(not c.replace('.', '').replace('-', '').isdigit() for c in row0_non_empty):
            if row1_non_empty and all(not c.replace('.', '').replace('-', '').isdigit() for c in row1_non_empty):
                has_multi_header = True

        if has_multi_header and len(matrix) >= 3:
            filled_row0 = []
            last_val = ""
            for c in matrix[0]:
                if c:
                    last_val = c
                filled_row0.append(last_val)

            composite_headers = [matrix[1][0]]
            for i in range(1, max_cols):
                h0 = filled_row0[i] if i < len(filled_row0) else ""
                h1 = matrix[1][i] if i < len(matrix[1]) else ""
                composite_headers.append("{} {}".format(h0, h1).strip())
            data_start = 2
        else:
            composite_headers = matrix[0]
            data_start = 1

        # Transpose
        field_names = [matrix[r][0] for r in range(data_start, len(matrix))]
        transposed_header = "Sample\t" + "\t".join(field_names)
        transposed_lines = [transposed_header]

        for col_idx in range(1, max_cols):
            sample_label = composite_headers[col_idx] if col_idx < len(composite_headers) else ""
            values = [matrix[r][col_idx] if col_idx < len(matrix[r]) else ""
                      for r in range(data_start, len(matrix))]
            row_line = sample_label + "\t" + "\t".join(values)
            transposed_lines.append(row_line)

        transposed_text = "\n".join(transposed_lines)
        synthetic_section = dict(section)
        synthetic_section["text"] = transposed_text

        return self.extract(
            synthetic_section, target_field_names, metadata_keywords,
            pmid, verified_accessions,
        )

    # ── 内部方法 ──────────────────────────────────────────

    def _split_to_lines(self, text: str) -> List[str]:
        return [l for l in text.split("\n") if l.strip()]

    def _detect_header_and_data(self, lines: List[str]) -> Tuple[Optional[str], List[str]]:
        header_idx = None
        for i, line in enumerate(lines):
            if line.count("\t") >= 1:
                header_idx = i
                break
        if header_idx is None:
            return None, []
        header = lines[header_idx]
        data = [l for l in lines[header_idx + 1:] if "\t" in l]
        return header, data

    def _extract_accessions_from_cells(
        self, cells: List[str], acc_indices: List[int],
        verified: Set[str],
    ) -> List[str]:
        result = []
        for i in acc_indices:
            if i >= len(cells):
                continue
            val = cells[i].strip()
            if not val:
                continue
            matches = ACC_PATTERN.findall(val)
            if matches:
                result.extend(matches)
            elif val in verified:
                result.append(val)
            elif val and not val.startswith(("NA", "na", "-", "none", "None")):
                result.append(val)
        return list(dict.fromkeys(result))

    def _extract_labels_from_cells(
        self, cells: List[str], label_indices: List[int],
    ) -> List[str]:
        result = []
        for i in label_indices:
            if i >= len(cells):
                continue
            val = cells[i].strip()
            if val and val.lower() not in ("na", "none", "-", ""):
                if ACC_PATTERN.fullmatch(val):
                    continue
                result.append(val)
        return list(dict.fromkeys(result))

    def _extract_metadata_from_cells(
        self,
        cells: List[str],
        meta_indices: List[int],
        col_roles: Dict[int, Tuple[str, str]],
    ) -> List[str]:
        result = []
        for i in meta_indices:
            if i >= len(cells):
                continue
            val = cells[i].strip()
            if not val or val.lower() in ("na", "none", "-", "nd", "n/a", ""):
                continue
            display_name = col_roles[i][1]
            result.append("%s: %s" % (display_name, val))
        return result
