"""Unified INSDC accession regex patterns."""
import re
from typing import Final

INSDC_ACCESSION_RE: Final[re.Pattern] = re.compile(
    r"\b("
    r"PRJ[NEDB][AB]\d+"          # BioProject (NCBI/EBI/DDBJ, A/B)
    r"|SAM[EN]A?\d+|SAMD\d+"    # BioSample (NCBI SAMN, EBI SAME/SAMEA, DDBJ SAMD)
    r"|[SED]RX\d+"               # Experiment (SRX/ERX/DRX)
    r"|[SED]RR\d+"               # Run (SRR/ERR/DRR)
    r"|[SED]RP\d+"               # Study (SRP/ERP/DRP)
    r"|[SED]RS\d+"               # Sample SRA-level (SRS/ERS/DRS)
    r")\b"
)
