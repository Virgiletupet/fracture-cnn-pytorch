"""Step 1. Data quality audit: decoding, duplicates, label conflicts, shortcuts."""
from fracture.audit import run_audit

if __name__ == "__main__":
    run_audit()
