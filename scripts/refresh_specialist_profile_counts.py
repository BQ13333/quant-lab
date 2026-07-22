#!/usr/bin/env python3
"""Synchronize displayed specialist counts with canonical primary/secondary indexes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="PhD_Math_Agent_Library")
    args = parser.parse_args()
    library = Path(args.library).expanduser().resolve()
    folders = sorted((library / "specialists").glob("[0-9][0-9]_*"))
    if len(folders) != 20:
        raise SystemExit(f"Expected 20 specialist folders, found {len(folders)}")

    counts: dict[str, tuple[int, int]] = {}
    changed_profiles: list[str] = []
    for folder in folders:
        agent = folder.name[3:]
        primary = len([row for row in read_csv(folder / "primary_index.csv") if row["retrieval_status"].startswith("active")])
        secondary = len([row for row in read_csv(folder / "secondary_index.csv") if row["retrieval_status"].startswith("active")])
        counts[agent] = (primary, secondary)
        profile_path = folder / "SPECIALIST_PROFILE.md"
        before = profile_path.read_text(encoding="utf-8")
        after, replacements = re.subn(
            r"(?m)^Active primary papers: \d+[ \t]*\r?\n"
            r"Secondary-routed papers: \d+[ \t]*(?:\r?\n)?(?:\r?\n)?",
            f"Active primary papers: {primary}  \nSecondary-routed papers: {secondary}\n\n",
            before,
            count=1,
        )
        if replacements != 1:
            raise SystemExit(f"Count lines missing or duplicated in {profile_path}")
        if after != before:
            profile_path.write_text(after, encoding="utf-8")
            changed_profiles.append(agent)

    roster_path = library / "indexes/agent_roster.csv"
    roster = read_csv(roster_path)
    fields = list(roster[0])
    if {row["agent_id"] for row in roster} != set(counts):
        raise SystemExit("Roster and specialist folders do not match")
    for row in roster:
        primary, secondary = counts[row["agent_id"]]
        row["primary_papers"] = str(primary)
        row["secondary_routes"] = str(secondary)
    write_csv(roster_path, fields, roster)

    audit = {
        "status": "pass",
        "specialists": len(folders),
        "changed_profiles": changed_profiles,
        "active_primary_total": sum(primary for primary, _ in counts.values()),
        "secondary_route_total": sum(secondary for _, secondary in counts.values()),
    }
    (library / "indexes/profile_count_refresh_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
