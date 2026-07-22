from __future__ import annotations

from quant_lab.research.asean import run_asean_research


def main() -> None:
    result = run_asean_research()

    print("Run id:", result.run_id)
    print("Artifacts:", result.artifact_dir.resolve())
    print("\nLeaderboard:")
    print(result.leaderboard.to_string(index=False) if not result.leaderboard.empty else "No leaderboard produced")
    print("\nCoverage summary:")
    print(result.coverage.groupby(["download_tier", "country"])["rows"].agg(["count", "sum"]).to_string())
    print("\nObsidian notes:")
    for path in result.note_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
