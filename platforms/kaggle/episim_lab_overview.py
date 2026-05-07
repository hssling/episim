"""Minimal Kaggle-facing overview script for EPISIM."""

from episim.lab import list_designs, run_design


def main() -> None:
    print("EPISIM design catalog")
    for design in list_designs():
        print(f"- {design.key}: {design.title}")

    study = run_design("cross_sectional", seed_value=20260508, n=800)
    print("\nCross-sectional summary")
    for key, value in study.results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
