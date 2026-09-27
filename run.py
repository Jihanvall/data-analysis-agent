import argparse
import sys

from src.pipeline import run_pipeline
from src.report import generate_report, save_report


def main():
    parser = argparse.ArgumentParser(
        description="Run the autonomous data analysis pipeline on a CSV file."
    )
    parser.add_argument("--file", required=True, help="Path to the input CSV file")
    parser.add_argument("--target", help="Target column name (omit for clustering)")
    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression", "clustering"],
        help="Type of ML task",
    )
    parser.add_argument(
        "--name",
        default="dataset",
        help="Dataset name used in the report filename (default: dataset)",
    )
    args = parser.parse_args()

    if args.task != "clustering" and not args.target:
        parser.error("--target is required unless --task is clustering")

    print(f"Running pipeline on {args.file} (task: {args.task})...")

    try:
        result = run_pipeline(args.file, target=args.target, task_type=args.task)
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Best model: {result['model_name']}")
    print(f"Test metrics: {result['test_metrics']}")

    report_text = generate_report(result, args.name)
    path = save_report(report_text, args.name)
    print(f"Report saved to: {path}")


if __name__ == "__main__":
    main()