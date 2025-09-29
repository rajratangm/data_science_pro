# data_science_pro/pipeline.py
from data_science_pro.cycle.controller import DataScienceController


class DataSciencePro:
    """Main entry point for the package."""

    def __init__(self, api_key: str):
        self.controller = DataScienceController(api_key)

    def run(self, csv_path: str) -> str:
        """Run the workflow for a given CSV path."""
        return self.controller.run(csv_path)


# For backward compatibility
def run_pipeline(csv_path: str, api_key: str = "") -> str:
    dsp = DataSciencePro(api_key=api_key)
    return dsp.run(csv_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data Science Pro - CSV analysis pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--api_key", required=False, default="", help="OpenAI API key")
    parser.add_argument("--goal", required=False, default="Automated EDA, preprocessing, training and evaluation", help="User goal or instruction for the agent")
    parser.add_argument("--target", required=False, default=None, help="Target column name (optional)")
    args = parser.parse_args()

    dsp = DataSciencePro(api_key=args.api_key)
    # Pass goal and optional target into controller
    dsp.controller.goal = args.goal
    if args.target:
        dsp.controller.user_target = args.target
    report = dsp.run(args.data)
    # Print final report if available
    if isinstance(report, dict) and "report" in report:
        print(report["report"])
    else:
        print(report)


if __name__ == "__main__":
    main()
