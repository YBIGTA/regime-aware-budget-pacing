from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(1, str(PROJECT_ROOT / "external" / "yahoo-budget-pacing"))

from controller.fixed_pi import FixedPI
import src.configuration as config
from src import constants
from src import system_generation_utils
from src.system.clock import Clock
from src.system.marketplace import Marketplace
from src.system.serving_system import ServingSystem


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--campaigns", required=True, type=int)
    parser.add_argument("--minutes", required=True, type=int)
    parser.add_argument("--kp", required=True, type=float)
    parser.add_argument("--ki", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.campaigns < 1 or args.minutes < 1:
        raise ValueError("campaigns and minutes must be positive")

    np.random.seed(config.campaign_generation_seed_value)
    Clock.reset()

    campaigns = system_generation_utils.generate_campaigns(args.campaigns)
    pacing_system = FixedPI(kp=args.kp, ki=args.ki)
    serving_system = ServingSystem(
        pacing_system=pacing_system,
        tracked_campaigns=campaigns,
    )
    marketplace = Marketplace(
        serving_system=serving_system,
        auction_type=config.auction_type,
        traffic_mean_cos_wave=config.traffic_mean_cos_wave,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_id",
        "environment_seed",
        "minute",
        "traffic_mean",
        "actual_auctions",
        "campaign_id",
        "daily_budget",
        "target_spend_cumulative",
        "actual_spend_cumulative",
        "spend_error",
        "spend_this_minute",
        "wins_this_minute",
        "remaining_budget",
        "pacing_signal_used",
        "pacing_signal_next",
        "integral_error",
    ]

    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(args.minutes):
            minute = Clock.minute_in_day()
            traffic_mean = marketplace._calculate_current_mean_num_of_auctions()
            auctions = marketplace._generate_auctions()

            spend_before = {
                campaign.id: campaign.spent_today()
                for campaign in campaigns
            }
            wins_before = {
                campaign.id: campaign.num_auctions_won_today()
                for campaign in campaigns
            }
            signal_before = {
                campaign.id: pacing_system.get_pacing_signal(campaign.id)
                for campaign in campaigns
            }

            marketplace._run_auctions(auctions)
            serving_system.end_iteration()

            for campaign in campaigns:
                state = pacing_system.get_state(campaign.id)
                actual_spend = campaign.spent_today()
                target_spend = (
                    campaign.daily_budget
                    * (minute + 1)
                    / constants.num_minutes_in_day
                )

                writer.writerow(
                    {
                        "scenario_id": args.scenario_id,
                        "environment_seed": config.campaign_generation_seed_value,
                        "minute": minute,
                        "traffic_mean": traffic_mean,
                        "actual_auctions": len(auctions),
                        "campaign_id": campaign.id,
                        "daily_budget": campaign.daily_budget,
                        "target_spend_cumulative": target_spend,
                        "actual_spend_cumulative": actual_spend,
                        "spend_error": target_spend - actual_spend,
                        "spend_this_minute": actual_spend - spend_before[campaign.id],
                        "wins_this_minute": (
                            campaign.num_auctions_won_today()
                            - wins_before[campaign.id]
                        ),
                        "remaining_budget": campaign.daily_budget - actual_spend,
                        "pacing_signal_used": signal_before[campaign.id],
                        "pacing_signal_next": state.pacing_signal,
                        "integral_error": state.integral_error,
                    }
                )

            Clock.advance()

    print(f"trace_path={args.output}")
    print(f"rows={args.campaigns * args.minutes}")


if __name__ == "__main__":
    main()