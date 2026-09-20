from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAHOO_ROOT = PROJECT_ROOT / "external" / "yahoo-budget-pacing"
sys.path.insert(0, str(YAHOO_ROOT))

import src.configuration as config
from src import system_generation_utils
from src.system.clock import Clock
from src.system.marketplace import Marketplace
from src.system.serving_system import ServingSystem

np.random.seed(config.campaign_generation_seed_value)
Clock.reset()

campaigns = system_generation_utils.generate_campaigns(1)
pacing_system = system_generation_utils.generate_pacing_system(config.pacing_algorithm)
serving_system = ServingSystem(
    pacing_system=pacing_system,
    tracked_campaigns=campaigns,
)
marketplace = Marketplace(
    serving_system=serving_system,
    auction_type=config.auction_type,
    traffic_mean_cos_wave=config.traffic_mean_cos_wave,
)

expected_auctions = marketplace._calculate_current_mean_num_of_auctions()
auctions = marketplace._generate_auctions()
marketplace._run_auctions(auctions)
serving_system.end_iteration()

campaign = campaigns[0]
print(f"minute={Clock.minute_in_day()}")
print(f"traffic_mean={expected_auctions:.6f}")
print(f"actual_auctions={len(auctions)}")
print(f"campaign_id={campaign.id}")
print(f"daily_budget={campaign.daily_budget:.6f}")
print(f"wins={campaign.num_auctions_won_today()}")
print(f"spend={campaign.spent_today():.6f}")
print(f"remaining_budget={campaign.daily_budget - campaign.spent_today():.6f}")

Clock.advance()
print(f"next_minute={Clock.minute_in_day()}")
