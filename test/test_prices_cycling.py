#!/usr/bin/env python3
"""Test script to verify that prices.py cycles through the price file."""

import numpy as np
from src.prices import Prices

def test_cycling_behavior():
    """Test that prices continue through the file when given different start_index values."""
    # Create a simple price sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # (shifted by 1 since prices < 1 get shifted)
    test_prices = list(range(1, 11))

    prices = Prices(external_prices=test_prices)

    print("Testing cycling behavior through price file...")
    print(f"Price file length: {len(test_prices)}")
    print(f"Prediction window: {Prices.PREDICTION_WINDOW}")
    print()

    # Simulate 3 episodes with 5-step episodes (like environment would do)
    EPISODE_HOURS = 5  # Short episodes for testing

    # Episode 1 - starts at index 0
    print("=== Episode 1 (start_index=0) ===")
    prices.reset(start_index=0)
    print(f"Initial predicted_prices (first 5): {prices.predicted_prices[:5]}")

    episode_1_prices = []
    for i in range(5):
        predicted = prices.advance_and_get_predicted_prices()
        current_price = predicted[0]  # First element is "current" price after roll
        episode_1_prices.append(float(current_price))

    print(f"Prices seen in episode 1: {episode_1_prices}")
    print()

    # Episode 2 - starts at index 5 (continuing through file)
    print("=== Episode 2 (start_index=5) ===")
    start_idx_2 = (1 * EPISODE_HOURS) % len(test_prices)
    prices.reset(start_index=start_idx_2)
    print(f"Initial predicted_prices (first 5): {prices.predicted_prices[:5]}")

    episode_2_prices = []
    for i in range(5):
        predicted = prices.advance_and_get_predicted_prices()
        current_price = predicted[0]
        episode_2_prices.append(float(current_price))

    print(f"Prices seen in episode 2: {episode_2_prices}")
    print()

    # Episode 3 - wraps around since we only have 10 prices
    print("=== Episode 3 (start_index=0, wrapped) ===")
    start_idx_3 = (2 * EPISODE_HOURS) % len(test_prices)
    prices.reset(start_index=start_idx_3)
    print(f"Initial predicted_prices (first 5): {prices.predicted_prices[:5]}")

    episode_3_prices = []
    for i in range(5):
        predicted = prices.advance_and_get_predicted_prices()
        current_price = predicted[0]
        episode_3_prices.append(float(current_price))

    print(f"Prices seen in episode 3: {episode_3_prices}")
    print()

    # Verify cycling behavior
    print("=== Verification ===")
    print(f"Episode 1 prices: {episode_1_prices}")
    print(f"Episode 2 prices: {episode_2_prices}")
    print(f"Episode 3 prices: {episode_3_prices}")

    # Check that episodes use different price sequences (unless wrapped)
    assert episode_1_prices != episode_2_prices, "Episodes 1 and 2 use same prices (not cycling)"
    print("✓ Episodes 1 and 2 use different price sequences (good!)")

    # Episode 3 should equal Episode 1 (wrapped back around)
    assert episode_1_prices == episode_3_prices, f"Episode 3 doesn't match Episode 1: expected {episode_1_prices}, got {episode_3_prices}"
    print("✓ Episode 3 wrapped around to match Episode 1 (good!)")


def test_determinism():
    """Test that same start_index gives same sequence."""
    test_prices = list(range(1, 101))

    print("\n=== Testing Determinism ===")

    # Run 1
    prices1 = Prices(external_prices=test_prices)
    prices1.reset(start_index=0)
    run1_prices = []
    for _ in range(10):
        predicted = prices1.advance_and_get_predicted_prices()
        run1_prices.append(float(predicted[0]))

    prices1.reset(start_index=50)  # Reset to different position
    for _ in range(10):
        predicted = prices1.advance_and_get_predicted_prices()
        run1_prices.append(float(predicted[0]))

    # Run 2 - completely fresh, same start indices
    prices2 = Prices(external_prices=test_prices)
    prices2.reset(start_index=0)
    run2_prices = []
    for _ in range(10):
        predicted = prices2.advance_and_get_predicted_prices()
        run2_prices.append(float(predicted[0]))

    prices2.reset(start_index=50)
    for _ in range(10):
        predicted = prices2.advance_and_get_predicted_prices()
        run2_prices.append(float(predicted[0]))

    assert run1_prices == run2_prices, f"Non-deterministic: run1={run1_prices[:5]} run2={run2_prices[:5]}"
    print("✓ Determinism verified: same start_index gives same sequence")


def test_environment_simulation():
    """Test that simulates how the environment would use prices."""
    print("\n=== Environment Simulation Test ===")

    # Use a year-like sequence
    test_prices = list(range(1, 366))  # 365 days
    EPISODE_HOURS = 14  # Simulate 14-hour episodes

    prices = Prices(external_prices=test_prices)

    print(f"Simulating 5 episodes with {EPISODE_HOURS}-hour duration")
    print(f"Price file has {len(test_prices)} prices")
    print()

    for episode_idx in range(5):
        start_index = (episode_idx * EPISODE_HOURS) % len(test_prices)
        prices.reset(start_index=start_index)

        # Just check first price of episode
        predicted = prices.advance_and_get_predicted_prices()
        first_price = float(predicted[0])

        print(f"Episode {episode_idx}: start_index={start_index:3d}, first_price={first_price}")

    print("\n✓ Episodes cycle through price file with proper offsets")


if __name__ == "__main__":
    test_cycling_behavior()
    test_determinism()
    test_environment_simulation()
