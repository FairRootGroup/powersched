#!/usr/bin/env python3
"""Test that price_history correctly maintains a 24-hour rolling window."""

from src.prices_deterministic import Prices

def test_price_history_bounded():
    """Verify price_history is bounded to HISTORY_WINDOW."""
    test_prices = list(range(1, 101))
    prices = Prices(external_prices=test_prices)

    print(f"HISTORY_WINDOW = {Prices.HISTORY_WINDOW}")
    print(f"Testing that price_history stays bounded during episode\n")

    # Simulate a long episode (336 hours like environment)
    prices.reset(start_index=0)

    steps_taken = 0
    for target_step in [0, 10, 24, 50, 100, 200, 336]:
        # Advance to target_step
        while steps_taken < target_step:
            prices.get_next_price()
            steps_taken += 1

        history_len = len(prices.price_history)
        expected_len = min(target_step, Prices.HISTORY_WINDOW)

        print(f"After {target_step:3d} steps: price_history length = {history_len:2d} (expected: {expected_len:2d})", end="")

        if history_len == expected_len:
            print(" ✓")
        else:
            print(f" ✗ FAILED")
            return False

    print(f"\n✓ price_history correctly bounded to {Prices.HISTORY_WINDOW} entries")
    return True


def test_rolling_window_behavior():
    """Test that old prices are evicted as new ones arrive."""
    test_prices = list(range(1, 101))
    prices = Prices(external_prices=test_prices)

    print("\n=== Testing Rolling Window Behavior ===")

    prices.reset(start_index=0)

    # Fill the history window (24 prices)
    for _ in range(24):
        prices.get_next_price()

    # Record the first entry
    first_24_prices = list(prices.price_history)
    print(f"First 24 prices: {first_24_prices[:5]}...{first_24_prices[-3:]}")

    # Add one more price - should evict the oldest
    price_25 = prices.get_next_price()
    current_history = list(prices.price_history)

    print(f"After adding price #{25} ({price_25}):")
    print(f"  Current history: {current_history[:5]}...{current_history[-3:]}")
    print(f"  Length: {len(current_history)}")

    # Check that oldest was evicted
    if len(current_history) == 24:
        print("  ✓ Length remains at 24")
    else:
        print(f"  ✗ Length is {len(current_history)}, expected 24")
        return False

    # The first element should have been evicted
    if current_history[0] == first_24_prices[1]:
        print("  ✓ Oldest price was evicted")
    else:
        print(f"  ✗ Expected first element to be {first_24_prices[1]}, got {current_history[0]}")
        return False

    # The newest element should be at the end
    if current_history[-1] == price_25:
        print("  ✓ Newest price is at the end")
    else:
        print(f"  ✗ Expected last element to be {price_25}, got {current_history[-1]}")
        return False

    return True


def test_get_price_context_uses_window():
    """Test that get_price_context uses rolling window, not entire episode."""
    test_prices = list(range(1, 101))
    prices = Prices(external_prices=test_prices)

    print("\n=== Testing get_price_context() ===")

    prices.reset(start_index=0)

    # Advance 50 steps into episode
    for _ in range(50):
        prices.get_next_price()

    history_avg, future_avg = prices.get_price_context()

    # Manual calculation of what history_avg should be
    # Should be average of last 24 prices only
    recent_prices = list(prices.price_history)
    expected_avg = sum(recent_prices) / len(recent_prices)

    print(f"After 50 steps:")
    print(f"  price_history length: {len(prices.price_history)}")
    print(f"  Recent prices (last 5): {recent_prices[-5:]}")
    print(f"  history_avg from get_price_context: {history_avg:.2f}")
    print(f"  Expected (avg of last 24): {expected_avg:.2f}")

    if abs(history_avg - expected_avg) < 0.01:
        print("  ✓ history_avg uses rolling 24-hour window")
        return True
    else:
        print("  ✗ history_avg calculation incorrect")
        return False


if __name__ == "__main__":
    all_pass = True
    all_pass &= test_price_history_bounded()
    all_pass &= test_rolling_window_behavior()
    all_pass &= test_get_price_context_uses_window()

    print("\n" + "="*50)
    if all_pass:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
