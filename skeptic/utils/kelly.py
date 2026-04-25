"""Kelly-style stake sizing: scale from fixed_usdc (minimum) up to kelly_max_usdc."""

KELLY_MAX_USDC = 2.00          # absolute max stake regardless of edge
MOMENTUM_EDGE_THRESHOLD = 0.10  # momentum: edge at which scaling begins (max_pm_price - pm_ask)

# Imbalance multiplier constants
# imbalance = bid_volume / (bid_volume + ask_volume)
# ask-heavy (low imbalance) → more supply, better fills, higher edge → scale up
# bid-heavy (high imbalance) → more demand pressure against us → scale down
IMBALANCE_NEUTRAL   = 0.51   # midpoint of balanced tercile (0.50–0.52)
IMBALANCE_SENSITIVITY = 5.0  # multiplier range: ±(sensitivity * delta)
IMBALANCE_MIN_MULT  = 0.5    # floor: never go below 50% of Kelly stake
IMBALANCE_MAX_MULT  = 1.5    # ceiling: never exceed 150% of Kelly stake


def imbalance_kelly_multiplier(imbalance: float) -> float:
    """
    Returns a stake multiplier based on real-time orderbook imbalance.

    imbalance = bid_volume / (bid_volume + ask_volume) for the traded token.

    ask-heavy (imbalance < 0.50) → multiplier > 1.0  (scale up)
    balanced  (imbalance ≈ 0.51) → multiplier ≈ 1.0  (no change)
    bid-heavy (imbalance > 0.52) → multiplier < 1.0  (scale down)

    Linear around IMBALANCE_NEUTRAL, clamped to [IMBALANCE_MIN_MULT, IMBALANCE_MAX_MULT].

    Examples (sensitivity=5.0, neutral=0.51):
      imbalance=0.40 → 1.50   imbalance=0.51 → 1.00   imbalance=0.62 → 0.50
    """
    mult = 1.0 + (IMBALANCE_NEUTRAL - imbalance) * IMBALANCE_SENSITIVITY
    return max(IMBALANCE_MIN_MULT, min(IMBALANCE_MAX_MULT, mult))


def kelly_usdc(
    edge: float,
    edge_threshold: float,
    fixed_usdc: float,
    kelly_max_usdc: float = KELLY_MAX_USDC,
) -> float:
    """
    Scale stake proportionally to edge strength. fixed_usdc is the floor.

    At edge == edge_threshold  →  fixed_usdc  (minimum stake)
    At edge > edge_threshold   →  scales up, capped at kelly_max_usdc
    At edge < edge_threshold   →  fixed_usdc  (no scaling down)

    Formula:
        multiplier = edge / edge_threshold
        stake = fixed_usdc × multiplier
        stake = clamp(stake, fixed_usdc, kelly_max_usdc)

    Args:
        edge:            actual edge at trigger time
                         momentum: max_pm_price - pm_ask
                         model:    predicted_win - pm_ask
        edge_threshold:  edge at which stake equals fixed_usdc (scale origin)
                         momentum: use MOMENTUM_EDGE_THRESHOLD (0.10)
                         model:    use the model edge_threshold from config
        fixed_usdc:      minimum stake (returned when edge <= edge_threshold)
        kelly_max_usdc:  maximum stake regardless of edge
    """
    if edge <= 0 or edge_threshold <= 0:
        return fixed_usdc
    multiplier = edge / edge_threshold
    stake = fixed_usdc * multiplier
    return max(fixed_usdc, min(kelly_max_usdc, round(stake, 4)))
