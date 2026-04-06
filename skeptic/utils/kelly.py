"""Kelly-style stake sizing: scale from fixed_usdc (minimum) up to kelly_max_usdc."""

KELLY_MAX_USDC = 3.00          # absolute max stake regardless of edge
MOMENTUM_EDGE_THRESHOLD = 0.10  # momentum: edge at which scaling begins (max_pm_price - pm_ask)


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
