from mlx_quant_fidelity.costs import kv_bytes_per_token


def test_kv_bytes_per_token_hand_value():
    # elements/token = 2*K,V * 2 layers * 4 kv-heads * 64 head_dim = 1024
    # bytes/element  = 4/8 + 4/64 = 0.5 + 0.0625 = 0.5625
    # total          = 1024 * 0.5625 = 576
    assert kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=64) == 576


def test_kv_bytes_smaller_group_costs_more():
    big = kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=64)
    small = kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=32)
    assert small > big  # the group scale/bias overhead must be visible to the Pareto


def test_kv_bytes_higher_bits_costs_more():
    four = kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=4, group_size=64)
    eight = kv_bytes_per_token(n_layers=2, n_kv_heads=4, head_dim=64, bits=8, group_size=64)
    assert eight > four
