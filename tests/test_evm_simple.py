"""Simple EVM test to verify BERCalculator.calculate_evm works correctly."""

import numpy as np

from physics_nlo_eval.metrics.ber_evm import BERCalculator


def test_evm_basic():
    """Test EVM with simple known inputs."""
    print("=" * 60)
    print("EVM Basic Test")
    print("=" * 60)

    ber_calc = BERCalculator(ignore_headers=0)

    # QPSK constellation: indices 0,1,2,3 -> complex symbols
    # Standard QPSK: (1+j), (-1+j), (-1-j), (1-j) normalized
    constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128) / np.sqrt(2)

    print(f"Constellation (normalized): {constellation}")

    # Test 1: Perfect reception (rx == ideal)
    print("\n--- Test 1: Perfect reception ---")
    rx_symbols = constellation.copy()  # shape (4,)
    # Need shape (n_pol, n_symbols) -> let's say n_pol=1 for now
    rx_1pol = constellation.reshape(1, -1)  # (1, 4)
    # All indices are 0, 1, 2, 3
    indices = np.array([0, 1, 2, 3])

    evm = ber_calc._calculate_evm_arrays(rx_1pol, constellation, indices)
    print(f"EVM (perfect): {evm:.6f}% (expected ~0%)")

    # Test 2: Small noise added
    print("\n--- Test 2: Small noise ---")
    np.random.seed(42)
    noise = 0.1 * (np.random.randn(4) + 1j * np.random.randn(4))
    rx_noisy = constellation + noise
    rx_noisy_1pol = rx_noisy.reshape(1, -1)

    evm_noisy = ber_calc._calculate_evm_arrays(rx_noisy_1pol, constellation, indices)
    print(f"EVM (noisy): {evm_noisy:.6f}%")

    # Test 3: Large noise
    print("\n--- Test 3: Large noise ---")
    noise_large = 0.5 * (np.random.randn(4) + 1j * np.random.randn(4))
    rx_large = constellation + noise_large
    rx_large_1pol = rx_large.reshape(1, -1)

    evm_large = ber_calc._calculate_evm_arrays(rx_large_1pol, constellation, indices)
    print(f"EVM (large noise): {evm_large:.6f}%")

    # Test 4: Dual polarization (DP-QPSK)
    print("\n--- Test 4: Dual polarization ---")
    # For DP, rx_symbols shape is (2, 4) - 2 pols, 4 symbols each
    # constellation_indices must be interleaved: [pol0_t0, pol1_t0, pol0_t1, pol1_t1, ...]
    # length = n_symbols * n_pol = 4 * 2 = 8

    # pol0 symbols: [0, 1, 2, 3], pol1 symbols: [0, 2, 1, 3]
    # Interleaved: [pol0[0], pol1[0], pol0[1], pol1[1], pol0[2], pol1[2], pol0[3], pol1[3]]
    # = [0, 0, 1, 2, 2, 1, 3, 3]
    indices_dp = np.array([0, 0, 1, 2, 2, 1, 3, 3], dtype=int)  # length = 8

    rx_dp = np.array(
        [
            [constellation[0], constellation[1], constellation[2], constellation[3]],  # pol0: 0,1,2,3
            [constellation[0], constellation[2], constellation[1], constellation[3]],  # pol1: 0,2,1,3
        ],
        dtype=np.complex128,
    )  # shape (2, 4)

    evm_dp = ber_calc._calculate_evm_arrays(rx_dp, constellation, indices_dp)
    print(f"EVM (DP perfect): {evm_dp:.6f}% (expected ~0%)")

    # Test 5: DP with noise
    print("\n--- Test 5: DP with noise ---")
    np.random.seed(42)
    noise_dp = 0.1 * (np.random.randn(2, 4) + 1j * np.random.randn(2, 4))
    rx_dp_noisy = rx_dp + noise_dp

    evm_dp_noisy = ber_calc._calculate_evm_arrays(rx_dp_noisy, constellation, indices_dp)
    print(f"EVM (DP noisy): {evm_dp_noisy:.6f}%")

    print("\n" + "=" * 60)
    print("All basic EVM tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_evm_basic()
