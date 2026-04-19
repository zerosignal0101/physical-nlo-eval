import numpy as np

from physics_nlo_eval.channel.fiber import (
    EDFA,  # Assuming EDFA is also in channel or noise
    FiberChannel,
)

# 导入所有模块
from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.pipeline import Pipeline
from physics_nlo_eval.dsp.carrier import PhaseNoiseCompensation
from physics_nlo_eval.dsp.freq_domain import FOFine, OFDM_Demodulator
from physics_nlo_eval.dsp.mimo import MIMO_ChannelEstimation
from physics_nlo_eval.dsp.time_domain import FOCoarse, Resampler, TimingRecovery
from physics_nlo_eval.metrics.ber_evm import BERCalculator, Demapper
from physics_nlo_eval.rx_front.adc import ADC
from physics_nlo_eval.rx_front.coherent import CoherentReceiver
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.ofdm import OFDMModulator
from physics_nlo_eval.tx.optics import IQModulator


def run_simulation() -> None:
    """运行一个完整的相干光通信仿真流程。"""
    print("Starting coherent optical communication simulation...")

    # --- 0. 全局仿真参数 ---
    # Optical parameters
    optical_wavelength_nm = 1550.0
    optical_center_freq = 2.99792458e8 / (optical_wavelength_nm * 1e-9)  # Hz

    # System parameters
    symbol_rate_baud = 50e9  # 50 Gbaud
    samples_per_symbol_tx = 4  # TX electrical signal upsampling
    samples_per_symbol_rx = 2  # RX ADC sampling, then downsample to 1 or 2 for DSP

    n_pol = 2  # Dual Polarization
    modulation_format = "DP-16QAM"
    bits_per_symbol = 4  # log2(16)

    # Data length
    num_symbols_per_pol = 100000  # Number of QAM symbols per polarization
    num_bits_per_pol = num_symbols_per_pol * bits_per_symbol
    total_tx_bits = num_bits_per_pol * n_pol

    # OFDM parameters (if OFDM is used)
    is_ofdm_system = True
    ofdm_n_subcarriers = 256
    ofdm_cp_len = 32
    ofdm_ts_pattern = np.tile(
        np.exp(1j * np.random.uniform(0, 2 * np.pi, ofdm_n_subcarriers)), (n_pol, 1)
    )  # Random phase pilots for TS

    # Fiber parameters
    fiber_length_km = 80.0
    fiber_alpha_db_per_km = 0.2
    fiber_beta2_ps2_per_m = -21.6e-27
    fiber_beta3_ps3_per_m = 0.12e-39
    fiber_d_pmd_ps_per_sqrt_km = 0.1  # ps/sqrt(km)

    # EDFA parameters
    edfa_gain_db = fiber_length_km * fiber_alpha_db_per_km  # Compensate loss
    edfa_nf_db = 4.5

    # ADC parameters
    adc_resolution_bits = 8
    adc_sampling_rate_hz = symbol_rate_baud * samples_per_symbol_rx  # 2 samples/symbol

    # --- 1. 发送端 (TX) ---
    print("\n--- TX Chain ---")
    tx_pipeline_blocks = [
        Mapper(modulation=modulation_format, power_norm=True),
    ]

    if is_ofdm_system:
        tx_pipeline_blocks.append(
            OFDMModulator(
                n_subcarriers=ofdm_n_subcarriers,
                cp_len=ofdm_cp_len,
                ts_pattern=ofdm_ts_pattern,
                num_data_symbols_per_frame=100,
            )
        )
        tx_electrical_sample_rate = (
            symbol_rate_baud * samples_per_symbol_tx * (ofdm_n_subcarriers + ofdm_cp_len) / ofdm_n_subcarriers
        )
    else:
        tx_electrical_sample_rate = symbol_rate_baud * samples_per_symbol_tx
        # If not OFDM, a pulse shaping filter would be here (e.g., RRC)
        # For simplicity, we skip pulse shaping in this example for non-OFDM.

    tx_pipeline_blocks.append(IQModulator(optical_power_per_pol_mw=10.0, wavelength_nm=optical_wavelength_nm))
    # PolarizationMultiplexer might be implicit in IQModulator if it directly outputs DP.
    # If not, you'd add PolarizationMultiplexer here for SP inputs to form DP.
    # tx_pipeline_blocks.append(PolarizationMultiplexer()) # If Mapper/IQModulator output SP

    tx_pipeline = Pipeline(tx_pipeline_blocks, name="TX_Pipeline")

    # Generate random bits
    tx_bits = np.random.randint(0, 2, total_tx_bits, dtype=bool)
    # Store ideal symbols for EVM calculation if needed later (need to capture from Mapper output)

    # Run TX
    tx_optical_signal = tx_pipeline(tx_bits)
    print(
        f"TX Output Signal: Data Shape {tx_optical_signal.data.shape}, "
        f"Sample Rate {tx_optical_signal.ctx.sample_rate / 1e9:.2f} GSa/s, "
        f"Center Freq {tx_optical_signal.ctx.center_freq / 1e12:.2f} THz"
    )
    print(f"TX Signal Average Power (per pol): {np.mean(np.abs(tx_optical_signal.data) ** 2):.2f} mW")
    tx_ideal_symbols = tx_pipeline.blocks[0].forward(tx_bits)  # Capture ideal symbols from Mapper

    # --- 2. 信道 (Channel) ---
    print("\n--- Channel Chain ---")
    channel_pipeline = Pipeline(
        [
            FiberChannel(
                length_m=fiber_length_km * 1000,
                alpha_db_per_km=fiber_alpha_db_per_km,
                beta2_ps2_per_km=fiber_beta2_ps2_per_m,
                beta3_ps3_per_km=fiber_beta3_ps3_per_m,
                d_pmd_ps_per_sqrt_km=fiber_d_pmd_ps_per_sqrt_km,
                coarse_step_size_m=10000.0,  # 10 km step
            ),
            EDFA(gain_db=edfa_gain_db, nf_db=edfa_nf_db),
        ],
        name="Channel_Pipeline",
    )

    rx_optical_signal = channel_pipeline(tx_optical_signal)
    print(
        f"Channel Output Signal: Data Shape {rx_optical_signal.data.shape}, "
        f"Sample Rate {rx_optical_signal.ctx.sample_rate / 1e9:.2f} GSa/s, "
        f"Center Freq {rx_optical_signal.ctx.center_freq / 1e12:.2f} THz"
    )
    print(f"RX Signal Average Power (per pol): {np.mean(np.abs(rx_optical_signal.data) ** 2):.2f} mW")

    # --- 3. 接收前端 (RX Front-end) ---
    print("\n--- RX Front-end Chain ---")
    rx_frontend_pipeline = Pipeline(
        [
            CoherentReceiver(
                lo_power_dbm=13.0,
                lo_linewidth_hz=100e3,
                lo_freq_offset_hz=5e6,  # Example LO frequency offset
            ),
            ADC(resolution_bits=adc_resolution_bits, sampling_rate_hz=adc_sampling_rate_hz, clip_ratio=1.2),
        ],
        name="RX_Front_End_Pipeline",
    )

    rx_electrical_signal = rx_frontend_pipeline(rx_optical_signal)
    print(
        f"RX Front-end Output Signal: Data Shape {rx_electrical_signal.data.shape}, "
        f"Sample Rate {rx_electrical_signal.ctx.sample_rate / 1e9:.2f} GSa/s"
    )

    # --- 4. 接收 DSP (RX DSP) ---
    print("\n--- RX DSP Chain ---")
    rx_dsp_pipeline_blocks: list[BaseBlock] = []

    # Time Domain DSP
    # rx_dsp_pipeline_blocks.append(ChromaticDispersionComp(
    #     fiber_length_km=fiber_length_km, # Compensate known fiber length
    #     beta2_ps2_per_km=fiber_beta2_ps2_per_km
    # ))
    rx_dsp_pipeline_blocks.append(FOCoarse(search_range_hz=10e6))  # Compensate LO freq offset

    # Resample to 2 sps for timing recovery, or 1 sps directly
    # Assuming ADC outputs 2 sps. Target 1 sps for symbol decision
    if rx_electrical_signal.ctx.sps > 1:
        rx_dsp_pipeline_blocks.append(Resampler(target_sps=2))  # To 2 sps if not already

    rx_dsp_pipeline_blocks.append(TimingRecovery(sps_in=2))  # Assuming input is 2sps, output 1sps

    if is_ofdm_system:
        # OFDM Demodulation
        rx_dsp_pipeline_blocks.append(OFDM_Demodulator(n_subcarriers=ofdm_n_subcarriers, cp_len=ofdm_cp_len))
        # Frequency Domain DSP for OFDM
        rx_dsp_pipeline_blocks.append(MIMO_ChannelEstimation(method="LS", ts_interval=10))  # Using TS for CE
        rx_dsp_pipeline_blocks.append(FOFine(method="BPS"))  # Or pilot_assisted for OFDM
        rx_dsp_pipeline_blocks.append(PhaseNoiseCompensation(method="BPS"))  # Or pilot_assisted
    else:
        # Single-carrier DSP
        rx_dsp_pipeline_blocks.append(MIMO_ChannelEstimation(method="CMA", equalizer_length=11, cma_mu=1e-4))
        rx_dsp_pipeline_blocks.append(FOFine(method="data_aided"))
        rx_dsp_pipeline_blocks.append(PhaseNoiseCompensation(method="V&V"))  # BPS or V&V for SC

    rx_dsp_pipeline = Pipeline(rx_dsp_pipeline_blocks, name="RX_DSP_Pipeline")

    rx_final_symbols = rx_dsp_pipeline(rx_electrical_signal)
    print(
        f"RX DSP Output Symbols: Data Shape {rx_final_symbols.data.shape}, "
        f"Sample Rate {rx_final_symbols.ctx.sample_rate / 1e9:.2f} GSa/s, "
        f"SPS {rx_final_symbols.ctx.sps}"
    )

    # --- 5. Output: 判决与 BER 计算 ---
    print("\n--- Output & Metrics ---")
    demapper = Demapper(modulation=modulation_format, power_norm=True)
    rx_bits = demapper(rx_final_symbols)
    print(f"Demapped RX Bits: {rx_bits.shape}")

    # Align TX and RX bits (e.g., if some symbols were lost/truncated)
    min_bits_len = min(len(tx_bits), len(rx_bits))
    tx_bits_aligned = tx_bits[:min_bits_len]
    rx_bits_aligned = rx_bits[:min_bits_len]

    ber_calculator = BERCalculator(ignore_headers=0)  # Adjust ignore_headers if needed for specific frame structures
    ber = ber_calculator.calculate_ber(tx_bits_aligned, rx_bits_aligned)
    print(f"Calculated BER: {ber:.2e}")

    # For EVM, we need the original TX symbols after mapper, before any upsampling
    # This might be tricky to get if the TX pipeline upsamples.
    # For now, let's just re-create the ideal TX symbols from the Mapper.
    # Note: For OFDM, EVM is typically per subcarrier, per symbol.
    tx_ideal_qam_symbols = tx_ideal_symbols.data  # (n_pol, N_symbols)

    # Align RX symbols to TX symbols (shape will depend on OFDM vs SC)
    if is_ofdm_system:
        # For OFDM, tx_ideal_qam_symbols will be (n_pol, N_subcarriers * N_data_symbols_per_frame * num_frames)
        # rx_final_symbols.data will be (n_pol, N_subcarriers, N_symbols)
        # Need to reshape tx_ideal_qam_symbols for EVM comparison
        # Also, need to skip TS symbols for EVM calc.

        # This part requires careful matching of symbol counts and indices.
        # For simplicity, let's just take the first N_symbols_for_evm from both.
        num_symbols_for_evm = min(
            tx_ideal_qam_symbols.shape[1],  # Total QAM symbols from mapper
            rx_final_symbols.data.shape[1] * rx_final_symbols.data.shape[2],  # Total demodulated symbols
        )
        tx_symbols_evm = tx_ideal_qam_symbols[:, :num_symbols_for_evm]
        rx_symbols_evm = rx_final_symbols.data.reshape(n_pol, -1)[:, :num_symbols_for_evm]

    else:  # Single-carrier
        num_symbols_for_evm = min(tx_ideal_qam_symbols.shape[1], rx_final_symbols.data.shape[1])
        tx_symbols_evm = tx_ideal_qam_symbols[:, :num_symbols_for_evm]
        rx_symbols_evm = rx_final_symbols.data[:, :num_symbols_for_evm]

    evm = ber_calculator.calculate_evm(tx_symbols_evm, rx_symbols_evm)
    print(f"Calculated EVM: {evm:.2f}%")
