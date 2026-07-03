import numpy as np
import pywt


def wavelet_denoise(data, wavelet='db4', level=2):
    """
    De-noises a 1D price series using the Discrete Wavelet Transform (DWT)
    with universal soft thresholding (VisuShrink).

    Parameters:
    -----------
    data : list or np.ndarray
        The raw, noisy price data (e.g., 1-minute close prices for the last 60 minutes).
    wavelet : str, optional
        The type of wavelet family to use.
        - 'db4' (Daubechies 4) is excellent for financial data because it captures
          sudden price shifts well without over-smoothing.
        - 'sym8' (Symlets 8) is another popular choice.
    level : int, optional
        The decomposition level. Higher levels smooth the data more, but can introduce
        edge distortion. For 60 data points, a level of 1 or 2 is optimal.

    Returns:
    --------
    np.ndarray
        The smoothed, de-noised price series matching the input length.
    """
    # Ensure data is a float numpy array
    data_arr = np.array(data, dtype=np.float64)
    n = len(data_arr)

    # 1. Multi-level DWT Decomposition
    # Returns: [cA_n, cD_n, cD_n-1, ..., cD_1]
    # cA_n is the coarsest trend. The cD's are the high-frequency detail lists.
    coefficients = pywt.wavedec(data_arr, wavelet, mode='symmetric', level=level)

    # 2. Noise Estimation (Sigma)
    # Estimate the noise level using the Median Absolute Deviation (MAD)
    # of the highest frequency detail coefficients (cD_1, last item in the list).
    # The constant 0.6745 scales the MAD to align with normal distribution std dev.
    highest_freq_details = coefficients[-1]
    mad = np.median(np.abs(highest_freq_details - np.median(highest_freq_details)))
    sigma = mad / 0.6745

    # Avoid division by zero if the signal is entirely flat
    if sigma < 1e-8:
        return data_arr

    # 3. Calculate the Universal Threshold (VisuShrink)
    # This formula calculates the mathematical boundary between signal and noise.
    threshold = sigma * np.sqrt(2 * np.log(n))

    # 4. Apply Soft Thresholding
    # Leave the approximation coefficients (coefficients[0]) completely untouched
    # because they contain the core trend. Apply thresholding ONLY to detail coefficients.
    denoised_coefficients = [coefficients[0]]
    for detail_coeff in coefficients[1:]:
        # Soft thresholding reduces coefficient values towards zero by the threshold amount.
        # Preferred over "hard" thresholding as it prevents sharp, unnatural jumps.
        cleaned_details = pywt.threshold(detail_coeff, threshold, mode='soft')
        denoised_coefficients.append(cleaned_details)

    # 5. Reconstruct the Signal (IDWT)
    denoised_signal = pywt.waverec(denoised_coefficients, wavelet, mode='symmetric')

    # Reconstruction padding can sometimes slightly change the output array length,
    # so truncate to match the exact input length.
    return denoised_signal[:n]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create dummy "noisy" price data (a clean trend + random noise)
    np.random.seed(42)
    time = np.linspace(0, 10, 100)
    clean_trend = np.sin(time) * 10 + 100   # Base trend
    noise = np.random.normal(0, 1.5, size=100)  # Random noise
    noisy_prices = clean_trend + noise           # Observed price

    # Apply Wavelet De-noising
    smooth_prices = wavelet_denoise(noisy_prices, wavelet='db4', level=2)

    # Calculate a Simple Moving Average (SMA) for comparison
    window_size = 10
    sma_prices = np.convolve(noisy_prices, np.ones(window_size) / window_size, mode='same')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(noisy_prices, label="Raw Noisy Prices (Tick Data)", color='lightgray', alpha=0.8)
    plt.plot(clean_trend, label="True Trend (Hidden)", color='black', linestyle='--', alpha=0.7)
    plt.plot(smooth_prices, label="Wavelet De-noised Signal (db4, lvl 2)", color='blue', linewidth=2)
    plt.plot(sma_prices, label=f"Standard Moving Average (SMA {window_size})", color='red', linestyle=':')
    plt.title("Wavelet De-noising vs. Simple Moving Average")
    plt.xlabel("Time (Ticks)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
