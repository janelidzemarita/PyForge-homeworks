import numpy as np
from scipy.io import wavfile

# Constants
SAMPLING_RATE = 44100  # Sampling rate in Hz (44.1 KHz)
DURATION_SECONDS = 5  # Default duration of the sound in seconds
SOUND_ARRAY_LEN = SAMPLING_RATE * DURATION_SECONDS
MAX_AMPLITUDE = 2 ** 13  # Maximum amplitude of the sound wave

# Dictionary of note frequencies
NOTES = {
    '0': 0, 'e0': 20.60172, 'f0': 21.82676, 'f#0': 23.12465, 'g0': 24.49971, 'g#0': 25.95654, 'a0': 27.50000,
    'a#0': 29.13524, 'b0': 30.86771, 'c0': 32.70320, 'c#0': 34.64783, 'd0': 36.70810, 'd#0': 38.89087,
    'e1': 41.20344, 'f1': 43.65353, 'f#1': 46.24930, 'g1': 48.99943, 'g#1': 51.91309, 'a1': 55.00000, 'a#1': 58.27047,
    'b1': 61.73541, 'c1': 65.40639, 'c#1': 69.29566, 'd1': 73.41619, 'd#1': 77.78175,
    'e2': 82.40689, 'f2': 87.30706, 'f#2': 92.49861, 'g2': 97.99886, 'g#2': 103.8262, 'a2': 110.0000, 'a#2': 116.5409,
    'b2': 123.4708, 'c2': 130.8128, 'c#2': 138.5913, 'd2': 146.8324, 'd#2': 155.5635,
    'e3': 164.8138, 'f3': 174.6141, 'f#3': 184.9972, 'g3': 195.9977, 'g#3': 207.6523, 'a3': 220.0000, 'a#3': 233.0819,
    'b3': 246.9417, 'c3': 261.6256, 'c#3': 277.1826, 'd3': 293.6648, 'd#3': 311.1270,
    'e4': 329.6276, 'f4': 349.2282, 'f#4': 369.9944, 'g4': 391.9954, 'g#4': 415.3047, 'a4': 440.0000, 'a#4': 466.1638,
    'b4': 493.8833, 'c4': 523.2511, 'c#4': 554.3653, 'd4': 587.3295, 'd#4': 622.2540,
    'e5': 659.2551, 'f5': 698.4565, 'f#5': 739.9888, 'g5': 783.9909, 'g#5': 830.6094, 'a5': 880.0000, 'a#5': 932.3275,
    'b5': 987.7666, 'c5': 1046.502, 'c#5': 1108.731, 'd5': 1174.659, 'd#5': 1244.508,
    'e6': 1318.510, 'f6': 1396.913, 'f#6': 1479.978, 'g6': 1567.982, 'g#6': 1661.219, 'a6': 1760.000, 'a#6': 1864.655,
    'b6': 1975.533, 'c6': 2093.005, 'c#6': 2217.461, 'd6': 2349.318, 'd#6': 2489.016,
    'e7': 2637.020, 'f7': 2793.826, 'f#7': 2959.955, 'g7': 3135.963, 'g#7': 3322.438, 'a7': 3520.000, 'a#7': 3729.310,
    'b7': 3951.066, 'c7': 4186.009, 'c#7': 4434.922, 'd7': 4698.636, 'd#7': 4978.032,
}

class SoundWaveFactory:
    """
    A factory class for generating sound waves with different frequencies, wave types, and handling audio file operations.

    Attributes:
        sampling_rate (int): The number of samples per second.
        duration_seconds (int): The duration of the sound in seconds.
        max_amplitude (int): The maximum amplitude of the sound wave.
        timeline (ndarray): The time axis for generating waveforms.
    """

    def __init__(self, sampling_rate=SAMPLING_RATE, duration_seconds=DURATION_SECONDS, max_amplitude=MAX_AMPLITUDE):
        """
        Initialize the SoundWaveFactory with default or user-defined settings.

        Args:
            sampling_rate (int): Sampling rate in Hz.
            duration_seconds (int): Duration of the sound in seconds.
            max_amplitude (int): Maximum amplitude of the sound wave.
        """
        self.sampling_rate = sampling_rate
        self.duration_seconds = duration_seconds
        self.max_amplitude = max_amplitude
        self.timeline = np.linspace(0, self.duration_seconds, num=self.sampling_rate * self.duration_seconds)

    def get_normed_wave(self, frequency, wave_type='sin'):
        """
        Generate a normalized waveform based on the specified frequency and wave type.

        Args:
            frequency (float): Frequency of the waveform.
            wave_type (str): Type of wave ('sin', 'square', or 'triangle').

        Returns:
            ndarray: Normalized waveform array.
        """
        if wave_type not in ['sin', 'square', 'triangle']:
            raise ValueError("Invalid wave type. Supported types: 'sin', 'square', 'triangle'.")

        try:
            if wave_type == 'sin':
                return self.max_amplitude * np.sin(2 * np.pi * frequency * self.timeline)
            elif wave_type == 'square':
                return self.max_amplitude * np.sign(np.sin(2 * np.pi * frequency * self.timeline))
            elif wave_type == 'triangle':
                return self.max_amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * self.timeline))
        except Exception as e:
            raise RuntimeError(f"Error generating waveform: {e}")

    def create_wave(self, note="a4", name=None, wave_type='sin'):
        """
        Create and save a sound wave as a WAV file.

        Args:
            note (str): Musical note to generate (e.g., 'a4').
            name (str): Optional file name for the generated wave.
            wave_type (str): Type of wave ('sin', 'square', 'triangle').

        Returns:
            ndarray: The generated sound wave array.
        """
        frequency = NOTES.get(note)
        if frequency is None:
            raise ValueError(f"Note '{note}' is not defined in the NOTES dictionary.")

        try:
            sound_wave = self.get_normed_wave(frequency, wave_type).astype(np.int16)
            file_name = f"{note}_wave.wav" if name is None else f"{name}.wav"
            wavfile.write(file_name, self.sampling_rate, sound_wave)
            return sound_wave
        except Exception as e:
            raise RuntimeError(f"Error creating wave for note '{note}': {e}")

    @staticmethod
    def read_wave_from_txt(filename):
        """
        Read a sound wave from a text file.

        Args:
            filename (str): The name of the text file containing the sound wave.

        Returns:
            ndarray: The sound wave array.
        """
        try:
            return np.loadtxt(filename, converters={0: lambda s: int(float(s))}, dtype=np.int16)
        except Exception as e:
            raise RuntimeError(f"Error reading wave from text file '{filename}': {e}")

    @staticmethod
    def print_wave_details(wave):
        """
        Print details of the wave array.

        Args:
            wave (ndarray): The sound wave array.
        """
        if not isinstance(wave, np.ndarray):
            raise ValueError("The wave must be a numpy ndarray.")
        print(f"Wave Details:\n- Length: {len(wave)}\n- Max Amplitude: {np.max(wave)}\n- Min Amplitude: {np.min(wave)}")

    def normalize_sound_waves(self, waves):
        """
        Normalize the amplitude of the given sound waves.

        Args:
            waves (list of ndarray): List of sound wave arrays to be normalized.

        Returns:
            list of ndarray: List of normalized sound wave arrays.
        """
        if not all(isinstance(wave, np.ndarray) for wave in waves):
            raise ValueError("All elements in 'waves' must be numpy ndarray objects.")

        try:
            return [wave / max(np.abs(wave)) * self.max_amplitude for wave in waves]
        except Exception as e:
            raise RuntimeError(f"Error normalizing sound waves: {e}")


# Example Usage
if __name__ == "__main__":
    factory = SoundWaveFactory()
    wave_a4 = factory.create_wave("a4")
    wave_c4 = factory.create_wave("c4")
    wave_e4 = factory.create_wave("e4")
    factory.print_wave_details(wave_a4)
    factory.print_wave_details(wave_c4)
    factory.print_wave_details(wave_e4)

    # Read wave from text file
    wave_from_txt = factory.read_wave_from_txt("wave.txt")
    factory.print_wave_details(wave_from_txt)

    # Normalize sound waves
    normalized_waves = factory.normalize_sound_waves([wave_a4, wave_c4, wave_e4, wave_from_txt])
    factory.print_wave_details(normalized_waves[0])
    factory.print_wave_details(normalized_waves[1])
    factory.print_wave_details(normalized_waves[2])
    factory.print_wave_details(normalized_waves[3])
