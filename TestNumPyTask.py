import unittest
import numpy as np
from numpy_task_impl import SoundWaveFactory, NOTES

class TestSoundWaveFactory(unittest.TestCase):

    def setUp(self):
        # Set up the SoundWaveFactory object before each test
        self.factory = SoundWaveFactory()
        self.default_note = 'a4'
        self.invalid_note = 'z9'
        self.valid_wave_types = ['sin', 'square', 'triangle']
        self.invalid_wave_type = 'sawtooth'

    def test_get_normed_wave_valid(self):
        # Test generating normalized waves with valid wave types
        for wave_type in self.valid_wave_types:
            wave = self.factory.get_normed_wave(NOTES[self.default_note], wave_type)
            self.assertIsInstance(wave, np.ndarray)
            self.assertEqual(len(wave), len(self.factory.timeline))

    def test_get_normed_wave_invalid_wave_type(self):
        # Test generating a wave with an invalid wave type
        with self.assertRaises(ValueError):
            self.factory.get_normed_wave(NOTES[self.default_note], self.invalid_wave_type)

    def test_create_wave_valid(self):
        # Test creating a sound wave with a valid note and wave type
        wave = self.factory.create_wave(self.default_note)
        self.assertIsInstance(wave, np.ndarray)
        self.assertEqual(wave.dtype, np.int16)

    def test_create_wave_invalid_note(self):
        # Test creating a sound wave with an invalid note
        with self.assertRaises(ValueError):
            self.factory.create_wave(self.invalid_note)

    def test_read_wave_from_txt_valid(self):
        # Test reading a wave from a valid text file
        # Creating a temporary wave file for testing
        filename = 'test_wave.txt'
        np.savetxt(filename, np.random.randint(-32768, 32767, size=1000, dtype=np.int16))
        wave = self.factory.read_wave_from_txt(filename)
        self.assertIsInstance(wave, np.ndarray)

    def test_read_wave_from_txt_invalid_file(self):
        # Test reading a wave from a non-existent text file
        with self.assertRaises(RuntimeError):
            self.factory.read_wave_from_txt('non_existent_file.txt')

    def test_print_wave_details_valid(self):
        # Test printing wave details with a valid wave array
        wave = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
        try:
            self.factory.print_wave_details(wave)
        except Exception as e:
            self.fail(f"print_wave_details raised an exception: {e}")

    def test_print_wave_details_invalid(self):
        # Test printing wave details with an invalid input (not a numpy array)
        with self.assertRaises(ValueError):
            self.factory.print_wave_details([1, 2, 3, 4])

    def test_normalize_sound_waves_valid(self):
        # Test normalizing a list of valid sound wave arrays
        waves = [np.random.randint(-32768, 32767, size=1000, dtype=np.int16) for _ in range(3)]
        normalized_waves = self.factory.normalize_sound_waves(waves)
        for normalized_wave in normalized_waves:
            self.assertIsInstance(normalized_wave, np.ndarray)

    def test_normalize_sound_waves_invalid(self):
        # Test normalizing a list containing non-ndarray elements
        with self.assertRaises(ValueError):
            self.factory.normalize_sound_waves([np.array([1, 2, 3]), [4, 5, 6]])

if __name__ == '__main__':
    unittest.main()
