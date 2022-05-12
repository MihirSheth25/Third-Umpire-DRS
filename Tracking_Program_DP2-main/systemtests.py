import datetime as dt
from os import environ
from balltrack import BallTracker
from general import Board, slope

class TestCustom:
    @staticmethod
    def setup_class():
        print(f'\n[{dt.datetime.now().strftime("%d %B %Y, %I:%M:%S %p")}]')
        print("Initializing PyTest testing...\n")

    @staticmethod
    def teardown_class():
        print("\nAll PyTest cases completed!")

    @staticmethod
    def setup_method():
        current_test = environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')
        print(f'\nNow testing: {current_test[0][5:]}')

    @staticmethod
    def teardown_method():
        print(f"\n{'-' * 30}")

    @staticmethod
    def test_ball_tracker():
        """
        Tests the BallTracker class and default properties.
        """
        sample_file = "ramp_trial.mp4"
        testcam = BallTracker(sample_file)
        testboard = Board()

        assert testcam.file == sample_file
        assert not testcam.is_running
        assert len(testcam.ball) == 0
        assert testcam.board.WIDTH == testboard.WIDTH
        assert testcam.board.LENGTH == testboard.LENGTH
        del testcam

    @staticmethod
    def test_math_functions():
        """
        Tests the general functions.
        """
        p1, p2 = (0, 0), (8, 4)
        assert slope(p1, p2) == 0.5
        assert slope(p2, p1) == 0.5
