
from balltrack import BallTracker


def main():
    MAIN_FILE = "trial2-1.mp4"
    SIM_FILE = "sim-edit.mp4"
    ballcam = BallTracker(MAIN_FILE, simulation=False)
    simcam = BallTracker(SIM_FILE, simulation=True)
    ballcam.track()
    simcam.track()


if __name__ == "__main__":
    main()