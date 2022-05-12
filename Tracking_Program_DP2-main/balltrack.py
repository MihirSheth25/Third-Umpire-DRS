"""
BALLTRACK.PY

Main ball-tracking class and tools for the project. Uses
OpenCV for video capture and processing, then NumPy and
custom modules for post-processing and computations. Data
is plotted visually using Matplotlib.
"""
# Libraries and modules
import cv2
import time
import csv
import logging as lg
import numpy as np
import datetime as dt
import traceback as tb
from os import path
from imutils import resize
from tabulate import tabulate
from matplotlib import pyplot as plt
# Local files
from timing import timer, time_interval
from datamodels import Board, Point, Region, Vector


# noinspection PyUnresolvedReferences
class BallTracker(object):
    def __init__(self, file_name: str, simulation: bool = False):
        """
        Camera object initialization.
        """
        # Basic run properties
        self.file = file_name
        self.video = cv2.VideoCapture(self.file)
        self.is_running = False
        self.board = Board()
        self.output = None

        # Video general properties
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.backsub = cv2.createBackgroundSubtractorKNN()
        self.simulation = simulation
        if simulation:
            self.lower = np.array([35, 110, 50])
            self.upper = np.array([70, 255, 110])
            self.radius_bounds = (10, 20)
        else:
            self.lower = np.array([52, 35, 0])
            self.upper = np.array([255, 255, 255])
            self.radius_bounds = (12, 25)
        self.ball = []
        self.radius_log = []
        self.ball_detected = False
        self.frame_dim = None
        self.roi_dim = None
        self.cm_pixel_ratio = 1

        # Video scaling properties
        self.x_adj_factor = 6 if self.simulation else 0
        self.y_adj_factor = 38 if self.simulation else 34
        self.x_axis_factor = 2 if self.simulation else 1.8
        self.y_axis_factor = 1.9 if self.simulation else 2

        # Configure logging
        log_fmt = "[%(levelname)s] %(asctime)s | %(message)s"
        date_fmt = "%I:%M:%S %p"
        lg.basicConfig(
            level=lg.DEBUG,
            format=log_fmt,
            datefmt=date_fmt
        )
        lg.getLogger('matplotlib.font_manager').setLevel(lg.WARNING)

        # Notification of completion
        lg.info("Ball tracker created.")
        print(f'Board dimensions: {self.board.WIDTH}mm x {self.board.HEIGHT}mm')
        print("-"*25)

    def __repr__(self):
        return f'<BallTracker file={self.file}, bounds={self.lower}, {self.upper}>'

    def __len__(self):
        if self.video is not None:
            return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def __getitem__(self, item: int):
        if type(item) != int:
            type_name = lambda v: str(type(v)).split("'")[1].lower()
            raise TypeError(f'{type_name(item)} was given but int was required.')
        return self.ball[item]

    def __iter__(self):
        for item in self.ball:
            yield item

    def __convert_units(self, item, precision: int = 0):
        """
        Converts pixel units to centimeters.

        Args:
            item: Value in pixels
            precision (int): Decimal places for result

        Returns:
            unit length in centimeters
        """
        if precision < 0:
            raise ValueError("Precision must be non-negative.")

        if precision:
            return round(self.cm_pixel_ratio * item, precision)

        return round(self.cm_pixel_ratio * item)

    def __trace_path(self, frame):
        """
        Traces path line of ball in video.
        """
        for i in range(1, len(self.ball)):
            prev = self.ball[i-1]
            curr = self.ball[i]
            if abs(prev.x - curr.x) < 20 and abs(prev.y - curr.y) < 20:
                p_prev = (prev.x, self.roi_dim.width[1] - prev.y)
                p_curr = (curr.x, self.roi_dim.width[1] - curr.y)
                cv2.line(frame, p_prev, p_curr, (10, 10, 255), 3)

    @staticmethod
    def __scale(value, factor: int = 1):
        """
        Scales the value according to a given integer factor
        """
        if factor <= 0:
            if factor:
                raise ValueError("Scaling factor must be a positive number.")
            raise ZeroDivisionError("Scaling factor cannot be zero.")

        else:
            return round(value / factor)

    @staticmethod
    def __show_trajectory(image, center, point):
        """
        Draw arrows to depict ball's trajectory and components.

        Args:
            image (Frame): Image to draw on
            center (tuple): Coordinates of the ball center
            point (Point): Ball point data object
        """
        # Set coordinate points and establish vector direction
        x_c, y_c = center[0], center[1]
        v_x = (round(x_c+point.velocity.x), y_c)
        v_y = (x_c, round(y_c-point.velocity.y))
        direction = (round(x_c+point.velocity.x), round(y_c-point.velocity.y))

        # Draw arrows
        cv2.arrowedLine(image, center, direction, (10, 130, 250), 2, 8, 0, 0.1)
        cv2.arrowedLine(image, center, v_x, (10, 255, 50), 2, 8, 0, 0.1)
        cv2.arrowedLine(image, center, v_y, (10, 255, 50), 2, 8, 0, 0.1)

    def __scale_data(self, data, axis: str):
        """
        Scales data set to compensate for video differences.
        """
        resize_factor = 0.9 if self.simulation else 1
        if axis.upper() == "X":
            out = [self.x_axis_factor * (self.__convert_units(p.x, 2) - self.x_adj_factor) / resize_factor for p in data]
        elif axis.upper() == "Y":
            out = [self.y_axis_factor * (self.__convert_units(p.y, 2) - self.y_adj_factor) for p in data]
        else:
            raise Exception(f'Invalid axis: {axis} was given but \"X\" or \"Y\" was expected.')

        return out

    @timer
    def track(self):
        """
        Track ball and plot position markers on frame.
        """
        # Configure starting variables and preprocessing
        lg.info("Beginning tracking sequence.")
        start_time = time.time()
        radius = 0
        pos = {'x': 0, 'y': 0}
        speed, angle = 0, 0
        baseline = time.perf_counter()

        while self.video.isOpened():
            # Grab frame from video
            ret, frame = self.video.read()
            if not ret:
                break  # End loop if out of frames

            # Setting up frame and region-of-interest
            width, length, _ = frame.shape
            w_r, h_r = self.board.WIDTH / width, self.board.HEIGHT / length
            self.cm_pixel_ratio = (w_r + h_r) / 2

            # Crop video
            if self.simulation:
                self.roi_dim = Region(length=(100, length), width=(0, width))
            else:
                self.roi_dim = Region(length=(400, length-450), width=(200, width-75))

            self.frame_dim = Region(length=(0, length), width=(0, width))
            roi = frame[
                  self.roi_dim.width[0]:self.roi_dim.width[1],
                  self.roi_dim.length[0]:self.roi_dim.length[1]
            ]
            frame = resize(frame, height=540)
            roi = resize(roi, height=540)

            r_height, r_width, _ = roi.shape
            self.board.r_width, self.board.r_length = r_height, r_width

            # Isolate ball shape via HSV bounds
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_hue = np.array([52, 35, 0])
            upper_hue = np.array([255, 255, 255])
            mask = cv2.inRange(hsv, lower_hue, upper_hue)
            (contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # When "ball" is detected
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                mts = cv2.moments(c)

                # Highlight ball shape and collect position data
                try:
                    if self.radius_bounds[0] < radius < self.radius_bounds[1] and x < 700:
                        self.ball_detected = True
                        stamp = time.perf_counter()
                        center = (int(mts["m10"]/mts["m00"]), int(mts["m01"]/mts["m00"]))
                        pos['x'], pos['y'] = round(x), round(y)
                        x_c, y_c = center[0], self.roi_dim.width[1] - center[1]
                        interval = stamp - baseline

                        if len(self.ball):
                            prev = self.ball[-1]
                            x_p, y_p = prev.x, prev.y
                            t_c, t_p = interval, prev.time
                            vx = round((x_c - x_p) / (t_c - t_p), 2)
                            vy = round((y_c - y_p) / (t_c - t_p), 2)
                        else:
                            vx, vy = 0, 0

                        # Store data and highlight ball
                        frame_data = Point(
                            time=interval,
                            x=x_c,
                            y=y_c,
                            velocity=Vector(x=vx, y=vy)
                        )
                        self.ball.append(frame_data)
                        self.radius_log.append(round(radius, 2))
                        self.__show_trajectory(roi, center, frame_data)
                        speed = self.__convert_units(frame_data.velocity.net(), 2)
                        angle = round(frame_data.velocity.angle(), 1)
                        cv2.circle(roi, center, 15, (30, 200, 255), 3)

                    else:
                        self.ball_detected = False

                    # Put tracer line to show ball path
                    if len(self.ball) > 2:
                        self.__trace_path(roi)

                except Exception as e:
                    lg.error(f'Error during ball trace: {e}')
                    tb.print_exc()

            # Put window text and frame details
            cv2.putText(frame, "Video feed", (10, 30), self.font, 0.7, (10, 255, 100), 2)
            time_stamp = dt.datetime.now().strftime("%I:%M:%S %p")
            pos_x_cm = self.__convert_units(pos["x"])
            pos_y_cm = self.__convert_units(pos["y"])
            roi_text = [
                time_stamp,
                f'Frame size: {width} x {length}',
                f'Runtime: {time_interval(start_time)}',
                f'Radius: {round(radius, 2)}',
                f'Ball in frame: {"Yes" if self.ball_detected else "No"}',
                f'Speed: {speed} cm/s ({angle:.1f} deg)',
                f'Last position: ({pos_x_cm}, {pos_y_cm})',
            ]
            for i, label in enumerate(roi_text):
                cv2.putText(roi, str(label), (10, 20 + (20 * i)),
                            self.font, 0.5, (10, 255, 100), 1)

            cv2.imshow("Ball Tracking", roi)

            # Close windows with 'Esc' key
            key = cv2.waitKey(10)
            if key == 27:
                lg.info("Ending tracking session.")
                break

        self.video.release()
        cv2.destroyAllWindows()
        lg.info("Tracker session ended.")

        # Postprocessing functions for data analysis
        self.get_run_statistics()

    def get_run_statistics(self):
        """
        Summarize data gathered during tracking run.
        """
        avg = lambda d: round(sum(d) / len(d), 2)
        try:
            # General run information
            print("\n***** TRACK DATA *****")
            print(f'Radius range: [{min(self.radius_log)}, {max(self.radius_log)}]')
            print(f'Average radius: {avg(self.radius_log)}')
            print(f'Total points: {len(self.ball)}')
            print(f'ROI area: W = {self.roi_dim.width}, L = {self.roi_dim.length}')
            print("*" * 22)
            self.plot_data()
            lg.info("Presenting recorded data.")

            # Tabulate point data
            headers = ["TIME", "POSITION", "VELOCITY", "SPEED", "ANGLE"]
            data = []
            for point in self.ball:
                t = round(point.time, 3)
                v_x = self.__convert_units(point.velocity.x, 2)
                v_y = self.__convert_units(point.velocity.y, 2)
                p_x = self.__convert_units(point.x, 2)
                p_y = self.__convert_units(point.y, 2)
                res = self.__convert_units(point.velocity.net(), 3)
                angle = round(point.velocity.angle(), 3)
                data_row = [t, (p_x, p_y), (v_x, v_y), res, f'{angle:.1f}°']
                data.append(data_row)
            print(tabulate(data, headers=headers, tablefmt="orgtbl"))

        except ValueError as ve:
            lg.error(f'Empty data: {ve}')

    def plot_data(self):
        """
        Plot 2D data acquired from tracking session.
        """
        # Separate coordinates data into X and Y parameters
        adj_set = self.ball[30:] if self.simulation else self.ball
        x_raw = self.__scale_data(adj_set, axis="x")
        y_raw = self.__scale_data(adj_set, axis="y")
        x_data = np.array(x_raw, dtype=float)
        y_data = np.array(y_raw, dtype=float)
        print("-" * 25)

        # Generate plot
        plot_title = f'Ball tracking {"of simulation" if self.simulation else "across board"}'
        plot_color = "darkgreen" if self.simulation else "firebrick"
        plt.scatter(x_data, y_data, color=plot_color)
        plt.title(plot_title)
        plt.xlim(0, self.board.WIDTH)
        plt.ylim(0, self.board.HEIGHT)
        plt.xlabel("X-coordinate (cm)")
        plt.ylabel("Y-coordinate (cm)")
        plt.grid()
        plt.show()
        lg.info("Showing data plot.")

    def export_data(self):
        """
        Export gathered data as CSV file.
        """
        plain_file_name = self.file.split(".")[0]
        csv_file = f'data_({plain_file_name}).csv'

        try:
            if not path.exists(csv_file):
                with open(csv_file, 'w', encoding='UTF8') as file:
                    # Establish CSV headers
                    header = ["TIME", "POSITION", "VELOCITY", "SPEED", "ANGLE"]
                    writer = csv.writer(file)
                    writer.writerow(header)

                    # Go through list of stored points, write to file
                    self.ball = self.ball[30:] if self.simulation else self.ball
                    resize_factor = 0.9 if self.simulation else 1
                    for point in self.ball:
                        t = round(point.time, 3)
                        v_x = self.__convert_units(point.velocity.x, 1)
                        v_y = self.__convert_units(point.velocity.y, 1)
                        p_x = self.x_axis_factor * (self.__convert_units(point.x, 2) - self.x_adj_factor) / resize_factor
                        p_y = self.y_axis_factor * (self.__convert_units(point.y, 2) - self.y_adj_factor)
                        res = self.__convert_units(point.velocity.net(), 2)
                        angle = round(point.velocity.angle(), 1)
                        data_row = [t, (p_x, p_y), (v_x, v_y), res, angle]
                        writer.writerow(list(map(lambda x: str(x), data_row)))

                print("Data compiled and exported to CSV file.")

            else:
                print(f'Data for \"{self.file}\" has already been exported to CSV file.')

        except Exception as e:
            lg.error(f'Error during data export: {e}')

