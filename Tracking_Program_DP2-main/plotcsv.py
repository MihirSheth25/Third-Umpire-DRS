import csv
from matplotlib import pyplot as plt
import pandas as pd
def plot_graph():
    #plot_title = f'Ball tracking {"of simulation " if self.simulation else "across board"}'
       #plot_color = "darkgreen" if self.simulation else "firebrick"
    col_list = ["POSITION_X", "POSITION_Y"]
    df1 = pd.read_csv("data_(trial_suc1).csv", usecols=col_list)
    df2 = pd.read_csv("data_(trial_suc2).csv", usecols=col_list)
    df3 = pd.read_csv("data_(trial_fail1).csv", usecols=col_list)
    df4 = pd.read_csv("data_(trial_fail2).csv", usecols=col_list)
    plt.plot(df1["POSITION_X"], df1["POSITION_Y"], color="darkgreen", label="Success1")
    plt.plot(df2["POSITION_X"], df2["POSITION_Y"], color="blue", label="Success2")
    plt.plot(df3["POSITION_X"], df3["POSITION_Y"], color="red", label="Failure1")
    plt.plot(df4["POSITION_X"], df4["POSITION_Y"], color="black", label="Failure2")
    #plt.title(plot_title)
    plt.xlim(0, 110)
    plt.ylim(0, 70)
    plt.xlabel("X-coordinate (cm)")
    plt.ylabel("Y-coordinate (cm)")
    plt.legend()
    plt.grid()
    plt.show()
    #lg.info("Showing data plot.")