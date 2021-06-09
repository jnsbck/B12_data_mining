from B12_webscraping_tools import *
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess as cmd


def update_plots():
    data = import_logged_data()
    data = prune_data(data)

    fig, ax = plot_capacity_matrix(
        data, y_axis="weekday", x_axis="min", x_increment=30, figsize=(15, 5)
    )
    ax.set_title("Average traffic at the B12.")
    plt.savefig("capacity_matrix.png", facecolor="white")
    plt.close()

    ax = plot_avg_capacity(
        data, time_binsize="30T", start_datetime=datetime.today().date(), figsize=(15, 5)
    )
    ax.set_title("Todays traffic at the B12.")
    plt.savefig("capacity_timeline.png", facecolor="white")
    plt.close()

    # Push everything to git
    msg = "Update to plots. @ %s" % datetime.strftime(
        datetime.now(), "%H:%M, %m/%d/%Y")
    cmd.run("git add capacity_matrix.png capacity_timeline.png",
            check=True, shell=True, stdout=cmd.DEVNULL)
    cmd.run(f"git commit -m '{msg}'", check=True,
            shell=True, stdout=cmd.DEVNULL)
    cmd.run("git push", check=True,
            shell=True, stdout=cmd.DEVNULL)

    print("[Success] Update plots were pushed to the webpage.")


# needs to be run before saving the figures, as they need to be changed in webpage
    cmd.run("git checkout webpage",
            check=True, shell=True, stdout=cmd.DEVNULL)


deploy_data_logger(update_interval=5, update_func=update_plots, run_every=1)
