import pandas as pd

from termcolor import colored


def main():
    delivery = pd.read_csv("data/onlinedeliverydata.csv")

    # Printing the information of dataset
    print("[INFO] The shape of the  data is (row, column):" + str(delivery.shape))
    print(colored("read_csv: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    return



if __name__ == "__main__":
    main()