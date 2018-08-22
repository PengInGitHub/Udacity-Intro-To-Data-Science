import pandas

def add_full_name(path_to_csv, path_to_new_csv):
    baseball = pandas.read_csv(path_to_csv)
    baseball['nameFull'] = baseball['nameFirst'] + " " + baseball['nameLast']
    baseball.to_csv(path_to_new_csv)



if __name__ == "__main__":
    # For local use only
    # If you are running this on your own machine add the path to the
    # Lahman baseball csv and a path for the new csv.
    # The dataset can be downloaded from this website: http://www.seanlahman.com/baseball-archive/statistics
    # We are using the file Master.csv
    path_to_csv = ""
    path_to_new_csv = ""
    add_full_name(path_to_csv, path_to_new_csv)
