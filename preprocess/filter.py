import pandas as pd
#load data/airline_ticket_dataset.xlsx
df = pd.read_excel("data/airline_ticket_dataset.xlsx")
#print columns names and basic statistics like number of rows
if __name__ == "__main__":
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
