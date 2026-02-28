import pandas as pd
#load data/airline_ticket_dataset.xlsx
df = pd.read_excel("data/airline_ticket_dataset.xlsx")
#print columns names and basic statistics like number of rows

#remove front dollar signs in fare and fare_lg column
df['fare'] = df['fare'].str.replace('$', '', regex=False)
df['fare_lg'] = df['fare_lg'].str.replace('$', '', regex=False)
#export to 
if __name__ == "__main__":
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
