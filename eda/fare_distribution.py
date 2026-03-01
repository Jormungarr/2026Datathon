import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess.data_utils import import_unit_removed_dataset
# Load your data (replace with your actual data loading code)
# Example: df = pd.read_csv('your_data.csv')
# For demonstration, let's assume df is already loaded and has a 'fare' column

# Plot the marginal distribution of 'fare'
df = import_unit_removed_dataset()
plt.figure(figsize=(8, 5))
sns.histplot(df['fare'], kde=True)
plt.title('Marginal Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
