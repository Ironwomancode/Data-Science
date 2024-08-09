
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import csv
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# the data set used exercise 1
#filename = 'iris.data'
#column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
#df = pd.read_csv(filename, header=None, names=column_names)

# Display the first few rows of the dataframe
#print(df.head())

#sns.pairplot(df, hue='species')
#plt.show()

iris_data = []
with open('iris.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        # Ensure that each row has exactly 5 elements
        if len(row) == 5:
            iris_data.append(row)

# Convert the zip object to a list before slicing
zipped_data = list(zip(*iris_data))  # Convert the zip object to a list

# Separate the measurements (columns 0-3) and convert them to floats
# measurements list stores each measurement column separately

measurement_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
measurements = [list(map(float, col)) for col in zipped_data[:4]]

# Compute and print the mean and standard deviation for each measurement across all species
print("Overall Statistics for All Species:")
for i, col in enumerate(measurements):
    mean = statistics.mean(col)
    stdev = statistics.stdev(col)

    #prints only the measurement
    #print(f"Measurement {i+1}: Mean = {mean:.2f}, Standard Deviation = {stdev:.2f}")
    print(f"{measurement_names[i]}: Mean = {mean:.2f}, Standard Deviation = {stdev:.2f}")
# Compute mean and standard deviation separately for each species
species_data = {'Iris-setosa': [], 'Iris-versicolor': [], 'Iris-virginica': []}

# Populate species_data dictionary with data for each species
for row in iris_data:
    species = row[-1]  # Last element in the row is the species name
    species_data[species].append(list(map(float, row[:-1])))  # Add measurements to the respective species

print("\nStatistics per Species:")
for species, data in species_data.items():
    print(f"\nSpecies: {species}")
    measurements = [list(col) for col in zip(*data)]  # Extract measurements for this species
    for i, col in enumerate(measurements):
        mean = statistics.mean(col)
        stdev = statistics.stdev(col)
        #print(f"Measurement {i+1}: Mean = {mean:.2f}, Standard Deviation = {stdev:.2f}")
        print(f"{measurement_names[i]}: Mean = {mean:.2f}, Standard Deviation = {stdev:.2f}")


#Based on the considerations of Exercise 3, assign the flowers with the
# following measurements to what you consider would be the most likely species.

# Convert the iris_data to a DataFrame for visualization
df = pd.DataFrame(iris_data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"])

# Convert measurement columns to numeric values
df["Sepal Length"] = pd.to_numeric(df["Sepal Length"])
df["Sepal Width"] = pd.to_numeric(df["Sepal Width"])
df["Petal Length"] = pd.to_numeric(df["Petal Length"])
df["Petal Width"] = pd.to_numeric(df["Petal Width"])

# 1. Pair Plot: Show pairwise relationships in the dataset
#Provides a grid of scatter plots for each pair of measurements, colored by species
sns.pairplot(df, hue="Species")
plt.suptitle("Each Pair of Iris Measurements, colored by species", y=1.02)
plt.show()

# 2. Scatter Plot: Sepal Length vs Sepal Width colored by species
sns.scatterplot(data=df, x="Sepal Length", y="Sepal Width", hue="Species", style="Species")
plt.title("Sepal Length versus Sepal Width")
plt.show()

# 3. Box Plot: Show distribution of values for each measurement, grouped by species
plt.figure(figsize=(10, 8))
sns.boxplot(x="Species", y="Sepal Length", data=df)
plt.title("Box Plot of Sepal Length by Species")
plt.show()

# Repeat for other measurements
plt.figure(figsize=(10, 8))
sns.boxplot(x="Species", y="Sepal Width", data=df)
plt.title("Box Plot of Sepal Width by Species")
plt.show()

plt.figure(figsize=(10, 8))
sns.boxplot(x="Species", y="Petal Length", data=df)
plt.title("Box Plot of Petal Length by Species")
plt.show()

plt.figure(figsize=(10, 8))
sns.boxplot(x="Species", y="Petal Width", data=df)
plt.title("Box Plot of Petal Width by Species")
plt.show()






#Extras
#Violin Plots
#Violin plots combine aspects of box plots and kernel density plots.
#They provide a deeper insight into the distribution of the data.
plt.figure(figsize=(10, 8))
sns.violinplot(x="Species", y="Sepal Length", data=df)
plt.title("Violin Plot of Sepal Length by Species")
#plt.show()

plt.figure(figsize=(10, 8))
sns.violinplot(x="Species", y="Sepal Width", data=df)
plt.title("Violin Plot of Sepal Width by Species")
#plt.show()

plt.figure(figsize=(10, 8))
sns.violinplot(x="Species", y="Petal Length", data=df)
plt.title("Violin Plot of Petal Length by Species")
#plt.show()

plt.figure(figsize=(10, 8))
sns.violinplot(x="Species", y="Petal Width", data=df)
plt.title("Violin Plot of Petal Width by Species")
#plt.show()


# KDE Plots
#The plots provide a smoothed estimate of the distribution of a dataset.

plt.figure(figsize=(10, 8))
sns.kdeplot(data=df, x="Sepal Length", hue="Species", fill=True)
plt.title("KDE Plot of Sepal Length by Species")
#plt.show()

plt.figure(figsize=(10, 8))
sns.kdeplot(data=df, x="Sepal Width", hue="Species", fill=True)
plt.title("KDE Plot of Sepal Width by Species")
plt.legend("You might notice that some species have more compact distributions, while others have broader or more spread-out distributions")
#plt.show()

plt.figure(figsize=(10, 8))
sns.kdeplot(data=df, x="Petal Length", hue="Species", fill=True)
plt.title("KDE Plot of Petal Length by Species")
#plt.show()

plt.figure(figsize=(10, 8))
sns.kdeplot(data=df, x="Petal Width", hue="Species", fill=True)
plt.title("KDE Plot of Petal Width by Species")
#plt.show()

# Perform ANOVA
#You can use statistical tests to understand the relationships between variables.
#For example, ANOVA (Analysis of Variance) can be used to determine if there are
#statistically significant differences between the means of different groups.
for col in measurement_names:
    print("ANOVA (Analysis of Variance) can be used to determine if there are statistically significant differences between the means of different groups")
    print(f"\nANOVA for {col}:")
    groups = [df[df['Species'] == species][col] for species in df['Species'].unique()]
    f_val, p_val = stats.f_oneway(*groups)
    print(f"F-value: {f_val:.2f}, p-value: {p_val:.2f}")

# Prepare the data
X = df[measurement_names]  # Features
y = df["Species"]          # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Print the results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))







#Question 5
# New measurements
new_measurements = pd.DataFrame([
    [5.2, 3.1, 4.0, 1.2],
    [4.9, 2.5, 5.6, 2.0],
    [5.4, 3.2, 1.9, 0.4]
], columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
# Predict species
predictions = knn.predict(new_measurements)

print("Question 5 in Lab exercise")
# Output predictions
for i, measurement in enumerate(new_measurements.values.tolist()):

    print(f"Measurements {measurement} is classified as: {predictions[i]}")