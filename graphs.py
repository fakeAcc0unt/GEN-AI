import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load a built-in dataset (Iris dataset)
iris = sns.load_dataset("iris")

print(iris.head(5))
# 1. Histogram (Distribution of petal length)
plt.figure(figsize=(6,4))
sns.histplot(iris['petal_length'],bins=20)
plt.title("Histogram of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()


# 2. Scatter Plot (Sepal length vs Petal length by species)
plt.figure(figsize=(6,4))
sns.scatterplot(data=iris, x="sepal_length", y="petal_length", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.show()


# 3. Correlation Heatmap
plt.figure(figsize=(6,4))
corr = iris.drop("species", axis=1).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.show()
