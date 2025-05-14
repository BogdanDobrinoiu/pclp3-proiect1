import pandas as pd
from faker import Faker
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

fake = Faker('en_US')
cities = ["London", "Liverpool", "Manchester", "Oxford", "Cambridge", "Cardiff",
        "Leicester", "Birmingham", "York", "Leeds", "Bradford", "Bristol"]
number_of_rooms = [random.randint(1, 10) for _ in range(1000)]
  
houseRegister =  {
    "Owner_name": [fake.name() for _ in range(1000)],
    "City": [random.choice(cities) for _ in range(1000)],
    "Contruction_year": [random.randint(1850, 2015) for _ in range(1000)],
    "Acquisition_year": [],
    "Number_of_rooms": number_of_rooms,
    "Number_of_bathrooms": [random.randint(1, min(3, rooms)) for rooms in number_of_rooms],
    "Surface_sqm": [round(random.uniform(30, 200), 1) for _ in range(1000)],
    "Initial_price_EUR": [random.randint(50000, 1000000) for _ in range(1000)],
    "Current_price_EUR": [] # to be determined
}

for year in houseRegister["Contruction_year"]:
    houseRegister["Acquisition_year"].append(year + random.randint(0, 10))

for i in range(1000):
    inflation = 0.03
    years = 2015 - houseRegister["Acquisition_year"][i]
    price = houseRegister["Initial_price_EUR"][i] * (1 + inflation) ** years
    price += houseRegister["Surface_sqm"][i] * 1
    price += houseRegister["Number_of_rooms"][i] * 2
    price += random.uniform(-10, 10)
    houseRegister["Current_price_EUR"].append(round(price, 2))

df = pd.DataFrame(houseRegister)
df.to_csv("House_register.csv", index=False)

df = pd.read_csv("House_register.csv")

X = df.drop(columns = ["Current_price_EUR", "Owner_name"])
y = df["Current_price_EUR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    train_size = 0.715, 
    test_size = 0.285,
    random_state = 42
)

train_set = pd.concat([X_train, y_train], axis = 1)
test_set = pd.concat([X_test, y_test], axis = 1)

train_set.to_csv("train.csv", index = False)
test_set.to_csv("test.csv", index = False)

missing_vals = train_set.isnull().sum()
missing_percent = 100 * missing_vals / len(train_set)

missing_df = pd.DataFrame({
    "Missing_values": missing_vals, 
    "Percentage (%)": missing_percent
})

print(missing_df[missing_df["Missing_values"] > 0])
# we don't have missing values, so no methods of resolving them

print(train_set.describe())
print(train_set.describe(include = ["object"]))

numeric_col = train_set.select_dtypes(include = ["float64", "int64"]).columns
train_set[numeric_col].hist(figsize = (15, 10), bins = 30)
plt.tight_layout()
plt.show()

categorical_col = train_set.select_dtypes(include = ["object"]).columns
for col in categorical_col:
    plt.figure(figsize = (8, 4))
    sns.countplot(data = train_set, x = col, order = train_set[col].value_counts().index)
    plt.title(f"Value distribution for {col}")
    plt.xticks(rotation = 45)
    plt.show()

for col in numeric_col:
    plt.figure(figsize = (8, 4))
    sns.boxplot(data = train_set, x = col)
    plt.title(f"Boxplot for {col}")
    plt.show()

correlaton_matrix = train_set.corr(numeric_only = True)
plt.figure(figsize = (10, 8))
sns.heatmap(correlaton_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", square = True)
plt.title("Correlation matrix")
plt.show()

target = "Current_price_EUR"
for col in numeric_col:
    if col != target:
        plt.figure(figsize = (6, 4))
        sns.scatterplot(data = train_set, x = col, y = target)
        plt.title(f"{col} vs {target}")
        plt.show()

for col in categorical_col:
    plt.figure(figsize = (8,4))
    sns.violinplot(data = train_set, x = col, y = target)
    plt.title(f"{col} vs {target}")
    plt.xticks(rotation = 45)
    plt.show()

X_train = pd.get_dummies(X_train, columns = ['City'], drop_first = True)
X_test = pd.get_dummies(X_test, columns = ["City"], drop_first = True)
X_test = X_test.reindex(columns = X_train.columns, fill_value = 0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R²: {r2:.4f}")

plt.figure(figsize = (8, 6))
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("Real values")
plt.ylabel("Predicted values")
plt.title("Predictions vs. Real values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  
plt.grid()
plt.show()

errors = y_test - y_pred
plt.figure(figsize = (8, 5))
sns.histplot(errors, kde = True, bins = 30, color = "red")
plt.title("Error distribution")
plt.xlabel("Error (Real value - Predicted value)")
plt.ylabel("Frequency")
plt.grid()
plt.show()

plt.figure(figsize = (8, 5))
plt.scatter(y_test, errors, alpha = 0.5)
plt.axhline(0, color = 'green', linestyle = '--')
plt.title("Errors depending on real values")
plt.xlabel("Real values")
plt.ylabel("Error")
plt.grid()
plt.show()
