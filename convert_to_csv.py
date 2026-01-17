import pandas as pd

# --------------------------------------------------
# STEP 1: READ COLUMN NAMES FROM adult.names
# --------------------------------------------------
column_names = []

with open("data/adult.names", "r") as file:
    for line in file:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("|"):
            continue

        # Extract column name before ':'
        if ":" in line:
            column_name = line.split(":")[0]
            column_names.append(column_name)

# Add target column manually (not listed with :)
column_names.append("income")

print("Columns extracted from adult.names:")
print(column_names)

# --------------------------------------------------
# STEP 2: CONVERT adult.data → adult_train.csv
# --------------------------------------------------
df_train = pd.read_csv(
    "data/adult.data",
    names=column_names,
    sep=",",
    skipinitialspace=True
)

df_train.to_csv("adult_train.csv", index=False)
print("✅ adult.data converted to adult_train.csv")

# --------------------------------------------------
# STEP 3: CONVERT adult.test → adult_test.csv
# --------------------------------------------------
df_test = pd.read_csv(
    "data/adult.test",
    names=column_names,
    sep=",",
    skipinitialspace=True,
    skiprows=1  # skip header line
)

# Remove trailing '.' from income column
df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

df_test.to_csv("adult_test.csv", index=False)
print("✅ adult.test converted to adult_test.csv")
