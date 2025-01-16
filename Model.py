import pandas as pd

df = pd.read_csv('D:\Related to programing Language\Python\Train_Data.csv')
print(df.shape)  # 3630 rows, 7 columns
print(df.head())
age = df['age']
raw_sex = df['sex']         # categorical
bmi = df['bmi']
raw_smoker = df['smoker']   # categorical
raw_region = df['region']   # categorical
children = df['children']
# printing unique categorical values
print(f"sex --> {list(set(raw_sex))}")
print(f"smoker --> {list(set(raw_smoker))}")
print(f"region --> {list(set(raw_region))}")
proc_sex = []  # processed age
proc_smoker = []
proc_region = []

for i in range(len(age)):

    bag_sex = [0] * len(list(set(raw_sex)))
    if raw_sex[i] ==  'male': bag_sex[0] = 1
    elif raw_sex[i] == 'female': bag_sex[1] = 1
    proc_sex.append(bag_sex)

    bag_smoker = [0] * len(list(set(raw_smoker)))
    if raw_smoker[i] == 'yes': bag_smoker[0] = 1
    elif raw_smoker[i] == 'no': bag_smoker[1] = 1
    proc_smoker.append(bag_smoker)

    bag_region = [0] * len(list(set(raw_region)))
    if raw_region[i] == 'southwest': bag_region[0] = 1
    elif raw_region[i] == 'northeast': bag_region[1] = 1
    elif raw_region[i] == 'southeast': bag_region[2] = 1
    elif raw_region[i] == 'northwest': bag_region[3] = 1
    proc_region.append(bag_region)

print(len(proc_sex))
print(proc_sex[:10])
print(proc_smoker[:10])
print(proc_region[:5])
# creating training dataset
train_x = []

for i in range(len(age)):
    temp = [
        age[i], proc_sex[i][0], proc_sex[i][1], bmi[i], children[i],
        proc_smoker[i][0], proc_smoker[i][1],
        proc_region[i][0], proc_region[i][1], proc_region[i][2], proc_region[i][3]
    ]
    train_x.append(temp)
    import numpy as np

train_x = np.array(train_x)

train_y = df['charges']
train_y = np.array(train_y)

print(train_x.shape, train_y.shape)
print(train_x[:2])
print(train_y[:2])
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


model = Sequential([
    layers.Dense(64, input_shape=[11], activation='relu'),
    # layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    # layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    
    layers.Dense(1),
])

model.compile(
    loss='mae',
    optimizer='adam'
)
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=30,
    patience=20,
    restore_best_weights=True,
)
history = model.fit(
    train_x, train_y,
    validation_split=0.1,
    epochs=300,
    callbacks=[early_stopping],
    verbose=0
)

print("loss :", model.evaluate(train_x, train_y))
import matplotlib.pyplot as plt

plt.plot( history.history['loss'] )
plt.plot( history.history['val_loss'] )
plt.show()
# checking how good the model fits on training dataset

train_predictions = model.predict(train_x)

plt.plot(train_predictions[:30])
plt.plot(train_y[:30])
plt.show()
# for i in range(len(train_predictions)):
for i in range(20):
    print(f"{int(train_y[i]//1)} predicted --> {int(train_predictions[i][0]//1)},\t\tdifference --> { ( int(train_predictions[i][0]//1) - int(train_y[i]//1) ) }")

df = pd.read_csv('D:\Related to programing Language\Python\Test_Data.csv')

print(df.shape)  # 492 rows, 6 columns
print(df.head())
age = df['age']
raw_sex = df['sex']         # categorical
bmi = df['bmi']
raw_smoker = df['smoker']   # categorical
raw_region = df['region']   # categorical
children = df['children']

# printing unique categorical values
print(f"sex --> {list(set(raw_sex))}")
print(f"smoker --> {list(set(raw_smoker))}")
print(f"region --> {list(set(raw_region))}")
# feature engineering on categorical values
proc_sex = []  # processed age
proc_smoker = []
proc_region = []

for i in range(len(age)):

    bag_sex = [0] * len(list(set(raw_sex)))
    if raw_sex[i] ==  'male': bag_sex[0] = 1
    elif raw_sex[i] == 'female': bag_sex[1] = 1
    proc_sex.append(bag_sex)

    bag_smoker = [0] * len(list(set(raw_smoker)))
    if raw_smoker[i] == 'yes': bag_smoker[0] = 1
    elif raw_smoker[i] == 'no': bag_smoker[1] = 1
    proc_smoker.append(bag_smoker)

    bag_region = [0] * len(list(set(raw_region)))
    if raw_region[i] == 'southwest': bag_region[0] = 1
    elif raw_region[i] == 'northeast': bag_region[1] = 1
    elif raw_region[i] == 'southeast': bag_region[2] = 1
    elif raw_region[i] == 'northwest': bag_region[3] = 1
    proc_region.append(bag_region)

print(len(proc_sex))
print(proc_sex[:10])
print(proc_smoker[:10])
print(proc_region[:5])
# creating testing dataset
test_x = []

for i in range(len(age)):
    temp = [
        age[i], proc_sex[i][0], proc_sex[i][1], bmi[i], children[i],
        proc_smoker[i][0], proc_smoker[i][1],
        proc_region[i][0], proc_region[i][1], proc_region[i][2], proc_region[i][3]
    ]
    test_x.append(temp)

import numpy as np

test_x = np.array(test_x)

print(test_x.shape)
print(test_x[:2])
predictions = model.predict(test_x)

print("AGE\tSEX\tBMI\tSMOKER\t REGION\t\t  CHARGES")
# for i in range(len(predictions)):
for i in range(20):
    print(f"{df['age'][i]//1}\t{df['sex'][i]}\t{df['bmi'][i]//1}\t{df['smoker'][i]}\t{df['region'][i]}", end='\t')
    print(f"predicted --> $ {int(predictions[i][0]//1)}")
