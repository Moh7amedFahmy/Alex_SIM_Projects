import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

# change data and labels to arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into 2 parts xtrian 80% and xtest 20%
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Use Random forestclassifier model object
model = RandomForestClassifier()

# To fit model we gave them the x,y train
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save model to test input data
f = open('model.p', 'wb')
pickle.dump({'model':model}, f)
f.close()
