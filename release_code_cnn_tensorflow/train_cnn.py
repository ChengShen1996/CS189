from data_manager import data_manager
from cnn import CNN
from trainer import Solver
from viz_features import Viz_Feat
import random


import matplotlib.pyplot as plt
print("version 1.0")
CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple', 
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

LITTLE_CLASS_LABELS = ['apple','banana','eggplant']

CLASS_LABELS = ['Truck', 'Van', 'Misc', 'Pedestrian', 'Tram', 'Cyclist', 'Person_sitting', 'Car'] #modified

image_size1 = 300
image_size2 = 300

random.seed(0)

classes = CLASS_LABELS
dm = data_manager(classes, image_size1, image_size2)

weights_regularizer = [0.05, 0.005, 0.0005, 0.00005, 0.000005]

# for lambda_ in weights_regularizer:
lambda_= 0.0005
cnn = CNN(classes,image_size1, image_size2, lambda_)

print("Training CNN...")
max_iter = 5000
solver = Solver(cnn,dm,max_iter)

solver.optimize()

print("Ploting")
plt.plot(solver.test_accuracy,label = 'Validation')
plt.plot(solver.train_accuracy, label = 'Training')
plt.legend()
plt.xlabel('Iterations (in 200s)')
plt.ylabel('Accuracy')
plt.title('5 filters 4 layers at each convolutional layer')
	# plt.show()
plt.savefig('images/Accuracy_KITTI.png')

# print("Visualizing...")
# val_data = dm.val_data
# train_data = dm.train_data

# sess = solver.sess

# cm = Viz_Feat(val_data,train_data,CLASS_LABELS,sess)

# cm.vizualize_features(cnn)

print("Done!")


