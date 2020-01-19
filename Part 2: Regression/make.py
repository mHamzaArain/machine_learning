# Step 6: Making the Confusion Matrix -> This contain right/wromg predictions to make our model robust 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, # correct data,
                      y_pred) #  predicted data

