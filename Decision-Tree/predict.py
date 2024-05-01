import joblib 
import numpy as np
import matplotlib.pyplot as plt

clf = joblib.load('treemodel.pkl')

# Sample data
sample_digit = np.array([0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0, 0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0, 0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0, 0, 2, 14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0]).reshape(1, -1)

# use the loaded model to make prediction
predicted_digit = clf.predict(sample_digit)

print(f"prdeicted digit: {predicted_digit[0]}")


# check the digit through visualization
plt.imshow(sample_digit.reshape(8,8), cmap='gray')
plt.title(f"Predicted digit: {predicted_digit[0]}")
plt.savefig('digit.png')
