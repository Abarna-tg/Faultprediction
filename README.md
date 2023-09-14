# Faultprediction
## Electrical fault detection and classification
###The transmission line is the most crucial part of the power system. The requirement of power and its allegiance has grown up exponentially over the modern era, and the prominent role of a transmission line is to transmit electric power from the source area to the distribution network. The electrical power system consists of so many complex dynamic and interacting elements that are always prone to disturbance or an 
electrical fault.
The power system consists of 4 generators of 11 × 10^3 V, each pair located at each end of the transmission line. Transformers are present in between to simulate and study the various faults at the midpoint of the transmission line.

##What are Electrical Faults?

###Normally, a power system operates under balanced conditions. When the system becomes unbalanced due to the failures of insulation at any point or due to the contact of live wires, a short–circuit or fault, is said to occur in the line. Faults may occur in the power system due to the number of reasons like natural disturbances (lightning, high-speed winds, earthquakes), insulation breakdown, falling of a tree, bird shorting, etc.

##Types of Faults?

###Normally, a power system operates under balanced conditions. When the system becomes unbalanced due to the failures of insulation at any point or due to the contact of live wires, a short–circuit or fault, is said to occur in the line. Faults may occur in the power system due to the number of reasons like natural disturbances (lightning, high-speed winds, earthquakes), insulation breakdown, falling of a tree, bird shorting, etc.

###Faults can be brodly categorised into two types:
###1. Open-circuit Fault
###2. Short-Circuit Faults

##Short-Circuit Faults
###1. Symmetrical
###2. Asymmetrical Faults

##Symmetrical and Asymmetrical Faults
###Symmetrical Faults
###In symmetrical faults, all phases are shorted to each other or to earth (L-L-L) or (L-L-L-G).
###The nature of this type of fault is balanced.
###In this type of fault,fault currents in all phases are symmetrical i.e. their magnitudes are equal and they are equally displaced by ###angle 120 degree.
###It is more severe type of fault but it occurs rarely.

###Asymmetrical Faults
###These faults involve only one or two phases.
###In this type of fault, three phase lines become unbalanced.
###There are mainly three types namely line to ground (L-G), line to line (L-L) and double line to ground (LL-G) faults.
###These type of faults mostly occur on power system..
###About this dataset file
###This file contains the dataset to classify the types of fault.
###Inputs - [Ia,Ib,Ic,Va,Vb,Vc]
###Ia = Current in line A
###Ib = Current in line B
###Ic = Current in line C
###Va = Voltage in line A
###Vb = Voltage in line B

##Vc = Voltage in line C
###Examples :
###[G C B A] - Outputs
###[0 0 0 0] - No Fault
###[1 0 0 0] - Ground Fault
###[0 0 0 1] - Fault in Line A
###[0 0 1 0] - Fault in Line B
###[0 1 0 0] - Fault in Line C
###[1 0 0 1] - LG fault (Between Phase A and Ground)
###[1 0 1 0] - LG fault (Between Phase B and Ground)
###[1 1 0 0] - LG fault (Between Phase C and Ground)
###[0 0 1 1] - LL fault (Between Phase B and Phase A)
###[0 1 1 0] - LL fault (Between Phase C and Phase B)
###[0 1 0 1] - LL fault (Between Phase C and Phase A)
###[1 1 0 0] - LG fault (Between Phase C and Ground)
###[1 0 1 0] - LG fault (Between Phase B and Ground)
###[1 0 0 1] - LG fault (Between Phase A and Ground)
###[1 0 1 1] - LLG Fault (Between Phases A,B and Ground)
###[1 1 0 1] - LLG Fault (Between Phases A,C and Ground)
###[1 1 1 0] - LLG Fault (Between Phases C,B and Ground)
###[0 1 1 1] - LLL Fault(Between all three phases)
###[1 1 1 1] - LLLG fault( Three phase symmetrical fault)

##Objectives of Notebook
###1. Dataset exploration using various types of data visualization.
###2. Build various Machine Learning models that can predict the Fault type in transmission Line.

##Voltage is in Per Unit value (pu)
###If there is any confusion regarding the values of the Line voltages, then let it be clarify that they are most probably in p.u. value i.e.
###Vp.u.=VVbase
 ###In actual the power system consists of 4 generators of 11 × 10^3 V. so we can convert by multiplying them by $11000$ Volts provided they have taken 11k as their base.
###Data Preparation: Gather and preprocess your dataset. Ensure it's structured with features and a target variable indicating whether a fault occurred or not.
###Data Splitting: Split your dataset into a training set and a testing set. This helps in evaluating the model's performance.
###Model Training: Train a Random Forest Classifier on the training data.
###Model Evaluation: Evaluate the model's performance on the testing data using appropriate metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.
###Prediction: Once you have a trained model, you can use it to predict faults in new data.
###Here's a sample Python code using scikit-learn to implement this:
##example
### Import necessary libraries
###from sklearn.model_selection import train_test_split
###from sklearn.ensemble import RandomForestClassifier
###from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### Load your dataset (replace 'X' and 'y' with your actual data)
###X = your_feature_data
###y = your_target_data
### Split the data into training and testing sets
###X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
### Initialize the Random Forest Classifier
###rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
### Train the model on the training data
###rf_classifier.fit(X_train, y_train)

## Make predictions on the testing data
###y_pred = rf_classifier.predict(X_test)

## Evaluate the model
###accuracy = accuracy_score(y_test, y_pred)
###print(f"Accuracy: {accuracy:.2f}")

## Generate a classification report
###print(classification_report(y_test, y_pred))

## Create a confusion matrix
###conf_matrix = confusion_matrix(y_test, y_pred)
###print("Confusion Matrix:")
###print(conf_matrix)
###Make sure to replace 'your_feature_data' and 'your_target_data' with your actual data. You can also adjust the hyperparameters of the Random ###Forest classifier, such as n_estimators and other options, to fine-tune the model's performance.
###Once you have a trained model, you can use it to predict faults in new data by providing the feature values as input to the predict method of the classifier.
