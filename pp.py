
import numpy as np 


A = np.array([1,2,3,4,5])
B = np.array([6,7,8,9,10])


sum_result = A+B
print("Element-wise sum", sum_result)



diff_result = A - B
print("Element-wise difference: " , diff_result)

dot_result = A*B
print("Element_ wise product" , dot_result)

div_result = A/B
print("Elements-wise division ", div_result)

mean_A = np.mean(A)
print("Mean of A: ", mean_A)

max_B =  np.max(B)
print("Maximum element of B: ", max_B)



min_A = np.min(A)
print("Minimum elemnts of A: ", min_A)




#Plotting Sales Data
#You have been given sales data for a company over a period 10 months .The data includes the month and the correspending sales amount in thousands of dollars. Plot the sales data using Matplotib to visualize the sales trend over time


#Sales data:
#Month:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct']
#Sales Amount: [20,25,18,22,30,28,35,32,26,29
import matplotlib.pyplot as plt 

months = ['Jan','Feb ','Mar', 'Apr','May','Jun','July','Aug','Sep','Oct']
sales = [20,25,28,22,30,28,35,32,26,29]


plt.plot(months,sales,marker='o')



plt.title('Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales Amount (in $1000s)')


plt.show()

#SIMPLE LINEAR REGRESSION

#You have been a dataset that contains the number of hours studied and the corresponding scores achieved by a group of students Your task is to build a simple linear regression model using Tenser Flow to predict the score based on the number of hours studied train the model 
#using TenserFlow to predict the score based on the number of hours stidied. Train the model 
#using the provided dataset and evaluate its performance 

#Dataset:
#Nunber of hours studied :  [2,3,4,5,6,7,8,9,10]
#Scores acheived :[25,50,42,61,75,68,80,92,88]

import tensorflow as tf
import numpy as np


hours_studied = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
scores_achieved = np.array([25, 50, 42, 61, 75, 68, 80, 92, 88], dtype=np.float32)


W = tf.Variable(0.0)
b = tf.Variable(0.0)
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)


Y_pred = W * X + b


loss = tf.reduce_mean(tf.square(Y_pred - Y))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(100):
        _, current_loss = sess.run([train_op, loss], feed_dict={X: hours_studied, Y: scores_achieved})
        print("Epoch:", epoch + 1, "Loss:", current_loss)
    
    
    final_W, final_b = sess.run([W, b])

print("Linear Regression Model: Y =", final_W, "* X +", final_b)


#READING AND MANUPILATING JSON DATA
#You have been given a JSON file that contains information about employees in a company
#your task is read the json file,extract relevant information, and perform some muniplation on the data 


import json

with open ('employees.json') as f:
    data = json.load(f)
    
    
employees = data['employees']
for employees in employees:
    print("ID:", employee['id'])
    print("Name:", employee['name'])
    print("Department", employee['department'])
    print("Salary:", employee['salary'])
    print()
    


salaries = [employee['salary'] for employee in employees ]
average_salary = sum(salaries) / len(salaries)
print("Avergae Salary: ",average_salary )

employee_id = 2 
new_department = 'HR'
for employee in employees:
    if employee['id'] == employee_id:
        employee['department'] = new_department
        
        
        
        
updated_data = {'employees':employees}
with open('updated_employees.json','w')as f:
    json.dump(updated_data, f , indent = 2)
    print("Data updated and saved to upload_employees.json.")
