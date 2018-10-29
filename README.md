# MachineLearning

#Exercise 2
(1) From a 2D-gaussian distribution with zero mean, sigma_1 = 3.0, sigma_2 = 1.0, no cross correlation
• sample 50 data points,
• calculate the sample mean and the covariance matrix,
• rotate the data by 30°, calculate the covariance matrix again
• rotate the original true covariance matrix by 30°
• sample again 50 data points and show them with a different colour together with the rotated data points

 (2) Read in the 3D, time dependend data in DatAccel.txt and DatGyr.txt. (you need the reader from csv modul)
a) visualize the data with two subplots over linear time (time stamps are given in the first column, read in row by row using datetime.datetime)
b) check if/which data are correlated: calculate and print the 3D normalized correlation matrix of each dataset, show a scatter plot of the data pair with highest correlation
c) (*) look at the DatAccel.txt visualization, interpret what you see with respect to the sampling theorem, given these are acceleration data of a car.