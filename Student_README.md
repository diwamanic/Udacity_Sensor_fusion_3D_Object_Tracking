## Writeup to address the project rubric points and explanation of the programming script

#### FP.1 Match 3D Objects
- the function 'matchBoundingBoxes' is implemented in the camFusion_Student.cpp

- Using a vector<vector<int>> as a selection map, we counted the number of occurences of keymatches in different combinations of currFrame bounding box and prevFrame bounding box.

- And then, for every bounding box in currFrame, I checked for the one with most number of keymatches and put them in the 'std::map<int, int> bbBestMatches map(more like a dictionary)'

#### FP.2 Compute Lidar-Based TTC
- Based on the theory learned for the TTC computation in "Lesson 3: Engineering a collision detection system", I estimated the TTC for subsequent lidar clouds.
- But, to compensate for the outliers present in lidar point clouds, I estimated the standard deviation and eliminated the points which are three standard deviations away from the mean.

#### FP.3 Associate Keypoint Correspondences with Bounding Boxes
-  Before finding the keypoints which are enclosed in the curr bounding box, as we ought to eliminate the outlier matches, I took the distance estimation of all the matches between the currFrame and prevFrame and found the mean of the distances measured.
- After providing a tolerance of 10%, I eliminated all of the other matches which shows distances more than that.
- Then, using the roi member variable in bounding box, after checking the keypoint matches, if it is enclosed in the bounding box, I then push_back that keypoint matches to the keypoint matches variable inside the bounding box.

#### FP.4 Compute Camera-Based TTC
- Based on the theory learned for the TTC computation in "Lesson 3: Engineering a collision detection system", I estimated the TTC for subsequent lidar clouds.
- But, to compensate for the outliers present in distance ratios, instead of choosing a mean for the final distance ratio, I sorted them and took the median value to find the final estimation of TTC Camera.

#### FP.5 Performance Evaluation 1
- In the TTCLidar values, for the image 12 and 17, we have negative values. It signifies that the relative velocity of the vehicles decreases, hence resulting in negative values.
- And for the image 7, the TTC value is way-off, as I think it is due to the erroneous Lidar measurements. 
- Since it is based on the first lidar point X value in both the frames, and the fps will be always high in real time applications, the effect of the noise in the measurements will be too high in the final result.
- Because the difference between the X value in two frames will be very close. Suppose if there were some vibrations during the lidar measurements, that will disrupt the whole TTC calculation.
- Although, this can be addressed with a kdtree clustering algorithm to form groups of clusters and choose the closest point in the largest cluster.
- To make it more robust, we can reduce the variances obtained with both Lidar and Camera, by employing Kalman filter.


#### FP.6 Performance Evaluation 2
- Average TTC Camera is noted down for different combination of detector / descriptor.
- And the difference between the TTC lidar and camera is calculated from the average value and tabled in the following picture.
- Also, when the TTC lidar and camera values are way off, they are ignored in the difference measurement, as it will disrupt the process of estimation of best combination.

##### AVERAGED DIFFERENCE BETWEEN TTC LIDAR AND TTC CAMERA

![Image1](https://user-images.githubusercontent.com/22639337/100371841-ad484b80-3008-11eb-8cdd-c4e823cf1b3f.JPG)

(If the above picture is not working, please refer to the following link)
(https://github.com/diwamanic/Sensor_fusion_working_directory/blob/main/Image2.JPG)

Also, the time taken table with respect to the combination is below:

##### AVERAGE TIME TAKEN

![Image2](https://user-images.githubusercontent.com/22639337/100371965-e1237100-3008-11eb-9b62-2ff17b31b3f6.JPG)

(If the above picture is not working, please refer to the following link)
(https://github.com/diwamanic/Sensor_fusion_working_directory/blob/main/Image2.JPG)

- From the above table, it is very clear and as it is obvious, SIFT descriptor proves it to be best method to have the accuracy of TTC Camera to be reliable.
- But, to also account for the time taken, I have chosen following three to be my best choices:
    1. Detector - HARRIS, Descriptor - SIFT
    2. Detector - HARRIS, Descriptor - FREAK
    3. Detector - ORB, Descriptor - ORB
- But, despite that, it would be better if we could eliminate those outliers by employing the following methods:
    *  Instead of just working with two frames, the chances of outliers could be reduced to a greater extent, if we taken multiple frames
    *  A Kalman filter could be employed to reduce the variance.
Possible causes of outliers:
    *  keypoint mismatching is possible, if there are two similar features in the tailgate of a car (Rear light on both ends). 
    *  Scale change and contrast change, if there is a lighting change between subsequent frames.



