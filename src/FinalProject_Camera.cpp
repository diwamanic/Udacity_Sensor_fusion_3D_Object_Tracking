
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <unistd.h>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

void initialize_log_vector(std::vector<Log_Database>& total_log_data, int n_images) {
    const std::vector<std::string> detectorTypes{ "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT" };
    //const std::vector<std::string> detectorTypes{ "SHITOMASI" };
    
    const std::vector<std::string> descriptorTypes{ "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT" };
    //const std::vector<std::string> descriptorTypes{ "BRISK"};
    const std::vector<std::string> matcherTypes{ "MAT_BF" };
    const std::vector<std::string> selectorTypes{ "SEL_KNN" };

    for(auto detType : detectorTypes)
        for(auto descType: descriptorTypes)
            for(auto matType: matcherTypes)
                for (auto selType : selectorTypes) {
                    if ((descType.compare("AKAZE") == 0 && detType.compare("AKAZE") != 0)||
                        (descType.compare("ORB") == 0 && detType.compare("SIFT") == 0))
                        
                    {
                        continue;
                    }
                    Log_Database temp(n_images, detType, descType, matType, selType);
                    //cout << "DETECTOR/ DESCRIPTOR COMBINATION: " << temp.detectorType << "-" << temp.descriptorType << endl; 
            
                    total_log_data.push_back(temp);
                }
}

void writeToFile(const std::vector<Log_Database>& total_log_data)
{
    std::string fileName{ "../log/Diwakar_Manickavelu_FinalProject_Camera.csv" };
    std::cout << "Writing to O/P file" << fileName << std::endl;

    std::ofstream file{ fileName };
    file << "Image Index: " << ",";
    file << "Detector Type: " << ",";
    file << "Descriptor Type: " << ",";
    file << "Matcher Type: " << ",";
    file << "Selector Type: " << ",";
    file << "Detection timetaken: " << ",";
    file << "Total No. of detected points: " << ",";
    file << "Num. of Filtered points: " << ",";
    file << "Descriptor timetaken: " << ",";
    file << "Matcher timetaken: " << ",";
    file << "No. of Matched pairs: " << ",";
    file << "No. of Matched pairs in CurrBB in front: " << ",";
    file << "TTC Camera" << ",";
    file << "TTC Lidar" << ",";
    file << std::endl;

    for (auto &log_data : total_log_data)
    {
        for (size_t i = 0; i < log_data.n_images; i++)
        {
            file << i << ",";
            file << log_data.detectorType << ",";
            file << log_data.descriptorType << ",";
            file << log_data.matcherType << ",";
            file << log_data.selectorType << ",";
            file << log_data.det_timetaken[i] << ",";
            file << log_data.total_kpts[i] << ",";
            file << log_data.filtered_kpts[i] << ",";
            file << log_data.desc_timetaken[i] << ",";
            file << log_data.mat_timetaken[i] << ",";
            file << log_data.matched_pairs_n[i] << ",";
            if( i > 0){
                if(log_data.kptmatches_in_bb[i].size()>0)
                {
                    file << *(std::max_element(log_data.kptmatches_in_bb[i].begin(), 
                            log_data.kptmatches_in_bb[i].end()));
                } file << ",";
                if(log_data.TTCcamera[i].size()>0)
                {
                    file << (int)*(std::max_element(log_data.TTCcamera[i].begin(),
                            log_data.TTCcamera[i].end()));
                } file << ",";
                if(log_data.TTClidar[i].size()>0)
                {
                    file << (int)*(std::max_element(log_data.TTClidar[i].begin(),
                            log_data.TTClidar[i].end()));
                } file << ",";
            }
            else { file << ",,,";}
            file << std::endl;
        }
        file << std::endl;
    } file.close();
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{    
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    /* INIT VARIABLES AND DATA STRUCTURES */
    std::vector<Log_Database> total_log_data;
    initialize_log_vector(total_log_data, imgEndIndex+1);

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */
    for (auto &log_data : total_log_data)
    {
        dataBuffer.clear();

        for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
        {
            /* LOAD IMAGE INTO BUFFER */

            // assemble filenames for current index
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            // load image from file 
            cv::Mat img = cv::imread(imgFullFilename);

            // push image into data frame buffer
            DataFrame frame;
            frame.cameraImg = img;
            dataBuffer.push_back(frame);

            cout << "IMAGE NO: " << imgIndex << endl;
            //cout << "DETECTOR/ DESCRIPTOR COMBINATION: " << log_data.detectorType << "-" << log_data.descriptorType << endl; 
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


            /* DETECT & CLASSIFY OBJECTS */

            float confThreshold = 0.2;
            float nmsThreshold = 0.4;        
            detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                        yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

            cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


            /* CROP LIDAR POINTS */

            // load 3D Lidar points from file
            string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
            std::vector<LidarPoint> lidarPoints;
            loadLidarFromFile(lidarPoints, lidarFullFilename);

            // remove Lidar points based on distance properties
            float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
            cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
        
            (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

            cout << "#3 : CROP LIDAR POINTS done" << endl;


            /* CLUSTER LIDAR POINT CLOUD */

            // associate Lidar points with camera-based ROI
            float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
            clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

            // Visualize 3D objects
            bVis = false;
            if(bVis)
            {
                show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(1000, 1000), true);
            }
            bVis = false;

            cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
            
            
            // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
            //continue; // skips directly to the next image without processing what comes beneath

            /* DETECT IMAGE KEYPOINTS */

            // convert current image to grayscale
            cv::Mat imgGray;
            cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

            
            // extract 2D keypoints from current image
            vector<cv::KeyPoint> keypoints; // create empty feature list for current image
            string detectorType = log_data.detectorType;

            std::pair<int, double> detInfo;

            if (detectorType.compare("SHITOMASI") == 0)
            {
                //std::cout << "Probe point" << std::endl;
                detInfo = detKeypointsShiTomasi(keypoints, imgGray, false);
            }
            else if (detectorType.compare("HARRIS") == 0)
            {
                detInfo = detKeypointsHarris(keypoints, imgGray, false);
            }
            else
            {
                detInfo = detKeypointsModern(keypoints, imgGray, detectorType, false);
            }
            log_data.total_kpts[imgIndex] = detInfo.first;
            log_data.det_timetaken[imgIndex] = detInfo.second;

            // optional : limit number of keypoints (helpful for debugging and learning)
            bool bLimitKpts = false;
            if (bLimitKpts)
            {
                int maxKeypoints = 50;

                if (detectorType.compare("SHITOMASI") == 0)
                { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                }
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                cout << " NOTE: Keypoints have been limited!" << endl;
            }

            // push keypoints and descriptor for current frame to end of data buffer
            (dataBuffer.end() - 1)->keypoints = keypoints;

            cout << "#5 : DETECT KEYPOINTS done" << endl;


            /* EXTRACT KEYPOINT DESCRIPTORS */

            cv::Mat descriptors;
            string descriptorType = log_data.descriptorType; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
            double tt = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
            log_data.desc_timetaken[imgIndex] = tt;

            // push descriptors for current frame to end of data buffer
            (dataBuffer.end() - 1)->descriptors = descriptors;

            cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


            if (dataBuffer.size() > 1) // wait until at least two images have been processed
            {

                /* MATCH KEYPOINT DESCRIPTORS */

                vector<cv::DMatch> matches;
                string matcherType = log_data.matcherType;        // MAT_BF, MAT_FLANN
                string descriptorKind;
                if (log_data.descriptorType == "SIFT")
                    descriptorKind = "DES_HOG";
                else
                    descriptorKind = "DES_BINARY"; // DES_BINARY, DES_HOG
                string selectorType = log_data.selectorType;       // SEL_NN, SEL_KNN

                std::pair<int, double> matInfo = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                matches, descriptorKind, matcherType, selectorType);
                log_data.matched_pairs_n[imgIndex] = matInfo.first;
                log_data.mat_timetaken[imgIndex] = matInfo.second;

                // store matches in current data frame
                (dataBuffer.end() - 1)->kptMatches = matches;

                cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                
                /* TRACK 3D OBJECT BOUNDING BOXES */

                //// STUDENT ASSIGNMENT
                //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
                map<int, int> bbBestMatches;
                matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
                //// EOF STUDENT ASSIGNMENT

                // store matches in current data frame
                (dataBuffer.end()-1)->bbMatches = bbBestMatches;

                cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


                /* COMPUTE TTC ON OBJECT IN FRONT */

                // loop over all BB match pairs
                for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
                {
                    // find bounding boxes associates with current match
                    BoundingBox *prevBB, *currBB;
                    for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                    {
                        if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                        {
                            currBB = &(*it2);
                        }
                    }

                    for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                    {
                        if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                        {
                            prevBB = &(*it2);
                        }
                    }
                    //cout << "Probe point" << currBB->lidarPoints.size() << endl;
                    //cout << prevBB->lidarPoints.size() << endl;

                    // compute TTC for current match
                    if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                    {
                        //// STUDENT ASSIGNMENT
                        //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                        double ttcLidar; 
                        computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                        //cout << "Probe point2" << endl;
                        log_data.TTClidar.at(imgIndex).push_back(ttcLidar);
                        //cout << "TTC Lidar: " << ttcLidar << endl;
                        //// EOF STUDENT ASSIGNMENT

                        //// STUDENT ASSIGNMENT
                        //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                        //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                        double ttcCamera;
                        //cout << "Probe point3" << endl;
                        clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                        log_data.kptmatches_in_bb.at(imgIndex).push_back (currBB->kptMatches.size());
                        //cout << "Probe point 4" << endl;
                        //cout << "keypoint sizes: " << ((dataBuffer.end() - 2)->keypoints).size() << " and " << ((dataBuffer.end() - 1)->keypoints).size() << endl;
                        //cout << "kpt matches in currBB: " << (currBB->kptMatches).size() << endl;

                        computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                        //cout << "Probe point 5" << endl;
                        log_data.TTCcamera.at(imgIndex).push_back(ttcCamera);
                        cout << "TTC Camera: " << ttcCamera << endl;
                        //// EOF STUDENT ASSIGNMENT

                        cout << "#8 : COMPUTE TTC FOR CAMERA AND LIDAR done" << endl;

                        bVis = false;
                        if (bVis)
                        {
                            cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                            showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                            cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                            
                            char str[200];
                            sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                            putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                            string windowName = "Final Results : TTC";
                            cv::namedWindow(windowName, 4);
                            cv::imshow(windowName, visImg);
                            cout << "Press key to continue to next frame" << endl;
                            cv::waitKey(0);
                        }
                        bVis = false;

                    } // eof TTC computation
                } // eof loop over all BB matches 
                
            }

        } // eof loop over all images
    }
    writeToFile(total_log_data);
    return 0;
}
