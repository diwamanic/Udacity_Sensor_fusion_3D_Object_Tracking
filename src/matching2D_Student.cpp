
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
std::pair<int, double> matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorKind, std::string matcherType, std::string selectorType)
{
    int num_total_Kpts;
    double time_taken;
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType == "MAT_BF")
    {
        /*int normType = cv::NORM_HAMMING;
        if (descriptorKind.compare("SIFT") == 0){
            normType = cv::NORM_L1;
        }*/

        int normType { ((descriptorKind == "DES_BINARY") ? cv::NORM_HAMMING : cv::NORM_L2) };
        matcher = cv::BFMatcher::create(normType, crossCheck);
        if (descSource.type() == CV_32F)
        {
            //OpenCV bug workaround: convert binary descriptors to floating point due to a bug in current OpenCV implementation
            //descSource.convertTo(descSource, CV_8U);
            //descRef.convertTo(descRef, CV_8U);
        }
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            //OpenCV bug workaround: convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    double t = (double)cv::getTickCount();
    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k = 2;
        std::vector<std::vector<cv::DMatch>> knn_Matches;
        matcher->knnMatch(descSource, descRef, knn_Matches, k); //Finds the first two best matches for each descriptor

        const float ratio_thresh = 0.8f;
        for (size_t i = 0; i < (int) knn_Matches.size(); i++)
        {
            if( knn_Matches[i][0].distance < ratio_thresh * knn_Matches[i][1].distance)
            {
                matches.push_back(knn_Matches[i][0]);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << matcherType << " matching for " << matches.size() << " pairs in " << 1000 * t / 1.0 << " ms" << endl;
    time_taken = 1000 * t / 1.0;
    num_total_Kpts = matches.size();
    return std::make_pair(num_total_Kpts, time_taken);
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    //BRIEF; ORB; FREAK; AKAZE; SIFT
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return (1000 * t / 1.0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
std::pair<int, double> detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int num_total_Kpts;
    double time_taken;
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    time_taken = 1000 * t / 1.0;
    num_total_Kpts = keypoints.size();

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
    return std::make_pair(num_total_Kpts, time_taken);
}

// Detect keypoints in image using the traditional Harris detector
std::pair<int, double> detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int num_total_Kpts;
    double time_taken;
    // compute detector parameters based on image size
    int blockSize = 2;       // size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int apertureSize = 3;    // aperture parameter for the sobel operator
    double k = 0.04;         // harris detector free parameter
    int thresh = 100;
    
    // Apply corner detection
    double t = (double)cv::getTickCount();
    cv::Mat dst = cv::Mat::zeros( img.size(), CV_32FC1);
    cv::cornerHarris( img, dst, blockSize, apertureSize, k);

    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs( dst_norm, dst_norm_scaled);

    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if((int) dst_norm.at<float>(i,j) > thresh)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f( j, i);
                newKeyPoint.size = blockSize;
                keypoints.push_back(newKeyPoint);
            }
        }
    }
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    time_taken = 1000 * t / 1.0;
    num_total_Kpts = keypoints.size();

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return std::make_pair(num_total_Kpts, time_taken);
}

// Detect keypoints in image using the FAST, BRISK, ORB, AKAZE, SIFT detector
std::pair<int, double> detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{   
    int num_total_Kpts;
    double time_taken;
    // Step 1: Detect the keypoints using SIFT Detector
    cv::Ptr<cv::FeatureDetector> detector;
    double t = (double)cv::getTickCount();
    if(detectorType.compare("FAST") == 0)
    {
        detector = cv::FastFeatureDetector::create();
    } 
    else if (detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SiftFeatureDetector::create();
    }
    
    detector->detect(img, keypoints);
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    time_taken = 1000 * t / 1.0;
    num_total_Kpts = keypoints.size();

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return std::make_pair(num_total_Kpts, time_taken);
}