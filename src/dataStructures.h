
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <algorithm>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
    int count = 0;
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

struct Log_Database { // represents the log information of every possible combinations of detector/descriptor types
    
    std::string detectorType, descriptorType, matcherType, selectorType;
    const size_t n_images;

    std::array<double, 30> det_timetaken, desc_timetaken, mat_timetaken;
    std::array<int, 30> total_kpts, filtered_kpts, matched_pairs_n;
    std::vector<std::vector<int>> TTCcamera {std::vector<std::vector<int>>(n_images, std::vector<int>(0))};
    std::vector<std::vector<int>> kptmatches_in_bb {std::vector<std::vector<int>>(n_images, std::vector<int>(0))};
    std::vector<std::vector<int>> TTClidar {std::vector<std::vector<int>>(n_images, std::vector<int>(0))};
    
    //constructor
    Log_Database(size_t num_of_images, std::string detectorT, std::string descriptorT, std::string matcherT, std::string selectorT)
        : detectorType{detectorT}, descriptorType{descriptorT}, matcherType{matcherT}, selectorType{selectorT}, n_images{num_of_images} {}

};
#endif /* dataStructures_h */
