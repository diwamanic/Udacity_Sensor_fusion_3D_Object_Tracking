
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>
#include <algorithm>

#include "camFusion.hpp"
#include "dataStructures.h"


using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
                if(enclosingBoxes.size()>1)
                {
                    lidarPoints.erase(it1);
                    it1--;
                    break;
                }
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }
        enclosingBoxes.clear();

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<double> euclidean_distance(kptMatches.size());
    for( auto it = kptMatches.begin(); it!= kptMatches.end(); it++)
    {
        auto Curr_kpt = kptsCurr[it->trainIdx];
        auto Prev_kpt = kptsPrev[it->queryIdx];
        auto diff = Curr_kpt.pt - Prev_kpt.pt;
        euclidean_distance.push_back(cv::sqrt(diff.x*diff.x + diff.y*diff.y));
    }
    double mean = std::accumulate(euclidean_distance.begin(), euclidean_distance.end(), 0)/(int) (euclidean_distance.size());
    double mean_with_tolerance = mean * 1.1;

    for(auto i = 0; i< kptMatches.size(); i++)
    {
        if(euclidean_distance[i] < mean_with_tolerance){
            if(boundingBox.roi.contains(kptsCurr[kptMatches[i].trainIdx].pt))
            {
                boundingBox.kptMatches.push_back(kptMatches[i]);
                boundingBox.keypoints.push_back(kptsCurr[kptMatches[i].trainIdx]);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    if (kptMatches.size()>0)
    {
        // compute distance ratios between all matched keypoints
        vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
        for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
        { // outer keypoint loop

            // get current keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
            cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

            for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
            { // inner keypoint loop

                double minDist = 100; // min. required distance

                // get next keypoint and its matched partner in the prev. frame
                cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
                cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

                //cout << "Probe point 6" << endl;
                // compute distances and distance ratios
                double distCurr = cv::norm(cv::Mat(kpOuterCurr.pt), cv::Mat(kpInnerCurr.pt));
                double distPrev = cv::norm(cv::Mat(kpOuterPrev.pt), cv::Mat(kpInnerPrev.pt));

                //cout << "Probe point 7" << endl;
                if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist
                    && distCurr != distPrev)
                { // avoid division by zero

                    double distRatio = distCurr / distPrev;
                    distRatios.push_back(distRatio);
                }
            } // eof inner loop over all matched kpts
        }     // eof outer loop over all matched kpts
        //cout << "Probe point 8" << distRatios.size() << endl;
        // only continue if list of distance ratios is not empty
        if (distRatios.size() == 0)
        {
            TTC = NAN;
            return;
        }

        // compute camera-based TTC from distance ratios
        //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

        double dT = 1 / frameRate;
        //TTC = -dT / (1 - meanDistRatio);

        // TODO: STUDENT TASK (replacement for meanDistRatio)
        std::sort(distRatios.begin(), distRatios.end(), std::greater<int>());
        //cout << "Probe point 9" << endl;
        double medianDistRatio;
        size_t size = distRatios.size();

        if(size != 0)
        {
            if(size %2 == 0)
                medianDistRatio = 0.5 * (distRatios.at(size/2) + distRatios.at(size/2 + 1));
            else
                medianDistRatio = distRatios.at(size/2 + 1);     
        }
        //cout << "Probe point 10" << medianDistRatio << endl;
        TTC = -dT / (1 - medianDistRatio);
        //cout << "Probe point 11- dT: " << dT << "- TTC: " << TTC << endl;
    } else {
        TTC = NAN;
        cout << "There are no kptmatches in the bounding box" << endl;
    }
}



void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double minXPrev = 1e9, minXCurr = 1e9;
    double mean_X_prev {0}; double mean_X_curr {0};
    for (const LidarPoint &point: lidarPointsPrev){
        mean_X_prev = mean_X_prev + point.x;
    }
    mean_X_prev = mean_X_prev/lidarPointsPrev.size();

    double var = 0.0;
    for(const LidarPoint &point: lidarPointsPrev) {
        var = var + (point.x - mean_X_prev)*(point.x - mean_X_prev); 
    } double std_dev = cv::sqrt(var);
    //Keeping one S.D. away from the mean
    double threshold = mean_X_prev - 3 * std_dev;
    lidarPointsPrev.erase(std::remove_if(lidarPointsPrev.begin(), lidarPointsPrev.end(),
                            [&](LidarPoint pt){return (pt.x < threshold);})
                            ,lidarPointsPrev.end());

    for(auto it = lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); it++){
        minXPrev = minXPrev > it->x ? it->x : minXPrev;  
    }
    cout << "first X value of a Lidar points in prev: " << minXPrev << endl;

    for (const LidarPoint &point: lidarPointsCurr){
        mean_X_curr = mean_X_curr + point.x;
    }
    mean_X_curr = mean_X_curr / lidarPointsCurr.size();

    var = 0.0;
    for(const LidarPoint &point: lidarPointsPrev) {
        var = var + (point.x - mean_X_prev)*(point.x - mean_X_prev); 
    } std_dev = cv::sqrt(var);
    //Keeping one S.D. away from the mean
    threshold = mean_X_prev - 3 * std_dev;
    lidarPointsCurr.erase(std::remove_if(lidarPointsCurr.begin(), lidarPointsCurr.end(),
                            [&](LidarPoint pt){return (pt.x < threshold);})
                            ,lidarPointsCurr.end());

    for(auto it = lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); it++){
        minXCurr = minXCurr > it->x ? it->x : minXCurr;  
    }
    cout << "first X value of a Lidar points in curr: " << minXCurr << endl;
    cout << "frame Rate: " << frameRate << endl;
    
    if(minXPrev != minXCurr){
        TTC = minXCurr / (frameRate * (minXPrev - minXCurr));
        cout << "TTC Lidar:inside  "<< TTC << endl;
    }
    //else TTC = NAN;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::vector<std::vector<int>> map_bb_selection(currFrame.boundingBoxes.size()+1, 
                                                    std::vector<int>(prevFrame.boundingBoxes.size()+1,0));
    std::cout << "BOUNDING BOX MATCHING STARTS" << std::endl;
    for(auto it = matches.begin(); it!= matches.end(); it++)
    {
        auto m1_prevKpt = prevFrame.keypoints[it->queryIdx];
        auto m1_currKpt = currFrame.keypoints[it->trainIdx];
        
        int m1_currBoxID {(int) (currFrame.boundingBoxes.size())}, m1_prevBoxID {(int) (prevFrame.boundingBoxes.size())};

        int count = 0;
        for(auto it2 = currFrame.boundingBoxes.begin(); it2!= currFrame.boundingBoxes.end(); it2++)
        {
            if(!it2->roi.contains(m1_currKpt.pt))
                continue;
            if(++count > 1){
                matches.erase(it);
                it--;
                m1_currBoxID = (int) (currFrame.boundingBoxes.size());
                break;
            }
            if(count == 1) m1_currBoxID = it2->boxID;
        }
        if(count == 1)
        {
            count = 0;
            for(auto it2 = prevFrame.boundingBoxes.begin(); it2!= prevFrame.boundingBoxes.end(); it2++)
            {
                if(!it2->roi.contains(m1_prevKpt.pt))
                    continue;
                if(++count > 1){
                    matches.erase(it);
                    it--;
                    m1_prevBoxID = (int) (prevFrame.boundingBoxes.size());
                    break;
                }
                if(count == 1) m1_prevBoxID = it2->boxID;
            }
        }
        map_bb_selection.at(m1_currBoxID).at(m1_prevBoxID)++;
    }
    for(auto it = map_bb_selection.begin(); it!=map_bb_selection.end(); it++)
    {
        int b2_prev = (std::max_element(it->begin(), it->end()) - it->begin());
        bbBestMatches.insert(std::make_pair(b2_prev, (it-map_bb_selection.begin())));
    }
    /*for(auto it = prevFrame.boundingBoxes.begin(); it!= prevFrame.boundingBoxes.end(); it++)
    {
        prevFrame.keypoints.erase(std::remove_if(prevFrame.keypoints.begin(), prevFrame.keypoints.end(), 
                                    [&](cv::KeyPoint kpt){
                                        if(it->roi.contains(kpt.pt))
                                        {
                                            it->keypoints.push_back(kpt);
                                            return true;
                                        }
                                        return false;
                                    })
                                ,prevFrame.keypoints.end());
    }
    //By looping over matches and bounding boxes, group the matches that is enclosed by each bounding box and store those
    //matches in Boundingbox object 
    
    for(auto it = currFrame.boundingBoxes.begin(); it != currFrame.boundingBoxes.end(); it++)
    {
        currFrame.kptMatches.erase(std::remove_if(currFrame.kptMatches.begin(), currFrame.kptMatches.end(),
                                    [&](cv::DMatch kptmatch){
                                        if(it->roi.contains(currFrame.keypoints[kptmatch.trainIdx].pt))
                                        {   
                                            it->kptMatches.push_back(kptmatch);
                                            it->keypoints.push_back(currFrame.keypoints[kptmatch.trainIdx]);
                                            return true; 
                                        }
                                        return false;
                                    })
                                    ,currFrame.kptMatches.end());
        
        int max_count = 0;
        int bb_max_index = 0;
        for(auto it2= prevFrame.boundingBoxes.begin(); it2!=prevFrame.boundingBoxes.end(); it2++)
        {
            for(auto it3 = it->kptMatches.begin(); it3!=it->kptMatches.end(); it3++)
            {   auto x = it3->queryIdx;
                cv::KeyPoint kpt_Prev_frame = prevFrame.keypoints[x];
            
                //int temp = it2->count;
                std::for_each(it2->keypoints.begin(), it2->keypoints.end(),
                            [&](cv::KeyPoint kpt){
                                if ((kpt.pt.x == kpt_Prev_frame.pt.x) && (kpt.pt.y == kpt_Prev_frame.pt.y))
                                {   
                                    it2->count++;
                                }
                            });
                //if(it2->count > temp) { break; };
            }
            if(it2->count > max_count)
            {
                max_count = it2->count;
                bb_max_index = it2->boxID;
            }
            //auto loc = prevFrame.boundingBoxes.begin();
            /*auto loc = std::find_if(prevFrame.boundingBoxes.begin(), prevFrame.boundingBoxes.end(), 
                                  [&](BoundingBox b1){ 
                                      return (std::find(b1.keypoints.begin(), b1.keypoints.end(), kpt_Prev_frame)!=b1.keypoints.end());
                                    });
            if(loc != prevFrame.boundingBoxes.end())
                loc->count++;
               
        }*/


        /*BoundingBox bmax = std::max(prevFrame.boundingBoxes, [&](const BoundingBox &b1, const BoundingBox &b2)
                                            {
                                                return (b1.count < b2.count);
                                            });
                                            */
    /*    bbBestMatches.insert(std::make_pair(bb_max_index,it->boxID)); 
    }
*/
}

