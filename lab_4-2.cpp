#include <cv.h>
#include <highgui.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/Legacy/Legacy.hpp>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;

void main()
{
	//Load two source image.
    Mat leftImg = imread( "s8.jpg" ) ;
	Mat rightImg = imread( "s9.jpg" ) ;

	//Variable of 1.left & right sift image 2.left & right temp image 3.merged image
    Mat siftLeftImg , siftRightImg , ltemp , rtemp , mrgImg ;

	//Descriptor 1 & 2.
	Mat des1 , des2 ;

    vector< KeyPoint > keypointsLeft , keypointsRight;
	vector< KeyPoint > vecContainer ;
	vector< DMatch > matchVec ;

	SIFT siftLeft , siftRight ;

	//Detect feature point from "leftImg" and draw it out.
	siftLeft.detect( leftImg , keypointsLeft , Mat( ) ) ;
	drawKeypoints( leftImg , keypointsLeft , ltemp , Scalar( 0 , 255 , 255 ) ) ;
	//imshow( "Left Image" , ltemp ) ;

	//Detect feature point from "rightImg" and draw it out.
	siftRight.detect( rightImg , keypointsRight , Mat( ) ) ;
	drawKeypoints( rightImg , keypointsRight , rtemp , Scalar( 0 , 255 , 255 ) ) ;
	//imshow( "Right Image" , rtemp ) ;

	//Compute the descriptor for a set of key points detected in image "leftImg" & "rightImg".
	siftLeft.compute( leftImg , keypointsLeft , des1 ) ;
	siftRight.compute( rightImg , keypointsRight , des2 ) ;

	//Match
	BruteForceMatcher<L2<float>> matcher ;
	matcher.match( des1 , des2 , matchVec ) ;

	//Filter the matches.
	nth_element( matchVec.begin( ) , matchVec.begin( ) + 24 , matchVec.end( ) ) ;
	matchVec.erase( matchVec.begin( ) + 25 , matchVec.end( ) ) ;

	//Declaration of variable for file name.
	string fileName ;
	char buffer[ 10 ] ;

	//Merge the left and right image into a bigger image.
	for( int i = 0 ; i < matchVec.size( ) ; i ++ )
	{
		mrgImg.create( leftImg.rows , keypointsRight[ matchVec[ i ].trainIdx ].pt.x + rightImg.cols , leftImg.type( ) ) ;
		leftImg.copyTo( mrgImg( Rect( 0 , 0 , leftImg.cols , leftImg.rows ) ) ) ;
		rightImg.copyTo( mrgImg( Rect( keypointsRight[ matchVec[ i ].trainIdx ].pt.x , 0 , rightImg.cols , rightImg.rows ) ) ) ;

		imshow( "merge image" , mrgImg ) ;

		//Write the image into disk, and the 10th image seems like the best one.
		fileName = itoa( i + 1 , buffer , 10 ) ;
		fileName += ".jpg" ;
		imwrite( fileName , mrgImg ) ;
		waitKey( 1000 ) ;
	}

	//===============Different kind of matches.===============
	//Knn match
	//vector< vector < DMatch >> knn_matches ;
	//matcher.knnMatch( des1 , des2 , knn_matches , 5 ) ;

	//Radus match
	//int RADIUS_MAX = 200 ;
	//vector< vector < DMatch >> radius_matches , finalMatch;
	//matcher.radiusMatch( des1 , des2 , radius_matches , RADIUS_MAX ) ;

	//drawMatches( leftImg , keypointsLeft , rightImg , keypointsRight , matchVec , mrgImg , Scalar :: all( -1 ) ) ;//match
	//drawMatches( leftImg , keypointsLeft , rightImg , keypointsRight , knn_matches , mrgImg , Scalar :: all( -1 ) ) ;//knnMatch
	//drawMatches( leftImg , keypointsLeft , rightImg , keypointsRight , radius_matches , mrgImg , Scalar :: all( -1 ) ) ;//radiusMatch

	//imshow( "Merge Image" , mrgImg ) ;

	std :: cout << "It seems the 10th image like the best one." << endl ;
    waitKey( 0 ) ;
    return 0 ;
}
