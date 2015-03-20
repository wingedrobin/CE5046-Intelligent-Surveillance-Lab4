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
    Mat image = imread( "frequency.jpg" ) ;
    Mat siftImg , surf64Img , surf128Img ;
    vector< KeyPoint > keypointsSIFT ;
	vector< KeyPoint > keypointsSURF64 ;
	vector< KeyPoint > keypointsSURF128 ;
	
	SIFT sift ;
	SURF surf64( 400 , 4 , 2 , false , false ) ;
	SURF surf128( 400 , 4 , 2 , true , false ) ;

	sift.detect( image , keypointsSIFT , Mat( ) ) ;
	drawKeypoints( image , keypointsSIFT , siftImg , Scalar( -1 , -1 , -1 , -1 ) , DrawMatchesFlags :: DRAW_RICH_KEYPOINTS ) ;
	imshow( "SIFT" , siftImg ) ;
	imwrite( "sift.jpg", siftImg );

	surf64.detect( image , keypointsSURF64 , Mat( ) ) ;
	drawKeypoints( image , keypointsSURF64 , surf64Img , Scalar( -1 , -1 , -1 , -1 ) , DrawMatchesFlags :: DRAW_RICH_KEYPOINTS ) ;
	imshow( "SURF64" , surf64Img ) ;
	imwrite( "suft64.jpg", surf64Img );

	surf128.detect( image , keypointsSURF128 , Mat( ) ) ;
	drawKeypoints( image , keypointsSURF128 , surf128Img , Scalar( -1 , -1 , -1 , -1 ) , DrawMatchesFlags :: DRAW_RICH_KEYPOINTS ) ;
	imshow( "SURF128" , surf128Img ) ;
	imwrite( "suft128.jpg", surf128Img );

    waitKey( 0 ) ;
}
