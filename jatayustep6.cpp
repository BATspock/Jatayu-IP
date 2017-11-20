#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>


#include<math.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>


#include<python2.7/Python.h>
#include<fstream>
#include<iostream>








#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include<stdio.h>

#include <sys/time.h>
#include<iomanip>

#include<stdio.h>
#define MAX_IMGS 1000 //Max image count
#define MAX_ERROR 16 //Shape max error
#define PI 3.141592653589793238462643383279

//#define semi_view 0.5984188935 //Canon S110 semi vertical angle of camera in radians
//#define semi_view 0.508850743 //Canon IXUS - Vasishta semi vertical angle of camera in radians

#define semi_view 0.476783142 //Canon IXUS - Vasishta semi vertical angle of camera in radians new

void preprocessing(cv::Mat src);
void notarget(cv::Mat src);
void k_means(cv::Mat src);
void shapedetect(cv::Mat src);
int ocr(cv::Mat input1,cv::Mat input2, int x1, int x2);
void orient(float letter);
int bar(char *);
void gps(cv::Mat src);
void lookup(int r1, int g1, int b1, int r2, int g2, int b2 );
void lab(cv::Mat image1, cv::Mat image2);
void closest(float l,float a,float b, int fl);
void deltaE2000( double lab1[3], double lab2[3], double &delta_e );

float err(float ideal,float detected)
 {
    return(abs(ideal-detected));
 }

 static double length(cv::Point pt1, cv::Point pt0) //finds length between pt1 and pt0
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;

	return (sqrt(dx1*dx1+dy1*dy1));
}

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)  //calculate cosine of angle between 2 lines pt1-pt0 and pt2-pt0
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

using namespace std;
//using namespace zbar;
using namespace cv ;

int i = 0; //Input Image index
int noflag = 1; // Flag is set when no targets are detected

//fields for output.txt - Interoperability format
float output2, output3;

//fields for output - Jatayu.txt
/*
field 1 - Serial Numbers of detected targets
field 2 - Type
field 31, 32, 33 - Latitude
field 41, 42, 43 - Longitude
field 5 - Orientation
field 6 - Shape
field 7 - Background color
field 8 - Alphanumeric
field 9 - Alphanumeric color
field 10 - Image file name
field 11 - Description
*/

int field1;
char field2[3],field5[2],field6[20],field7[10],field8[1],field9[10],field10[10],field11[50];
int field31, field32, field41, field42;
float field33, field43;

int c1,c2,c3; //waiting variables for transmission
int target_counter = 0; //up counter for detected targets
int up=0; //supporting up counter for target_counter
float heading = 0;  //Yaw of the plane
int qrcflag;

//Lookup structure for standard colors in Lab colorspace
struct color
{
    float l;
    float a;
    float b;
    char name[20];
}clab[10];

int main()
{
    Mat src = imread("/home/rishita/Desktop/check/im6.jpg",CV_LOAD_IMAGE_UNCHANGED);
        FILE* fp; //file pointer for Jatayu.txt
    FILE* op; //file pointer for output.txt5
    int exit_count=0; //used to quit

    char buf[12] = "im1.JPG";
    /*ckfor(;;)
    {
        exit_count = 0;
        strcpy(field2,"STD"); strcpy(field5," "); strcpy(field6," "); strcpy(field7," "); strcpy(field8," "); strcpy(field9," "); strcpy(field10," ");
        strcpy(field11,"AutoDetect");
        qrcflag = 0; //set when QRC target is detected
        ++i;
        sprintf(buf,"im%d.jpeg",i);

        FILE* fp1;
        fp1=fopen("gps.txt","w");
        fprintf(fp1,"im%d",i);
        fclose(fp1);

        cout<<buf<<endl;
        Mat src = imread(buf,CV_LOAD_IMAGE_UNCHANGED);
        if(!src.data)
        {
            cout<<"No image with index "<<i<<endl;

            /*
            Py_Initialize();   //initialise python embedded code
            PyObject* PyFileObject = PyFile_FromString("getfile.py", "r");
            PyRun_SimpleFileEx(PyFile_AsFile(PyFileObject), "getfile.py", 1);
            Py_Finalize();

            src = imread(buf,CV_LOAD_IMAGE_UNCHANGED);
            if(!src.data)
            {
                for(c1=0 ; c1<1000; c1++)
                    for(c2=0 ; c2<1000; c2++)

                        for(c3=0 ; c3<1000; c3++)
                        {}
               waitKey(60*60*60*60*1000*5); //wait and check 3 times before exiting
                i--;
                exit_count++;
                if(exit_count == 3)
                    {
                        cout<<"End of task! Thank you :)";
                        exit(0);
                    }
                else
                    {
                        continue;
                    }
            }
              //retain

        }
        */  //ck

        if(src.data)
        {

     //Extracting metadata from image

    /*ck Py_Initialize();   //initialise python embedded code
     PyObject* PyFileObject = PyFile_FromString("extract.py", "r");
     PyRun_SimpleFileEx(PyFile_AsFile(PyFileObject), "extract.py", 1);
     Py_Finalize();
     */  //ck

            //qr code

            //qrcflag = bar(buf);    //calls QRC decoder on all images
//            target_counter += qrcflag;  //increases if QR code is detected
           // if(!qrcflag)
                notarget(src);  //continues to process to eliminate images without targets
          /*  field1 = target_counter;
            strcpy(field10,buf);

            //File I/O

            if(noflag==0)
            {
                gps(src);

                fp=fopen("final/Jatayu.txt","a");
                if(target_counter<10)
                    fprintf(fp,"0%d\t",field1);
                else
                    fprintf(fp,"%d\t",field1);

                 fprintf(fp,"%c%c%c\t",field2[0],field2[1],field2[2]);

                if(field41>=0 && field41<=99 )
                    fprintf(fp,"N%d %d %.3f\tW0%d %d %.3f\t",field31,field32,field33,field41,field42,field43);
                else
                    fprintf(fp,"N%d %d %.3f\tW%d %d %.3f\t",field31,field32,field33,field41,field42,field43);

                if(field8[0] == NULL || !((field8[0]>'A' && field8[0]<'Z') || (field8[0]>'a' && field8[0]<'z') || (field8[0]>'0' && field8[0]<'9')) )
                    fprintf(fp,"%-2s\t%-15s\t%-10s\t \t%-10s\t%-10s\t%-20s\n",field5,field6,field7,field9,field10,field11);
                else
                    fprintf(fp,"%-2s\t%-15s\t%-10s\t%c\t%-10s\t%-10s\t%-20s\n",field5,field6,field7,field8[0],field9,field10,field11);
                fclose(fp);

                op=fopen("final/output.txt","a");
                if(target_counter == 1)
                    fprintf(fp,"[ {");
                else
                    fprintf(fp,",\n{");
                fclose(op);

                if(!qrcflag)
                {
                    op=fopen("final/output.txt","a");
                    if(field8[0] == NULL)
                        fprintf(op,"\"type\": \"standard\", \"latitude\": %f, \"longitude\": %f, \"orientation\": \"%s\", \"shape\": \"%s\", \"background_color\": \"%s\", \"alphanumeric_color\": \"%s\", ",output2, output3, field5, field6, field7, field9);
                    else
                        fprintf(op,"\"type\": \"standard\", \"latitude\": %f, \"longitude\": %f, \"orientation\": \"%s\", \"shape\": \"%s\", \"background_color\": \"%s\", \"alphanumeric\": \"%c\", \"alphanumeric_color\": \"%s\", ",output2, output3, field5, field6, field7, field8[0], field9);
                    if(target_counter<7)
                    {
                        fprintf(op,"\"autonomous\": True");
                    }
                    fclose(op);
                }

                else
                {
                    op=fopen("final/output.txt","a");
                    fprintf(op,"\"type\": \"qrc\", \"latitude\": %f, \"longitude\": %f, \"description\": \"%s\", ",output2, output3, field11);

                    if(target_counter<7)
                    {
                        fprintf(op,"\"autonomous\": True");
                    }

                    fclose(op);
                }
                op=fopen("final/output.txt","a");
                fprintf(op," }");
                fclose(op);

            }
        }
        if(i == MAX_IMGS)
        {
            op=fopen("final/output.txt","a");
            fprintf(op," ]");
            fclose(op);

            fclose(fp);
            exit(0);
        }
        */
    }
 }



void notarget(cv::Mat src)
{
    int flag = 0;   //flag is sent when contour is found
    RNG rng(12345);     //random colours
    Mat image,img;
    pyrDown( src, img, Size( src.cols/2, src.rows/2 ),BORDER_DEFAULT );     //resizing
    fastNlMeansDenoisingColored(img,img,10,10,0,5);     //denoising algorithm
    Mat bgr[3];
    split(img,bgr);     //splits B, G, R components

    for(int j=0;j<3;j++)
        {
            image = bgr[j];
            //blur( image, image, Size(3,3) );
            Canny(image, image, 80, 255, 3);
                   //edge detection
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            for( int i = 0; i< contours.size(); i++ )
            {
                if(hierarchy[i][2]!=-1 && arcLength(contours[i],true)>60)           //Accept only closed contours with children and arc length of contour > 60
                    {
                        flag = 1;
                        break;
                    }
            }
        }
        if(flag==1)
            preprocessing(src);
        else
        {
            cout<<"no target"<<endl;
            noflag = 1;
        }

    return;
}

void preprocessing(cv::Mat src)
{
    int flag = 0;
    RNG rng(12345);
    Mat img = src;
    pyrDown( src, src, Size( src.cols/2, src.rows/2 ) );
    Mat shape = Mat::zeros( src.size(), CV_8UC3 );

   fastNlMeansDenoisingColored(src,src,10,10,0,21);    //increase factor to eliminate further images with no targets - takes a while to compute

    Mat image,imageROI,finalimage;
    Mat bgr[3];
    split(src,bgr);
     namedWindow("image5",CV_WINDOW_NORMAL);
    imshow( "image5",src );
    namedWindow("image6",CV_WINDOW_NORMAL);
    imshow( "image6", bgr[1] );
    namedWindow("image7",CV_WINDOW_NORMAL);
    imshow( "image7", bgr[2] );
    waitKey(0);

Mat mask = Mat::zeros(src.size(), CV_8UC1);
    for(int j=0;j<3;j++)
        {
        image = bgr[j];
  //  blur( image, image, Size(3,3) );

        Canny(image, image, 80, 255, 3);
        namedWindow("canny",CV_WINDOW_NORMAL);
        imshow("canny",image);

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        RNG rng(12345);
        findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );
        Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
        Mat shape = Mat::zeros( image.size(), CV_8UC3 );

        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            if(hierarchy[i][2]!=-1 && arcLength(contours[i],true)>60)///all contours that have child & length > 60
                {
                flag = 1;
                drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
                if(hierarchy[hierarchy[i][2]][2]!=-1)
                    {
                        drawContours( shape, contours, i, color, 1, 8, hierarchy, 0, Point() );
                    }

                drawContours(mask, contours, i, Scalar(255), CV_FILLED);
                }
            Rect roi = boundingRect(contours[i]);
            src.copyTo(imageROI, mask);

        }
         namedWindow("im12",CV_WINDOW_NORMAL);
    imshow( "im12", imageROI );
   // waitKey(0);
    /*  namedWindow("image5",CV_WINDOW_NORMAL);
    imshow( "image5", drawing );
    namedWindow("image6",CV_WINDOW_NORMAL);
    imshow( "image6", shape );
    namedWindow("image7",CV_WINDOW_NORMAL);
    imshow( "image7", mask );
*/
        }

        if(flag==0)
            {
            cout<<"no target"<<endl;
            noflag = 1;
            return;
            }

    noflag = 0;
    if(!qrcflag)
        target_counter++;

    namedWindow("image1",CV_WINDOW_NORMAL);
    imshow( "image1", src );

    int largest_area=0;
    int largest_contour_index=0;
    Rect bounding_rect;

    Mat thr,imageROI2;
    imageROI2 = imageROI;

    //cvtColor( imageROI2, thr, COLOR_BGR2GRAY ); //Convert to gray
    //threshold( thr, thr, 125, 255, THRESH_BINARY ); //Threshold the gray

    Mat stuff = Mat::zeros(imageROI2.size(), CV_8UC1);

    vector<vector<Point> > contours; // Vector for storing contours

    findContours( mask, contours,RETR_CCOMP, CHAIN_APPROX_SIMPLE ); // Find the contours in the image
    for( size_t i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

    drawContours( stuff, contours, i, color,1,8);

        double area = contourArea( contours[i] );  //  Find the area of contour

        if( area > largest_area )
        {
            largest_area = area;
            largest_contour_index = i;               //Store the index of largest contour
        }
    }
    namedWindow("check",CV_WINDOW_NORMAL);
    imshow("check",stuff);
bounding_rect = boundingRect( contours[largest_contour_index] );
char location[100];
sprintf(location,"final/target%d.jpeg",target_counter);

imwrite( location, imageROI(bounding_rect) );
if(!qrcflag)
    k_means(imageROI(bounding_rect));
shape=cv::Scalar(0,0,0);

Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
drawContours( shape, contours, largest_contour_index, Scalar(255,255,255),CV_FILLED);
int morph_size = 2;
Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
morphologyEx( shape, shape, MORPH_OPEN, element );
namedWindow("sh",CV_WINDOW_NORMAL);
imshow("sh",shape);
waitKey(0);

if(!qrcflag)
    shapedetect(shape);

    return;
}

void k_means(cv::Mat src)
{
    namedWindow("image",CV_WINDOW_NORMAL);
    imshow( "image", src );
int flag1 = -1,flag2 = -1, ocrflag = -1;
int r1,r2,b1,b2,g1,g2;
Mat samples(src.rows * src.cols, 3, CV_32F);
  for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
      for( int z = 0; z < 3; z++)
        samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];

  int clusterCount = 3;
  Mat labels;
  int attempts = 5;
  Mat centers;
  kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 5000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

   Mat new_image8( src.size(), src.type() );
   Mat new_image9( src.size(), src.type() );
   Mat new_image10( src.size(), src.type() );

   Mat imgg2( src.size(), src.type() );
   Mat imgg3( src.size(), src.type() );
   Mat imgg4( src.size(), src.type() );

flag1 = -1;

       for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
    {
      int cluster_idx = labels.at<int>(y + x*src.rows,0);
      if( cluster_idx == 2)
      {
      new_image8.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
      new_image8.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
      new_image8.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);

      if(flag1 == -1)

        if(!(   new_image8.at<Vec3b>(y,x)[0] >= 0 && new_image8.at<Vec3b>(y,x)[0] <= 30 &&
                new_image8.at<Vec3b>(y,x)[1] >= 0 && new_image8.at<Vec3b>(y,x)[1] <= 30 &&
                new_image8.at<Vec3b>(y,x)[2] >= 0 && new_image8.at<Vec3b>(y,x)[2] <= 30 ))
            flag1 = 0;
      }
      else
       {
      new_image8.at<Vec3b>(y,x)[0] = 1;
      new_image8.at<Vec3b>(y,x)[1] = 1;
      new_image8.at<Vec3b>(y,x)[2] = 1;
      }

    }

if(flag1 == 0 )
{
for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
    {
      imgg2.at<Vec3b>(y,x)[0] =centers.at<float>(2, 0);
      imgg2.at<Vec3b>(y,x)[1] =centers.at<float>(2, 1);
      imgg2.at<Vec3b>(y,x)[2] =centers.at<float>(2, 2);
    }
}
else
{
    flag2 = 2 ;
}

flag1 = -1;

    for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
    {
      int cluster_idx = labels.at<int>(y + x*src.rows,0);
      if( cluster_idx == 1)
      {
      new_image9.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
      new_image9.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
      new_image9.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);

      if(flag1 == -1)

        if(!(   new_image9.at<Vec3b>(y,x)[0] >= 0 && new_image9.at<Vec3b>(y,x)[0] <= 30 &&
                new_image9.at<Vec3b>(y,x)[1] >= 0 && new_image9.at<Vec3b>(y,x)[1] <= 30 &&
                new_image9.at<Vec3b>(y,x)[2] >= 0 && new_image9.at<Vec3b>(y,x)[2] <= 30 ))
            flag1 = 0;
      }
      else
       {
      new_image9.at<Vec3b>(y,x)[0] = 1;
      new_image9.at<Vec3b>(y,x)[1] = 1;
      new_image9.at<Vec3b>(y,x)[2] = 1;
      }

    }
if(flag1 == 0)
{
for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
    {
      imgg3.at<Vec3b>(y,x)[0] =centers.at<float>(1, 0);
      imgg3.at<Vec3b>(y,x)[1] =centers.at<float>(1, 1);
      imgg3.at<Vec3b>(y,x)[2] =centers.at<float>(1, 2);
    }
}
else
{
    flag2 = 1 ;
}

flag1 = -1;

    for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
    {
      int cluster_idx = labels.at<int>(y + x*src.rows,0);
      if( cluster_idx == 0)
      {
      new_image10.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
      new_image10.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
      new_image10.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);

      if(flag1 == -1)
      if(!(   new_image10.at<Vec3b>(y,x)[0] >= 0 && new_image10.at<Vec3b>(y,x)[0] <= 30 &&
                new_image10.at<Vec3b>(y,x)[1] >= 0 && new_image10.at<Vec3b>(y,x)[1] <= 30 &&
                new_image10.at<Vec3b>(y,x)[2] >= 0 && new_image10.at<Vec3b>(y,x)[2] <= 30 ))
            flag1 = 0;
      }
      else
       {
      new_image10.at<Vec3b>(y,x)[0] = 1;
      new_image10.at<Vec3b>(y,x)[1] = 1;
      new_image10.at<Vec3b>(y,x)[2] = 1;
      }

    }
if(flag1 == 0)
{
for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
    {
      imgg4.at<Vec3b>(y,x)[0] =centers.at<float>(0, 0);
      imgg4.at<Vec3b>(y,x)[1] =centers.at<float>(0, 1);
      imgg4.at<Vec3b>(y,x)[2] =centers.at<float>(0, 2);
    }
}
else
{
    flag2 = 0 ;
}

/*namedWindow("8",CV_WINDOW_NORMAL);
imshow( "8", new_image8 );
namedWindow("9",CV_WINDOW_NORMAL);
imshow( "9", new_image9 );
namedWindow("10",CV_WINDOW_NORMAL);
imshow( "10", new_image10 );
*/


int l = 1,s = 1;
switch(flag2)
{
    case 0: l = 8; s = 9;
            ocrflag = ocr(new_image8,new_image9,8,9);
            if(ocrflag == 8)
            {
                b1 = imgg3.at<Vec3b>(1,1)[0];
                g1 = imgg3.at<Vec3b>(1,1)[1];
                r1 = imgg3.at<Vec3b>(1,1)[2];

                b2 = imgg2.at<Vec3b>(1,1)[0];
                g2 = imgg2.at<Vec3b>(1,1)[1];
                r2 = imgg2.at<Vec3b>(1,1)[2];
                namedWindow("11",CV_WINDOW_NORMAL);
imshow( "11", imgg3 );
namedWindow("12",CV_WINDOW_NORMAL);
imshow( "12", imgg4 );


                lab(imgg3,imgg2);
            }
            else
            {
                b1 = imgg2.at<Vec3b>(1,1)[0];
                g1 = imgg2.at<Vec3b>(1,1)[1];
                r1 = imgg2.at<Vec3b>(1,1)[2];

                b2 = imgg3.at<Vec3b>(1,1)[0];
                g2 = imgg3.at<Vec3b>(1,1)[1];
                r2 = imgg3.at<Vec3b>(1,1)[2];

                lab(imgg2,imgg3);
                 namedWindow("11",CV_WINDOW_NORMAL);
imshow( "11", imgg3 );
namedWindow("12",CV_WINDOW_NORMAL);
imshow( "12", imgg2 );

            }
            //return;
            break;

    case 1: l = 8; s = 10;
            ocrflag = ocr(new_image8,new_image10,8,10);
            if(ocrflag == 8)
            {
                b2 = imgg4.at<Vec3b>(1,1)[0];
                g2 = imgg4.at<Vec3b>(1,1)[1];
                r2 = imgg4.at<Vec3b>(1,1)[2];

                b1 = imgg2.at<Vec3b>(1,1)[0];
                g1 = imgg2.at<Vec3b>(1,1)[1];
                r1 = imgg2.at<Vec3b>(1,1)[2];

                lab(imgg4,imgg2);
                 namedWindow("11",CV_WINDOW_NORMAL);
imshow( "11", imgg2 );
namedWindow("12",CV_WINDOW_NORMAL);
imshow( "12", imgg4 );
            }
            else
            {
                b1 = imgg2.at<Vec3b>(1,1)[0];
                g1 = imgg2.at<Vec3b>(1,1)[1];
                r1 = imgg2.at<Vec3b>(1,1)[2];

                b2 = imgg4.at<Vec3b>(1,1)[0];
                g2 = imgg4.at<Vec3b>(1,1)[1];
                r2 = imgg4.at<Vec3b>(1,1)[2];

                lab(imgg2,imgg4);
                 namedWindow("11",CV_WINDOW_NORMAL);
imshow( "11", imgg2 );
namedWindow("12",CV_WINDOW_NORMAL);
imshow( "12", imgg4 );waitKey(3000);
            }
            //return;
            break;

    case 2: l = 9; s = 10;
            ocrflag = ocr(new_image9,new_image10,9,10);
            if(ocrflag == 9)
            {
                b2 = imgg4.at<Vec3b>(1,1)[0];
                g2 = imgg4.at<Vec3b>(1,1)[1];
                r2 = imgg4.at<Vec3b>(1,1)[2];

                b1 = imgg3.at<Vec3b>(1,1)[0];
                g1 = imgg3.at<Vec3b>(1,1)[1];
                r1 = imgg3.at<Vec3b>(1,1)[2];

                lab(imgg4,imgg3); namedWindow("11",CV_WINDOW_NORMAL);
imshow( "11", imgg3 );
namedWindow("12",CV_WINDOW_NORMAL);
imshow( "12", imgg4 );waitKey(3000);
            }
            else
            {
                b1 = imgg3.at<Vec3b>(1,1)[0];
                g1 = imgg3.at<Vec3b>(1,1)[1];
                r1 = imgg3.at<Vec3b>(1,1)[2];

                b2 = imgg4.at<Vec3b>(1,1)[0];
                g2 = imgg4.at<Vec3b>(1,1)[1];
                r2 = imgg4.at<Vec3b>(1,1)[2];

                lab(imgg3,imgg4);
                 namedWindow("11",CV_WINDOW_NORMAL);
imshow( "11", imgg3 );
namedWindow("12",CV_WINDOW_NORMAL);
imshow( "12", imgg4 );waitKey(3000);
            }
            //return;
            break;
}
  return;
}

void shapedetect(cv::Mat src)
{
    float a=MAX_ERROR;
    int radius;
    float cs,ca,sum=0,f_err,wt=0.6;int j,flag=0,e_flag=0;
    float l_err[3];
    string index;
	cv::Mat bw;
	//cv::cvtColor(src, bw, CV_BGR2GRAY);
bitwise_not(src,bw);
    //blur(bw,bw,Size(3,3));

	cv::Canny(bw, bw, 80,200,3);
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point> approx;
    cout<<"Shape: ";

 vector<Vec3f> circles;
 HoughCircles( bw, circles, CV_HOUGH_GRADIENT, 1, bw.rows/8, 200, 30, 0, 0 );

 for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      radius = cvRound(circles[i][2]);
      circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
      circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
      cout<<"circle "<<endl;
      strcpy(field6,"circle");
      return;
   }

	for (int i = 0; i < contours.size(); i++)
	{
      cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
      if (std::fabs(cv::contourArea(contours[i])) < 10)
			continue;
   int vtc = approx.size(); int p;

   std::vector<double> l_vec;
   for (int j = 2; j < vtc+1; j++)
   l_vec.push_back(length(approx[j%vtc], approx[j-1]));

   std::vector<double> cos;
   for (int j = 2; j < vtc+1; j++)
   cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));
   std::sort(cos.begin(), cos.end());

 //triangle

  cs=err(3,vtc);
    for(j=0;j<cos.size();j++)
  {
    ca=err(0.5,cos[j]);
    sum+=ca;
  }
  sum /= j;
  f_err=wt*sum+(1-wt)*cs;
if(f_err<a)
 {
    a=f_err;
    index="Triangle";
}
    sum=0;

  //square/rect

  cs=err(4,vtc);
 for(j=0;j<cos.size();j++)
  { ca=err(0,cos[j]);
  sum+=ca;
  }
 sum /= j;
  f_err=wt*sum+(1-wt)*cs;
if(f_err<a)
 {a=f_err;index="Sq/Rect";}

    sum=0;

  //pentagon

  cs=err(5,vtc);
  for( j=0;j<cos.size();j++)
  { ca=err(-0.309,cos[j]);
  sum+=ca;
  }
   sum /= j;
  f_err=wt*sum+(1-wt)*cs;
if(f_err<a)
 {a=f_err;index="Pentagon";}
    sum=0;

  //hexagon

  cs=err(6,vtc);
  for( j=0;j<cos.size();j++)
  { ca=err(-0.5,cos[j]);
 sum+=ca;
  }
   sum /= j;
  f_err=wt*sum+(1-wt)*cs;
if(f_err<a)
 {a=f_err;index="Hexagon";}
 sum=0;

    //hept

  cs=err(7,vtc);
  for( j=0;j<cos.size();j++)
  {
  ca=err(-0.623,cos[j]);
  sum+=ca;
  }
   sum /= j;
  f_err=wt*sum+(1-wt)*cs;
if(f_err<a)
 {a=f_err;index="Heptagon";}
 sum=0;

  //oct

  cs=err(8,vtc);
  for( j=0;j<cos.size();j++)
  {
  ca=err(-0.707,cos[j]);
  sum+=ca;
  }
   sum /= j;
  f_err=wt*sum+(1-wt)*cs;
if(f_err<a)
 {a=f_err;index="Octagon";}
 sum=0;

  //cross

  cs=err(12,vtc);
  for( j=0;j<cos.size();j++)
  {
  ca=err(0,cos[j]);
  sum+=ca;
  }
   sum /= j;
  f_err=wt*sum+(1-wt)*cs;
if(f_err<a)
 {a=f_err;index="Cross";}
 sum=0;

  //star

  float ca1,ca2;
  cs=err(10,vtc);
  for( j=0;j<cos.size();j++)
  {
  ca1=err(-0.309,cos[j]);
  ca2=err(0.809,cos[j]);
  ca1<ca2?sum+=ca1:sum+=ca2;
  }
   sum /= j;
  f_err=wt*sum+(1-wt)*cs;
if(f_err<a)
 {a=f_err;index="Star";}
 sum=0;


 if(index=="Sq/Rect")
{
  for(j=0;j<cos.size();j++)
  {
  if(!(cos[j]<0.1&&cos[j]>-0.1))
        flag=1;
  }

  if(flag==1)
  {
      index="Trapezoid";
  }
  else
  {
      l_err[0]=err(l_vec[0],l_vec[1]);
      l_err[1]=err(l_vec[1],l_vec[2]);
      l_err[2]=err(l_vec[0],l_vec[2]);
      for(i=0;i<3;i++)
        {
        if(l_err[i]>2)
          {
            e_flag=1;
            break;
          }

        }
    if(e_flag==1)
    index="Rectangle";
    else index="Square";

  }

 }

cout<<index<<endl;
 strcpy(field6,index.c_str());

}
	return;
}

int ocr(cv::Mat input1,cv::Mat input2, int x1, int x2)
{
  int m,m1=0,send=-1,sflag=-1,orif = -1;
  char m2;
  char t;
  Mat src, src_gray, dst,dst1,mst,input;
    waitKey(1000);

    for(int ocrk=0; ocrk<2;ocrk++)
    {
    if(ocrk==0)
        input = input1;
    else
        input = input2;

cvtColor( input, src_gray, CV_BGR2GRAY );
  cv::blur( src_gray, mst, Size( 3,5 ) );
    cv::threshold(mst, dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  namedWindow("tess",CV_WINDOW_NORMAL);
  imshow("tess",dst);
   // imwrite("/home/image.jpg",dst);
    //waitKey(5000);
    vector<vector<Point> > contours2;
    vector<Vec4i> hierarchy2;
    findContours(dst,contours2,hierarchy2,RETR_TREE,CV_CHAIN_APPROX_NONE);

    int max_area=0,max_index;
    for(int k=0;k<contours2.size();k++)
    {
        double area = contourArea(contours2[k]);
        if(area>max_area)
        {
            max_area = area;
            max_index = k;
        }
    }
     Mat tess1 = Mat::zeros(dst.size(), CV_8UC1);
    drawContours(tess1,contours2,max_index,Scalar(255,255,255),CV_FILLED);
    int alternate=0;
    while(hierarchy2[max_index][2]!= -1)
    {
        max_index = hierarchy2[max_index][2];
        if(alternate = 0)
        drawContours(tess1,contours2,max_index,Scalar(0),CV_FILLED);
        else
            drawContours(tess1,contours2,max_index,Scalar(255,255,255),CV_FILLED);
        alternate =~alternate;
    }

namedWindow("dst",CV_WINDOW_NORMAL);
imshow("dst",tess1);
waitKey(0);




for(int i=0; i<360; i=i+5)
{
    int len = std::max(tess1.cols, tess1.rows);
    cv::Point2f pt(len/2.0, len/2.0);
    cv::Mat r = cv::getRotationMatrix2D(pt, i, 1.0);

    cv::warpAffine(tess1, dst1, r, cv::Size(len, len));

    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    tess.SetImage((uchar*)dst1.data, dst1.cols, dst1.rows, 1, dst1.cols);

   char* out = tess.GetUTF8Text();
   int* conf = tess.AllWordConfidences();
   t= *out;

    for(int j=0;;j++)
  {
     if(conf[j] == -1 )
        break;
     else
        {
            if(conf[j] >= 80 && (( t>='0' && t<='9') || (t>='a' && t<= 'z')|| (t>='A' && t<= 'Z')))
            {
              m=conf[j];
              if(m>m1)
                    {
                        m1=m;
                        m2 = t;
                        sflag = ocrk;
                        orif = i;
                    }
            }
        }
 }

}
}
namedWindow("tessinput1",CV_WINDOW_NORMAL);
     namedWindow("tessinput2",CV_WINDOW_NORMAL);
     imshow("tessinput1",input1);
      imshow("tessinput1",input2);
      waitKey(0);
    std::cout << "The Identified Alphanumeric code is ";
    std::cout << m2 << std::endl;
    field8[0] = m2;
    field8[1] = '\0';
    orient(orif);
if(sflag == 0)
    return x1;
else
    return x2;

}

void orient(float letter)
{
    float plane;
    plane = heading;
    char ori[3];
    //float letter,plane,f;
    float f;
    //f = 216;
    //plane = 220;
    //letter = 180;
    f = ( plane + 360 + letter ) ;
    while(f>360)
        f-=360;
    if(f>=22.5 && f<=202.5)
    {
        if(f>=22.5 && f<67.5)
            strcpy(ori,"NE");
        else if(f>=67.5 && f<112.5)
            strcpy(ori,"E");
        else if(f>=112.5 && f<157.5)
            strcpy(ori,"SE");
        else
            strcpy(ori,"S");
    }
    else
    {
        if(f>=202.5 && f<247.5)
            strcpy(ori,"SW");
        else if(f>=247.5 && f<292.5)
            strcpy(ori,"W");
        else if(f>=292.5 && f<337.5)
            strcpy(ori,"NW");
        else
            strcpy(ori,"N");
    }
    cout<<"Orientation: "<<ori<<endl;
    strcpy(field5,ori);
    return;
}
/*ck
int bar(char * name)
{
    int up;
      ImageScanner scanner;
      scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
      int loop=0;
      Mat img = imread(name,0);
      Mat imgout;
      char result[100];
      cvtColor(img,imgout,CV_GRAY2RGB);
      int width = img.cols;
      int height = img.rows;
   uchar *raw = (uchar *)img.data;
   // wrap image data
   Image image(width, height, "Y800", raw, width * height);
   // scan the image for barcodes
   int n = scanner.scan(image);
   // extract results
   up = target_counter;
   for(Image::SymbolIterator symbol = image.symbol_begin();
     symbol != image.symbol_end();
     ++symbol) {
     cout << symbol->get_data() << endl;
     strcpy(field11,symbol->get_data().c_str());
     ++loop;
   }
   target_counter=up;
   image.set_data(NULL, 0);
   if(loop == 1)
    {
        noflag = 0;
        strcpy(field2,"QRC");
    }
   return loop;
}
*/


void gps(cv::Mat src)
{

 double pitch=5.0;
 double roll_a;
 double pl_angle=20.0, alt=300.0, la=28.0, lo=68.0;
 double f_la,f_lo;
 double pi=22/7;pl_angle=pl_angle*pi/180;
 double cam_angle=28*pi/180+pitch, theta_l,theta_r,shift_l,shift_r;
 double meter_to_feet=3.28084;

     FILE* fp2;
     void img_centre(float cam_angle,float pl_angle,float alt,float la,float lo,float disx,float disy);
     char loc[15],temp[100], data[500];
     int pos=0;
     sprintf(loc,"im%d.txt",i);
     fp2=fopen(loc,"r");
     if(fp2 == NULL)
        cout<<"Unable to open file";
    fseek(fp2,0,2);
    int lod = ftell(fp2);
    rewind(fp2);

        char c;
     while(!feof(fp2))
    {
        fscanf(fp2,"%c",&data[pos++]);
     }
     fclose(fp2);
    int pos1 = 0;
   // new_line = '\n';
    pos=0;
     while(data[pos]!='\n')
     {
        temp[pos1++] = data[pos];
        pos++;
     }
     ++pos;
     pos1=0;
     alt = atof(temp);
     alt *= meter_to_feet;
     while(data[pos]!='\n')
     {
        temp[pos1++] = data[pos];
           pos++;

     }
     ++pos;
     pos1=0;
     la = atof(temp);
      while(data[pos]!='\n')
     {
        temp[pos1++] = data[pos];
           pos++;

     }
     ++pos;
     pos1=0;
     lo = atof(temp);
      while(data[pos]!='\n')
     {
        temp[pos1++] = data[pos];
           pos++;

     }
     ++pos;
     pos1=0;
     heading = pl_angle = atof(temp);
      while(data[pos]!='\n')
     {
        temp[pos1++] = data[pos];
           pos++;

     }
     pitch = atof(temp);
     ++pos;
     pos1=0;
     while(data[pos]!='\n' && pos < lod)
     {
        temp[pos1++] = data[pos];
           pos++;

     }

     roll_a = atof(temp);

    cv::Mat gray;cv::Mat bw;
	cv::cvtColor(src, gray, CV_BGR2GRAY);
    blur(gray,gray,Size(3,3));

	cv::Canny(gray, bw, 80,200,3);

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	std::vector<cv::Point> approx;
	cv::Mat dst = src.clone();

    for (int i = 0; i < contours.size(); i++) ///runs the loop for each polygon
	{

cv::approxPolyDP(cv::Mat(contours[0]), approx, cv::arcLength(cv::Mat(contours[0]), true)*0.02, true);///incase of for loop, use i  instead of zero
      if (std::fabs(cv::contourArea(contours[0])) < 10)///incase of for loop, use i  instead of zero
			continue;
      int vtc = approx.size(); int p;
}
   /// dlat,dlon is initally the x,y coordinates of the dispalcement of the center of image from the guven gps posiiton.
   /// cc,rc is the centre of the image
   ///approx[0].x/y corresponds to the first point returned by the target
double cc = (src.cols)/2.0 ;
double rc = (src.rows)/2.0;
  cc = approx[0].x - cc;
   rc = approx[0].y - rc;
   /// cc and rc have to be scaled up to the real value from pixel length
double disp_roll=alt*tan(abs(roll_a));
double x_roll,y_roll;
if(pl_angle<=pi/2 && pl_angle>=0)
    {
    x_roll=-sin((pi/2)-pl_angle);
    y_roll=cos((pi/2)-pl_angle);
    }
else if(pl_angle<=pi && pl_angle>=pi/2)
    {
    x_roll=cos((pi)-pl_angle);
    y_roll=sin((pi)-pl_angle);
    }
else if(pl_angle<=3*pi*0.5 && pl_angle>=pi)
    {
    x_roll= sin((3*pi*0.5)-pl_angle);
    y_roll= -cos((3*pi*0.5)-pl_angle);
    }
else if (pl_angle<=2*pi && pl_angle>=3*pi*0.5)
    {
    x_roll= -(cos((2*pi)-pl_angle));
    y_roll= -(sin((2*pi)-pl_angle));
    }

if(roll_a <0)
    {
    x_roll=-x_roll;
    y_roll=-y_roll;
    }
x_roll=x_roll*disp_roll;
y_roll=y_roll*disp_roll;

double refer=alt*tan(semi_view);
   theta_l=abs(cam_angle-semi_view);
   theta_r=abs(cam_angle+semi_view);
   shift_r=alt*tan(theta_r);
   shift_l=alt*tan(theta_l);
   shift_r+=refer;
   if(cam_angle<semi_view)
   {
        shift_l+=refer;
   }
   else
     shift_l=refer-shift_l;

double D=abs(shift_r-shift_l);
float t=D/10;
float A=48*t*t;

//12 megapixel
//double pixels=4000*3000;
//8 megapixel
double pixels=3264*2448;

float factor=sqrt(abs(A/pixels)); // ft per pixel
cc = cc*factor;
rc = rc*factor;

double d_img_center=alt*(tan(cam_angle));
double x=d_img_center*sin(pl_angle);
double y=d_img_center*cos(pl_angle);
x=x+cc+x_roll;
y=y+rc+y_roll;

double radius = 6371*1000*3.2;
double   ft_deg_lo = (2 * pi * radius * cos(38*pi/180)) / 360 ;
  double ft_deg_la=(2*pi*radius)/360;

x=(x/ft_deg_lo)+lo;y=(y/ft_deg_la)+la;

if(x<0)
    x = x*-1;
if(y<0)
    y = y*-1;

output2 = y;
output3 = x;

int deg_lo=x;
int deg_la=y;
double min_lo_temp=(x-deg_lo)*60;
double min_la_temp=(y-deg_la)*60;

int min_la=min_la_temp;
int min_lo=min_lo_temp;

cout<<"deg_la = "<<deg_la<<endl;
cout<<"y = "<<y<<endl;
cout<<"min_la_temp = "<<min_la_temp<<endl;
cout<<"min_la = "<<min_la<<endl;

long double sec_la=(min_la_temp-min_la)*60;
long double sec_lo=(min_lo_temp-min_lo)*60;

cout<<"GPS: N"<<" "<<deg_la<<" "<<min_la<<" "<<sec_la;
cout<<" W"<<" "<<deg_lo<<" "<<min_lo<<" "<<sec_lo;
field31 = deg_la;
field32 = min_la;
field33 = sec_la;
field41 = deg_lo;
field42 = min_lo;
field43 = sec_lo;
return;

}

void lab(cv::Mat image1, cv::Mat image2)
{

Mat frame;
frame = image1;
int i=10,j=10;

for(int k=0;k<2;k++)
{
      Vec3b pix_bgr = frame.ptr<Vec3b>(i)[j];
            float B = pix_bgr.val[0];
            float G = pix_bgr.val[1];
            float R = pix_bgr.val[2];

        float var_R = ( R / 255 )  ;
        float var_G = ( G / 255 )  ;
        float var_B = ( B / 255 )  ;

        if ( var_R > 0.04045 )
            var_R = pow(( ( var_R + 0.055 ) / 1.055 ),2.4) ;
        else
            var_R = var_R / 12.92;

        if ( var_G > 0.04045 )
            var_G = pow(( ( var_G + 0.055 ) / 1.055 ), 2.4 );
        else
            var_G = var_G / 12.92;

        if ( var_B > 0.04045 )
            var_B = pow(( ( var_B + 0.055 ) / 1.055 ), 2.4 );
        else
            var_B = var_B / 12.92;

var_R = var_R * 100;
var_G = var_G * 100;
var_B = var_B * 100;

float X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
float Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
float Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

            float Xn = X / 95.047;
            float Yn = Y / 100.00;
            float Zn = Z / 108.883;

            if(Xn > 0.008856)
                Xn = pow(Xn,(1/3.0));
            else
                Xn = (7.787 * Xn) + (16.0 /116);

            if(Yn > 0.008856)
                Yn = pow(Yn,(1/3.0));
            else
                Yn = (7.787 * Yn) + (16.0 /116);
            if(Zn > 0.008856)
                Zn = pow(Zn,(1/3.0));
            else
                Zn = (7.787 * Zn) + (16.0 /116);

            float L = (116.0 * Yn) - 16.0;
            float a = 500.0 * (Xn - Yn);
            float b = 200.0 * (Yn - Zn);

closest(L,a,b,k);

if(k==0)
    frame = image2;

 }
    return;
}

void closest(float l,float a,float b,int fl)
{
    clab[0].l = 53.232881 ;
    clab[0].a = 80.109309 ;
    clab[0].b = 67.220068 ;
    strcpy(clab[0].name,"red");

    clab[1].l = 87.737033 ;
    clab[1].a = -86.184636 ;
    clab[1].b = 83.181164 ;
    strcpy(clab[1].name,"green");

    clab[2].l = 32.302586 ;
    clab[2].a = 79.196661 ;
    clab[2].b = -107.863681 ;
    strcpy(clab[2].name,"blue");

    clab[3].l = 44.221893 ;
    clab[3].a = 85.286819 ;
    clab[3].b = -87.724381 ;
    strcpy(clab[3].name,"purple");

    clab[4].l = 97.138246 ;
    clab[4].a = -21.555908 ;
    clab[4].b = 94.482485 ;
    strcpy(clab[4].name,"yellow");

    clab[5].l = 59.668564 ;
    clab[5].a = 62.058117 ;
    clab[5].b = 69.970523 ;
    strcpy(clab[5].name,"orange");

    clab[6].l = 27.276118 ;
    clab[6].a = 19.632727 ;
    clab[6].b = 37.715538 ;
    strcpy(clab[6].name,"brown");

    clab[7].l = 100 ;
    clab[7].a = 0.0052604 ;
    clab[7].b = -0.010408 ;
    strcpy(clab[7].name,"white");

    clab[8].l = 42.374603 ;
    clab[8].a = 0.002647 ;
    clab[8].b = -0.0052377 ;
    strcpy(clab[8].name,"black");

    clab[9].l = 72.574782 ;
    clab[9].a = 0.004016 ;
    clab[9].b = -0.007947 ;
    strcpy(clab[9].name,"gray");

    double lab1[3], lab2[3];
    double delta;
    double mini=100000;
    int mini_val = -1;

    for(int m=0;m<10;m++)
    {
    lab1[0]=l;
    lab1[1]=a;
    lab1[2]=b;

    lab2[0]= clab[m].l;
    lab2[1]= clab[m].a;
    lab2[2]= clab[m].b;

    deltaE2000(lab1, lab2,  delta );

    if(delta<mini)
    {
        mini = delta;
        mini_val = m;
    }
    }

    if(fl==0)
    {
        cout<<"CLOSEST COLOR LAB THINGY OUT = "<<clab[mini_val].name<<endl;
        strcpy(field7,clab[mini_val].name);
    }
    else
    {
        cout<<"CLOSEST COLOR LAB THINGY IN = "<<clab[mini_val].name<<endl;
        strcpy(field9,clab[mini_val].name);
    }
    if (strcmp(field7,field9)==0)
    {
        target_counter--;
        noflag = 1;
    }

    return;
}

void deltaE2000( double lab1[3], double lab2[3], double &delta_e )
{
	double Lstd = lab1[0];
	double astd = lab1[1];
	double bstd = lab1[2];

	double Lsample = lab2[0];
	double asample = lab2[1];
	double bsample = lab2[2];

	double _kL = 1.0;
	double _kC = 1.0;
	double _kH = 1.0;

	double Cabstd= sqrt(astd*astd+bstd*bstd);
	double Cabsample= sqrt(asample*asample+bsample*bsample);

	double Cabarithmean= (Cabstd + Cabsample)/2.0;

	double G= 0.5*( 1.0 - sqrt( pow(Cabarithmean,7.0)/(pow(Cabarithmean,7.0) + pow(25.0,7.0))));

	double apstd= (1.0+G)*astd;
	double apsample= (1.0+G)*asample;
	double Cpsample= sqrt(apsample*apsample+bsample*bsample);

	double Cpstd= sqrt(apstd*apstd+bstd*bstd);
	double Cpprod= (Cpsample*Cpstd);

	double hpstd= atan2(bstd,apstd);
	if (hpstd<0) hpstd+= 2.0*PI;

	double hpsample= atan2(bsample,apsample);
	if (hpsample<0) hpsample+= 2.0*PI;
	if ( (fabs(apsample)+fabs(bsample))==0.0)  hpsample= 0.0;

	double dL= (Lsample-Lstd);
	double dC= (Cpsample-Cpstd);

	double dhp= (hpsample-hpstd);
	if (dhp>PI)  dhp-= 2.0*PI;
	if (dhp<-PI) dhp+= 2.0*PI;
	if (Cpprod == 0.0) dhp= 0.0;

	double dH= 2.0*sqrt(Cpprod)*sin(dhp/2.0);

	double Lp= (Lsample+Lstd)/2.0;
	double Cp= (Cpstd+Cpsample)/2.0;

	double hp= (hpstd+hpsample)/2.0;

	if ( fabs(hpstd-hpsample)  > PI ) hp-= PI;

	if (hp<0) hp+= 2.0*PI;

    if (Cpprod==0.0) hp= hpsample+hpstd;

	double Lpm502= (Lp-50.0)*(Lp-50.0);;
	double Sl= 1.0+0.015*Lpm502/sqrt(20.0+Lpm502);
	double Sc= 1.0+0.045*Cp;
	double T= 1.0 - 0.17*cos(hp - PI/6.0) + 0.24*cos(2.0*hp) + 0.32*cos(3.0*hp+PI/30.0) - 0.20*cos(4.0*hp-63.0*PI/180.0);
	double Sh= 1.0 + 0.015*Cp*T;
	double delthetarad= (30.0*PI/180.0)*exp(- pow(( (180.0/PI*hp-275.0)/25.0),2.0));
	double Rc=  2.0*sqrt(pow(Cp,7.0)/(pow(Cp,7.0) + pow(25.0,7.0)));
	double RT= -sin(2.0*delthetarad)*Rc;

	delta_e = sqrt( pow((dL/Sl),2.0) + pow((dC/Sc),2.0) + pow((dH/Sh),2.0) + RT*(dC/Sc)*(dH/Sh) );
	return;
}

