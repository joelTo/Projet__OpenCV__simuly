#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <iostream>
#include <stdio.h>

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <io.h>
#include <direct.h>
#include <string>
#include <math.h>
#include <fstream>
#include <time.h>
#include <vector>

#include <QDir>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/// Global Variables
Mat img; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method = CV_TM_SQDIFF;
int max_Trackbar = 5;

 int scale = 1;
 int delta = 0;
 int ddepth = CV_16S;

int nbr_file_FILENAME=0;
int nbr_file_FILENAME_REFERENCE=0;

int con_file();
String *go_to_image();
String *go_to_image_reference();
void sleep(int nbr_seconds);
void ascending_sort(String name[],double val[],int taille);
void descending_sort(String name[],double val[],int taille);
double *add_in_list_double(double a[],int taille,double valeur);
String *add_in_list_String(String a[],int taille,String name);
void affiche( String name[],double val[], int tailleTableau);
void enregistrement(String name[], String name_file, double val[], int tailleTableau,QString norme);
Mat contour(Mat img);

/** @function main */
int main(int argc, char** argv)
{

    std::vector<int> v;
    //v.push_back( 0 );
    v.push_back( CV_TM_SQDIFF_NORMED );
    v.push_back( CV_TM_CCORR );
    v.push_back( CV_TM_CCORR_NORMED );
    v.push_back( CV_TM_CCOEFF );
    v.push_back( CV_TM_CCOEFF_NORMED );


    std::vector<QString> n;
    //n.push_back( "SURF" );
    n.push_back( "CV_TM_SQDIFF_NORMED" );
    n.push_back( "CV_TM_CCORR" );
    n.push_back( "CV_TM_CCORR_NORMED" );
    n.push_back( "CV_TM_CCOEFF" );
    n.push_back( "CV_TM_CCOEFF_NORMED" );


    for (int ite_methode=0;ite_methode <v.size() ; ite_methode++)
    {
        int match_method = v[ite_methode];
        QString norme=n[ite_methode];
        std::cout << v[ite_methode] << '\n';
        std::cout << (n[ite_methode]).toStdString() << '\n';


        //** Initialisation de toutes les variables **/

        String *FILENAME = go_to_image();
        String *FILENAME_REFERENCE = go_to_image_reference();


        //** Fin < Initialisation de toutes les variables > **/


        for(int ite_im_ref=0;ite_im_ref <(nbr_file_FILENAME_REFERENCE);ite_im_ref++ )
        {
            std::cout<<"ite_im_ref ="<<ite_im_ref<<"  :"<<FILENAME_REFERENCE[ite_im_ref]<<std::endl;


            std::vector<double> stokage_valeur;
            std::vector<String> stokage_name;


            String choix;
            for(int ite_im=0;ite_im <(nbr_file_FILENAME);ite_im++)
            {
                std::cout<<"ite_im ="<<ite_im <<"  :"<<FILENAME[ite_im]<<std::endl;

                //_chdir("Base_de_donnee");// everywhere version
                _chdir("C:/Users/Projet TER/Documents/build-simily-Desktop_Qt_5_4_2_MinGW_32bit-Debug/Base_de_donnee");
                //std::cout<<"nre de dossier :"<<²      ²on_file()<<std::endl;
                img = imread(FILENAME[ite_im], 1);
                //blur( img, img, Size(5,5) );
                //GaussianBlur( img, img, Size( 5, 5 ), 0, 0 );
                // test 7
                //img=contour(img);
                //imshow( "Sobel Demo - Simple Edge Detector", img );
                //waitKey(0);

                /// Apply Histogram Equalization
                //equalizeHist( img, img );

                //_chdir("..");
                //_chdir("image_reference");// //  everywhere version
                _chdir("C:/Users/Projet TER/Documents/build-simily-Desktop_Qt_5_4_2_MinGW_32bit-Debug/image_reference");

                //std::cout<<"nre de dossier :"<<con_file()<<std::endl;
                templ = imread(FILENAME_REFERENCE[ite_im_ref], 1);

                // Test 7
                //templ=contour(templ);
                //imshow( "Sobel Demo - Simple Edge Detector", templ );
                //waitKey(0);
                //blur( templ, templ, Size(5,5) );
                //GaussianBlur( templ, templ, Size( 5, 5 ), 0, 0 );
                //imshow( "Sobel Demo - Simple Edge Detector", templ );
                //                waitKey(0);
                //equalizeHist( templ, templ );
                //_chdir("..");

                //******
                if( !templ.data || !img.data )
                { printf(" --(!) Error reading images \n"); return -1; }

                //-- Step 1: Detect the keypoints using SURF Detector
                int minHessian = 400;
                Ptr<SURF> detector = SURF::create();
                detector->setHessianThreshold(minHessian);

                std::vector<KeyPoint> keypoints_1, keypoints_2;
                Mat descriptors_1, descriptors_2;

                detector->detectAndCompute( templ, Mat(), keypoints_1, descriptors_1 );
                detector->detectAndCompute( img, Mat(), keypoints_2, descriptors_2 );


                //-- Step 2: Matching descriptor vectors using FLANN matcher
                FlannBasedMatcher matcher;
                std::vector< DMatch > matches;
                matcher.match( descriptors_1, descriptors_2, matches );

                double max_dist = 0; double min_dist = 100;

                //-- Quick calculation of max and min distances between keypoints
                for( int i = 0; i < descriptors_1.rows; i++ )
                { double dist = matches[i].distance;
                    if( dist < min_dist ) min_dist = dist;
                    if( dist > max_dist ) max_dist = dist;
                }

                printf("-- Max dist : %f \n", max_dist );
                printf("-- Min dist : %f \n", min_dist );

                double som = sqrt(max_dist*max_dist +min_dist*min_dist);
                std::cout << "som :" << som << std::endl;


                //********

                stokage_valeur.push_back(abs(som));
                stokage_name.push_back(FILENAME[ite_im]);
                //cout << stokage_name.size() << '\n';
                //cout << stokage_valeur.size() << '\n';
                std::cout << "\n  ref: "<<FILENAME_REFERENCE[ite_im_ref]<<" / img: " <<FILENAME[ite_im]<< " / som :" << abs(som) << endl<< endl;


            }



            int lengt_vector=stokage_valeur.size();
            double val[lengt_vector];
            String name[lengt_vector];
            for (int u=0;u<lengt_vector;u++)
            {
                val[u]=stokage_valeur[u];
                name[u]=stokage_name[u];
            }
            ascending_sort(name, val,lengt_vector);
            affiche(name,val,lengt_vector);
            enregistrement(name, FILENAME_REFERENCE[ite_im_ref], val,lengt_vector,norme);

        }

    }
    return 0;
}



/** @function to list folder (files)*/
int con_file()
{
    int i=0;
    struct _finddata_t D;
    int done = 0;
    int hd;
    hd = _findfirst("*.*", &D);
    if (hd == -1)
    {
        cout<<"Il n'y a rien dans ce dossier. J'aime les pâtes";
    }
    while (!done)
    {
        //printf("%s\n", D.name);
        i++;
        done = _findnext(hd, &D);
    }
    return i;
}





String *go_to_image()
{
    _chdir("Base_de_donnee");// read name of fil
    nbr_file_FILENAME=con_file()-2; //jump .. & .
    String *array= new String[nbr_file_FILENAME];
    std::cout<<"Nombre de fichier est de :"<< nbr_file_FILENAME<<std::endl;


    int i=0;
    struct _finddata_t D;
    int done = 0;
    int hd;
    hd = _findfirst("*.*", &D);
    done = _findnext(hd, &D);
    done = _findnext(hd, &D);
    if (hd == -1)
    {
        cout<<"Il n'y a rien dans ce dossier";
    }
    while (!done)
    {
        array[i]=D.name;
        i++;
        done = _findnext(hd, &D);
    }
    _chdir("..");// read name of fil
    return array;
}

String *go_to_image_reference()
{
    _chdir("image_reference");// read name of fil
    nbr_file_FILENAME_REFERENCE=con_file()-2; //jump .. & .
    String *array= new String[nbr_file_FILENAME_REFERENCE];
    std::cout<<"Nombre de fichier est de :"<< nbr_file_FILENAME_REFERENCE <<std::endl;

    int i=0;
    struct _finddata_t D;
    int done = 0;
    int hd;
    hd = _findfirst("*.*", &D);
    done = _findnext(hd, &D);
    done = _findnext(hd, &D);
    if (hd == -1)
    {
        cout<<"Il n'y a rien dans ce dossier";
    }
    while (!done)
    {
        array[i]=D.name;
        i++;
        done = _findnext(hd, &D);
    }

    _chdir("..");// read name of fil
    return array;
}

void sleep(int nbr_seconds)
{
    clock_t goal;
    goal = (nbr_seconds * CLOCKS_PER_SEC) + clock();

    while(goal > clock())
    {
        ;
    }
}

/** @function to sort the list "ascebding" */
void ascending_sort(String name[],double val[],int taille)
{

    int i,j;
    String name_tmp;
    double tmp;

    for(i=0; i<taille ; i++)
    {
        for(j=i; j<taille; j++)
        {
            if(val[j]<val[i]){

                tmp = val[i];
                val[i] = val[j];
                val[j] = tmp;

                name_tmp=name[i];
                name[i]=name[j];
                name[j]=name_tmp;
            }
        }
    }
}

/** @function to sort the list "descebding" */
void descending_sort(String name[],double val[],int taille)
{

    int i,j;
    String name_tmp;
    double tmp;

    for(i=0; i<taille ; i++)
    {
        for(j=i; j<taille; j++)
        {
            if(val[j]>val[i]){

                tmp = val[i];
                val[i] = val[j];
                val[j] = tmp;

                name_tmp=name[i];
                name[i]=name[j];
                name[j]=name_tmp;
            }
        }
    }
}

double *add_in_list_double(double a[],int taille,double valeur)
{
    double *b=new double[taille];
    std::cout<<"Il y a "<< taille+1<<"dans le tableau \n"; // affichage de +1 car on ne prends pas encompte 0
    for(int i=0;i<taille;i++ )
    {
        b[i]=a[i];
    }
    b[taille-1]=valeur;
    return b;
}

String *add_in_list_String(String a[],int taille,String name)
{
    String *b=new String[taille];
    //std::cout<<"Il y a "<< taille+1<<"dans le tableau \n";
    //std::cout<<"Le premier element est :"<<a[0];

    for(int i=0;i<taille;i++ )
    {
        b[i]=a[i];
    }

    b[taille-1]=name;
    return b;
}

/** @function to show elements*/
void affiche( String name[],double val[], int tailleTableau)
{
    int i;
    for (i = 0 ; i < tailleTableau ; i++)
    {
        std::cout<<i+1<< " :" << name[i]<<" --> "<< val[i]<<std::endl;

    }
    printf("\n\n");
}


/** @function to record element to <name>.jpg */
void enregistrement(String name[], String name_file, double val[],int tailleTableau,QString norme)
{

    _chdir("C:/Users/Projet TER/Documents/build-simily-Desktop_Qt_5_4_2_MinGW_32bit-Debug/Resultats");
    QDir dossier;
    dossier.mkdir(name_file.c_str());
    _chdir("C:/Users/Projet TER/Documents/build-simily-Desktop_Qt_5_4_2_MinGW_32bit-Debug/Base_de_donnee");


    for(int i=0;i<tailleTableau ; i++)
    {
        //initialize a 120X700 matrix of black pixels:
        Mat output = imread(name[i]);

        //write text on the matrix:
        putText(output,name[i],cvPoint(15,140),FONT_HERSHEY_PLAIN,10,cvScalar(0,255,0),10);

        QString name_f= "Rang :"+ QString::number(i)+ " valeur :"+QString::number(val[i]);
        //write text on the matrix:
        putText(output,(name_f.toStdString()),cvPoint(15,260),FONT_HERSHEY_PLAIN,10,cvScalar(0,255,0),10);

        QString name2=norme;
        name2 = name2 + QString::number(i)+".jpg";
        String name3=name2.toStdString();


        _chdir("C:/Users/Projet TER/Documents/build-simily-Desktop_Qt_5_4_2_MinGW_32bit-Debug/Resultats");
        _chdir(name_file.c_str());
        vector<int> compression_params;
        //vector that stores the compression parameters of the image
        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
        //specify the compression technique
        compression_params.push_back(20);

        imwrite(name3, output, compression_params);
        _chdir("C:/Users/Projet TER/Documents/build-simily-Desktop_Qt_5_4_2_MinGW_32bit-Debug/Base_de_donnee");


    }
}

Mat contour(Mat img)
{
     Mat src_gray,grad;
    /// Convert it to gray
    cvtColor( img, src_gray, CV_BGR2GRAY );

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );



    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    return(grad);
}

