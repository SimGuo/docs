```CPP
#include <shogun/base/init.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/OpenCV/CV2FeaturesFactory.h>
#include <shogun/lib/OpenCV/CV2SGMatrixFactory.h>
#include <shogun/features/DataGenerator.h>



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include<iostream>

using namespace std;
using namespace shogun;
using namespace cv;
#define k 10

int main()
{

init_shogun_with_defaults();

    CvMLData mlData;
    mlData.read_csv("car.data");

    const CvMat* temp = mlData.get_values();
    int numfeatures = temp->cols-1;
    mlData.set_response_idx(numfeatures);


    CvTrainTestSplit spl((float)0.5);
    mlData.set_train_test_split(&spl);


    const CvMat* traindata_idx = mlData.get_train_sample_idx();
    const CvMat* testdata_idx = mlData.get_test_sample_idx();
    Mat mytraindataidx(traindata_idx);
    Mat mytestdataidx(testdata_idx);

    Mat all_Data(temp);
    Mat all_responses = mlData.get_responses();
    Mat traindata(mytraindataidx.cols,numfeatures,CV_32F);
    Mat shogun_trainresponse(mytraindataidx.cols,1,CV_32S);
    Mat testdata(mytestdataidx.cols,numfeatures,CV_32F);
    Mat shogun_testresponse(mytestdataidx.cols,1,CV_32S);
    Mat shogun_all_responses = Mat::ones(all_responses.rows, 1, CV_32F);
    Mat opencv_trainresponse(mytraindataidx.cols,1,CV_32S);
    Mat opencv_testresponse(mytestdataidx.cols,1,CV_32S);



    // making responses compatible to Shogun.
    for (int h=0; h<all_responses.rows; h++)
    {
        if (all_responses.at<float>(h) == 4 )
        {
            shogun_all_responses.at<float>(h)=0;
        }
        else if (all_responses.at<float>(h) == 10)
        {
            shogun_all_responses.at<float>(h)=1;
        }
        else if (all_responses.at<float>(h) == 11)
        {
            shogun_all_responses.at<float>(h)=2;
        }
        else 
        {
            shogun_all_responses.at<float>(h)=3;
        }
    }


//filling out shogun_testresponse, shogun_trainresponse, traindata and testdata mats in there.
   for(int i=0; i<mytraindataidx.cols; i++)
    {
       
        opencv_trainresponse.at<int>(i)=all_responses.at<float>(mytraindataidx.at<int>(i));
        shogun_trainresponse.at<int>(i)=shogun_all_responses.at<float>(mytraindataidx.at<int>(i));    
        for(int j=0; j<=numfeatures; j++)
        {
            traindata.at<float>(i, j)=all_Data.at<float>(mytraindataidx.at<int>(i), j);
        }
    }

    for(int i=0; i<mytestdataidx.cols; i++)
    {
       
        opencv_testresponse.at<int>(i)=all_responses.at<float>(mytestdataidx.at<int>(i));
        shogun_testresponse.at<int>(i)=shogun_all_responses.at<float>(mytestdataidx.at<int>(i));
        for(int j=0; j<=numfeatures; j++)
        {
            testdata.at<float>(i, j)=all_Data.at<float>(mytestdataidx.at<int>(i), j);
        }   
    }

    CvKNearest opencv_knn(traindata, opencv_trainresponse);
    opencv_knn.train(traindata, opencv_trainresponse);

    Mat results(1,1,CV_32F);
    Mat neighbourResponses = Mat::ones(1,10,CV_32F);
    Mat dist = Mat::ones(1, 10, CV_32F);

    // estimate the response and get the neighbors' labels
 
    int ko=0;

    for (int i=0;i<testdata.rows;++i)
    {
        opencv_knn.find_nearest(testdata.row(i),10,results, neighbourResponses, dist);
        if (results.at<float>(0,0) == opencv_testresponse.at<int>(i))
        {
            //cout<<results.at<float>(0,0)<<endl;
            ++ko;
            //cout<<ko<<endl;
        }
    }


cout<< "the efficiency of opencv knn is: "<<100.0 * ko/testdata.rows  <<endl;

    SGMatrix<float64_t> shogun_traindata = CV2SGMatrixFactory::getSGMatrix<float64_t>(traindata, CV2SG_MANUAL);
    SGMatrix<float64_t>::transpose_matrix(shogun_traindata.matrix, shogun_traindata.num_rows, shogun_traindata.num_cols);
    CDenseFeatures<float64_t>* shogun_trainfeatures = new CDenseFeatures<float64_t>(shogun_traindata);

    CDenseFeatures<float64_t>* shogun_dense_response = CV2FeaturesFactory::getDenseFeatures<float64_t>(shogun_trainresponse, CV2SG_MANUAL);
    SGVector<float64_t> shogun_vector_response = shogun_dense_response->get_feature_vector(0);
    CMulticlassLabels* labels = new CMulticlassLabels(shogun_vector_response);

    SGMatrix<float64_t> shogun_testdata = CV2SGMatrixFactory::getSGMatrix<float64_t>(testdata, CV2SG_MANUAL);
    SGMatrix<float64_t>::transpose_matrix(shogun_testdata.matrix, shogun_testdata.num_rows, shogun_testdata.num_cols);
    CDenseFeatures<float64_t>* shogun_testfeatures = new CDenseFeatures<float64_t>(shogun_testdata);
    
    // Create KNN classifier
	CKNN* knn = new CKNN(k, new CEuclideanDistance(shogun_trainfeatures, shogun_trainfeatures), labels);

	// Train classifier
	knn->train();
	
    CMulticlassLabels* output = knn->apply_multiclass(shogun_testfeatures);
    SGMatrix<int32_t> multiple_k_output = knn->classify_for_multiple_k();
    SGVector<float64_t> sgvec = output->get_labels();

    int ki=0;
    for(int i=0; i<sgvec.vlen; ++i)
    { 
        if(shogun_testresponse.at<float>(i) == sgvec[i])
        ++ki;
    }

    cout << "the efficiency of KNN for Shogun is: "<<(float)100.0 *ki/sgvec.vlen <<endl;
 	
	SG_UNREF(knn)
	SG_UNREF(output)  

    return 0;
}

```

