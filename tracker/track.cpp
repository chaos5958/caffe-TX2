// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cvconfig.h>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>

//for tracking
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utility.hpp>


//for networking
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/epoll.h>
#include <pthread.h>
#include <semaphore.h>

//for json
#include <json/json.h>

//for debugging
#include <stdarg.h>

//for clock
#include <time.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

#define TRACKING_METHOD "KCF"
#define CROP_RATIO 0.5

#define MIN_CROP_WIDTH  300 //caffe-ssd input width
#define MIN_CROP_HEIGHT 300 //caffe-ssd input height

//for networking
#define BUF_SIZE 4096 
#define LISTEN_PORT "44444"
int clnt_sock;

//for debugging and logging
#define USE_STREAM 1
#define GCS_STREAM 0 
#define NORM_LOG_ENABLED 1
#define TEST_LOG_ENABLED 1 
#define USE_TrackerKCF 0

typedef std::ostream& (*manip) (std::ostream&);
struct normlogger
{
    template< typename T >
        normlogger &operator<<(const T &val)
        {
            if(NORM_LOG_ENABLED)
                std::cout<<val;
            return *this;
        }

    normlogger &operator<<(manip manipulator)
    {
        if(NORM_LOG_ENABLED)
            std::cout<<manipulator;
        return *this;
    }
};

struct testlogger
{
    template< typename T >
        testlogger &operator<<(const T &val)
        {
            if(TEST_LOG_ENABLED)
                std::cout<<val;
            return *this;
        }

    testlogger &operator<<(manip manipulator)
    {
        if(TEST_LOG_ENABLED)
            std::cout<<manipulator;
        return *this;
    }
};


#define NETWORK_DEBUG 0 
static normlogger logout = normlogger(); 
static testlogger testout = testlogger();

//for developing
#define NEW_VERSION 0

//#define COMMAND_BUF_SIZE 128
sem_t mutex;
sem_t empty;
sem_t full;
pthread_cond_t track_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t track_mutex = PTHREAD_MUTEX_INITIALIZER;

//Variables shared between threads
bool is_detect_run = false;
bool is_detect_thisframe = false;
bool is_stream = false;
bool is_quit = false;
int index_obj = 0;
int object = 0; //NOT USED YET VER.1.0
double send_track_period = 0;
int frame_width = 1280;
int frame_height = 720;
int frame_rate = 10;
int buffer_size = 1; 
int send_msg_per_frame = 1; // send tracking result per XXX frame, 1 is default sending every frame's result

bool initial_crop_enable = true;
bool detection_crop_enable = true;
bool tracking_crop_enable = true;
bool visualize_detection_enable = true;
bool visualize_tracking_enable = true;

int track_frame_num = 30;
int object_type = 15; 
int selection_policy = 1;
int color_confidence_ratio = 0.05;

int iLowH = 0;
int iHighH = 35;
int iLowS = 100;
int iHighS = 255;
int iLowV = 100;
int iHighV = 255;

#define CAR 0
#define HUMAN 1


int port_num;

void error_handling(char * buf);
void * network_handler(void * arg);
void test_json();
void *detection_handler(void *arg);
int write_log(const char *foramat, ...);
bool send_imageresult(Json::Value msg);

char command_buf[BUF_SIZE];

class Detector {
    public:
        Detector(const string& model_file,
                const string& weights_file,
                const string& mean_file,
                const string& mean_value);

        std::vector<vector<float> > Detect(const cv::Mat& img);

    private:
        void SetMean(const string& mean_file, const string& mean_value);

        void WrapInputLayer(std::vector<cv::Mat>* input_channels);

        void Preprocess(const cv::Mat& img,
                std::vector<cv::Mat>* input_channels);

    private:
        shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
};


Detector::Detector(const string& model_file,
        const string& weights_file,
        const string& mean_file,
        const string& mean_value) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
            input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
            "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) <<
            "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
            "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                    cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Detector::Preprocess(const cv::Mat& img,
        std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
            == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
        "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
        "If specified, can be one value or can be same as image channels"
        " - would subtract from the corresponding channel). Separated by ','."
        "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "video",
        "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
        "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.1,
        "Only store detections with score higher than the threshold.");


bool containPoint(cv::Rect2d Rect, float pointX, float pointY) 
{
    // Just had to change around the math
    if (pointX < (Rect.x + Rect.width) && pointX > Rect.x &&
            pointY < (Rect.y + Rect.height) && pointY > Rect.y)
        return true;
    else
        return false;
}



int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Do detection using SSD mode.\n"
            "Usage:\n"
            "    ssd_detect [FLAGS] model_file weights_file list_file\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
        return 1;
    }

    vector<string> input_args;
    input_args.push_back(argv[1]);
    input_args.push_back(argv[2]);
    input_args.push_back(argv[3]);

    int serv_sock;
    struct sockaddr_in serv_adr, clnt_adr;
    socklen_t clnt_adr_sz;
    const char *port = LISTEN_PORT;
    int read_len = 0; 
    int write_len = 0;
    // client handler thread creation. thread will take a task in working queue.    
    serv_sock = socket (PF_INET, SOCK_STREAM, 0);

    //handle only one client
    memset(&serv_adr, 0, sizeof(serv_adr));
    serv_adr.sin_family= AF_INET;
    serv_adr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_adr.sin_port = htons(atoi(port));

    if(bind(serv_sock, (struct sockaddr *) &serv_adr, sizeof(serv_adr)) == -1)
    {
        error_handling((char *)"bind() error");
    }
    if(listen(serv_sock, 5) == -1){
        error_handling((char*)"listen() error");
    }
    int enable = 1;

    if (setsockopt(serv_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0){
        error_handling("setsockopt(SO_REUSEADDR) failed");
    }

    logout << " waiting connection ... \n" << std::endl;
    memset(&clnt_adr, 0, sizeof(clnt_adr));
    memset(&clnt_adr_sz, 0, sizeof(clnt_adr_sz));
    clnt_sock = accept(serv_sock,(struct sockaddr *)&clnt_adr, &clnt_adr_sz);
    logout <<" connected \n" << std::endl;

    //Network handler thread start!

    pthread_t network_thread, detection_thread;
    pthread_create(&network_thread, NULL, network_handler, NULL);

#if (NETWORK_DEBUG != 1)
    pthread_create(&detection_thread, NULL, detection_handler, &input_args);
#endif

    testout << "main: join network_thread" << std::endl;
    pthread_join(network_thread, NULL);

    testout << "main: join detection_thread" << std::endl;
#if (NETWORK_DEBUG != 1)
    pthread_join(detection_thread, NULL);
#endif

    testout << "main thread end" << std::endl;
    close(serv_sock);   
    return 0;
}

void *detection_handler(void *arg)
{
    //detector initialization 
    vector<string> *input_args = (vector<string> *)arg;
    const string& model_file = input_args->operator[](0);
    const string& weights_file = input_args->operator[](1);
    const string& mean_file = FLAGS_mean_file;
    const string& mean_value = FLAGS_mean_value;
    const string& file_type = FLAGS_file_type;
    const string& out_file = FLAGS_out_file;
    const float confidence_threshold = FLAGS_confidence_threshold;

    Detector detector(model_file, weights_file, mean_file, mean_value);

    //initialization input video & GCS stream 
    std::string file; //TODO: is this variable necceary?
    cv::VideoCapture cap;
    cv::VideoWriter writer;

    if(GCS_STREAM)
    {
        writer.open("appsrc ! videoconvert ! x264enc tune=zerolatency ! rtph264pay ! udpsink host=223.171.33.71 port=5000", 0, (double)30, cv::Size(640, 480), true); 

        if (!writer.isOpened()) {
            printf("=ERR= can't create video writer\n");
            return NULL;
        }
    }

    if(!USE_STREAM)
        cap = cv::VideoCapture((input_args->operator[](2)));
    else{
        cap = cv::VideoCapture(1);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, frame_width);	
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);
        cap.set(CV_CAP_PROP_FPS, frame_rate);
        cap.set(CV_CAP_PROP_BUFFERSIZE, buffer_size);
    }
    if (!cap.isOpened()) {
        LOG(FATAL) << "Failed to open video: " << file;
        return NULL;
    }

    //initialization window 
    cv::namedWindow("output", 1);

    int top_left_x = 0;
    int top_left_y = 0;
    int crop_box_width = 0;
    int crop_box_height = 0;

    cv::Rect2d bbox;
    cv::Rect2d draw_bbox;

    while(1)
    {
        pthread_mutex_lock(&track_mutex);
        while(!is_detect_run && !is_quit)
            pthread_cond_wait(&track_cond, &track_mutex);  

        if(is_quit)
        {
            pthread_mutex_unlock(&track_mutex);
            cap.release();
            testout << "detection thread ends" << std::endl;
            pthread_exit(NULL);
        }
        pthread_mutex_unlock(&track_mutex);

        cv::Mat img, img_process;

        bool success = cap.read(img);
        if(!success)
        {
            LOG(INFO) << "Process " << std::endl;
            break;
        }
        
        img_process = img;

        //preprocess
        if(initial_crop_enable)
        {
            //our case 1280 * 720 
            if(img.cols > img.rows){
                top_left_x = (img.cols - img.rows )/2; // (1280 - 720 )/2 = 280;
                top_left_y = 0;
                crop_box_width = 0, crop_box_height = 0;
            }
            //maybe if we get a long height input 
            else{
                top_left_x = 0;
                top_left_y = (img.rows - img.cols)/2;
                crop_box_width = 0, crop_box_height = 0;
            }
            //img(cv::Rect(280,0,720,720));
            img_process = img(cv::Rect(top_left_x,top_left_y, img.rows,img.rows));
            initial_crop_enable = false;
			
			cv::Rect2d debug_bbox(top_left_x, top_left_y, crop_box_width, crop_box_height);
            rectangle(img, debug_bbox, cv::Scalar(100,100,0), 2, 1);
			cv::imshow("output", img);
	        cv::waitKey(30); 


        }
        else if(detection_crop_enable)
        {
            if(tracking_crop_enable){
                top_left_x = std::max(static_cast<int>(draw_bbox.x - draw_bbox.width* CROP_RATIO), 0);
                top_left_y = std::max(static_cast<int>(draw_bbox.y - draw_bbox.height* CROP_RATIO), 0);
                crop_box_width = std::max(static_cast<int>((draw_bbox.x - top_left_x) * 2 + draw_bbox.width), MIN_CROP_WIDTH);
                crop_box_height = std::max(static_cast<int>((draw_bbox.y - top_left_y) * 2 + draw_bbox.height), MIN_CROP_HEIGHT);
            }
            else{
                top_left_x = std::max(static_cast<int>(bbox.x - bbox.width* CROP_RATIO), 0);
                top_left_y = std::max(static_cast<int>(bbox.y - bbox.height* CROP_RATIO), 0);
                crop_box_width = std::max(static_cast<int>((bbox.x - top_left_x) * 2 + bbox.width), MIN_CROP_WIDTH);
                crop_box_height = std::max(static_cast<int>((bbox.y - top_left_y) * 2 + bbox.height), MIN_CROP_HEIGHT);   
            }
            //minimum cropped image size is caffe input size
            int max_crop_x;
            int max_crop_y;
            max_crop_x = img.cols - MIN_CROP_WIDTH; //1280-300 = 980;
            max_crop_y = img.rows - MIN_CROP_HEIGHT; //720-300 = 420;
            
            if(top_left_x >= max_crop_x ){
                top_left_x = max_crop_x;
                crop_box_width = MIN_CROP_WIDTH;
            }
            if(top_left_y >= max_crop_y){
                top_left_y = max_crop_y;
                crop_box_height = MIN_CROP_HEIGHT;
            }
            if (top_left_x + crop_box_width > img.cols)
            {
                //YHH's code
                //crop_box_width = (img.cols - top_left_x)/2;
                crop_box_width = (img.cols - top_left_x);
            }
            
            //error handling
            else if(top_left_x <= 0){
                top_left_x = 0;
            }
            if (top_left_y + crop_box_height > img.rows)
            {
                //YHH's code
                //crop_box_height = (img.rows - top_left_y)/2;
                crop_box_height = (img.rows - top_left_y);
            }
            else if(top_left_y <= 0){
                top_left_y = 0;
            }
            img_process = img(cv::Rect(top_left_x, top_left_y, crop_box_width, crop_box_height));
			//for debugging
			cv::Rect2d debug_bbox(top_left_x, top_left_y, crop_box_width, crop_box_height);
            rectangle(img, debug_bbox, cv::Scalar(128,128,0), 2, 1);
			cv::imshow("output", img);
	        cv::waitKey(30); 

			
        }
        else
        {
        }
        //detection: run  
        std::vector<vector<float> > detections = detector.Detect(img_process);
        //dtection: threshold filter
        std::vector<vector<float> > targets;
        for (int i = 0; i < detections.size(); ++i) {
            vector<float> &d_ = detections[i];
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            CHECK_EQ(d_.size(), 7);
            const float score = d_[2];

            if (d_[1] == object_type && score >= confidence_threshold) {
                targets.push_back(d_);
            }
        } 
        vector<float> d;
        //detection: fail
        if(targets.size() == 0)
        {
            Json::Value msg;

            msg["data"]["status"] = "NO OBJECTS";
            msg["type"] = "imageresult";
            
            send_imageresult(msg);
		
            
			continue;
        }
        //detection: multiple objects
        else if(targets.size() > 1)
        {
            //selection policy
            switch(selection_policy)
            {
                //neareset neighbor
                case 0:
					//for debugging 
                    break;
                //color based detection
                case 1:
                    {
                    int max_index = -1;
                    double max_value = 0;
                    for(int i = 0; i < targets.size(); i++)
                    {
                        cv::Mat roi_HSV, roi_thresholded;

						int target_x, target_y, target_width, target_height;
						target_x =  std::max(static_cast<int>(targets[i][3] * img_process.cols), 0);
						target_y =  std::max(static_cast<int>(targets[i][4] * img_process.rows), 0);
                        target_width = static_cast<int>((targets[i][5] - targets[i][3]) * img_process.cols);
                        target_height = static_cast<int>((targets[i][6] - targets[i][4]) * img_process.rows);
						if(target_x + target_width > img_process.cols ){
							target_width = img_process.cols - target_x;
						}
						if(target_y + target_height > img_process.rows){
							target_height = img_process.rows - target_y;
						}
						cv::Rect2d debug_bbox(target_x, target_y, target_width, target_height);
						std::cout << "cols(x) " << img_process.cols << "rows(y) " << img_process.rows ;
						std::cout << "debug bbox " << debug_bbox << std::endl;
						
						cv::Mat roi(img_process, debug_bbox);
                        //cv::Mat roi(img_process, cv::Rect(targets[i][3] * img_process.cols, targets[i][4] * img_process.rows, 
                        //        (targets[i][5] - targets[i][3]) * img_process.cols,
                        //        (targets[i][6] - targets[i][4]) * img_process.rows));
						cv::cvtColor(roi, roi_HSV, cv::COLOR_BGR2HSV);
                        cv::inRange(roi_HSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), roi_thresholded); 
                        int score_input = cv::sum(roi_thresholded)[0] / 255; 
                        double score = (double)score_input / (double)roi_thresholded.total();

                        //std::cout << "score_input: " << score_input << "roi_total: " << roi_thresholded.total() << "score: " << score << std::endl;

                        if(score > color_confidence_ratio)
                        {
                            if(score > max_value)
                            {
                                max_value = score;
                                max_index = i;
                            }
                        }
                    }

                    //error - no objects 
                    if(max_index == -1)
                    {
                        Json::Value msg;
                        msg["data"]["status"] = "MULTIPLE OBJECTS NO TARGET";
                        msg["type"] = "imageresult";
                        send_imageresult(msg);

                        std::cerr << "multiple objects: no targets" << std::endl
                            << "max_score: " << max_value << std::endl;   
                        continue;
                    }

                    d = targets[max_index];

                    /* debugging - hhyeo
                    cv::Mat testimg(img, cv::Rect(targets[max_index][3] * img.cols, targets[max_index][4] * img.rows, 
                                (targets[max_index][5] - targets[max_index][3]) * img.cols,
                                (targets[max_index][6] - targets[max_index][4]) * img.rows));

                    cv::imshow("output", testimg);
                    cv::waitKey(30); 
                    */

                    break;
                    }
                default:
                    break;
            }

        }
        //detection: single object
        else
        {
            d = targets[0];
        }

        //continue;


        int track_max_width = 200;
        int track_max_height = 200;

        int reduce_x = 0;
        int reduce_y = 0;
        int reduce_width = 0;
        int reduce_height = 0;
        int my_width, my_height;

        // //postprocess - determine bbox
        // if(initial_crop_enable)
        // {

        // }
        // // detection and tracking crop should be separated with this if-statement
        // // before we start tracking, we should re-cordinate bbox or draw bbox
        // else if(detection_crop_enable)
        // {

        // }
        if(tracking_crop_enable)
        {
            //draw bbox for drawing
            //Tracker uses only cropped bbox
            if(initial_crop_enable ||detection_crop_enable){
                draw_bbox.width = static_cast<int> (d[5] * img_process.cols - d[3] * img_process.cols);
                draw_bbox.height = static_cast<int> (d[6] * img_process.rows - d[4] * img_process.rows);
                draw_bbox.x = static_cast<int> (d[3]* img_process.cols + top_left_x);  
                draw_bbox.y = static_cast<int> (d[4]* img_process.rows + top_left_y);
            }
            else{
                draw_bbox.width = static_cast<int> (d[5] * img_process.cols - d[3] * img_process.cols);
                draw_bbox.height = static_cast<int> (d[6] * img_process.rows - d[4] * img_process.rows);
                draw_bbox.x = static_cast<int> (d[3]* img_process.cols);  
                draw_bbox.y = static_cast<int> (d[4]* img_process.rows);
            }
        

            my_width = draw_bbox.width;
            my_height = draw_bbox.height;
            
            //reduce x calculation
            if(my_width > track_max_width){
                reduce_width = my_width - track_max_width;
                reduce_x = (my_width - track_max_width)/2;
                bbox.x = draw_bbox.x + reduce_x;
                bbox.width = track_max_width;
            }
            else{
                reduce_width = 0;
                reduce_x = 0;
                bbox.x = draw_bbox.x;
                bbox.width = my_width;
            }
            //reduce y calculation
            if(my_height > track_max_height){
                reduce_height = my_height - track_max_height;
                reduce_y = (my_height - track_max_height)/2;
                bbox.y = draw_bbox.y + reduce_y;
                bbox.height = track_max_height;
            }
            else{
                reduce_height = 0;
                reduce_y = 0;
                bbox.y = draw_bbox.y;
                bbox.height = my_height;
            }

        }
        else
        {
            if (initial_crop_enable ||detection_crop_enable){
                bbox.width = static_cast<int> (d[5] * img_process.cols - d[3] * img_process.cols);
                bbox.height = static_cast<int> (d[6] * img_process.rows - d[4] * img_process.rows);
                bbox.x = static_cast<int> (d[3]* img_process.cols + top_left_x);  
                bbox.y = static_cast<int> (d[4]* img_process.rows + top_left_y); 
            }
            else{
                bbox.width = static_cast<int> (d[5] * img_process.cols - d[3] * img_process.cols);
                bbox.height = static_cast<int> (d[6] * img_process.rows - d[4] * img_process.rows);
                bbox.x = static_cast<int> (d[3]* img_process.cols);  
                bbox.y = static_cast<int> (d[4]* img_process.rows);    
            }
        }

        //visualize detection
        if(visualize_detection_enable)
        {
            rectangle(img, bbox, cv::Scalar(255,0,0), 2, 1);
            cv::imshow("output", img);
            cv::waitKey(30);   
        }
        //tracker initialization
        //cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
#if USE_TrackerKCF
        cv::Ptr<cv::TrackerKCF> tracker = cv::TrackerKCF::create();
#else
        cv::Ptr<cv::Tracker> tracker = cv::Tracker::create("KCF");
#endif
        std::cout << bbox << std::endl;
        tracker->init(img, bbox);

        //tracking
        double start_time = clock();
        double elapsed_time;

        for(int i = 0; i < track_frame_num; i++)
        {
            if(is_quit)
                break;

            bool success = cap.read(img);
            if(!success)
            {
                LOG(INFO) << "Process " << std::endl;
                break;
            }
            tracker->update(img, bbox);

            if(i % send_msg_per_frame == 0)
            {
                elapsed_time = (clock() - start_time) / CLOCKS_PER_SEC;
                
                Json::Value msg;
                msg["type"] = "imageresult";
                msg["data"]["status"] = "SUCCESS";
                msg["data"]["x_min"] = bbox.x;
                msg["data"]["y_min"] = bbox.y;
                msg["data"]["width"] = bbox.width;
                msg["data"]["height"] = bbox.height;
                msg["data"]["time"] = elapsed_time; 
                msg["data"]["v_width"] = frame_width;
                msg["data"]["v_height"] = frame_height;

                send_imageresult(msg);

                start_time = clock();
            }

            //visualize tracking
            if(visualize_tracking_enable)
            {
                if(tracking_crop_enable){
                    //draw_bbox update
                    draw_bbox.x = bbox.x - reduce_x;
                    draw_bbox.y = bbox.y - reduce_y;
                    draw_bbox.width = bbox.width +reduce_width;
                    draw_bbox.height = bbox.height + reduce_height;

                    if(draw_bbox.x <=0){
                        draw_bbox.x = 0;
                    }
                    if(draw_bbox.y <=0){
                        draw_bbox.y = 0;
                    }

                    rectangle(img, draw_bbox, cv::Scalar(255,255,0), 2, 1);
                    rectangle(img, bbox, cv::Scalar(255,0,0), 2, 1);
                    cv::imshow("output", img);
                    cv::waitKey(30);   
                }
                else{
                    rectangle(img, bbox, cv::Scalar(255,0,0), 2, 1);
                    cv::imshow("output", img);
                    cv::waitKey(30);   
                }
            }
        }
        tracker->clear();
    }

    //exit
    if (cap.isOpened()) {
        cap.release();
    }
}

void *network_handler(void *arg)
{
    char buf[BUF_SIZE];
    int read_len;

    std::string rcv, str;
    Json::Reader reader;
    Json::Value data;
    Json::StyledWriter writer;

    /***********************SERVER START****************************/
    while(1){
        memset(buf, 0, sizeof(buf));
        read_len = read(clnt_sock,buf, BUF_SIZE);
        if (read_len == 0){
            std::cout << "Connection closed " <<std::endl;
            break;
        }
        rcv = string(buf);

        bool parsingRet = reader.parse(rcv, data);
        if (!parsingRet)
        {
            std::cerr << "Failed to parse Json: "  << reader.getFormattedErrorMessages();
            continue;
        }

        //Parse json data
        Json::Value type= data["type"];
        Json::Value cmd = data["data"]["command"];

        Json::Value object_json = data["target"]["object"];
        Json::Value index_json = data["target"]["index"];


        if(cmd.isNull() || type.isNull())
        {
            std::cerr << "Json format is wrong :" << buf << std::endl;
            continue;
        }

        //Handle each command from Drone Net
        if(cmd.asString().compare("track") == 0)
        {
            Json::Value action_json = data["data"]["action"];
            Json::Value object_json = data["data"]["object"];
            Json::Value index_json = data["data"]["index"];

            testout << "track comes" << std::endl;
            std::string tmp_str;
            if(action_json.isNull())
            {
                std::cerr << "[Track] Json format is wrong (none actino field) :" << buf << std::endl;
            }

            tmp_str = action_json.asString();

            if(tmp_str.compare("start") == 0)
            {
                testout << "track start comes" << std::endl;
                if(object_json.isNull() || index_json.isNull())
                {
                    std::cerr << "[Track] Json format is wrong :" << buf << std::endl;
                }

                //Get the object value from json
                tmp_str = object_json.asString();
                if(tmp_str.compare("car"))
                {
                    object = CAR;
                }
                else if(tmp_str.compare("human"))
                {
                    object =  HUMAN; 
                }
                else
                {
                    std::cerr << "Object is invalid: " << buf << std::endl;
                    continue;
                }

                //Get the index value from json 
                index_obj = atoi(index_json.asString().c_str());

                pthread_mutex_lock(&track_mutex);
                is_detect_run = true;
                is_detect_thisframe = true; 
                pthread_mutex_unlock(&track_mutex);
                pthread_cond_signal(&track_cond);
            }
            else if(tmp_str.compare("stop") == 0)
            {
                pthread_mutex_lock(&track_mutex);
                is_detect_run = false;
                pthread_mutex_unlock(&track_mutex);
            }
            else
            {
                std::cerr << "[Track] Action is invalid: " << buf << std::endl;
                continue;
            }
        }
        else if(cmd.asString().compare("stream") == 0)
        {
            Json::Value action_json = data["data"]["action"];

            std::string tmp_str;
            if(action_json.isNull())
            {
                std::cerr << "[Stream] Json format is wrong (none actino field) :" << buf << std::endl;
            }

            tmp_str = action_json.asString();
            if(tmp_str.compare("start") == 0)
            {
                pthread_mutex_lock(&track_mutex);
                is_stream = true;
                pthread_mutex_unlock(&track_mutex);
            }
            else if(tmp_str.compare("stop") == 0)
            {
                pthread_mutex_lock(&track_mutex);
                is_stream = false;
                pthread_mutex_unlock(&track_mutex);
            }
            else
            {
                std::cerr << "[Stream] Action is invalid: " << buf << std::endl;
                continue;
            }
        }
        //Detect on a whole image
        else if(cmd.asString().compare("redetect") == 0)
        {
            pthread_mutex_lock(&track_mutex);
            is_detect_run = true;
            is_detect_thisframe = true; 
            pthread_mutex_unlock(&track_mutex);
            pthread_cond_signal(&track_cond);

        }
        else if(cmd.asString().compare("exit") == 0)
        {
            pthread_mutex_lock(&track_mutex);
            is_quit = true;
            pthread_mutex_unlock(&track_mutex);
            pthread_cond_signal(&track_cond);
            testout << "network thread ends" << std::endl;
            pthread_exit(NULL);

        }
        else if(cmd.asString().compare("state") == 0)
        {
            Json::Value data_json;

            data_json["type"] = "return";

            if(is_detect_run)
                data_json["data"]["track"] = "on";
            else
                data_json["data"]["track"] = "off";

            if(is_stream)
                data_json["data"]["stream"] = "on";
            else
                data_json["data"]["stream"] = "off";

            Json::StyledWriter writer;
            std::string str = writer.write(data_json);

            if(send(clnt_sock, str.data(), str.size(), 0) < 0)
            {
                perror("tracker sends error");
                continue;
            }
        }
        else if(cmd.asString().compare("config") == 0)
        {
            Json::Value interval_json= data["data"]["interval"];
            int input_period = atoi(interval_json.asString().c_str());
            pthread_mutex_lock(&track_mutex);
            send_track_period = input_period;
            pthread_mutex_unlock(&track_mutex);
            testout << "track_preriod: " << input_period << std::endl;

        }
        else
        {
            std::cerr << "Command is invalid" << std::endl;
            continue;
        } 
        testout << "cmd: " << cmd.asString() << std::endl;
    }
    return NULL;
}

bool send_imageresult(Json::Value msg)
{
    Json::StyledWriter writer;
    std::string str = writer.write(msg);

    if(send(clnt_sock, str.data(), str.size(), 0) < 0)
    {
        perror("imagemresult: send error");
        return false;
    }

    return true;
}

void error_handling(char * buf){
    fputs(buf, stderr);
    fputs(" ", stderr);
    exit(1);
}

void test_json()
{
    Json::Value root;
    std::string str;
    root["id"] = "Luna";
    root["name"] = "Silver";
    root["age"] = 19;
    root["hasCar"] = false;

    Json::Value items;
    items.append("nootbook");
    items.append("ipadmini2");
    items.append("iphone5s");
    root["items"] = items;

    Json::Value friends;
    Json::Value tom;
    tom["name"] = "Tom";
    tom["age"] = 21;
    Json::Value jane;
    jane["name"] = "jane";
    jane["age"] = 23;
    friends.append(tom);
    friends.append(jane);
    root["friends"] = friends;

    Json::StyledWriter writer;
    str = writer.write(root);
    logout << "test_json: " << str << std::endl;
}



#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV




