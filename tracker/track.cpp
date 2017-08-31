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

//for networking
#define BUF_SIZE 4096 
#define LISTEN_PORT "44444"
int clnt_sock;

//for debugging and logging
#define USE_STREAM 1
#define GCS_STREAM 0 
#define NORM_LOG_ENABLED 0
#define TEST_LOG_ENABLED 1 

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

bool initial_crop_enable = false;
bool detection_crop_enable = false;
bool tracking_crop_enable = false;
bool visualize_detection_enable = true;
bool visualize_tracking_enable = true;

int track_frame_num = 30;


#define CAR 0
#define HUMAN 1


int port_num;

void error_handling(char * buf);
void * network_handler(void * arg);
void test_json();
void *detection_handler(void *arg);
int write_log(const char *foramat, ...);

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

    while(1)
    {
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
        }
        else if(detection_crop_enable)
        {
        }
        else
        {
        }

        int object_number = 0;
        //detection: run  
        std::vector<vector<float> > detections = detector.Detect(img_process);

        //dtection: threshold filter
        vector<float> d;
        for (int i = 0; i < detections.size(); ++i) {
            vector<float> &d_ = detections[i];
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            CHECK_EQ(d.size(), 7);
            const float score = d[2];

            if (score >= confidence_threshold) {
                object_number++;
            }

            d = d_;
        } 

        logout << "object number: " << object_number << std::endl;

        //detection: fail
        if(object_number == 0)
        {
        }
        //detection: multiple objects
        else if(object_number > 1)
        {
        }
        //detection: single object
        else
        {
        }

        cv::Rect2d bbox;

        //postprocess - determine bbox
        if(initial_crop_enable)
        {
        }
        else if(detection_crop_enable)
        {
        }
        else if(tracking_crop_enable)
        {
        }
        else
        {
            bbox.width = d[5] * img.cols - d[3] * img.cols;
            bbox.height = d[6] * img.rows - d[4] * img.rows;
            bbox.x = d[3];  
            bbox.y = d[4]; 
        }

        //visualize detection
        if(visualize_detection_enable)
        {
            rectangle(img, bbox, cv::Scalar(255,0,0), 2, 1);
            cv::imshow("output", img);
        }

        //tracker initialization
        cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
        tracker->init(img, bbox);

        //tracking
        for(int i = 0; i < track_frame_num; i++)
        {
            bool success = cap.read(img);
            if(!success)
            {
                LOG(INFO) << "Process " << std::endl;
                break;
            }
            tracker->update(img, bbox);

            //visualize tracking
            if(visualize_tracking_enable)
            {
                rectangle(img, bbox, cv::Scalar(255,0,0), 2, 1);
                cv::imshow("output", img);
            }
        }
        tracker->clear();
    }

    //exit
    if (cap.isOpened()) {
        cap.release();
    }


    /*
    cv::namedWindow("output",1);


    cv::Rect2d bbox(600,150,100,100);
    cv::Rect2d draw_bbox(600,150,100,100);
    float reduce_x = 0;
    float reduce_y = 0;
    float reduce_width = 0;
    float reduce_height = 0;

    bool is_first_detect = true;
    bool detect_success = false;
    bool send_track_result= false;
    tracker->init(img, bbox);
    int frame_count = 0, top_left_x = 0, top_left_y = 0, tmp_width = 0, tmp_height = 0;

    clock_t send_track_timer;

    while (true) {
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

        clock_t before_read_img;
        double time_diff;
        before_read_img = clock();
        success = cap.read(img);
        if (!success) {
            LOG(INFO) << "Process " << frame_count << " frames from " << file;
            //    pthread_mutex_unlock(&track_mutex);
            break;
        }
        CHECK(!img.empty()) << "Error when read frame";

        // detect objects 30 frames
        if (frame_count % 30 == 0 || is_detect_thisframe){
            //Crop image using prior tracking result

            detect_success = false;

            if(!is_first_detect || is_detect_thisframe)
            {
                logout << "image row: " << img.rows << " image col: " << img.cols << std::endl;
                //top_left_x = std::max(static_cast<int>(bbox.x - bbox.width* CROP_RATIO), 0); 
                //top_left_x = std::min(top_left_x, img.cols);
                //top_left_y = std::max(static_cast<int>(bbox.y - bbox.height* CROP_RATIO), 0); 
                //top_left_y = std::min(top_left_y, img.rows);
                //tmp_width = (bbox.x - top_left_x) * 2 + bbox.width;
                //tmp_height = (bbox.y - top_left_y) * 2 + bbox.height;
                top_left_x = std::max(static_cast<int>(draw_bbox.x - draw_bbox.width* CROP_RATIO), 0); 
                top_left_x = std::min(top_left_x, img.cols);
                top_left_y = std::max(static_cast<int>(draw_bbox.y - draw_bbox.height* CROP_RATIO), 0); 
                top_left_y = std::min(top_left_y, img.rows);

                tmp_width = (draw_bbox.x - top_left_x) * 2 + draw_bbox.width;
                tmp_height = (draw_bbox.y - top_left_y) * 2 + draw_bbox.height;

                if (top_left_x + tmp_width > img.cols)
                {
                    //YHH's code
                    //tmp_width = (img.cols - top_left_x)/2;
                    tmp_width = (img.cols - top_left_x);
                }
                //error handling
                else if(top_left_x <= 0){
                    top_left_x = 0;
                }

                if (top_left_y + tmp_height > img.rows)
                {
                    //YHH's code
                    //tmp_height = (img.rows - top_left_y)/2;
                    tmp_height = (img.rows - top_left_y);
                }
                else if(top_left_y <= 0){
                    top_left_y = 0;
                }
                testout<< "before sub_img " << endl;
                testout<< "x" << top_left_x <<" y"<< 
                    top_left_y << "tmp_width" << tmp_width << "tmp_height " << tmp_height << endl;
                sub_img = img(cv::Rect(top_left_x, top_left_y, tmp_width, tmp_height));  
                testout<< "after  sub img " << endl;

            }
            else
            {
                testout<< "first detect !!!!!!!!!!!!!!!!!"<<endl;
                top_left_x = 280, top_left_y = 0, tmp_width = 0, tmp_height = 0;
                sub_img = img(cv::Rect(280,0,720,720));
                is_first_detect = false;
            }

            std::vector<vector<float> > detections = detector.Detect(sub_img);

            int my_width, my_height;
            int x_avg, y_avg, count_car = 0, count_person = 0;
            cv::Rect2d min_rect(0,0,1,1);
            float min_distance = 0;
            float max_score = 0;
            // Print the detection results. 
            for (int i = 0; i < detections.size(); ++i) {
                const vector<float>& d = detections[i];
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                CHECK_EQ(d.size(), 7);
                const float score = d[2];

                if (score >= confidence_threshold) {

                    logout << file << "_";
                    logout << std::setfill('0') << std::setw(6) << frame_count << " ";
                    logout << static_cast<int>(d[1]) << " ";
                    logout << score << " ";
                    logout << static_cast<int>(d[3] * sub_img.cols) << " ";
                    logout << static_cast<int>(d[4] * sub_img.rows) << " ";
                    logout << static_cast<int>(d[5] * sub_img.cols) << " ";
                    logout << static_cast<int>(d[6] * sub_img.rows) << std::endl;
                    //          cv::line(sub_img, cv::Point(d[3]* sub_img.cols,d[4] * sub_img.rows), \
                    //            cv::Point(d[5]*sub_img.cols,d[6]*sub_img.rows), cv::Scalar(255,255,0));
                    my_width = d[5]* sub_img.cols - d[3]* sub_img.cols;
                    my_height = d[6] * sub_img.rows - d[4] * sub_img.rows;

                    x_avg =( d[3]* sub_img.cols + d[5] *sub_img.cols)/2;
                    x_avg = x_avg + top_left_x; 
                    y_avg =( d[4]* sub_img.rows + d[6] *sub_img.rows)/2;
                    y_avg = y_avg + top_left_y;
                    int baseline = 0;
                    string text;
                    int fontFace = cv::FONT_HERSHEY_PLAIN;
                    double fontScale = 2;
                    int thickness = 2;
                    
                    // TODO: Multi obeject detection handling
                    // 1) Box object should be saved in an array 
                    // 2) Multi-object tracking should be implemented

                    //Person
                    if (d[1] == 15){
                        //if(count_person == 1)
                        //  {
                        //  logout << "Multiple person: detection failed\n" << std::endl;
                        //  count_person++;
                        //  break;
                        //  }

                        text = "person";
                        cv::Size textSize = cv::getTextSize(text, fontFace,
                                fontScale, thickness, &baseline);
                        baseline += thickness;
                        cv::Point textOrg(x_avg - textSize.width/2 ,y_avg - textSize.height/2);
                        putText(img, text, textOrg, fontFace, fontScale,
                                cv::Scalar::all(0), thickness, 8);

                        rectangle(img, cv::Point(d[3]*sub_img.cols + top_left_x, d[4]*sub_img.rows + top_left_y), 
                                cv::Point(d[5]*sub_img.cols + top_left_x, d[6]*sub_img.rows + top_left_y),
                                cv::Scalar(255,0,0),2,8);

                        logout << "bbox.x + bbox.width/2: " << bbox.x + bbox.width/2 << "crop (x_avg ): " << x_avg << std::endl;
                        //to track the hightest score object 
                        if(score > max_score){
                            draw_bbox.x = d[3]*sub_img.cols + top_left_x;
                            draw_bbox.y = d[4]*sub_img.rows + top_left_y;
                            draw_bbox.height = my_height;
                            draw_bbox.width = my_width;
                            //if my_width or my_height is greater than 40, we resize it to 40 for fast tracking	i
                            int max_width = 200;
                            int max_height = 200;
                            if(my_width > max_width){
                                //	printf("exceed width !!!!!!!!!!!\n");
                                reduce_width = my_width - max_width;
                                reduce_x = (my_width - max_width)/2;
                                min_rect.x = d[3]*sub_img.cols + top_left_x + reduce_x;
                                min_rect.width = max_width;
                            }
                            else{
                                reduce_width = 0;
                                reduce_x = 0;
                                min_rect.x = d[3]*sub_img.cols + top_left_x;
                                min_rect.width = my_width;
                            }
                            if(my_height > max_height){
                                //printf("exceed height !!!!!!!!!!\n");
                                reduce_height = my_height - max_height;
                                reduce_y = (my_height - max_height)/2;
                                min_rect.y = d[4]*sub_img.rows + top_left_y + reduce_y;
                                min_rect.height = max_height;
                            }
                            else{
                                reduce_height = 0;
                                reduce_y = 0;
                                min_rect.y = d[4]*sub_img.rows + top_left_y;
                                min_rect.height = my_height;
                            }
                               //min_rect.x = d[3]*sub_img.cols + top_left_x;
                               //min_rect.y = d[4]*sub_img.rows + top_left_y;
                               //min_rect.height = my_height;
                               //min_rect.width = my_width; 
                            max_score = score;
                        }

                        logout << "det_height: " << min_rect << std::endl;
                        testout << "det_height: " << min_rect << std::endl;
                        logout << "min_distance: " << min_distance << std::endl;
                        testout << "min_distance: " << min_distance << std::endl;
                        detect_success = true;
                        count_person++;
                    }
                    //Car
                    else if (d[1] == 7){
#if NEW_VERSION 
                        text = "car";
                        cv::Size textSize = cv::getTextSize(text, fontFace,
                                fontScale, thickness, &baseline);
                        baseline += thickness;
                        cv::Point textOrg(x_avg - textSize.width/2 ,y_avg - textSize.height/2);
                        putText(img, text, textOrg, fontFace, fontScale,
                                cv::Scalar::all(0), thickness, 8);

                        rectangle(img, cv::Point(d[3]*sub_img.cols + top_left_x, d[4]*sub_img.rows + top_left_y), 
                                cv::Point(d[5]*sub_img.cols + top_left_x, d[6]*sub_img.rows + top_left_y),
                                cv::Scalar(255,0,0),2,8);
                        float cur_distance = sqrt((bbox.x + bbox.width/2 - (x_avg)) *(bbox.x + bbox.width/2 - (x_avg)) +
                                (bbox.y + bbox.height/2 - (y_avg)) *(bbox.y + bbox.height/2 - (y_avg)));

                        logout << "bbox.x + bbox.width/2 : " << bbox.x + bbox.width/2 << "crop (x_avg ): " << x_avg << std::endl; 

                        if ( cur_distance < min_distance || min_distance == 0){
                            min_distance = cur_distance;

                            min_rect.x = d[3]*sub_img.cols + top_left_x;
                            min_rect.y = d[4]*sub_img.rows + top_left_y;
                            min_rect.height = my_height;
                            min_rect.width = my_width;

                        } 
                           //bbox.height = my_height;
                           //bbox.width = my_width;
                           //bbox.x = d[3]*sub_img.cols + top_left_x;
                           //bbox.y = d[4]*sub_img.rows + top_left_y;
                        logout << min_rect << std::endl;
                        logout << "min_distance: " << min_distance << std::endl;
                        detect_success = true;
                        count_car++;
#endif
                    }
                    logout << "count_car: " << count_car << std::endl;
                }
            }
            //TODO : if minimum distance is larger than bbox, tracker use old box.
            // do not update bbox

            //std::cout << "minx" << min_rect.x + min_rect.width/2 << "miny" << min_rect.y + min_rect.height/2 << std::endl;
            //containPoint(bbox, min_rect.x + min_rect.width/2, min_rect.y + min_rect.height/2)
            //if (detect_success && min_distance < std::max(bbox.width, bbox.height) * 1)
            if(detect_success)
            {

                Json::Value data_json;

                //Multi-object error
                if(count_person > 1)
                {
                    pthread_mutex_lock(&track_mutex);
                    is_detect_run = true;
                    is_detect_thisframe = true;
                    pthread_mutex_unlock(&track_mutex);

                    data_json["data"]["status"] = "MULTI_OBJECTS";
                    data_json["type"] = "imageresult";
                    Json::StyledWriter writer;
                    std::string str = writer.write(data_json);

                    if(send(clnt_sock, str.data(), str.size(), 0) < 0)
                    {
                        perror("tracker sends error");
                        continue;
                    }
                    //for debugging 	
                    bbox.height = min_rect.height;
                    bbox.width = min_rect.width;
                    bbox.x = min_rect.x;
                    bbox.y = min_rect.y;

                    tracker->clear();
                    tracker = cv::Tracker::create(TRACKING_METHOD);
                    tracker->init(img,bbox);

                }
                //Single-object detection
                else
                {
                    bbox.height = min_rect.height;
                    bbox.width = min_rect.width;
                    bbox.x = min_rect.x;
                    bbox.y = min_rect.y;
                    tracker->clear();
                    tracker = cv::Tracker::create(TRACKING_METHOD);
                    tracker->init(img,bbox);

                }
                testout << "in dectect_success loop (bbox)" << bbox << std::endl;

            }   
            //Detection fail error
            else
            {
                pthread_mutex_lock(&track_mutex);
                is_detect_run = true;
                is_detect_thisframe = true;
                is_first_detect = true;
                pthread_mutex_unlock(&track_mutex);

                Json::Value data_json;
                data_json["data"]["status"] = "NO_OBJECTS";
                data_json["type"] = "imageresult";
                Json::StyledWriter writer;
                std::string str = writer.write(data_json);

                if(send(clnt_sock, str.data(), str.size(), 0) < 0)
                {
                    perror("tracker sends error");
                    continue;
                }

                logout << "detection fail" << std::endl;
            }
            if(count_person > 1){
                pthread_mutex_lock(&track_mutex);
                is_detect_thisframe = true;
                pthread_mutex_unlock(&track_mutex);
            }
            else{
                pthread_mutex_lock(&track_mutex);
                is_detect_thisframe = false;
                pthread_mutex_unlock(&track_mutex);
            }
            count_car = 0;
            count_person = 0;
            max_score = 0;

        }
        //Handle tracking 
        else{
            tracker -> update(img, bbox); 

            //update draw box information 

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
            testout << "draw_bbox " << draw_bbox <<endl;	

            rectangle(img,draw_bbox, cv::Scalar(255,0,0),2,1);
            rectangle(img, bbox, cv::Scalar(255,0,0),2,1);


            logout << bbox << std::endl;

            if(!send_track_result)
            {
                send_track_result = true;
                send_track_timer = clock();
            }
            else
            {
                double time_elapsed = (clock() - send_track_timer) / CLOCKS_PER_SEC;
                if(time_elapsed >= send_track_period)
                {
                    Json::Value data_json;
                    data_json["type"] = "imageresult";
                    data_json["data"]["status"] = "SUCCESS";
                    data_json["data"]["x_min"] = bbox.x;
                    data_json["data"]["y_min"] = bbox.y;
                    data_json["data"]["width"] = bbox.width;
                    data_json["data"]["height"] = bbox.height;
                    data_json["data"]["time"] = time_elapsed; 
                    data_json["data"]["v_width"] = frame_width;
                    data_json["data"]["v_height"] = frame_height;

                    Json::StyledWriter writer;
                    std::string str = writer.write(data_json);

                    if(send(clnt_sock, str.data(), str.size(), 0) < 0)
                    {
                        perror("tracker sends error");
                        continue;
                    }

                    send_track_result = false;
                }
            }
        }
        time_diff = (double) (clock() -before_read_img) /CLOCKS_PER_SEC;

        char frame_text[200];
        int frame_fontFace = cv::FONT_HERSHEY_PLAIN;
        double frame_fontScale = 2;
        int frame_thickness = 2;
        int frame_baseline = 0;
        sprintf(frame_text,"%lf",(1/time_diff));
        testout << "frames : "<< frame_text << endl;
        cv::Size frame_textSize = cv::getTextSize(frame_text, frame_fontFace,
                frame_fontScale, frame_thickness, &frame_baseline);
        frame_baseline += frame_thickness;
        cv::Point frame_textOrg(100 - frame_textSize.width/2 ,100 - frame_textSize.height/2);
        putText(img, frame_text, frame_textOrg, frame_fontFace, frame_fontScale,
                cv::Scalar(128,128,128), frame_thickness, 8);
        cv::imshow("test",img);
        cv::waitKey(30);   
        ++frame_count;

        //Stream boxed image (result of tracking or detectiion
        if(is_stream && GCS_STREAM)
        {
            cv::Mat img_str;
            cv::resize(img,img_str,cv::Size(640,480));
            writer << img_str;
        }
        //    pthread_mutex_unlock(&track_mutex);
    }
    */
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




