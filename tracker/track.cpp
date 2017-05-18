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
//
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

#define TRACKING_METHOD "KCF"
#define CROP_RATIO 0.5

//for networking
#define BUF_SIZE 4096 
#define LISTEN_PORT "44444"
int clnt_sock;

//for debugging
#define NETWORK_DEBUG 0 

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
int index_obj = 0;
int object = 0; //NOT USED YET VER.1.0
#define CAR 0
#define HUMAN 1


int port_num;

void error_handling(char * buf);
void * network_handler(void * arg);
void test_json();
void *detection_handler(void *arg);

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
DEFINE_double(confidence_threshold, 0.4,
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

    printf(" waiting connection ... \n");
    memset(&clnt_adr, 0, sizeof(clnt_adr));
    memset(&clnt_adr_sz, 0, sizeof(clnt_adr_sz));
    clnt_sock = accept(serv_sock,(struct sockaddr *)&clnt_adr, &clnt_adr_sz);
    printf(" connected \n");

    //Network handler thread start!

    pthread_t network_thread, detection_thread;
    pthread_create(&network_thread, NULL, network_handler, NULL);

#if (NETWORK_DEBUG != 1)
    pthread_create(&detection_thread, NULL, detection_handler, &input_args);
#endif
    pthread_join(network_thread, NULL);

#if (NETWORK_DEBUG != 1)
    pthread_join(detection_thread, NULL);
#endif

    close(serv_sock);   
    return 0;
}

void *detection_handler(void *arg)
{
    vector<string> *input_args = (vector<string> *)arg;

    const string& model_file = input_args->operator[](0);
    const string& weights_file = input_args->operator[](1);
    const string& mean_file = FLAGS_mean_file;
    const string& mean_value = FLAGS_mean_value;
    const string& file_type = FLAGS_file_type;
    const string& out_file = FLAGS_out_file;
    const float confidence_threshold = FLAGS_confidence_threshold;

    std::cout << model_file << std::endl << weights_file << std::endl;

    // Initialize the network.

    Detector detector(model_file, weights_file, mean_file, mean_value);

    // Set the output mode.
    std::streambuf* buf = std::cout.rdbuf();
    std::ofstream outfile;
    if (!out_file.empty()) {
        outfile.open(out_file.c_str());
        if (outfile.good()) {
            buf = outfile.rdbuf();
        }
    }
    std::ostream out(buf);

    // Process image one by one.
    std::string file;

    if (file_type == "image") {
        cv::Mat img = cv::imread(input_args->operator[](2), -1);
        CHECK(!img.empty()) << "Unable to decode image " << file;
        std::vector<vector<float> > detections = detector.Detect(img);

        /* Print the detection results. */
        for (int i = 0; i < detections.size(); ++i) {
            const vector<float>& d = detections[i];
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            CHECK_EQ(d.size(), 7);
            const float score = d[2];
            if (score >= confidence_threshold) {
                out << file << " ";
                out << static_cast<int>(d[1]) << " ";
                out << score << " ";
                out << static_cast<int>(d[3] * img.cols) << " ";
                out << static_cast<int>(d[4] * img.rows) << " ";
                out << static_cast<int>(d[5] * img.cols) << " ";
                out << static_cast<int>(d[6] * img.rows) << std::endl;
            }
        }
    } else if (file_type == "video") {
        cv::VideoCapture cap(input_args->operator[](2));
        //cv::VideoCapture cap(file);
        cv::namedWindow("test",1);

        cv::Ptr<cv::Tracker> tracker = cv::Tracker::create(TRACKING_METHOD);
        cv::Rect2d bbox(600,150,100,100);
        if (!cap.isOpened()) {
            LOG(FATAL) << "Failed to open video: " << file;
        }
        cv::Mat img, sub_img;
        bool success = cap.read(img);
        bool is_first_detect = true;
        bool detect_success = false;
        tracker->init(img, bbox);
        int frame_count = 0, top_left_x = 0, top_left_y = 0, tmp_width = 0, tmp_height = 0;
        
        while (true) {
            pthread_mutex_lock(&track_mutex);
            while(!is_detect_run)
                pthread_cond_wait(&track_cond, &track_mutex);
                
            success = cap.read(img);

            if (!success) {
                LOG(INFO) << "Process " << frame_count << " frames from " << file;
                pthread_mutex_unlock(&track_mutex);
                break;
            }
            CHECK(!img.empty()) << "Error when read frame";

            // detect objects per 10 frames
            if (frame_count % 30 == 0 || is_detect_thisframe){
                //Crop image using prior tracking result

                detect_success = false;

                if(!is_first_detect || is_detect_thisframe)
                {
                    std::cout << "img row: " << img.rows<< "img col " << img.cols << std::endl;
                    top_left_x = std::max(static_cast<int>(bbox.x - bbox.width* CROP_RATIO), 0); 
                    top_left_x = std::min(top_left_x, img.cols);
                    top_left_y = std::max(static_cast<int>(bbox.y - bbox.height* CROP_RATIO), 0); 
                    top_left_y = std::min(top_left_y, img.rows);

                    tmp_width = (bbox.x - top_left_x) * 2 + bbox.width;
                    tmp_height = (bbox.y - top_left_y) * 2 + bbox.height;

                    if (top_left_x + tmp_width > img.cols)
                    {
                        tmp_width = (img.cols - top_left_x)/2;
                    }

                    if (top_left_y + tmp_height > img.rows)
                    {
                        tmp_height = (img.rows - top_left_y)/2;
                    }

                    sub_img = img(cv::Rect(top_left_x, top_left_y, tmp_width, tmp_height));  


                }
                else
                {
                    sub_img = img(cv::Rect(280,0,720,720));
                    is_first_detect = false;
                }

                std::vector<vector<float> > detections = detector.Detect(sub_img);

                int my_width, my_height;
                int x_avg, y_avg, count_car = 0, count_person = 0;
                cv::Rect2d min_rect(0,0,1,1);
                float min_distance = 0;
                /* Print the detection results. */
                for (int i = 0; i < detections.size(); ++i) {
                    const vector<float>& d = detections[i];
                    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                    CHECK_EQ(d.size(), 7);
                    const float score = d[2];

                    if (score >= confidence_threshold) {

                        out << file << "_";
                        out << std::setfill('0') << std::setw(6) << frame_count << " ";
                        out << static_cast<int>(d[1]) << " ";
                        out << score << " ";
                        out << static_cast<int>(d[3] * sub_img.cols) << " ";
                        out << static_cast<int>(d[4] * sub_img.rows) << " ";
                        out << static_cast<int>(d[5] * sub_img.cols) << " ";
                        out << static_cast<int>(d[6] * sub_img.rows) << std::endl;
                        /*          cv::line(sub_img, cv::Point(d[3]* sub_img.cols,d[4] * sub_img.rows), \
                                    cv::Point(d[5]*sub_img.cols,d[6]*sub_img.rows), cv::Scalar(255,255,0));
                                    */
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
                        /************************SHOULD BE UPDATED-START*************************/
                        // TODO: Multi obeject detection handling
                        // 1) Box object should be saved in an array 
                        // 2) Multi-object tracking should be implemented

                        //Person
                        if (d[1] == 15){
                            if(count_person == 1)
                            {
                                printf("Multiple person: detection failed\n");
                                count_person++;
                                break;
                            }

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

                            printf("bbox.x + bbox.width/2 : %f crop (x_avg ) %d \n"
                                    , bbox.x + bbox.width/2, (x_avg ));

                            min_rect.x = d[3]*sub_img.cols + top_left_x;
                            min_rect.y = d[4]*sub_img.rows + top_left_y;
                            min_rect.height = my_height;
                            min_rect.width = my_width;

                            printf("det_height %f det_width %f det_x %f det_y %f\n",
                                    min_rect.height, min_rect.width, min_rect.x, min_rect.y);
                            printf("min_distance : %f\n",min_distance);
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

                            printf("bbox.x + bbox.width/2 : %f crop (x_avg ) %d \n"
                                    , bbox.x + bbox.width/2, (x_avg ));

                            if ( cur_distance < min_distance || min_distance == 0){
                                min_distance = cur_distance;
                                min_rect.x = d[3]*sub_img.cols + top_left_x;
                                min_rect.y = d[4]*sub_img.rows + top_left_y;
                                min_rect.height = my_height;
                                min_rect.width = my_width;
                            } 
                            /*
                               bbox.height = my_height;
                               bbox.width = my_width;
                               bbox.x = d[3]*sub_img.cols + top_left_x;
                               bbox.y = d[4]*sub_img.rows + top_left_y;
                               */
                            printf("det_height %f det_width %f det_x %f det_y %f\n",
                                    min_rect.height, min_rect.width, min_rect.x, min_rect.y);
                            printf("min_distance : %f\n",min_distance);
                            detect_success = true;
                            count_car++;
#endif
                        }
                        /************************SHOULD BE UPDATED-END*************************/
                        printf("count_car : %d\n", count_car);
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
                        is_detect_run = false; 

                        data_json["status"] = "MULTI_OBJECTS";
                        Json::StyledWriter writer;
                        std::string str = writer.write(data_json);

                        if(send(clnt_sock, str.data(), str.size(), 0) < 0)
                        {
                            perror("tracker sends error");
                            continue;
                        }
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


                        /* TODO: Should we send detection result?  
                        Json::Value data_json;
                        data_json["status"] = "SUCCESS";
                        Json::StyledWriter writer;
                        std::string str = writer.write(data_json);

                        if(send(clnt_sock, str.data(), str.size(), 0) < 0)
                        {
                            perror("tracker sends error");
                            continue;
                        }
                        */

                    }

                }   
                //Detection fail error
                else
                {
                    is_detect_run = false; 

                    Json::Value data_json;
                    data_json["status"] = "NO_OBJECTS";
                    Json::StyledWriter writer;
                    std::string str = writer.write(data_json);

                    if(send(clnt_sock, str.data(), str.size(), 0) < 0)
                    {
                        perror("tracker sends error");
                        continue;
                    }

                    std::cout << "detection fail" << std::endl;
                    std::cout << "min_distance: " << min_distance << "bbox.width: " << bbox.width << "bbox.height: " << bbox.height << std::endl;
                }
                count_car = 0;
                count_person = 0;
                is_detect_thisframe = false;
            }
            //Handle tracking 
            else{
                tracker -> update(img, bbox); 
                rectangle(img, bbox, cv::Scalar(255,0,0),2,1);
                printf("height %f width %f x %f y %f\n", bbox.height, bbox.width, bbox.x, bbox.y);

                Json::Value data_json;
                data_json["status"] = "SUCCESS";
                data_json["data"]["x_min"] = bbox.x;
                data_json["data"]["y_min"] = bbox.y;
                data_json["data"]["width"] = bbox.width;
                data_json["data"]["height"] = bbox.height;

                Json::StyledWriter writer;
                std::string str = writer.write(data_json);

                if(send(clnt_sock, str.data(), str.size(), 0) < 0)
                {
                    perror("tracker sends error");
                    continue;
                }
            }
            cv::imshow("test",img);
            cv::waitKey(30);   
            ++frame_count;

            pthread_mutex_unlock(&track_mutex);
        }

        if (cap.isOpened()) {
            cap.release();
        }
    } else {
        LOG(FATAL) << "Unknown file_type: " << file_type;
    }

    std::cout << confidence_threshold << std::endl;
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

        printf("input process\n");
        bool parsingRet = reader.parse(rcv, data);
        if (!parsingRet)
        {
            std::cerr << "Failed to parse Json: "  << reader.getFormattedErrorMessages();
            continue;
        }

        //Parse json data
        Json::Value cmd = data["cmd"];
        Json::Value object_json = data["target"]["object"];
        Json::Value index_json = data["target"]["index"];

        if(cmd.isNull())
        {
            std::cerr << "Json format is wrong :" << buf << std::endl;
            continue;
        }

        //Handle each command from Drone Net
        if(cmd.asString().compare("track") == 0)
        {
            if(object_json.isNull() || index_json.isNull())
            {
                std::cerr << "[Track] Json format is wrong :" << buf << std::endl;
            }

            //Get the object value from json
            std::string tmp_str = object_json.asString();
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
            pthread_cond_signal(&track_cond);
            pthread_mutex_unlock(&track_mutex);
        }
        //Detect on a whole image
        else if(cmd.asString().compare("redetect") == 0)
        {
            pthread_mutex_lock(&track_mutex);
            is_detect_run = true;
            is_detect_thisframe = true; 
            pthread_cond_signal(&track_cond);
            pthread_mutex_unlock(&track_mutex);

        }
        else if(cmd.asString().compare("stop") == 0)
        {
            pthread_mutex_lock(&track_mutex);
            is_detect_run = false;
            pthread_mutex_unlock(&track_mutex);

        }
        else
        {
            std::cerr << "Command is invalid" << std::endl;
            continue;
        }


        std::cout << cmd.asString() << std::endl;
        std::cout << object_json.asString() << std::endl;
        std::cout << index_json.asString() << std::endl;
    }

    return 0;
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
    std::cout << str << std::endl << std::endl;
}



#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
