#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <list>
#include <sys/wait.h>

using namespace std;
using namespace cv;

int total_frame_num = 50;
int total_process_num = 6;
int total_task_num = 60;
int box_lenght = 100;

int main( int argc, char** argv ){
	//TODO: total_frame_num , total_process_num input args
	// show help
	if(argc<2){
		cout<<
			" Usage: tracker <video_name>\n"
			" examples:\n"
			" example_tracking_kcf Bolt/img/%04d.jpg\n"
			" example_tracking_kcf faceocc2.webm\n"
			<< endl;
		return 0;
	}

	if(argc == 3)
	{
		total_process_num = atoi(argv[2]);
	}

	// declares all required variables
	Mat frame;
	// create a tracker object
	Ptr<Tracker> tracker = Tracker::create( "KCF" );
	// set input video
	std::string video = argv[1];
	VideoCapture cap(video);
	// get bounding box
	cap >> frame;
	printf("initialize tracker\n");
	//roi=selectROI("tracker",frame);
	//quit if ROI was not selected
	//if(roi.width==0 || roi.height==0)
	//		return 0;
	// initialize the tracker

	cv::Rect2d roi = cv::Rect2d(400, 400, 200, 200);
	tracker->init(frame,roi);

	list<Mat> frames = list<Mat>();
	//list<Mat>::iterator iter;
	// perform the tracking process
	printf("data prepare\n");
	printf("total_process_num: %d\n", total_process_num);
	printf("total_task_num: %d\n", total_task_num);
	
	//sequential process
	for (int i = 0; i < total_frame_num ; i++)
	{
		cap >> frame;
		frames.push_back(frame);
	}
	printf("data size: %d\n", frames.size());

	printf("start tracking\n");
	int i;
	double elapsed_time, fps;
	struct timeval t1, t0;

	//start_time = clock();
	/*
	gettimeofday(&t0, 0);
	for (int i = 0; i < total_process_num; i++)
	{
		for (list<Mat>::iterator iter = frames.begin() ; iter!=frames.end(); iter++){
			tracker->update(*iter,roi);
		}
	}
	gettimeofday(&t1, 0);
	elapsed_time = (t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec) / (double)1000000;
	fps = total_frame_num * total_process_num / elapsed_time;
	printf("sequential process: %f sec, fpusec: %f \n", elapsed_time, fps);
	//printf("sequatial process: %f sec, fps: %f \n", elapsed_time, fps);
	*/

	//parallel process
	//start_time = clock();
	pid_t *pids = (pid_t*)malloc(sizeof(pid_t) * total_process_num);
	int status;
	int remained_process_num = total_process_num;
	pid_t pid;

	gettimeofday(&t0, 0);
	for (int i = 0; i < total_process_num; i++)
	{
		if ((pids[i] = fork()) < 0)
		{
			perror("fork");
			abort();
		}
		else if(pids[i] == 0)
		{
			//child
			printf("process %d fork\n", i);
			printf("per process task num: %d\n", total_task_num / total_process_num);
			for(int j = 0; j < (total_task_num / total_process_num); j++)
			{
				for (list<Mat>::iterator iter = frames.begin() ; iter!=frames.end(); iter++){
					tracker->update(*iter,roi);
				}
			}
			exit(0);
		}

	}

	while (remained_process_num > 0)
	{
		pid = wait(&status);
		printf("Child with PID %ld exited with status  0x%x.\n", (long)pid, status);
		--remained_process_num;
	}

	//elapsed_time = (clock() - start_time) / CLOCKS_PER_SEC;
	gettimeofday(&t1, 0);
	elapsed_time = (t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec) / (double)1000000;
	fps = total_frame_num * total_task_num / elapsed_time;
	printf("parallel process: %f sec, fpusec: %f \n", elapsed_time, fps);

	return 0;
}

