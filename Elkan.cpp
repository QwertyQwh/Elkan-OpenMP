#include <iostream>
#include <fstream>
#include <string>
#include <float.h>
#include <vector>
#include "gif.h"
#include <cmath>
#include <string.h>
#include <omp.h>

using namespace std;
#include <climits>
#define PAGE 0x80

const int width = 1000;
const int height = 1000;

double dist(double* vec1, double* vec2, int dimension) {
	double sum = 0;
	for (int i = 0; i < dimension; i++) {
		sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return sqrt(sum);
}

int inline mat_ind_to_array(int row, int col, int dimension) {
	return row * dimension + col;
}

void recompute_inter_center(double* inter_center, double* centers, int clusters, int dimensions) {
	double tmp = 0;
	for (int i = 0; i < clusters; i++) {
		for (int j = 0; j < i; j++) {
			tmp = dist(centers + i * dimensions, centers + j * dimensions, dimensions);
			inter_center[mat_ind_to_array(i, j, clusters)] = tmp;
			inter_center[mat_ind_to_array(j, i, clusters)] = tmp;
		}
	}
}

void recompute_s(double* s, double* inter_center, int clusters) {
	for (int i = 0; i < clusters; i++) {
		double min = DBL_MAX;
		for (int j = 0; j < clusters ; j++) {
			if (j == i)
				continue;
			double cur = inter_center[mat_ind_to_array(i, j, clusters)];
			if (cur < min) 
				min = cur;
		}
		s[i] = min / 2;
	}
}

void update_step(double* features, double* centers, double* ys , double* zs, int* labels, int clusters, int dimensions, int samples) {
	for (int i = 0; i < clusters; i++) {
		for(int j = 0; j<dimensions ;j++)
			ys[i*dimensions+j] = 0; 
		zs[i] = 0;
	}
	for (int i = 0; i < samples; i++) {
		int label = labels[i];
		for (int j = 0; j < dimensions; j++) 
			ys[mat_ind_to_array(label, j, dimensions)] += features[mat_ind_to_array(i, j, dimensions)];
		zs[label] += 1;
	}
	for (int j = 0; j < clusters; j++) {
		for (int i = 0; i < dimensions; i++) {
			centers[mat_ind_to_array(j, i, dimensions)] = ys[mat_ind_to_array(j, i, dimensions)] / zs[j];
		}
	}
}

void draw_point(vector<uint8_t>& image, double x, double y, int radius, int* color) {
	int xc = (int)(x * width);
	int yc = (int)(y * height);
	int xl = max(xc - radius,0);
	int xh = min(xc+radius,width);
	int yl = max(yc-radius,0);
	int yh = min(yc+radius,height);
	for (unsigned wi = xl; wi < xh; ++wi)
	{
		for (unsigned hi = yl; hi < yh; ++hi)
		{
			
			if ((wi - xc) * (wi - xc) + (hi - yc) * (hi - yc) < radius * radius) {
				image[(hi * width * 4) + (wi * 4) + 0] = color[0];
				image[(hi * width * 4) + (wi * 4) + 1] = color[1];
				image[(hi * width * 4) + (wi * 4) + 2] = color[2];
				image[(hi * width * 4) + (wi * 4) + 3] = color[3];
			}

		}
	}
}

int main(int argc, char *argv[])
{
#pragma region Read Input
	//ReadInput
	std::ifstream data;
	string input = "DelayedFlights.csv";
	int labelled = 0;
	int dimension = 0;
	int samples = 0;
	int clusters = 9;
	int max_iter = 30;
	int threads =16;
	int gifDraw = 0;
	if(argc!=7){
		printf("Incorrect Parameters\n");
		return -1;
	}else{
		input = argv[1];
		sscanf(argv[2], "%d",&threads);
		sscanf(argv[3], "%d",&labelled);
		sscanf(argv[4], "%d",&clusters);
		sscanf(argv[5], "%d",&max_iter);
		sscanf(argv[6], "%d",&gifDraw);

	}
	
	


	data.open(input);
	std::string delimiter = ",";
	string output;
	//First pass, read dimension and samples
	if (data.is_open()) {
		getline(data, output);
		int start = 0;
		start = output.find(delimiter, start);
		dimension++;
		while (start != string::npos) {
			start = output.find(delimiter, start + 1);
			dimension++;
		}
		dimension-=labelled;// subtract the label
		while (data) {
			getline(data, output);
			if (output[0] == ',') {
				break;
			}
			samples++;
		}
	}
	double* features = (double*)calloc(dimension * samples, sizeof(double));
	double* data_max = (double*)calloc(dimension, sizeof(double));
	double* data_min = (double*)calloc(dimension, sizeof(double));
	for (int i = 0; i < dimension; i++) {
		data_max[i] = -INT_MAX;
		data_min[i] = INT_MAX;
	}

	//Second pass, read data
	data.clear();                 // clear fail and eof bits
	data.seekg(0, std::ios::beg); // back to the start!
	int tempIter = 0;
	if (data.is_open()) {
		getline(data, output);
		while (data) {
			getline(data, output);
			int start = 0;
			int prev = -1;
			double num = 0;
			string token;
				for(int i = 0; i<dimension; i++){
					start = output.find(delimiter, prev + 1);
					string token = output.substr(prev+1, start-prev-1);
					if (token.empty()) {
						num = (data_max[i] + data_min[i])/2;
					}else {
						num = stod(token);
					}
					if (num > data_max[i]) 
						data_max[i] = num;
					if (num < data_min[i])
						data_min[i] = num;
					features[tempIter++] = num;
					prev = start;
				}
			}

	}

#pragma endregion

#pragma region Image

	std::vector<uint8_t> canvas(width * height * 4, 255);
	int delay = 100; 
	int point_radius = 10;
	auto fileName = "kmeans.gif";
	GifWriter g;
	GifBegin(&g, fileName, width, height, delay);
	int red[4] = { 255,0,0,1 };
	int dim_visualize_x = 0;
	int dim_visualize_y = 1;

#pragma endregion

	//TODO: USE OPENMP
#pragma region Loop
	// int clusters = 128;
	int* colors = (int*)calloc(4*clusters, sizeof(int));
	srand(time(NULL));
	for (int j = 0; j < clusters; j++) {
		colors[j *4 + 0] = (double)rand() / RAND_MAX *200+20;
		colors[j *4 + 1] = (double)rand() / RAND_MAX * 200+20;
		colors[j *4 + 2] = (double)rand() / RAND_MAX * 200+20;
		colors[j *4 + 3] = 1;
	}
	int* labels = (int*)calloc(samples, sizeof(int));
	// c1.x, c1.y || c2.x, c2.y
	double* centers = (double*)calloc(dimension * clusters, sizeof(double));
	double* centers_prev = (double*)calloc(dimension*clusters, sizeof(double));
	for (int i = 0; i < dimension; i++) {
		srand(time(NULL));
		for (int j = 0; j < clusters; j++) {
			centers[j * dimension + i] = (double)rand()/RAND_MAX * (data_max[i] - data_min[i])/2 +data_min[i]+ (data_max[i] - data_min[i]) / 4;
		}
	}
	double* ys = (double*)calloc(dimension * clusters, sizeof(double));
	double* zs = (double*)calloc(clusters, sizeof(double));

	double* uppers = (double*)calloc(samples, sizeof(double));
	for (int i = 0; i < samples; i++) 
		uppers[i] = INT_MAX;
	//samples * cluster
	double* lowers = (double*)calloc(samples * clusters, sizeof(double));
	//The inter-center dist matrix cluster*cluster
	double* inter_center_dist = (double*)calloc(clusters*clusters, sizeof(double));
	double* s = (double*)calloc(clusters, sizeof(double));
	double* delta = (double*)calloc(clusters, sizeof(double));
	double r = true;
	double z;
	double tmp;
	bool changed = false;
	
	
	double loop_time = 0;
	omp_set_num_threads(threads);
	int chunk_size = max(samples / threads,PAGE);
	double tstart = omp_get_wtime();
	int iter = 0;
		for (; iter < max_iter; iter++) {
				changed = false;
				recompute_inter_center(inter_center_dist, centers, clusters, dimension);
				recompute_s(s, inter_center_dist, clusters);
#pragma omp parallel for schedule(static,chunk_size)  private(z,r,tmp)
					for (int i = 0; i < samples ; i++) {
					// int tn = omp_get_thread_num();
					// 	printf("%i is working on iteration %i\n", tn, i);
						if (uppers[i] > s[labels[i]]) {
							r = true;
							for (int k = 0; k < clusters; k++) {
								z = max(lowers[mat_ind_to_array(i, k, clusters)], inter_center_dist[mat_ind_to_array(labels[i], k, clusters)] / 2);
								if (k == labels[i] || uppers[i] <= z)
									continue;
								if (r) {
									uppers[i] = dist(features + i * dimension, centers + labels[i] * dimension, dimension);
									r = false;
									if (uppers[i] <= z)
										continue;
								}
								tmp = dist(features + i * dimension, centers + k * dimension, dimension);
								lowers[mat_ind_to_array(i, k, clusters)] = tmp;
								if (tmp < uppers[i]) {
									labels[i] = k;
									if(gifDraw==1){
										changed = true;
									}
									
									uppers[i] = tmp;
								}
							}
						}
					}
				memcpy(centers_prev, centers, dimension* clusters * sizeof(double));
				update_step(features, centers, ys, zs, labels, clusters, dimension, samples);

			for (int i = 0; i < clusters; i++) {
				delta[i] = dist(centers_prev + i * dimension, centers + i * dimension, dimension);
			}
#pragma omp parallel for  schedule(static,chunk_size)
			for (int i = 0; i < samples; i++) {
				uppers[i] += delta[labels[i]];
				for (int k = 0; k < clusters; k++) {
					lowers[mat_ind_to_array(i, k, clusters)] -= delta[k];
				}
			}
			if(gifDraw==1){
				for (int i = 0; i < samples; i++) {
					draw_point(canvas, (features[dimension * i + dim_visualize_x] - data_min[dim_visualize_x]) / (data_max[dim_visualize_x] - data_min[dim_visualize_x]), (features[dimension * i + dim_visualize_y] - data_min[dim_visualize_y]) / (data_max[dim_visualize_y] - data_min[dim_visualize_y]), point_radius, colors + labels[i] * 4);
				}
				GifWriteFrame(&g, canvas.data(), width, height, delay);
				for (int i = 0; i < canvas.size(); i++) {
					canvas[i] = 255;
				}
				if (!changed) {
					break;
				}

			}
			//printf("Time taken for the loop: % f\n", omp_get_wtime()-beforeloop);
	}
	double ttaken = omp_get_wtime() - tstart;
	printf("Time taken for %i iterations : % f\n", iter+1,ttaken);
	free(features);
	free(data_max);
	free(data_min);
	free(labels);
	free(centers);
	free(centers_prev);
	free(uppers);
	free(lowers);
	free(inter_center_dist);
	free(s);
	free(ys);
	free(zs);
	free(delta);
#pragma endregion

	GifEnd(&g);
	return 1;
}
