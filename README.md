# Elkan-OpenMP

This is a parallelized version of Elkan's algorithm, the program reads in a csv file either labelled or unlabelled and perform K-means clustering on it. The output is a gif file illustrating the clustering iterations. 

 Usage:
-------------------
To run the compiled program, type in 

./Elkan input_file.csv num_threads labelled num_clusters max_iteration should_draw

Parameters:

input_file: input file path.

num_threads: the number of threads used, this also depends on the physical machine.

labelled: is the input labelled or not, 0-1 value. (the label is assumed to be stored in the last column)

num_clusters: the number of initialization clusters.

max_iteration: the maximum number of iterations K-means is allowed to run.

should_draw: should we generate the gif or not, 0-1 value. Turn this off when testing running time. 

