#include <SImage.h>
#include <SImageIO.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <utility>
#include <DrawText.h>

using namespace std;

// The simple image class is called SDoublePlane, with each pixel represented as
// a double (floating point) type. This means that an SDoublePlane can represent
// values outside the range 0-255, and thus can represent squared gradient magnitudes,
// harris corner scores, etc. 
//
// The SImageIO class supports reading and writing PNG files. It will read in
// a color PNG file, convert it to grayscale, and then return it to you in 
// an SDoublePlane. The values in this SDoublePlane will be in the range [0,255].
//
// To write out an image, call write_png_file(). It takes three separate planes,
// one for each primary color (red, green, blue). To write a grayscale image,
// just pass the same SDoublePlane for all 3 planes. In order to get sensible
// results, the values in the SDoublePlane should be in the range [0,255].
//

// Below is a helper functions that overlays rectangles
// on an image plane for visualization purpose. 

// Draws a rectangle on an image plane, using the specified gray level value and line width.
//
void overlay_rectangle(SDoublePlane &input, int _top, int _left, int _bottom, int _right, double graylevel, int width)
{
	for(int w=-width/2; w<=width/2; w++) {
		int top = _top+w, left = _left+w, right=_right+w, bottom=_bottom+w;

		// if any of the coordinates are out-of-bounds, truncate them 
		top = min( max( top, 0 ), input.rows()-1);
		bottom = min( max( bottom, 0 ), input.rows()-1);
		left = min( max( left, 0 ), input.cols()-1);
		right = min( max( right, 0 ), input.cols()-1);

		// draw top and bottom lines
		for(int j=left; j<=right; j++)
			input[top][j] = input[bottom][j] = graylevel;
		// draw left and right lines
		for(int i=top; i<=bottom; i++)
			input[i][left] = input[i][right] = graylevel;
	}
}

// DetectedSymbol class may be helpful!
//  Feel free to modify.
//
typedef enum {NOTEHEAD=0, QUARTERREST=1, EIGHTHREST=2} Type;
class DetectedSymbol {
	public:
		int row, col, width, height;
		Type type;
		char pitch;
		double confidence;
};

// Function that outputs the ascii detection output file
void  write_detection_txt(const string &filename, const vector<struct DetectedSymbol> &symbols)
{
	ofstream ofs(filename.c_str());

	for(int i=0; i<symbols.size(); i++)
	{
		const DetectedSymbol &s = symbols[i];
		ofs << s.row << " " << s.col << " " << s.width << " " << s.height << " ";
		if(s.type == NOTEHEAD)
			ofs << "filled_note " << s.pitch;
		else if(s.type == EIGHTHREST)
			ofs << "eighth_rest _";
		else 
			ofs << "quarter_rest _";
		ofs << " " << s.confidence << endl;
	}
}

void write_image(const string &filename, const SDoublePlane &input)
{
	SDoublePlane output_planes[3];
	for(int i=0; i<3; i++)
	{
		output_planes[i] = input;
	}	
	SImageIO::write_png_file(filename.c_str(), output_planes[0], output_planes[1], output_planes[2]);
}

// Function that outputs a visualization of detected symbols
void  write_detection_image(const string &filename, const vector<DetectedSymbol> &symbols, const SDoublePlane &input)
{
	SDoublePlane output_planes[3];
	for(int i=0; i<3; i++)
		output_planes[i] = input;

	for(int i=0; i<symbols.size(); i++)
	{
		const DetectedSymbol &s = symbols[i];

		overlay_rectangle(output_planes[s.type], s.row, s.col, s.row+s.height-1, s.col+s.width-1, 255, 2);
		overlay_rectangle(output_planes[(s.type+1) % 3], s.row, s.col, s.row+s.height-1, s.col+s.width-1, 0, 2);
		overlay_rectangle(output_planes[(s.type+2) % 3], s.row, s.col, s.row+s.height-1, s.col+s.width-1, 0, 2);

		if(s.type == NOTEHEAD)
		{
			char str[] = {s.pitch, 0};
			draw_text(output_planes[0], str, s.row, s.col+s.width+1, 0, 2);
			draw_text(output_planes[1], str, s.row, s.col+s.width+1, 0, 2);
			draw_text(output_planes[2], str, s.row, s.col+s.width+1, 0, 2);
		}
	}

	SImageIO::write_png_file(filename.c_str(), output_planes[0], output_planes[1], output_planes[2]);
}



// The rest of these functions are incomplete. These are just suggestions to 
// get you started -- feel free to add extra functions, change function
// parameters, etc.
//
//
// Print the value of the image
//
void print_image_value(const SDoublePlane &input)
{
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			cout << input[i][j] << " ";			
		}
		cout << endl;
	}

}

// compare two image values 
// returns true if they are similar, false otherwise
bool compare_image_value(const SDoublePlane &image1, const SDoublePlane &image2)
{
	if (image1.rows() != image2.rows() || image1.cols() != image2.cols())
		return false;
	for (int i = 0; i < image1.rows(); ++i)
	{
		for (int j = 0; j < image1.cols(); ++j)
		{
			if (image1[i][j] != image2[i][j])
				return false;	
		}
	}
	return true;
}

// Convolve an image with a separable convolution kernel
SDoublePlane convolve_separable(const SDoublePlane &input, const SDoublePlane &row_filter, const SDoublePlane &col_filter)
{
	SDoublePlane output(input.rows(), input.cols());
	SDoublePlane temp(input.rows(), input.cols());

	// convolve with row filter
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
				
			double sum = 0.0;
			for (int k = 0; k < row_filter.cols(); ++k)
			{
				int l = j + row_filter.cols()/2 - k;
				// Doing reflection if out of boundary
				if (l < 0) l = -l;
				if (l >= input.cols()) l = 2 * input.cols() - l - 1;
				sum += row_filter[0][k] * input[i][l];
			}
			temp[i][j] = sum;
		}
	}
	
	// convolve with col filter
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			double sum = 0.0;
			for (int k = 0; k < col_filter.rows(); ++k)
			{
				int l = i + col_filter.rows()/2 - k;
				// Doing reflection if out of boundary
				if (l < 0) l = -l;
				if (l >= input.rows()) l = 2 * input.rows() - l - 1;
				sum += col_filter[k][0] * temp[l][j];
			}
			output[i][j] = sum;
		}
	}

	return output;
}

// Convolve an image with a general convolution kernel
//
SDoublePlane convolve_general(const SDoublePlane &input, const SDoublePlane &filter)
{
	SDoublePlane output(input.rows(), input.cols());
	
	// Convolution code here
	int pixel_value=0;
	int frow2,fcol2,frow,fcol;
	
	for(int irow=0;irow<input.rows();irow++){
		for(int icol=0;icol<input.cols();icol++){
			pixel_value=0;
			for(frow=-filter.rows()/2,frow2=0;frow<=filter.rows()/2;frow++,frow2++){
				for(fcol=-filter.cols()/2,fcol2=0;fcol<=filter.cols()/2;fcol++,fcol2++){
					if( (irow+frow) >= 0 && (icol+fcol) >= 0 && (irow+frow) < input.rows() && (icol+fcol) < input.cols()){
						pixel_value=pixel_value+(input[irow+frow][icol+fcol] * filter[frow2][fcol2]); 
					}
				}
			}
		output[irow][icol]=pixel_value;	
		}	
	}
	
	return output;
}

// Clamp all pixel values within 0-255
void clamp_image(const SDoublePlane &input)
{
	double min = input[0][0], max = input[0][0];
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			if (input[i][j] > max)
				max = input[i][j];
			else if (input[i][j] < min)
				min = input[i][j];
		}
	}

	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			input[i][j] = ((input[i][j]-min)/(max-min))*255;
		}
	}
}

// Apply a sobel operator to an image, returns the result
// _gx=true for horizontal gradient, false for vertical gradient
SDoublePlane sobel_gradient_filter(const SDoublePlane &input, bool _gx)
{
	SDoublePlane output(input.rows(), input.cols());
	SDoublePlane row_filter(1, 3), col_filter(3, 1);
	if (_gx)
	{
		row_filter[0][0] = -1.0;
		row_filter[0][1] = 0.0;	
		row_filter[0][2] = 1.0;
			
		col_filter[0][0] = 1.0/8.0;
		col_filter[1][0] = 2.0/8.0;	
		col_filter[2][0] = 1.0/8.0;
	}
	else
	{
		row_filter[0][0] = 1.0/8.0;
		row_filter[0][1] = 2.0/8.0;
		row_filter[0][2] = 1.0/8.0;
			
		col_filter[0][0] = 1.0;
		col_filter[1][0] = 0.0;	
		col_filter[2][0] = -1.0;	
	}	

	SDoublePlane sobel = convolve_separable(input, row_filter, col_filter);	
	return sobel;
}

// Apply an edge detector to an image, returns the binary edge map
// Pass thresh=0 to ignore binary map, else pass thresh [1-255]
SDoublePlane find_edges(const SDoublePlane &input, double thresh=0)
{	
	SDoublePlane G(input.rows(), input.cols());
	SDoublePlane Gx, Gy;

	Gx = sobel_gradient_filter(input, true);
	Gy = sobel_gradient_filter(input, false);
	
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			G[i][j] = sqrt(Gx[i][j]*Gx[i][j]+Gy[i][j]*Gy[i][j]);						
			if (G[i][j] > 255) G[i][j] = 255;			
		}
	}

	if (abs(thresh) > 0.0001)
	{
		for (int i = 0; i < G.rows(); ++i)		
			for (int j = 0; j < G.cols(); ++j)
				G[i][j] = (G[i][j]>thresh?1:0);
	}

	return G;
}

struct compare_priority_queue
{
	bool operator()(const pair<int,double> &lhs, const pair<int,double> &rhs) const
	{
		return rhs.second < lhs.second;
	}
};

SDoublePlane compute_distance_matrix(SDoublePlane &edge_map)
{
	SDoublePlane D(edge_map.rows(), edge_map.cols());	

	// Do a dijkstra in O(nlgn), n=total number of pixel in edge_map	
	priority_queue< pair<int,double>, vector< pair<int,double> >, compare_priority_queue> Q;
	const int n_col = D.cols();
	const int n_row = D.rows();
	for (int i = 0; i < n_row; ++i)		
	{
		for (int j = 0; j < n_col; ++j)
		{			
			if (abs(edge_map[i][j] - 1) < 0.0001)
			{
				D[i][j] = 0;
				Q.push(make_pair(i*n_col+j, 0.0));
			}
			else
				D[i][j] = -1;
		}
	}
	
	while (Q.empty() == false)
	{
		pair<int,double> u = Q.top();		
		int row = u.first / n_col;
		int col = u.first % n_col;
		double w;
		Q.pop();
		for (int i = -1; i <= 1; ++i)
		{
			for (int j = -1; j <= 1; ++j)
			{
				if (row+i<0 || row+i>=n_row || col+j<0 || col+j>=n_col || (i==0 && j==0))
					continue;

				w = (i*j==0?1:1.414);
				if ( abs(D[row+i][col+j]+1) < 0.0001 || D[row+i][col+j] > D[row][col] + w)
				{
					D[row+i][col+j] = D[row][col] + w;
					Q.push( make_pair( (row+i)*n_col+(col+j), D[row+i][col+j] ) );
				}
			}
		}
	}
	
	return D;
}

// Match template using edge detection method
vector<DetectedSymbol> match_template_by_edge(const SDoublePlane &input, const SDoublePlane &template_image,
												double edge_threshold, double score_threshold)
{
	// Compute binary edge map with threshold value
	SDoublePlane edge_map = find_edges(input, edge_threshold);
	SDoublePlane edge_map_template = find_edges(template_image, edge_threshold);

	// Compute D: min distance to an edge pixel for all (i,j) in edge_map
	SDoublePlane D = compute_distance_matrix(edge_map);

	SDoublePlane score(input.rows(), input.cols());

	vector<DetectedSymbol> symbols;

	for (int i = 0; i < input.rows(); ++i)	
		for (int j = input.cols()-template_image.cols()+1; j < input.cols(); ++j)
			score[i][j] = -1;

	for (int i = input.rows()-template_image.rows()+1; i < input.rows(); ++i)	
		for (int j = 0; j < input.cols(); ++j)
			score[i][j] = -1;

	for (int i = 0; i < input.rows()-template_image.rows()+1; ++i)
	{
		for (int j = 0; j < input.cols()-template_image.cols()+1; ++j)
		{			
			for (int k = 0; k < template_image.rows(); ++k)
			{
				for (int l = 0; l < template_image.cols(); ++l)	
				{
					score[i][j] += edge_map_template[k][l] * D[i+k][j+l];
				}
			}

			if (score[i][j] < score_threshold)
			{
				DetectedSymbol s;
				s.row = i;
				s.col = j;
				s.width = 17;
				s.height = 11;
				s.type = (Type) (0);
				s.confidence = rand();
				s.pitch = (rand() % 7) + 'A';
				symbols.push_back(s);
			}
		}
	}

	return symbols;
}

double find_max_vote(const SDoublePlane &acc)
{
	double max=0;
	for(int i=0;i<acc.rows();i++){
		if(acc[i][0] > max)max=acc[i][0];
	}
	return max;
}
double find_min_vote(const SDoublePlane &acc)
{
	double min=acc[0][0];
	for(int i=1;i<acc.rows();i++){
		if(acc[i][0] < min ) min=acc[i][0]; 
	}
	return min;
}
SDoublePlane normalize_votes(const SDoublePlane &acc)
{
	SDoublePlane normalized(acc.rows(),acc.cols());
	double min = find_min_vote(acc);
	double max = find_max_vote(acc);
	for(int i=0;i<acc.rows();i++){
		normalized[i][0] = (acc[i][0] - min)/(max-min);
	}
	return normalized;
}
SDoublePlane hough_transform(const SDoublePlane &edges,double threshold=120.0)
{
        SDoublePlane accumulator(edges.rows(),1);

        for(int i=0;i<edges.rows();i++){
                for(int j=0;j<edges.cols();j++){
                        if(edges[i][j] > threshold){

                                accumulator[i][0]=accumulator[i][0] + 1;

                        }
                }
        }
        print_image_value(normalize_votes(accumulator));
        return accumulator;
}

// Get Hamming distance map
SDoublePlane get_Hamming_distance(const SDoublePlane &input, const SDoublePlane &target)
{
	SDoublePlane output(input.rows(), input.cols());
	
	// change to convolution function later
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			double sum = 0;
			for (int u = 0; u < target.rows(); u++)
			{
				for (int v = 0; v < target.cols(); v++)
				{
					int k = i + u, l = j + v;
					if (k >= input.rows())
					{
						k = input.rows() - 1 - (k - input.rows() + 1);
					}
					if (l >= input.cols())
					{
						l = input.cols() - 1 - (l - input.cols() + 1);
					}
					double a = input[k][l] / 255;
					double b = target[u][v] / 255;
					sum += a * b;
					sum += (1 - a) * (1 - b);
				}
			}
			output[i][j] = sum / (target.rows() * target.cols()) * 255;
		}
	}
	
	return output;
}

double plane_max(const SDoublePlane &input)
{
	double max = 0;
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			if (input[i][j] > max)
			{
				max = input[i][j];
			}
		}
	}
	return max;
}

bool is_max_in_neighbour(const SDoublePlane &input, int y, int x, int w, int h)
{
	
	int hw = w / 2;
	int hh = h / 2;
	
	for (int i = -hh; i <= hh; i++)
	{
		for (int j = -hw; j <= hw; j++)
		{
			int k = y + i;
			int l = x + j;
			if (k < 0 || k >= input.rows())
			{
				continue;
			}
			if (l < 0 || l >= input.cols())
			{
				continue;
			}
			if (k == y && l == x)
			{
				continue;
			}
			if (input[k][l] > input[y][x])
			{
				return false;
			}
			else if (input[k][l] == input[y][x])
			{
				if (i < 0 || j < 0)
				{
					return false;
				}
			}
		}
	}
	return true;
}

SDoublePlane non_maximum_suppress(const SDoublePlane &input, int w, int h)
{
	double threshold = 0.84 * 255;
	SDoublePlane output(input.rows(), input.cols());
	
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			if (input[i][j] > threshold &&
					is_max_in_neighbour(input, i, j, w, h))
			{
				output[i][j] = 255;
			}
			else
			{
				output[i][j] = 0;
			}
		}
	}
	return output;
}

void get_symbols(const SDoublePlane &input, vector<DetectedSymbol> &symbols, Type type, int w, int h)
{
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			if (input[i][j] == 255)
			{
				DetectedSymbol s;
				s.row = i;
				s.col = j;
				s.width = w;
				s.height = h;
				s.type = type;
				s.confidence = 0;
				s.pitch = 'A';
				symbols.push_back(s);
			}
		}
	}
	return;
}

int get_notes_possitions(const SDoublePlane &input, SDoublePlane &pl_note,
		SDoublePlane &pl_quarterrest, SDoublePlane &pl_eighthrest, vector<DetectedSymbol> &symbols)
{
	// non-maximum suppress size
	int sup_w,  sup_h;
	
	//shawn calc hamming distance
	// get template image
	SDoublePlane template_note = SImageIO::read_png_file("template1.png");
	// get distance
	SDoublePlane hammdis_note = get_Hamming_distance(input, template_note);
	write_image("hamming_dist_note.png", hammdis_note);
	// print_image_value(hammdis_note);
	// cout << plane_max(hammdis_note) / 255 << endl;
	// non-maximum suppress
	SDoublePlane sup_note = non_maximum_suppress(hammdis_note, template_note.cols(), template_note.rows());
	write_image("sup_hamming_dist_note.png", sup_note);
	get_symbols(sup_note, symbols, NOTEHEAD, template_note.cols(), template_note.rows());
	
	// quarter_rest
	SDoublePlane template_quarterrest = SImageIO::read_png_file("template2.png");
	// get distance
	SDoublePlane hammdis_quarterrest = get_Hamming_distance(input, template_quarterrest);
	write_image("hamming_dist_quarterrest.png", hammdis_quarterrest);
	// cout << plane_max(hammdis_quarterrest) / 255 << endl;
	// non-maximum suppress
	SDoublePlane sup_quarterrest = non_maximum_suppress(hammdis_quarterrest, template_quarterrest.cols(), template_quarterrest.rows());
	write_image("sup_hamming_dist_quarterrest.png", sup_quarterrest);
	get_symbols(sup_quarterrest, symbols, QUARTERREST, template_quarterrest.cols(), template_quarterrest.rows());
	
	// quarter_rest
	SDoublePlane template_eighthrest = SImageIO::read_png_file("template3.png");
	// get distance
	SDoublePlane hammdis_eighthrest = get_Hamming_distance(input, template_eighthrest);
	write_image("hamming_dist_eighthrest.png", hammdis_eighthrest);
	// cout << plane_max(hammdis_eighthrest) / 255 << endl;
	// non-maximum suppress
	SDoublePlane sup_eighthrest = non_maximum_suppress(hammdis_eighthrest, template_eighthrest.cols(), template_eighthrest.rows());
	write_image("sup_hamming_dist_eighthrest.png", sup_eighthrest);
	get_symbols(sup_eighthrest, symbols, EIGHTHREST, template_eighthrest.cols(), template_eighthrest.rows());

	pl_note = sup_note;
	pl_quarterrest = sup_quarterrest;
	pl_eighthrest = sup_eighthrest;
	
	return 0;
}


//
// This main file just outputs a few test images. You'll want to change it to do 
//  something more interesting!
//
int main(int argc, char *argv[])
{
	if(!(argc == 2))
	{
		cerr << "usage: " << argv[0] << " input_image" << endl;
		return 1;
	}

	string input_filename(argv[1]);
	SDoublePlane input_image= SImageIO::read_png_file(input_filename.c_str());
	//test	
	SDoublePlane acc=hough_transform(find_edges(non_maximum_suppress(input_image,1,1)));
	//test end
	/////////// Step 2 //////////
	/*
	SDoublePlane mean_filter(3,3);
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			mean_filter[i][j] = 1/9.0;
	SDoublePlane output_image = convolve_general(input_image, mean_filter);
	*/

	////////// Step 4 //////////
	/*	
	SDoublePlane pl_note(input_image.rows(), input_image.cols());
	SDoublePlane pl_quarterrest(input_image.rows(), input_image.cols());
	SDoublePlane pl_eighthrest(input_image.rows(), input_image.cols());
		
	vector<DetectedSymbol> symbols;
	get_notes_possitions(input_image, pl_note, pl_quarterrest, pl_eighthrest, symbols);
	// for(int i=0; i<10; i++)
	// {
		// DetectedSymbol s;
		// s.row = rand() % input_image.rows();
		// s.col = rand() % input_image.cols();
		// s.width = 20;
		// s.height = 20;
		// s.type = (Type) (rand() % 3);
		// s.confidence = rand();
		// s.pitch = (rand() % 7) + 'A';
		// symbols.push_back(s);
	// }

	//write_detection_txt("detected.txt", symbols);
	//write_detection_image("detected.png", symbols, input_image);	
	*/

	////////// Step 5 //////////
	
	write_image("edges.png", find_edges(input_image, 30));	
	SDoublePlane template_image= SImageIO::read_png_file("template1.png");	
	vector<DetectedSymbol> symbols = match_template_by_edge(input_image, template_image, 30, 6);	
	write_detection_image("detected.png", symbols, input_image);	
}
