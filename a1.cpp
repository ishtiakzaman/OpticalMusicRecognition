#include <SImage.h>
#include <SImageIO.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
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
//
SDoublePlane convolve_separable(const SDoublePlane &input, const SDoublePlane &row_filter, const SDoublePlane &col_filter)
{
	SDoublePlane output(input.rows(), input.cols());

	// Convolution code here

	return output;
}

// Convolve an image with a separable convolution kernel
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


// Apply a sobel operator to an image, returns the result
// 
SDoublePlane sobel_gradient_filter(const SDoublePlane &input, bool _gx)
{
	SDoublePlane output(input.rows(), input.cols());

	// Implement a sobel gradient estimation filter with 1-d filters


	return output;
}

// Apply an edge detector to an image, returns the binary edge map
// 
SDoublePlane find_edges(const SDoublePlane &input, double thresh=0)
{
	SDoublePlane output(input.rows(), input.cols());

	// Implement an edge detector of your choice, e.g.
	// use your sobel gradient operator to compute the gradient magnitude and threshold

	return output;
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

	// To print the values of an image use the following call	
	//print_image_value(input_image);
	//cout <<  compare_image_value(input_image, input_image) << endl;
	// test step 2 by applying mean filters to the input image
	SDoublePlane mean_filter(3,3);
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			mean_filter[i][j] = 1/9.0;
	SDoublePlane output_image = convolve_general(input_image, mean_filter);

	//
	SDoublePlane pl_note(input_image.rows(), input_image.cols());
	SDoublePlane pl_quarterrest(input_image.rows(), input_image.cols());
	SDoublePlane pl_eighthrest(input_image.rows(), input_image.cols());
	

	// randomly generate some detected symbols -- you'll want to replace this
	//  with your symbol detection code obviously!
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
}
