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
#include <set>

using namespace std;

#define PI 3.14159

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

double image_max(const SDoublePlane &input)
{
	double max=0;
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			if (input[i][j] > max)
			{
				max = input[i][j];
			}
		}
	}
	return max;
}

double image_min(const SDoublePlane &input)
{
	double min=0;
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			if (input[i][j] < min)
			{
				min = input[i][j];
			}
		}
	}
	return min;
}

SDoublePlane normalize_image(const SDoublePlane &input)
{
	SDoublePlane output(input);
	double max = image_max(output);
	double min = image_min(output);
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			output[i][j] = (output[i][j] - min) / (max - min) * 255;
		}
	}
	return output;
}

SDoublePlane complement_image(const SDoublePlane &input)
{
	SDoublePlane output(input);
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			output[i][j] = 255 - output[i][j];
		}
	}
	return output;
}

SDoublePlane scale_image(const SDoublePlane &input, double ratio)
{
	int m = input.rows();
	int n = input.cols();
	int m2 = input.rows()*ratio;
	int n2 = input.cols()*ratio;
	
	SDoublePlane output(m2, n2);
	
	if (ratio > 0.5)
	{
		for (int i = 0; i < m2; i++)
		{
			int sk = i/ratio;
			int ek = (i + 1)/ratio - 0.00001;
			for (int j = 0; j < n2; j++)
			{
				int sl = j/ratio;
				int el = (j + 1)/ratio - 0.00001;
				
				output[i][j] = input[sk][sl];
				output[i][j] += input[sk][el];
				output[i][j] += input[ek][sl];
				output[i][j] += input[ek][el];
				output[i][j] /= 4.0;
			}
		}
	}
	else
	{
		int span = 1.0/ratio + 0.5;
		for (int i = 0; i < m2; i++)
		{
			int sk = i/ratio;
			int ek = (i - 1)/ratio - 0.00001;
			for (int j = 0; j < n2; j++)
			{
				int sl = j/ratio;
				int el = (j - 1)/ratio - 0.00001;
				
				output[i][j] = 0;
				for (int u = sk; u <= ek; u++)
				{
					for (int v = sl; v <= el; v++)
					{
						output[i][j] += input[u][v];
					}
				}
				output[i][j] /= (ek - sk + 1)*(el - sl + 1);
			}
		}
	}
	
	
	return output;
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
void print_image_value1(const SDoublePlane &input)
{	int sum=0;
	for (int i = 0; i < input.cols(); ++i)
	{	sum=0;
		for (int j = 0; j < input.rows(); ++j)
		{	sum=sum+input[j][i];
			//cout << input[i][j] << " ";			
		}
		//cout << endl;
		cout<<sum<<"|||"<<i<<endl;
	}

}

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
// white_value only applies when thresh!=0, pass 1 for 0-1 image, or 255 for 0-255 binary image
// Returns edge_map and gradient_angle
pair<SDoublePlane, SDoublePlane> find_edges(const SDoublePlane &input, double thresh=0, double white_value=1)
{	
	SDoublePlane G(input.rows(), input.cols());
	SDoublePlane Rotation(input.rows(), input.cols());
	SDoublePlane Gx, Gy;

	Gx = sobel_gradient_filter(input, true);
	Gy = sobel_gradient_filter(input, false);
	
	for (int i = 0; i < input.rows(); ++i)
	{
		for (int j = 0; j < input.cols(); ++j)
		{
			G[i][j] = sqrt(Gx[i][j]*Gx[i][j]+Gy[i][j]*Gy[i][j]);						
			if (G[i][j] > 255) G[i][j] = 255;			
			if ( abs(Gx[i][j]) < 0.0001)
				Rotation[i][j] = PI / 2.0;
			else
				Rotation[i][j] = atan(Gy[i][j] / Gx[i][j]);			
		}		
	}

	if (abs(thresh) > 0.0001)
	{
		for (int i = 0; i < G.rows(); ++i)		
			for (int j = 0; j < G.cols(); ++j)
				G[i][j] = (G[i][j]>thresh?white_value:0);
	}
	return make_pair(G, Rotation);
}

SDoublePlane create_gaussian_filter(int size, double sigma)
{
	SDoublePlane filter(size, size);
	if (size % 2 == 0)
	{
		printf("Gaussian filter size must be odd.\n");
		return filter;
	}
	
	for(int i = -size/2; i <= size/2; i++)
	{
		for(int j = -size/2; j <= size/2; j++)			
		{
			filter[i+size/2][j+size/2] = 1.0/(2.0*PI*sigma*sigma)*exp(-1.0*(i*i+j*j)/(2*sigma*sigma));
		}	
	}
	return filter;
}

SDoublePlane edge_thinning_non_maximum_suppress(const pair<SDoublePlane, SDoublePlane> &edge, const double threshold,
														const double range, double white_value=1, bool double_pass=true)
{
	SDoublePlane gradient_value = edge.first;
	SDoublePlane gradient_angle = edge.second;

	SDoublePlane edge_map(gradient_value.rows(), gradient_value.cols());

	for (int i = 0; i < gradient_value.rows(); ++i)
	{
		for (int j = 0; j < gradient_value.cols(); ++j)
		{
			
			if (gradient_value[i][j] < threshold)
			{
				edge_map[i][j] = 0;
				continue;
			}

			bool is_local_maxima = true;						
			for (double d = -range/2.0; d <= range/2.0; ++d)
			{
				if ( abs(d) < 0.0001)
					continue;
				
				int r = round( i - d * sin(gradient_angle[i][j]) );
				int c = round( j + d * cos(gradient_angle[i][j]) );

				if (r < 0 || c < 0 || r >= gradient_value.rows() || c >= gradient_value.cols())
					break;				

				if (gradient_value[r][c] > gradient_value[i][j] + 0.0001)
				{
					is_local_maxima = false;
					break;
				}			
			}
			

			edge_map[i][j] = is_local_maxima?white_value:0;

		}
	}

	if (double_pass == true)
	{
		for (int i = 0; i < edge_map.rows(); ++i)
		{
			for (int j = 0; j < edge_map.cols(); ++j)
			{
				if (abs(edge_map[i][j]) < 0.001)
				{				
					continue;
				}
				double avgr = i, avgc = j, cnt = 1;
				edge_map[i][j] = 0;
				for (double d = -range/2.0; d <= range/2.0; ++d)
				{
					
					int r = round( i - d * sin(gradient_angle[i][j]) );
					int c = round( j + d * cos(gradient_angle[i][j]) );

					if (r < 0 || c < 0 || r >= edge_map.rows() || c >= edge_map.cols() || (r == i && j == c))
						continue;				

					if ( edge_map[r][c] > 0.0001 )
					{
						// If there gradient angle is different, skip
						if ( abs(gradient_angle[i][j]-gradient_angle[r][c]) > 0.1) 
							break;
						edge_map[r][c] = 0;
						avgr += r;
						avgc += c;
						cnt++;				
					}			
				}

				edge_map[int(avgr/cnt)][int(avgc/cnt)] = white_value;

			}
		}
	}

	return edge_map;	
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
vector<DetectedSymbol> match_template_by_edge(const SDoublePlane &input, const vector<SDoublePlane> &template_image,
												double edge_threshold, vector<double> &template_threshold)
{
	// Compute binary edge map with threshold value
	SDoublePlane gaussian = create_gaussian_filter(5, 1);		
	SDoublePlane edge_map = edge_thinning_non_maximum_suppress(find_edges(input), edge_threshold, 7, 1, true);	
	write_image("edges.png", edge_thinning_non_maximum_suppress(find_edges(input), edge_threshold, 7, 255, true));
	// Compute D: min distance to an edge pixel for all (i,j) in edge_map
	SDoublePlane D = compute_distance_matrix(edge_map);

	SDoublePlane score(input.rows(), input.cols());

	vector<DetectedSymbol> symbols;

	for (int template_type = 0; template_type < template_image.size(); ++template_type)	
	{		
		SDoublePlane edge_map_template = edge_thinning_non_maximum_suppress(find_edges(template_image[template_type]), edge_threshold, 7, 1, true);
		
		SDoublePlane D_template = compute_distance_matrix(edge_map_template);

		for (int i = 0; i < input.rows()-template_image[template_type].rows()+1; ++i)
		{
			for (int j = 0; j < input.cols()-template_image[template_type].cols()+1; ++j)
			{			
				score[i][j] = 0;

				for (int k = 0; k < template_image[template_type].rows(); ++k)
				{
					for (int l = 0; l < template_image[template_type].cols(); ++l)	
					{
						if (edge_map_template[k][l] > 0.001)
							score[i][j] += edge_map_template[k][l] * D[i+k][j+l];
						else
							score[i][j] += abs( D[i+k][j+l] - D_template[k][l]);
					}
				}

				if (score[i][j] < template_threshold[template_type])
				{
					bool skip_match = false;
					for (vector<DetectedSymbol>::iterator it = symbols.begin(); it != symbols.end(); ++it)
					{
						// Skip nearby matched to avoid double detection
						if ( abs(it->row - i) < it->height && abs(it->col - j) < it->width)
						{
							skip_match = true;
							break;
						}
					}
					if (skip_match == true)
						continue;

					DetectedSymbol s;
					s.row = i;
					s.col = j;
					s.width = template_image[template_type].cols();
					s.height = template_image[template_type].rows();
					s.type = (Type) (template_type);					
					s.confidence = rand();
					s.pitch = (rand() % 7) + 'A';
					symbols.push_back(s);

					// Skip the next 5 column to avoid multiple selection					
					j += 4;
				}
			}
		}	
	}

	return symbols;
}
//find max votes from accumulator
double find_max_vote(const SDoublePlane &acc)
{
	double max=0;
	for(int i=0;i<acc.rows();i++){
		if(acc[i][0] > max)max=acc[i][0];
	}
	return max;
}

//find min votes from accumulator

double find_min_vote(const SDoublePlane &acc)
{
	double min=acc[0][0];
	for(int i=1;i<acc.rows();i++){
		if(acc[i][0] < min ) min=acc[i][0]; 
	}
	return min;
}
//Do Min Max Normalization on the votes of accumulator

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

//for every first line of the staff set the other 4 lines of staff using the best spacing found

SDoublePlane set_staff(const SDoublePlane &row_votes,int best_space,int intercept_value,int staff_number)
{
        set<int> row_nums;
        int last_row=0;

        for(int i=0;i<5;i++){
                row_nums.insert(intercept_value + (i*best_space) );
                last_row=intercept_value+(i*best_space);
        }

        for(int j=staff_number;j<=last_row;j++){
			if(j <row_votes.rows()){
            	if(row_nums.count(j) == 1){
                    	row_votes[j][best_space]=255;
            	}
            	else{
                    	row_votes[j][best_space]=0;
            	}		
    		}		
		}

        return row_votes;
}

//using the normalized votes find the best row co-ordinates for staff lines
SDoublePlane find_best_line_intercepts(const SDoublePlane &row_votes,const SDoublePlane &normed_votes,int best_space,double norm_threshold=0.55,int neighbour_threshold=8,int start=0)
{	
	SDoublePlane row_spacing=row_votes;
	if(start < row_votes.rows()){
        SDoublePlane staff_lines(row_votes.rows(),1);
	int i=0;	
	double best_value=0;
        int intercept_value=0;
        while(i<row_votes.rows()){
                if(normed_votes[i][0] > norm_threshold){
                        best_value=normed_votes[i][0];
                        intercept_value=i;
			for(int j=1;j<neighbour_threshold;j++){
                                if(normed_votes[i+j][0] > best_value ){
                                        best_value=normed_votes[i+j][0];
                                        intercept_value=i+j;
                                }
                        }
		row_spacing=set_staff(row_spacing,best_space,intercept_value,start);
                i=intercept_value+(4*(best_space))+neighbour_threshold;
		start=intercept_value+(4*best_space)+neighbour_threshold;

                }
		
     		i++;
        }
	
        }
	return row_spacing;
}

//from the row co-ordinates/best space matrix find the space parameter with high votes
int find_best_spacing(const SDoublePlane &row_spacing)
{
	long max=0,sum=0;
	int best_space=0;
	for(int i=2;i<row_spacing.cols();i++){
		sum=0;
		for(int j=0;j<row_spacing.rows();j++){
			sum=sum+row_spacing[j][i];
		}
		if(sum > max){
			max=sum;
			best_space=i;
		}
	}
	return best_space;
}

bool is_max_in_neighbour_for_hough(const SDoublePlane &input, int y, int x, int w, int h)
{

        int hw = w / 2;
        int hh = h / 2;
        
	for (int i = -hh; i <= hh; i++)
        {       int k = y + i;

                for (int j = -hw; j <= hw; j++)
                {

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

                        if (input[k][l] < input[y][x] )
                        {
                                 return false;
                        }

                }
        }

        return true;
}


SDoublePlane non_maximum_suppress_for_hough(const SDoublePlane &input, int w, int h)
{
        double threshold = 170;
        SDoublePlane output(input.rows(), input.cols());

        for (int i = 0; i < input.rows(); i++)
        {
                for (int j = 0; j < input.cols(); j++)
                {
                        if (input[i][j] < threshold &&
                                        is_max_in_neighbour_for_hough(input, i, j, w, h))
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
//return a pair of 1-d SDoublePlane with 255 set for staff lines and integer representing distance between the staff lines 
//max_suppress = 1 to do a line thinning before hough, 0 otherwise
pair<SDoublePlane,int> hough_transform(const SDoublePlane &input,int max_suppress,double threshold=0.001,int neighborhood=8,double norm_threshold=0.55)
{	
	SDoublePlane edges(input.rows(),input.cols());		
	if(max_suppress = 1){
		 edges=non_maximum_suppress_for_hough(input,0,2);
	}
	else{
		 edges=input;
	}
	
	SDoublePlane accumulator(edges.rows(),1);
    	
	SDoublePlane row_spacing(edges.rows(),edges.rows());
	
	 for(int i=0;i<edges.rows();i++){
                for(int j=0;j<edges.cols();j++){
			
                        if(edges[i][j] > threshold ){

                        	accumulator[i][0]=accumulator[i][0] + 1;
                                
                        }
                }
        }
	
	
	SDoublePlane normed_votes=normalize_votes(accumulator);
	int cur=0,count=0;
        for(int i=0;i<edges.rows();i++){
                for(int j=0;j<edges.cols();j++){
                        if(edges[i][j] > threshold && normed_votes[i][0] > norm_threshold){
				count=0;
      				for(int z=i-1;z>0;z--){
					if(normed_votes[z][0]>=norm_threshold &&  abs(i-z) > neighborhood && z!=i && count<1){cur=i;count++;row_spacing[z][abs(i-z)]++;}
				}
                        }
                }
        }
	
	int best_space=find_best_spacing(row_spacing);
	
	SDoublePlane best_row_intercepts= find_best_line_intercepts(row_spacing,normed_votes,best_space);
	for(int i=0;i<best_row_intercepts.rows();i++){
	accumulator[i][0]=best_row_intercepts[i][best_space];
	}
	return make_pair(accumulator,best_space);
}

//draw lines on the image after hough transform
SDoublePlane get_lines(const SDoublePlane &acc,const SDoublePlane &input)
{
	SDoublePlane lines(input.rows(),input.cols());

	for(int i=0;i<acc.rows();i++){
		if(acc[i][0] == 255){
			for(int j=0;j<input.cols();j++){
				lines[i][j]=255;
			}
		}
	}
	return lines;
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

SDoublePlane non_maximum_suppress(const SDoublePlane &input, double threshold, int w, int h)
{
	//double threshold = 0.84 * 255;
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
		for (int j = 40; j < input.cols(); j++)
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

int get_notes_possitions(const SDoublePlane &input, const SDoublePlane &tmpl,
		double threshold, SDoublePlane &output, Type t,
		vector<DetectedSymbol> &symbols)
{
	// non-maximum suppress size
	int sup_w,  sup_h;
	
	//shawn calc hamming distance
	
	// get template image
	//SDoublePlane template_note = SImageIO::read_png_file("template1.png");
	// get distance
	SDoublePlane hammdis_note = get_Hamming_distance(input, tmpl);
	//write_image("hamming_dist_note.png", hammdis_note);
	// print_image_value(hammdis_note);
	// cout << plane_max(hammdis_note) / 255 << endl;
	// non-maximum suppress
	SDoublePlane sup_note = non_maximum_suppress(hammdis_note, threshold*255, tmpl.cols()*0.5, tmpl.rows()-(int)(tmpl.rows()*0.4));
	//write_image("sup_hamming_dist_note.png", sup_note);
	get_symbols(sup_note, symbols, t, tmpl.cols(), tmpl.rows());

	
	return 0;
}

int get_notes_pitch(vector<DetectedSymbol> &symbols, const SDoublePlane &lines, int interval)
{
	vector<int> line_pos;
	for (int i = 0; i < lines.rows(); i++)
	{
		if (lines[i][0] == 255)
		{
			line_pos.push_back(i);
		}
	}
	
	int last = 0;
	vector<int> groups;
	for(vector<int>::iterator iter = line_pos.begin(); iter < line_pos.end(); iter++)
	{
		int y = *iter;
		if (y - last > interval * 2)
		{
			groups.push_back(y);
		}
		last = y;
	}
	
	int ginterval = groups[1] - groups[0];
	int upper_bound = -4 * interval;
	int lower_bound = ginterval - 4 * interval;
	
	for(vector<DetectedSymbol>::iterator symiter = symbols.begin(); symiter < symbols.end(); symiter++)
	{
		if (symiter->type != NOTEHEAD)
		{
			continue;
		}
		for(vector<int>::iterator giter = groups.begin(); giter < groups.end(); giter++)
		{
			int gy = *giter;
			int dy = symiter->row - 0.1*interval - gy;
			//if (dy < upper_bound)
			//{
			//	// missing some group
			//	break;
			//}
			if(dy > lower_bound)
			{
				// belong to next group
				continue;
			}
			int np = 4 - dy * 2 / interval + 28;
			if ((giter - groups.begin()) % 2 != 0)
			{
				np += 2;
			}
			np %= 7;
			symiter->pitch = 'A' + np;
			break;
		}
	}
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
	//SDoublePlane acc=hough_transform(find_edges(input_image).first);
	//SDoublePlane lines=get_lines(acc,input_image);
	//SImageIO::write_png_file("lines1.png",input_image,lines,lines);
	pair<SDoublePlane,int> intercept_space = hough_transform(input_image,1);
	SDoublePlane lines = get_lines(intercept_space.first, input_image);
	SImageIO::write_png_file("lines1.png", input_image, lines, lines);

	//
	//testend
	/////////// Step 2 //////////
	
	// scale temple
	SDoublePlane tmpl_note = SImageIO::read_png_file("template1.png");
	SDoublePlane tmpl_quarterrest = SImageIO::read_png_file("template2.png");
	SDoublePlane tmpl_eighthrest = SImageIO::read_png_file("template3.png");
	
	double scale_ratio = (double)(intercept_space.second+1) / tmpl_note.rows();

	cout << "scale_ratio:" << scale_ratio << endl;
	
	if (scale_ratio < 2.0)
	{
		tmpl_note = scale_image(tmpl_note, scale_ratio);
		tmpl_quarterrest = scale_image(tmpl_quarterrest, scale_ratio);
		tmpl_eighthrest = scale_image(tmpl_eighthrest, scale_ratio);
	}
	else if(scale_ratio >= 1.5 && scale_ratio <= 5.0)
	{
		scale_ratio = 1/scale_ratio;
		input_image = scale_image(input_image, scale_ratio);
	}
	
	write_image("rs_music.png", input_image);
	write_image("rs_tmpl_1.png", tmpl_note);
	write_image("rs_tmpl_2.png", tmpl_quarterrest);
	write_image("rs_tmpl_3.png", tmpl_eighthrest);

	////////// Step 4 //////////
	
	SDoublePlane pl_note(input_image.rows(), input_image.cols());
	SDoublePlane pl_quarterrest(input_image.rows(), input_image.cols());
	SDoublePlane pl_eighthrest(input_image.rows(), input_image.cols());
	
	
	vector<DetectedSymbol> symbols_hamming;
	//get_notes_possitions(input_image, pl_note, pl_quarterrest, pl_eighthrest, symbols_hamming);
	get_notes_possitions(input_image, tmpl_note, 0.78, pl_note, NOTEHEAD, symbols_hamming);
	get_notes_possitions(input_image, tmpl_quarterrest, 0.76, pl_quarterrest, QUARTERREST, symbols_hamming);
	get_notes_possitions(input_image, tmpl_eighthrest, 0.78, pl_eighthrest, EIGHTHREST, symbols_hamming);
	
	get_notes_pitch(symbols_hamming, intercept_space.first, intercept_space.second - 1);
	
	write_detection_image("detected_hamming.png", symbols_hamming, input_image);
	
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
	
	
	////////// Step 5 //////////

	/*
	SDoublePlane tmpl_note = SImageIO::read_png_file("template1.png");
	SDoublePlane tmpl_quarterrest = SImageIO::read_png_file("template2.png");
	SDoublePlane tmpl_eighthrest = SImageIO::read_png_file("template3.png");
	*/

	SDoublePlane gaussian = create_gaussian_filter(5, 1);	
	write_image("edges.png", edge_thinning_non_maximum_suppress(find_edges(convolve_general(input_image, gaussian)), 11, 7, 255));	
	
	vector<SDoublePlane> template_image;	
	template_image.push_back(tmpl_note);
	template_image.push_back(tmpl_quarterrest);
	template_image.push_back(tmpl_eighthrest);
	vector<double> template_threshold;
	
	template_threshold.push_back(160.0*scale_ratio*scale_ratio);
	template_threshold.push_back(450.0*scale_ratio*scale_ratio);
	template_threshold.push_back(395.0*scale_ratio*scale_ratio);
	
	/*
	template_threshold.push_back(89.0*scale_ratio*scale_ratio);
	template_threshold.push_back(250.0*scale_ratio*scale_ratio);
	template_threshold.push_back(195.0*scale_ratio*scale_ratio);
	*/
	vector<DetectedSymbol> symbols = match_template_by_edge(input_image, template_image, 11, template_threshold);
	get_notes_pitch(symbols, intercept_space.first, intercept_space.second - 1);
	write_detection_image("detected.png", symbols, input_image);	
}
