
#ifndef MATHUTILITY
#define MATHUTILITY

#include <vector>
#include <algorithm> // sort

using namespace std;

class MathUtility {

   public:
   // vector
   template <typename T>
   static T v_median(vector<T> vec);
   template <typename T>
   static vector<T> v_add(vector<T> a,vector<T> b);
   template <typename T>
   static vector<T> v_subtr(vector<T> a,vector<T> b);
   template <typename T>
   static vector<T> v_dot(vector<T> a,vector<T> b);
   
   // matrix
   template <typename T>
   static vector< vector<T> > m_zeros(unsigned int n1, unsigned int n2);
   template <typename T>
   static vector< vector<T> > m_mult(vector< vector<T> > a,vector< vector<T> > b);
  
   // matrix vector
   template <typename T>
   static vector<T> mv_mult(vector< vector<T> > a,vector<T> b);
  
};

/*--------------------*/
/* Template functions */
/*--------------------*/

/*
 * Add two vectors.
 */
template < typename T >
vector<T> MathUtility::v_add(vector<T> a,vector<T> b){
	
	if(a.size() != b.size())
		throw("Can not subtract to vectors with unequal sizes.");
	
	for(unsigned int ii=0; ii<a.size(); ++ii){
		a[ii] += b[ii];
	}

	return a;
}

/*
 * Subtract two vectors.
 */
template < typename T >
vector<T> MathUtility::v_subtr(vector<T> a,vector<T> b){
	
	if(a.size() != b.size())
		throw("Can not subtract to vectors with unequal sizes.");
	
	for(unsigned int ii=0; ii<a.size(); ++ii){
		a[ii] -= b[ii];
	}

	return a;
}

/*
 * Median of a vector.
 */
template < typename T >
T MathUtility::v_median(vector<T> vec){
	
	if(vec.empty())
		return 0.;
	
	sort(vec.begin(), vec.end());
	
	if(vec.size() % 2 == 0){
		return (vec[vec.size()/2 - 1] + vec[vec.size()/2]) / 2;
    } else {
		return vec[vec.size()/2];
	}
}

/*
 * Dot product of two vectors.
 */
template < typename T >
vector<T> MathUtility::v_dot(vector<T> a,vector<T> b){
	
	if(a.size() != b.size())
		throw("Can not dot multiply vectors with unequal sizes.");
	
	for(unsigned int ii=0; ii<a.size(); ++ii){
		a[ii] *= b[ii];
	}

	return a;
}


/*
 * Initialize a matrix with zeros.
 */
template < typename T >
vector< vector<T> > MathUtility::m_zeros(unsigned int n1,unsigned int n2){
	
	if(n1 == 0)
		throw("matrix_zeros dimension 1 must be at least 1.");
	if(n2 == 0)
		throw("matrix_zeros dimension 2 must be at least 1.");
	
	vector< vector<T> > matrix(n1);
	for(unsigned int ii=0; ii<n1; ii++){
		vector<T> matrix2(n2);
		matrix[ii] = matrix2;
		//matrix[ii].reserve(n2);
	}
	
	return matrix;
	
}


/*
 * Product of two matrices.
 */
template < typename T >
vector< vector<T> > MathUtility::m_mult(vector< vector<T> > a,vector< vector<T> > b){
	
	if((a.size() < 1) | (b.size() < 1))
		throw("Can not dot multiply empty containers.");
	
	vector< vector<T> > res = MathUtility::m_zeros(a.size(),b[0].size());
	
	for(unsigned int ii=0; ii<res.size(); ++ii){
		if(a[ii].size() != b.size())
			throw("Matrix a and b do not meet dimensions.");
		for(unsigned int jj=0; jj<res[ii].size(); ++jj){
			for(unsigned int kk=0; kk < b.size(); ++kk)
				res[ii][jj] += a[ii][kk]*b[kk][jj];
		}
	}

	return res;
}


/*
 * Product of a matrix with a vector.
 */
template < typename T >
vector<T> MathUtility::mv_mult(vector< vector<T> > a,vector<T> b){
	
	if((a.size() < 1) | (b.size() < 1))
		throw("Can not dot multiply empty containers.");
	
	vector<T> res(b.size());
	
	for(unsigned int ii=0; ii<b.size(); ++ii){	
		if(a[ii].size() != b.size())
			throw("Matrix a and vector b do not meet dimensions.");
		for(unsigned int jj=0; jj<b.size(); ++jj){
			res[ii] += a[ii][jj]*b[jj];
		}
	}
	
	return res;
}

#endif
