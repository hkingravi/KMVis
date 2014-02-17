# returns evaluations at a user-defined function
# data assumed to be in column form 
from numpy import zeros, power, dot 
def poly_func( data, weight_vec, degree ):
   # need matrix for feature map
   feature_map = zeros((degree+1,data.shape[0]));
  
   for i in xrange(0,degree):
     feature_map[i,:] = power(data,i)

   feature_map = feature_map.T
   f_vals = dot(feature_map,weight_vec)

   return f_vals, feature_map
