from colorsys import hsv_to_rgb
def gencolorarray(numcolors):
   # ensure numcolors is an integer by using exception 
   color_list = []
   try: 
     for i in xrange(1, numcolors+1):
       p_color = float(i)/numcolors 
       color_val = hsv_to_rgb(p_color,1,1)
       color_list.append(color_val)
   except: 
     print "numcolors must be an integer\n"
   
   return color_list
