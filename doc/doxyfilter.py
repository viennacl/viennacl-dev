#!/usr/bin/python

#
# Filter for setting up a nice example pages in doxygen.
# Quick instructions on setting examples up properly:
#  - All text blocks need to be inside '/**' and '**/'.
#  - The first block MUST contain '\example' or '@example'
#    in the same line as the opening '/**'
#
# ------- Start of example input ------------
#
# /** \example file.cpp   Some short description
#  *
#  * Some comments here
# **/
#
# int func()
# {
# /** Here we explain something
#     very important over multiple lines
#  **/
#   std::cout << "Hello world!" << std::endl;
# }
#
# int main()
# {
#   /** Some more comments and what we do **/
#   func();
# }
#
# ------- End of example input ------------
#
# This gets modified in such a way that the whole file is a single
# Doxygen comment block. Thus, all '/**' except for the one line including
# '\example' or '@example' are replaced by '\code', while all '**/'
# are replaced by '\endcode'.
#
# For the example input above, this results in:
#
# ------- Start of example output ------------
#
# /** \example file.cpp   Some short description
#
#    Some comments here
# \code
#
# int func()
# {
#   \endcode Here we explain something
#   very important over multiple lines
#   \code
#   std::cout << "Hello world!" << std::endl;
# }
#
# int main()
# {
#   \endcode Some more comments and what we do \code
#   func();
# }
#
# \endcode <h2>Full Example Code</h2> */
# ------- End of example output ------------
#
# which is exactly what is needed to let Doxygen produce an example
# page with text blocks explaining things nicely. :-)
# Note that a compressed version of the sourcefile with Doxygen comments
# removed is listed at the bottom of the page automatically by Doxygen.
#

import sys

if len(sys.argv) < 2:
  print 'Missing file argument!'
  sys.exit(1)

filehandle = open( sys.argv[1], "r" )
is_example = False

for line in filehandle:
  if "/**" in line:
    if "\\example" in line or "@example" in line:
      is_example = True
      sys.stdout.write(line.replace("**/", "\code"))
    elif is_example:
      sys.stdout.write(line.replace("/**", "\endcode").replace("**/", "\code"))
    else:
      sys.stdout.write(line)
  elif "**/" in line and is_example:
    sys.stdout.write(line.replace("**/", "\code"))
  else:
    sys.stdout.write(line)

# Terminating comment sequence:
if is_example:
  sys.stdout.write("\endcode <h2>Full Example Code</h2> */\n")
filehandle.close()


