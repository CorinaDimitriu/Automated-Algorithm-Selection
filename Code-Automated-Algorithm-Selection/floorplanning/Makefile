CCC = g++ 
FLAGS = -g -Wall -Wextra 
GRBPATH = $(GUROBI_HOME)
SOURCE_DIR = ./src
SOURCES := $(shell find $(SOURCE_DIR) -name '*.cpp')

exec: $(SOURCES)
	$(CCC) $(FLAGS) $(SOURCES) -o floorplan -I$(GRBPATH)/include -L$(GRBPATH)/lib -gurobi_c++ -gurobi100 -I$(GRAPHICS_CPP_INCLUDE) -L$(GRAPHICS_CPP_LIB) -graph

clean: 
	rm -f floorplan *.log
