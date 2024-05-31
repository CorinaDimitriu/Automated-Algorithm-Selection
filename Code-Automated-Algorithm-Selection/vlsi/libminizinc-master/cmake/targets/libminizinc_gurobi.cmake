### MiniZinc Gurobi Solver Target

if (NOT DEFINED EMSCRIPTEN)
	### Compile target for the Gurobi interface
	add_library(minizinc_gurobi OBJECT
		solvers/MIP/MIP_gurobi_solverfactory.cpp
		solvers/MIP/MIP_gurobi_wrap.cpp

		include/minizinc/solvers/MIP/MIP_gurobi_solverfactory.hh
		include/minizinc/solvers/MIP/MIP_gurobi_wrap.hh
		)
	add_dependencies(minizinc_gurobi minizinc_mip)

	### Setup correct compilation into the MiniZinc library
  	target_compile_definitions(mzn PRIVATE HAS_GUROBI)
	target_sources(mzn PRIVATE $<TARGET_OBJECTS:minizinc_gurobi>)
endif()
