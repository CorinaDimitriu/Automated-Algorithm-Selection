import os
from smt_utils import Vlsi_smt_abstract
import subprocess

class Vlsi_smt(Vlsi_smt_abstract):
    """Class for solving the VLSI problem in SMT, using the encoding 2C.

    It inherits from the class `Vlsi_smt_abstract`.

    It is like the encoding 2A. SMT variables 'coordX[i]', 'coordY[i]' and 'l'. 
    The only difference is about the optimization procedure: the linear search is still 
    performed, but starting from the bottom instead of the top (i.e. starting from `l_min` instead of `l_max`).
    Still incremental solving and linear search, but from the bottom.

    The optimization procedure is the following.
    The SMT encoding is generated only one time, at the beginning. And the solver is started only one time, at the
    beginning (the solver is run on the terminal in incremental mode).
    Cycle. At each iteration we have a certain current 'l' that we want to try. We have already tested all the lengths below 
    'l', thus, if this 'l' is SAT, then this 'l' is the best possible 'l'.
    We inject into the solver a new constraint, ensuring that the actual length of the plate is equal to 'l'. 
    (Since we are sure that all the lengths below 'l' are not SAT, we are basically testing if 'l' is the best length for our
    problem). 
    Then, we solve that current solver instance.
    If SAT, then we simply take the best solution and we exit. If UNSAT, we update l<-l+1, we remove the last constraint 
    injected into the solver (i.e. the one ensuring that the actual length of the plate is equal to 'l').
    Then we repeat. 
    At the beginning, l<-l_min.

    INCREMENTAL SOLVING: the solver is created only at the beginning, then we dinamically inject/remove constraints from that
    solver. For obtaining this behaviour, we use the assertions stack (we push/pop levels into that stack).

    See the `__optimize` method.


    --- SUPPORTED SOLVERS ---
    Only solvers 'z3' and 'cvc5' are supported. No 'yices-smt2' ('yices-smt2' recquires that a specific logic is specified).

    """
    def __generate_encoding(self, l_max):
        """Generates the SMT encoding for the specific instance of the VLSI problem, according to the encoding 2C.

        The SMT encoding is generated as a single string, containing the SMT-LIB code, which will be passed to the solver.

        Remark: this returned string contains only the encoding, and it does not contain the SAT check and the "get model"
        instruction.

        Parameters
        ----------
        l_max : int
            Maximum length of the plate.

        Returns
        -------
        encoding_string : str
            Single string containing the encoding. More specifically, it contains the SMT-LIB lines.
            (Without the SAT check and the "get model" instruction).

        Notes
        ------
        The following SMT variables are used.
        - coordX[i], where 'i' in [0,n-1].
          coordX[i] is the x coordinate of the bottom-left vertex of the 'i'-th circuit.
        - coordY[i], where 'i' in [0,n-1].
          coordY[i] is the y coordinate of the bottom-left vertex of the 'i'-th circuit.
        - l
          It represents the length of the plate.
          
        """
        w, n, dimsX, dimsY = self.w, self.n, self.dimsX, self.dimsY

        lines = []  # List of lines for the SMT-LIB encoding. It is a list of strings.

        # For getting the model
        lines.append('(set-option :produce-models true)')

        # VARIABLES
        # We are defining the function "coordX": we are declaring n variables "coordX[i]".
        lines.append('(declare-fun coordX (Int) Int)')
        # We are defining the function "coordY": we are declaring n variables "coordY[i]".
        lines.append('(declare-fun coordY (Int) Int)')
        # We are defining the variable "l".
        lines.append('(declare-fun l () Int)')

        # CONSTRAINTS

        # 1- Domains
        # We add a single string, stating that l<=l_max
        lines.append(f'(assert (<= l {l_max}))')

        # We create a list of strings, one for each variable "coordX[i]".
        # For each "coordX[i]", we say that 0<="coordX[i]"<=w-dimsX[i]:
        #                        "(assert (and (>= (coordX {i}) 0) (<= (coordX {i}) (- {w} {dimsX[i]}))))".
        lines += [f'(assert (and (>= (coordX {i}) 0) (<= (coordX {i}) (- {w} {dimsX[i]}))))' for i in range(n)]

        # We create a list of strings, one for each variable "coordY[i]".
        # For each "coordY[i]", we say that 0<="coordY[i]"<=l_max-dimsY[i]:
        #                        "(assert (and (>= (coordY {i}) 0) (<= (coordY {i}) (- {l_max} {dimsY[i]}))))".
        lines += [f'(assert (and (>= (coordY {i}) 0) (<= (coordY {i}) (- {l_max} {dimsY[i]}))))' for i in range(n)]

        # 2- Non-overlapping
        # For each pair of circuits (i,j), where i<j, we impose the non-overlapping constraint: 
        #            coordX[i]+dimsX[i]<=coordX[j] \/ coordX[j]+dimsX[j]<=coordX[i] \/ coordY[i]+dimsY[i]<=coordY[j] \/ 
        #                                             coordY[j]+dimsY[j]<=coordY[i]
        lines += [f'(assert (or (<= (+ (coordX {i}) {dimsX[i]}) (coordX {j})) (<= (+ (coordX {j}) {dimsX[j]}) (coordX {i})) (<= (+ (coordY {i}) {dimsY[i]}) (coordY {j})) (<= (+ (coordY {j}) {dimsY[j]}) (coordY {i}))))' 
                for i in range(n) for j in range(n) if i<j]
        
        # 3- Length of the plate
        # For each circuit 'i', we impose that coordY[i]+dimsY[i]<=l
        lines += [f'(assert (<= (+ (coordY {i}) {dimsY[i]}) l))' for i in range(n)]

        # FINAL REFINEMENTS
        # Join all these strings, by means of the new line '\n'. Now we have a single string.
        encoding_string = '\n'.join(lines)
        # Add a final new line
        encoding_string += '\n'

        return encoding_string

    
    def __generate_new_constraint(self, curr_l, first_time):
        """Generates the new constraint to add into the solver (incremental solving).

        More precisely, this new constraint involves the given `curr_l`: we want to impose that the length of the plate must
        be equal to `curr_l`.

        Parameters
        ----------
        curr_l : int
            Current length of interest of the plate.
        first_time : bool
            It says whether it is the first time this method has been called or not.
            If it's not the first time, we must be careful in popping out the last imposed constraints.

        Returns
        -------
        constraint_string : str
            Single string containing the encoding of the constraint (i.e. SMT-LIB lines). 

        """
        # Strings representing the SMT-LIB encoding of the new constraint to add
        lines = []

        # Check if it's not the first time this method has been called
        if not first_time:
            # It's not the first time: we must pop out the last constraint imposed.
            lines.append('(pop 1)')
        
        # We push the new constraint into a new level of the assertions stack of the solver: in this way, we can retract 
        # this constraint, if necessary (if UNSAT).  
        lines.append('(push 1)')

        # We impose that the length of the plate must be equal than the current given length: l==curr_l
        lines.append(f'(assert (= l {curr_l}))')

        # Check satisfiability and get the model
        lines.append('(check-sat)')
        # String "(get-value ((coordX 0) (coordX 1) ... (coordX N) (coordY 0) (coordY 1) ... (coordY N)))"
        lines.append(f'(get-value ({" ".join([f"(coordX {i}) (coordY {i}) " for i in range(self.n)])}))')

        # Join all these strings, by means of the new line '\n'. Now we have a single string.
        constraint_string = '\n'.join(lines)
        # Add a final new line.
        constraint_string += '\n'

        return constraint_string


    def __optimize(self):
        """Solves the given VLSI instance, using the SAT encoding 2C.

        It performs optimization: the best solution is found (if any).
        (If this class is used as a parallel process with a time limit, there is not gurantee of founding the optimal 
        solution, but only the best solution found so far).

        The optimization procedure is the following.
        The SMT encoding is generated only one time, at the beginning. And the solver is started only one time, at the
        beginning (the solver is run on the terminal in incremental mode).
        Cycle. At each iteration we have a certain current 'l' that we want to try. We have already tested all the lengths below 
        'l', thus, if this 'l' is SAT, then this 'l' is the best possible 'l'.
        We inject into the solver a new constraint, ensuring that the actual length of the plate is equal to 'l'. 
        (Since we are sure that all the lengths below 'l' are not SAT, we are basically testing if 'l' is the best length for our
        problem). 
        Then, we solve that current solver instance.
        If SAT, then we simply take the best solution and we exit. If UNSAT, we update l<-l+1, we remove the last constraint 
        injected into the solver (i.e. the one ensuring that the actual length of the plate is equal to 'l').
        Then we repeat. 
        At the beginning, l<-l_min.

        INCREMENTAL SOLVING: the solver is created only at the beginning, then we dinamically inject/remove constraints from that
        solver. For obtaining this behaviour, we use the assertions stack (we push/pop levels into that stack).

        Notes
        ------
        - The solver is run on the terminal in **incremental mode**. This means that the solver process remains alive, and 
          always waits for the user new constraints. 
        - The solution is communicated to the user through the `results` dictionary, which is shared between the class and the 
          user. 
          Each time a better solution is found, it is injected to the `results` dictionary.

        """
        solver_name = self.solver_name
        solver_path = os.path.join(os.getcwd(), 'vlsi', 'src', 'smt', 'solvers', solver_name)

        w, n, dimsX, dimsY = self.w, self.n, self.dimsX, self.dimsY

        # Computing the lowe and upper bound of the length of the plate (i.e. `l_min` and `l_max`)
        areas = [dimsX[i]*dimsY[i] for i in range(n)]  # The areas of the circuits
        A_tot = sum(areas)  # The overall area of all the given circuits
        h_max = max(dimsY)  # The maximum height of a circuit
        w_max = max(dimsX)  # The maximum width of a circuit
        l_min = max([h_max, A_tot // w])  # The lower bound for the length
        min_rects_per_row = w // w_max  # Minimum number of circuits per row
        if min_rects_per_row==0:  # UNSAT
            self.results['finish'] = True
            self.results['unsat'] = True  
            return
        sorted_dimsY = sorted(dimsY, reverse=True)  
        if min_rects_per_row==1:
            l_max = sum(dimsY)
        else:
            l_max = sum([sorted_dimsY[i] for i in range(n) if i % min_rects_per_row == 0])  # The upper bound for the length

        # Command to execute on the terminal, for starting the solver process in incremental mode
        if solver_name=='z3':
            command = f'{solver_path} -in -T:{self.time_limit}'
        elif solver_name=='cvc5':
            command = f'{solver_path} --incremental --tlimit={self.time_limit*1000}'
        
        # Start the solver process on the terminal.
        # We use pipes for the stdin and the stdout of the solver process, for communicating with it.
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        # Generate the string representing the SMT encoding
        encoding = self.__generate_encoding(l_max=l_max)

        # Pass to the solver process the SMT encoding
        process.stdin.write(encoding.encode('utf-8'))
        process.stdin.flush()
    
        # Starting length of interest: it is `l_min`
        l = l_min

        while l<=l_max:        
            # Generate the new constraint to pass to the solver
            constraint = self.__generate_new_constraint(l, first_time=(l==l_min))

            # Pass the new constraint into the solver
            process.stdin.write(constraint.encode('utf-8'))
            process.stdin.flush()

            # Get the solver response
            output = process.stdout.read1().decode('utf-8') # String sat/unsat
            output += process.stdout.read1().decode('utf-8') # String containing the model (if any)

            # Check if SAT
            if 'sat' in output and 'unsat' not in output:
                # SAT: we have found the best solution

                # Parse the output of the solver into the list of coords, i.e. `coords`
                if solver_name=='z3':
                    coords = [int(s.split(')')[1]) for s in output.split('\n')[1:-1]]
                elif solver_name=='cvc5':
                    coords = [int(s.split(')')[1]) for s in output.split('\n')[1:-1][0].split(' (')]
                # List of coords, for each circuit 'i'. For each circuit 'i', we have the list of two elements [coordX_i, coordY_i]
                coords = [[coords[2*i], coords[2*i+1]] for i in range((len(coords))//2)]
    
                # TODO: remove
                #print(l)
                #print(coords)

                # Best solution
                self.results['best_coords'] = coords
                self.results['best_l'] = l
                break

            # UNSAT: update `l` and do next iteration       
            l = l+1

        # The computation is finished
        self.results['finish'] = True
                
        if l>l_max:  # No solution has been found: UNSAT
            self.results['unsat'] = True         


    def run(self):
        # Code executed by the process when it is run
        self.__optimize()