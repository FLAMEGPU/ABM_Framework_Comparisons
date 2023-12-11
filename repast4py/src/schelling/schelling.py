from repast4py import core, space, schedule, logging, random
from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType
from repast4py.geometry import find_2d_nghs_periodic

from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

import numpy as np
from typing import Final, Dict, Tuple
from mpi4py import MPI

import math, sys, csv, time, itertools

from functools import reduce

mainTimer_start = time.monotonic()
prePopulationTimer_start = time.monotonic()

# Configurable properties
THRESHOLD: Final = 0.375 # 0.375 == 3/8s to match integer models

A: Final = 0
B: Final = 1
UNOCCUPIED: Final = 2

def dp_to_array(dp):
    if dp is not None:
        return np.array((dp.x, dp.y))
    return None
    
def array_to_dp(arr):
    if arr is not None:
        return dpt(arr[0], arr[1])
    return None


class Cell(core.Agent): 

    TYPE = 0

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Cell.TYPE, rank=rank)
        self.current_type = UNOCCUPIED
        self.winner = None
        
    def save(self) -> Tuple:
        """Saves the state of this Boid as a Tuple.

        Used to move this Boid from one MPI rank to another.

        Returns:
            The saved state of this Boid.
        """
        return (self.uid, self.current_type, self.winner)
        
        
    def update(self, data: Tuple):    
        """Updates the state of this agent when it is a ghost
        agent on a rank other than its local one.

        Args:
            data: the new agent state
        """
        # The example only updates if there is a change, is there a performance reason for doing this?
        self.current_type      = data[1]
        self.winner            = data[2]
        
    def determineStatus(self):
        pt = model.grid.get_location(self)
        nghs = np.transpose(find_2d_nghs_periodic(np.array((pt.x, pt.y)), model.grid_box))

        same_type_neighbours = 0
        diff_type_neighbours = 0

        # Iterate 3x3 Moore neighbourhood (but skip ourself)
        for ngh in nghs:
            at = array_to_dp(ngh) # at._reset_from_array(ngh) does not work correctly???
            if pt == at:
                continue
            # Iterate cell agents within current grid cell
            for obj in model.grid.get_agents(at):
                if obj.uid[1] == Cell.TYPE:
                    same_type_neighbours += int(self.current_type == obj.current_type)
                    diff_type_neighbours += int(self.current_type != obj.current_type and obj.current_type != UNOCCUPIED)
        
        isHappy = (float(same_type_neighbours) / (same_type_neighbours + diff_type_neighbours)) > THRESHOLD if same_type_neighbours else False
        if not isHappy and self.winner is not None:
            old_winner = model.context.agent(self.winner)
            old_winner.movement_resolved = False
            self.winner = None
            self.current_type = UNOCCUPIED
        
    def selectWinner(self):
        if self.winner is not None:
            return
        pt = model.grid.get_location(self)
        
        bid_agents = [x for x in model.grid.get_agents(pt) if x.uid[1] == Agent.TYPE]
        # Random agent in the bucket wins
        if len(bid_agents):
            winner = random.default_rng.choice(bid_agents)
            winner.movement_resolved = True
            self.winner = winner.uid
            self.current_type = winner.my_type

class Agent(core.Agent): 

    TYPE = 1

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Agent.TYPE, rank=rank)
        self.my_type = UNOCCUPIED
        self.movement_resolved = True
        
    def save(self) -> Tuple:
        """Saves the state of this Boid as a Tuple.

        Used to move this Boid from one MPI rank to another.

        Returns:
            The saved state of this Boid.
        """
        return (self.uid, self.my_type, self.movement_resolved)
        
        
    def update(self, data: Tuple):    
        """Updates the state of this agent when it is a ghost
        agent on a rank other than its local one.

        Args:
            data: the new agent state
        """
        # The example only updates if there is a change, is there a performance reason for doing this?
        self.my_type           = data[1]
        self.movement_resolved = data[2]
            
        
    def bid(self, available_locations):
        if self.movement_resolved:
            return
        
        # Select and bid for a random location
        # DO NOT USE random.default_rng.choice() HERE, VERY SLOW!!!
        selected_location = available_locations[random.default_rng.integers(len(available_locations))]
        model.grid.move(self, array_to_dp(selected_location))
        
agent_cache = {}

def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """
    uid = agent_data[0]
    
    if uid in agent_cache:
        h = agent_cache[uid]
    else:
        if uid[1] == Cell.TYPE:
            h = Cell(uid[0], uid[2])
        elif uid[1] == Agent.TYPE:
            h = Agent(uid[0], uid[2])
        agent_cache[uid] = h
    # restore the agent state from the agent_data tuple
    h.update(agent_data)
    return h

class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()
        self.csv_log = params['csv.log']

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        GRID_WIDTH = int(params['grid.width'])
        grid_box = space.BoundingBox(int(0), int(GRID_WIDTH), int(0), int(GRID_WIDTH), 0, 0)
        self.grid_box = np.array((grid_box.xmin, grid_box.xmin + grid_box.xextent - 1, grid_box.ymin, grid_box.ymin + grid_box.yextent - 1))
        self.grid = space.SharedGrid('grid', bounds=grid_box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple, buffer_size=0, comm=comm)

        self.context.add_projection(self.grid)

        # Each rank must generate a unique seed
        # https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds
        random.default_rng = np.random.default_rng([self.rank, random.default_rng.bytes(4)])
        
        prePopulationTimer_stop = time.monotonic()
        if self.rank == 0:
            print("pre population (s): %.6f"%(prePopulationTimer_stop - prePopulationTimer_start))
        
        populationGenerationTimer_start = time.monotonic()

        # Only rank zero generates agents, for simplicity/to avoid conflict
        if self.rank == 0:
            # Calculate the total number of cells
            CELL_COUNT = GRID_WIDTH * GRID_WIDTH
            POPULATED_COUNT = int(params['population.count'])
            # Select POPULATED_COUNT unique random integers in the range [0, CELL_COUNT), by shuffling the iota.
            shuffledIota = [i for i in range(CELL_COUNT)]
            random.default_rng.shuffle(shuffledIota)
            for elementIdx in range(CELL_COUNT):
                # Create cell agent
                cell = Cell(elementIdx, self.rank)
                self.context.add(cell)
                # Setup it's location
                idx = shuffledIota[elementIdx]
                x = int(idx % GRID_WIDTH)
                y = int(idx / GRID_WIDTH)
                self.move(cell, x, y)
                # If the elementIDX is below the populated count, generated a mobile agent and make it the current winner
                if elementIdx < POPULATED_COUNT:
                    agent = Agent(elementIdx, self.rank)
                    self.context.add(agent)
                    agent.my_type = A if random.default_rng.uniform() < 0.5 else B
                    cell.current_type = agent.my_type
                    cell.winner = agent.uid
                    self.move(agent, x, y)
                
                
        populationGenerationTimer_stop = time.monotonic()
        if self.rank == 0:
            print("population generation (s): %.6f"%(populationGenerationTimer_stop - populationGenerationTimer_start))


    def at_end(self):
        if self.csv_log:
            # Log final agent positions to file
            self.csv_log = self.csv_log.replace(".csv", "%d.csv"%(self.comm.rank))
            with open(self.csv_log, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['"x"', '"y"', '"fx"', '"fy"'])
                for b in self.context.agents(Cell.TYPE):
                    agent_xy = self.grid.get_location(b)
                    t = b.current_type
                    writer.writerow([agent_xy.x, agent_xy.y, int(t==A or t==UNOCCUPIED), int(t==B or t==UNOCCUPIED)])
                    
    def remove(self, agent):
        self.grid.remove(agent)
        
    def move(self, agent, x, y):
        # timer.start_timer('grid_move')
        self.grid.move(agent, dpt(x, y))
        # timer.stop_timer('grid_move')

    def step(self):
        # print("{}: {}".format(self.rank, len(self.context.local_agents)))
        tick = self.runner.schedule.tick
        self.context.synchronize(restore_agent)

        # timer.start_timer('b_step')
        for c in self.context.agents(Cell.TYPE):
            c.determineStatus()
        # Plan Movement Submodel
        # Break from submodel once all agents have resolved their movement
        while True:
            self.context.synchronize(restore_agent)
            # Available agents output their location
            # Perform a local gather
            my_available_spaces = [dp_to_array(self.grid.get_location(c)) for c in self.context.agents(Cell.TYPE) if c.winner is None]
            # Collectively do a global gather of available spaces
            # TODO: this could potentially be made cheaper with a 2-pass
            #       step 1: move agents to a rank based on ratio of available spaces in each rank
            #       step 2: move agent to a random available space within it's new rank
            available_spaces = list(itertools.chain.from_iterable(self.comm.allgather(my_available_spaces)))
            # Unhappy agents bid for a new location
            for a in self.context.agents(Agent.TYPE):
                a.bid(available_spaces)

            self.context.synchronize(restore_agent)
            # Available locations check if anyone wants to move to them. If so, approve one and mark as unavailable
            # Update next type to the type of the mover
            # todo request winner as a ghost so their UID can be marked as winner
            for c in self.context.agents(Cell.TYPE):
                c.selectWinner()
                
            my_mvmt_resolved = any([not a.movement_resolved for a in self.context.agents(Agent.TYPE)])
            if not any(self.comm.allgather(my_mvmt_resolved)):
            #if not self.comm.allreduce(my_mvmt_resolved, op=MPI.ANY):
                break
        # timer.stop_timer('b_step')

    def run(self):
        simulateTimer_start = time.monotonic()
        self.runner.execute()
        simulateTimer_stop = time.monotonic()
        if self.rank == 0:
            print("simulate (s): %.6f"%(simulateTimer_stop - simulateTimer_start))

    def remove_agent(self, agent):
        self.context.remove(agent)

def run(params: Dict):
    """Creates and runs the Schelling Model.

    Args:
        params: the model input parameters
    """
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()
    mainTimer_stop = time.monotonic()
    if model.rank == 0:
        print("main (s): %.6f\n"%(mainTimer_stop - mainTimer_start))


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)
