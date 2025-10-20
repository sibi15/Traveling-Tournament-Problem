import xml.etree.ElementTree as ET
from ortools.sat.python import cp_model
import time
import csv

class TTPExactSolverComplete:
    def __init__(self, filename):
        """Initialize solver with XML data"""
        self.filename = filename
        (self.n_teams, self.team_names, self.distances, self.L, self.U) = self.parse_ttp_data(filename)
        self.num_rounds = 2 * (self.n_teams - 1)
        print(f"Initialized Complete TTP Solver:")
        print(f" Teams: {self.team_names}")
        print(f" Rounds: {self.num_rounds}, Capacity L={self.L}, U={self.U}")

    def parse_ttp_data(self, filename):
        """Parse XML file and extract tournament data"""
        tree = ET.parse(filename)
        root = tree.getroot()

        # Teams
        teams = sorted(root.findall('.//Teams/team'), key=lambda t: int(t.attrib['id']))
        team_names = [t.attrib['name'] for t in teams]
        n = len(team_names)

        # Distances
        dist = [[0]*n for _ in range(n)]
        for e in root.findall('.//Distances/distance'):
            i, j, d = int(e.attrib['team1']), int(e.attrib['team2']), int(e.attrib['dist'])
            dist[i][j] = d

        # Capacity constraints L/U
        L = U = None
        for c in root.findall('.//intp'):
            if c.get('mode1') == 'H':
                L, U = int(c.get('min')), int(c.get('max'))
                break
        if L is None:
            L, U = 1, 3

        return n, team_names, dist, L, U

    def calculate_travel_cost_complete(self, schedule):
        """Complete travel cost calculation including intermediate returns home"""
        total_cost = 0
        team_locations = list(range(self.n_teams))  # All teams start at home
        
        for r in range(self.num_rounds):
            new_locations = team_locations.copy()
            
            # Step 1: Home teams return home if needed
            for away_team, home_team in schedule[r]:
                if team_locations[home_team] != home_team:
                    return_home_cost = self.distances[team_locations[home_team]][home_team]
                    total_cost += return_home_cost
                    new_locations[home_team] = home_team
            
            # Update locations after home returns
            team_locations = new_locations.copy()
            
            # Step 2: Away teams travel to opponent's city
            for away_team, home_team in schedule[r]:
                from_city = team_locations[away_team]
                to_city = home_team
                
                if from_city != to_city:
                    travel_cost = self.distances[from_city][to_city]
                    total_cost += travel_cost
                
                new_locations[away_team] = home_team
            
            team_locations = new_locations
        
        # Final return home after tournament
        for team in range(self.n_teams):
            if team_locations[team] != team:
                final_return = self.distances[team_locations[team]][team]
                total_cost += final_return
        
        return total_cost

    def print_schedule_with_complete_distances(self, schedule):
        """Print schedule with COMPLETE distance calculation"""
        total_distance = 0
        team_locations = list(range(self.n_teams))  # start at home

        for r in range(self.num_rounds):
            if r in schedule and schedule[r]:
                print(f"Round {r+1}:")
                round_distance = 0
                new_locations = team_locations.copy()
                
                # Step 1: Home teams return home if needed
                for away_team, home_team in schedule[r]:
                    if team_locations[home_team] != home_team:
                        return_dist = self.distances[team_locations[home_team]][home_team]
                        round_distance += return_dist
                        total_distance += return_dist
                        new_locations[home_team] = home_team
                        print(f"  → {self.team_names[home_team]} returns home from {self.team_names[team_locations[home_team]]} for home game, dist={return_dist}")
                
                # Update locations after home returns
                team_locations = new_locations.copy()
                
                # Step 2: Away teams travel
                for away_team, home_team in schedule[r]:
                    from_city = team_locations[away_team]
                    to_city = home_team
                    dist = self.distances[from_city][to_city] if from_city != to_city else 0
                    round_distance += dist
                    total_distance += dist
                    new_locations[away_team] = to_city

                    print(f"  → {self.team_names[away_team]} @ {self.team_names[home_team]} "
                          f"(from {self.team_names[from_city]} to {self.team_names[to_city]}, dist={dist})")

                print(f"  Round {r+1} total distance: {round_distance}\n")
                team_locations = new_locations

        # Final return home
        return_distance = 0
        print("Final return home trips:")
        for team in range(self.n_teams):
            if team_locations[team] != team:
                dist = self.distances[team_locations[team]][team]
                return_distance += dist
                total_distance += dist
                print(f"  → {self.team_names[team]} returns home from {self.team_names[team_locations[team]]}, dist={dist}")

        print(f"\nFinal return distance: {return_distance}")
        print(f"COMPLETE travel distance: {total_distance}")

    def extract_schedule(self, solver, x):
        """Extract schedule from CP-SAT variables"""
        sched = {}
        for r in range(self.num_rounds):
            sched[r] = []
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j and solver.Value(x[i,j,r]) == 1:
                        sched[r].append((i,j))
        return sched

    def solve_exact(self, time_limit=300):
        """Build and solve CP-SAT model with complete travel cost"""
        model = cp_model.CpModel()
        x, home, away = {}, {}, {}

        # x[i,j,r] - only create when i != j
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i != j:
                    for r in range(self.num_rounds):
                        x[i,j,r] = model.NewBoolVar(f'x_{i}_{j}_{r}')

        # home/away flags
        for i in range(self.n_teams):
            for r in range(self.num_rounds):
                home[i,r] = model.NewBoolVar(f'home_{i}_{r}')
                away[i,r] = model.NewBoolVar(f'away_{i}_{r}')

        # y[i,s,t,r] - travel coupling variables
        y = {}
        for i in range(self.n_teams):
            for s in range(self.n_teams):
                for t in range(self.n_teams):
                    if s != t:
                        for r in range(self.num_rounds-1):
                            y[i,s,t,r] = model.NewBoolVar(f'y_{i}_{s}_{t}_{r}')

        print("Adding constraints...")

        # 1) Double round robin
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i != j:
                    model.Add(sum(x[i,j,r] for r in range(self.num_rounds)) == 1)
                    model.Add(sum(x[j,i,r] for r in range(self.num_rounds)) == 1)

        # 2) One game per round
        for i in range(self.n_teams):
            for r in range(self.num_rounds):
                games = [x[i,j,r] for j in range(self.n_teams) if i != j] + \
                        [x[j,i,r] for j in range(self.n_teams) if j != i]
                model.Add(sum(games) == 1)

        # 3) Home/Away definition
        for i in range(self.n_teams):
            for r in range(self.num_rounds):
                model.Add(home[i,r] == sum(x[j,i,r] for j in range(self.n_teams) if j != i))
                model.Add(away[i,r] == sum(x[i,j,r] for j in range(self.n_teams) if j != i))

        # 4) Capacity constraints L/U
        L, U = self.L, self.U
        for i in range(self.n_teams):
            # Max U consecutive
            for r in range(self.num_rounds - U):
                model.Add(sum(home[i,r+k] for k in range(U+1)) <= U)
                model.Add(sum(away[i,r+k] for k in range(U+1)) <= U)

            if L > 1:
                for r in range(self.num_rounds - L + 1):
                    # If starting a home streak, must continue for at least L rounds
                    if r > 0:
                        model.Add(home[i,r] - home[i,r-1] <= sum(home[i,r+k] for k in range(min(L, self.num_rounds-r))) / L)
                    # Similar for away streaks
                    if r > 0:
                        model.Add(away[i,r] - away[i,r-1] <= sum(away[i,r+k] for k in range(min(L, self.num_rounds-r))) / L)

        # 5) No repeaters (strengthened)
        for i in range(self.n_teams):
            for k in range(self.n_teams):
                if i != k:
                    for r in range(self.num_rounds-1):
                        model.Add(x[i,k,r] + x[k,i,r] + x[i,k,r+1] + x[k,i,r+1] <= 1)

        # 6) Travel coupling constraints
        for i in range(self.n_teams):
            for s in range(self.n_teams):
                for t in range(self.n_teams):
                    if s != t and i != s and i != t:
                        for r in range(self.num_rounds-1):
                            model.Add(y[i,s,t,r] >= x[i,s,r] + x[i,t,r+1] - 1)
                            model.Add(y[i,s,t,r] <= x[i,s,r])
                            model.Add(y[i,s,t,r] <= x[i,t,r+1])

        # Objective - minimize travel (using y variables for coupling)
        obj_terms = []
        
        # Travel between rounds
        for i in range(self.n_teams):
            for s in range(self.n_teams):
                for t in range(self.n_teams):
                    if s != t and i != s and i != t:
                        for r in range(self.num_rounds-1):
                            obj_terms.append(self.distances[s][t] * y[i,s,t,r])
        
        # Return home cost after last round
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i != j:
                    obj_terms.append(self.distances[j][i] * x[i,j,self.num_rounds-1])

        if obj_terms:
            model.Minimize(sum(obj_terms))

        # Solve
        print("Solving...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit

        start = time.time()
        status = solver.Solve(model)
        t_solve = time.time() - start

        schedule = {}
        total_cost = None
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = self.extract_schedule(solver, x)
            total_cost = self.calculate_travel_cost_complete(schedule)
            
            print(f"\nOR-Tools Results:")
            print(f"Status: {solver.StatusName(status)}")
            print(f"Solve time: {t_solve:.2f}s")
            print(f"Complete travel cost: {total_cost}\n")
            
            # Print detailed schedule
            self.print_schedule_with_complete_distances(schedule)
            
            # Print home/away patterns
            print("\nHome/Away Pattern:")
            print("=" * 60)
            for i, name in enumerate(self.team_names):
                pattern = ''.join('H' if solver.Value(home[i,r]) == 1 else 'A'
                                  for r in range(self.num_rounds))
                print(f" {name}: {pattern}")
                
        else:
            print(f"No solution found. Status: {solver.StatusName(status)}")
            print(f"Solve time: {t_solve:.2f} seconds")

        return {
            'status': solver.StatusName(status),
            'time': t_solve,
            'cost': total_cost,
            'schedule': schedule
        }

    def print_xml_format(self, schedule):
        
        for round_num in range(self.num_rounds):
            if round_num in schedule:
                for away_team, home_team in schedule[round_num]:
                    print(f' ScheduledMatch away="{away_team}" home="{home_team}" slot="{round_num}"')
        
    def export_schedule_to_csv(self, schedule, filename):
        """Export schedule to CSV"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Away_ID', 'Away_Name', 'Home_ID', 'Home_Name', 'Match'])
            
            for round_num in range(self.num_rounds):
                if round_num in schedule:
                    for away_id, home_id in schedule[round_num]:
                        writer.writerow([
                            round_num + 1,
                            away_id,
                            self.team_names[away_id],
                            home_id,
                            self.team_names[home_id],
                            f"{self.team_names[away_id]} @ {self.team_names[home_id]}"
                        ])
        
        print(f"Schedule exported to: {filename}")

if __name__ == "__main__":
    # Create solver
    solver = TTPExactSolverComplete("Data/NL6.xml")
    
    # Solve
    result = solver.solve_exact(time_limit=300)
    
    if result['status'] in ['OPTIMAL', 'FEASIBLE']:
        # Export XML and CSV
        solver.print_xml_format(result['schedule'])
        solver.export_schedule_to_csv(result['schedule'], 'schedule_NL6.csv')
