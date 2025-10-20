import xml.etree.ElementTree as ET
import random
import copy
import time
from collections import defaultdict

class TTPGeneticSolver:
    def __init__(self, filename, population_size=50, generations=250, mutation_rate=0.25):
        self.filename = filename
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.teams, self.distances, self.L, self.U = self.load_instance(filename)
        self.n_teams = len(self.teams)
        self.n_rounds = 2 * (self.n_teams - 1)

    def load_instance(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        teams = [t.attrib["name"] for t in root.findall(".//team")]
        n = len(teams)
        distances = [[0]*n for _ in range(n)]
        for d in root.findall(".//distance"):
            i = int(d.attrib["team1"])
            j = int(d.attrib["team2"])
            dist = int(d.attrib["dist"])
            distances[i][j] = dist
            distances[j][i] = dist
        L, U = 1, n
        sep = root.find(".//SeparationConstraints")
        if sep is not None:
            for sc in sep.findall("SE1"):
                L = int(sc.attrib.get("min", L))
                U = int(sc.attrib.get("max", U))
        return teams, distances, L, U

    def generate_feasible_schedule(self):
        n = self.n_teams
        teams = list(range(n))
        rounds = []
        for round_idx in range(n-1):
            pairs = []
            for i in range(n // 2):
                t1 = teams[i]
                t2 = teams[n - 1 - i]
                if round_idx % 2 == 0:
                    pairs.append([t1, t2])
                else:
                    pairs.append([t2, t1])
            rounds.append(pairs)
            teams = [teams[0]] + [teams[-1]] + teams[1:-1]
        rounds += [[[a, b] for (b, a) in round_] for round_ in rounds]
        return rounds

    def is_valid(self, schedule):
        L, U = self.L, self.U
        n_teams = self.n_teams
        matches_home = defaultdict(set)
        matches_away = defaultdict(set)
        streaks = [0]*n_teams
        last_hw = [None]*n_teams
        last_opponent = [None]*n_teams

        for rnd in schedule:
            played = set()
            for home, away in rnd:
                if home == away or home in played or away in played:
                    return False
                played.add(home)
                played.add(away)
                matches_home[home].add(away)
                matches_away[away].add(home)
                if last_opponent[home] == away or last_opponent[away] == home:
                    return False
                last_opponent[home] = away
                last_opponent[away] = home
            for home, away in rnd:
                for t, hw in ((home, 'H'), (away, 'A')):
                    if last_hw[t] == hw:
                        streaks[t] += 1
                    else:
                        if last_hw[t] is not None and streaks[t] < L:
                            return False
                        streaks[t] = 1
                        last_hw[t] = hw
                    if streaks[t] > U:
                        return False
        for t in range(n_teams):
            if streaks[t] < L:
                return False
        for i in range(n_teams):
            if len(matches_home[i]) != n_teams - 1 or len(matches_away[i]) != n_teams - 1:
                return False
            expected_opponents = set(range(n_teams)) - {i}
            if matches_home[i] != expected_opponents or matches_away[i] != expected_opponents:
                return False
        return True

    def calculate_travel(self, schedule):
        n_teams = self.n_teams
        distances = self.distances
        location = list(range(n_teams))
        total_travel = 0
        for rnd in schedule:
            for home, away in rnd:
                if location[away] != home:
                    total_travel += distances[location[away]][home]
                    location[away] = home
                if location[home] != home:
                    total_travel += distances[location[home]][home]
                    location[home] = home
        for team in range(n_teams):
            if location[team] != team:
                total_travel += distances[location[team]][team]
        return total_travel

    def mutate(self, schedule):
        new_schedule = copy.deepcopy(schedule)
        r1, r2 = random.sample(range(len(new_schedule)), 2)
        new_schedule[r1], new_schedule[r2] = new_schedule[r2], new_schedule[r1]
        return new_schedule

    def crossover(self, p1, p2):
        point = random.randint(1, len(p1)-2)
        return copy.deepcopy(p1[:point] + p2[point:])

    def solve(self):
        base_schedule = self.generate_feasible_schedule()
        print("\n" + "="*70)
        if not self.is_valid(base_schedule):
            print("ERROR: Base Schedule is invalid!")
            return base_schedule, float('inf')
        print(f"\nBase Schedule Travel Cost: {self.calculate_travel(base_schedule)}")
        population = []
        attempts = 0
        max_attempts = self.population_size * 100
        while len(population) < self.population_size and attempts < max_attempts:
            candidate = copy.deepcopy(base_schedule)
            random.shuffle(candidate)
            if self.is_valid(candidate):
                population.append(candidate)
            attempts += 1
        if len(population) == 0:
            print("Could not generate any valid schedules!")
            return base_schedule, self.calculate_travel(base_schedule)
        print(f"Generated {len(population)} valid schedules in initial population")
        best_solution = min(population, key=lambda s: self.calculate_travel(s))
        best_cost = self.calculate_travel(best_solution)
        print(f"\nInitial Best Cost: {best_cost}\n")
        start_time = time.time()
        for gen in range(self.generations):
            new_population = []
            attempts = 0
            max_gen_attempts = self.population_size * 50
            while len(new_population) < self.population_size and attempts < max_gen_attempts:
                parents = random.sample(population, 2)
                if random.random() < 0.5:
                    child = self.mutate(parents[0])
                else:
                    child = self.crossover(parents[0], parents[1])
                    if random.random() < self.mutation_rate:
                        child = self.mutate(child)
                if self.is_valid(child):
                    new_population.append(child)
                attempts += 1
            if len(new_population) < self.population_size:
                needed = self.population_size - len(new_population)
                new_population.extend(random.sample(population, min(needed, len(population))))
            population = new_population
            current_best = min(population, key=lambda s: self.calculate_travel(s))
            current_cost = self.calculate_travel(current_best)
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_best
                print(f"Generation {gen+1}: New best found! Travel Cost: {best_cost}")
            elif gen % 10 == 0:
                print(f"Generation {gen+1}: Best Travel Cost: {best_cost}")
        return best_solution, best_cost

    def print_schedule(self, schedule):
        print("\nSchedule:")
        for i, rnd in enumerate(schedule):
            games = ' '.join(f"{self.teams[a]}@{self.teams[h]}" for h, a in rnd)
            print(f" Round {i+1}: {games}")
        print("\nH/A Patterns:")
        patterns = [''] * self.n_teams
        for rnd in schedule:
            for h, a in rnd:
                patterns[h] += 'H'
                patterns[a] += 'A'
        for idx, pat in enumerate(patterns):
            print(f" {self.teams[idx]}: {pat}")

        print("\nTravel Breakdown:")
        # Calculate travel for each team
        n_teams = self.n_teams
        distances = self.distances
        location = list(range(n_teams))
        travel = [0] * n_teams
        for rnd in schedule:
            for h, a in rnd:
                if location[a] != h:
                    travel[a] += distances[location[a]][h]
                    location[a] = h
                if location[h] != h:
                    travel[h] += distances[location[h]][h]
                    location[h] = h
        for team in range(n_teams):
            if location[team] != team:
                travel[team] += distances[location[team]][team]
        for idx, t in enumerate(self.teams):
            print(f" {t}: {travel[idx]}")

        print("\nScheduledMatch output:")
        for slot, rnd in enumerate(schedule):
            for h, a in rnd:
                print(f'ScheduledMatch away="{a}" home="{h}" slot="{slot}"')
        print("\n")

if __name__ == "__main__":
    start=time.perf_counter()
    solver = TTPGeneticSolver("Data/NL4.xml", population_size=50, generations=150)
    solution, cost = solver.solve()
    print("\n" + "="*70)
    solver.print_schedule(solution)
    print("="*70)
    print(f"\nTotal Travel Cost: {cost}")
    if solver.is_valid(solution):
        print("✓ Schedule is VALID\n")
    else:
        print("✗ Schedule is INVALID")
    end = time.perf_counter()
    print(f"\nTime taken: {end - start:.2f} seconds\n")
