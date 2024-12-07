# Running the testing script repeatedly and timing it

N_RUNS = 2  # Number of times test is run

import cProfile as profile
import pstats

for i in range(N_RUNS):
    # Run the test script and create stats object with name "profile stats_i"
    # That also creates a stats file with the same name
    profile.run("exec(open('test.py').read())", f"profile_stats_{i}")

# Read all the stats objects into one. All the times are added
overall_stats = pstats.Stats("profile_stats_0")
for i in range(1, N_RUNS):
    overall_stats.add(f"profile_stats_{i}")

overall_stats.strip_dirs()
overall_stats.sort_stats(pstats.SortKey.TIME)
overall_stats.print_stats(30)
overall_stats.print_callers(30)

# For writing out a stats file that can be visualised in the browser
overall_stats.dump_stats("overall_stats.prof")