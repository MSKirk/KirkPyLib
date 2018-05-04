import os
import SpikesModule as spm
import timeit

# Location of database file referencing the so-called "spikes files"). Update accordingly
data_dir = os.path.abspath('/Users/rattie/Data/Michael/spike_files/')
db_filepath = os.path.join(data_dir, 'Table_SpikesDB2.h5')
# Output directory where the filtered spikes fits files will be written.
output_dir = os.path.join(data_dir, 'filtered')
# Instantiate the lookup table used as read-only by and shared with the child processes (Linux & MacOS only!!)
# https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing (2nd answer from EelkeSpaak)
lut = spm.SpikesLookup(db_filepath, data_dir, output_dir, n_co_spikes=2)

## Uncomment for testing single processing on a few groups.
groups = list(range(4))
g = spm.process_map_IO(groups, lut)
#spm.multiprocess_IO(groups, lut, nworkers=4)

## Benchmark single and parallel processing.

groups = list(range(100))
benchmarks = []
nworkers_list = [1, 2, 4, 6]

for nworkers in nworkers_list:
    # Delete previously existing files in the output directory.
    for filename in os.listdir(output_dir):
        os.remove(os.path.abspath(os.path.join(output_dir, filename)))

    if nworkers == 1:
        benchmarks.append(
            timeit.Timer('spm.process_map_IO(groups, lut)',
                         'from __main__ import spm, groups, lut').timeit(number=1))
    else:
        benchmarks.append(
            timeit.Timer('spm.multiprocess_IO(groups, lut, nworkers)',
                     'from __main__ import spm, groups, lut, nworkers').timeit(number=1))

print('Timing with full IO:')
total_days = [btime / (len(groups) * 7) * 131e6 / (24*3600) for btime in benchmarks]
for i in range(len(nworkers_list)):
    print('%d core(s): %d days'%(nworkers_list[i], total_days[i]))


lut.spikes_db.close()
