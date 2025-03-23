import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('performance_data.txt', skiprows=1)

N_COL       = 0
NPROC_COL   = 4
AVGTIME_COL = 9

Ns = np.unique(data[:, N_COL])

plt.figure(figsize=(10, 6))
for N in Ns:
    subset = data[data[:, N_COL] == N]
    plt.plot(
        subset[:, NPROC_COL],
        subset[:, AVGTIME_COL],
        marker='o',
        label=f'N={int(N)}'
    )

plt.xlabel('Number of MPI Processes')
plt.ylabel('Average Runtime (s)')
plt.title('Scaling of Forest Fire Simulation')
plt.grid(True)
plt.legend()

# Make the y-axis log scale
plt.yscale('log')

plt.savefig("my_plot_log.png")
plt.show()  # optional