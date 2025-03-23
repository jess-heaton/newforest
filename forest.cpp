#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>     
#include <string>
#include <cstdlib>    
#include <ctime>       
#include <algorithm>     
#include <sstream>       

// Cell states
static const int EMPTY   = 0;
static const int TREE    = 1;
static const int BURNING = 2;
static const int DEAD    = 3;


// Divide grid into blocks for different processes to handle
void distribute_grid(int N, int iproc, int nproc, int &i0, int &i1)
{
    int block = N / nproc;
    i0 = iproc * block;
    
    // Checks if process is last one 
    if (iproc == nproc - 1) {
        i1 = N;
    } else {
        i1 = i0 + block;
    }    
}

// Read grid from file and broadcast
std::vector<int> readGrid(const std::string &filename, int &N, int iproc, int nproc) {
    std::vector<int> grid;

    // First process reads the file
    if (iproc == 0) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "No file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read the file line by line
        std::string line;
        std::vector<std::vector<int>> rows;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;
            std::stringstream ss(line);
            std::vector<int> row;
            int number;
            while (ss >> number) {
                row.push_back(number);
            }
            if (!row.empty()) {
                rows.push_back(row);
            }
        }
        file.close();

        // Check file has data
        int rowCount = rows.size();
        if (rowCount == 0) {
            std::cerr << "File empty"<< std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Set the grid size N and flatten row by row
        N = rowCount;
        grid.resize(N * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                grid[i * N + j] = rows[i][j];
            }
        }
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize grid for all other processes too
    if (iproc != 0) {
        grid.resize(N * N);
    }
    // Broadcast the grid data
    MPI_Bcast(grid.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    return grid;
}

// Generate grid randomly and broadcast
std::vector<int> generate_random_grid(int N, double p, int iproc, int nproc, int run, unsigned int baseSeed = 42) {
    
    std::vector<int> grid(N * N, EMPTY);

    // First process generates the grid
    if (iproc == 0) {

        // Random number gen with seed based on baseSeed and run number
        unsigned int seed = baseSeed + 1000 * run;
        std::mt19937 generator(seed);
        std::uniform_real_distribution<double> randomNumber(0.0, 1.0);

        // Fill the grid
        for (int i = 0; i < N * N; i++) {
            double r = randomNumber(generator);
            if (r < p) {
                // Assign TREE if the random number is less than p
                grid[i] = TREE;
            } else {

                // Otherwise EMPTY
                grid[i] = EMPTY;
            }
        }

        // Ignite top row of trees
        for (int j = 0; j < N; j++) {
            if (grid[j] == TREE) {
                grid[j] = BURNING;
            }
        }
    }

    // Broadcast generated grid to all the processes
    MPI_Bcast(grid.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    return grid;
}

// Simulation step 
// (added Moore Neighbourhood option as an extension - hard coded parameter)
bool step(std::vector<int> &oldGrid, int N, int iproc, int nproc, int i0, int i1, bool moore=false) {
    
    std::vector<int> newGrid(oldGrid); 
    bool localBurning = false;

    // Work out neighbours

    // Von Neumann Method
    int d4i[4] = {-1,  1,  0,  0};
    int d4j[4] = { 0,  0, -1,  1};

    // Moore 
    int d8i[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
    int d8j[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

    // Only rows assigned to this process
    for (int i = i0; i < i1; i++) {

        // Over columns
        for (int j = 0; j < N; j++) {
            int idx = i*N + j;
            if (oldGrid[idx] == BURNING) {
                
                // Transform burning cells to dead trees
                newGrid[idx] = DEAD;

                // Work out neighbours according to chosen method
                int nCount = (moore ? 8 : 4);
                for (int k = 0; k < nCount; k++) {
                    int ni = i + (moore ? d8i[k] : d4i[k]);
                    int nj = j + (moore ? d8j[k] : d4j[k]);
                    if (ni >= 0 && ni < N && nj >= 0 && nj < N) {
                        int nidx = ni*N + nj;
                        
                        // Set neighbourding trees alight
                        if (oldGrid[nidx] == TREE) {
                            newGrid[nidx] = BURNING;
                        }
                    }
                }
            }
        }
    }

    // Update oldGrid
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N + j;
            oldGrid[idx] = newGrid[idx];
        }
    }

    // Update all processes
    MPI_Allreduce(MPI_IN_PLACE, oldGrid.data(), N*N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Check if any cells are burning
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N + j;
            if (oldGrid[idx] == BURNING) {
                localBurning = true;
                break;
            }
        }

        if (localBurning) break;
    }

    int localFlag = localBurning ? 1 : 0;
    int globalFlag;

    // Includes all flags across processes
    MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // If any burning cells remain, keep sim going
    return (globalFlag > 0);
}


int main(int argc, char* argv[]) {

    // Init mpi
    MPI_Init(&argc, &argv);
    int iproc, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Default params
    int    N       = 100;
    double p       = 0.5;
    int    M       = 50;
    bool   useFile = false;

    std::string filename;

    // Rank 0 parses cmd args
    if (iproc == 0) {
        if (argc >= 2) N = std::atoi(argv[1]);
        if (argc >= 3) p = std::atof(argv[2]);
        if (argc >= 4) M = std::atoi(argv[3]);
        if (argc >= 5) {
            useFile = true;
            filename = argv[4];
        }
    }

    // Broadcast
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int useFileInt = (useFile ? 1 : 0);
    MPI_Bcast(&useFileInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    useFile = (useFileInt == 1);

    // Divide grid over ranks
    int i0, i1;
    distribute_grid(N, iproc, nproc, i0, i1);

    // Vars for total M runs
    double totalSteps = 0.0;
    double totalTime  = 0.0;
    int    totalHitBottom = 0;

    // Read grid from file if given
    std::vector<int> fileGridData;
    if (useFile) {
        fileGridData = readGrid(filename, N, iproc, nproc);
    }

    // Run M simulations
    for (int run = 0; run < M; run++) {
        
        // Build initial grid 
        std::vector<int> gridData;
        
        if (useFile) {
            // Copy the data from file
            gridData = fileGridData;

            // Ignite top row
            if (iproc == 0) {
                for (int j = 0; j < N; j++) {
                    int idx = j;
                    if (gridData[idx] == TREE) {
                        gridData[idx] = BURNING;
                    }
                }
            }

            // Synchronize
            MPI_Bcast(gridData.data(), N*N, MPI_INT, 0, MPI_COMM_WORLD);
        } else {

            // Generate a random grid (top row ignited in function)
            gridData = generate_random_grid(N, p, iproc, nproc, run);
        }

        double startTime = MPI_Wtime();
        int steps = 0;

        // Keep running sim until no cells are burning
        while (true) {
            bool burning = step(gridData, N, iproc, nproc, i0, i1, true);
            if (burning) {
                steps++;
            } else {
                break;
            }
        }

        // Calc time & totals
        double runTime = MPI_Wtime() - startTime;
        totalSteps += steps;
        totalTime  += runTime;

        // Check the bottom row for burning cells
        if (iproc == 0) {
            bool bottomHit = false;
            for (int j = 0; j < N; j++) {
                int idx = (N - 1) * N + j;
                if (gridData[idx] == DEAD || gridData[idx] == BURNING) {
                    bottomHit = true;
                    break;
                }
            }
            if (bottomHit) {
                totalHitBottom++;
            }
        }
    }

    // Averages
    double avgSteps = totalSteps / M;
    double avgTime  = totalTime  / M;
    double bottomFraction = 0.0;

    if (iproc == 0) {
        bottomFraction = double(totalHitBottom) / double(M);
    }

    // Print results
    if (iproc == 0) {
        std::cout << nproc << " "
                  << N << " "
                  << p << " "
                  << M << " "
                  << avgSteps << " "
                  << avgTime << " "
                  << bottomFraction
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}