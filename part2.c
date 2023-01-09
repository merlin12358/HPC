#include <stdio.h>				// needed for printing
#include <math.h>				// needed for tanh, used in init function
#include <mpi.h>
const int N 	= 128;			// domain size
const int M		= 50000;		// number of time steps
const double a 	= 0.3;			// model parameter a
const double b 	= 0.1;			// model parameter b
const double c 	= 0.01;			// model parameter c
const double d 	= 0.0;			// model parameter d
const double dt	= 0.1;			// time step
const double dx	= 2.0;			// spatial resolution
const double DD = 1.0/(dx*dx);	// diffusion scaling
const int m		= (int)(1/dt);	// Norm calculation

void init(double u[N][N], double v[N][N]){
    int size, rank;
    double uhi, ulo, vhi, vlo;
    uhi = 0.5; ulo = -0.5; vhi = 0.1; vlo = -0.1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Determine the portion of the grid that this process will handle
    int rows_per_process = N / size;
    int start_row = rows_per_process * rank;
    int end_row = start_row + rows_per_process;

    // Initialize the portion of the grid assigned to this process
    for (int i = start_row; i < end_row; i++){
        for (int j = 0; j < N; j++){
            u[i][j] = ulo + (uhi-ulo)*0.5*(1.0 + tanh((i-N/2)/16.0));
            v[i][j] = vlo + (vhi-vlo)*0.5*(1.0 + tanh((j-N/2)/16.0));
        }
    }
}


void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]){
    int size, rank;
    double lapu, lapv;
    int up, down, left, right;
    double global_lapu, global_lapv;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Determine the portion of the grid that this process will handle
    int rows_per_process = N / size;
    int start_row = rows_per_process * rank;
    int end_row = start_row + rows_per_process;

    // Calculate the diffusion term for the portion of the grid assigned to this process
    for (int i = start_row; i < end_row; i++){
        for (int j = 0; j < N; j++){
            if (i == 0){
                down = i;
            }
            else{
                down = i-1;
            }
            if (i == N-1){
                up = i;
            }
            else{
                up = i+1;
            }
            if (j == 0){
                left = j;
            }
            else{
                left = j-1;
            }
            if (j == N-1){
                right = j;
            }
            else{
                right = j+1;
            }
            lapu = u[up][j] + u[down][j] + u[i][left] + u[i][right] + -4.0*u[i][j];
            lapv = v[up][j] + v[down][j] + v[i][left] + v[i][right] + -4.0*v[i][j];
            du[i][j] = DD*lapu + u[i][j]*(1.0 - u[i][j])*(u[i][j]-b) - v[i][j];
            dv[i][j] = d*DD*lapv + c*(a*u[i][j] - v[i][j]);
        }
    }

    // Perform a reduction operation on the diffusion term values from all ranks
    MPI_Allreduce(&lapu, &global_lapu, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&lapv, &global_lapv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // Use the global sum of the diffusion term values to update the du and dv arrays
    for (int i = start_row; i < end_row; i++){
        for (int j = 0; j < N; j++){
            du[i][j] = DD*global_lapu + u[i][j]*(1.0 - u[i][j])*(u[i][j]-b) - v[i][j];
            dv[i][j] = d*DD*global_lapv + c*(a*u[i][j] - v[i][j]);
        }
    }
}


void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]){
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Determine the portion of the grid that this process will handle
    int rows_per_process = N / size;
    int start_row = rows_per_process * rank;
    int end_row = start_row + rows_per_process;

    // Send and receive values of the border indices to/from neighboring processes
    if (rank > 0) {
        MPI_Send(&u[start_row][0], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
        MPI_Recv(&u[start_row-1][0], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size-1) {
        MPI_Send(&u[end_row-1][0], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
        MPI_Recv(&u[end_row][0], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Update the portion of the grid assigned to this process
    for (int i = start_row; i < end_row; i++){
        for (int j = 0; j < N; j++){
            u[i][j] += dt*du[i][j];
            v[i][j] += dt*dv[i][j];
        }
    }
}

double norm(double x[N][N]){
    int size, rank;
    double local_norm, global_norm;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calculate the norm of the portion of the array assigned to this process
    local_norm = 0.0;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            local_norm += x[i][j]*x[i][j];
        }
    }

    // Perform a reduction operation on the norm values from all ranks
    MPI_Reduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_norm;
}


int main(int argc, char** argv){
    MPI_Init( &argc, &argv );
    double t = 0.0, nrmu, nrmv;
    double u[N][N], v[N][N], du[N][N], dv[N][N];

    FILE *fptr = fopen("nrms.txt", "w");
    fprintf(fptr, "# t\t\tnrmu\t\tnrmv\n");

    // initialize the state
    init(u, v);

    // time-loop
    for (int k=0; k < M; k++){
        // track the time
        t = dt*k;
        // evaluate the PDE
        dxdt(du, dv, u, v);
        // update the state variables u,v
        step(du, dv, u, v);
        if (k%m == 0){
            // calculate the norms
            nrmu = norm(u);
            nrmv = norm(v);
            printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
            fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
        }
    }

    fclose(fptr);
    MPI_Finalize();
    return 0;
}
