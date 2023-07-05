#include "cudapfem2particle.cuh"

template<int dim>
__host__ __device__ cudaPfem2Particle<dim>::cudaPfem2Particle()
{
    for(int i = 0; i < dim; ++i){
        this->location[i] = 0.0;
        this->reference_location[i] = 0.0;
        this->velocity[i] = 0.0;
        this->velocity_ext[i] = 0.0;
    }

    this->id = 0.0;
    this->cell = nullptr;
}

template<int dim>
__host__ __device__ cudaPfem2Particle<dim>::cudaPfem2Particle(const double *location, const double *reference_location, const unsigned id)
{
    for(int i = 0; i < dim; ++i){
        this->location[i] = *(location + i);
        this->reference_location[i] = *(reference_location + i);
        this->velocity[i] = 0.0;
        this->velocity_ext[i] = 0.0;
    }

    this->id = id;
    this->cell = nullptr;
}
