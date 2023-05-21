#include "pfem2particle.h"

#include <deal.II/particles/property_pool.h>

using namespace dealii;

template<int dim>
pfem2Particle<dim>::pfem2Particle(const Point<dim> &location, const Point<dim> &reference_location, const unsigned id)
	: id (id)
{
	for(int i = 0; i < dim; ++i){
		this->location[i] = location[i];
		this->reference_location[i] = reference_location[i];
		this->velocity[i] = 0.0;
		this->velocity_ext[i] = 0.0;
	}
}

template<int dim>
pfem2Particle<dim>::pfem2Particle(const void* &data)
{
	const types::particle_index *id_data = static_cast<const types::particle_index*> (data);
    id = *id_data++;
    
    const int *triaData = reinterpret_cast<const int*> (id_data);
    tria_position = *triaData++;
    
    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
		cell_dofs[i] = *triaData++;

    const double *pdata = reinterpret_cast<const double*> (triaData);

    for (unsigned int i = 0; i < dim; ++i) 
		location[i] = *pdata++;

    for (unsigned int i = 0; i < dim; ++i)
		reference_location[i] = *pdata++;

    for (unsigned int i = 0; i < dim; ++i)
		velocity[i] = *pdata++;
    
	for (unsigned int i = 0; i < dim; ++i)
		velocity_ext[i] = *pdata++;

    data = static_cast<const void*> (pdata);
}

template<int dim>
pfem2Particle<dim>::pfem2Particle()
	: pfem2Particle(Point<dim>(), Point<dim>(), 0)
{
}

template<int dim>
void pfem2Particle<dim>::set_location (const Point<dim> &new_location)
{
	for(int i = 0; i < dim; ++i)
		location[i] = new_location[i];
}

template<int dim>
const Point<dim> &pfem2Particle<dim>::get_location () const
{
    Point<dim> res;
    for(int i = 0; i < dim; ++i)
		res[i] = location[i];
    
    return res;
}

template<int dim>
void pfem2Particle<dim>::set_reference_location (const Point<dim> &new_reference_location)
{
    for(int i = 0; i < dim; ++i)
		reference_location[i] = new_reference_location[i];
}

template<int dim>
const Point<dim> &pfem2Particle<dim>::get_reference_location () const
{
	Point<dim> res;
    for(int i = 0; i < dim; ++i)
		res[i] = reference_location[i];
    
    return res;
}

template<int dim>
unsigned int pfem2Particle<dim>::get_id () const
{
    return id;
}

template<int dim>
void pfem2Particle<dim>::set_cell_dofs(const typename DoFHandler<dim>::active_cell_iterator &cell)
{
	for(unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
		cell_dofs[i] = cell->vertex_dof_index(i, 0);
}

#ifdef WITH_CUDA
template<int dim>
void pfem2Particle<dim>::set_cuda_constants()
{
	cudaError_t err = cudaSuccess;
	
	size_t pfem2ParticleSize = sizeof(pfem2Particle<dim>);
	err = cudaMemcpyToSymbol(particleSize, &pfem2ParticleSize,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t locationOffset = offsetof(pfem2Particle<dim>, location);
	err = cudaMemcpyToSymbol(locationPos, &locationOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t refLocationOffset = offsetof(pfem2Particle<dim>, reference_location);
	err = cudaMemcpyToSymbol(refLocationPos, &refLocationOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t velocityOffset = offsetof(pfem2Particle<dim>, velocity);
	err = cudaMemcpyToSymbol(velocityPos, &velocityOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t velocityExtOffset = offsetof(pfem2Particle<dim>, velocity_ext);
	err = cudaMemcpyToSymbol(velocityExtPos, &velocityExtOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
	
	size_t cellDofsOffset = offsetof(pfem2Particle<dim>, cell_dofs);
	err = cudaMemcpyToSymbol(cellDoFsPos, &cellDofsOffset,  sizeof(size_t));
	if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
}
#endif

template<int dim>
int pfem2Particle<dim>::get_tria_position() const
{
	return tria_position;
}

template<int dim>
void pfem2Particle<dim>::set_tria_position(const int &new_position)
{
	tria_position = new_position;
}

template<int dim>
const Tensor<1,dim> &pfem2Particle<dim>::get_velocity() const
{
	Tensor<1,dim> res;
    for(int i = 0; i < dim; ++i)
		res[i] = velocity[i];
    
    return res;
}

template<int dim>
const Tensor<1,dim> &pfem2Particle<dim>::get_velocity_ext() const
{
	Tensor<1,dim> res;
    for(int i = 0; i < dim; ++i)
		res[i] = velocity_ext[i];
    
    return res;
}

template<int dim>
double pfem2Particle<dim>::get_velocity_component(int component) const
{
	return velocity[component];
}

template<int dim>
void pfem2Particle<dim>::set_velocity (const Tensor<1,dim> &new_velocity)
{
	for(int i = 0; i < dim; ++i)
		velocity[i] = new_velocity[i];
}

template<int dim>
void pfem2Particle<dim>::set_velocity_component (const double value, int component)
{
	velocity[component] = value;
}

template<int dim>
void pfem2Particle<dim>::set_velocity_ext (const Tensor<1,dim> &new_ext_velocity)
{
	for(int i = 0; i < dim; ++i)
		velocity_ext[i] = new_ext_velocity[i];
}

template<int dim>
typename Triangulation<dim>::cell_iterator pfem2Particle<dim>::get_surrounding_cell(const Triangulation<dim> &triangulation) const
{
	const typename Triangulation<dim>::cell_iterator cell(&triangulation, triangulation.n_levels() - 1, tria_position);
	
	return cell;
}

template<int dim>
typename DoFHandler<dim>::cell_iterator pfem2Particle<dim>::get_surrounding_cell(const Triangulation<dim> &triangulation, const DoFHandler<dim> &dof_handler) const
{
	const typename DoFHandler<dim>::cell_iterator cell(&triangulation, triangulation.n_levels() - 1, tria_position, &dof_handler);
	
	return cell;
}


template<int dim>
unsigned int pfem2Particle<dim>::find_closest_vertex_of_cell(const typename Triangulation<dim>::active_cell_iterator &cell, const Mapping<dim> &mapping)
{
	//transformation of local particle coordinates transformation is required as the global particle coordinates have already been updated by the time this function is called
	const Point<dim> old_position = mapping.transform_unit_to_real_cell(cell, reference_location);
	
	Tensor<1,dim> velocity_normalized = velocity_ext / velocity_ext.norm();
	Tensor<1,dim> particle_to_vertex = cell->vertex(0) - old_position;
    particle_to_vertex /= particle_to_vertex.norm();
    
    double maximum_angle = velocity_normalized * particle_to_vertex;
    unsigned int closest_vertex = 0;
    
    for (unsigned int v = 1; v < GeometryInfo<dim>::vertices_per_cell; ++v){
		particle_to_vertex = cell->vertex(v) - old_position;
		particle_to_vertex /= particle_to_vertex.norm();
		const double v_angle = velocity_normalized * particle_to_vertex;
		
		if (v_angle > maximum_angle){
			closest_vertex = v;
			maximum_angle = v_angle;
		}
	}
	
	return closest_vertex;
}

template<int dim>
std::size_t pfem2Particle<dim>::serialized_size_in_bytes() const
{
	std::size_t size = sizeof(id) + sizeof(location) + sizeof(reference_location) + sizeof(tria_position)
		+ sizeof(velocity) + sizeof(velocity_ext);

	return size;
}

template<int dim>
void pfem2Particle<dim>::write_data (void* &data) const
{
	unsigned int *id_data  = static_cast<unsigned int*> (data);
    *id_data = id;
    ++id_data;
    
    int *triaData = reinterpret_cast<int*> (id_data);
    *triaData = tria_position;
    ++triaData;
    
    // Write cell DoFs numbers
    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i,++triaData)
		*triaData = cell_dofs[i];
    
    double *pdata = reinterpret_cast<double*> (id_data);

    // Write location data
    for (unsigned int i = 0; i < dim; ++i,++pdata)
		*pdata = location[i];

    // Write reference location data
    for (unsigned int i = 0; i < dim; ++i,++pdata)
		*pdata = reference_location[i];
    
    // Write velocity
    for (unsigned int i = 0; i < dim; ++i,++pdata)
		*pdata = velocity[i];
    
    // Write streamline velocity
    for (unsigned int i = 0; i < dim; ++i,++pdata)
		*pdata = velocity_ext[i];
     
    data = static_cast<void*> (pdata);
}
