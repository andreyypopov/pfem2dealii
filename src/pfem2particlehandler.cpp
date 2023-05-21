#include "pfem2particlehandler.h"

#include <deal.II/grid/grid_tools.h>

#include "pfem2solver.h"
#include "pfem2parameterhandler.h"
#include "pfem2fem.h"

template<int dim>
pfem2ParticleHandler<dim>::pfem2ParticleHandler(const FE_Q<dim> *finite_element)
	: particles()
    , feq(finite_element)
{
	this->mapping = MappingQ1<dim>();
}

template<int dim>
pfem2ParticleHandler<dim>::~pfem2ParticleHandler()
{
	clear_particles();
}

template<int dim>
const pfem2Solver<dim>* pfem2ParticleHandler<dim>::getPfem2Solver() const
{
	return mainSolver;
}

template<int dim>
void pfem2ParticleHandler<dim>::setPfem2Solver(const pfem2Solver<dim> *mainSolver)
{
	this->mainSolver = mainSolver;

    particle_integration_steps = mainSolver->getParameterHandler().getParticleIntegrationSteps();
    if(particle_integration_steps == 0)
        particle_integration_steps = 1;

    this->particle_transport_timestep = mainSolver->getParameterHandler().getTimeStep() / (double)particle_integration_steps;
    this->maxParticlesPerCellPart = mainSolver->getParameterHandler().getMaxParticlesPerCellPart();
}

template <int dim>
void pfem2ParticleHandler<dim>::seed_particles()
{
    TimerOutput::Scope timer_section(mainSolver->getTimer(), "Particle seeding");
    
    const auto& quantities = mainSolver->getParameterHandler().getParticlesPerCell();
    
    //generate possible combinations of indices for cell parts within a single cell
    int cellPartsCount = 1;
    for(int i = 0; i < dim; ++i)
        cellPartsCount *= quantities[i];

    cellPartsIndices.reserve(cellPartsCount);  
    std::array<int, dim> cellPartIndex;
    for(int i = 0; i < quantities[0]; ++i){
        cellPartIndex[0] = i;

        if(dim > 1)
            for(int j = 0; j < quantities[1]; ++j){
                cellPartIndex[1] = j;

                if(dim > 2)
                    for(int k = 0; k < quantities[2]; ++k)
                        cellPartIndex[2] = k;
            }

        cellPartsIndices.push_back(cellPartIndex);
    }

    particles.reserve(cellPartsIndices * mainSolver->getTriangulation()->n_active_cells());
    const auto& solutionV = mainSolver->getFemSolver()->getSolutionV();
    double shapeValue;
    
    double h[dim];
    for(int i = 0; i < dim; ++i)
        h[i] = 1.0 / quantities[i];

    for(const auto& cell : mainSolver->getFemSolver()->getDoFhandler().active_cell_iterators())
        if(cell->is_locally_owned())
            for(const auto& index : cellPartsIndices){
                Point<dim> newPoint;
                for(int i = 0; i < dim; ++i)
                    newPoint[i] = (index[i] + 1.0 / 2) * h[i];
                
                pfem2Particle particle(mapping.transform_unit_to_real_cell(cell, newPoint), newPoint, ++particleCount);
                particle.set_cell_dofs(cell);

                Tensor<1,dim> newVelocity(0.0);
                
                for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
                    shapeValue = feq->shape_value(vertex, particle.get_reference_location());

                    for(int i = 0; i < dim; ++i)
                        newVelocity[i] += shapeValue * solutionV[i](particle.cell_dofs[vertex]);
                }

                particle.setVelocity(newVelocity);
                insert_particle(particle, cell);
            }        

    std::cout << "Created and placed " << particleCount << " particles on process "
         << Utilities::MPI::this_mpi_process(mainSolver->getCommunicator()) << std::endl;    
}

template<int dim>
void pfem2ParticleHandler<dim>::correct_particle_velocity()
{
    TimerOutput::Scope timer_section(mainSolver->getTimer(), "Particle velocity correction");

    double shapeValue;
    const auto& solutionV = mainSolver->getFemSolver()->getSolutionV();
    const auto& oldSolutionV = mainSolver->getFemSolver()->getOldSolutionV();

    std::array<TrilinosWrappers::MPI::Vector, dim> deltaV;
    for(int i = 0; i < dim; ++i)
        deltaV[i] = solutionV[i] - oldSolutionV[i];

    Tensor<1, dim> velocityCorr;
    for(auto& particleIndex : particles){
        velocityCorr = 0;

        for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
            shapeValue = feq->shape_value(vertex, particleIndex.get_reference_location());

            for(int i = 0; i < dim; ++i)
                velocityCorr[i] += shapeValue * deltaV[i](particleIndex.cell_dofs[vertex]);
        }

        particleIndex.setVelocity(particleIndex.getVelocity() + velocityCorr);
    }
}

template<int dim>
void pfem2ParticleHandler<dim>::move_particles()
{
    TimerOutput::Scope timer_section(mainSolver->getTimer(), "Particle transport");

    Tensor<1, dim> particleTransportVel;
    double shapeValue;
    const auto& solutionV = mainSolver->getFemSolver()->getSolutionV();

    for (int np_m = 0; np_m < particle_integration_steps; ++np_m) {
        for(auto& particleIndex : particles){
            particleTransportVel = 0;

            for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
                shapeValue = feq->shape_value(vertex, particleIndex.get_reference_location());

                for(int i = 0; i < dim; ++i)
                    particleTransportVel[i] += shapeValue * solutionV[i](particleIndex.cell_dofs[vertex]);
            }

            particleIndex.set_velocity_ext(particleTransportVel);
            particleIndex.set_location(particleIndex.get_location() + particleTransportVel * particle_transport_timestep);
        }

        //update particle reference positions and their binding to cells
        sort_particles_into_subdomains_and_cells(mainSolver->getFemSolver()->getDoFhandler());
    }

    //check number of particles in cells, add new particles, delete excessive particles
    check_particle_distribution();
}

template<int dim>
void pfem2ParticleHandler<dim>::project_particle_fields()
{
    TimerOutput::Scope timer_section(mainSolver->getTimer(), "Particle fields projection");

    std::array<TrilinosWrappers::MPI::Vector, dim> nodeVelocity;
    TrilinosWrappers::MPI::Vector nodeWeights;

    for(int i = 0; i < dim; ++i){
        nodeVelocity[i].reinit(mainSolver->getFemSolver()->getLocallyOwnedDofs(), mainSolver->getCommunicator());
        nodeVelocity[i] = 0;
    }

    nodeWeights = 0;

    double shapeValue;

    for(const auto& particleIndex : particles)
        for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
            shapeValue = feq->shape_value(vertex, particleIndex.get_reference_location());

            for(int i = 0; i < dim; ++i)
                nodeVelocity[i][particleIndex.cell_dofs[vertex]] += shapeValue * particleIndex->get_velocity_component(i);

            nodeWeights[particleIndex.cell_dofs[vertex]] += shapeValue;
        }

    for(int i = 0; i < dim; ++i)
        nodeVelocity[i].compress(VectorOperation::add);

    nodeWeights.compress(VectorOperation::add);

    for(unsigned int k = nodeWeights.local_range().first; k < nodeWeights.local_range().second; ++k)
        for(int i = 0; i < dim; ++i){
            nodeVelocity[i][k] /= nodeWeights[k];
        }

    for(int i = 0; i < dim; ++i){
        nodeVelocity[i].compress(VectorOperation::insert);
        mainSolver->getFemSolver()->setSolutionV(nodeVelocity[i], i);
    }
}

template <int dim>
void pfem2ParticleHandler<dim>::output_particle_solution(int timestep_number)
{
    const std::string filename =  "particles-" + Utilities::int_to_string (timestep_number, 2) + "." + Utilities::int_to_string(Utilities::MPI::this_mpi_process(mainSolver->getCommunicator()), 3) + ".vtu";
	std::ofstream output (filename.c_str());
	
	//header
	output << "<?xml version=\"1.0\" ?> " << std::endl;
	output << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
	output << "  <UnstructuredGrid>" << std::endl;
	output << "    <Piece NumberOfPoints=\"" << particles.size() <<  "\" NumberOfCells=\"" << particles.size() << "\">" << std::endl;
	
	//particle positions
	output << "      <Points>" << std::endl;
	output << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;
	for(const auto& particleIndex : particles){
		output << "          " << particleIndex.get_location()[0] << " ";
        output << ((dim > 1) ? particleIndex.get_location()[1] : "0.0") << " ";
        output << ((dim > 2) ? particleIndex.get_location()[2] : "0.0") << std::endl;
    }
	output << "        </DataArray>" << std::endl;
	output << "      </Points>" << std::endl;

	//"cells" (one per each particle)
	output << "      <Cells>" << std::endl;
	output << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << std::endl;
	for (unsigned int i = 0; i < particles.size(); ++i)
        output << "          " << i << std::endl; 
	output << "        </DataArray>" << std::endl;

	//"offsets"
    output << "        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" << std::endl;
	output << "        ";
	for (unsigned int i = 0; i < particles.size(); ++i)
        output << "  " << i + 1;
	output << std::endl;
	output << "        </DataArray>" << std::endl;

	//"types"
    output << "        <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">" << std::endl;
	output << "        ";
	for (unsigned int i = 0; i < particles.size(); ++i)
        output << "  " << 1;
	output << std::endl;
	output << "        </DataArray>" << std::endl;
	output << "      </Cells>" << std::endl;

	//particle data
	output << "      <PointData Scalars=\"scalars\">" << std::endl;
	
	//velocity
	output << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;
	for(const auto& particleIndex : particles){
		output << "          " << particleIndex.get_velocity_component(0) << " ";
        output << ((dim > 1) ? particleIndex.get_velocity_component(1) : "0.0") << " ";
        output << ((dim > 2) ? particleIndex.get_velocity_component(2) : "0.0") << std::endl;
    }
	output << "        </DataArray>" << std::endl;

	//subdomain number
	output << "        <DataArray type=\"Float32\" Name=\"subdomain\" Format=\"ascii\">" << std::endl;
	output << "        ";

    const auto locallyOwnedSubdomain = mainSolver->getTriangulation().locally_owned_subdomain();
	for (unsigned int i = 0; i < particles.size(); ++i)
        output << "  " << locallyOwnedSubdomain;
	output << std::endl;
	output << "        </DataArray>" << std::endl;
	
	output << "      </PointData>" << std::endl;

	//footer
	output << "    </Piece>" << std::endl;
	output << "  </UnstructuredGrid>" << std::endl;
	output << "</VTKFile>" << std::endl;
    output.flush();
    output.close();
	
	if (Utilities::MPI::this_mpi_process==0) {
        std::ofstream master_output (("particles-" + Utilities::int_to_string (timestep_number, 2) + ".pvtu").c_str());
        
        master_output << "<?xml version=\"1.0\" ?> " << std::endl;
		master_output << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
		master_output << "  <PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
		master_output << "    <PPointData Scalars=\"scalars\">" << std::endl;
		master_output << "      <PDataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\"/>" << std::endl;
		master_output << "      <PDataArray type=\"Float32\" Name=\"subdomain\" format=\"ascii\"/>" << std::endl;
		master_output << "    </PPointData>" << std::endl;
		master_output << "    <PPoints>" << std::endl;
        master_output << "      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>" << std::endl;
		master_output << "    </PPoints>" << std::endl;
        
        for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mainSolver->getCommunicator()); ++i)
			master_output << "    <Piece Source=\"particles-" << Utilities::int_to_string (timestep_number, 2) << "." << Utilities::int_to_string(i, 3) << ".vtu\"/>";
			
		master_output << "  </PUnstructuredGrid>" << std::endl;
		master_output << "</VTKFile>" << std::endl;
        master_output.flush();
        master_output.close();
	}//if
}

template<int dim>
void pfem2ParticleHandler<dim>::initialize_maps()
{
	vertex_to_cells = std::vector<std::set<typename Triangulation<dim>::active_cell_iterator>>(GridTools::vertex_to_cell_map(mainSolver->getTriangulation()));
    vertex_to_cell_centers = std::vector<std::vector<Tensor<1,dim>>>(GridTools::vertex_to_cell_centers_directions(mainSolver->getTriangulation(),vertex_to_cells));	  
}

template<int dim>
void pfem2ParticleHandler<dim>::clear_particles()
{
	particles.clear();
}

template<int dim>
pfem2ParticleIterator<dim> pfem2ParticleHandler<dim>::remove_particle(pfem2ParticleIterator<dim> particleIndex)
{
	return particles.erase(particleIndex);
}

template<int dim>
void pfem2ParticleHandler<dim>::insert_particle(pfem2Particle<dim> &particle, const typename DoFHandler<dim>::active_cell_iterator &cell)
{
	particle.set_tria_position(cell->index());
	particle.set_cell_dofs(cell);
	particles.push_back(particle);
}

template<int dim>
bool compare_particle_association(const unsigned int a, const unsigned int b, const Tensor<1,dim> &particle_direction, const std::vector<Tensor<1,dim>> &center_directions)
{
	const double scalar_product_a = center_directions[a] * particle_direction;
    const double scalar_product_b = center_directions[b] * particle_direction;

    return scalar_product_a > scalar_product_b;
}

template<int dim>
void pfem2ParticleHandler<dim>::check_particle_distribution(const DoFHandler<dim> &dof_handler)
{
    const auto& triangulation = mainSolver->getTriangulation();

    std::map<int, std::map<std::array<int, dim>, int>> particlesInCellParts;

    double h[dim];
    for(int i = 0; i < dim; ++i)
        h[i] = 1.0 / quantities[i];

    //1. Get information about particle distribution within cells (per each cell part) and delete excessive particles
    int particleCellPart[dim];
    for(const auto& particleIndex = particles.begin(); particleIndex != particles.end(); ){
        for(int i = 0; i < dim; ++i)
            particleCellPart[i] = particleIndex->get_reference_location()[i] / h[i];

        if(particlesInCellParts[particleIndex->get_tria_position()][particleCellPart] > maxParticlesPerCellPart){
            *particleIndex = std::move(particles.back());
            particles.pop_back();
        } else {
            particlesInCellParts[particleIndex->get_tria_position()][particleCellPart]++;
            ++particleIndex;
        }
    }

    //2. Add a particle in each empty cell part
    const auto& solutionV = mainSolver->getFemSolver()->getSolutionV();
    double shapeValue;
    
    for(const auto& cellInfo : particlesInCellParts){
        const typename DoFHandler<dim>::cell_iterator dofCell(triangulation, triangulation->n_levels() - 1, cellInfo->first, &dof_handler);
        
        for(const auto& cellPartIndex : cellPartsIndices)
            if(cellInfo->second[cellPartIndex] == 0){
                Point<dim> newPoint;
                for(int i = 0; i < dim; ++i)
                    newPoint[i] = (cellPartIndex[i] + 1.0 / 2) * h[i];
                
                pfem2Particle particle(mapping.transform_unit_to_real_cell(dofCell, newPoint), newPoint, ++particleCount);
                particle.set_cell_dofs(dofCell);

                Tensor<1,dim> newVelocity(0.0);
                
                for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
                    shapeValue = feq->shape_value(vertex, particle.get_reference_location());

                    for(int i = 0; i < dim; ++i)
                        newVelocity[i] += shapeValue * solutionV[i](particle.cell_dofs[vertex]);
                }

                particle.setVelocity(newVelocity);
                insert_particle(particle, dofCell);
            }
    }
}

template<int dim>
void pfem2ParticleHandler<dim>::sort_particles_into_subdomains_and_cells(const DoFHandler<dim> &dof_handler)
{
    const auto& triangulation = mainSolver->getTriangulation();

	std::map<unsigned int, std::vector<pfem2Particle<dim>>> moved_particles;
	const std::set<unsigned int> ghost_owners = triangulation->ghost_owners();

	for(auto it = begin(); it != end(); ){
		const typename Triangulation<dim>::cell_iterator cell = (*it).get_surrounding_cell(triangulation);
		
		bool found_cell = false;
		try{
			const Point<dim> p_unit = mapping.transform_real_to_unit_cell(cell, (*it).get_location());
		
			if(GeometryInfo<dim>::is_inside_unit_cell(p_unit)){
				(*it).set_reference_location(p_unit);
				found_cell = true;
				++it;
			}
		} catch(typename Mapping<dim>::ExcTransformationFailed &){
#ifdef VERBOSE_OUTPUT
			std::cout << "Transformation failed for particle with global coordinates " << (*it).get_location() << " (checked cell index #" << cell->index() << ")" << std::endl;
#endif // VERBOSE_OUTPUT
		}

		if(!found_cell){
			std::vector<unsigned int> neighbor_permutation;
			
			Point<dim> current_reference_position;
			typename Triangulation<dim>::active_cell_iterator current_cell = (*it).get_surrounding_cell(triangulation);

			const unsigned int closest_vertex = (*it).find_closest_vertex_of_cell(current_cell, &mapping);
			Tensor<1,dim> vertex_to_particle = (*it).get_location() - current_cell->vertex(closest_vertex);
			vertex_to_particle /= vertex_to_particle.norm();

			const unsigned int closest_vertex_index = current_cell->vertex_index(closest_vertex);
			const unsigned int n_neighbor_cells = vertex_to_cells[closest_vertex_index].size();

			neighbor_permutation.resize(n_neighbor_cells);		  
			for (unsigned int i=0; i<n_neighbor_cells; ++i) neighbor_permutation[i] = i;

			std::sort(neighbor_permutation.begin(), neighbor_permutation.end(),
				std::bind(&compare_particle_association, std::placeholders::_1, std::placeholders::_2, std::cref(vertex_to_particle), std::cref(vertex_to_cell_centers[closest_vertex_index])));

			for (unsigned int i=0; i<n_neighbor_cells; ++i){
				typename std::set<typename Triangulation<dim>::active_cell_iterator>::const_iterator cell = vertex_to_cells[closest_vertex_index].begin();
				std::advance(cell,neighbor_permutation[i]);
              
				try {
					const Point<dim> p_unit = mapping.transform_real_to_unit_cell(*cell, (*it).get_location());
					if (GeometryInfo<dim>::is_inside_unit_cell(p_unit)){
						current_cell = *cell;
						(*it).set_reference_location(p_unit);
						(*it).set_tria_position(current_cell->index());
						
						const typename DoFHandler<dim>::cell_iterator dofCell(triangulation, triangulation->n_levels() - 1, current_cell->index(), &dof_handler);
						(*it).set_cell_dofs(dofCell);
						
						found_cell = true;
						
						break; 
					}
				} catch(typename Mapping<dim>::ExcTransformationFailed &)
                { }
            }

			if (found_cell) {
				if(current_cell->is_locally_owned()) ++it;
				else {
					moved_particles[current_cell->subdomain_id()].push_back(*it);
					*it = std::move(particles.back());
					particles.pop_back();
				}
			} else {
				*it = std::move(particles.back());
				particles.pop_back();
			}
		}                                                        
	}
	
#ifdef DEAL_II_WITH_MPI
	if(dealii::Utilities::MPI::n_mpi_processes(mainSolver->getCommunicator()) > 1) send_recv_particles(moved_particles);
	
	for (auto ghost_domain_id = ghost_owners.begin(); ghost_domain_id != ghost_owners.end(); ++ghost_domain_id) moved_particles[*ghost_domain_id].clear();
	moved_particles.clear();
#endif //DEAL_II_WITH_MPI
}

#ifdef DEAL_II_WITH_MPI
template<int dim>
void pfem2ParticleHandler<dim>::send_recv_particles(const std::map<unsigned int, std::vector<pfem2Particle<dim>>> &particles_to_send)
{
	// Determine the communication pattern
    const std::set<unsigned int> ghost_owners = mainSolver->getTriangulation()->ghost_owners();
    const std::vector<unsigned int> neighbors (ghost_owners.begin(), ghost_owners.end());
    const unsigned int n_neighbors = neighbors.size();

	// If we do not know the subdomain this particle needs to be send to, throw an error
    Assert (particles_to_send.find(numbers::artificial_subdomain_id) == particles_to_send.end(), ExcInternalError());

    // TODO: Implement the shipping of particles to processes that are not ghost owners of the local domain
    for (auto send_particles = particles_to_send.begin(); send_particles != particles_to_send.end(); ++send_particles)
      Assert(ghost_owners.find(send_particles->first) != ghost_owners.end(), ExcNotImplemented());

    unsigned int n_send_particles = 0;
    for (auto send_particles = particles_to_send.begin(); send_particles != particles_to_send.end(); ++send_particles)
		n_send_particles += send_particles->second.size();

    // Containers for the amount and offsets of data we will send to other processors and the data itself.
    std::vector<unsigned int> n_send_data(n_neighbors,0);
    std::vector<unsigned int> send_offsets(n_neighbors,0);
    std::vector<char> send_data;

    // Only serialize things if there are particles to be send.
    // We can not return early even if no particles are send, because we might receive particles from other processes
    if (n_send_particles){
        // Allocate space for sending particle data
        auto firstParticle = begin();
        const unsigned int particle_size = firstParticle->serialized_size_in_bytes();
        send_data.resize(n_send_particles * particle_size);
        void *data = static_cast<void *> (&send_data.front());

        // Serialize the data sorted by receiving process
        for (unsigned int i = 0; i<n_neighbors; ++i){
            send_offsets[i] = reinterpret_cast<std::size_t> (data) - reinterpret_cast<std::size_t> (&send_data.front());

            if(particles_to_send.count(neighbors[i]))
				for (unsigned int j = 0; j < particles_to_send.at(neighbors[i]).size(); ++j){
					auto particleIndex = particles_to_send.at(neighbors[i])[j];
					particleIndex.write_data(data);
				}

            n_send_data[i] = reinterpret_cast<std::size_t> (data) - send_offsets[i] - reinterpret_cast<std::size_t> (&send_data.front());
        }
    }

    // Containers for the data we will receive from other processors
    std::vector<unsigned int> n_recv_data(n_neighbors);
    std::vector<unsigned int> recv_offsets(n_neighbors);

    // Notify other processors how many particles we will send
    {
      std::vector<MPI_Request> n_requests(2*n_neighbors);
      for (unsigned int i=0; i<n_neighbors; ++i){
          const int ierr = MPI_Irecv(&(n_recv_data[i]), 1, MPI_INT, neighbors[i], 0, mainSolver->getCommunicator(), &(n_requests[2*i]));
          AssertThrowMPI(ierr);
      }
      
      for (unsigned int i=0; i<n_neighbors; ++i){
          const int ierr = MPI_Isend(&(n_send_data[i]), 1, MPI_INT, neighbors[i], 0, mainSolver->getCommunicator(), &(n_requests[2*i+1]));
          AssertThrowMPI(ierr);
      }

      const int ierr = MPI_Waitall(2*n_neighbors,&n_requests[0],MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }

    // Determine how many particles and data we will receive
    unsigned int total_recv_data = 0;
    for (unsigned int neighbor_id = 0; neighbor_id < n_neighbors; ++neighbor_id){
        recv_offsets[neighbor_id] = total_recv_data;
        total_recv_data += n_recv_data[neighbor_id];
    }

    // Set up the space for the received particle data
    std::vector<char> recv_data(total_recv_data);

    // Exchange the particle data between domains
    {
      std::vector<MPI_Request> requests(2*n_neighbors);
      unsigned int send_ops = 0;
      unsigned int recv_ops = 0;

      for (unsigned int i=0; i<n_neighbors; ++i)
          if (n_recv_data[i] > 0){
              const int ierr = MPI_Irecv(&(recv_data[recv_offsets[i]]), n_recv_data[i], MPI_CHAR, neighbors[i], 1, mainSolver->getCommunicator(), &(requests[send_ops]));
              AssertThrowMPI(ierr);
              send_ops++;
          }

      for (unsigned int i=0; i<n_neighbors; ++i)
          if (n_send_data[i] > 0){
              const int ierr = MPI_Isend(&(send_data[send_offsets[i]]), n_send_data[i], MPI_CHAR, neighbors[i], 1, mainSolver->getCommunicator(), &(requests[send_ops+recv_ops]));
              AssertThrowMPI(ierr);
              recv_ops++;
          }
          
      const int ierr = MPI_Waitall(send_ops+recv_ops, &requests[0], MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }

    // Put the received particles into the domain if they are in the triangulation
    const void *recv_data_it = static_cast<const void *> (recv_data.data());

    while (reinterpret_cast<std::size_t> (recv_data_it) - reinterpret_cast<std::size_t> (recv_data.data()) < total_recv_data){
		pfem2Particle newParticle(recv_data_it);
        particles.push_back(newParticle);
    }
	
    AssertThrow(recv_data_it == recv_data.data() + recv_data.size(),
                ExcMessage("The amount of data that was read into new particles does not match the amount of data sent around."));
}
#endif

template<int dim>
pfem2ParticleIterator<dim> pfem2ParticleHandler<dim>::begin()
{
	return particles.begin();
}

template<int dim>
pfem2ParticleIterator<dim> pfem2ParticleHandler<dim>::end()
{
	return particles.end();
}
