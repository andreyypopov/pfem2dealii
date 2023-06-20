#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <iomanip>

int main (int argc, char *argv[])
{
	if(argc < 2){
		std::cout << "3D UNV mesh repair tool for deal.II import (removal of excessive UNV-file parts, internal boundaries)" << std::endl << std::endl;
		std::cout << "Usage: ./repairMesh3D <meshFileName>.unv" << std::endl;

		return 0;
	}

	std::ifstream in(argv[1]);
	std::ofstream out(std::string("repaired_" + std::string(argv[1])).c_str());
	out << std::setw(6) << -1 << std::endl;

	std::string line;

	std::getline(in, line);
	std::size_t found = line.find("2411");

	//skip all (preliminary) sections until 2411 (vertices)
	while(found == std::string::npos){
		std::getline(in, line);
		found = line.find("2411");
	}

	out << line << std::endl;

	int tmp, dummy;
	double x[3];

	in >> tmp;

	//section 2411 (vertices) is copied straight into the output file
	while(tmp != -1){
		out << std::setw(10) << tmp;
		in >> dummy >> dummy >> dummy;

		out << std::setw(10) << 1;
		out << std::setw(10) << 1;
		out << std::setw(10) << 11;
		out << std::endl;

		in >> x[0] >> x[1] >> x[2];
		out << std::scientific << std::uppercase << std::right << std::setprecision(16) << std::setw(25) << x[0];
		out << std::scientific << std::uppercase << std::right << std::setprecision(16) << std::setw(25) << x[1];
		out << std::scientific << std::uppercase << std::right << std::setprecision(16) << std::setw(25) << x[2];
		out << std::endl;

		in >> tmp;
	}

	out << std::setw(6) << -1 << std::endl;
	out << std::setw(6) << -1 << std::endl;
	out << std::setw(6) << 2412 << std::endl;

	in >> tmp >> tmp >> tmp;

	std::map<int,std::vector<int>> lines;
	std::map<int,std::vector<int>> faces;
	std::map<int,std::vector<int>> hexahedra;

	int no, type;

	//read the edge and element data
	while(tmp != -1){
		no = tmp;

		in >> type >> dummy >> dummy >> dummy >> dummy;
		std::vector<int> vertices;

		if(type == 11){		//edge (2 vertices)
			in >> dummy >> dummy >> dummy;

			vertices.resize(2);
			for(int i = 0; i < 2; i++){
				in >> tmp;
				vertices[i] = tmp;
			}

			lines[no] = vertices;
		} else if(type == 44){//element (4 vertices)
			vertices.resize(4);

			for(int i = 0; i < 4; i++){
				in >> tmp;
				vertices[i] = tmp;
			}

			faces[no] = vertices;
		} else if(type == 115){//hexahedra (8 vertices)
			vertices.resize(8);

			for(int i = 0; i < 8; ++i){
				in >> tmp;
				vertices[i] = tmp;
			}

			hexahedra[no] = vertices;
		}

		in >> tmp;
	}

	in >> tmp >> tmp;

	std::map<int, std::vector<int>> boundary;

	//read the boundary patch data
	if(tmp == 2467){
		in >> tmp;

		int n_entities, id;

		while(tmp != -1){
			in >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> n_entities;
			in >> id;

			std::vector<int> patchFaces(n_entities);

			for(int i = 0; i < n_entities; ++i){
				in >> dummy >> no >> dummy >> dummy;

				patchFaces[i] = no;
			}

			boundary[id] = patchFaces;

			in >> tmp;
		}
	}

	in.close();		//finished reading the input file

	//the list of "necessary" boundary faces, which will be used for boundary description
	//(to be written to the new UNV-file)
	std::set<int> necessaryBoundaryFaces;
	for(auto it = boundary.cbegin(); it != boundary.cend(); ++it)
		for(unsigned int i = 0; i < it->second.size(); i++)	//include all faces that constitute this boundary patch
			necessaryBoundaryFaces.emplace(it->second[i]);

	std::cout << "original list contains " << faces.size() << " faces" << std::endl;
	std::cout << "filtered list contains " << necessaryBoundaryFaces.size() << " faces" << std::endl;

	//writing edges, elements, hexahedra and boundary patches to the new UNV-file
	//edges
	for(auto it = lines.cbegin(); it != lines.cend(); ++it){
		out << std::setw(10) << it->first;
		out << std::setw(10) << 11;
		out << std::setw(10) << 2;
		out << std::setw(10) << 1;
		out << std::setw(10) << 7;
		out << std::setw(10) << 2;
		out << std::endl;
		out << std::setw(10) << 0;
		out << std::setw(10) << 1;
		out << std::setw(10) << 1;
		out << std::endl;

		for(int i = 0; i < 2; ++i)
			out << std::setw(10) << it->second[i];
		out << std::endl;
	}

	//faces (only necessary for boundary patches description)
	for(auto it = necessaryBoundaryFaces.cbegin(); it != necessaryBoundaryFaces.cend(); ++it){
		out << std::setw(10) << *it;
		out << std::setw(10) << 44;
		out << std::setw(10) << 2;
		out << std::setw(10) << 1;
		out << std::setw(10) << 7;
		out << std::setw(10) << 4;
		out << std::endl;

		for(int i = 0; i < 4; ++i)
			out << std::setw(10) << faces[*it][i];
		out << std::endl;
	}

	//hexahedra
	for(auto it = hexahedra.cbegin(); it != hexahedra.cend(); ++it){
		out << std::setw(10) << it->first;
		out << std::setw(10) << 115;
		out << std::setw(10) << 2;
		out << std::setw(10) << 1;
		out << std::setw(10) << 7;
		out << std::setw(10) << 8;
		out << std::endl;

		for(int i = 0; i < 8; ++i)
			out << std::setw(10) << it->second[i];
		out << std::endl;
	}

	out << std::setw(6) << -1 << std::endl;
	out << std::setw(6) << -1 << std::endl;

	//boundary patches (list of faces for each patch)
	out << std::setw(6) << 2467 << std::endl;
	int boundaryPatchNum = 0;
	for(auto it = boundary.cbegin(); it != boundary.cend(); ++it){
		out << std::setw(10) << ++boundaryPatchNum;

		for(int i = 0; i < 6; i++)
			out << std::setw(10) << 0;

		out << std::setw(10) << it->second.size();
		out << std::endl;

		out << it->first << std::endl;

		unsigned int j = 0;
		while(j < it->second.size()){
			out << std::setw(10) << 8;
			out << std::setw(10) << it->second[j];
			out << std::setw(10) << 0;
			out << std::setw(10) << 0;

			j++;

			if(j < it->second.size()){
				out << std::setw(10) << 8;
				out << std::setw(10) << it->second[j];
				out << std::setw(10) << 0;
				out << std::setw(10) << 0;

				j++;
			}

			out << std::endl;
		}
	}

	out << std::setw(6) << -1 << std::endl;
	out.close();

	return 0;
}
