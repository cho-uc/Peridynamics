//Edited version
#include <vtkSmartPointer.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkLookupTable.h>
#include <vtkGlyph3DMapper.h>
#include <vtkSphereSource.h>

#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>

#include <vtkNamedColors.h>
#include <fstream> // for I/O to file
#include <vector>
#include <algorithm> // find max, min of array

// For compatibility with new VTK generic data arrays
#ifdef vtkGenericDataArray_h
#define InsertNextTupleValue InsertNextTypedTuple
#endif

using namespace std;

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf( "Please input argument : usage = ./ColoredPoints filename.txt \n");
		return 0;
	}
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	ifstream myfile;
    myfile.open(argv[1]);
    vector<float> u_n1; //to set color based on u_n1
   
    if (myfile.is_open()) {
		while ( !myfile.eof()) {
			float x_pos,y_pos,z_pos, u_n1_temp;
			myfile >> x_pos;
			myfile >> y_pos;
			myfile >> z_pos;
			myfile >> u_n1_temp;
			u_n1.push_back(u_n1_temp);
			points->InsertNextPoint(x_pos, y_pos, z_pos);
		  }
    }
	
    myfile.close();
	
	const auto [u_n1_min, u_n1_max] = minmax_element(begin(u_n1), end(u_n1));
    cout << "u_n1_min = " << *u_n1_min << ", u_n1_max = " << *u_n1_max << '\n';
	
	cout<<"No of points of points = "<<points->GetNumberOfPoints()<<endl;

  vtkSmartPointer<vtkPolyData> pointsPolydata = vtkSmartPointer<vtkPolyData>::New();

  pointsPolydata->SetPoints(points);

  vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
  vertexFilter->SetInputData(pointsPolydata);
  vertexFilter->Update();

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->ShallowCopy(vertexFilter->GetOutput());

  // -----Setup colors--------------------------------------
  vtkSmartPointer<vtkNamedColors> namedColors = vtkSmartPointer<vtkNamedColors>::New();

  vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
  colors->SetNumberOfComponents(3);
  colors->SetName ("Colors");
 
  
   double bounds[6];
	pointsPolydata->GetBounds(bounds);
	double minx = bounds[0]; double maxx = bounds[1];
	double miny = bounds[2]; double maxy = bounds[3];
  double minz = bounds[4];  double maxz = bounds[5];
  
  cout << "minx= " << minx << endl; cout << "maxx= " << maxx << endl;
  cout << "miny= " << miny << endl; cout << "maxy= " << maxy << endl;
  cout << "minz= " << minz << endl; cout << "maxz= " << maxz << endl;
  
   // Create the color map
  vtkSmartPointer<vtkLookupTable> colorLookupTable = vtkSmartPointer<vtkLookupTable>::New();
  //colorLookupTable->SetTableRange(minx, maxx); //coloring according to x
  colorLookupTable->SetTableRange(*u_n1_min, *u_n1_max); //coloring according to u_n1 (use dereference for pointer)
  colorLookupTable->Build();

  // Generate the colors for each point based on the color map

  for(int i = 0; i < pointsPolydata->GetNumberOfPoints(); i++) {
    double position[3];
    pointsPolydata->GetPoint(i,position);
    double dcolor[3];
    //colorLookupTable->GetColor(position[0], dcolor); //coloring according to X
	colorLookupTable->GetColor(u_n1[i], dcolor); //coloring according to u_n1
    //cout << "dcolor: " << dcolor[0] << " " << dcolor[1] << " " << dcolor[2] << std::endl;
    unsigned char color[3];
    for(unsigned int j = 0; j < 3; j++) {
      color[j] = static_cast<unsigned char>(255.0 * dcolor[j]);
    }
    //cout << "color: " << (int)color[0] << " "<< (int)color[1] << " " << (int)color[2] << std::endl;

    colors->InsertNextTupleValue(color);
  }
  
  
  polydata->GetPointData()->SetScalars(colors);

  // Visualization
  vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
  sphere->SetPhiResolution(21);
  sphere->SetThetaResolution(21);
  sphere->SetRadius(.008);  
  
  vtkSmartPointer<vtkPolyDataMapper> pointMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  pointMapper->SetInputData(polydata);
 //pointMapper->SetSourceConnection(sphere->GetOutputPort());

  vtkSmartPointer<vtkActor> pointActor = vtkSmartPointer<vtkActor>::New();
  pointActor->SetMapper(pointMapper);
 pointActor->GetProperty()->SetPointSize(10);
pointActor->GetProperty()->SetSpecular(.6);
  pointActor->GetProperty()->SetSpecularColor(1.0,1.0,1.0);
  pointActor->GetProperty()->SetSpecularPower(100);
  
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);

  renderer->AddActor(pointActor);
   renderer->SetBackground(0.6,1.0,0.8);  //light green
  renderWindow->Render();
  renderWindowInteractor->Start();

  return EXIT_SUCCESS;
}
