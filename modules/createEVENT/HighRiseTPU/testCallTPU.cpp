int 
callHighRise_TPU(const char *alpha,   
		 const char *breadthDepth, 
		 const char *depthHeight, 
		 int incidenceAngle,
		 const char *outputFilename);

int main(int argc, char *argv[])
{
  return callHighRise_TPU("1/4","2:1","1:3",10,"test.mat");
}
