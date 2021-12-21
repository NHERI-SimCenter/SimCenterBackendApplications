#!/bin/sh

# create a function to facilitate logging
log_msg() {
  echo "$(TZ='America/Los_Angeles' date +%Y-%m-%dT%H:%M:%S) $1"
}

log_msg 'Starting calculation...'

# load modules
module load intel
module load dakota
module load launcher/3.1

echo "-------------------------------------------------------------------------"
log_msg "INPUTS"
echo "-------------------------------------------------------------------------"

dataFile=${dataFile}
configFile=${configFile}

log_msg "dataFile: $dataFile"
log_msg "configFile: $configFile"

if [[ -z "${buildingFilter}" ]]
then
    log_msg "Building filter not provided, trying to use the filter specified in the config file"
    buildingFilter=""
else
    buildingFilter=${buildingFilter}
fi

log_msg "buildingFilter: $buildingFilter"

if [[ -z "${buildingsPerTask}" ]]
then
    log_msg "Number of buildings per task is not provided, defaulting to 10"
	  buildingsPerTask="10"
else
    buildingsPerTask=${buildingsPerTask}
fi

log_msg "buildingsPerTask: $buildingsPerTask"

echo ""
echo "-------------------------------------------------------------------------"
log_msg "PREPARE FILES ON HEAD NODE"
echo "-------------------------------------------------------------------------"
log_msg "Current Job Directory: $PWD"

echo ""
log_msg "Preparing Workflow Applications..."
cp /home1/04972/zs_adam/SimCenterBackendApplications/backend_apps.tar.gz ./
tar zxBf backend_apps.tar.gz
mv backend_apps rWHALE_Adam

echo ""
log_msg "Preparing Input Data (dataFile)..."

/home1/00477/tg457427/p7zip/p7zip_16.02/bin/7za x $dataFile -orWHALE_Adam/input_data

echo ""
log_msg "Preparing Configuration File.."
cp $configFile ./rWHALE_Adam/

echo ""
log_msg "Files in Job Dir:"
echo ""
ls -lh

echo ""
log_msg "Files in rWHALE dir:"
echo ""
ls -lh ./rWHALE_Adam

echo ""
log_msg "Input data files:"
echo ""
ls -lh ./rWHALE_Adam/input_data

echo ""
echo "-------------------------------------------------------------------------"
log_msg "CREATE JOBS"
echo "-------------------------------------------------------------------------"
log_msg "..."
# add our python interpreter to the PATH
export PATH=$PWD/rWHALE_Adam/applications/Workflow/python/bin:$PATH

python3 ./rWHALE_Adam/applications/Workflow/CreateWorkflowJobs.py \
  -buildingFilter $buildingFilter \
  -configFile $configFile \
  -outputDir $PWD \
  -buildingsPerTask $buildingsPerTask \
  -rWHALE_dir "rWHALE_Adam"

echo ""
log_msg "$(wc -l < WorkflowJobs.txt) jobs created."
echo ""

echo ""
echo "-------------------------------------------------------------------------"
log_msg "PREPARE SIMULATION ENVIRONMENT ON COMPUTE NODES"
echo "-------------------------------------------------------------------------"

log_msg "List of compute nodes: "
log_msg $SLURM_NODELIST

log_msg "Extracting the backend apps..."
for node in $(scontrol show hostnames $SLURM_NODELIST) ; do
  srun -N 1 -n 1 -w $node tar zxBf $PWD/backend_apps.tar.gz -C /tmp &
done
wait

log_msg "Moving the apps to the rWHALE dir..."
for node in $(scontrol show hostnames $SLURM_NODELIST) ; do
  srun -N 1 -n 1 -w $node mv /tmp/backend_apps /tmp/rWHALE_Adam  &
done
wait

log_msg "Copying the input data..."
for node in $(scontrol show hostnames $SLURM_NODELIST) ; do
  srun -N 1 -n 1 -w $node cp -fr $PWD/rWHALE_Adam/input_data /tmp/rWHALE_Adam/ &
done

log_msg "Copying the configuration file..."
for node in $(scontrol show hostnames $SLURM_NODELIST) ; do
  srun -N 1 -n 1 -w $node cp $PWD/rWHALE_Adam/$configFile /tmp/rWHALE_Adam/ &
done
wait

# add the rWHALE dirs to the PATH
export PATH=/tmp/rWHALE_Adam/applications:/tmp/rWHALE_Adam/applications/Workflow::/tmp/rWHALE_Adam/applications/Workflow/python/bin:$PATH

log_msg "Installing python dependencies..."
for node in $(scontrol show hostnames $SLURM_NODELIST) ; do
  srun -N 1 -n 1 -w $node python3 -m pip install --upgrade pip
  srun -N 1 -n 1 -w $node python3 -m pip install openseespy==3.2.2.3
  srun -N 1 -n 1 -w $node python3 -m pip install nheri_simcenter
done
wait

echo ""
log_msg "Finished setting up compute nodes."

echo ""
echo "-------------------------------------------------------------------------"
log_msg "RUN SIMULATION"
echo "-------------------------------------------------------------------------"

echo ""
log_msg "Setting up Launcher"
echo ""
export LAUNCHER_JOB_FILE=$PWD/WorkflowJobs.txt
export LAUNCHER_WORKDIR=/tmp/rWHALE_Adam/

echo ""
log_msg "Running simulation on nodes"
echo ""
$TACC_LAUNCHER_DIR/paramrun > launcher.out 2> launcher.err

echo ""
log_msg "Simulation completed"
echo ""

echo ""
echo "-------------------------------------------------------------------------"
log_msg "COLLECT RESULTS" $date
echo "-------------------------------------------------------------------------"

echo ""
log_msg "Aggregating Results..."
python3 ./rWHALE_Adam/applications/Workflow/AggregateResults.py

echo ""
log_msg "Archiving Log Files..."
/home1/00477/tg457427/p7zip/p7zip_16.02/bin/7za a -tzip logs.zip ./logs/*

echo ""
log_msg "Cleaning up..."
for node in $(scontrol show hostnames $SLURM_NODELIST) ; do
  srun -N 1 -n 1 -w $node rm -fr /tmp/rWHALE_Adam &
done
wait

rm -rf rWHALE_Adam
rm -rf results
rm -rf logs
rm backend_apps.tar.gz

rm $dataFile

rm $configFile

echo ""
log_msg "Calculation completed."

