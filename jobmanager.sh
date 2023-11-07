#!/bin/bash
# Set the threshold for the maximum number of running jobs
threshold=164
echo "Job Threshold set to $threshold"
job_number=0
# Read the list of job commands from a text file
while read -r line
do
	# Check the current number of running jobs
	num_jobs=$(squeue -h | wc -l)
	echo "Jobs already running: $num_jobs"
	# Wait until the number of running jobs is below the threshold
	while [[ $num_jobs -ge $threshold ]]
	do
		sleep 60
		num_jobs=$(squeue | wc -l)
		echo "waiting, $num_jobs currently running"
	done
	echo "Line: $line"
	# Generate temp script
	script_name="temp_job_$(date +%s%N).sh"
	# Write line to script
	echo -e "#!/bin/bash\n" $line > $script_name
	# Add executable permission for the script
	chmod +x $script_name
	echo "Added permission for $script_name"
	run=1
	while [[ $run -le 50 ]]
	do
		echo "Run number $run"
		# Submit the job using the sbatch command
		sbatch -J "${run}_${job_number}" $script_name
		((run++))
	done
	# Remove temp script
	rm $script_name
	((job_number++))
done < jobs.txt

echo "FINISHED"
