#!/bin/bash
# Set the threshold for the maximum number of running jobs
threshold=464
echo "Job Threshold set to $threshold"
# Read the list of job commands from a text file
while read -r line
do
	# Check the current number of running jobs
	num_jobs=$(squeue -h | wc -l)
	# Wait until the number of running jobs is below the threshold
	while [[ $num_jobs -ge $threshold ]]
	do
		sleep 60
		num_jobs=$(squeue | wc -l)
	done
	echo "Line: $line"
	# Generate temp script
	script_name="temp_job_$(date +%s%N).sh"
	# Write line to script
	echo -e "#!/bin/bash\n" $line > $script_name
	# Add executable permission for the script
	chmod +x $script_name
	echo "Added permission for $script_name"
	# Submit the job using the sbatch command
	sbatch $script_name
	# Remove temp script
	#rm $script_name
done < jobs.txt