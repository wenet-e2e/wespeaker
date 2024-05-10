#!/bin/bash

# Define the path to the template file
template_file="job_extract.sh"

for i in {1..19}
do
  # Create a temporary script file
  temp_script="temp_script_${i}.sh"

  # Replace MODEL_ID in the template with the current number
  sed "s/MODEL_ID/${i}/g" $template_file > $temp_script
  cat $temp_script

  # sbatch the script
  sbatch $temp_script

  # Optionally remove the temporary script after execution
  rm $temp_script
done
