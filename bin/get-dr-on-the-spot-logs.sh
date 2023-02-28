cd logs
if [[ $# != 2 ]] ; then
echo 'USAGE: .bin/get-dr-on-the-spot-logs.sh s3://your_bucket/your_log_location logs/local_folder_to_copy_to'
exit 0
fi
echo "Copying Deep Racer logs from S3 to local for Log analysis"
echo "Copying from $1"
echo "Copying to $2"
echo "If it doesn't work correctly drop the final / from $1"
mkdir $2
#Copy logs and metrics
aws s3 cp $1/logs $2/logs/training --recursive
aws s3 cp $1/metrics $2/metrics/training --recursive
#Manipulate robomaker container log names so as not to overwrite the logs from each worker when using multiple robomaker containers
COUNTER=$(aws s3 ls $1 | grep [0-9]/ -c)
COUNTER=$(($COUNTER+1))
for i in seq 1 $COUNTER;
do
for file in $(ls $2/logs/training/robomaker.$i..log)
do
mv -T 
{file%%..}"$i-"robomaker.""${file##.}"
done
done
#Manipulate log names for sim-trace so as not to overwrite the logs from each worker when using multiple robomaker containers
#aws s3 cp $1/training-simtrace $2/sim-trace/training/training-simtrace --recursive
for dir in $(aws s3 ls $1/ | grep [0-9]/ | sed 's/PRE//g;s/.$//')
do
echo "copying $1/$dir/training-simtrace to $2/sim-trace/training/training-simtrace"
aws s3 cp $1/$dir/training-simtrace $2/sim-trace/training/training-simtrace --recursive
for file in $(ls $2/sim-trace/training/training-simtrace/iteration.csv)
do
mv -T $file "${file%%..}"-worker$dir."${file##*.}"
done
done
cd ..