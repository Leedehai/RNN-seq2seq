# Remove directories created during testing.
curr_dir=${PWD##*/}
cd ..
rm -rf RUN_V/test_results.txt
rm -rf RUN_V/test_log.txt
cd $curr_dir
