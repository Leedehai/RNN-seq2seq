# Remove directories created during testing.
curr_dir=${PWD##*/}
cd ..
rm -rf RUN_V
cd $curr_dir
