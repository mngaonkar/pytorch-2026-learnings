# dry_run_fake_backend.sh
NGPU=2 COMM_BACKEND=fake MODULE=my_recipes CONFIG=llama3_8b_cp2 ./run_train.sh
