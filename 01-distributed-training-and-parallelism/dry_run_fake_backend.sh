# dry_run_fake_backend.sh
NGPU=2 COMM_MODE=fake_backend MODULE=my_recipes CONFIG=llama3_debugmodel_cp2 ./run_train.sh
